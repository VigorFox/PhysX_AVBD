// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef DY_AVBD_SOFT_BODY_COMPONENT_H
#define DY_AVBD_SOFT_BODY_COMPONENT_H

// =============================================================================
// Internal AVBD Soft Body / Cloth -- energy-based deformable component
//
// Elastic energies use position-level VBD blocks. Contacts use a persistent
// augmented-Lagrangian normal/tangent state; pins currently use adaptive
// penalty state. The authoritative global schedule is serial vertex-block
// nonlinear Gauss-Seidel.
//
// Elastic forces: StVK (triangles), Neo-Hookean (tetrahedra), dihedral bending
// Constraints: contact (ground/soft-soft/soft-rigid), kinematic pins
//
// This Scene-external component remains private until a real CPU deformable
// actor/factory/buffer contract exists. Validation Snippets may consume it
// through an explicit private include path; it is not a public PhysX API.
//
// References: VBD (SIGGRAPH 2024), AVBD (SIGGRAPH 2025)
// =============================================================================

#include "foundation/PxAllocator.h"
#include "foundation/PxAlignedMalloc.h"
#include "foundation/PxArray.h"
#include "foundation/PxAssert.h"
#include "foundation/PxBounds3.h"
#include "foundation/PxMat33.h"
#include "foundation/PxMathUtils.h"
#include "foundation/PxQuat.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxSort.h"
#include "foundation/PxTime.h"
#include "foundation/PxVec3.h"
#include "PxMaterial.h"

#include <cstdlib>
#include <cstdint>
#include <cstring>

#include "DyAvbdConstraint.h"
#include "DyAvbdCpuIsa.h"

namespace physx
{
namespace Dy
{

// Private bridge supplied by PhysX core for component headers that are also
// compiled by validation Snippets.  Deliberately undecorated here: static
// PhysX sub-libraries resolve it locally, while the Snippet import library
// resolves the exported C symbol.
extern "C" AvbdCpuIsaCorotationalTetPacket8Fn PX_CALL_CONV
PxAvbdCpuIsaCorotationalTetPacket8FunctionInternal();

// Internal experiment controls are process/module policy.  Every caller below
// stores this result in a function-local static before entering a simulation
// stage; never call this reader directly from an epoch, task or kernel.
PX_FORCE_INLINE bool avbdReadProcessExactOneFlag(const char* name)
{
	const char* value = std::getenv(name);
	return value && value[0] == '1' && value[1] == '\0';
}

// Temporary physical A/B for the particle elastic proximal stabilization.
// Preserve the legacy route unless the process starts with the exact value
// "0".  This policy must only be sampled at timestep/outer-stage boundaries;
// in particular, the particle primal continues to have no new runtime branch.
PX_FORCE_INLINE bool avbdUseSoftElasticProximal()
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
PX_FORCE_INLINE bool avbdUseSoftAdaptivePrimalInitialization()
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
PX_FORCE_INLINE bool avbdUseSoftRigidPrimalInitialization()
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
PX_FORCE_INLINE bool avbdUseGroundTetPatchProbe()
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
PX_FORCE_INLINE bool avbdUseVelocityTangentOwner()
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
PX_FORCE_INLINE bool avbdUseParticlePrimalWorkCensus()
{
	static const bool enabled = []()
	{
		const char* value = std::getenv("PHYSX_AVBD_P8_PRIMAL_CENSUS");
		return value && value[0] == '1' && value[1] == '\0';
	}();
	return enabled;
}

// The admitted AVX2+FMA corotational backend is on by default. Keep one
// process-start rollback switch for differential validation; ISA selection
// and rollback sampling happen once, outside every particle/tet loop.
PX_FORCE_INLINE bool avbdUseCorotationalTetPacketKernel()
{
	static const bool enabled = []()
	{
		const char* value = std::getenv(
			"PHYSX_AVBD_DISABLE_COROTATIONAL_TET_PACKET_KERNEL");
		return !(value && value[0] == '1' && value[1] == '\0');
	}();
	return enabled;
}

PX_FORCE_INLINE AvbdCpuIsaCorotationalTetPacket8Fn
avbdGetCorotationalTetPacketKernel()
{
	static const AvbdCpuIsaCorotationalTetPacket8Fn kernel =
		avbdUseCorotationalTetPacketKernel()
			? PxAvbdCpuIsaCorotationalTetPacket8FunctionInternal()
			: NULL;
	return kernel;
}

// Packet IR is part of the admitted backend and is never built when the
// rollback switch or the runtime ISA gate selects scalar SSE2 authority.
PX_FORCE_INLINE bool avbdUseCorotationalTetPacketIr()
{
	static const bool enabled =
		avbdGetCorotationalTetPacketKernel() != NULL;
	return enabled;
}

// The hierarchy is on by default whenever the compiled body has valid
// topology.  This intentionally remains an internal diagnostic switch: it
// permits the performance runner to compare the refit path with the exact
// retained full traversal in the same executable.  Environment selection is
// process/module policy: sampling it inside every OGC epoch serializes on the
// CRT environment lock and can dominate small self-collision workloads.
PX_FORCE_INLINE bool avbdUseSurfaceTriangleBvh()
{
	static const bool enabled = []()
	{
		const char* value = std::getenv("PHYSX_AVBD_SURFACE_TRIANGLE_BVH");
		return !value || value[0] != '0';
	}();
	return enabled;
}

PX_FORCE_INLINE bool avbdUseSurfaceEdgeBvh()
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
PX_FORCE_INLINE bool avbdUseRigidTriangleSurfaceBvh()
{
	static const bool enabled = []()
	{
		const char* value = std::getenv("PHYSX_AVBD_RIGID_TRIANGLE_BVH");
		return !value || value[0] != '0';
	}();
	return enabled;
}

// P5.8b is independently opt-in. Triangle topology and pose are read-only
// only after the P5.8a range leaf supplies per-task BVH candidate scratch.
PX_FORCE_INLINE bool avbdUseRigidTriangleSurfaceContactTaskFanIn()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_RIGID_TRIANGLE_SURFACE_TASK_FANIN");
	return enabled;
}

PX_FORCE_INLINE bool avbdForceRigidTriangleSurfaceContactTaskFanIn()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_FORCE_RIGID_TRIANGLE_SURFACE_TASK_FANIN");
	return enabled;
}

// P5.20's sole threshold candidate is intentionally a literal test switch,
// rather than a general runtime tuning knob. It can matter only after the
// independently opt-in triangle task route is already selected; absent this
// switch, the accepted 128-particles-per-child policy remains exact.
PX_FORCE_INLINE bool avbdUseRigidTriangleSurfaceContactTaskThreshold96()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_RIGID_TRIANGLE_SURFACE_TASK_THRESHOLD_96");
	return enabled;
}

// P5.27 is a correctness-gated feature-only repartition experiment. It never
// enables the enclosing triangle task route and, when selected, only changes
// which existing child owns a canonical feature-plan row. The parent still
// merges one private output stream per canonical row.
PX_FORCE_INLINE bool avbdUseRigidTriangleSurfaceFeatureRoundRobinTaskPlan()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_RIGID_TRIANGLE_SURFACE_FEATURE_ROUND_ROBIN_TASKS");
	return enabled;
}

// P5.29 is a control for P5.27's measurement: it preserves accepted
// contiguous row ownership but uses P5.27's one-private-output-per-row
// execution and canonical parent reconstruction. It isolates that cost from
// row distribution and never enables the enclosing triangle task route.
PX_FORCE_INLINE bool avbdUseRigidTriangleSurfaceFeatureRowPrivateOutputTaskPlan()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_RIGID_TRIANGLE_SURFACE_FEATURE_ROW_PRIVATE_OUTPUTS");
	return enabled;
}

// P5.30 instruments only the already default-off triangle task route. It is
// intentionally a diagnostic switch: normal and accepted task behavior take
// no per-primitive clocks.
PX_FORCE_INLINE bool avbdUseRigidTriangleSurfaceFeatureSweptSubstageTiming()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_RIGID_TRIANGLE_SURFACE_FEATURE_SWEPT_SUBSTAGE_TIMING");
	return enabled;
}

// P5.31 counts exact (surface, particle) forward-owner query multiplicity in
// the default-off triangle task route. It is diagnostic-only and does not
// cache, bypass or otherwise alter the owner predicate.
PX_FORCE_INLINE bool avbdUseRigidTriangleSurfaceFeatureForwardOwnerQueryStats()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_RIGID_TRIANGLE_SURFACE_FEATURE_FORWARD_OWNER_QUERY_STATS");
	return enabled;
}

// P5.38 observes the discrete OGC query shape without a clock, allocation or
// result-cache. It is restricted to the already opt-in triangle task route so
// normal serial and accepted task execution retain no telemetry state.
PX_FORCE_INLINE bool avbdUseRigidTriangleSurfaceFeatureDiscreteQueryStats()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_RIGID_TRIANGLE_SURFACE_FEATURE_DISCRETE_QUERY_STATS");
	return enabled;
}

// P5.39 is an explicitly gated task-route candidate. It rejects a complete
// discrete body/surface feature row only when the exact eight-corner local
// image of the body's world AABB misses the rigid surface's local bounds.
PX_FORCE_INLINE bool avbdUseRigidTriangleSurfaceFeatureDiscreteBodyLocalBoundsCull()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_RIGID_TRIANGLE_SURFACE_FEATURE_DISCRETE_BODY_LOCAL_BOUNDS_CULL");
	return enabled;
}

// P5.41 promotes P5.39 only inside the already opt-in Scene triangle task
// route. The explicit disable switch retains the historical unculled task
// predicate for regression and measurement; it never changes serial/global
// behavior and must override any historical force switch.
PX_FORCE_INLINE bool avbdDisableRigidTriangleSurfaceFeatureDiscreteBodyLocalBoundsCull()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_RIGID_TRIANGLE_SURFACE_FEATURE_DISCRETE_BODY_LOCAL_BOUNDS_CULL_DISABLE");
	return enabled;
}

// P5.32 introduced this result cache as a separately opt-in experiment. P5.35
// promotes it for the already opt-in triangle task route after P5.32/P5.34
// exactness and residency proof. The historical force switch remains accepted
// for reproducible P5.32 experiments; the explicit disable switch selects the
// legacy task predicate for A/B measurement and regression control.
PX_FORCE_INLINE bool avbdUseRigidTriangleSurfaceFeatureForwardOwnerResultCache()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_RIGID_TRIANGLE_SURFACE_FEATURE_FORWARD_OWNER_RESULT_CACHE");
	return enabled;
}

PX_FORCE_INLINE bool avbdDisableRigidTriangleSurfaceFeatureForwardOwnerResultCache()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_RIGID_TRIANGLE_SURFACE_FEATURE_FORWARD_OWNER_RESULT_CACHE_DISABLE");
	return enabled;
}

// P5.9d is the first soft-pair OGC leaf. Its parent owns the mutable
// plan/refit epoch; children consume only frozen pair-plan ranges with their
// own output and query scratch. The route is opt-in while that boundary is
// being proven against the serial contact trace.
PX_FORCE_INLINE bool avbdUseSoftPairContactTaskFanIn()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_SOFT_PAIR_TASK_FANIN");
	return enabled;
}

PX_FORCE_INLINE bool avbdForceSoftPairContactTaskFanIn()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_FORCE_SOFT_PAIR_TASK_FANIN");
	return enabled;
}

// P5.10b consumes the already-proven self-BVH range contract. The parent
// refits one self body once; children only query disjoint VF/EE outer ranges.
// It remains opt-in until the Scene two-phase merge is accepted.
PX_FORCE_INLINE bool avbdUseSelfBvhContactTaskFanIn()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_SELF_BVH_TASK_FANIN");
	return enabled;
}

PX_FORCE_INLINE bool avbdForceSelfBvhContactTaskFanIn()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_FORCE_SELF_BVH_TASK_FANIN");
	return enabled;
}

// P4.2 builds the complete structural-plus-dynamic particle access plan only
// when explicitly requested.  Until a colored solve consumes it, keeping this
// diagnostic off preserves the authoritative serial GS hot path exactly.
PX_FORCE_INLINE bool avbdValidateParticlePrimalAccessPlan()
{
	static const bool enabled = []()
	{
		const char* value =
			std::getenv("PHYSX_AVBD_P4_VALIDATE_ACCESS_PLAN");
		return value && value[0] && value[0] != '0';
	}();
	return enabled;
}

// The ordered schedule preserves the legacy nonlinear-GS dependency order for
// reference and enhanced-determinism runs.  The relaxed schedule uses the
// same complete conflict graph, but applies a compact ordinary coloring so a
// production task graph can trade trajectory identity for throughput without
// ever admitting a shared read/write conflict.
enum class AvbdParticlePrimalSchedule : PxU8
{
	eDEFAULT,
	eSERIAL_LINEAR,
	eCOLORED_SERIAL,
	eRELAXED_COLOR
};

PX_FORCE_INLINE bool avbdUsesColoredParticlePrimalSchedule(
	AvbdParticlePrimalSchedule schedule)
{
	return schedule == AvbdParticlePrimalSchedule::eCOLORED_SERIAL ||
		schedule == AvbdParticlePrimalSchedule::eRELAXED_COLOR;
}

// Returns only an explicit process policy.  Scene resolves eDEFAULT against
// its worker count and determinism contract so ordinary production callers do
// not need an environment switch merely to reach the relaxed fast path.
PX_FORCE_INLINE AvbdParticlePrimalSchedule
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

PX_FORCE_INLINE AvbdParticlePrimalSchedule avbdGetParticlePrimalSchedule()
{
	const AvbdParticlePrimalSchedule configured =
		avbdGetConfiguredParticlePrimalSchedule();
	return configured == AvbdParticlePrimalSchedule::eDEFAULT
		? AvbdParticlePrimalSchedule::eSERIAL_LINEAR : configured;
}

// =============================================================================
// PxMat33 helper utilities (column-major <-> element access)
// =============================================================================

PX_FORCE_INLINE PxMat33 avbdOuter(const PxVec3& a, const PxVec3& b)
{
	return PxMat33(a * b.x, a * b.y, a * b.z);
}

PX_FORCE_INLINE PxMat33 avbdSkew(const PxVec3& v)
{
	return PxMat33(
		PxVec3(0.0f,  v.z, -v.y),
		PxVec3(-v.z,  0.0f,  v.x),
		PxVec3( v.y, -v.x,  0.0f));
}

PX_FORCE_INLINE PxVec3 avbdMatRow(const PxMat33& m, int row)
{
	return PxVec3(m.column0[row], m.column1[row], m.column2[row]);
}

PX_FORCE_INLINE PxReal avbdColSum(const PxVec3& col)
{
	return col.x + col.y + col.z;
}

PX_FORCE_INLINE PxVec3 avbdSolveSymmetric33(
	const PxMat33& matrix, const PxVec3& rhs)
{
	const PxReal a = matrix.column0.x;
	const PxReal b = matrix.column1.x;
	const PxReal c = matrix.column2.x;
	const PxReal d = matrix.column1.y;
	const PxReal e = matrix.column2.y;
	const PxReal f = matrix.column2.z;
	const PxReal adj00 = d * f - e * e;
	const PxReal adj01 = c * e - b * f;
	const PxReal adj02 = b * e - c * d;
	const PxReal adj11 = a * f - c * c;
	const PxReal adj12 = b * c - a * e;
	const PxReal adj22 = a * d - b * b;
	const PxReal determinant =
		a * adj00 + b * adj01 + c * adj02;
	if(determinant == 0.0f)
		return rhs;
	const PxReal inverseDeterminant = 1.0f / determinant;
	return PxVec3(
		adj00 * rhs.x + adj01 * rhs.y + adj02 * rhs.z,
		adj01 * rhs.x + adj11 * rhs.y + adj12 * rhs.z,
		adj02 * rhs.x + adj12 * rhs.y + adj22 * rhs.z) *
		inverseDeterminant;
}

// =============================================================================
// AvbdSoftParticle -- 3-DOF mass point (no rotation)
// =============================================================================

struct PX_ALIGN_PREFIX(16) AvbdSoftParticle
{
	PxVec3 position;
	PxReal invMass;

	PxVec3 velocity;
	PxReal mass;

	PxVec3 prevVelocity;
	PxReal damping;

	PxVec3 initialPosition;
	PxReal gravityScale;

	PxVec3 predictedPosition;
	PxReal elasticK;         // AVBD elastic proximal weight (adaptive)

	PxVec3 outerPosition;    // position snapshot at start of outer iteration (proximal anchor)
	PxReal elasticKMax;      // AVBD elastic proximal upper bound

	AvbdSoftParticle()
		: position(0.0f), invMass(1.0f), velocity(0.0f), mass(1.0f),
		  prevVelocity(0.0f), damping(0.0f), initialPosition(0.0f), gravityScale(1.0f),
		  predictedPosition(0.0f), elasticK(0.0f), outerPosition(0.0f), elasticKMax(1e6f) {}

	PX_FORCE_INLINE bool isStatic() const { return invMass <= 0.0f; }

	PX_FORCE_INLINE void computePrediction(PxReal dt, const PxVec3& gravity)
	{
		if (invMass <= 0.0f) return;
		predictedPosition =
			position + velocity * dt +
			gravity * (gravityScale * dt * dt);
		initialPosition = position;
	}

	// The VBD adaptive initial guess is intentionally separate from the
	// prediction above: predictedPosition remains the complete inertial target
	// and initialPosition remains the previous accepted state.  This lets the
	// position-level solve start closer to that target without changing the
	// velocity reconstruction contract.
	PX_FORCE_INLINE void computePredictionWithAdaptiveInitialGuess(
		PxReal dt, const PxVec3& gravity)
	{
		computePrediction(dt, gravity);
		if(invMass <= 0.0f || dt <= 0.0f)
			return;

		const PxReal gravityMagnitudeSq = gravity.magnitudeSquared();
		PxReal accelerationWeight = 0.0f;
		if(gravityMagnitudeSq > 1.0e-12f)
		{
			const PxVec3 acceleration = (velocity - prevVelocity) *
				(1.0f / dt);
			accelerationWeight = PxClamp(
				acceleration.dot(gravity) / gravityMagnitudeSq,
				0.0f, 1.0f);
		}

		const PxVec3 initialGuess =
			initialPosition + velocity * dt + gravity *
				(gravityScale * accelerationWeight * dt * dt);
		if(initialGuess.isFinite())
			position = initialGuess;
	}

	PX_FORCE_INLINE void updateVelocityFromPosition(PxReal invDt)
	{
		if (invMass <= 0.0f) return;
		prevVelocity = velocity;
		PxVec3 v = (position - initialPosition) * invDt;
		// NaN/inf guard: reset to zero on degenerate values
		if (v.x != v.x || PxAbs(v.x) > 1e6f ||
		    v.y != v.y || PxAbs(v.y) > 1e6f ||
		    v.z != v.z || PxAbs(v.z) > 1e6f)
		{
			velocity = PxVec3(0.0f);
			position = initialPosition;
		}
		else
			velocity = v;
	}
} PX_ALIGN_SUFFIX(16);

// =============================================================================
// VBD Element types -- precomputed rest-state data
// =============================================================================

struct AvbdTriElement
{
	PxU32 p0, p1, p2;
	PxU32 sourceElementIndex;
	PxReal DmInv00, DmInv01;
	PxReal DmInv10, DmInv11;
	PxReal restArea;
};

struct AvbdTetElement
{
	PxU32 p0, p1, p2, p3;
	PxU32 sourceElementIndex;
	PxMat33 DmInv;
	PxReal restVolume;
	// For vertex i, F*m_i = Ds*deformationGradientWeights[i].
	// The local block kernel also needs only ||m_i||^2 and det(DmInv);
	// precomputing them removes a matrix-matrix product and repeated
	// rest-gradient reconstruction from every nonlinear GS visit.
	PxVec3 deformationGradientWeights[4];
	PxVec3 shapeGradients[4];
	PxReal shapeGradientNormSq[4];
	PxReal inverseRestDeterminant;
};

struct AvbdTetVertexLinearization
{
	PxVec3 determinantGradient;
	PxReal determinant;
};

struct AvbdBendingElement
{
	PxU32 opp0, opp1;
	PxU32 edgeStart, edgeEnd;
	PxReal restShapeAngle;
	PxReal restAngle;
	PxReal restLength;
};

struct AvbdEdgeInfo
{
	PxU32 p0, p1;
	PxReal restLength;
	// Surface collision needs edges of the piecewise-smooth boundary, not
	// triangulation seams inside one planar patch. Structural edges and
	// manually authored fixtures remain active by default; buildSurfaceTriangles
	// classifies manifold surface edges from their two rest-pose face normals.
	PxU32 adjacentSurfaceFace0, adjacentSurfaceFace1;
	bool collisionFeature;

	AvbdEdgeInfo()
		: p0(PX_MAX_U32), p1(PX_MAX_U32), restLength(0.0f),
		  adjacentSurfaceFace0(PX_MAX_U32),
		  adjacentSurfaceFace1(PX_MAX_U32), collisionFeature(true)
	{
	}
};

// =============================================================================
// AVBD constraint types
// =============================================================================

// A soft attachment endpoint is one physical point sampled from up to four
// particles.  Vertex objectives are the count=1, weight=1 degenerate case.
// Keeping this point intact is important: a triangle/tetrahedron attachment is
// one vector equality with one AL state, not several independent vertex pins.
struct AvbdSoftPoint
{
	PxU32 particleIndices[4];
	PxReal weights[4];
	PxU32 particleCount;

	AvbdSoftPoint()
		: particleCount(1)
	{
		setVertex(0);
	}

	PX_FORCE_INLINE void setVertex(PxU32 particleIndex)
	{
		particleIndices[0] = particleIndex;
		weights[0] = 1.0f;
		for(PxU32 i = 1; i < 4; i++)
		{
			particleIndices[i] = PX_MAX_U32;
			weights[i] = 0.0f;
		}
		particleCount = 1;
	}

	PX_FORCE_INLINE bool operator==(const AvbdSoftPoint& other) const
	{
		if(particleCount != other.particleCount)
			return false;
		for(PxU32 i = 0; i < 4; i++)
		{
			if(particleIndices[i] != other.particleIndices[i] ||
				weights[i] != other.weights[i])
				return false;
		}
		return true;
	}
};

// Collision vertices are embedded in one simulation tetrahedron, while a
// point on a collision triangle can span three different tetrahedra.  Keep
// the fully expanded point in the prepared contact IR so every solver path
// consumes the same exact Jacobian instead of projecting the point back to a
// single simulation element.
static const PxU32 AVBD_EMBEDDED_VERTEX_SUPPORT = 4;
static const PxU32 AVBD_CONTACT_POINT_MAX_SUPPORT = 12;
static const PxU32 AVBD_CONTACT_MAX_PARTICLES = 24;

struct AvbdWeightedContactPoint
{
	PxU32 particleIndices[AVBD_CONTACT_POINT_MAX_SUPPORT];
	PxReal weights[AVBD_CONTACT_POINT_MAX_SUPPORT];
	PxU8 count;

	AvbdWeightedContactPoint() : count(0)
	{
		for(PxU32 i = 0; i < AVBD_CONTACT_POINT_MAX_SUPPORT; ++i)
		{
			particleIndices[i] = PX_MAX_U32;
			weights[i] = 0.0f;
		}
	}

	PX_FORCE_INLINE void clear()
	{
		count = 0;
		for(PxU32 i = 0; i < AVBD_CONTACT_POINT_MAX_SUPPORT; ++i)
		{
			particleIndices[i] = PX_MAX_U32;
			weights[i] = 0.0f;
		}
	}

	PX_FORCE_INLINE bool appendMerged(PxU32 particleIndex, PxReal weight)
	{
		if(particleIndex == PX_MAX_U32 || !PxIsFinite(weight))
			return false;
		for(PxU32 i = 0; i < count; ++i)
		{
			if(particleIndices[i] == particleIndex)
			{
				weights[i] += weight;
				return PxIsFinite(weights[i]);
			}
		}
		if(count >= AVBD_CONTACT_POINT_MAX_SUPPORT)
			return false;
		particleIndices[count] = particleIndex;
		weights[count] = weight;
		++count;
		return true;
	}

	PX_FORCE_INLINE void removeNearZero(PxReal epsilon = 1.0e-8f)
	{
		PxU32 writeIndex = 0;
		for(PxU32 i = 0; i < count; ++i)
		{
			if(PxAbs(weights[i]) <= epsilon)
				continue;
			particleIndices[writeIndex] = particleIndices[i];
			weights[writeIndex] = weights[i];
			++writeIndex;
		}
		for(PxU32 i = writeIndex; i < AVBD_CONTACT_POINT_MAX_SUPPORT; ++i)
		{
			particleIndices[i] = PX_MAX_U32;
			weights[i] = 0.0f;
		}
		count = PxU8(writeIndex);
	}

	PX_FORCE_INLINE void setVertex(PxU32 particleIndex)
	{
		clear();
		particleIndices[0] = particleIndex;
		weights[0] = 1.0f;
		count = 1;
	}
};

PX_FORCE_INLINE bool avbdIsSoftPointValid(
	const AvbdSoftPoint& point, PxU32 particleStart, PxU32 particleCount)
{
	if(point.particleCount == 0 || point.particleCount > 4)
		return false;
	for(PxU32 i = 0; i < point.particleCount; i++)
	{
		if(point.particleIndices[i] - particleStart >= particleCount ||
			!PxIsFinite(point.weights[i]))
			return false;
	}
	return true;
}

PX_FORCE_INLINE PxVec3 avbdGetSoftPointPosition(
	const AvbdSoftPoint& point, const AvbdSoftParticle* particles)
{
	PxVec3 position(0.0f);
	for(PxU32 i = 0; i < point.particleCount; i++)
		position +=
			particles[point.particleIndices[i]].position * point.weights[i];
	return position;
}

PX_FORCE_INLINE PxReal avbdGetSoftPointJacobianWeight(
	const AvbdSoftPoint& point, PxU32 particleIndex)
{
	PxReal weight = 0.0f;
	for(PxU32 i = 0; i < point.particleCount; i++)
	{
		if(point.particleIndices[i] == particleIndex)
			weight += point.weights[i];
	}
	return weight;
}

PX_FORCE_INLINE PxReal avbdGetSoftPointInverseMass(
	const AvbdSoftPoint& point, const AvbdSoftParticle* particles,
	PxU32 numParticles)
{
	PxReal inverseMass = 0.0f;
	for(PxU32 i = 0; i < point.particleCount; i++)
	{
		const PxU32 particleIndex = point.particleIndices[i];
		if(particleIndex >= numParticles)
			return 0.0f;
		bool firstOccurrence = true;
		for(PxU32 previous = 0; previous < i; previous++)
		{
			if(point.particleIndices[previous] == particleIndex)
			{
				firstOccurrence = false;
				break;
			}
		}
		if(!firstOccurrence)
			continue;
		const PxReal jacobianWeight =
			avbdGetSoftPointJacobianWeight(point, particleIndex);
		inverseMass += jacobianWeight * jacobianWeight *
			particles[particleIndex].invMass;
	}
	return inverseMass;
}

enum class AvbdSoftAttachmentTargetKind : PxU8
{
	eDYNAMIC_RIGID,
	eARTICULATION_LINK,
	eDYNAMIC_SOFT,
	eUNSUPPORTED
};

struct AvbdSoftAttachment
{
	AvbdSoftPoint point;
	AvbdSoftPoint targetPoint;
	PxU32 rigidBodyIdx;
	PxU32 sourceHandle;
	AvbdSoftAttachmentTargetKind targetKind;
	PxVec3 localOffset;
	PxVec3 alLambda;
	PxReal k;
	PxReal kMax;

	AvbdSoftAttachment()
		: rigidBodyIdx(0), sourceHandle(PX_MAX_U32),
		  targetKind(AvbdSoftAttachmentTargetKind::eDYNAMIC_RIGID),
		  localOffset(0.0f),
		  alLambda(0.0f), k(1e3f), kMax(1e5f) {}
};

enum class AvbdSoftPinTargetKind : PxU8
{
	eWORLD_FIXED,
	ePRESCRIBED_RIGID,
	eDEFORMABLE_KINEMATIC,
	eUNSUPPORTED
};

struct AvbdKinematicPin
{
	AvbdSoftPoint point;
	PxU32 sourceHandle;
	AvbdSoftPinTargetKind targetKind;
	PxVec3 worldTarget;
	PxVec3 previousWorldTarget;
	PxVec3 alLambda;
	PxReal k;
	PxReal kMax;

	AvbdKinematicPin()
		: sourceHandle(PX_MAX_U32),
		  targetKind(AvbdSoftPinTargetKind::eWORLD_FIXED),
		  worldTarget(0.0f), previousWorldTarget(0.0f),
		  alLambda(0.0f),
		  k(1e4f), kMax(1e6f) {}
};

// Closest-point feature type for OGC block classification.
enum AvbdClosestFeature
{
	AVBD_FEATURE_FACE,
	AVBD_FEATURE_EDGE,
	AVBD_FEATURE_VERTEX,
	AVBD_FEATURE_UNKNOWN
};

// Stable identity assigned during contact prep.  The solver state must follow
// the physical source objective, not whichever contact happens to be closest
// in the rebuilt array.
struct AvbdSoftContactSource
{
	enum Type
	{
		eINVALID,
		eGROUND,
		eRIGID_SDF,
		eSOFT_SURFACE,
		eSELF_SURFACE
	};

	Type type;
	PxU32 targetBodyIndex;
	PxU64 primitiveKey;
	PxU64 featureKey;

	AvbdSoftContactSource()
		: type(eINVALID), targetBodyIndex(PX_MAX_U32),
		  primitiveKey(0), featureKey(0)
	{
	}

	AvbdSoftContactSource(
		Type sourceType, PxU32 bodyIndex,
		PxU64 sourcePrimitiveKey, PxU64 sourceFeatureKey)
		: type(sourceType), targetBodyIndex(bodyIndex),
		  primitiveKey(sourcePrimitiveKey), featureKey(sourceFeatureKey)
	{
	}

	PX_FORCE_INLINE bool isValid() const
	{
		return type != eINVALID;
	}

	PX_FORCE_INLINE bool operator==(const AvbdSoftContactSource& other) const
	{
		return type == other.type &&
			targetBodyIndex == other.targetBodyIndex &&
			primitiveKey == other.primitiveKey &&
			featureKey == other.featureKey;
	}
};

PX_FORCE_INLINE PxU64 avbdSoftContactHashValue(PxU64 hash, PxU32 value)
{
	hash ^= PxU64(value);
	return hash * 1099511628211ull;
}

PX_FORCE_INLINE PxU64 avbdSoftTrianglePrimitiveKey(
	PxU32 vertex0, PxU32 vertex1, PxU32 vertex2)
{
	if(vertex1 < vertex0)
	{
		const PxU32 tmp = vertex0;
		vertex0 = vertex1;
		vertex1 = tmp;
	}
	if(vertex2 < vertex1)
	{
		const PxU32 tmp = vertex1;
		vertex1 = vertex2;
		vertex2 = tmp;
	}
	if(vertex1 < vertex0)
	{
		const PxU32 tmp = vertex0;
		vertex0 = vertex1;
		vertex1 = tmp;
	}

	PxU64 hash = 1469598103934665603ull;
	hash = avbdSoftContactHashValue(hash, vertex0);
	hash = avbdSoftContactHashValue(hash, vertex1);
	return avbdSoftContactHashValue(hash, vertex2);
}

PX_FORCE_INLINE PxU64 avbdSoftTriangleFeatureKey(
	PxU32 vertex0, PxU32 vertex1, PxU32 vertex2,
	AvbdClosestFeature feature, PxU32 localFeatureIndex)
{
	PxU32 featureVertex0 = vertex0;
	PxU32 featureVertex1 = vertex1;
	PxU32 featureVertex2 = vertex2;
	PxU32 featureVertexCount = 3;

	if(feature == AVBD_FEATURE_VERTEX)
	{
		const PxU32 vertices[3] = {vertex0, vertex1, vertex2};
		featureVertex0 = vertices[PxMin(localFeatureIndex, 2u)];
		featureVertexCount = 1;
	}
	else if(feature == AVBD_FEATURE_EDGE)
	{
		if(localFeatureIndex == 0)
		{
			featureVertex0 = vertex0;
			featureVertex1 = vertex1;
		}
		else if(localFeatureIndex == 1)
		{
			featureVertex0 = vertex0;
			featureVertex1 = vertex2;
		}
		else
		{
			featureVertex0 = vertex1;
			featureVertex1 = vertex2;
		}
		if(featureVertex1 < featureVertex0)
		{
			const PxU32 tmp = featureVertex0;
			featureVertex0 = featureVertex1;
			featureVertex1 = tmp;
		}
		featureVertexCount = 2;
	}
	else
	{
		if(featureVertex1 < featureVertex0)
		{
			const PxU32 tmp = featureVertex0;
			featureVertex0 = featureVertex1;
			featureVertex1 = tmp;
		}
		if(featureVertex2 < featureVertex1)
		{
			const PxU32 tmp = featureVertex1;
			featureVertex1 = featureVertex2;
			featureVertex2 = tmp;
		}
		if(featureVertex1 < featureVertex0)
		{
			const PxU32 tmp = featureVertex0;
			featureVertex0 = featureVertex1;
			featureVertex1 = tmp;
		}
	}

	PxU64 hash = 1469598103934665603ull;
	hash = avbdSoftContactHashValue(hash, PxU32(feature));
	hash = avbdSoftContactHashValue(hash, featureVertex0);
	if(featureVertexCount > 1)
		hash = avbdSoftContactHashValue(hash, featureVertex1);
	if(featureVertexCount > 2)
		hash = avbdSoftContactHashValue(hash, featureVertex2);
	return hash;
}

enum class AvbdSoftContactTargetKind : PxU8
{
	eWORLD_STATIC,
	eKINEMATIC_RIGID,
	eDEFORMABLE_SURFACE,
	eRIGID_BODY,
	eUNSUPPORTED
};

// Tangential contact response can have a different owner from the geometric
// normal.  This is deliberately separate from AvbdVelocityObjectiveOwner:
// changing the latter would alter the normal Position-AL / rigid IR program.
enum class AvbdSoftContactTangentOwner : PxU8
{
	ePOSITION_AL,
	eVELOCITY
};

// Immutable-for-the-solve output of contact prep.  This record identifies the
// physical objective and contains only geometry/material data.  Target kind
// and velocity owner are explicit prep IR; no sentinel, target kind inference,
// or repeated bit-flag tests may select a later solve stage.
struct AvbdSoftContactGeometry
{
	AvbdSoftContactSource source;
	PxU32 particleIdx;
	PxU32 collisionFeatureParticleIdx;
	// Explicit ownership and collision-domain element identities.  The legacy
	// representative particle remains only as detector feature identity.
	PxU32 queryBodyIndex;
	PxU32 queryCollisionElementIndex;
	PxU32 targetCollisionElementIndex;
	AvbdSoftContactTargetKind targetKind;
	AvbdVelocityObjectiveOwner velocityOwner;
	AvbdSoftContactTangentOwner tangentOwner;
	PxU32 targetIndex;
	// The deformable point on the query side. Legacy vertex contacts leave
	// queryParticleIndices[0] invalid and use particleIdx with unit weight.
	// Edge/face contacts store a barycentric point here so one geometric
	// contact owns one AL state while its block contributions are distributed
	// to every incident particle.
	PxU32 queryParticleIndices[3];
	PxReal queryWeights[3];
	AvbdWeightedContactPoint queryPoint;
	// Source-domain element on the deformable target selected by contact
	// prep. For a Surface target this is the public triangle index.
	PxU32 targetSourceElementIndex;
	PxVec3 normal;          // penalty direction (closest-point, VBD-stable)
	PxVec3 projNormal;      // projection direction (face-normal corrected, always outward)
	PxReal depth;
	PxReal margin;          // contact shell thickness (used for proximity contacts)
	PxReal friction;
	PxVec3 tangent1, tangent2;
	PxVec3 surfacePoint;    // reference point on the other body's surface (world space)
	PxVec3 kinematicSurfacePointPrevious;
	PxU32 surfaceParticleIndices[3];
	PxReal surfaceWeights[3];
	AvbdWeightedContactPoint targetPoint;
	PxVec3 rigidLocalPoint;
	// A current-pose rigid-box SDF row keeps enough immutable box geometry for
	// a later post-AL recovery to re-query the *current* expanded soft point.
	// For a dynamic rigid target rigidBoxPose is shape-to-rigid-body; for a
	// world-static target it is the world-space box pose. Ground and non-box
	// SDF rows leave this flag false.
	PxVec3 rigidBoxHalfExtent;
	PxTransform rigidBoxPose;
	bool hasRigidBoxSdf;
	// A triangle/box core row is a discrete current-pose witness for a
	// collision triangle whose interior cuts an OBB. Its regular barycentric
	// query is sufficient for the AL solve,
	// but not necessarily for a final whole-body escape: the query can be
	// outside while another patch of the same triangle remains inside. Keep
	// the minimum rigid translation that separates that complete triangle.
	// This is set only by the current-pose triangle/box core detector, never
	// by a swept/CCD path.
	// Box-local unit normal. The detector stores the certificate in the box
	// frame so a dynamic target can rotate it to its current pose before a
	// fresh endpoint recovery consumes it.
	PxVec3 rigidBoxTriangleCoreExitNormalLocal;
	PxReal rigidBoxTriangleCoreExitDistance;
	bool hasRigidBoxTriangleCoreExit;
	// Detection-time bounds of the complete collision triangle in the rigid
	// box frame.  A single centroid query is enough for Position-AL, whereas a
	// terminal whole-body DCD escape must be able to choose one common box face
	// for every core triangle of a soft body.  These bounds make that choice
	// exact without treating the OGC shell or a swept path as penetration.
	PxVec3 rigidBoxTriangleCoreMinimumLocal;
	PxVec3 rigidBoxTriangleCoreMaximumLocal;
	// Proxy-domain identity of the complete collision triangle.  Core rows may
	// use one vertex as their AL query, but expansion still needs all three
	// embedded vertices to keep pair-level geometry exact.
	PxU32 rigidBoxTriangleCoreCollisionParticleIndices[3];
	// The three collision-triangle vertices expanded independently into the
	// simulation domain.  Unlike queryPoint (the compact centroid objective),
	// these supports let the unified soft/rigid OGC trust region re-evaluate the
	// complete triangle against the *current candidate* OBB pose.
	AvbdWeightedContactPoint rigidBoxTriangleCorePoints[3];

	AvbdSoftContactGeometry()
		: source(), particleIdx(0), collisionFeatureParticleIdx(PX_MAX_U32),
		  queryBodyIndex(PX_MAX_U32),
		  queryCollisionElementIndex(PX_MAX_U32),
		  targetCollisionElementIndex(PX_MAX_U32),
		  targetKind(AvbdSoftContactTargetKind::eUNSUPPORTED),
		  velocityOwner(
			  AvbdVelocityObjectiveOwner::Unsupported),
		  tangentOwner(AvbdSoftContactTangentOwner::ePOSITION_AL),
		  targetIndex(PX_MAX_U32),
		  queryParticleIndices{
			  PX_MAX_U32, PX_MAX_U32, PX_MAX_U32},
		  queryWeights{0.0f, 0.0f, 0.0f},
		  targetSourceElementIndex(PX_MAX_U32),
		  normal(0.0f, 1.0f, 0.0f),
		  projNormal(0.0f, 1.0f, 0.0f),
		  depth(0.0f), margin(0.0f), friction(0.5f),
		  tangent1(1.0f, 0.0f, 0.0f),
		  tangent2(0.0f, 0.0f, 1.0f), surfacePoint(0.0f),
		  kinematicSurfacePointPrevious(0.0f),
		  surfaceParticleIndices{PX_MAX_U32, PX_MAX_U32, PX_MAX_U32},
		  surfaceWeights{0.0f, 0.0f, 0.0f},
		  rigidLocalPoint(0.0f), rigidBoxHalfExtent(0.0f),
		  rigidBoxPose(PxIdentity), hasRigidBoxSdf(false),
		  rigidBoxTriangleCoreExitNormalLocal(0.0f),
		  rigidBoxTriangleCoreExitDistance(0.0f),
		  hasRigidBoxTriangleCoreExit(false),
		  rigidBoxTriangleCoreMinimumLocal(0.0f),
		  rigidBoxTriangleCoreMaximumLocal(0.0f),
		  rigidBoxTriangleCoreCollisionParticleIndices{
			  PX_MAX_U32, PX_MAX_U32, PX_MAX_U32}
	{
	}

	PX_FORCE_INLINE bool hasBarycentricQueryPoint() const
	{
		return queryParticleIndices[0] != PX_MAX_U32;
	}

	PX_FORCE_INLINE bool hasWeightedQueryPoint() const
	{
		return queryPoint.count != 0;
	}

	PX_FORCE_INLINE bool hasWeightedTargetPoint() const
	{
		return targetPoint.count != 0;
	}

	PX_FORCE_INLINE bool hasDeformableSurfaceTarget() const
	{
		return
			targetKind ==
				AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE &&
			surfaceParticleIndices[0] != PX_MAX_U32;
	}

	PX_FORCE_INLINE bool hasWorldStaticTarget() const
	{
		return
			targetKind == AvbdSoftContactTargetKind::eWORLD_STATIC;
	}

	PX_FORCE_INLINE bool hasKinematicRigidTarget() const
	{
		return
			targetKind ==
				AvbdSoftContactTargetKind::eKINEMATIC_RIGID;
	}

	PX_FORCE_INLINE bool hasRigidBodyTarget() const
	{
		return
			targetKind == AvbdSoftContactTargetKind::eRIGID_BODY &&
			targetIndex != PX_MAX_U32;
	}
};

// The rigid-side shell path is valid only when the complete deformable query
// point is prescribed.  A collision vertex embedded in a simulation tet can
// have a pinned first endpoint and movable remaining endpoints, so the legacy
// representative particle is not an ownership test.
PX_FORCE_INLINE bool avbdIsSoftContactQueryFullyKinematic(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftParticle* particles, PxU32 numParticles)
{
	bool hasSupport = false;
	if(geometry.hasWeightedQueryPoint())
	{
		for(PxU32 i = 0; i < geometry.queryPoint.count; ++i)
		{
			if(PxAbs(geometry.queryPoint.weights[i]) <= 1.0e-8f)
				continue;
			const PxU32 particleIndex =
				geometry.queryPoint.particleIndices[i];
			if(particleIndex >= numParticles ||
				particles[particleIndex].invMass > 0.0f)
				return false;
			hasSupport = true;
		}
		return hasSupport;
	}
	if(geometry.hasBarycentricQueryPoint())
	{
		for(PxU32 i = 0; i < 3; ++i)
		{
			if(geometry.queryParticleIndices[i] == PX_MAX_U32)
				break;
			if(PxAbs(geometry.queryWeights[i]) <= 1.0e-8f)
				continue;
			const PxU32 particleIndex =
				geometry.queryParticleIndices[i];
			if(particleIndex >= numParticles ||
				particles[particleIndex].invMass > 0.0f)
				return false;
			hasSupport = true;
		}
		return hasSupport;
	}
	return geometry.particleIdx < numParticles &&
		particles[geometry.particleIdx].invMass <= 0.0f;
}

// Dynamic rigid feedback must use the same complete query support as the
// shell classification above.  Keep this as an explicit positive predicate:
// a malformed or empty weighted point is neither a valid dynamic endpoint nor
// a reason to dereference the legacy representative particle.
PX_FORCE_INLINE bool avbdHasSoftContactDynamicQuerySupport(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftParticle* particles, PxU32 numParticles)
{
	bool hasSupport = false;
	bool hasDynamicSupport = false;
	if(geometry.hasWeightedQueryPoint())
	{
		for(PxU32 i = 0; i < geometry.queryPoint.count; ++i)
		{
			if(PxAbs(geometry.queryPoint.weights[i]) <= 1.0e-8f)
				continue;
			const PxU32 particleIndex =
				geometry.queryPoint.particleIndices[i];
			if(particleIndex >= numParticles)
				return false;
			hasSupport = true;
			hasDynamicSupport = hasDynamicSupport ||
				particles[particleIndex].invMass > 0.0f;
		}
		return hasSupport && hasDynamicSupport;
	}
	if(geometry.hasBarycentricQueryPoint())
	{
		for(PxU32 i = 0; i < 3; ++i)
		{
			if(geometry.queryParticleIndices[i] == PX_MAX_U32)
				break;
			if(PxAbs(geometry.queryWeights[i]) <= 1.0e-8f)
				continue;
			const PxU32 particleIndex = geometry.queryParticleIndices[i];
			if(particleIndex >= numParticles)
				return false;
			hasSupport = true;
			hasDynamicSupport = hasDynamicSupport ||
				particles[particleIndex].invMass > 0.0f;
		}
		return hasSupport && hasDynamicSupport;
	}
	return geometry.particleIdx < numParticles &&
		particles[geometry.particleIdx].invMass > 0.0f;
}

// Persistent primal/dual state owned by exactly one prepared contact source.
// Detection may rebuild geometry, but this state is transferred only when the
// source identity matches.
struct AvbdSoftContactAugmentedState
{
	// Kinematic shell rigid coupling (invMass=0 shell particle, moving surfacePoint).
	PxVec3 surfacePointPrev;
	PxVec3 particlePointPrev;
	PxReal alLambda;
	PxReal alLambdaTangent[2];
	PxReal penTangent[2];
	bool frictionStick;
	// The permitted residual normal constraint for this frame. A negative
	// value leaves deep overlap in place after exactly
	// maxDepenetrationVelocity * dt of contact-owned recovery. Redetection
	// transfers this anchor, but the next timestep explicitly rebuilds it.
	PxReal depenetrationConstraintOffset;
	bool depenetrationLimitInitialized;

	PxReal k;
	PxReal ke;

	AvbdSoftContactAugmentedState()
		: surfacePointPrev(0.0f), particlePointPrev(0.0f),
		  alLambda(0.0f), alLambdaTangent{0.0f, 0.0f},
		  penTangent{1000.0f, 1000.0f}, frictionStick(false),
		  depenetrationConstraintOffset(0.0f),
		  depenetrationLimitInitialized(false),
		  k(1e4f), ke(1e6f)
	{
	}
};

// Prepared contact objective.  Geometry and augmented state have distinct
// ownership and may only be consumed through their named records.
struct AvbdSoftContact
{
	AvbdSoftContactGeometry geometry;
	AvbdSoftContactAugmentedState state;
};

// Shared OGC epoch state.  A pair has exactly one identity across the
// component fallback and the native mixed-island solver; only the target
// response differs (deformable/static here, 6DOF rigid in the native path).
// Keeping the scheduler state above either solve path prevents a per-row
// repair from silently becoming a second contact owner.
struct AvbdOgcPairState
{
	AvbdSoftContactSource::Type sourceType;
	AvbdSoftContactTargetKind targetKind;
	PxU32 sourceBodyIndex;
	PxU32 targetBodyIndex;
	PxU64 primitiveKey;
	PxU32 contactCount;
	PxU32 representativeContact;
	PxU32 admissionContact;
	PxU8 triangleCoreFace;
	PxReal triangleCoreFaceExit;
	bool hasTriangleCoreManifold;
	bool triangleCoreLocallyResolved;
	PxVec3 representativeNormal;
	PxVec3 representativeRigidOffset;
	// Relative query-to-surface vector at the beginning of this DCD epoch.
	// It lets both the native 6DOF path and the component scheduler decide
	// whether subsequent position updates consumed this pair's discrete
	// safety domain, without treating a mere OGC shell admission as a refresh.
	PxVec3 referenceRelativePoint;
	PxReal representativeGap;
	PxReal accumulatedNormalLambda;
	// Normal work admitted at the current mixed OGC boundary.  Endpoint
	// admission may clip the rigid endpoint to the trust-region boundary;
	// retain the clipped relative motion as a pair-owned load so the soft
	// material solve can absorb it instead of losing the pressure entirely.
	PxReal admittedNormalDisplacement;
	PxReal admittedNormalLoad;
	// These values describe a DCD epoch, not the AL multiplier state.
	// Component scheduling consumes the same pair-wide safe displacement
	// budget that native soft/rigid candidate filtering consumes.
	PxReal safetyGap;
	PxReal remainingSafeDisplacement;
	PxReal accumulatedRelativeDisplacement;
	PxReal referenceGap;
	PxReal minimumGap;
	PxU32 epoch;
	bool active;
	bool admittedAtBoundary;
	bool refreshRequested;

	AvbdOgcPairState()
		: sourceType(AvbdSoftContactSource::eINVALID),
		  targetKind(AvbdSoftContactTargetKind::eUNSUPPORTED),
		  sourceBodyIndex(PX_MAX_U32), targetBodyIndex(PX_MAX_U32),
		  primitiveKey(~PxU64(0)), contactCount(0),
		  representativeContact(PX_MAX_U32), admissionContact(PX_MAX_U32),
		  triangleCoreFace(PX_MAX_U8), triangleCoreFaceExit(0.0f),
		  hasTriangleCoreManifold(false),
		  triangleCoreLocallyResolved(false), representativeNormal(0.0f),
		  representativeRigidOffset(0.0f), referenceRelativePoint(0.0f),
		  representativeGap(PX_MAX_F32),
		  accumulatedNormalLambda(0.0f), admittedNormalDisplacement(0.0f),
		  admittedNormalLoad(0.0f), safetyGap(PX_MAX_F32),
		  remainingSafeDisplacement(0.0f),
		  accumulatedRelativeDisplacement(0.0f), referenceGap(PX_MAX_F32),
		  minimumGap(PX_MAX_F32), epoch(0), active(false),
		  admittedAtBoundary(false), refreshRequested(false)
	{
	}
};

PX_FORCE_INLINE PxReal avbdGetOgcPairNormalLoadPerContact(
	const AvbdOgcPairState& pair)
{
	if(!pair.active || !pair.admittedAtBoundary ||
		!PxIsFinite(pair.admittedNormalLoad) ||
		pair.admittedNormalLoad <= 0.0f || pair.contactCount == 0u)
		return 0.0f;
	return pair.admittedNormalLoad /
		static_cast<PxReal>(PxMax(1u, pair.contactCount));
}

PX_FORCE_INLINE PxVec3 avbdGetSoftContactSurfacePoint(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftParticle* particles)
{
	if(geometry.hasWeightedTargetPoint())
	{
		PxVec3 point(0.0f);
		for(PxU32 i = 0; i < geometry.targetPoint.count; ++i)
			point += particles[geometry.targetPoint.particleIndices[i]].position *
				geometry.targetPoint.weights[i];
		return point;
	}
	if(!geometry.hasDeformableSurfaceTarget())
		return geometry.surfacePoint;

	PxVec3 surfacePoint(0.0f);
	for(PxU32 i = 0; i < 3; i++)
	{
		if(geometry.surfaceParticleIndices[i] == PX_MAX_U32)
			break;
		surfacePoint +=
			particles[geometry.surfaceParticleIndices[i]].position *
			geometry.surfaceWeights[i];
	}
	return surfacePoint;
}

PX_FORCE_INLINE PxVec3 avbdGetSoftContactQueryPoint(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftParticle* particles)
{
	if(geometry.hasWeightedQueryPoint())
	{
		PxVec3 point(0.0f);
		for(PxU32 i = 0; i < geometry.queryPoint.count; ++i)
			point += particles[geometry.queryPoint.particleIndices[i]].position *
				geometry.queryPoint.weights[i];
		return point;
	}
	if(!geometry.hasBarycentricQueryPoint())
		return particles[geometry.particleIdx].position;

	PxVec3 queryPoint(0.0f);
	for(PxU32 i = 0; i < 3; i++)
	{
		if(geometry.queryParticleIndices[i] == PX_MAX_U32)
			break;
		queryPoint +=
			particles[geometry.queryParticleIndices[i]].position *
			geometry.queryWeights[i];
	}
	return queryPoint;
}

PX_FORCE_INLINE PxVec3 avbdGetSoftContactInitialSurfacePoint(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftParticle* particles)
{
	if(geometry.hasWeightedTargetPoint())
	{
		PxVec3 point(0.0f);
		for(PxU32 i = 0; i < geometry.targetPoint.count; ++i)
			point += particles[geometry.targetPoint.particleIndices[i]].initialPosition *
				geometry.targetPoint.weights[i];
		return point;
	}
	if(!geometry.hasDeformableSurfaceTarget())
		return geometry.surfacePoint;

	PxVec3 surfacePoint(0.0f);
	for(PxU32 i = 0; i < 3; i++)
	{
		if(geometry.surfaceParticleIndices[i] == PX_MAX_U32)
			break;
		surfacePoint +=
			particles[geometry.surfaceParticleIndices[i]].
				initialPosition * geometry.surfaceWeights[i];
	}
	return surfacePoint;
}

PX_FORCE_INLINE PxVec3 avbdGetSoftContactInitialQueryPoint(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftParticle* particles)
{
	if(geometry.hasWeightedQueryPoint())
	{
		PxVec3 point(0.0f);
		for(PxU32 i = 0; i < geometry.queryPoint.count; ++i)
			point += particles[geometry.queryPoint.particleIndices[i]].initialPosition *
				geometry.queryPoint.weights[i];
		return point;
	}
	if(!geometry.hasBarycentricQueryPoint())
		return particles[geometry.particleIdx].initialPosition;

	PxVec3 queryPoint(0.0f);
	for(PxU32 i = 0; i < 3; i++)
	{
		if(geometry.queryParticleIndices[i] == PX_MAX_U32)
			break;
		queryPoint +=
			particles[geometry.queryParticleIndices[i]].
				initialPosition * geometry.queryWeights[i];
	}
	return queryPoint;
}

PX_FORCE_INLINE PxReal avbdEvaluateSoftContactNormalConstraint(
	const AvbdSoftContactGeometry& geometry,
	const PxVec3& queryPoint,
	const PxVec3& currentSurfacePoint)
{
	return
		(queryPoint - currentSurfacePoint).dot(geometry.normal) -
		(geometry.source.type == AvbdSoftContactSource::eGROUND
			? 0.0f : geometry.margin);
}

PX_FORCE_INLINE PxReal avbdGetSoftContactParticleJacobianScale(
	const AvbdSoftContactGeometry& geometry, PxU32 particleIdx)
{
	PxReal scale = 0.0f;
	if(geometry.hasWeightedQueryPoint())
	{
		for(PxU32 i = 0; i < geometry.queryPoint.count; ++i)
			if(geometry.queryPoint.particleIndices[i] == particleIdx)
				scale += geometry.queryPoint.weights[i];
	}
	else if(geometry.hasBarycentricQueryPoint())
	{
		for(PxU32 i = 0; i < 3; i++)
		{
			if(geometry.queryParticleIndices[i] == PX_MAX_U32)
				break;
			if(geometry.queryParticleIndices[i] == particleIdx)
				scale += geometry.queryWeights[i];
		}
	}
	else if(geometry.particleIdx == particleIdx)
		scale = 1.0f;
	if(geometry.hasWeightedTargetPoint())
	{
		for(PxU32 i = 0; i < geometry.targetPoint.count; ++i)
			if(geometry.targetPoint.particleIndices[i] == particleIdx)
				scale -= geometry.targetPoint.weights[i];
	}
	else if(geometry.hasDeformableSurfaceTarget())
	{
		for(PxU32 i = 0; i < 3; i++)
		{
			if(geometry.surfaceParticleIndices[i] == PX_MAX_U32)
				break;
			if(geometry.surfaceParticleIndices[i] == particleIdx)
				scale -= geometry.surfaceWeights[i];
		}
	}
	return scale;
}

PX_FORCE_INLINE PxU32 avbdCollectSoftContactParticleIndices(
	const AvbdSoftContactGeometry& geometry,
	PxU32 (&indices)[AVBD_CONTACT_MAX_PARTICLES])
{
	PxU32 count = 0;
	auto appendUnique = [&indices, &count](PxU32 particleIndex)
	{
		if(particleIndex == PX_MAX_U32)
			return;
		for(PxU32 i = 0; i < count; i++)
			if(indices[i] == particleIndex)
				return;
		indices[count++] = particleIndex;
	};

	if(geometry.hasWeightedQueryPoint())
	{
		for(PxU32 i = 0; i < geometry.queryPoint.count; ++i)
			appendUnique(geometry.queryPoint.particleIndices[i]);
	}
	else if(geometry.hasBarycentricQueryPoint())
	{
		for(PxU32 i = 0; i < 3; i++)
			appendUnique(geometry.queryParticleIndices[i]);
	}
	else
		appendUnique(geometry.particleIdx);

	if(geometry.hasWeightedTargetPoint())
	{
		for(PxU32 i = 0; i < geometry.targetPoint.count; ++i)
			appendUnique(geometry.targetPoint.particleIndices[i]);
	}
	else if(geometry.hasDeformableSurfaceTarget())
	{
		for(PxU32 i = 0; i < 3; i++)
			appendUnique(geometry.surfaceParticleIndices[i]);
	}
	return count;
}

// Persistent OGC sweep entries.  These are deliberately compact, temporary
// query records rather than compiled contact identity: every redetection still
// refits their bounds from the authoritative particle positions and preserves
// the existing canonical traversal/order.
struct AvbdSelfCollisionTriangleBounds
{
	PxU32 triangleOffset;
	PxVec3 minimum;
	PxVec3 maximum;
};

struct AvbdSelfCollisionVertexSweepEntry
{
	PxU32 localIndex;
	PxReal minimumX;
	PxReal maximumX;
};

struct AvbdSelfCollisionEdgeBounds
{
	PxU32 edgeIndex;
	PxVec3 minimum;
	PxVec3 maximum;
};

struct AvbdSoftPairEdgeBounds
{
	PxU32 edgeIndex;
	PxVec3 minimum;
	PxVec3 maximum;
	PxVec3 adjacentNormal0;
	PxVec3 adjacentNormal1;
	bool hasExteriorNormalCone;
};

// A convex surface edge owns only the positive cone spanned by the outward
// normals of its two incident faces. Distance alone is not sufficient for an
// edge-edge contact: accepting a direction outside either edge's normal cone
// adds a separating plane that is not a feature of the Minkowski boundary.
// Such planes are especially visible at box corners, where they form an
// artificial wedge and pin an otherwise sliding contact.
PX_FORCE_INLINE bool avbdIsDirectionInSurfaceEdgeNormalCone(
	const PxVec3& direction,
	const PxVec3& adjacentNormal0,
	const PxVec3& adjacentNormal1)
{
	const PxReal directionLengthSq = direction.magnitudeSquared();
	const PxReal normalLengthSq0 = adjacentNormal0.magnitudeSquared();
	const PxReal normalLengthSq1 = adjacentNormal1.magnitudeSquared();
	if(directionLengthSq <= 1.0e-12f ||
		normalLengthSq0 <= 1.0e-12f ||
		normalLengthSq1 <= 1.0e-12f)
		return true;

	const PxVec3 unitDirection =
		direction * PxRecipSqrt(directionLengthSq);
	const PxVec3 normal0 =
		adjacentNormal0 * PxRecipSqrt(normalLengthSq0);
	const PxVec3 normal1 =
		adjacentNormal1 * PxRecipSqrt(normalLengthSq1);
	const PxReal normalDot = PxClamp(normal0.dot(normal1), -1.0f, 1.0f);
	const PxReal determinant = 1.0f - normalDot * normalDot;
	// A folded or numerically degenerate two-face edge has no stable 2-D cone.
	// Preserve the historical contact in that rare case; the normal-owner test
	// below is for well-defined collision creases, not a topology repair path.
	if(determinant <= 1.0e-6f)
		return true;

	const PxReal directionDot0 = unitDirection.dot(normal0);
	const PxReal directionDot1 = unitDirection.dot(normal1);
	const PxReal coefficient0 =
		(directionDot0 - normalDot * directionDot1) / determinant;
	const PxReal coefficient1 =
		(directionDot1 - normalDot * directionDot0) / determinant;
	const PxReal coefficientTolerance = 1.0e-3f;
	if(coefficient0 < -coefficientTolerance ||
		coefficient1 < -coefficientTolerance)
		return false;

	// Segment closest-point normals should be perpendicular to the shared edge
	// and therefore lie in the incident-normal plane. Reject a direction with
	// a material out-of-plane component instead of silently projecting it into
	// a valid-looking cone.
	const PxVec3 reconstructed =
		normal0 * coefficient0 + normal1 * coefficient1;
	return (unitDirection - reconstructed).magnitudeSquared() <= 1.0e-3f;
}

// Per-body current and swept bounds prepared immediately after prediction.
// The values are valid for one OGC rebuild only: later outer iterations mutate
// particle positions, so the contact path invalidates this cache after use.
// Keeping the two variants separate preserves the legacy pair policy where a
// swept bound is selected iff either body enables speculative CCD.
struct AvbdSoftBodyBounds
{
	PxVec3 currentMinimum;
	PxVec3 currentMaximum;
	PxVec3 sweptMinimum;
	PxVec3 sweptMaximum;

	AvbdSoftBodyBounds()
		: currentMinimum(PX_MAX_F32), currentMaximum(-PX_MAX_F32),
		  sweptMinimum(PX_MAX_F32), sweptMaximum(-PX_MAX_F32)
	{
	}
};

// Serial broadphase result used as the stable input to later refit planning.
// Entries are appended in canonical (bodyA, bodyB) order and retain the exact
// current/swept pair mode chosen by the legacy loop.
struct AvbdSoftPairDetectionPlan
{
	PxU32 bodyA;
	PxU32 bodyB;
	bool swept;
	PxVec3 minimumA;
	PxVec3 maximumA;
	PxVec3 minimumB;
	PxVec3 maximumB;
};

// Detection-epoch bounds are intentionally separated from the immutable BVH
// topology held by AvbdSurface*BvhNode. A caller owns sizing and lifetime of
// this span; refit/query code only writes/reads its matching node index.
struct AvbdSurfaceBvhNodeBounds
{
	PxVec3 minimum;
	PxVec3 maximum;
};

// Mutable hierarchy bounds for one body in one detection epoch.  A body may
// appear in both current and swept pairs, so each mode owns an independent
// span and validity stamp.  The arrays are workspace-owned: compiled topology
// never stores detection-period state.
struct AvbdSoftPairBvhEpochSpans
{
	PxArray<AvbdSurfaceBvhNodeBounds> currentBounds;
	PxArray<AvbdSurfaceBvhNodeBounds> sweptBounds;
	PxU32 currentRequiredEpoch;
	PxU32 sweptRequiredEpoch;
	PxU32 currentRefitEpoch;
	PxU32 sweptRefitEpoch;

	AvbdSoftPairBvhEpochSpans()
		: currentRequiredEpoch(0), sweptRequiredEpoch(0),
		  currentRefitEpoch(0), sweptRefitEpoch(0)
	{
	}
};

// P5.2a's immutable redetection phase IR.  This records the exact outer
// contact-source order and the canonical source-domain range for one complete
// OGC rebuild.  It is intentionally a phase plan, not a worker plan yet:
// nested soft-pair/self query scratch and the final contact-array merge remain
// parent-owned until P5.2b has private output contracts for them.
struct AvbdSoftContactRedetectionPhase
{
	enum Type : PxU8
	{
		eWORLD_PLANES,
		eLEGACY_GROUND,
		eRIGID_BOXES,
		eRIGID_SPHERES,
		eRIGID_CAPSULES,
		eRIGID_CONVEXES,
		eRIGID_TRIANGLE_SURFACES,
		eSOFT_SOFT,
		eSELF_BODY
	};

	Type type;
	PxU32 sourceBegin;
	PxU32 sourceEnd;

	AvbdSoftContactRedetectionPhase()
		: type(eLEGACY_GROUND), sourceBegin(0), sourceEnd(0)
	{
	}

	AvbdSoftContactRedetectionPhase(
		Type inputType, PxU32 inputSourceBegin, PxU32 inputSourceEnd)
		: type(inputType), sourceBegin(inputSourceBegin),
		  sourceEnd(inputSourceEnd)
	{
	}
};

PX_FORCE_INLINE bool avbdValidateRedetectionPhasePlan()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_VALIDATE_REDETECTION_PLAN");
	return enabled;
}

// Topology-compiled binary hierarchy over boundary triangles.  Its child and
// leaf ranges are immutable; only minimum/maximum are refitted from current
// (or current+initial swept) particle positions at detection time.
struct AvbdSurfaceTriangleBvhNode
{
	PxVec3 minimum;
	PxVec3 maximum;
	PxU32 leftChild;
	PxU32 rightChild;
	PxU32 firstTriangle;
	PxU32 triangleCount;

	PX_FORCE_INLINE bool isLeaf() const
	{
		return leftChild == PX_MAX_U32;
	}
};

// The edge hierarchy follows the same immutable-topology/refittable-bounds
// contract as the triangle hierarchy.  It is intentionally a distinct tree:
// self EE needs segment bounds and must not infer edge ownership from facet
// leaves or alter its canonical endpoint-pair identity.
struct AvbdSurfaceEdgeBvhNode
{
	PxVec3 minimum;
	PxVec3 maximum;
	PxU32 leftChild;
	PxU32 rightChild;
	PxU32 firstEdge;
	PxU32 edgeCount;

	PX_FORCE_INLINE bool isLeaf() const
	{
		return leftChild == PX_MAX_U32;
	}
};

// All mutable state used by one soft-pair query.  The serial workspace owns
// one reusable instance, while a future plan-range child supplies its own
// instance after the parent has frozen the pair plan and BVH refit epoch.
struct AvbdSoftSoftPairQueryScratch
{
	PxArray<AvbdSoftPairEdgeBounds> edgeBoundsA;
	PxArray<AvbdSoftPairEdgeBounds> edgeBoundsB;
	PxArray<PxU32> triangleCandidates;

	void reserve(PxU32 edgeCountA, PxU32 edgeCountB,
		PxU32 triangleCandidateCapacity = 0)
	{
		edgeBoundsA.reserve(edgeCountA);
		edgeBoundsB.reserve(edgeCountB);
		triangleCandidates.reserve(triangleCandidateCapacity);
	}

	void reset()
	{
		edgeBoundsA.reset();
		edgeBoundsB.reset();
		triangleCandidates.reset();
	}
};

// One redetection-epoch cache entry for a rigid convex edge. The points and
// bounds are rebuilt from the authoritative convex pose before each feature
// suffix; only the O(soft edges * rigid edges) pair loop consumes them.
struct AvbdRigidConvexEdgeBounds
{
	PxVec3 point0;
	PxVec3 point1;
	PxVec3 minimum;
	PxVec3 maximum;
};

// Caller-owned scratch for contact rebuild, state transfer and OGC sweep
// refit. Keeping this separate from the contact records lets detection reuse
// capacity without making persistent solver state depend on array order.
struct AvbdSoftContactWorkspace
{
	PxArray<AvbdSoftContact> previousContacts;
	PxArray<PxU8> previousUsed;
	PxArray<PxReal> selfTetStressCoefficients;
	// Safety-bound reduction state is rebuilt for every OGC outer epoch.  Keep
	// its scalar minima alongside the reusable self-contact sweep records so
	// the bound calculation does not allocate transient arrays per epoch.
	PxArray<PxReal> selfSafetyTriangleMinimums;
	PxArray<PxReal> selfSafetyEdgeMinimums;
	PxArray<AvbdSelfCollisionTriangleBounds> selfTriangleBounds;
	PxArray<AvbdSelfCollisionVertexSweepEntry> selfSortedVertices;
	PxArray<PxU32> selfActiveTriangles;
	PxArray<PxU64> selfEmittedFeatureKeys;
	PxArray<AvbdSelfCollisionEdgeBounds> selfEdgeBounds;
	PxArray<PxU32> selfEdgeCandidates;
	// P5.9a keeps self-VF candidate ownership separate from soft-pair VF
	// queries.  Both phases remain serial today, but a future soft-pair child
	// must not inherit a mutable buffer that self collision can also consume.
	PxArray<PxU32> selfTriangleCandidates;
	// One byte per particle, reused by the parent-owned analytic sphere/capsule
	// and convex swept-feature suffixes. A surface vertex's forward SDF/sweep
	// ownership depends only on one body/shape/redetection epoch, not on each
	// adjacent edge or face.
	PxArray<PxU8> rigidConvexForwardOwnerScratch;
	// Tri-state (unknown/false/true) scratch for the parent-owned rigid
	// triangle-surface swept-feature suffix.  The predicate is invariant for
	// one body/surface/redetection epoch and is evaluated lazily in canonical
	// edge/face traversal order.
	PxArray<PxU8> rigidTriangleSurfaceForwardOwnerScratch;
	AvbdSoftSoftPairQueryScratch softPairQueryScratch;
	PxArray<AvbdSoftPairBvhEpochSpans> softPairTriangleBvhEpochSpans;
	PxU32 softPairTriangleBvhEpoch;
	PxArray<AvbdSurfaceBvhNodeBounds> selfTriangleBvhBounds;
	PxArray<AvbdSurfaceBvhNodeBounds> selfEdgeBvhBounds;
	PxArray<AvbdSoftPairDetectionPlan> softPairDetectionPlan;
	PxArray<AvbdSoftContactRedetectionPhase> redetectionPhasePlan;
	PxU32 redetectionOutputCapacityBefore;
	PxArray<AvbdSoftBodyBounds> softBodyBounds;
	PxArray<PxU8> softBodyBoundsReady;
	bool softBodyBoundsValid;
	PxU64 growthEvents;
	PxU64 growthBytes;
	PxU64 sweepScratchGrowthEvents;
	PxU64 sweepScratchGrowthBytes;
	PxU64 outputGrowthEvents;
	PxU64 outputGrowthBytes;
	PxU32 peakOutputContactCount;
	PxU32 peakOutputContactCapacity;
	PxU32 peakPreviousContactCount;
	PxU32 peakPreviousContactCapacity;
	PxU32 peakPreviousUsedCapacity;

	AvbdSoftContactWorkspace()
		: softPairTriangleBvhEpoch(0), redetectionOutputCapacityBefore(0),
		  softBodyBoundsValid(false),
		  growthEvents(0), growthBytes(0),
		  sweepScratchGrowthEvents(0), sweepScratchGrowthBytes(0),
		  outputGrowthEvents(0), outputGrowthBytes(0),
		  peakOutputContactCount(0), peakOutputContactCapacity(0),
		  peakPreviousContactCount(0), peakPreviousContactCapacity(0),
		  peakPreviousUsedCapacity(0)
	{
	}

	void reserve(PxU32 contactCapacity)
	{
		previousContacts.reserve(contactCapacity);
		previousUsed.reserve(contactCapacity);
	}

	template<typename T>
	void reserveSweepScratch(PxArray<T>& array, PxU32 capacity)
	{
		if(capacity > array.capacity())
		{
			growthEvents++;
			growthBytes +=
				PxU64(capacity - array.capacity()) * sizeof(T);
			sweepScratchGrowthEvents++;
			sweepScratchGrowthBytes +=
				PxU64(capacity - array.capacity()) * sizeof(T);
			array.reserve(capacity);
		}
	}

	void reserveSelfCollisionSweep(
		PxU32 tetCount, PxU32 triangleCount,
		PxU32 vertexCount, PxU32 edgeCount)
	{
		reserveSweepScratch(selfTetStressCoefficients, tetCount);
		reserveSweepScratch(selfSafetyTriangleMinimums, triangleCount);
		reserveSweepScratch(selfSafetyEdgeMinimums, edgeCount);
		reserveSweepScratch(selfTriangleBounds, triangleCount);
		reserveSweepScratch(selfSortedVertices, vertexCount);
		reserveSweepScratch(selfActiveTriangles, triangleCount);
		reserveSweepScratch(selfEmittedFeatureKeys, triangleCount);
		reserveSweepScratch(selfEdgeBounds, edgeCount);
		reserveSweepScratch(selfEdgeCandidates, edgeCount);
		reserveSweepScratch(selfTriangleCandidates, triangleCount);
	}

	void reserveSoftPairSweep(
		PxU32 edgeCountA, PxU32 edgeCountB,
		PxU32 triangleCandidateCapacity = 0)
	{
		reserveSweepScratch(softPairQueryScratch.edgeBoundsA, edgeCountA);
		reserveSweepScratch(softPairQueryScratch.edgeBoundsB, edgeCountB);
		reserveSweepScratch(
			softPairQueryScratch.triangleCandidates,
			triangleCandidateCapacity);
	}

	// Start a distinct soft-pair detection epoch. Callers first mark the body
	// and mode spans that a canonical pair plan needs, then refit each marked
	// span exactly once. A later pair consumer may only read a span stamped
	// with this epoch.
	void beginSoftPairTriangleBvhEpoch(PxU32 bodyCount)
	{
		reserveSweepScratch(softPairTriangleBvhEpochSpans, bodyCount);
		softPairTriangleBvhEpochSpans.resize(bodyCount);
		softPairTriangleBvhEpoch++;
		if(softPairTriangleBvhEpoch == 0)
		{
			softPairTriangleBvhEpoch = 1;
			for(PxU32 bodyIndex = 0;
				bodyIndex < softPairTriangleBvhEpochSpans.size();
				++bodyIndex)
			{
				AvbdSoftPairBvhEpochSpans& spans =
					softPairTriangleBvhEpochSpans[bodyIndex];
				spans.currentRequiredEpoch = 0;
				spans.sweptRequiredEpoch = 0;
				spans.currentRefitEpoch = 0;
				spans.sweptRefitEpoch = 0;
			}
		}
	}

	void requireSoftPairTriangleBvhBounds(
		PxU32 bodyIndex, bool swept, PxU32 nodeCount)
	{
		PX_ASSERT(bodyIndex < softPairTriangleBvhEpochSpans.size());
		PX_ASSERT(softPairTriangleBvhEpoch != 0);
		AvbdSoftPairBvhEpochSpans& spans =
			softPairTriangleBvhEpochSpans[bodyIndex];
		PxArray<AvbdSurfaceBvhNodeBounds>& bounds = swept
			? spans.sweptBounds : spans.currentBounds;
		reserveSweepScratch(bounds, nodeCount);
		bounds.resize(nodeCount);
		if(swept)
			spans.sweptRequiredEpoch = softPairTriangleBvhEpoch;
		else
			spans.currentRequiredEpoch = softPairTriangleBvhEpoch;
	}

	bool isSoftPairTriangleBvhBoundsRequired(
		PxU32 bodyIndex, bool swept) const
	{
		PX_ASSERT(bodyIndex < softPairTriangleBvhEpochSpans.size());
		const AvbdSoftPairBvhEpochSpans& spans =
			softPairTriangleBvhEpochSpans[bodyIndex];
		return swept ?
			spans.sweptRequiredEpoch == softPairTriangleBvhEpoch :
			spans.currentRequiredEpoch == softPairTriangleBvhEpoch;
	}

	PxArray<AvbdSurfaceBvhNodeBounds>& getSoftPairTriangleBvhBoundsForRefit(
		PxU32 bodyIndex, bool swept)
	{
		PX_ASSERT(isSoftPairTriangleBvhBoundsRequired(bodyIndex, swept));
		AvbdSoftPairBvhEpochSpans& spans =
			softPairTriangleBvhEpochSpans[bodyIndex];
		return swept ? spans.sweptBounds : spans.currentBounds;
	}

	void markSoftPairTriangleBvhBoundsRefit(
		PxU32 bodyIndex, bool swept)
	{
		PX_ASSERT(isSoftPairTriangleBvhBoundsRequired(bodyIndex, swept));
		AvbdSoftPairBvhEpochSpans& spans =
			softPairTriangleBvhEpochSpans[bodyIndex];
		if(swept)
			spans.sweptRefitEpoch = softPairTriangleBvhEpoch;
		else
			spans.currentRefitEpoch = softPairTriangleBvhEpoch;
	}

	const PxArray<AvbdSurfaceBvhNodeBounds>& getSoftPairTriangleBvhBounds(
		PxU32 bodyIndex, bool swept) const
	{
		PX_ASSERT(bodyIndex < softPairTriangleBvhEpochSpans.size());
		const AvbdSoftPairBvhEpochSpans& spans =
			softPairTriangleBvhEpochSpans[bodyIndex];
		PX_ASSERT(swept ?
			spans.sweptRefitEpoch == softPairTriangleBvhEpoch :
			spans.currentRefitEpoch == softPairTriangleBvhEpoch);
		return swept ? spans.sweptBounds : spans.currentBounds;
	}

	void prepareSelfBvhBounds(
		PxU32 triangleNodeCount, PxU32 edgeNodeCount)
	{
		reserveSweepScratch(selfTriangleBvhBounds, triangleNodeCount);
		reserveSweepScratch(selfEdgeBvhBounds, edgeNodeCount);
		selfTriangleBvhBounds.resize(triangleNodeCount);
		selfEdgeBvhBounds.resize(edgeNodeCount);
	}

	void beginSoftPairDetectionPlan()
	{
		softPairDetectionPlan.clear();
	}

	void appendSoftPairDetectionPlan(
		const AvbdSoftPairDetectionPlan& plan)
	{
		if(softPairDetectionPlan.size() ==
			softPairDetectionPlan.capacity())
		{
			const PxU32 currentCapacity =
				softPairDetectionPlan.capacity();
			const PxU32 nextCapacity = currentCapacity == 0
				? 8u : currentCapacity <= PX_MAX_U32 / 2
					? currentCapacity * 2u : PX_MAX_U32;
			reserveSweepScratch(softPairDetectionPlan, nextCapacity);
		}
		softPairDetectionPlan.pushBack(plan);
	}

	bool validateSoftPairDetectionPlan(PxU32 bodyCount) const
	{
		PxU32 previousBodyA = 0;
		PxU32 previousBodyB = 0;
		for(PxU32 planIndex = 0;
			planIndex < softPairDetectionPlan.size(); ++planIndex)
		{
			const AvbdSoftPairDetectionPlan& plan =
				softPairDetectionPlan[planIndex];
			if(plan.bodyA >= plan.bodyB || plan.bodyB >= bodyCount)
				return false;
			if(planIndex > 0 &&
				(plan.bodyA < previousBodyA ||
				 (plan.bodyA == previousBodyA &&
				  plan.bodyB <= previousBodyB)))
				return false;
			previousBodyA = plan.bodyA;
			previousBodyB = plan.bodyB;
		}
		return true;
	}

	// P5.2a: phase-plan publication happens before any candidate append.  Its
	// order is the serial OGC order, and the parent retains all mutable arrays.
	void beginRedetectionPhasePlan()
	{
		redetectionPhasePlan.clear();
	}

	void appendRedetectionPhasePlan(
		AvbdSoftContactRedetectionPhase::Type type,
		PxU32 sourceBegin, PxU32 sourceEnd)
	{
		PX_ASSERT(sourceBegin < sourceEnd);
		if(redetectionPhasePlan.size() ==
			redetectionPhasePlan.capacity())
		{
			const PxU32 currentCapacity =
				redetectionPhasePlan.capacity();
			const PxU32 nextCapacity = currentCapacity == 0
				? 8u : currentCapacity <= PX_MAX_U32 / 2
					? currentCapacity * 2u : PX_MAX_U32;
			reserveSweepScratch(redetectionPhasePlan, nextCapacity);
		}
		redetectionPhasePlan.pushBack(
			AvbdSoftContactRedetectionPhase(
				type, sourceBegin, sourceEnd));
	}

	bool validateRedetectionPhasePlan() const
	{
		if(redetectionPhasePlan.empty())
			return true;
		PxU32 previousType = 0;
		for(PxU32 phaseIndex = 0;
			phaseIndex < redetectionPhasePlan.size(); ++phaseIndex)
		{
			const AvbdSoftContactRedetectionPhase& phase =
				redetectionPhasePlan[phaseIndex];
			if(phase.sourceBegin >= phase.sourceEnd ||
				PxU32(phase.type) < previousType)
				return false;
			previousType = PxU32(phase.type);
		}
		return true;
	}

	// This is prepared before prediction tasks are submitted. Child tasks then
	// write distinct body slots only; no task may resize either array.
	void prepareSoftBodyBounds(PxU32 bodyCount)
	{
		reserveSweepScratch(softBodyBounds, bodyCount);
		reserveSweepScratch(softBodyBoundsReady, bodyCount);
		softBodyBounds.resize(bodyCount);
		softBodyBoundsReady.resize(bodyCount);
		for(PxU32 bodyIndex = 0; bodyIndex < bodyCount; ++bodyIndex)
			softBodyBoundsReady[bodyIndex] = 0;
		softBodyBoundsValid = false;
	}

	void markSoftBodyBoundsReady()
	{
		softBodyBoundsValid = !softBodyBounds.empty();
		for(PxU32 bodyIndex = 0;
			bodyIndex < softBodyBoundsReady.size(); ++bodyIndex)
		{
			if(!softBodyBoundsReady[bodyIndex])
			{
				softBodyBoundsValid = false;
				break;
			}
		}
	}

	void invalidateSoftBodyBounds()
	{
		softBodyBoundsValid = false;
	}

	void beginStep()
	{
		growthEvents = 0;
		growthBytes = 0;
		sweepScratchGrowthEvents = 0;
		sweepScratchGrowthBytes = 0;
		outputGrowthEvents = 0;
		outputGrowthBytes = 0;
		peakOutputContactCount = 0;
		peakOutputContactCapacity = 0;
		peakPreviousContactCount = 0;
		peakPreviousContactCapacity = 0;
		peakPreviousUsedCapacity = 0;
	}

	void recordOutputCapacityGrowth(
		PxU32 capacityBefore, PxU32 capacityAfter)
	{
		if(capacityAfter > capacityBefore)
		{
			outputGrowthEvents++;
			outputGrowthBytes +=
				PxU64(capacityAfter - capacityBefore) *
				sizeof(AvbdSoftContact);
		}
	}

	void recordOutputWatermark(PxU32 count, PxU32 capacity)
	{
		peakOutputContactCount = PxMax(peakOutputContactCount, count);
		peakOutputContactCapacity = PxMax(peakOutputContactCapacity, capacity);
	}

	void copyPreviousContacts(const PxArray<AvbdSoftContact>& contacts)
	{
		if(contacts.size() > previousContacts.capacity())
		{
			growthEvents++;
			growthBytes +=
				PxU64(contacts.size() - previousContacts.capacity()) *
				sizeof(AvbdSoftContact);
		}
		previousContacts.assign(contacts.begin(), contacts.end());
		peakPreviousContactCount = PxMax(
			peakPreviousContactCount, contacts.size());
		peakPreviousContactCapacity = PxMax(
			peakPreviousContactCapacity, previousContacts.capacity());
	}

	void resizePreviousUsed(PxU32 size)
	{
		if(size > previousUsed.capacity())
		{
			growthEvents++;
			growthBytes +=
				PxU64(size - previousUsed.capacity()) * sizeof(PxU8);
		}
		previousUsed.resize(size);
		peakPreviousUsedCapacity = PxMax(
			peakPreviousUsedCapacity, previousUsed.capacity());
	}

	void reset()
	{
		previousContacts.reset();
		previousUsed.reset();
		selfTetStressCoefficients.reset();
		selfSafetyTriangleMinimums.reset();
		selfSafetyEdgeMinimums.reset();
		selfTriangleBounds.reset();
		selfSortedVertices.reset();
		selfActiveTriangles.reset();
		selfEmittedFeatureKeys.reset();
		selfEdgeBounds.reset();
		selfEdgeCandidates.reset();
		selfTriangleCandidates.reset();
		rigidConvexForwardOwnerScratch.reset();
		rigidTriangleSurfaceForwardOwnerScratch.reset();
		softPairQueryScratch.reset();
		softPairTriangleBvhEpochSpans.reset();
		softPairTriangleBvhEpoch = 0;
		selfTriangleBvhBounds.reset();
		selfEdgeBvhBounds.reset();
		softPairDetectionPlan.reset();
		redetectionPhasePlan.reset();
		softBodyBounds.reset();
		softBodyBoundsReady.reset();
		softBodyBoundsValid = false;
		beginStep();
	}
};

PX_FORCE_INLINE void avbdInitializeSoftContactAnchors(
	const AvbdSoftContactGeometry& geometry,
	AvbdSoftContactAugmentedState& state,
	const AvbdSoftParticle* particles)
{
	state.particlePointPrev =
		avbdGetSoftContactQueryPoint(geometry, particles);
	state.surfacePointPrev =
		avbdGetSoftContactSurfacePoint(geometry, particles);
}

// A velocity-owned tangent must never inherit a Position-AL spring or its
// frame anchor.  The normal AL state intentionally stays intact: this helper
// changes only the tangential owner.
PX_FORCE_INLINE void avbdResetSoftContactTangentState(
	const AvbdSoftContactGeometry& geometry,
	AvbdSoftContactAugmentedState& state,
	const AvbdSoftParticle* particles)
{
	state.alLambdaTangent[0] = 0.0f;
	state.alLambdaTangent[1] = 0.0f;
	state.penTangent[0] = 1000.0f;
	state.penTangent[1] = 1000.0f;
	state.frictionStick = false;
	avbdInitializeSoftContactAnchors(geometry, state, particles);
}

PX_FORCE_INLINE void avbdBuildSoftContactTangents(
	AvbdSoftContactGeometry& geometry)
{
	if(PxAbs(geometry.normal.x) < 0.9f)
		geometry.tangent1 =
			geometry.normal.cross(PxVec3(1.0f, 0.0f, 0.0f)).getNormalized();
	else
		geometry.tangent1 =
			geometry.normal.cross(PxVec3(0.0f, 1.0f, 0.0f)).getNormalized();
	geometry.tangent2 = geometry.normal.cross(geometry.tangent1);
}

// The only production boundary from detection into the solver.  Geometry is
// the prepared-contact IR; this function creates and initializes its unique
// augmented state before publishing the aggregate to the solver.
PX_FORCE_INLINE void avbdAppendPreparedSoftContact(
	const AvbdSoftContactGeometry& geometry,
	PxReal k, PxReal ke,
	const AvbdSoftParticle* particles,
	PxArray<AvbdSoftContact>& contacts)
{
	PX_ASSERT(geometry.source.isValid());
	PX_ASSERT(
		geometry.targetKind !=
			AvbdSoftContactTargetKind::eUNSUPPORTED);
	PX_ASSERT(geometry.normal.isFinite());
	PX_ASSERT(geometry.tangent1.isFinite());
	PX_ASSERT(geometry.tangent2.isFinite());

	AvbdSoftContact contact;
	contact.geometry = geometry;
	AvbdSoftContactAugmentedState& state = contact.state;
	state.k = k;
	state.ke = ke;
	avbdInitializeSoftContactAnchors(geometry, state, particles);
	contacts.pushBack(contact);
}

// =============================================================================
// Per-particle element adjacency
// =============================================================================

struct AvbdParticleElementRef
{
	PxU32 index;
	PxU8 vOrder;
	PxU8 padding[3];
};

struct AvbdParticleElementAdjacency
{
	PxArray<AvbdParticleElementRef> triRefs;
	PxArray<AvbdParticleElementRef> tetRefs;
	PxArray<AvbdParticleElementRef> bendRefs;
};

// P8.2's packet record has a deliberately canonical layout: lane N refers to
// tetRefs[packetOrdinal * 8 + N] for the owning particle. It may accelerate
// evaluation later, but it is not permitted to define a new reduction order.
static const PxU32 eAVBD_TET_INCIDENCE_PACKET_WIDTH = 8;

struct AvbdTetIncidencePacket8
{
	PxU32 tetIndices[eAVBD_TET_INCIDENCE_PACKET_WIDTH];
	PxU8 vertexOrders[eAVBD_TET_INCIDENCE_PACKET_WIDTH];
	PxU8 validMask;
	PxU8 padding[3];

	AvbdTetIncidencePacket8()
		: validMask(0), padding{0, 0, 0}
	{
		for(PxU32 lane = 0; lane < eAVBD_TET_INCIDENCE_PACKET_WIDTH;
			lane++)
		{
			tetIndices[lane] = PX_MAX_U32;
			vertexOrders[lane] = PX_MAX_U8;
		}
	}
};

struct AvbdTetIncidencePacketRange
{
	PxU32 packetStart;
	PxU32 packetCount;

	AvbdTetIncidencePacketRange() : packetStart(0), packetCount(0) {}
};

enum class AvbdSoftObjectiveOwner : PxU8
{
	eKINEMATIC_PIN_POSITION_AL,
	eDEFORMABLE_KINEMATIC_POSITION_AL,
	eKINEMATIC_ATTACHMENT_POSITION_AL,
	eRIGID_ATTACHMENT_POSITION_AL,
	eARTICULATION_ATTACHMENT_POSITION_AL,
	eSOFT_PAIR_ATTACHMENT_POSITION_AL,
	eUNSUPPORTED
};

PX_FORCE_INLINE AvbdSoftObjectiveOwner avbdGetAttachmentObjectiveOwner(
	const AvbdSoftAttachment& attachment, bool particleIsValid)
{
	if(!particleIsValid)
		return AvbdSoftObjectiveOwner::eUNSUPPORTED;
	switch(attachment.targetKind)
	{
	case AvbdSoftAttachmentTargetKind::eDYNAMIC_RIGID:
		return AvbdSoftObjectiveOwner::
			eRIGID_ATTACHMENT_POSITION_AL;
	case AvbdSoftAttachmentTargetKind::eARTICULATION_LINK:
		return AvbdSoftObjectiveOwner::
			eARTICULATION_ATTACHMENT_POSITION_AL;
	case AvbdSoftAttachmentTargetKind::eDYNAMIC_SOFT:
		if(attachment.targetPoint.particleCount == 0 ||
			attachment.targetPoint.particleCount > 4)
			return AvbdSoftObjectiveOwner::eUNSUPPORTED;
		for(PxU32 endpoint = 0;
			endpoint < attachment.targetPoint.particleCount; endpoint++)
		{
			if(attachment.targetPoint.particleIndices[endpoint] ==
					PX_MAX_U32 ||
				!PxIsFinite(
					attachment.targetPoint.weights[endpoint]))
				return AvbdSoftObjectiveOwner::eUNSUPPORTED;
		}
		return AvbdSoftObjectiveOwner::
			eSOFT_PAIR_ATTACHMENT_POSITION_AL;
	case AvbdSoftAttachmentTargetKind::eUNSUPPORTED:
	default:
		return AvbdSoftObjectiveOwner::eUNSUPPORTED;
	}
}

PX_FORCE_INLINE bool avbdIsAttachmentPositionOwner(
	AvbdSoftObjectiveOwner owner)
{
	return owner ==
			AvbdSoftObjectiveOwner::eRIGID_ATTACHMENT_POSITION_AL ||
		owner == AvbdSoftObjectiveOwner::
			eARTICULATION_ATTACHMENT_POSITION_AL ||
		owner == AvbdSoftObjectiveOwner::
			eSOFT_PAIR_ATTACHMENT_POSITION_AL;
}

PX_FORCE_INLINE AvbdSoftObjectiveOwner avbdGetPinObjectiveOwner(
	const AvbdKinematicPin& pin, bool particleIsValid)
{
	if(!particleIsValid)
		return AvbdSoftObjectiveOwner::eUNSUPPORTED;
	switch(pin.targetKind)
	{
	case AvbdSoftPinTargetKind::eWORLD_FIXED:
		return AvbdSoftObjectiveOwner::eKINEMATIC_PIN_POSITION_AL;
	case AvbdSoftPinTargetKind::eDEFORMABLE_KINEMATIC:
		return AvbdSoftObjectiveOwner::
			eDEFORMABLE_KINEMATIC_POSITION_AL;
	case AvbdSoftPinTargetKind::ePRESCRIBED_RIGID:
		return AvbdSoftObjectiveOwner::
			eKINEMATIC_ATTACHMENT_POSITION_AL;
	case AvbdSoftPinTargetKind::eUNSUPPORTED:
	default:
		return AvbdSoftObjectiveOwner::eUNSUPPORTED;
	}
}

PX_FORCE_INLINE bool avbdIsPinPositionOwner(
	AvbdSoftObjectiveOwner owner)
{
	return owner ==
			AvbdSoftObjectiveOwner::eKINEMATIC_PIN_POSITION_AL ||
		owner ==
			AvbdSoftObjectiveOwner::
				eDEFORMABLE_KINEMATIC_POSITION_AL ||
		owner ==
			AvbdSoftObjectiveOwner::
				eKINEMATIC_ATTACHMENT_POSITION_AL;
}

struct AvbdCompiledSoftObjective
{
	AvbdSoftObjectiveOwner owner;
	PxU32 runtimeStateIndex;
	AvbdSoftPoint point;
	AvbdSoftPoint targetPoint;
	PxU32 rigidBodyIdx;

	AvbdCompiledSoftObjective()
		: owner(AvbdSoftObjectiveOwner::eUNSUPPORTED),
		  runtimeStateIndex(PX_MAX_U32), rigidBodyIdx(PX_MAX_U32)
	{
	}
};

struct AvbdParticleObjectiveAdjacency
{
	PxArray<PxU32> objectiveIndices;
};

// =============================================================================
// AvbdSoftBody -- explicit compiled/material/runtime ownership
// =============================================================================

struct AvbdSoftBodyMaterialData
{
	PxReal youngsModulus;
	PxReal poissonsRatio;
	PxReal density;
	PxReal damping;
	PxReal bendingStiffness;
	PxReal bendingDamping;
	PxReal thickness;
	PxReal dynamicFriction;
	bool coRotationalVolumeModel;

	PxReal mu;
	PxReal lambda;
	PxReal neoHookeanAlpha;

	AvbdSoftBodyMaterialData()
		: youngsModulus(1e5f), poissonsRatio(0.3f),
		  density(1000.0f), damping(0.0f),
		  bendingStiffness(0.0f), bendingDamping(0.0f),
		  thickness(0.01f), dynamicFriction(0.5f),
		  coRotationalVolumeModel(true),
		  mu(0.0f), lambda(0.0f), neoHookeanAlpha(1.0f)
	{
	}

	void computeLameParameters()
	{
		mu = youngsModulus / (2.0f * (1.0f + poissonsRatio));
		lambda = youngsModulus * poissonsRatio /
		         ((1.0f + poissonsRatio) * (1.0f - 2.0f * poissonsRatio));
		const PxReal lambdaSafe =
			PxAbs(lambda) < 1e-6f ? 1e-6f : lambda;
		neoHookeanAlpha = 1.0f + mu / lambdaSafe;
	}
};

struct AvbdSoftBodyRuntimeState
{
	PxArray<AvbdSoftAttachment> attachments;
	PxArray<AvbdKinematicPin> pins;
	PxArray<AvbdCompiledSoftObjective> compiledObjectives;
	PxArray<AvbdParticleObjectiveAdjacency> objectiveAdjacency;

	void compileObjectiveProgram(PxU32 particleStart, PxU32 particleCount)
	{
		compiledObjectives.clear();
		objectiveAdjacency.resize(particleCount);
		for (PxU32 i = 0; i < particleCount; i++)
			objectiveAdjacency[i].objectiveIndices.clear();

		for (PxU32 ai = 0; ai < attachments.size(); ai++)
		{
			const AvbdSoftAttachment& attachment = attachments[ai];
			const bool pointIsValid = avbdIsSoftPointValid(
				attachment.point, particleStart, particleCount);
			AvbdCompiledSoftObjective objective;
			objective.owner = avbdGetAttachmentObjectiveOwner(
				attachment, pointIsValid);
			objective.runtimeStateIndex = ai;
			objective.point = attachment.point;
			objective.targetPoint = attachment.targetPoint;
			objective.rigidBodyIdx = attachment.rigidBodyIdx;
			const PxU32 objectiveIndex = compiledObjectives.size();
			compiledObjectives.pushBack(objective);
			if(pointIsValid)
			{
				for(PxU32 endpoint = 0;
					endpoint < attachment.point.particleCount; endpoint++)
				{
					const PxU32 particleIndex =
						attachment.point.particleIndices[endpoint];
					bool firstOccurrence = true;
					for(PxU32 previous = 0; previous < endpoint; previous++)
					{
						if(attachment.point.particleIndices[previous] ==
							particleIndex)
						{
							firstOccurrence = false;
							break;
						}
					}
					if(firstOccurrence)
						objectiveAdjacency[
							particleIndex - particleStart].
							objectiveIndices.pushBack(objectiveIndex);
				}
			}
		}

		for (PxU32 pi = 0; pi < pins.size(); pi++)
		{
			const AvbdKinematicPin& pin = pins[pi];
			const bool pointIsValid = avbdIsSoftPointValid(
				pin.point, particleStart, particleCount);
			AvbdCompiledSoftObjective objective;
			objective.owner = avbdGetPinObjectiveOwner(
				pin, pointIsValid);
			objective.runtimeStateIndex = pi;
			objective.point = pin.point;
			objective.rigidBodyIdx = PX_MAX_U32;
			const PxU32 objectiveIndex = compiledObjectives.size();
			compiledObjectives.pushBack(objective);
			if(pointIsValid)
			{
				for(PxU32 endpoint = 0;
					endpoint < pin.point.particleCount; endpoint++)
				{
					const PxU32 particleIndex =
						pin.point.particleIndices[endpoint];
					bool firstOccurrence = true;
					for(PxU32 previous = 0; previous < endpoint; previous++)
					{
						if(pin.point.particleIndices[previous] ==
							particleIndex)
						{
							firstOccurrence = false;
							break;
						}
					}
					if(firstOccurrence)
						objectiveAdjacency[
							particleIndex - particleStart].
							objectiveIndices.pushBack(objectiveIndex);
				}
			}
		}
	}

	bool isObjectiveProgramCurrent(
		PxU32 particleStart, PxU32 particleCount) const
	{
		if (objectiveAdjacency.size() != particleCount ||
			compiledObjectives.size() !=
				attachments.size() + pins.size())
			return false;

		for (PxU32 ai = 0; ai < attachments.size(); ai++)
		{
			const AvbdSoftAttachment& attachment = attachments[ai];
			const AvbdCompiledSoftObjective& objective =
				compiledObjectives[ai];
			const bool pointIsValid = avbdIsSoftPointValid(
				attachment.point, particleStart, particleCount);
			const AvbdSoftObjectiveOwner expectedOwner =
				avbdGetAttachmentObjectiveOwner(
					attachment, pointIsValid);
			if (objective.owner != expectedOwner ||
				objective.runtimeStateIndex != ai ||
				!(objective.point == attachment.point) ||
				!(objective.targetPoint ==
					attachment.targetPoint) ||
				objective.rigidBodyIdx != attachment.rigidBodyIdx)
				return false;
		}

		for (PxU32 pi = 0; pi < pins.size(); pi++)
		{
			const AvbdKinematicPin& pin = pins[pi];
			const AvbdCompiledSoftObjective& objective =
				compiledObjectives[attachments.size() + pi];
			const bool pointIsValid = avbdIsSoftPointValid(
				pin.point, particleStart, particleCount);
			const AvbdSoftObjectiveOwner expectedOwner =
				avbdGetPinObjectiveOwner(
					pin, pointIsValid);
			if (objective.owner != expectedOwner ||
				objective.runtimeStateIndex != pi ||
				!(objective.point == pin.point) ||
				objective.rigidBodyIdx != PX_MAX_U32)
				return false;
		}
		return true;
	}
};

// Defined after the OGC closest-point implementation. Keeping this scalar
// declaration here lets immutable compiled metadata use the exact same
// point/triangle distance rule without depending on the later result type.
PX_FORCE_INLINE PxReal avbdGetRestPointTriangleDistance(
	const PxVec3& point, const PxVec3& vertex0,
	const PxVec3& vertex1, const PxVec3& vertex2);

struct AvbdSoftBodyCompiledData
{
	PxU32 particleStart;
	PxU32 particleCount;

	PxArray<PxU32> tetrahedra;
	PxArray<PxU32> triangles;

	PxArray<AvbdTriElement> triElements;
	PxArray<AvbdTetElement> tetElements;
	PxArray<AvbdBendingElement> bendElements;
	PxArray<AvbdEdgeInfo> edges;
	PxArray<AvbdParticleElementAdjacency> elementAdjacency;
	// P8.2 keeps a canonical, topology-versioned projection of tetRefs. A
	// particle's packet lanes are consecutive original adjacency ordinals, so
	// future packet arithmetic can retain scalar-order reduction and stable
	// positive-J linearization ownership.
	PxArray<AvbdTetIncidencePacket8> tetIncidencePackets;
	PxArray<AvbdTetIncidencePacketRange> tetIncidencePacketRanges;
	PxU32 tetIncidenceFullPacketCount;
	bool tetIncidencePacketProgramValid;
	// P4.1: immutable, body-local structural conflict CSR for particle-primal
	// blocks.  It contains the union of every tri/tet/bending hyperedge.  It is
	// intentionally not a complete color plan: contacts and point objectives
	// add dynamic supports after redetection and must be overlaid before a
	// colored schedule is allowed to consume it.
	PxArray<PxU32> particlePrimalStructuralConflictOffsets;
	PxArray<PxU32> particlePrimalStructuralConflictIndices;
	bool particlePrimalStructuralConflictValid;
	bool flatteningEnabled;

	// Immutable local rest-space input used to compile self-collision
	// candidates. The public filter distance is copied beside it during
	// prep so contact detection does not reinterpret actor flags or use the
	// global current-space contact radius as a rest-neighborhood policy.
	PxArray<PxVec3> selfCollisionRestPositions;
	PxReal selfCollisionFilterDistance;
	// Immutable rest-space exclusion cache for the self vertex/triangle
	// filter. Each local query vertex stores boundary-triangle indices whose
	// rest-space closest distance is within the actor filter distance. This is
	// deliberately separate from current-position bounds: it survives refits
	// but must be rebuilt when rest topology/positions or filter distance
	// changes. A bounded fallback preserves the direct query for unusually
	// dense rest-space neighborhoods.
	PxArray<PxArray<PxU32> > selfCollisionRestFilteredTriangles;
	PxReal selfCollisionRestFilterCacheDistance;
	bool selfCollisionRestFilterCacheValid;
	bool selfCollisionRestFilterCacheFallback;
	// Public per-actor collision-bias speed. This is consumed only by the
	// prepared contact owner; generic motion finalization must not clamp it.
	PxReal maxDepenetrationVelocity;
	// Volume self-collision contacts whose owning tetrahedron exceeds this
	// dimensionless co-rotated strain threshold are omitted, matching the
	// public GPU deformable-volume policy.
	PxReal selfCollisionStressTolerance;
	// Public deformable speculative-CCD policy. Swept candidates are
	// generated only for bodies that explicitly opt in.
	bool speculativeCCDEnabled;

	PxArray<PxU32> surfaceTriangles;  // boundary face indices (3 per tri, global particle indices)
	PxArray<PxU32> surfaceVertices;   // unique sorted boundary vertices, global particle indices
	PxArray<AvbdEdgeInfo> surfaceEdges; // unique boundary edges, global indices
	// Source element owning each boundary face. Surface entries are public
	// triangle indices; Volume entries are simulation tetrahedron indices.
	PxArray<PxU32> surfaceTriangleElementIndices;
	// For volume boundary faces, resolve the source tetrahedron to its compiled
	// tet-element slot once at topology build time. A non-existent slot is kept
	// as PX_MAX_U32 so the self-collision stress filter preserves its existing
	// permissive fallback for an invalid/omitted compiled source element.
	PxArray<PxU32> surfaceTriangleTetElementIndices;
	// The hierarchy's topology/permutation is compiled once. Bounds are mutable
	// detection scratch stored with the owning immutable topology, so a refit
	// cannot change feature identity or leaf membership.
	PxArray<PxU32> surfaceTriangleBvhTriangleIndices;
	PxArray<AvbdSurfaceTriangleBvhNode> surfaceTriangleBvhNodes;
	PxArray<PxU32> surfaceEdgeBvhEdgeIndices;
	PxArray<AvbdSurfaceEdgeBvhNode> surfaceEdgeBvhNodes;

	AvbdSoftBodyCompiledData()
		: particleStart(0), particleCount(0),
		  tetIncidenceFullPacketCount(0),
		  tetIncidencePacketProgramValid(false),
		  particlePrimalStructuralConflictValid(false),
		  flatteningEnabled(false),
		  selfCollisionFilterDistance(0.0f),
		  selfCollisionRestFilterCacheDistance(-PX_MAX_F32),
		  selfCollisionRestFilterCacheValid(false),
		  selfCollisionRestFilterCacheFallback(false),
		  maxDepenetrationVelocity(PX_MAX_F32),
		  selfCollisionStressTolerance(0.9f),
		  speculativeCCDEnabled(false)
	{
	}

	void compileBendingRestAngles(bool enabled)
	{
		if(flatteningEnabled == enabled)
			return;
		flatteningEnabled = enabled;
		for(PxU32 i = 0; i < bendElements.size(); i++)
			bendElements[i].restAngle =
				enabled ? 0.0f : bendElements[i].restShapeAngle;
	}

	static PxReal computeDihedralAngle(const PxVec3& x0, const PxVec3& x1,
	                                   const PxVec3& x2, const PxVec3& x3)
	{
		const PxReal eps = 1e-8f;
		PxVec3 e = x3 - x2;
		PxVec3 n1 = (x2 - x0).cross(x3 - x0);
		PxVec3 n2 = (x3 - x1).cross(x2 - x1);
		PxReal n1Norm = n1.magnitude();
		PxReal n2Norm = n2.magnitude();
		PxReal eNorm = e.magnitude();
		if (n1Norm < eps || n2Norm < eps || eNorm < eps)
			return 0.0f;
		PxVec3 n1Hat = n1 * (1.0f / n1Norm);
		PxVec3 n2Hat = n2 * (1.0f / n2Norm);
		PxVec3 eHat = e * (1.0f / eNorm);
		PxReal sinTheta = n1Hat.cross(n2Hat).dot(eHat);
		PxReal cosTheta = PxClamp(n1Hat.dot(n2Hat), -1.0f, 1.0f);
		return PxAtan2(sinTheta, cosTheta);
	}

	void buildTriElements(const PxArray<AvbdSoftParticle>& particles)
	{
		triElements.clear();
		for (PxU32 i = 0; i + 2 < triangles.size(); i += 3)
		{
			PxU32 i0 = triangles[i] + particleStart;
			PxU32 i1 = triangles[i + 1] + particleStart;
			PxU32 i2 = triangles[i + 2] + particleStart;

			PxVec3 x0 = particles[i0].position;
			PxVec3 e01 = particles[i1].position - x0;
			PxVec3 e02 = particles[i2].position - x0;

			PxVec3 t1 = e01.getNormalized();
			PxVec3 n = e01.cross(e02);
			PxReal area = n.magnitude() * 0.5f;
			if (area < 1e-12f) continue;
			PxVec3 t2 = n.cross(t1).getNormalized();

			PxReal d00 = e01.dot(t1), d10 = e01.dot(t2);
			PxReal d01 = e02.dot(t1), d11 = e02.dot(t2);
			PxReal det = d00 * d11 - d01 * d10;
			if (PxAbs(det) < 1e-12f) continue;
			PxReal invDet = 1.0f / det;

			AvbdTriElement tri;
			tri.p0 = i0; tri.p1 = i1; tri.p2 = i2;
			tri.sourceElementIndex = i / 3;
			tri.DmInv00 =  d11 * invDet;
			tri.DmInv01 = -d01 * invDet;
			tri.DmInv10 = -d10 * invDet;
			tri.DmInv11 =  d00 * invDet;
			tri.restArea = area;
			triElements.pushBack(tri);
		}
	}

	void buildTetElements(const PxArray<AvbdSoftParticle>& particles)
	{
		tetElements.clear();
		for (PxU32 i = 0; i + 3 < tetrahedra.size(); i += 4)
		{
			PxU32 i0 = tetrahedra[i] + particleStart;
			PxU32 i1 = tetrahedra[i + 1] + particleStart;
			PxU32 i2 = tetrahedra[i + 2] + particleStart;
			PxU32 i3 = tetrahedra[i + 3] + particleStart;

			PxVec3 x0 = particles[i0].position;
			PxVec3 e1 = particles[i1].position - x0;
			PxVec3 e2 = particles[i2].position - x0;
			PxVec3 e3 = particles[i3].position - x0;

			PxMat33 Dm(e1, e2, e3);
			PxReal det = Dm.getDeterminant();
			PxReal vol = PxAbs(det) / 6.0f;
			if (vol < 1e-15f) continue;

			AvbdTetElement tet;
			tet.p0 = i0; tet.p1 = i1; tet.p2 = i2; tet.p3 = i3;
			tet.sourceElementIndex = i / 4;
			tet.DmInv = Dm.getInverse();
			tet.restVolume = vol;
			const PxMat33& inverseRestShape = tet.DmInv;
			const PxVec3 shapeGradients[4] =
			{
				PxVec3(
					-avbdColSum(inverseRestShape.column0),
					-avbdColSum(inverseRestShape.column1),
					-avbdColSum(inverseRestShape.column2)),
				avbdMatRow(inverseRestShape, 0),
				avbdMatRow(inverseRestShape, 1),
				avbdMatRow(inverseRestShape, 2)
			};
			for(PxU32 vertexOrder = 0; vertexOrder < 4; vertexOrder++)
			{
				tet.shapeGradients[vertexOrder] =
					shapeGradients[vertexOrder];
				tet.deformationGradientWeights[vertexOrder] =
					inverseRestShape * shapeGradients[vertexOrder];
				tet.shapeGradientNormSq[vertexOrder] =
					shapeGradients[vertexOrder].magnitudeSquared();
			}
			tet.inverseRestDeterminant =
				inverseRestShape.getDeterminant();
			tetElements.pushBack(tet);
		}
	}

	void buildBendingElements(const PxArray<AvbdSoftParticle>& particles)
	{
		bendElements.clear();
		if (triangles.empty()) return;

		// Sort-and-scan approach: collect all half-edge records, sort by
		// canonical edge key, then scan for adjacent pairs sharing the same key.
		// This is O(e log e) with contiguous memory access -- friendly to SIMD
		// sort and parallel scan in future optimisations.

		struct HalfEdge
		{
			PxU64 key;       // canonical (lo << 32 | hi)
			PxU32 edgeV0;    // original (unsorted) first vertex
			PxU32 edgeV1;    // original (unsorted) second vertex
			PxU32 oppVertex;
		};

		const PxU32 numTris = triangles.size() / 3;
		PxArray<HalfEdge> halfEdges;
		halfEdges.reserve(numTris * 3);

		for (PxU32 ti = 0; ti < numTris; ti++)
		{
			PxU32 v[3] = {
				triangles[ti * 3]     + particleStart,
				triangles[ti * 3 + 1] + particleStart,
				triangles[ti * 3 + 2] + particleStart
			};
			for (int e = 0; e < 3; e++)
			{
				PxU32 ea = v[e], eb = v[(e + 1) % 3], opp = v[(e + 2) % 3];
				HalfEdge he;
				he.key = (PxU64(ea < eb ? ea : eb) << 32) | PxU64(ea < eb ? eb : ea);
				he.edgeV0 = ea;
				he.edgeV1 = eb;
				he.oppVertex = opp;
				halfEdges.pushBack(he);
			}
		}

		// Sort by key (PxSort for now, replaceable with parallel radix sort later)
		PxSort(halfEdges.begin(), halfEdges.size(),
			[](const HalfEdge& a, const HalfEdge& b) { return a.key < b.key; });

		// Scan: match consecutive pairs with same key (manifold edge = exactly 2)
		for (PxU32 i = 0; i + 1 < halfEdges.size(); i++)
		{
			if (halfEdges[i].key != halfEdges[i + 1].key)
				continue;

			// Count how many share this key (skip non-manifold > 2)
			PxU32 run = 1;
			while (i + run < halfEdges.size() && halfEdges[i + run].key == halfEdges[i].key)
				run++;
			if (run == 2)
			{
				PxU32 edgeA = PxU32(halfEdges[i].key >> 32);
				PxU32 edgeB = PxU32(halfEdges[i].key & 0xFFFFFFFF);

				AvbdBendingElement be;
				be.opp0 = halfEdges[i].oppVertex;
				be.opp1 = halfEdges[i + 1].oppVertex;
				be.edgeStart = edgeA;
				be.edgeEnd = edgeB;
				be.restShapeAngle = computeDihedralAngle(
					particles[be.opp0].position, particles[be.opp1].position,
					particles[edgeA].position, particles[edgeB].position);
				be.restAngle = be.restShapeAngle;
				be.restLength = (particles[edgeB].position - particles[edgeA].position).magnitude();
				bendElements.pushBack(be);
			}
			i += run - 1; // advance past run
		}
	}

	void buildEdges(const PxArray<AvbdSoftParticle>& particles)
	{
		edges.clear();

		// Collect all candidate edges as canonical (lo, hi) keys into a flat
		// array, sort, then deduplicate with a linear scan.

		PxArray<PxU64> keys;
		const PxU32 numTris = triangles.size() / 3;
		const PxU32 numTets = tetrahedra.size() / 4;
		keys.reserve(numTris * 3 + numTets * 6);

		auto pushEdge = [&](PxU32 a, PxU32 b) {
			PxU32 lo = a < b ? a : b;
			PxU32 hi = a < b ? b : a;
			keys.pushBack((PxU64(lo) << 32) | PxU64(hi));
		};

		for (PxU32 i = 0; i < numTris; i++)
		{
			PxU32 v[3] = {
				triangles[i * 3]     + particleStart,
				triangles[i * 3 + 1] + particleStart,
				triangles[i * 3 + 2] + particleStart
			};
			pushEdge(v[0], v[1]); pushEdge(v[1], v[2]); pushEdge(v[2], v[0]);
		}
		for (PxU32 i = 0; i < numTets; i++)
		{
			PxU32 v[4] = {
				tetrahedra[i * 4]     + particleStart,
				tetrahedra[i * 4 + 1] + particleStart,
				tetrahedra[i * 4 + 2] + particleStart,
				tetrahedra[i * 4 + 3] + particleStart
			};
			pushEdge(v[0], v[1]); pushEdge(v[0], v[2]); pushEdge(v[0], v[3]);
			pushEdge(v[1], v[2]); pushEdge(v[1], v[3]); pushEdge(v[2], v[3]);
		}

		PxSort(keys.begin(), keys.size());

		// Linear unique scan
		for (PxU32 i = 0; i < keys.size(); )
		{
			PxU64 k = keys[i];
			PxU32 a = PxU32(k >> 32);
			PxU32 b = PxU32(k & 0xFFFFFFFF);
			AvbdEdgeInfo ei;
			ei.p0 = a; ei.p1 = b;
			ei.restLength = (particles[a].position - particles[b].position).magnitude();
			edges.pushBack(ei);
			// Skip duplicates
			while (i < keys.size() && keys[i] == k) i++;
		}
	}

	void invalidateTetIncidencePacketProgram()
	{
		tetIncidencePackets.clear();
		tetIncidencePacketRanges.clear();
		tetIncidenceFullPacketCount = 0;
		tetIncidencePacketProgramValid = false;
	}

	bool validateTetIncidencePacketProgram() const
	{
		if(tetIncidencePacketRanges.size() != particleCount)
			return false;
		PxU32 expectedPacketStart = 0;
		for(PxU32 localParticleIndex = 0;
			localParticleIndex < particleCount; localParticleIndex++)
		{
			const PxArray<AvbdParticleElementRef>& refs =
				elementAdjacency[localParticleIndex].tetRefs;
			const AvbdTetIncidencePacketRange& range =
				tetIncidencePacketRanges[localParticleIndex];
			const PxU32 expectedPacketCount =
				(refs.size() + eAVBD_TET_INCIDENCE_PACKET_WIDTH - 1) /
				eAVBD_TET_INCIDENCE_PACKET_WIDTH;
			if(range.packetStart != expectedPacketStart ||
				range.packetCount != expectedPacketCount ||
				range.packetStart + range.packetCount >
					tetIncidencePackets.size())
				return false;
			for(PxU32 ordinal = 0; ordinal < refs.size(); ordinal++)
			{
				const PxU32 packetIndex = range.packetStart +
					ordinal / eAVBD_TET_INCIDENCE_PACKET_WIDTH;
				const PxU32 lane = ordinal %
					eAVBD_TET_INCIDENCE_PACKET_WIDTH;
				const AvbdTetIncidencePacket8& packet =
					tetIncidencePackets[packetIndex];
				if((packet.validMask & PxU8(1u << lane)) == 0u ||
					packet.tetIndices[lane] != refs[ordinal].index ||
					packet.vertexOrders[lane] != refs[ordinal].vOrder)
					return false;
			}
			if(expectedPacketCount)
			{
				const PxU32 tailLanes = refs.size() %
					eAVBD_TET_INCIDENCE_PACKET_WIDTH;
				if(tailLanes)
				{
					const PxU8 expectedMask = PxU8(
						(1u << tailLanes) - 1u);
					if(tetIncidencePackets[
						range.packetStart + range.packetCount - 1].validMask !=
						expectedMask)
						return false;
				}
			}
			expectedPacketStart += expectedPacketCount;
		}
		return expectedPacketStart == tetIncidencePackets.size();
	}

	void buildTetIncidencePacketProgram()
	{
		invalidateTetIncidencePacketProgram();
		if(!avbdUseCorotationalTetPacketIr() || tetElements.empty())
			return;
		tetIncidencePacketRanges.resize(particleCount);
		for(PxU32 localParticleIndex = 0;
			localParticleIndex < particleCount; localParticleIndex++)
		{
			const PxArray<AvbdParticleElementRef>& refs =
				elementAdjacency[localParticleIndex].tetRefs;
			tetIncidenceFullPacketCount +=
				refs.size() / eAVBD_TET_INCIDENCE_PACKET_WIDTH;
			AvbdTetIncidencePacketRange& range =
				tetIncidencePacketRanges[localParticleIndex];
			range.packetStart = tetIncidencePackets.size();
			range.packetCount = (refs.size() +
				eAVBD_TET_INCIDENCE_PACKET_WIDTH - 1) /
				eAVBD_TET_INCIDENCE_PACKET_WIDTH;
			for(PxU32 packetOrdinal = 0;
				packetOrdinal < range.packetCount; packetOrdinal++)
			{
				AvbdTetIncidencePacket8 packet;
				for(PxU32 lane = 0;
					lane < eAVBD_TET_INCIDENCE_PACKET_WIDTH; lane++)
				{
					const PxU32 ordinal = packetOrdinal *
						eAVBD_TET_INCIDENCE_PACKET_WIDTH + lane;
					if(ordinal >= refs.size())
						break;
					packet.tetIndices[lane] = refs[ordinal].index;
					packet.vertexOrders[lane] = refs[ordinal].vOrder;
					packet.validMask |= PxU8(1u << lane);
				}
				tetIncidencePackets.pushBack(packet);
			}
		}
		tetIncidencePacketProgramValid = validateTetIncidencePacketProgram();
		PX_ASSERT(tetIncidencePacketProgramValid);
		if(!tetIncidencePacketProgramValid)
			invalidateTetIncidencePacketProgram();
	}

	void buildAdjacency(AvbdSoftBodyRuntimeState& runtime)
	{
		elementAdjacency.resize(particleCount);
		for (PxU32 i = 0; i < particleCount; i++)
		{
			elementAdjacency[i].triRefs.clear();
			elementAdjacency[i].tetRefs.clear();
			elementAdjacency[i].bendRefs.clear();
		}

		for (PxU32 ei = 0; ei < triElements.size(); ei++)
		{
			const AvbdTriElement& tri = triElements[ei];
			PxU32 verts[3] = { tri.p0, tri.p1, tri.p2 };
			for (PxU8 v = 0; v < 3; v++)
			{
				PxU32 localIdx = verts[v] - particleStart;
				if (localIdx < particleCount)
				{
					AvbdParticleElementRef ref;
					ref.index = ei; ref.vOrder = v;
					ref.padding[0] = ref.padding[1] = ref.padding[2] = 0;
					elementAdjacency[localIdx].triRefs.pushBack(ref);
				}
			}
		}

		for (PxU32 ei = 0; ei < tetElements.size(); ei++)
		{
			const AvbdTetElement& tet = tetElements[ei];
			PxU32 verts[4] = { tet.p0, tet.p1, tet.p2, tet.p3 };
			for (PxU8 v = 0; v < 4; v++)
			{
				PxU32 localIdx = verts[v] - particleStart;
				if (localIdx < particleCount)
				{
					AvbdParticleElementRef ref;
					ref.index = ei; ref.vOrder = v;
					ref.padding[0] = ref.padding[1] = ref.padding[2] = 0;
					elementAdjacency[localIdx].tetRefs.pushBack(ref);
				}
			}
		}

		for (PxU32 ei = 0; ei < bendElements.size(); ei++)
		{
			const AvbdBendingElement& be = bendElements[ei];
			PxU32 verts[4] = { be.opp0, be.opp1, be.edgeStart, be.edgeEnd };
			for (PxU8 v = 0; v < 4; v++)
			{
				PxU32 localIdx = verts[v] - particleStart;
				if (localIdx < particleCount)
				{
					AvbdParticleElementRef ref;
					ref.index = ei; ref.vOrder = v;
					ref.padding[0] = ref.padding[1] = ref.padding[2] = 0;
					elementAdjacency[localIdx].bendRefs.pushBack(ref);
				}
			}
		}

		runtime.compileObjectiveProgram(particleStart, particleCount);
	}

	void buildParticlePrimalStructuralAccessDescriptor()
	{
		particlePrimalStructuralConflictOffsets.clear();
		particlePrimalStructuralConflictIndices.clear();
		particlePrimalStructuralConflictValid = false;
		particlePrimalStructuralConflictOffsets.resize(particleCount + 1);
		if(!particleCount)
		{
			particlePrimalStructuralConflictValid = true;
			return;
		}

		// This temporary adjacency exists only at topology-compile time.  The
		// persistent form below is compact CSR, so a P4 color plan can walk it
		// without nested PxArray access or allocation in a solve step.
		PxArray<PxArray<PxU32> > localConflicts;
		localConflicts.resize(particleCount);
		auto appendClique = [this, &localConflicts](
			const PxU32* vertices, PxU32 vertexCount)
		{
			for(PxU32 sourceOrder = 0; sourceOrder < vertexCount;
				sourceOrder++)
			{
				const PxU32 source = vertices[sourceOrder];
				if(source < particleStart ||
					source - particleStart >= particleCount)
					continue;
				PxArray<PxU32>& conflicts =
					localConflicts[source - particleStart];
				for(PxU32 targetOrder = 0; targetOrder < vertexCount;
					targetOrder++)
				{
					if(targetOrder == sourceOrder)
						continue;
					const PxU32 target = vertices[targetOrder];
					if(target < particleStart ||
						target - particleStart >= particleCount)
						continue;
					conflicts.pushBack(target - particleStart);
				}
			}
		};

		for(PxU32 elementIndex = 0;
			elementIndex < triElements.size(); elementIndex++)
		{
			const AvbdTriElement& element = triElements[elementIndex];
			const PxU32 vertices[3] =
				{element.p0, element.p1, element.p2};
			appendClique(vertices, 3);
		}
		for(PxU32 elementIndex = 0;
			elementIndex < tetElements.size(); elementIndex++)
		{
			const AvbdTetElement& element = tetElements[elementIndex];
			const PxU32 vertices[4] =
				{element.p0, element.p1, element.p2, element.p3};
			appendClique(vertices, 4);
		}
		for(PxU32 elementIndex = 0;
			elementIndex < bendElements.size(); elementIndex++)
		{
			const AvbdBendingElement& element = bendElements[elementIndex];
			const PxU32 vertices[4] =
				{element.opp0, element.opp1,
					element.edgeStart, element.edgeEnd};
			appendClique(vertices, 4);
		}

		for(PxU32 localIndex = 0; localIndex < particleCount;
			localIndex++)
		{
			particlePrimalStructuralConflictOffsets[localIndex] =
				particlePrimalStructuralConflictIndices.size();
			PxArray<PxU32>& conflicts = localConflicts[localIndex];
			PxSort(conflicts.begin(), conflicts.size());
			PxU32 uniqueCount = 0;
			for(PxU32 conflictIndex = 0;
				conflictIndex < conflicts.size(); conflictIndex++)
			{
				const PxU32 conflict = conflicts[conflictIndex];
				if(conflict == localIndex ||
					(uniqueCount &&
						conflicts[uniqueCount - 1] == conflict))
					continue;
				conflicts[uniqueCount++] = conflict;
				particlePrimalStructuralConflictIndices.pushBack(conflict);
			}
			conflicts.resize(uniqueCount);
		}
		particlePrimalStructuralConflictOffsets[particleCount] =
			particlePrimalStructuralConflictIndices.size();
		particlePrimalStructuralConflictValid = true;
	}

	bool validateParticlePrimalStructuralAccessDescriptor() const
	{
		if(!particlePrimalStructuralConflictValid ||
			particlePrimalStructuralConflictOffsets.size() !=
				particleCount + 1 ||
			particlePrimalStructuralConflictOffsets[0] != 0 ||
			particlePrimalStructuralConflictOffsets[particleCount] !=
				particlePrimalStructuralConflictIndices.size())
			return false;
		for(PxU32 localIndex = 0; localIndex < particleCount;
			localIndex++)
		{
			const PxU32 begin =
				particlePrimalStructuralConflictOffsets[localIndex];
			const PxU32 end =
				particlePrimalStructuralConflictOffsets[localIndex + 1];
			if(begin > end ||
				end > particlePrimalStructuralConflictIndices.size())
				return false;
			for(PxU32 conflictIndex = begin; conflictIndex < end;
				conflictIndex++)
			{
				const PxU32 conflict =
					particlePrimalStructuralConflictIndices[conflictIndex];
				if(conflict >= particleCount || conflict == localIndex ||
					(conflictIndex > begin &&
						particlePrimalStructuralConflictIndices[
							conflictIndex - 1] >= conflict))
					return false;
				const PxU32 reverseBegin =
					particlePrimalStructuralConflictOffsets[conflict];
				const PxU32 reverseEnd =
					particlePrimalStructuralConflictOffsets[conflict + 1];
				bool reverseFound = false;
				for(PxU32 reverseIndex = reverseBegin;
					reverseIndex < reverseEnd; reverseIndex++)
				{
					if(particlePrimalStructuralConflictIndices[
						reverseIndex] == localIndex)
					{
						reverseFound = true;
						break;
					}
				}
				if(!reverseFound)
					return false;
			}
		}
		return true;
	}

	void buildSurfaceTriangles(
		const PxArray<AvbdSoftParticle>& particles)
	{
		surfaceTriangles.clear();
		surfaceVertices.clear();
		surfaceEdges.clear();
		surfaceTriangleElementIndices.clear();
		surfaceTriangleTetElementIndices.clear();

		if (!tetrahedra.empty())
		{
			// Newton-style sort + unique-count: collect all tet faces with a
			// canonical (sorted) key alongside the original winding, sort by
			// key, then emit faces that appear exactly once (boundary).

			struct FaceRecord
			{
				PxU64 keyHi;   // upper 40 bits of canonical key (smallest vertex)
				PxU32 keyLo;   // lower 20 bits packed into PxU32 for simplicity
				PxU32 v0, v1, v2;  // original winding
			};

			// More compact: pack full key into single PxU64
			// key = (sorted_a << 40) | (sorted_b << 20) | sorted_c
			// This fits since vertex indices < 2^20 = 1M for practical meshes

			const PxU32 numTets = tetrahedra.size() / 4;
			PxArray<PxU32> faceIndices; // flat: 3 * (numTets * 4) original winding indices
			PxArray<PxU64> faceKeys;    // canonical sorted key per face
			PxArray<PxU32> faceElementIndices;
			faceIndices.reserve(numTets * 4 * 3);
			faceKeys.reserve(numTets * 4);
			faceElementIndices.reserve(numTets * 4);

			// 4 faces per tet, following Newton's winding convention
			static const int faceLUT[4][3] = {
				{0,2,1}, {1,2,3}, {0,1,3}, {0,3,2}
			};

			for (PxU32 ti = 0; ti < numTets; ti++)
			{
				PxU32 tv[4] = {
					tetrahedra[ti * 4]     + particleStart,
					tetrahedra[ti * 4 + 1] + particleStart,
					tetrahedra[ti * 4 + 2] + particleStart,
					tetrahedra[ti * 4 + 3] + particleStart
				};
				for (int f = 0; f < 4; f++)
				{
					PxU32 a = tv[faceLUT[f][0]];
					PxU32 b = tv[faceLUT[f][1]];
					PxU32 c = tv[faceLUT[f][2]];
					faceIndices.pushBack(a);
					faceIndices.pushBack(b);
					faceIndices.pushBack(c);
					faceElementIndices.pushBack(ti);

					// Canonical key: sort the 3 indices
					PxU32 sa = a, sb = b, sc = c;
					if (sa > sb) { PxU32 t = sa; sa = sb; sb = t; }
					if (sb > sc) { PxU32 t = sb; sb = sc; sc = t; }
					if (sa > sb) { PxU32 t = sa; sa = sb; sb = t; }
					faceKeys.pushBack((PxU64(sa) << 40) | (PxU64(sb) << 20) | PxU64(sc));
				}
			}

			// Build an index array and sort it by key (indirect sort preserves
			// the original winding in faceIndices)
			const PxU32 numFaces = faceKeys.size();
			PxArray<PxU32> order;
			order.resize(numFaces);
			for (PxU32 i = 0; i < numFaces; i++) order[i] = i;

			const PxU64* keyPtr = faceKeys.begin();
			PxSort(order.begin(), order.size(),
				[keyPtr](PxU32 a, PxU32 b) { return keyPtr[a] < keyPtr[b]; });

			// Linear scan: emit faces whose key appears exactly once
			for (PxU32 i = 0; i < numFaces; )
			{
				PxU32 run = 1;
				while (i + run < numFaces && faceKeys[order[i + run]] == faceKeys[order[i]])
					run++;
				if (run == 1)
				{
					PxU32 base = order[i] * 3;
					surfaceTriangles.pushBack(faceIndices[base]);
					surfaceTriangles.pushBack(faceIndices[base + 1]);
					surfaceTriangles.pushBack(faceIndices[base + 2]);
					surfaceTriangleElementIndices.pushBack(
						faceElementIndices[order[i]]);
				}
				i += run;
			}
		}
		else if (!triangles.empty())
		{
			// For triangle mesh: all triangles are surface
			for (PxU32 i = 0; i + 2 < triangles.size(); i += 3)
			{
				surfaceTriangles.pushBack(triangles[i] + particleStart);
				surfaceTriangles.pushBack(triangles[i+1] + particleStart);
				surfaceTriangles.pushBack(triangles[i+2] + particleStart);
				surfaceTriangleElementIndices.pushBack(i / 3);
			}
		}

		surfaceVertices = surfaceTriangles;
		PxSort(surfaceVertices.begin(), surfaceVertices.size());
		if(!surfaceVertices.empty())
		{
			PxU32 writeIndex = 1;
			for(PxU32 i = 1; i < surfaceVertices.size(); ++i)
			{
				if(surfaceVertices[i] !=
					surfaceVertices[writeIndex - 1])
					surfaceVertices[writeIndex++] =
						surfaceVertices[i];
			}
			surfaceVertices.resize(writeIndex);
		}

		struct SurfaceHalfEdge
		{
			PxU64 key;
			PxU32 faceIndex;
		};
		PxArray<SurfaceHalfEdge> halfEdges;
		halfEdges.reserve(surfaceTriangles.size());
		for(PxU32 i = 0; i + 2 < surfaceTriangles.size(); i += 3)
		{
			const PxU32 vertices[3] =
			{
				surfaceTriangles[i],
				surfaceTriangles[i + 1],
				surfaceTriangles[i + 2]
			};
			for(PxU32 edgeIndex = 0; edgeIndex < 3; edgeIndex++)
			{
				const PxU32 a = vertices[edgeIndex];
				const PxU32 b = vertices[(edgeIndex + 1) % 3];
				const PxU32 lo = PxMin(a, b);
				const PxU32 hi = PxMax(a, b);
				SurfaceHalfEdge halfEdge;
				halfEdge.key = (PxU64(lo) << 32) | PxU64(hi);
				halfEdge.faceIndex = i / 3;
				halfEdges.pushBack(halfEdge);
			}
		}
		PxSort(
			halfEdges.begin(), halfEdges.size(),
			[](const SurfaceHalfEdge& a, const SurfaceHalfEdge& b)
			{
				return a.key < b.key;
			});
		for(PxU32 i = 0; i < halfEdges.size();)
		{
			const PxU64 key = halfEdges[i].key;
			PxU32 run = 1;
			while(i + run < halfEdges.size() &&
				halfEdges[i + run].key == key)
				run++;
			AvbdEdgeInfo edge;
			edge.p0 = PxU32(key >> 32);
			edge.p1 = PxU32(key & 0xffffffffu);
			edge.restLength =
				(particles[edge.p1].position -
				 particles[edge.p0].position).magnitude();
			edge.adjacentSurfaceFace0 = halfEdges[i].faceIndex;
			if(run == 2)
			{
				edge.adjacentSurfaceFace1 =
					halfEdges[i + 1].faceIndex;
				const PxU32 face0 = edge.adjacentSurfaceFace0 * 3;
				const PxU32 face1 = edge.adjacentSurfaceFace1 * 3;
				const PxVec3 normal0 =
					(particles[surfaceTriangles[face0 + 1]].position -
					 particles[surfaceTriangles[face0]].position).cross(
						particles[surfaceTriangles[face0 + 2]].position -
						particles[surfaceTriangles[face0]].position);
				const PxVec3 normal1 =
					(particles[surfaceTriangles[face1 + 1]].position -
					 particles[surfaceTriangles[face1]].position).cross(
						particles[surfaceTriangles[face1 + 2]].position -
						particles[surfaceTriangles[face1]].position);
				const PxReal normalLengthProduct =
					normal0.magnitude() * normal1.magnitude();
				// Stable rest-topology ownership prevents a planar seam from
				// flickering into and out of the edge manifold as the FEM surface
				// deforms. A genuine crease, curved tessellation edge, open edge or
				// non-manifold edge remains active.
				if(normalLengthProduct > 1.0e-12f &&
					normal0.dot(normal1) >=
						normalLengthProduct * 0.9999f)
					edge.collisionFeature = false;
			}
			surfaceEdges.pushBack(edge);
			i += run;
		}
	}

	void buildSurfaceTriangleTetElementIndices()
	{
		// The mapping is a compiled-topology property: current and predicted
		// particle positions do not affect it.  Resolving it here removes the
		// former O(surfaceTriangles * tetElements) source-element scan from
		// every volume self-collision redetection.
		surfaceTriangleTetElementIndices.resize(
			surfaceTriangleElementIndices.size());
		for(PxU32 triangleIndex = 0;
			triangleIndex <
				surfaceTriangleTetElementIndices.size();
			triangleIndex++)
		{
			surfaceTriangleTetElementIndices[triangleIndex] =
				PX_MAX_U32;
		}
		if(tetElements.empty() || tetrahedra.empty())
			return;

		PxArray<PxU32> sourceToTetElement;
		sourceToTetElement.resize(tetrahedra.size() / 4);
		for(PxU32 sourceIndex = 0;
			sourceIndex < sourceToTetElement.size(); sourceIndex++)
			sourceToTetElement[sourceIndex] = PX_MAX_U32;
		for(PxU32 tetElementIndex = 0;
			tetElementIndex < tetElements.size(); tetElementIndex++)
		{
			const PxU32 sourceElementIndex =
				tetElements[tetElementIndex].sourceElementIndex;
			if(sourceElementIndex < sourceToTetElement.size())
				sourceToTetElement[sourceElementIndex] =
					tetElementIndex;
		}
		for(PxU32 triangleIndex = 0;
			triangleIndex <
				surfaceTriangleElementIndices.size(); triangleIndex++)
		{
			const PxU32 sourceElementIndex =
				surfaceTriangleElementIndices[triangleIndex];
			if(sourceElementIndex < sourceToTetElement.size())
				surfaceTriangleTetElementIndices[triangleIndex] =
					sourceToTetElement[sourceElementIndex];
		}
	}

	PxU32 buildSurfaceTriangleBvhNode(PxU32 first, PxU32 count)
	{
		const PxU32 nodeIndex = surfaceTriangleBvhNodes.size();
		AvbdSurfaceTriangleBvhNode node;
		node.minimum = PxVec3(PX_MAX_F32);
		node.maximum = PxVec3(-PX_MAX_F32);
		node.leftChild = PX_MAX_U32;
		node.rightChild = PX_MAX_U32;
		node.firstTriangle = first;
		node.triangleCount = count;
		for(PxU32 entry = first; entry < first + count; entry++)
		{
			const PxU32 triangleIndex =
				surfaceTriangleBvhTriangleIndices[entry];
			const PxU32 triangleOffset = triangleIndex * 3;
			const PxU32 vertex0 = surfaceTriangles[triangleOffset];
			const PxU32 vertex1 = surfaceTriangles[triangleOffset + 1];
			const PxU32 vertex2 = surfaceTriangles[triangleOffset + 2];
			const PxVec3& point0 =
				selfCollisionRestPositions[vertex0 - particleStart];
			const PxVec3& point1 =
				selfCollisionRestPositions[vertex1 - particleStart];
			const PxVec3& point2 =
				selfCollisionRestPositions[vertex2 - particleStart];
			node.minimum = node.minimum.minimum(point0).
				minimum(point1).minimum(point2);
			node.maximum = node.maximum.maximum(point0).
				maximum(point1).maximum(point2);
		}
		surfaceTriangleBvhNodes.pushBack(node);
		if(count <= 4)
			return nodeIndex;

		const PxVec3 extent = node.maximum - node.minimum;
		const PxU32 axis = extent.y > extent.x && extent.y >= extent.z
			? 1u : extent.z > extent.x && extent.z > extent.y ? 2u : 0u;
		PxSort(
			surfaceTriangleBvhTriangleIndices.begin() + first, count,
			[this, axis](PxU32 lhs, PxU32 rhs)
			{
				const PxU32 lhsOffset = lhs * 3;
				const PxU32 rhsOffset = rhs * 3;
				const PxU32 lhsVertices[3] =
				{
					surfaceTriangles[lhsOffset],
					surfaceTriangles[lhsOffset + 1],
					surfaceTriangles[lhsOffset + 2]
				};
				const PxU32 rhsVertices[3] =
				{
					surfaceTriangles[rhsOffset],
					surfaceTriangles[rhsOffset + 1],
					surfaceTriangles[rhsOffset + 2]
				};
				const PxVec3 lhsCenter =
					(selfCollisionRestPositions[
						lhsVertices[0] - particleStart] +
					 selfCollisionRestPositions[
						lhsVertices[1] - particleStart] +
					 selfCollisionRestPositions[
						lhsVertices[2] - particleStart]) *
					(1.0f / 3.0f);
				const PxVec3 rhsCenter =
					(selfCollisionRestPositions[
						rhsVertices[0] - particleStart] +
					 selfCollisionRestPositions[
						rhsVertices[1] - particleStart] +
					 selfCollisionRestPositions[
						rhsVertices[2] - particleStart]) *
					(1.0f / 3.0f);
				const PxReal lhsValue = axis == 0 ? lhsCenter.x :
					axis == 1 ? lhsCenter.y : lhsCenter.z;
				const PxReal rhsValue = axis == 0 ? rhsCenter.x :
					axis == 1 ? rhsCenter.y : rhsCenter.z;
				return lhsValue == rhsValue ? lhs < rhs : lhsValue < rhsValue;
			});
		const PxU32 leftCount = count / 2;
		const PxU32 leftChild = buildSurfaceTriangleBvhNode(
			first, leftCount);
		const PxU32 rightChild = buildSurfaceTriangleBvhNode(
			first + leftCount, count - leftCount);
		surfaceTriangleBvhNodes[nodeIndex].leftChild = leftChild;
		surfaceTriangleBvhNodes[nodeIndex].rightChild = rightChild;
		return nodeIndex;
	}

	void buildSurfaceTriangleBvh()
	{
		surfaceTriangleBvhTriangleIndices.clear();
		surfaceTriangleBvhNodes.clear();
		const PxU32 triangleCount = surfaceTriangles.size() / 3;
		if(triangleCount == 0 ||
			selfCollisionRestPositions.size() != particleCount)
			return;
		surfaceTriangleBvhTriangleIndices.resize(triangleCount);
		for(PxU32 triangleIndex = 0;
			triangleIndex < triangleCount; triangleIndex++)
			surfaceTriangleBvhTriangleIndices[triangleIndex] = triangleIndex;
		buildSurfaceTriangleBvhNode(0, triangleCount);
	}

	void refitSurfaceTriangleBvh(
		const AvbdSoftParticle* particles, bool swept,
		PxArray<AvbdSurfaceBvhNodeBounds>& bounds) const
	{
		PX_ASSERT(bounds.size() == surfaceTriangleBvhNodes.size());
		for(PxU32 index = surfaceTriangleBvhNodes.size(); index > 0;)
		{
			const AvbdSurfaceTriangleBvhNode& node =
				surfaceTriangleBvhNodes[--index];
			AvbdSurfaceBvhNodeBounds& nodeBounds = bounds[index];
			if(!node.isLeaf())
			{
				nodeBounds.minimum = bounds[node.leftChild].minimum.minimum(
					bounds[node.rightChild].minimum);
				nodeBounds.maximum = bounds[node.leftChild].maximum.maximum(
					bounds[node.rightChild].maximum);
				continue;
			}
			nodeBounds.minimum = PxVec3(PX_MAX_F32);
			nodeBounds.maximum = PxVec3(-PX_MAX_F32);
			for(PxU32 entry = node.firstTriangle;
				entry < node.firstTriangle + node.triangleCount; entry++)
			{
				const PxU32 triangleOffset =
					surfaceTriangleBvhTriangleIndices[entry] * 3;
				const PxU32 vertex0 = surfaceTriangles[triangleOffset];
				const PxU32 vertex1 = surfaceTriangles[triangleOffset + 1];
				const PxU32 vertex2 = surfaceTriangles[triangleOffset + 2];
				const AvbdSoftParticle& particle0 = particles[vertex0];
				const AvbdSoftParticle& particle1 = particles[vertex1];
				const AvbdSoftParticle& particle2 = particles[vertex2];
				nodeBounds.minimum = nodeBounds.minimum.minimum(particle0.position).
					minimum(particle1.position).minimum(particle2.position);
				nodeBounds.maximum = nodeBounds.maximum.maximum(particle0.position).
					maximum(particle1.position).maximum(particle2.position);
				if(swept)
				{
					nodeBounds.minimum = nodeBounds.minimum.minimum(
						particle0.initialPosition).minimum(
						particle1.initialPosition).minimum(
						particle2.initialPosition);
					nodeBounds.maximum = nodeBounds.maximum.maximum(
						particle0.initialPosition).maximum(
						particle1.initialPosition).maximum(
						particle2.initialPosition);
				}
			}
		}
	}

	void collectSurfaceTriangleBvhNodeCandidates(
		PxU32 nodeIndex,
		const PxArray<AvbdSurfaceBvhNodeBounds>& bounds,
		const PxVec3& queryMinimum,
		const PxVec3& queryMaximum, PxReal margin,
		PxArray<PxU32>& candidates) const
	{
		const AvbdSurfaceTriangleBvhNode& node =
			surfaceTriangleBvhNodes[nodeIndex];
		const AvbdSurfaceBvhNodeBounds& nodeBounds = bounds[nodeIndex];
		if(nodeBounds.minimum.x > queryMaximum.x + margin ||
			nodeBounds.maximum.x < queryMinimum.x - margin ||
			nodeBounds.minimum.y > queryMaximum.y + margin ||
			nodeBounds.maximum.y < queryMinimum.y - margin ||
			nodeBounds.minimum.z > queryMaximum.z + margin ||
			nodeBounds.maximum.z < queryMinimum.z - margin)
			return;
		if(!node.isLeaf())
		{
			collectSurfaceTriangleBvhNodeCandidates(
				node.leftChild, bounds, queryMinimum, queryMaximum,
				margin, candidates);
			collectSurfaceTriangleBvhNodeCandidates(
				node.rightChild, bounds, queryMinimum, queryMaximum,
				margin, candidates);
			return;
		}
		for(PxU32 entry = node.firstTriangle;
			entry < node.firstTriangle + node.triangleCount; entry++)
			candidates.pushBack(surfaceTriangleBvhTriangleIndices[entry]);
	}

	void collectSurfaceTriangleBvhCandidates(
		const PxVec3& queryMinimum, const PxVec3& queryMaximum,
		PxReal margin,
		const PxArray<AvbdSurfaceBvhNodeBounds>& bounds,
		PxArray<PxU32>& candidates) const
	{
		candidates.clear();
		if(surfaceTriangleBvhNodes.empty())
			return;
		PX_ASSERT(bounds.size() == surfaceTriangleBvhNodes.size());
		collectSurfaceTriangleBvhNodeCandidates(
			0, bounds, queryMinimum, queryMaximum, margin, candidates);
		PxSort(candidates.begin(), candidates.size());
	}

	PxU32 buildSurfaceEdgeBvhNode(PxU32 first, PxU32 count)
	{
		const PxU32 nodeIndex = surfaceEdgeBvhNodes.size();
		AvbdSurfaceEdgeBvhNode node;
		node.minimum = PxVec3(PX_MAX_F32);
		node.maximum = PxVec3(-PX_MAX_F32);
		node.leftChild = PX_MAX_U32;
		node.rightChild = PX_MAX_U32;
		node.firstEdge = first;
		node.edgeCount = count;
		for(PxU32 entry = first; entry < first + count; entry++)
		{
			const AvbdEdgeInfo& edge = surfaceEdges[
				surfaceEdgeBvhEdgeIndices[entry]];
			const PxVec3& point0 = selfCollisionRestPositions[
				edge.p0 - particleStart];
			const PxVec3& point1 = selfCollisionRestPositions[
				edge.p1 - particleStart];
			node.minimum = node.minimum.minimum(point0).minimum(point1);
			node.maximum = node.maximum.maximum(point0).maximum(point1);
		}
		surfaceEdgeBvhNodes.pushBack(node);
		if(count <= 4)
			return nodeIndex;

		const PxVec3 extent = node.maximum - node.minimum;
		const PxU32 axis = extent.y > extent.x && extent.y >= extent.z
			? 1u : extent.z > extent.x && extent.z > extent.y ? 2u : 0u;
		PxSort(
			surfaceEdgeBvhEdgeIndices.begin() + first, count,
			[this, axis](PxU32 lhs, PxU32 rhs)
			{
				const AvbdEdgeInfo& lhsEdge = surfaceEdges[lhs];
				const AvbdEdgeInfo& rhsEdge = surfaceEdges[rhs];
				const PxVec3 lhsCenter = (
					selfCollisionRestPositions[
						lhsEdge.p0 - particleStart] +
					selfCollisionRestPositions[
						lhsEdge.p1 - particleStart]) * 0.5f;
				const PxVec3 rhsCenter = (
					selfCollisionRestPositions[
						rhsEdge.p0 - particleStart] +
					selfCollisionRestPositions[
						rhsEdge.p1 - particleStart]) * 0.5f;
				const PxReal lhsValue = axis == 0 ? lhsCenter.x :
					axis == 1 ? lhsCenter.y : lhsCenter.z;
				const PxReal rhsValue = axis == 0 ? rhsCenter.x :
					axis == 1 ? rhsCenter.y : rhsCenter.z;
				return lhsValue == rhsValue ? lhs < rhs : lhsValue < rhsValue;
			});
		const PxU32 leftCount = count / 2;
		const PxU32 leftChild = buildSurfaceEdgeBvhNode(first, leftCount);
		const PxU32 rightChild = buildSurfaceEdgeBvhNode(
			first + leftCount, count - leftCount);
		surfaceEdgeBvhNodes[nodeIndex].leftChild = leftChild;
		surfaceEdgeBvhNodes[nodeIndex].rightChild = rightChild;
		return nodeIndex;
	}

	void buildSurfaceEdgeBvh()
	{
		surfaceEdgeBvhEdgeIndices.clear();
		surfaceEdgeBvhNodes.clear();
		if(surfaceEdges.empty() ||
			selfCollisionRestPositions.size() != particleCount)
			return;
		surfaceEdgeBvhEdgeIndices.resize(surfaceEdges.size());
		for(PxU32 edgeIndex = 0; edgeIndex < surfaceEdges.size(); edgeIndex++)
			surfaceEdgeBvhEdgeIndices[edgeIndex] = edgeIndex;
		buildSurfaceEdgeBvhNode(0, surfaceEdges.size());
	}

	void refitSurfaceEdgeBvh(
		const AvbdSoftParticle* particles, bool swept,
		PxArray<AvbdSurfaceBvhNodeBounds>& bounds) const
	{
		PX_ASSERT(bounds.size() == surfaceEdgeBvhNodes.size());
		for(PxU32 index = surfaceEdgeBvhNodes.size(); index > 0;)
		{
			const AvbdSurfaceEdgeBvhNode& node =
				surfaceEdgeBvhNodes[--index];
			AvbdSurfaceBvhNodeBounds& nodeBounds = bounds[index];
			if(!node.isLeaf())
			{
				nodeBounds.minimum = bounds[node.leftChild].minimum.
					minimum(bounds[node.rightChild].minimum);
				nodeBounds.maximum = bounds[node.leftChild].maximum.
					maximum(bounds[node.rightChild].maximum);
				continue;
			}
			nodeBounds.minimum = PxVec3(PX_MAX_F32);
			nodeBounds.maximum = PxVec3(-PX_MAX_F32);
			for(PxU32 entry = node.firstEdge;
				entry < node.firstEdge + node.edgeCount; entry++)
			{
				const AvbdEdgeInfo& edge = surfaceEdges[
					surfaceEdgeBvhEdgeIndices[entry]];
				const AvbdSoftParticle& particle0 = particles[edge.p0];
				const AvbdSoftParticle& particle1 = particles[edge.p1];
				nodeBounds.minimum = nodeBounds.minimum.minimum(particle0.position).
					minimum(particle1.position);
				nodeBounds.maximum = nodeBounds.maximum.maximum(particle0.position).
					maximum(particle1.position);
				if(swept)
				{
					nodeBounds.minimum = nodeBounds.minimum.minimum(
						particle0.initialPosition).minimum(
						particle1.initialPosition);
					nodeBounds.maximum = nodeBounds.maximum.maximum(
						particle0.initialPosition).maximum(
						particle1.initialPosition);
				}
			}
		}
	}

	void collectSurfaceEdgeBvhNodeCandidates(
		PxU32 nodeIndex,
		const PxArray<AvbdSurfaceBvhNodeBounds>& bounds,
		const PxVec3& queryMinimum,
		const PxVec3& queryMaximum, PxReal margin,
		PxArray<PxU32>& candidates) const
	{
		const AvbdSurfaceEdgeBvhNode& node = surfaceEdgeBvhNodes[nodeIndex];
		const AvbdSurfaceBvhNodeBounds& nodeBounds = bounds[nodeIndex];
		if(nodeBounds.minimum.x > queryMaximum.x + margin ||
			nodeBounds.maximum.x < queryMinimum.x - margin ||
			nodeBounds.minimum.y > queryMaximum.y + margin ||
			nodeBounds.maximum.y < queryMinimum.y - margin ||
			nodeBounds.minimum.z > queryMaximum.z + margin ||
			nodeBounds.maximum.z < queryMinimum.z - margin)
			return;
		if(!node.isLeaf())
		{
			collectSurfaceEdgeBvhNodeCandidates(
				node.leftChild, bounds, queryMinimum, queryMaximum, margin,
				candidates);
			collectSurfaceEdgeBvhNodeCandidates(
				node.rightChild, bounds, queryMinimum, queryMaximum, margin,
				candidates);
			return;
		}
		for(PxU32 entry = node.firstEdge;
			entry < node.firstEdge + node.edgeCount; entry++)
			candidates.pushBack(surfaceEdgeBvhEdgeIndices[entry]);
	}

	void collectSurfaceEdgeBvhCandidates(
		const PxVec3& queryMinimum, const PxVec3& queryMaximum,
		PxReal margin,
		const PxArray<AvbdSurfaceBvhNodeBounds>& bounds,
		PxArray<PxU32>& candidates) const
	{
		candidates.clear();
		if(surfaceEdgeBvhNodes.empty())
			return;
		PX_ASSERT(bounds.size() == surfaceEdgeBvhNodes.size());
		collectSurfaceEdgeBvhNodeCandidates(
			0, bounds, queryMinimum, queryMaximum, margin, candidates);
		// Self EE evaluates every eligible unordered pair exactly once using
		// min/max edge identity; unlike soft-soft VF it does not choose a
		// first triangle or a tie owner from candidate order.  The fixed tree
		// traversal is deterministic, and avoiding a per-query candidate sort
		// is essential when a dense cloth issues one query per boundary edge.
	}

	void buildSelfCollisionRestVertexTriangleFilter()
	{
		const PxReal filterDistance =
			PxMax(selfCollisionFilterDistance, 0.0f);
		selfCollisionRestFilteredTriangles.clear();
		selfCollisionRestFilterCacheDistance = filterDistance;
		selfCollisionRestFilterCacheValid = false;
		selfCollisionRestFilterCacheFallback = false;
		if(filterDistance <= 0.0f ||
			selfCollisionRestPositions.size() != particleCount)
			return;

		selfCollisionRestFilteredTriangles.resize(particleCount);
		const PxU64 pairBudget =
			(2ull * 1024ull * 1024ull) / sizeof(PxU32);
		PxU64 pairCount = 0;
		PxArray<AvbdSelfCollisionTriangleBounds> triangleBounds;
		triangleBounds.reserve(surfaceTriangles.size() / 3);
		for(PxU32 triangleOffset = 0;
			triangleOffset + 2 < surfaceTriangles.size();
			triangleOffset += 3)
		{
			const PxU32 vertex0 = surfaceTriangles[triangleOffset];
			const PxU32 vertex1 = surfaceTriangles[triangleOffset + 1];
			const PxU32 vertex2 = surfaceTriangles[triangleOffset + 2];
			if(vertex0 < particleStart || vertex1 < particleStart ||
				vertex2 < particleStart ||
				vertex0 - particleStart >= particleCount ||
				vertex1 - particleStart >= particleCount ||
				vertex2 - particleStart >= particleCount)
				continue;
			AvbdSelfCollisionTriangleBounds bounds;
			bounds.triangleOffset = triangleOffset;
			bounds.minimum =
				selfCollisionRestPositions[vertex0 - particleStart].minimum(
					selfCollisionRestPositions[vertex1 - particleStart]).minimum(
					selfCollisionRestPositions[vertex2 - particleStart]);
			bounds.maximum =
				selfCollisionRestPositions[vertex0 - particleStart].maximum(
					selfCollisionRestPositions[vertex1 - particleStart]).maximum(
					selfCollisionRestPositions[vertex2 - particleStart]);
			triangleBounds.pushBack(bounds);
		}
		PxSort(
			triangleBounds.begin(), triangleBounds.size(),
			[](const AvbdSelfCollisionTriangleBounds& a,
			   const AvbdSelfCollisionTriangleBounds& b)
			{
				return a.minimum.x < b.minimum.x;
			});

		PxArray<AvbdSelfCollisionVertexSweepEntry> sortedVertices;
		sortedVertices.resize(particleCount);
		for(PxU32 localIndex = 0; localIndex < particleCount;
			localIndex++)
		{
			sortedVertices[localIndex].localIndex = localIndex;
			sortedVertices[localIndex].minimumX =
				selfCollisionRestPositions[localIndex].x;
			sortedVertices[localIndex].maximumX =
				selfCollisionRestPositions[localIndex].x;
		}
		PxSort(
			sortedVertices.begin(), sortedVertices.size(),
			[](const AvbdSelfCollisionVertexSweepEntry& a,
			   const AvbdSelfCollisionVertexSweepEntry& b)
			{
				return a.minimumX < b.minimumX;
			});

		PxArray<PxU32> activeTriangles;
		activeTriangles.reserve(triangleBounds.size());
		PxU32 triangleCursor = 0;
		for(PxU32 sortedVertexIndex = 0;
			sortedVertexIndex < sortedVertices.size(); sortedVertexIndex++)
		{
			const PxU32 localIndex =
				sortedVertices[sortedVertexIndex].localIndex;
			const PxVec3& point = selfCollisionRestPositions[localIndex];
			while(triangleCursor < triangleBounds.size() &&
				triangleBounds[triangleCursor].minimum.x <=
					point.x + filterDistance)
				activeTriangles.pushBack(triangleCursor++);
			for(PxU32 activeIndex = 0;
				activeIndex < activeTriangles.size();)
			{
				const AvbdSelfCollisionTriangleBounds& bounds =
					triangleBounds[activeTriangles[activeIndex]];
				if(bounds.maximum.x < point.x - filterDistance)
				{
					activeTriangles[activeIndex] = activeTriangles.back();
					activeTriangles.popBack();
					continue;
				}
				activeIndex++;
			}
			for(PxU32 activeIndex = 0;
				activeIndex < activeTriangles.size(); activeIndex++)
			{
				const AvbdSelfCollisionTriangleBounds& bounds =
					triangleBounds[activeTriangles[activeIndex]];
				if(bounds.minimum.y > point.y + filterDistance ||
					bounds.maximum.y < point.y - filterDistance ||
					bounds.minimum.z > point.z + filterDistance ||
					bounds.maximum.z < point.z - filterDistance)
					continue;
				const PxU32 triangleOffset = bounds.triangleOffset;
				const PxU32 vertex0 = surfaceTriangles[triangleOffset];
				const PxU32 vertex1 = surfaceTriangles[triangleOffset + 1];
				const PxU32 vertex2 = surfaceTriangles[triangleOffset + 2];
				const PxU32 localVertex0 = vertex0 - particleStart;
				const PxU32 localVertex1 = vertex1 - particleStart;
				const PxU32 localVertex2 = vertex2 - particleStart;
				if(localIndex == localVertex0 || localIndex == localVertex1 ||
					localIndex == localVertex2)
					continue;
				const PxReal restDistance =
					avbdGetRestPointTriangleDistance(
						point,
						selfCollisionRestPositions[localVertex0],
						selfCollisionRestPositions[localVertex1],
						selfCollisionRestPositions[localVertex2]);
				if(restDistance > filterDistance)
					continue;
				if(pairCount >= pairBudget)
				{
					selfCollisionRestFilteredTriangles.clear();
					selfCollisionRestFilterCacheFallback = true;
					return;
				}
				selfCollisionRestFilteredTriangles[localIndex].pushBack(
					triangleOffset / 3);
				++pairCount;
			}
		}
		for(PxU32 localIndex = 0;
			localIndex < selfCollisionRestFilteredTriangles.size();
			localIndex++)
		{
			PxArray<PxU32>& filteredForVertex =
				selfCollisionRestFilteredTriangles[localIndex];
			PxSort(filteredForVertex.begin(), filteredForVertex.size());
		}
		selfCollisionRestFilterCacheValid = true;
	}

	void ensureSelfCollisionRestVertexTriangleFilter()
	{
		const PxReal filterDistance =
			PxMax(selfCollisionFilterDistance, 0.0f);
		if(selfCollisionRestFilterCacheDistance == filterDistance &&
			(selfCollisionRestFilterCacheValid ||
				selfCollisionRestFilterCacheFallback))
			return;
		buildSelfCollisionRestVertexTriangleFilter();
	}

	void buildElements(
		const PxArray<AvbdSoftParticle>& particles,
		AvbdSoftBodyMaterialData& material,
		AvbdSoftBodyRuntimeState& runtime)
	{
		material.computeLameParameters();
		if (!triangles.empty())
			buildTriElements(particles);
		if (!tetrahedra.empty())
			buildTetElements(particles);
		if (!triangles.empty())
			buildBendingElements(particles);
		buildEdges(particles);
		buildAdjacency(runtime);
		if(material.coRotationalVolumeModel)
			buildTetIncidencePacketProgram();
		else
			invalidateTetIncidencePacketProgram();
		buildParticlePrimalStructuralAccessDescriptor();
		PX_ASSERT(validateParticlePrimalStructuralAccessDescriptor());
		buildSurfaceTriangles(particles);
		buildSurfaceTriangleTetElementIndices();
		buildSurfaceTriangleBvh();
		buildSurfaceEdgeBvh();
		buildSelfCollisionRestVertexTriangleFilter();
	}
};

struct AvbdSoftBody
{
	AvbdSoftBodyCompiledData compiled;
	AvbdSoftBodyMaterialData material;
	AvbdSoftBodyRuntimeState runtime;

	void buildElements(const PxArray<AvbdSoftParticle>& particles)
	{
		compiled.buildElements(particles, material, runtime);
	}
};

// A speculative body relies on the full old-position -> predicted-position
// sweep to admit first-impact contacts.  This diagnostic is deliberately
// limited to an all-dynamic, unpinned and unattached component: a per-vertex
// initial guess would otherwise manufacture strain at a static or owned
// support.  Reject the entire component rather than shortening only one side
// of a soft-soft swept pair.
PX_FORCE_INLINE bool avbdCanUseSoftAdaptivePrimalInitialization(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies)
{
	if(!avbdUseSoftAdaptivePrimalInitialization())
		return false;
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		if(body.compiled.speculativeCCDEnabled ||
			!body.runtime.pins.empty() ||
			!body.runtime.attachments.empty() ||
			body.compiled.particleStart > numParticles ||
			body.compiled.particleCount >
				numParticles - body.compiled.particleStart)
			return false;
		for(PxU32 localIndex = 0;
			localIndex < body.compiled.particleCount; ++localIndex)
		{
			if(particles[body.compiled.particleStart + localIndex].invMass <=
				0.0f)
				return false;
		}
	}
	return true;
}

// A rigid initial guess changes every dynamic particle of a body together.
// Therefore it has the same whole-component swept-contact restriction as the
// adaptive guess, and additionally needs finite positive masses for its
// mass-weighted fit.  Reject the complete component rather than applying the
// transform to only a subset of bodies or vertices.
PX_FORCE_INLINE bool avbdCanUseSoftRigidPrimalInitialization(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies)
{
	if(!avbdUseSoftRigidPrimalInitialization() || !particles ||
		!softBodies || numParticles == 0 || numSoftBodies == 0)
		return false;
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		if(body.compiled.speculativeCCDEnabled ||
			!body.runtime.pins.empty() ||
			!body.runtime.attachments.empty() ||
			body.compiled.particleCount == 0 ||
			body.compiled.particleStart > numParticles ||
			body.compiled.particleCount >
				numParticles - body.compiled.particleStart)
			return false;
		for(PxU32 localIndex = 0;
			localIndex < body.compiled.particleCount; ++localIndex)
		{
			const AvbdSoftParticle& particle = particles[
				body.compiled.particleStart + localIndex];
			if(!PxIsFinite(particle.invMass) || particle.invMass <= 0.0f ||
				!PxIsFinite(particle.mass) || particle.mass <= 0.0f)
				return false;
		}
	}
	return true;
}

// Select the wide backend once at a sweep boundary. Bodies with another
// material model, invalid packet metadata, or no complete packet must stay on
// the canonical scalar call graph; paying candidate dispatch per particle is
// measurably more expensive than the scalar solve for those workloads.
PX_NOINLINE inline AvbdCpuIsaCorotationalTetPacket8Fn
avbdSelectCorotationalTetPacketKernel(
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies)
{
	AvbdCpuIsaCorotationalTetPacket8Fn kernel =
		avbdGetCorotationalTetPacketKernel();
	if(!kernel)
		return NULL;
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		if(body.material.coRotationalVolumeModel &&
			body.compiled.tetIncidencePacketProgramValid &&
			body.compiled.tetIncidenceFullPacketCount)
			return kernel;
	}
	return NULL;
}

// This uses the same per-particle min/max traversal as the legacy soft-pair
// broadphase. A caller may compute different body slots concurrently only
// after the output array has been serially sized.
PX_FORCE_INLINE void avbdComputeSoftBodyBounds(
	const AvbdSoftParticle* particles, const AvbdSoftBody& body,
	AvbdSoftBodyBounds& bounds)
{
	bounds = AvbdSoftBodyBounds();
	for(PxU32 particleIndex = 0;
		particleIndex < body.compiled.particleCount; ++particleIndex)
	{
		const AvbdSoftParticle& particle = particles[
			body.compiled.particleStart + particleIndex];
		bounds.currentMinimum = bounds.currentMinimum.minimum(
			particle.position);
		bounds.currentMaximum = bounds.currentMaximum.maximum(
			particle.position);
		// Retain the legacy swept reduction order exactly: current first,
		// then the initial position for each particle.
		bounds.sweptMinimum = bounds.sweptMinimum.minimum(
			particle.position);
		bounds.sweptMaximum = bounds.sweptMaximum.maximum(
			particle.position);
		bounds.sweptMinimum = bounds.sweptMinimum.minimum(
			particle.initialPosition);
		bounds.sweptMaximum = bounds.sweptMaximum.maximum(
			particle.initialPosition);
	}
}

PX_FORCE_INLINE const AvbdSoftBody* avbdFindSoftBodyForParticle(
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxU32 particleIndex)
{
	for(PxU32 i = 0; i < numSoftBodies; i++)
	{
		const AvbdSoftBody& body = softBodies[i];
		const PxU32 start = body.compiled.particleStart;
		if(particleIndex >= start &&
			particleIndex - start < body.compiled.particleCount)
			return &body;
	}
	return NULL;
}

PX_FORCE_INLINE bool avbdIsSoftBodySurfaceVertex(
	const AvbdSoftBody& body,
	PxU32 particleIndex)
{
	const PxArray<PxU32>& surfaceVertices =
		body.compiled.surfaceVertices;
	PxU32 lower = 0;
	PxU32 upper = surfaceVertices.size();
	while(lower < upper)
	{
		const PxU32 middle = lower + (upper - lower) / 2;
		if(surfaceVertices[middle] < particleIndex)
			lower = middle + 1;
		else
			upper = middle;
	}
	return lower < surfaceVertices.size() &&
		surfaceVertices[lower] == particleIndex;
}

PX_FORCE_INLINE bool avbdIsSelfRestVertexTriangleFiltered(
	const AvbdSoftBody& body, PxU32 localVertexIndex,
	PxU32 surfaceTriangleIndex)
{
	const PxArray<PxArray<PxU32> >& filteredTriangles =
		body.compiled.selfCollisionRestFilteredTriangles;
	if(!body.compiled.selfCollisionRestFilterCacheValid ||
		localVertexIndex >= filteredTriangles.size())
		return false;
	const PxArray<PxU32>& filteredForVertex =
		filteredTriangles[localVertexIndex];
	PxU32 lower = 0;
	PxU32 upper = filteredForVertex.size();
	while(lower < upper)
	{
		const PxU32 middle = lower + (upper - lower) / 2;
		if(filteredForVertex[middle] < surfaceTriangleIndex)
			lower = middle + 1;
		else
			upper = middle;
	}
	return lower < filteredForVertex.size() &&
		filteredForVertex[lower] == surfaceTriangleIndex;
}

PX_FORCE_INLINE void avbdResetSoftContactDepenetrationLimits(
	AvbdSoftContact* contacts, PxU32 numContacts)
{
	for(PxU32 contactIndex = 0;
		contactIndex < numContacts; contactIndex++)
	{
		contacts[contactIndex].state.
			depenetrationConstraintOffset = 0.0f;
		contacts[contactIndex].state.
			depenetrationLimitInitialized = false;
	}
}

PX_FORCE_INLINE void
avbdInitializeSoftContactDepenetrationLimitAtSurfacePoint(
	AvbdSoftContact& contact,
	const AvbdSoftParticle* particles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const PxVec3& initialSurfacePoint, PxReal dt)
{
	AvbdSoftContactAugmentedState& state = contact.state;
	if(state.depenetrationLimitInitialized)
		return;

	const AvbdSoftContactGeometry& geometry = contact.geometry;
	const PxU32 queryRepresentative = geometry.hasWeightedQueryPoint()
		? geometry.queryPoint.particleIndices[0]
		: geometry.particleIdx;
	const AvbdSoftBody* queryBody = geometry.queryBodyIndex < numSoftBodies
		? &softBodies[geometry.queryBodyIndex]
		: avbdFindSoftBodyForParticle(
			softBodies, numSoftBodies, queryRepresentative);
	PxReal maxDepenetrationVelocity = queryBody
		? queryBody->compiled.maxDepenetrationVelocity
		: PX_MAX_F32;
	if(geometry.hasDeformableSurfaceTarget())
	{
		const PxU32 targetRepresentative =
			geometry.hasWeightedTargetPoint()
				? geometry.targetPoint.particleIndices[0]
				: geometry.surfaceParticleIndices[0];
		const AvbdSoftBody* targetBody =
			geometry.source.targetBodyIndex < numSoftBodies
				? &softBodies[geometry.source.targetBodyIndex]
				: avbdFindSoftBodyForParticle(
					softBodies, numSoftBodies, targetRepresentative);
		if(targetBody)
			maxDepenetrationVelocity = PxMin(
				maxDepenetrationVelocity,
				targetBody->compiled.maxDepenetrationVelocity);
	}
	maxDepenetrationVelocity =
		PxMax(maxDepenetrationVelocity, 0.0f);
	state.depenetrationConstraintOffset = 0.0f;
	state.depenetrationLimitInitialized = true;
	if(dt <= 0.0f ||
		maxDepenetrationVelocity >= 1.0e20f)
		return;

	const PxVec3 initialQueryPoint =
		avbdGetSoftContactInitialQueryPoint(
			geometry, particles);
	const PxReal initialConstraint =
		avbdEvaluateSoftContactNormalConstraint(
			geometry, initialQueryPoint, initialSurfacePoint);
	const PxReal maxRecoveryDistance =
		maxDepenetrationVelocity * dt;
	state.depenetrationConstraintOffset =
		PxMin(0.0f, initialConstraint + maxRecoveryDistance);
	if(state.depenetrationConstraintOffset < 0.0f)
	{
		// A carried normal multiplier can otherwise spend the new frame's
		// finite bias budget before the shifted row has converged.
		state.alLambda = 0.0f;
	}
}

PX_FORCE_INLINE void avbdInitializeSoftContactDepenetrationLimits(
	AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdSoftParticle* particles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxReal dt)
{
	for(PxU32 contactIndex = 0;
		contactIndex < numContacts; contactIndex++)
	{
		AvbdSoftContact& contact = contacts[contactIndex];
		avbdInitializeSoftContactDepenetrationLimitAtSurfacePoint(
			contact, particles, softBodies, numSoftBodies,
			avbdGetSoftContactInitialSurfacePoint(
				contact.geometry, particles),
			dt);
	}
}

PX_FORCE_INLINE PxReal avbdCombineDeformableRigidFriction(
	PxReal deformableFriction, PxReal rigidFriction,
	PxU8 frictionCombineMode)
{
	const PxReal soft = PxMax(deformableFriction, 0.0f);
	const PxReal rigid = PxMax(rigidFriction, 0.0f);
	switch(static_cast<PxCombineMode::Enum>(frictionCombineMode))
	{
	case PxCombineMode::eMIN:
		return PxMin(soft, rigid);
	case PxCombineMode::eMULTIPLY:
		return soft * rigid;
	case PxCombineMode::eMAX:
		return PxMax(soft, rigid);
	case PxCombineMode::eAVERAGE:
	default:
		return 0.5f * (soft + rigid);
	}
}

// =============================================================================
// VBD Force/Hessian evaluators
// =============================================================================

PX_FORCE_INLINE void avbdEvaluateStVKForceHessian(
	const AvbdTriElement& tri, int vOrder,
	PxReal mu, PxReal lam,
	const AvbdSoftParticle* particles,
	PxVec3& outForce, PxMat33& outHessian)
{
	PxVec3 x0 = particles[tri.p0].position;
	PxVec3 x01 = particles[tri.p1].position - x0;
	PxVec3 x02 = particles[tri.p2].position - x0;

	PxReal D00 = tri.DmInv00, D01 = tri.DmInv01;
	PxReal D10 = tri.DmInv10, D11 = tri.DmInv11;

	PxVec3 f0 = x01 * D00 + x02 * D10;
	PxVec3 f1 = x01 * D01 + x02 * D11;

	PxReal f0f0 = f0.dot(f0);
	PxReal f1f1 = f1.dot(f1);
	PxReal f0f1 = f0.dot(f1);

	PxReal G00 = 0.5f * (f0f0 - 1.0f);
	PxReal G11 = 0.5f * (f1f1 - 1.0f);
	PxReal G01 = 0.5f * f0f1;

	PxReal Gfro2 = G00 * G00 + G11 * G11 + 2.0f * G01 * G01;
	if (Gfro2 < 1e-20f)
	{
		outForce = PxVec3(0.0f);
		outHessian = PxMat33(PxZero);
		return;
	}

	PxReal trG = G00 + G11;
	PxReal ltrG = lam * trG;
	PxReal twoMu = 2.0f * mu;
	PxVec3 PK1_0 = f0 * (twoMu * G00 + ltrG) + f1 * (twoMu * G01);
	PxVec3 PK1_1 = f0 * (twoMu * G01) + f1 * (twoMu * G11 + ltrG);

	PxReal df0, df1;
	if (vOrder == 0)      { df0 = -D00 - D10; df1 = -D01 - D11; }
	else if (vOrder == 1) { df0 = D00; df1 = D01; }
	else                  { df0 = D10; df1 = D11; }

	outForce = (PK1_0 * df0 + PK1_1 * df1) * (-tri.restArea);

	PxReal df0sq = df0 * df0;
	PxReal df1sq = df1 * df1;
	PxReal df0df1 = df0 * df1;

	PxReal Ic = f0f0 + f1f1;
	PxReal two_dpsi_dIc = -mu + (0.5f * Ic - 1.0f) * lam;
	PxMat33 I33 = PxMat33(PxIdentity);

	PxMat33 f0f0m = avbdOuter(f0, f0);
	PxMat33 f1f1m = avbdOuter(f1, f1);
	PxMat33 f0f1m = avbdOuter(f0, f1);
	PxMat33 f1f0m = avbdOuter(f1, f0);

	PxMat33 H00 = f0f0m * lam + I33 * two_dpsi_dIc
	            + (I33 * f0f0 + f0f0m * 2.0f + f1f1m) * mu;
	PxMat33 H01 = f0f1m * lam + (I33 * f0f1 + f1f0m) * mu;
	PxMat33 H11 = f1f1m * lam + I33 * two_dpsi_dIc
	            + (I33 * f1f1 + f1f1m * 2.0f + f0f0m) * mu;

	PxReal area = tri.restArea;
	outHessian = H00 * (df0sq * area) + H11 * (df1sq * area)
	           + (H01 + H01.getTranspose()) * (df0df1 * area);
}

PX_FORCE_INLINE void avbdEvaluateTetDeterminantAndGradient(
	const AvbdTetElement& tet, PxU32 vertexOrder,
	const PxVec3& e1, const PxVec3& e2, const PxVec3& e3,
	PxReal& outDeterminant, PxVec3& outGradient)
{
	PxVec3 currentFaceGradient;
	PxReal currentDeterminant;
	switch(vertexOrder)
	{
	case 0:
		currentFaceGradient = (e3 - e1).cross(e2 - e1);
		currentDeterminant = (-e1).dot(currentFaceGradient);
		break;
	case 1:
		currentFaceGradient = e2.cross(e3);
		currentDeterminant = e1.dot(currentFaceGradient);
		break;
	case 2:
		currentFaceGradient = e3.cross(e1);
		currentDeterminant = e2.dot(currentFaceGradient);
		break;
	default:
		currentFaceGradient = e1.cross(e2);
		currentDeterminant = e3.dot(currentFaceGradient);
		break;
	}
	outDeterminant =
		currentDeterminant * tet.inverseRestDeterminant;
	outGradient =
		currentFaceGradient * tet.inverseRestDeterminant;
}

PX_FORCE_INLINE bool avbdIsFiniteVector(const PxVec3& value)
{
	return PxIsFinite(value.x) && PxIsFinite(value.y) &&
		PxIsFinite(value.z);
}

// The polar iteration already computes and validates the determinant before
// inversion.  PxMat33::getInverse() would recompute that same determinant;
// consume the validated value while preserving its scalar cofactor algebra.
PX_FORCE_INLINE PxMat33 avbdGetInverseTransposeWithDeterminant(
	const PxMat33& matrix, PxReal determinant)
{
	const PxReal invDet = 1.0f / determinant;
	return PxMat33(
		PxVec3(
			invDet * (matrix.column1.y * matrix.column2.z -
				matrix.column2.y * matrix.column1.z),
			invDet * -(matrix.column1.x * matrix.column2.z -
				matrix.column1.z * matrix.column2.x),
			invDet * (matrix.column1.x * matrix.column2.y -
				matrix.column1.y * matrix.column2.x)),
		PxVec3(
			invDet * -(matrix.column0.y * matrix.column2.z -
				matrix.column2.y * matrix.column0.z),
			invDet * (matrix.column0.x * matrix.column2.z -
				matrix.column0.z * matrix.column2.x),
			invDet * -(matrix.column0.x * matrix.column2.y -
				matrix.column0.y * matrix.column2.x)),
		PxVec3(
			invDet * (matrix.column0.y * matrix.column1.z -
				matrix.column0.z * matrix.column1.y),
			invDet * -(matrix.column0.x * matrix.column1.z -
				matrix.column0.z * matrix.column1.x),
			invDet * (matrix.column0.x * matrix.column1.y -
				matrix.column1.x * matrix.column0.y)));
}

PX_FORCE_INLINE PxMat33 avbdExtractCorotationalRotation(
	const PxMat33& deformationGradient)
{
	PxMat33 rotation = deformationGradient;
	PxReal determinant = rotation.getDeterminant();
	if(!PxIsFinite(determinant) || PxAbs(determinant) <= 1.0e-9f)
		rotation = PxMat33(PxIdentity);
	else
	{
		for(PxU32 iteration = 0; iteration < 5; iteration++)
		{
			if(!PxIsFinite(determinant) ||
				PxAbs(determinant) <= 1.0e-9f)
				break;
			const PxMat33 inverseTranspose =
				avbdGetInverseTransposeWithDeterminant(
					rotation, determinant);
			if(!avbdIsFiniteVector(inverseTranspose.column0) ||
				!avbdIsFiniteVector(inverseTranspose.column1) ||
				!avbdIsFiniteVector(inverseTranspose.column2))
				break;
			rotation.column0 =
				(rotation.column0 + inverseTranspose.column0) * 0.5f;
			rotation.column1 =
				(rotation.column1 + inverseTranspose.column1) * 0.5f;
			rotation.column2 =
				(rotation.column2 + inverseTranspose.column2) * 0.5f;
			if(iteration + 1 < 5)
				determinant = rotation.getDeterminant();
		}
	}

	// Finish with an explicitly right-handed orthonormal basis.  The polar
	// iteration alone can retain a reflection for inverted configurations;
	// co-rotational elasticity requires the closest proper rotation.
	PxVec3 column0 = rotation.column0;
	if(!avbdIsFiniteVector(column0) ||
		column0.magnitudeSquared() <= 1.0e-12f)
		column0 = deformationGradient.column0;
	if(!avbdIsFiniteVector(column0) ||
		column0.magnitudeSquared() <= 1.0e-12f)
		column0 = PxVec3(1.0f, 0.0f, 0.0f);
	column0.normalize();

	PxVec3 column1 =
		rotation.column1 -
		column0 * rotation.column1.dot(column0);
	if(!avbdIsFiniteVector(column1) ||
		column1.magnitudeSquared() <= 1.0e-12f)
	{
		const PxVec3 reference =
			PxAbs(column0.x) < 0.8f
				? PxVec3(1.0f, 0.0f, 0.0f)
				: PxVec3(0.0f, 1.0f, 0.0f);
		column1 =
			reference - column0 * reference.dot(column0);
	}
	column1.normalize();
	PxVec3 column2 = column0.cross(column1);
	if(column2.dot(rotation.column2) < 0.0f)
	{
		column1 = -column1;
		column2 = -column2;
	}
	return PxMat33(column0, column1, column2);
}

// Projects only the primal starting point, not the inertial target.  The
// weighted covariance maps initialPosition to predictedPosition, so its polar
// factor is the proper Kabsch rotation R in x' = cPredicted + R(x-cInitial).
// Applying that transform to the complete body retains its current tet shape
// exactly; the ordinary VBD objective still pulls it toward each particle's
// untouched predictedPosition.
PX_FORCE_INLINE bool avbdApplySoftBodyRigidPrimalInitialGuess(
	AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody& body)
{
	const PxU32 particleStart = body.compiled.particleStart;
	const PxU32 particleCount = body.compiled.particleCount;
	if(!particles || particleCount == 0 || particleStart > numParticles ||
		particleCount > numParticles - particleStart)
		return false;

	PxReal totalMass = 0.0f;
	PxVec3 initialCentroid(0.0f);
	PxVec3 predictedCentroid(0.0f);
	for(PxU32 localIndex = 0; localIndex < particleCount; ++localIndex)
	{
		const AvbdSoftParticle& particle = particles[
			particleStart + localIndex];
		if(!PxIsFinite(particle.mass) || particle.mass <= 0.0f ||
			!avbdIsFiniteVector(particle.initialPosition) ||
			!avbdIsFiniteVector(particle.predictedPosition))
			return false;
		totalMass += particle.mass;
		initialCentroid += particle.initialPosition * particle.mass;
		predictedCentroid += particle.predictedPosition * particle.mass;
	}
	if(!PxIsFinite(totalMass) || totalMass <= 1.0e-12f ||
		!avbdIsFiniteVector(initialCentroid) ||
		!avbdIsFiniteVector(predictedCentroid))
		return false;
	const PxReal inverseMass = 1.0f / totalMass;
	initialCentroid *= inverseMass;
	predictedCentroid *= inverseMass;

	PxMat33 covariance(PxZero);
	for(PxU32 localIndex = 0; localIndex < particleCount; ++localIndex)
	{
		const AvbdSoftParticle& particle = particles[
			particleStart + localIndex];
		covariance += avbdOuter(
			particle.predictedPosition - predictedCentroid,
			particle.initialPosition - initialCentroid) * particle.mass;
	}
	if(!avbdIsFiniteVector(covariance.column0) ||
		!avbdIsFiniteVector(covariance.column1) ||
		!avbdIsFiniteVector(covariance.column2))
		return false;
	const PxMat33 rotation = avbdExtractCorotationalRotation(covariance);
	if(!avbdIsFiniteVector(rotation.column0) ||
		!avbdIsFiniteVector(rotation.column1) ||
		!avbdIsFiniteVector(rotation.column2))
		return false;

	// Validate every transformed point before writing any one of them: a bad
	// diagnostic fit must retain the ordinary prediction start wholesale.
	for(PxU32 localIndex = 0; localIndex < particleCount; ++localIndex)
	{
		const AvbdSoftParticle& particle = particles[
			particleStart + localIndex];
		const PxVec3 rigidPosition = predictedCentroid + rotation *
			(particle.initialPosition - initialCentroid);
		if(!avbdIsFiniteVector(rigidPosition))
			return false;
	}
	for(PxU32 localIndex = 0; localIndex < particleCount; ++localIndex)
	{
		AvbdSoftParticle& particle = particles[particleStart + localIndex];
		particle.position = predictedCentroid + rotation *
			(particle.initialPosition - initialCentroid);
	}
	return true;
}

PX_FORCE_INLINE PxReal avbdComputeTetStressCoefficient(
	const AvbdTetElement& tet,
	const AvbdSoftParticle* particles)
{
	const PxVec3 p0 = particles[tet.p0].position;
	const PxMat33 deformationGradient(
		particles[tet.p1].position - p0,
		particles[tet.p2].position - p0,
		particles[tet.p3].position - p0);
	const PxMat33 F = deformationGradient * tet.DmInv;
	const PxMat33 rotation =
		avbdExtractCorotationalRotation(F);
	const PxMat33 coRotatedF =
		rotation.getTranspose() * F;
	const PxMat33 strain =
		(coRotatedF + coRotatedF.getTranspose()) * 0.5f -
		PxMat33(PxIdentity);
	const PxReal q0 = strain.column0.x;
	const PxReal q1 = strain.column1.y;
	const PxReal q2 = strain.column2.z;
	const PxReal q01 = q0 - q1;
	const PxReal q12 = q1 - q2;
	const PxReal q20 = q2 - q0;
	const PxReal coefficient = PxSqrt(
		q01 * q01 + q12 * q12 + q20 * q20) *
		0.7071067811865475244f;
	return PxIsFinite(coefficient)
		? coefficient : PX_MAX_F32;
}

PX_FORCE_INLINE void
avbdEvaluateCorotationalForceHessianPrepared(
	const AvbdTetElement& tet, int vOrder,
	PxReal mu, PxReal lam,
	const AvbdSoftParticle* particles,
	PxVec3& outForce, PxMat33& outHessian,
	AvbdTetVertexLinearization* outLinearization = NULL)
{
	const PxVec3 p0 = particles[tet.p0].position;
	const PxVec3 e1 = particles[tet.p1].position - p0;
	const PxVec3 e2 = particles[tet.p2].position - p0;
	const PxVec3 e3 = particles[tet.p3].position - p0;
	const PxU32 vertexOrder =
		vOrder >= 0 && vOrder < 3 ? PxU32(vOrder) : 3;
	if(outLinearization)
	{
		avbdEvaluateTetDeterminantAndGradient(
			tet, vertexOrder, e1, e2, e3,
			outLinearization->determinant,
			outLinearization->determinantGradient);
	}

	const PxMat33 deformationGradient =
		PxMat33(e1, e2, e3) * tet.DmInv;
	const PxMat33 rotation =
		avbdExtractCorotationalRotation(deformationGradient);
	const PxReal strainTrace =
		rotation.column0.dot(deformationGradient.column0) +
		rotation.column1.dot(deformationGradient.column1) +
		rotation.column2.dot(deformationGradient.column2) -
		3.0f;
	const PxMat33 firstPiola =
		(deformationGradient - rotation) * (2.0f * mu) +
		rotation * (lam * strainTrace);
	const PxVec3& shapeGradient =
		tet.shapeGradients[vertexOrder];
	outForce =
		(firstPiola * shapeGradient) * (-tet.restVolume);

	// A frozen-rotation Gauss-Newton block is symmetric positive
	// semi-definite and gives the exact local stiffness for the linearized
	// co-rotational energy.
	const PxVec3 rotatedGradient =
		rotation * shapeGradient;
	const PxReal gradientNormSq =
		tet.shapeGradientNormSq[vertexOrder];
	outHessian =
		PxMat33::createDiagonal(
			PxVec3(2.0f * mu * gradientNormSq *
				tet.restVolume)) +
		avbdOuter(rotatedGradient, rotatedGradient) *
			(lam * tet.restVolume);
}

PX_FORCE_INLINE void avbdEvaluateNeoHookeanForceHessianPrepared(
	const AvbdTetElement& tet, int vOrder,
	PxReal mu, PxReal lam, PxReal alpha,
	const AvbdSoftParticle* particles,
	PxVec3& outForce, PxMat33& outHessian,
	AvbdTetVertexLinearization* outLinearization = NULL)
{
	PxVec3 p0 = particles[tet.p0].position;
	PxVec3 e1 = particles[tet.p1].position - p0;
	PxVec3 e2 = particles[tet.p2].position - p0;
	PxVec3 e3 = particles[tet.p3].position - p0;

	const PxU32 vertexOrder =
		vOrder >= 0 && vOrder < 3 ? PxU32(vOrder) : 3;
	PxReal J;
	PxVec3 cofm;
	avbdEvaluateTetDeterminantAndGradient(
		tet, vertexOrder, e1, e2, e3, J, cofm);
	if(outLinearization)
	{
		outLinearization->determinant = J;
		outLinearization->determinantGradient = cofm;
	}

	const PxVec3& deformationWeights =
		tet.deformationGradientWeights[vertexOrder];
	const PxVec3 Fm =
		e1 * deformationWeights.x +
		e2 * deformationWeights.y +
		e3 * deformationWeights.z;

	PxReal V0 = tet.restVolume;

	// Inversion protection: clamp J to a small positive value so that
	// fully inverted tets produce bounded restoration forces instead of
	// catastrophic blowup.  The force direction remains correct (cofactor
	// still points toward un-inverting the tet).
	const PxReal Jmin = 0.05f;
	const PxReal Jsafe = PxMax(J, Jmin);

	outForce =
		(Fm * mu + cofm * (lam * (Jsafe - alpha))) * (-V0);

	const PxReal m2 = tet.shapeGradientNormSq[vertexOrder];
	outHessian =
		PxMat33::createDiagonal(PxVec3(mu * m2 * V0)) +
		avbdOuter(cofm, cofm) * (lam * V0);

	// Extra diagonal regularization for severely compressed / inverted tets
	// to keep the Hessian well-conditioned.
	if(J < 0.5f)
	{
		const PxReal regularization =
			(0.5f - J) * lam * V0 * m2;
		outHessian.column0.x += regularization;
		outHessian.column1.y += regularization;
		outHessian.column2.z += regularization;
	}
}

PX_FORCE_INLINE void avbdEvaluateNeoHookeanForceHessian(
	const AvbdTetElement& tet, int vOrder,
	PxReal mu, PxReal lam,
	const AvbdSoftParticle* particles,
	PxVec3& outForce, PxMat33& outHessian)
{
	const PxReal lambdaSafe =
		PxAbs(lam) < 1e-6f ? 1e-6f : lam;
	avbdEvaluateNeoHookeanForceHessianPrepared(
		tet, vOrder, mu, lam, 1.0f + mu / lambdaSafe,
		particles, outForce, outHessian);
}

enum class AvbdSoftTetDisplacementLimitReason : PxU8
{
	eNONE,
	ePOSITIVE_J_LIMITED,
	ePOSITIVE_J_REJECTED,
	eNONFINITE_REJECTED
};

struct AvbdSoftTetDisplacementLimitResult
{
	PxVec3 appliedDisplacement;
	PxReal fraction;
	AvbdSoftTetDisplacementLimitReason reason;

	AvbdSoftTetDisplacementLimitResult()
		: appliedDisplacement(0.0f), fraction(0.0f),
		  reason(AvbdSoftTetDisplacementLimitReason::eNONFINITE_REJECTED)
	{
	}

	AvbdSoftTetDisplacementLimitResult(
		const PxVec3& displacement, PxReal appliedFraction,
		AvbdSoftTetDisplacementLimitReason limitReason)
		: appliedDisplacement(displacement),
		  fraction(appliedFraction), reason(limitReason)
	{
	}
};

// Observe the positive-J limiter separately from the local block solve.
// A rejected displacement is a feasibility signal, not a stationarity
// certificate: the applied displacement may be zero while H^-1 f is not.
struct AvbdSoftSweepConvergenceObservation
{
	PxReal maxLocalSolveDisplacementSq;
	PxReal maxAppliedDisplacementSq;
	PxU32 trustRegionLimitedSteps;
	PxU32 positiveJLimitedSteps;
	PxU32 positiveJRejectedSteps;
	PxU32 nonFiniteRejectedSteps;

	AvbdSoftSweepConvergenceObservation()
		: maxLocalSolveDisplacementSq(0.0f),
		  maxAppliedDisplacementSq(0.0f),
		  trustRegionLimitedSteps(0), positiveJLimitedSteps(0),
		  positiveJRejectedSteps(0), nonFiniteRejectedSteps(0)
	{
	}

	PX_FORCE_INLINE void observe(
		const PxVec3& localSolveDisplacement,
		bool trustRegionLimited,
		const AvbdSoftTetDisplacementLimitResult& limitResult)
	{
		const PxReal localSolveDisplacementSq =
			localSolveDisplacement.magnitudeSquared();
		if(localSolveDisplacement.isFinite() &&
			PxIsFinite(localSolveDisplacementSq))
		{
			maxLocalSolveDisplacementSq = PxMax(
				maxLocalSolveDisplacementSq,
				localSolveDisplacementSq);
		}
		else
			nonFiniteRejectedSteps++;

		const PxReal appliedDisplacementSq =
			limitResult.appliedDisplacement.magnitudeSquared();
		if(limitResult.appliedDisplacement.isFinite() &&
			PxIsFinite(appliedDisplacementSq))
		{
			maxAppliedDisplacementSq = PxMax(
				maxAppliedDisplacementSq,
				appliedDisplacementSq);
		}
		else
			nonFiniteRejectedSteps++;

		if(trustRegionLimited)
			trustRegionLimitedSteps++;

		switch(limitResult.reason)
		{
		case AvbdSoftTetDisplacementLimitReason::ePOSITIVE_J_LIMITED:
			positiveJLimitedSteps++;
			break;
		case AvbdSoftTetDisplacementLimitReason::ePOSITIVE_J_REJECTED:
			positiveJRejectedSteps++;
			break;
		case AvbdSoftTetDisplacementLimitReason::eNONFINITE_REJECTED:
			if(localSolveDisplacement.isFinite() &&
				PxIsFinite(localSolveDisplacementSq))
				nonFiniteRejectedSteps++;
			break;
		case AvbdSoftTetDisplacementLimitReason::eNONE:
			break;
		}
	}

	// P4.5 range tasks retain one observation each and the parent merges them
	// in a fixed causal-layer/range order.  This is deliberately a plain,
	// non-atomic reduction: the current serial sweep uses the same operation
	// and the future task route must never write this shared object directly.
	PX_FORCE_INLINE void merge(
		const AvbdSoftSweepConvergenceObservation& other)
	{
		maxLocalSolveDisplacementSq = PxMax(
			maxLocalSolveDisplacementSq,
			other.maxLocalSolveDisplacementSq);
		maxAppliedDisplacementSq = PxMax(
			maxAppliedDisplacementSq,
			other.maxAppliedDisplacementSq);
		trustRegionLimitedSteps += other.trustRegionLimitedSteps;
		positiveJLimitedSteps += other.positiveJLimitedSteps;
		positiveJRejectedSteps += other.positiveJRejectedSteps;
		nonFiniteRejectedSteps += other.nonFiniteRejectedSteps;
	}

	PX_FORCE_INLINE bool isAppliedDisplacementConverged(
		PxReal toleranceSq) const
	{
		return maxAppliedDisplacementSq < toleranceSq;
	}

	PX_FORCE_INLINE bool isResidualConverged(PxReal toleranceSq) const
	{
		return maxLocalSolveDisplacementSq < toleranceSq &&
			trustRegionLimitedSteps == 0 &&
			positiveJLimitedSteps == 0 &&
			positiveJRejectedSteps == 0 &&
			nonFiniteRejectedSteps == 0;
	}
};

struct AvbdSoftResidualConvergenceTracker
{
	PxReal toleranceSq;
	PxU32 requiredConsecutiveSweeps;
	PxU32 consecutiveSweeps;

	AvbdSoftResidualConvergenceTracker(
		PxReal solveToleranceSq, PxU32 requiredSweeps)
		: toleranceSq(solveToleranceSq),
		  requiredConsecutiveSweeps(PxMax(requiredSweeps, 1u)),
		  consecutiveSweeps(0)
	{
	}

	PX_FORCE_INLINE bool observe(
		const AvbdSoftSweepConvergenceObservation& observation)
	{
		consecutiveSweeps = observation.isResidualConverged(toleranceSq)
			? consecutiveSweeps + 1
			: 0;
		return consecutiveSweeps >= requiredConsecutiveSweeps;
	}
};

PX_FORCE_INLINE AvbdSoftTetDisplacementLimitResult
avbdLimitTetDisplacementFromLinearizations(
	const PxVec3& displacement,
	const AvbdTetVertexLinearization* linearizations,
	PxU32 linearizationCount, PxReal minDetF = 0.05f)
{
	if(!displacement.isFinite())
	{
		return AvbdSoftTetDisplacementLimitResult(
			PxVec3(0.0f), 0.0f,
			AvbdSoftTetDisplacementLimitReason::eNONFINITE_REJECTED);
	}

	PxReal fraction = 1.0f;
	for(PxU32 linearizationId = 0;
		linearizationId < linearizationCount; linearizationId++)
	{
		const AvbdTetVertexLinearization& linearization =
			linearizations[linearizationId];
		const PxReal currentDetF = linearization.determinant;
		const PxReal proposedDetF =
			currentDetF +
			linearization.determinantGradient.dot(displacement);
		if(!PxIsFinite(currentDetF) || !PxIsFinite(proposedDetF))
		{
			return AvbdSoftTetDisplacementLimitResult(
				PxVec3(0.0f), 0.0f,
				AvbdSoftTetDisplacementLimitReason::
					eNONFINITE_REJECTED);
		}
		if(proposedDetF >= minDetF ||
			proposedDetF >= currentDetF)
			continue;
		if(currentDetF <= minDetF)
		{
			return AvbdSoftTetDisplacementLimitResult(
				PxVec3(0.0f), 0.0f,
				AvbdSoftTetDisplacementLimitReason::
					ePOSITIVE_J_REJECTED);
		}
		const PxReal admissible =
			(currentDetF - minDetF) /
			(currentDetF - proposedDetF);
		fraction = PxMin(
			fraction,
			PxMax(0.0f, admissible * 0.99f));
	}
	return AvbdSoftTetDisplacementLimitResult(
		displacement * fraction, fraction,
		fraction < 1.0f
			? AvbdSoftTetDisplacementLimitReason::
				ePOSITIVE_J_LIMITED
			: AvbdSoftTetDisplacementLimitReason::eNONE);
}

// Limit a single-particle displacement so no incident tetrahedron is
// pushed through the same positive-J floor used by the Neo-Hookean model.
// For one moving vertex, det(F) is affine in the displacement, so the
// admissible fraction is available analytically without a global line search.
PX_FORCE_INLINE AvbdSoftTetDisplacementLimitResult
avbdLimitTetDisplacementObserved(
	const AvbdSoftBody& body, PxU32 particleIdx,
	const AvbdSoftParticle* particles, const PxVec3& displacement,
	PxReal minDetF = 0.05f)
{
	if(!displacement.isFinite())
	{
		return AvbdSoftTetDisplacementLimitResult(
			PxVec3(0.0f), 0.0f,
			AvbdSoftTetDisplacementLimitReason::eNONFINITE_REJECTED);
	}
	if(particleIdx < body.compiled.particleStart ||
		particleIdx >= body.compiled.particleStart + body.compiled.particleCount)
	{
		return AvbdSoftTetDisplacementLimitResult(
			displacement, 1.0f,
			AvbdSoftTetDisplacementLimitReason::eNONE);
	}

	const PxU32 localIdx = particleIdx - body.compiled.particleStart;
	const AvbdParticleElementAdjacency& adjacency =
		body.compiled.elementAdjacency[localIdx];
	PxReal fraction = 1.0f;
	for(PxU32 refId = 0; refId < adjacency.tetRefs.size(); ++refId)
	{
		const AvbdParticleElementRef& ref =
			adjacency.tetRefs[refId];
		const AvbdTetElement& tet =
			body.compiled.tetElements[ref.index];
		const PxVec3 current0 = particles[tet.p0].position;
		const PxVec3 e1 =
			particles[tet.p1].position - current0;
		const PxVec3 e2 =
			particles[tet.p2].position - current0;
		const PxVec3 e3 =
			particles[tet.p3].position - current0;
		PxReal currentDetF;
		PxVec3 determinantGradient;
		avbdEvaluateTetDeterminantAndGradient(
			tet, ref.vOrder, e1, e2, e3,
			currentDetF, determinantGradient);
		const PxReal proposedDetF =
			currentDetF + determinantGradient.dot(displacement);

		if(!PxIsFinite(currentDetF) || !PxIsFinite(proposedDetF))
		{
			return AvbdSoftTetDisplacementLimitResult(
				PxVec3(0.0f), 0.0f,
				AvbdSoftTetDisplacementLimitReason::
					eNONFINITE_REJECTED);
		}
		if(proposedDetF >= minDetF || proposedDetF >= currentDetF)
			continue;
		if(currentDetF <= minDetF)
		{
			return AvbdSoftTetDisplacementLimitResult(
				PxVec3(0.0f), 0.0f,
				AvbdSoftTetDisplacementLimitReason::
					ePOSITIVE_J_REJECTED);
		}
		const PxReal admissible =
			(currentDetF - minDetF) /
			(currentDetF - proposedDetF);
		fraction = PxMin(fraction, PxMax(0.0f, admissible * 0.99f));
	}
	const AvbdSoftTetDisplacementLimitReason reason =
		fraction < 1.0f
			? AvbdSoftTetDisplacementLimitReason::ePOSITIVE_J_LIMITED
			: AvbdSoftTetDisplacementLimitReason::eNONE;
	return AvbdSoftTetDisplacementLimitResult(
		displacement * fraction, fraction, reason);
}

PX_FORCE_INLINE PxVec3 avbdLimitTetDisplacement(
	const AvbdSoftBody& body, PxU32 particleIdx,
	const AvbdSoftParticle* particles, const PxVec3& displacement,
	PxReal minDetF = 0.05f)
{
	return avbdLimitTetDisplacementObserved(
		body, particleIdx, particles, displacement, minDetF).
			appliedDisplacement;
}

PX_FORCE_INLINE void avbdEvaluateBendingForceHessian(
	const AvbdBendingElement& be, int vOrder,
	PxReal stiffness,
	const AvbdSoftParticle* particles,
	PxVec3& outForce, PxMat33& outHessian)
{
	const PxReal eps = 1e-6f;

	PxVec3 x0 = particles[be.opp0].position;
	PxVec3 x1 = particles[be.opp1].position;
	PxVec3 x2 = particles[be.edgeStart].position;
	PxVec3 x3 = particles[be.edgeEnd].position;

	PxVec3 e = x3 - x2;
	PxVec3 x02 = x2 - x0, x03 = x3 - x0;
	PxVec3 x13 = x3 - x1, x12 = x2 - x1;

	PxVec3 n1 = x02.cross(x03);
	PxVec3 n2 = x13.cross(x12);

	PxReal n1Norm = n1.magnitude();
	PxReal n2Norm = n2.magnitude();
	PxReal eNorm = e.magnitude();

	if (n1Norm < eps || n2Norm < eps || eNorm < eps)
	{
		outForce = PxVec3(0.0f);
		outHessian = PxMat33(PxZero);
		return;
	}

	PxVec3 n1Hat = n1 * (1.0f / n1Norm);
	PxVec3 n2Hat = n2 * (1.0f / n2Norm);
	PxVec3 eHat = e * (1.0f / eNorm);

	PxReal sinTheta = n1Hat.cross(n2Hat).dot(eHat);
	PxReal cosTheta = PxClamp(n1Hat.dot(n2Hat), -1.0f, 1.0f);
	PxReal theta = PxAtan2(sinTheta, cosTheta);

	PxReal k = stiffness * be.restLength;
	PxReal dE_dtheta = k * (theta - be.restAngle);

	auto normalizedDerivative = [](PxReal unnormLen, const PxVec3& nHat,
	                                const PxMat33& dNdx) -> PxMat33 {
		PxMat33 P = PxMat33(PxIdentity) - avbdOuter(nHat, nHat);
		return (P * dNdx) * (1.0f / unnormLen);
	};

	auto angleDerivative = [](const PxVec3& n1h, const PxVec3& n2h, const PxVec3& eh,
	                          const PxMat33& dn1dx, const PxMat33& dn2dx,
	                          PxReal sinT, PxReal cosT,
	                          const PxMat33& skN1, const PxMat33& skN2) -> PxVec3 {
		PxMat33 dSinMat = skN1 * dn2dx - skN2 * dn1dx;
		PxVec3 dSin = dSinMat.getTranspose() * eh;
		PxVec3 dCos = dn1dx.getTranspose() * n2h + dn2dx.getTranspose() * n1h;
		return dSin * cosT - dCos * sinT;
	};

	PxMat33 skE = avbdSkew(e);
	PxMat33 skX03 = avbdSkew(x03);
	PxMat33 skX02 = avbdSkew(x02);
	PxMat33 skX13 = avbdSkew(x13);
	PxMat33 skX12 = avbdSkew(x12);
	PxMat33 skN1 = avbdSkew(n1Hat);
	PxMat33 skN2 = avbdSkew(n2Hat);

	PxMat33 dn1hat_dx0 = normalizedDerivative(n1Norm, n1Hat, skE);
	PxMat33 dn1hat_dx1(PxZero);
	PxMat33 dn1hat_dx2 = normalizedDerivative(n1Norm, n1Hat, skX03 * (-1.0f));
	PxMat33 dn1hat_dx3 = normalizedDerivative(n1Norm, n1Hat, skX02);

	PxMat33 dn2hat_dx0(PxZero);
	PxMat33 dn2hat_dx1 = normalizedDerivative(n2Norm, n2Hat, skE * (-1.0f));
	PxMat33 dn2hat_dx2 = normalizedDerivative(n2Norm, n2Hat, skX13);
	PxMat33 dn2hat_dx3 = normalizedDerivative(n2Norm, n2Hat, skX12 * (-1.0f));

	PxVec3 dtheta_dx0 = angleDerivative(n1Hat, n2Hat, eHat, dn1hat_dx0, dn2hat_dx0,
	                                     sinTheta, cosTheta, skN1, skN2);
	PxVec3 dtheta_dx1 = angleDerivative(n1Hat, n2Hat, eHat, dn1hat_dx1, dn2hat_dx1,
	                                     sinTheta, cosTheta, skN1, skN2);
	PxVec3 dtheta_dx2 = angleDerivative(n1Hat, n2Hat, eHat, dn1hat_dx2, dn2hat_dx2,
	                                     sinTheta, cosTheta, skN1, skN2);
	PxVec3 dtheta_dx3 = angleDerivative(n1Hat, n2Hat, eHat, dn1hat_dx3, dn2hat_dx3,
	                                     sinTheta, cosTheta, skN1, skN2);

	PxVec3 dtheta_dx;
	switch (vOrder)
	{
		case 0: dtheta_dx = dtheta_dx0; break;
		case 1: dtheta_dx = dtheta_dx1; break;
		case 2: dtheta_dx = dtheta_dx2; break;
		case 3: dtheta_dx = dtheta_dx3; break;
		default:
			outForce = PxVec3(0.0f);
			outHessian = PxMat33(PxZero);
			return;
	}

	outForce = dtheta_dx * (-dE_dtheta);
	outHessian = avbdOuter(dtheta_dx, dtheta_dx) * k;
}

// Bending damping is a once-per-step post-solve stage.  Keep its body/hinge
// traversal out of the particle primal solver's instruction footprint.
PX_NOINLINE inline void avbdApplyBendingDamping(
	AvbdSoftParticle* particles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxReal dt)
{
	if(!particles || !softBodies || dt <= 0.0f)
		return;
	for(PxU32 bodyIndex = 0;
		bodyIndex < numSoftBodies; bodyIndex++)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		const PxReal dampingFactor = PxClamp(
			body.material.bendingDamping * dt,
			0.0f, 1.0f);
		if(dampingFactor <= 0.0f ||
			body.material.bendingStiffness <= 0.0f ||
			body.compiled.bendElements.empty())
			continue;
		PxArray<PxVec3> deltaVelocities(
			body.compiled.particleCount);
		for(PxU32 localIndex = 0;
			localIndex < deltaVelocities.size(); localIndex++)
			deltaVelocities[localIndex] = PxVec3(0.0f);

		for(PxU32 bendingIndex = 0;
			bendingIndex < body.compiled.bendElements.size();
			bendingIndex++)
		{
			const AvbdBendingElement& bending =
				body.compiled.bendElements[bendingIndex];
			const PxU32 edgeStart = bending.edgeStart;
			const PxU32 edgeEnd = bending.edgeEnd;
			const PxU32 tip0 = bending.opp0;
			const PxU32 tip1 = bending.opp1;
			const PxU32 bodyEnd =
				body.compiled.particleStart +
				body.compiled.particleCount;
			if(edgeStart < body.compiled.particleStart ||
				edgeEnd < body.compiled.particleStart ||
				tip0 < body.compiled.particleStart ||
				tip1 < body.compiled.particleStart ||
				edgeStart >= bodyEnd || edgeEnd >= bodyEnd ||
				tip0 >= bodyEnd || tip1 >= bodyEnd)
				continue;

			const PxVec3 linearVelocity =
				(particles[edgeStart].velocity +
				 particles[edgeEnd].velocity) * 0.5f;
			PxVec3 edgeDirection =
				particles[edgeEnd].position -
				particles[edgeStart].position;
			if(edgeDirection.normalize() < 1.0e-6f)
				continue;
			PxVec3 tipDirection0 =
				edgeDirection.cross(
					particles[tip0].position -
					particles[edgeStart].position);
			PxVec3 tipDirection1 =
				edgeDirection.cross(
					particles[tip1].position -
					particles[edgeStart].position);
			const PxReal tipDistance0 = tipDirection0.normalize();
			const PxReal tipDistance1 = tipDirection1.normalize();
			if(tipDistance0 < 1.0e-6f ||
				tipDistance1 < 1.0e-6f)
				continue;
			const PxReal angularVelocity0 =
				tipDirection0.dot(
					particles[tip0].velocity -
					linearVelocity) /
				tipDistance0;
			const PxReal angularVelocity1 =
				tipDirection1.dot(
					particles[tip1].velocity -
					linearVelocity) /
				tipDistance1;
			const PxReal dampedAngularDifference =
				(angularVelocity1 - angularVelocity0) *
				dampingFactor;
			PxVec3 deltaEdgeStart(0.0f);
			PxVec3 deltaEdgeEnd(0.0f);
			PxVec3 deltaTip0 =
				tipDirection0 *
				(dampedAngularDifference * tipDistance0);
			PxVec3 deltaTip1 =
				tipDirection1 *
				(-dampedAngularDifference * tipDistance1);
			const PxReal inverseMassSum =
				particles[edgeStart].invMass +
				particles[edgeEnd].invMass +
				particles[tip0].invMass +
				particles[tip1].invMass;
			if(inverseMassSum <= 1.0e-12f)
				continue;
			const PxVec3 averageDelta =
				(deltaEdgeStart + deltaEdgeEnd +
				 deltaTip0 + deltaTip1) * 0.25f;
			deltaEdgeStart -= averageDelta;
			deltaEdgeEnd -= averageDelta;
			deltaTip0 -= averageDelta;
			deltaTip1 -= averageDelta;
			const PxReal weightFactor =
				1.0f / inverseMassSum;
			deltaVelocities[
				edgeStart - body.compiled.particleStart] +=
				deltaEdgeStart *
				(particles[edgeStart].invMass * weightFactor);
			deltaVelocities[
				edgeEnd - body.compiled.particleStart] +=
				deltaEdgeEnd *
				(particles[edgeEnd].invMass * weightFactor);
			deltaVelocities[
				tip0 - body.compiled.particleStart] +=
				deltaTip0 *
				(particles[tip0].invMass * weightFactor);
			deltaVelocities[
				tip1 - body.compiled.particleStart] +=
				deltaTip1 *
				(particles[tip1].invMass * weightFactor);
		}
		for(PxU32 localIndex = 0;
			localIndex < body.compiled.particleCount;
			localIndex++)
		{
			AvbdSoftParticle& particle =
				particles[
					body.compiled.particleStart + localIndex];
			if(particle.invMass > 0.0f)
				particle.velocity +=
					deltaVelocities[localIndex];
		}
	}
}

// =============================================================================
// AVBD contact/pin evaluators
// =============================================================================

struct AvbdSoftContactRowForces
{
	PxReal normal;
	PxReal tangent[2];
	bool tangentClamped;

	AvbdSoftContactRowForces()
		: normal(0.0f), tangent{0.0f, 0.0f}, tangentClamped(false)
	{
	}
};

PX_FORCE_INLINE AvbdSoftContactRowForces
avbdEvaluateSoftContactRowForces(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftContactAugmentedState& state,
	const AvbdSoftParticle* particles,
	const PxVec3& currentSurfacePoint)
{
	AvbdSoftContactRowForces forces;
	const PxVec3 queryPoint =
		avbdGetSoftContactQueryPoint(geometry, particles);
	const PxVec3 n = geometry.normal;
	const PxReal normalConstraint =
		avbdEvaluateSoftContactNormalConstraint(
			geometry, queryPoint, currentSurfacePoint) -
		state.depenetrationConstraintOffset;
	forces.normal =
		PxMin(
			0.0f,
			state.k * normalConstraint +
				state.alLambda);

	if(geometry.tangentOwner !=
			AvbdSoftContactTangentOwner::ePOSITION_AL ||
		geometry.friction <= 0.0f || forces.normal >= 0.0f)
		return forces;

	const PxVec3 relativeDisplacement =
		(queryPoint - state.particlePointPrev) -
		(currentSurfacePoint - state.surfacePointPrev);
	const PxReal tangentConstraint[2] =
	{
		relativeDisplacement.dot(geometry.tangent1),
		relativeDisplacement.dot(geometry.tangent2)
	};
	forces.tangent[0] =
		state.penTangent[0] * tangentConstraint[0] +
			state.alLambdaTangent[0];
	forces.tangent[1] =
		state.penTangent[1] * tangentConstraint[1] +
			state.alLambdaTangent[1];
	const PxReal frictionBound =
		geometry.friction * PxAbs(forces.normal);
	const PxReal tangentMagnitude = PxSqrt(
		forces.tangent[0] * forces.tangent[0] +
		forces.tangent[1] * forces.tangent[1]);
	if(tangentMagnitude > frictionBound && tangentMagnitude > 1e-12f)
	{
		const PxReal scale = frictionBound / tangentMagnitude;
		forces.tangent[0] *= scale;
		forces.tangent[1] *= scale;
		forces.tangentClamped = true;
	}
	return forces;
}

PX_FORCE_INLINE void avbdEvaluateContactParticleBlockAtSurfacePoint(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftContactAugmentedState& state,
	const AvbdSoftParticle* particles,
	const PxVec3& currentSurfacePoint,
	PxReal jacobianScale,
	PxVec3& outForce, PxMat33& outHessian)
{
	outForce = PxVec3(0.0f);
	outHessian = PxMat33(PxZero);
	if(PxAbs(jacobianScale) <= 1e-12f)
		return;

	const PxVec3 n = geometry.normal;
	const AvbdSoftContactRowForces rowForces =
		avbdEvaluateSoftContactRowForces(
			geometry, state, particles, currentSurfacePoint);
	outForce = n * (-jacobianScale * rowForces.normal);
	outHessian =
		avbdOuter(n, n) *
		(state.k * jacobianScale * jacobianScale);

	if(geometry.tangentOwner !=
			AvbdSoftContactTangentOwner::ePOSITION_AL ||
		geometry.friction <= 0.0f || rowForces.normal >= 0.0f)
		return;

	outForce -=
		(geometry.tangent1 * rowForces.tangent[0] +
		 geometry.tangent2 * rowForces.tangent[1]) * jacobianScale;
	// Once the trial tangent force is projected onto the Coulomb cone the
	// contact is sliding. Keeping the unprojected penalty Hessian here makes
	// Newton see a sticking spring even though the force itself is capped; a
	// dense edge manifold can then numerically pin two visibly sliding bodies.
	// Use a lagged Coulomb force for the sliding row. Inertia and material
	// curvature still regularize the particle block, while a row inside the
	// cone retains the full static-friction curvature below.
	if(!rowForces.tangentClamped)
	{
		outHessian = outHessian +
			avbdOuter(geometry.tangent1, geometry.tangent1) *
				(state.penTangent[0] *
				 jacobianScale * jacobianScale) +
			avbdOuter(geometry.tangent2, geometry.tangent2) *
				(state.penTangent[1] *
				 jacobianScale * jacobianScale);
	}
}
PX_FORCE_INLINE void avbdEvaluateContactParticleBlock(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftContactAugmentedState& state,
	const AvbdSoftParticle* particles,
	PxReal jacobianScale,
	PxVec3& outForce, PxMat33& outHessian)
{
	avbdEvaluateContactParticleBlockAtSurfacePoint(
		geometry, state, particles,
		avbdGetSoftContactSurfacePoint(geometry, particles),
		jacobianScale, outForce, outHessian);
}

PX_FORCE_INLINE void avbdEvaluateContactForceHessian(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftContactAugmentedState& state,
	const AvbdSoftParticle* particles,
	PxVec3& outForce, PxMat33& outHessian)
{
	avbdEvaluateContactParticleBlock(
		geometry, state, particles, 1.0f,
		outForce, outHessian);
}

PX_FORCE_INLINE void avbdEvaluatePinForceHessian(
	const AvbdSoftPoint& point,
	const AvbdKinematicPin& kp,
	const AvbdSoftParticle* particles,
	PxU32 particleIndex,
	PxVec3& outForce, PxMat33& outHessian)
{
	const PxReal jacobianWeight =
		avbdGetSoftPointJacobianWeight(point, particleIndex);
	const PxVec3 C =
		avbdGetSoftPointPosition(point, particles) - kp.worldTarget;
	outForce = -(C * kp.k + kp.alLambda) * jacobianWeight;
	outHessian = PxMat33::createDiagonal(
		PxVec3(kp.k * jacobianWeight * jacobianWeight));
}

// =============================================================================
// AVBD Dual updates
// =============================================================================

PX_FORCE_INLINE void avbdWarmstartAttachmentState(
	AvbdSoftAttachment& attachment,
	PxReal alpha, PxReal gamma, PxReal penaltyMin)
{
	attachment.alLambda *= alpha * gamma;
	attachment.k = PxMax(
		penaltyMin,
		PxMin(attachment.kMax, attachment.k * gamma));
}

PX_FORCE_INLINE void avbdWarmstartPinState(
	AvbdKinematicPin& kp,
	PxReal alpha, PxReal gamma, PxReal penaltyMin)
{
	kp.alLambda *= alpha * gamma;
	kp.k = PxMax(penaltyMin, PxMin(kp.kMax, kp.k * gamma));
}

PX_FORCE_INLINE void avbdUpdatePinDual(
	AvbdKinematicPin& kp,
	const AvbdSoftPoint& point,
	const AvbdSoftParticle* particles,
	PxReal beta)
{
	const PxVec3 C =
		avbdGetSoftPointPosition(point, particles) - kp.worldTarget;
	kp.alLambda += C * kp.k;
	const PxReal C_lin = C.magnitude();
	kp.k = PxMin(kp.k + beta * C_lin, kp.kMax);
}

PX_FORCE_INLINE void avbdUpdateSoftPairAttachmentDual(
	AvbdSoftAttachment& attachment,
	const AvbdSoftPoint& point,
	const AvbdSoftPoint& targetPoint,
	const AvbdSoftParticle* particles,
	PxReal beta)
{
	const PxVec3 constraint =
		avbdGetSoftPointPosition(point, particles) -
		avbdGetSoftPointPosition(targetPoint, particles);
	attachment.alLambda += constraint * attachment.k;
	attachment.k = PxMin(
		attachment.k + beta * constraint.magnitude(),
		attachment.kMax);
}

PX_FORCE_INLINE void avbdUpdateSoftContactDualAtSurfacePoint(
	const AvbdSoftContactGeometry& geometry,
	AvbdSoftContactAugmentedState& state,
	const AvbdSoftParticle* particles,
	const PxVec3& currentSurfacePoint,
	PxReal beta)
{
	PxVec3 n = geometry.normal;
	const PxVec3 queryPoint =
		avbdGetSoftContactQueryPoint(geometry, particles);
	const PxReal normalConstraint =
		avbdEvaluateSoftContactNormalConstraint(
			geometry, queryPoint, currentSurfacePoint) -
		state.depenetrationConstraintOffset;

	state.alLambda =
		PxMin(
			0.0f,
			state.k * normalConstraint +
				state.alLambda);
	if(state.alLambda < 0.0f)
		state.k = PxMin(
			state.k + beta * PxAbs(normalConstraint),
			state.ke);

	if(geometry.tangentOwner !=
		AvbdSoftContactTangentOwner::ePOSITION_AL)
	{
		avbdResetSoftContactTangentState(geometry, state, particles);
		return;
	}

	const PxVec3 relativeDisplacement =
		(queryPoint - state.particlePointPrev) -
		(currentSurfacePoint - state.surfacePointPrev);
	const PxReal tangentConstraint[2] =
	{
		relativeDisplacement.dot(geometry.tangent1),
		relativeDisplacement.dot(geometry.tangent2)
	};
	const PxReal frictionBound =
		geometry.friction * PxAbs(state.alLambda);
	PxReal tangentForce[2] =
	{
		state.penTangent[0] * tangentConstraint[0] +
			state.alLambdaTangent[0],
		state.penTangent[1] * tangentConstraint[1] +
			state.alLambdaTangent[1]
	};
	const PxReal rawTangentMagnitude = PxSqrt(
		tangentForce[0] * tangentForce[0] +
		tangentForce[1] * tangentForce[1]);
	const bool insideFrictionCone =
		rawTangentMagnitude <= frictionBound;
	if(!insideFrictionCone && rawTangentMagnitude > 1e-12f)
	{
		const PxReal scale = frictionBound / rawTangentMagnitude;
		tangentForce[0] *= scale;
		tangentForce[1] *= scale;
	}
	state.alLambdaTangent[0] = tangentForce[0];
	state.alLambdaTangent[1] = tangentForce[1];
	if(insideFrictionCone)
	{
		state.penTangent[0] = PxMin(
			state.penTangent[0] +
				beta * PxAbs(tangentConstraint[0]), state.ke);
		state.penTangent[1] = PxMin(
			state.penTangent[1] +
				beta * PxAbs(tangentConstraint[1]), state.ke);
	}
	state.frictionStick =
		insideFrictionCone &&
		tangentConstraint[0] * tangentConstraint[0] +
		tangentConstraint[1] * tangentConstraint[1] < 1e-10f;
}

enum class AvbdSoftComponentFinalizeMode : PxU8
{
	eMOMENTUM,
	eKINEMATIC_CONTACT,
	ePOSITION_OWNED,
	eUNSUPPORTED
};

struct AvbdSoftComponentMomentumTarget
{
	PxVec3 centroid;
	PxVec3 linearMomentum;
	PxVec3 angularMomentum;
	PxReal mass;
	bool valid;

	AvbdSoftComponentMomentumTarget()
		: centroid(0.0f), linearMomentum(0.0f), angularMomentum(0.0f),
		  mass(0.0f), valid(false)
	{
	}
};

// Minimal prep-time IR for a velocity objective.  It intentionally excludes
// contact dual/penalty state so a finalizer cannot accidentally re-enter the
// position AL program.
struct AvbdCompiledSoftVelocityObjective
{
	AvbdVelocityObjectiveOwner owner;
	AvbdSoftContactSource source;
	PxU32 bodyIndex;
	PxU32 particleIndex;
	AvbdWeightedContactPoint queryPoint;
	PxVec3 normal;
	PxVec3 surfacePoint;
	PxVec3 previousSurfacePoint;

	AvbdCompiledSoftVelocityObjective()
		: owner(AvbdVelocityObjectiveOwner::Unsupported),
		  source(), bodyIndex(PX_MAX_U32),
		  particleIndex(PX_MAX_U32), queryPoint(),
		  normal(0.0f, 1.0f, 0.0f),
		  surfacePoint(0.0f), previousSurfacePoint(0.0f)
	{
	}
};

PX_FORCE_INLINE bool avbdComputeSoftComponentMomentum(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody& body, bool usePrediction, PxReal invDt,
	PxVec3& centroid, PxVec3& linearMomentum,
	PxVec3& angularMomentum, PxMat33& inertia, PxReal& mass)
{
	centroid = PxVec3(0.0f);
	linearMomentum = PxVec3(0.0f);
	angularMomentum = PxVec3(0.0f);
	inertia = PxMat33::createDiagonal(PxVec3(0.0f));
	mass = 0.0f;
	const PxU32 particleStart = body.compiled.particleStart;
	const PxU32 particleCount = body.compiled.particleCount;
	if(particleStart > numParticles ||
		particleCount > numParticles - particleStart)
		return false;
	for(PxU32 localIndex = 0; localIndex < particleCount; localIndex++)
	{
		const AvbdSoftParticle& particle =
			particles[particleStart + localIndex];
		if(particle.invMass <= 0.0f || particle.mass <= 0.0f)
			continue;
		const PxVec3 position =
			usePrediction ? particle.initialPosition : particle.position;
		const PxVec3 velocity = usePrediction
			? (particle.predictedPosition -
			   particle.initialPosition) * invDt
			: particle.velocity;
		if(!position.isFinite() || !velocity.isFinite())
			return false;
		centroid += position * particle.mass;
		linearMomentum += velocity * particle.mass;
		mass += particle.mass;
	}
	if(mass <= 0.0f)
		return false;
	centroid *= 1.0f / mass;
	for(PxU32 localIndex = 0; localIndex < particleCount; localIndex++)
	{
		const AvbdSoftParticle& particle =
			particles[particleStart + localIndex];
		if(particle.invMass <= 0.0f || particle.mass <= 0.0f)
			continue;
		const PxVec3 position =
			usePrediction ? particle.initialPosition : particle.position;
		const PxVec3 velocity = usePrediction
			? (particle.predictedPosition -
			   particle.initialPosition) * invDt
			: particle.velocity;
		const PxVec3 offset = position - centroid;
		inertia = inertia +
			(PxMat33::createDiagonal(
				PxVec3(offset.magnitudeSquared())) -
			 avbdOuter(offset, offset)) * particle.mass;
		angularMomentum +=
			offset.cross(velocity) * particle.mass;
	}
	return linearMomentum.isFinite() &&
		angularMomentum.isFinite() &&
		PxIsFinite(inertia.getDeterminant());
}

PX_FORCE_INLINE void avbdApplySoftComponentDampingToMomentumTarget(
	AvbdSoftComponentMomentumTarget& target,
	const AvbdSoftBody& body, PxReal dt)
{
	if(!target.valid)
		return;
	// The particle solve damps deformation modes, but the component finalizer
	// subsequently restores the predicted rigid linear/angular momentum. Apply
	// the same timestep damping to that authoritative target so contact
	// ownership cannot resurrect an undamped rigid mode every frame (most
	// visibly as runaway rolling).
	const PxReal dampingScale = 1.0f /
		(1.0f + PxMax(body.material.damping, 0.0f) * dt);
	target.linearMomentum *= dampingScale;
	target.angularMomentum *= dampingScale;
}

// This is a once-per-step component finalization stage with several distinct
// contact-owner policies.  Keeping it out of the particle primal solve's code
// body limits instruction-cache and stack pressure without adding a call to
// the per-particle/per-sweep hot loop.
PX_NOINLINE inline void avbdFinalizeSoftComponentVelocities(
	AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftComponentMomentumTarget* momentumTargets,
	const AvbdSoftComponentFinalizeMode* finalizeModes,
	const AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdCompiledSoftVelocityObjective* velocityObjectives,
	PxU32 numVelocityObjectives, PxReal invDt)
{
	if(!particles || !softBodies || !momentumTargets ||
		!finalizeModes || invDt <= 0.0f)
		return;
	bool hasSpeculativeCcdBody = false;
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
		hasSpeculativeCcdBody = hasSpeculativeCcdBody ||
			softBodies[bodyIndex].compiled.speculativeCCDEnabled;
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		const AvbdSoftComponentMomentumTarget& target =
			momentumTargets[bodyIndex];
		const AvbdSoftComponentFinalizeMode mode =
			finalizeModes[bodyIndex];
		// Position-AL contacts already own local non-penetration in the particle
		// solve. Recasting their multipliers as one rigid component impulse
		// distributes a local impact across the whole deformable and injects
		// spurious translation/rotation at first contact.
		if(!target.valid ||
			mode == AvbdSoftComponentFinalizeMode::ePOSITION_OWNED ||
			mode == AvbdSoftComponentFinalizeMode::eUNSUPPORTED)
			continue;
		const AvbdSoftBody& body = softBodies[bodyIndex];
		PxVec3 centroid(0.0f);
		PxVec3 actualLinearMomentum(0.0f);
		PxVec3 actualAngularMomentum(0.0f);
		PxMat33 inertia(PxZero);
		PxReal mass = 0.0f;
		if(!avbdComputeSoftComponentMomentum(
				particles, numParticles, body, false, invDt,
				centroid, actualLinearMomentum,
				actualAngularMomentum, inertia, mass))
			continue;
		if(PxAbs(mass - target.mass) >
			PxMax(1.0e-5f, target.mass * 1.0e-5f))
			continue;

		const PxReal inertiaDeterminant = inertia.getDeterminant();
		const bool hasAngularResponse =
			PxIsFinite(inertiaDeterminant) &&
			PxAbs(inertiaDeterminant) > 1.0e-12f;
		const PxMat33 inverseInertia = hasAngularResponse
			? inertia.getInverse()
			: PxMat33::createDiagonal(PxVec3(0.0f));
		// Swept AL multipliers and contact-owned depenetration caps are not
		// discrete end-step impulses. Preserve their stable position-derived
		// component momentum; an opted-in swept body receives only the bounded
		// uniform velocity boundary below. Ordinary discrete collision batches
		// use AL's external force to restore their missing global angular impulse.
		const bool preservePositionDerivedMomentum =
			mode == AvbdSoftComponentFinalizeMode::ePOSITION_OWNED &&
			(hasSpeculativeCcdBody ||
			 body.compiled.maxDepenetrationVelocity < 1.0e20f);
		PxVec3 targetLinearMomentum = preservePositionDerivedMomentum
			? actualLinearMomentum : target.linearMomentum;
		PxVec3 targetAngularMomentum = preservePositionDerivedMomentum
			? actualAngularMomentum : target.angularMomentum;

		if(mode == AvbdSoftComponentFinalizeMode::ePOSITION_OWNED &&
			!preservePositionDerivedMomentum)
		{
			PxVec3 targetAbsoluteAngularMomentum =
				target.angularMomentum +
					target.centroid.cross(target.linearMomentum);
			const PxReal dt = 1.0f / invDt;
			const PxU32 particleStart = body.compiled.particleStart;
			const PxU32 particleEnd =
				particleStart + body.compiled.particleCount;
			for(PxU32 contactIndex = 0;
				contactIndex < numContacts; ++contactIndex)
			{
				const AvbdSoftContact& contact = contacts[contactIndex];
				const AvbdSoftContactGeometry& geometry = contact.geometry;
				if(geometry.velocityOwner !=
					AvbdVelocityObjectiveOwner::PositionAL ||
					contact.state.alLambda >= 0.0f ||
					geometry.hasWorldStaticTarget())
					continue;
				const PxVec3 contactForce =
					geometry.normal * (-contact.state.alLambda) -
					geometry.tangent1 *
						contact.state.alLambdaTangent[0] -
					geometry.tangent2 *
						contact.state.alLambdaTangent[1];
				if(!contactForce.isFinite())
					continue;
				PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
				const PxU32 particleCount =
					avbdCollectSoftContactParticleIndices(
						geometry, particleIndices);
				for(PxU32 supportIndex = 0;
					supportIndex < particleCount; ++supportIndex)
				{
					const PxU32 particleIndex =
						particleIndices[supportIndex];
					if(particleIndex < particleStart ||
						particleIndex >= particleEnd ||
						particleIndex >= numParticles)
						continue;
					const PxReal jacobianScale =
						avbdGetSoftContactParticleJacobianScale(
							geometry, particleIndex);
					if(PxAbs(jacobianScale) <= 1.0e-12f)
						continue;
					const PxVec3 impulse =
						contactForce * (jacobianScale * dt);
					targetLinearMomentum += impulse;
					targetAbsoluteAngularMomentum +=
						particles[particleIndex].position.cross(impulse);
				}
			}
			targetAngularMomentum =
				targetAbsoluteAngularMomentum -
				centroid.cross(targetLinearMomentum);

			// Position AL owns geometric non-penetration, but its accumulated
			// multiplier is not a discrete material impulse. Replaying that
			// multiplier against a world-static surface can overshoot the
			// inelastic velocity boundary every frame and pump both translation
			// and rotation. Rebuild the static response from the damped inertial
			// component momentum instead: each sequential row applies exactly the
			// non-negative impulse needed to remove inward point velocity, followed
			// by a Coulomb-bounded tangent impulse. This retains impact torque while
			// preventing a resting contact from creating separating kinetic energy.
			for(PxU32 contactIndex = 0;
				contactIndex < numContacts; ++contactIndex)
			{
				const AvbdSoftContact& contact = contacts[contactIndex];
				const AvbdSoftContactGeometry& geometry = contact.geometry;
				if(geometry.velocityOwner !=
						AvbdVelocityObjectiveOwner::PositionAL ||
					contact.state.alLambda >= 0.0f ||
					!geometry.hasWorldStaticTarget())
					continue;

				PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
				const PxU32 particleCount =
					avbdCollectSoftContactParticleIndices(
						geometry, particleIndices);
				bool belongsToBody = false;
				for(PxU32 supportIndex = 0;
					supportIndex < particleCount; ++supportIndex)
				{
					const PxU32 particleIndex =
						particleIndices[supportIndex];
					if(particleIndex >= particleStart &&
						particleIndex < particleEnd &&
						particleIndex < numParticles &&
						PxAbs(avbdGetSoftContactParticleJacobianScale(
							geometry, particleIndex)) > 1.0e-12f)
					{
						belongsToBody = true;
						break;
					}
				}
				if(!belongsToBody)
					continue;

				const PxVec3 normal = geometry.normal;
				const PxVec3 queryPoint =
					avbdGetSoftContactQueryPoint(geometry, particles);
				if(!normal.isFinite() || !queryPoint.isFinite())
					continue;
				const PxVec3 offset = queryPoint - centroid;
				const PxVec3 angularMomentum =
					targetAbsoluteAngularMomentum -
					centroid.cross(targetLinearMomentum);
				const PxVec3 angularVelocity = hasAngularResponse
					? inverseInertia * angularMomentum
					: PxVec3(0.0f);
				const PxVec3 linearVelocity =
					targetLinearMomentum * (1.0f / mass);
				const PxVec3 normalAngularJacobian =
					offset.cross(normal);
				const PxReal normalResponse =
					1.0f / mass +
					(hasAngularResponse
						? normalAngularJacobian.dot(
							inverseInertia * normalAngularJacobian)
						: 0.0f);
				const PxReal relativeNormalVelocity =
					(linearVelocity + angularVelocity.cross(offset)).
						dot(normal);
				if(normalResponse <= 1.0e-12f ||
					!PxIsFinite(relativeNormalVelocity) ||
					relativeNormalVelocity >= 0.0f)
					continue;

				const PxReal normalImpulseMagnitude =
					-relativeNormalVelocity / normalResponse;
				const PxVec3 normalImpulse =
					normal * normalImpulseMagnitude;
				targetLinearMomentum += normalImpulse;
				targetAbsoluteAngularMomentum +=
					queryPoint.cross(normalImpulse);

				const PxReal frictionLimit =
					PxMax(geometry.friction, 0.0f) *
						normalImpulseMagnitude;
				if(frictionLimit <= 0.0f)
					continue;
				const PxVec3 postNormalAngularMomentum =
					targetAbsoluteAngularMomentum -
					centroid.cross(targetLinearMomentum);
				const PxVec3 postNormalAngularVelocity =
					hasAngularResponse
						? inverseInertia * postNormalAngularMomentum
						: PxVec3(0.0f);
				const PxVec3 postNormalLinearVelocity =
					targetLinearMomentum * (1.0f / mass);
				const PxVec3 pointVelocity =
					postNormalLinearVelocity +
					postNormalAngularVelocity.cross(offset);
				const PxVec3 tangentAngularJacobian0 =
					offset.cross(geometry.tangent1);
				const PxVec3 tangentAngularJacobian1 =
					offset.cross(geometry.tangent2);
				const PxReal response00 =
					1.0f / mass +
					(hasAngularResponse
						? tangentAngularJacobian0.dot(
							inverseInertia * tangentAngularJacobian0)
						: 0.0f);
				const PxReal response11 =
					1.0f / mass +
					(hasAngularResponse
						? tangentAngularJacobian1.dot(
							inverseInertia * tangentAngularJacobian1)
						: 0.0f);
				const PxReal response01 = hasAngularResponse
					? tangentAngularJacobian0.dot(
						inverseInertia * tangentAngularJacobian1)
					: 0.0f;
				const PxReal determinant =
					response00 * response11 - response01 * response01;
				if(!PxIsFinite(determinant) ||
					PxAbs(determinant) <= 1.0e-12f)
					continue;
				const PxReal rhs0 =
					-pointVelocity.dot(geometry.tangent1);
				const PxReal rhs1 =
					-pointVelocity.dot(geometry.tangent2);
				PxReal tangentImpulse0 =
					(response11 * rhs0 - response01 * rhs1) /
						determinant;
				PxReal tangentImpulse1 =
					(response00 * rhs1 - response01 * rhs0) /
						determinant;
				const PxReal tangentMagnitude = PxSqrt(
					tangentImpulse0 * tangentImpulse0 +
					tangentImpulse1 * tangentImpulse1);
				if(tangentMagnitude > frictionLimit &&
					tangentMagnitude > 1.0e-12f)
				{
					const PxReal scale =
						frictionLimit / tangentMagnitude;
					tangentImpulse0 *= scale;
					tangentImpulse1 *= scale;
				}
				const PxVec3 tangentImpulse =
					geometry.tangent1 * tangentImpulse0 +
					geometry.tangent2 * tangentImpulse1;
				if(!tangentImpulse.isFinite())
					continue;
				targetLinearMomentum += tangentImpulse;
				targetAbsoluteAngularMomentum +=
					queryPoint.cross(tangentImpulse);
			}
			targetAngularMomentum =
				targetAbsoluteAngularMomentum -
				centroid.cross(targetLinearMomentum);
		}
		else if(preservePositionDerivedMomentum &&
			body.compiled.speculativeCCDEnabled)
		{
			// A uniform component correction preserves all relative particle
			// velocities and therefore cannot distort volume. It only removes
			// inward normal speed at active static/kinematic swept contacts.
			PxVec3 linearVelocityCorrection(0.0f);
			const PxU32 particleStart = body.compiled.particleStart;
			const PxU32 particleEnd =
				particleStart + body.compiled.particleCount;
			for(PxU32 contactIndex = 0;
				contactIndex < numContacts; ++contactIndex)
			{
				const AvbdSoftContact& contact = contacts[contactIndex];
				const AvbdSoftContactGeometry& geometry = contact.geometry;
				if(geometry.velocityOwner !=
						AvbdVelocityObjectiveOwner::PositionAL ||
					contact.state.alLambda >= 0.0f ||
					(!geometry.hasWorldStaticTarget() &&
					 !geometry.hasKinematicRigidTarget()))
					continue;
				PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
				const PxU32 particleCount =
					avbdCollectSoftContactParticleIndices(
						geometry, particleIndices);
				PxVec3 queryVelocity(0.0f);
				bool belongsToBody = false;
				for(PxU32 supportIndex = 0;
					supportIndex < particleCount; ++supportIndex)
				{
					const PxU32 particleIndex =
						particleIndices[supportIndex];
					if(particleIndex < particleStart ||
						particleIndex >= particleEnd ||
						particleIndex >= numParticles)
						continue;
					const PxReal jacobianScale =
						avbdGetSoftContactParticleJacobianScale(
							geometry, particleIndex);
					if(PxAbs(jacobianScale) <= 1.0e-12f)
						continue;
					belongsToBody = true;
					queryVelocity +=
						particles[particleIndex].velocity *
							jacobianScale;
				}
				if(!belongsToBody)
					continue;
				const PxVec3 surfaceVelocity =
					geometry.hasKinematicRigidTarget()
						? (geometry.surfacePoint -
						   geometry.kinematicSurfacePointPrevious) *
							invDt
						: PxVec3(0.0f);
				const PxReal relativeNormalVelocity =
					(queryVelocity + linearVelocityCorrection -
					 surfaceVelocity).dot(geometry.normal);
				if(relativeNormalVelocity < 0.0f &&
					PxIsFinite(relativeNormalVelocity))
					linearVelocityCorrection +=
						geometry.normal * (-relativeNormalVelocity);
			}
			if(linearVelocityCorrection.isFinite())
				targetLinearMomentum +=
					linearVelocityCorrection * mass;
		}

		if(mode ==
			AvbdSoftComponentFinalizeMode::eKINEMATIC_CONTACT)
		{
			for(PxU32 objectiveIndex = 0;
				objectiveIndex < numVelocityObjectives;
				objectiveIndex++)
			{
				const AvbdCompiledSoftVelocityObjective& objective =
					velocityObjectives[objectiveIndex];
				if(objective.owner !=
						AvbdVelocityObjectiveOwner::
							ComponentFinalize ||
					objective.bodyIndex != bodyIndex ||
					objective.particleIndex <
						body.compiled.particleStart ||
					objective.particleIndex >=
						body.compiled.particleStart +
							body.compiled.particleCount)
					continue;
				const PxVec3 normal = objective.normal;
				PxVec3 queryPoint =
					particles[objective.particleIndex].position;
				if(objective.queryPoint.count != 0)
				{
					queryPoint = PxVec3(0.0f);
					for(PxU32 queryVertex = 0;
						queryVertex < objective.queryPoint.count;
						queryVertex++)
					{
						const PxU32 queryParticle =
							objective.queryPoint.particleIndices[
								queryVertex];
						queryPoint +=
							particles[queryParticle].position *
							objective.queryPoint.weights[queryVertex];
					}
				}
				const PxVec3 offset = queryPoint - centroid;
				const PxVec3 targetLinearVelocity =
					targetLinearMomentum * (1.0f / mass);
				const PxVec3 targetAngularVelocity =
					hasAngularResponse
						? inverseInertia * targetAngularMomentum
						: PxVec3(0.0f);
				const PxVec3 surfaceVelocity =
					(objective.surfacePoint -
					 objective.previousSurfacePoint) *
						invDt;
				const PxReal relativeNormalVelocity =
					(targetLinearVelocity +
					 targetAngularVelocity.cross(offset) -
					 surfaceVelocity).dot(normal);
				const PxVec3 angularJacobian =
					offset.cross(normal);
				const PxReal response =
					1.0f / mass +
					(hasAngularResponse
						? angularJacobian.dot(
							inverseInertia * angularJacobian)
						: 0.0f);
				if(response <= 1.0e-12f ||
					!PxIsFinite(relativeNormalVelocity))
					continue;
				// Position AL already owns penetration.  This typed
				// component owner supplies only the e=0 velocity boundary
				// at the prescribed surface; it does not iterate impulses.
				const PxReal correction =
					-relativeNormalVelocity / response;
				const PxVec3 momentumDelta = normal * correction;
				targetLinearMomentum += momentumDelta;
				targetAngularMomentum +=
					offset.cross(momentumDelta);
			}
		}

		const PxVec3 actualLinearVelocity =
			actualLinearMomentum * (1.0f / mass);
		const PxVec3 targetLinearVelocity =
			targetLinearMomentum * (1.0f / mass);
		const PxVec3 actualAngularVelocity =
			hasAngularResponse
				? inverseInertia * actualAngularMomentum
				: PxVec3(0.0f);
		const PxVec3 targetAngularVelocity =
			hasAngularResponse
				? inverseInertia * targetAngularMomentum
				: actualAngularVelocity;
		for(PxU32 localIndex = 0;
			localIndex < body.compiled.particleCount;
			localIndex++)
		{
			AvbdSoftParticle& particle =
				particles[body.compiled.particleStart + localIndex];
			if(particle.invMass <= 0.0f)
				continue;
			const PxVec3 offset = particle.position - centroid;
			particle.velocity +=
				(targetLinearVelocity +
				 targetAngularVelocity.cross(offset)) -
				(actualLinearVelocity +
				 actualAngularVelocity.cross(offset));
			if(!particle.velocity.isFinite())
			{
				PX_ASSERT(false);
					particle.velocity = particle.prevVelocity;
			}
		}
	}
	}

PX_FORCE_INLINE void avbdUpdateSoftContactDual(
	const AvbdSoftContactGeometry& geometry,
	AvbdSoftContactAugmentedState& state,
	const AvbdSoftParticle* particles,
	PxReal beta)
{
	avbdUpdateSoftContactDualAtSurfacePoint(
		geometry, state, particles,
		avbdGetSoftContactSurfacePoint(geometry, particles), beta);
}

// =============================================================================
// Ground contact detection
// =============================================================================

struct AvbdWorldPlane
{
	PxVec3 normal;
	PxReal offset;
	PxReal friction;
	PxU8 frictionCombineMode;
	PxU64 primitiveKey;

	AvbdWorldPlane()
		: normal(0.0f, 1.0f, 0.0f), offset(0.0f),
		  friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE)),
		  primitiveKey(0)
	{
	}
};

// P5.3a private-output leaf: a worker owns one canonical particle interval
// and appends only to its local contact array. Parent merge in ascending range
// order exactly reproduces the legacy particle-major/plane-minor stream.
inline void avbdDetectSoftWorldPlaneContactsRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdWorldPlane* planes, PxU32 numPlanes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.02f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	PX_ASSERT(particleBegin <= particleEnd && particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for(PxU32 i = particleBegin; i < particleEnd; i++)
	{
		if(particles[i].invMass <= 0.0f)
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies, i);
		if(sourceBody &&
			!avbdIsSoftBodySurfaceVertex(*sourceBody, i))
			continue;
		const PxVec3& position = particles[i].position;
		for(PxU32 planeIndex = 0;
			planeIndex < numPlanes; planeIndex++)
		{
			const AvbdWorldPlane& plane = planes[planeIndex];
			const PxReal normalMagnitudeSq =
				plane.normal.magnitudeSquared();
			if(normalMagnitudeSq <= 1e-12f ||
				!PxIsFinite(normalMagnitudeSq))
				continue;
			const PxVec3 normal =
				plane.normal * PxRecipSqrt(normalMagnitudeSq);
			const PxReal distance =
				normal.dot(position) - plane.offset;
			bool speculativeCandidate = false;
			if(distance >= margin)
			{
				if(!sourceBody ||
					!sourceBody->compiled.speculativeCCDEnabled ||
					!particles[i].predictedPosition.isFinite())
					continue;
				const PxReal predictedDistance =
					normal.dot(particles[i].predictedPosition) -
						plane.offset;
				if(predictedDistance >= margin)
					continue;
				speculativeCandidate = true;
			}

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eGROUND, PX_MAX_U32,
				plane.primitiveKey, 0);
			geometry.particleIdx = i;
			geometry.targetKind =
				AvbdSoftContactTargetKind::eWORLD_STATIC;
			geometry.velocityOwner =
				AvbdVelocityObjectiveOwner::PositionAL;
			geometry.targetIndex = planeIndex;
			geometry.normal = normal;
			geometry.projNormal = normal;
			geometry.depth = PxMax(0.0f, -distance);
			geometry.margin = margin;
			geometry.surfacePoint =
				position - normal * distance;
			geometry.friction = sourceBody
				? avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction,
					plane.friction, plane.frictionCombineMode)
				: PxMax(plane.friction, 0.0f);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry,
				speculativeCandidate ? 1e6f : 1e4f,
				1e6f, particles, contacts);
		}
	}
}

inline void avbdDetectSoftWorldPlaneContacts(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdWorldPlane* planes, PxU32 numPlanes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.02f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	avbdDetectSoftWorldPlaneContactsRange(
		particles, numParticles, 0, numParticles,
		planes, numPlanes, contacts, margin,
		softBodies, numSoftBodies);
}

inline void avbdDetectSoftGroundContacts(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxArray<AvbdSoftContact>& contacts,
	PxReal groundY = 0.0f, PxReal margin = 0.02f,
	PxReal friction = 0.5f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	contacts.clear();
	AvbdWorldPlane plane;
	plane.offset = groundY;
	plane.friction = friction;
	avbdDetectSoftWorldPlaneContacts(
		particles, numParticles, &plane, 1, contacts, margin,
		softBodies, numSoftBodies);
}

// =============================================================================
// Rigid box descriptor for soft-rigid collision
// =============================================================================

struct AvbdRigidBox
{
	PxVec3 center;
	PxQuat rotation;
	PxVec3 halfExtent;
	PxReal friction;
	PxU8 frictionCombineMode;
	PxU64 primitiveKey;
	AvbdSoftContactTargetKind targetKind;
	PxU32 targetIndex;
	PxVec3 previousCenter;
	PxQuat previousRotation;
	// Shape-to-rigid-body transform.  Dynamic contacts store their surface
	// anchor in this body-local frame so the rigid 6x6 block and the soft 3x3
	// block evaluate the same moving position objective.
	PxTransform shapeToRigidBody;

	AvbdRigidBox()
		: center(0.0f), rotation(PxIdentity), halfExtent(0.0f),
		  friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE)),
		  primitiveKey(0),
		  targetKind(AvbdSoftContactTargetKind::eWORLD_STATIC),
		  targetIndex(PX_MAX_U32), previousCenter(0.0f),
		  previousRotation(PxIdentity), shapeToRigidBody(PxIdentity)
	{
	}
};

// Analytical sphere descriptor. E31 first admitted world-static spheres; the
// prescribed/dynamic fields keep later moving-target ownership explicit rather
// than treating every sphere as an immobile Position-AL target.
struct AvbdRigidSphere
{
	PxVec3 center;
	PxQuat rotation;
	PxReal radius;
	PxReal friction;
	PxU8 frictionCombineMode;
	PxU64 primitiveKey;
	AvbdSoftContactTargetKind targetKind;
	PxU32 targetIndex;
	PxVec3 previousCenter;
	PxQuat previousRotation;
	// Native dynamic spheres retain their current shape pose in center /
	// rotation and publish the current-frame rigid prediction separately.
	// This keeps the E31 discrete and E34 prescribed-target meanings intact.
	PxVec3 predictedCenter;
	PxQuat predictedRotation;
	bool predictedPoseValid;
	PxTransform shapeToRigidBody;

	AvbdRigidSphere()
		: center(0.0f), rotation(PxIdentity), radius(0.0f),
		  friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE)),
		  primitiveKey(0),
		  targetKind(AvbdSoftContactTargetKind::eWORLD_STATIC),
		  targetIndex(PX_MAX_U32), previousCenter(0.0f),
		  previousRotation(PxIdentity), predictedCenter(0.0f),
		  predictedRotation(PxIdentity), predictedPoseValid(false),
		  shapeToRigidBody(PxIdentity)
	{
	}
};

// Analytical capsule descriptor. PhysX capsules use the shape-local X axis;
// halfHeight is the half length of the medial segment and radius is the
// spherical sweep radius. The target fields deliberately match box/sphere so
// static, prescribed kinematic, and native dynamic contacts share one owner
// contract.
struct AvbdRigidCapsule
{
	PxVec3 center;
	PxQuat rotation;
	PxReal radius;
	PxReal halfHeight;
	PxReal friction;
	PxU8 frictionCombineMode;
	PxU64 primitiveKey;
	AvbdSoftContactTargetKind targetKind;
	PxU32 targetIndex;
	PxVec3 previousCenter;
	PxQuat previousRotation;
	PxVec3 predictedCenter;
	PxQuat predictedRotation;
	bool predictedPoseValid;
	PxTransform shapeToRigidBody;

	AvbdRigidCapsule()
		: center(0.0f), rotation(PxIdentity), radius(0.0f),
		  halfHeight(0.0f), friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE)),
		  primitiveKey(0),
		  targetKind(AvbdSoftContactTargetKind::eWORLD_STATIC),
		  targetIndex(PX_MAX_U32), previousCenter(0.0f),
		  previousRotation(PxIdentity), predictedCenter(0.0f),
		  predictedRotation(PxIdentity), predictedPoseValid(false),
		  shapeToRigidBody(PxIdentity)
	{
	}
};

struct AvbdRigidConvexFace
{
	PxVec3 normal;
	PxReal offset;

	AvbdRigidConvexFace()
		: normal(0.0f, 1.0f, 0.0f), offset(0.0f)
	{
	}
};

struct AvbdRigidConvexEdge
{
	PxU32 p0;
	PxU32 p1;
	PxVec3 outward;

	AvbdRigidConvexEdge()
		: p0(PX_MAX_U32), p1(PX_MAX_U32),
		  outward(0.0f, 1.0f, 0.0f)
	{
	}
};

struct AvbdRigidConvexTriangle
{
	PxU32 p0;
	PxU32 p1;
	PxU32 p2;
	PxU32 faceIndex;

	AvbdRigidConvexTriangle()
		: p0(PX_MAX_U32), p1(PX_MAX_U32),
		  p2(PX_MAX_U32), faceIndex(PX_MAX_U32)
	{
	}
};

// Scene-owned convex topology. Mesh scaling is baked into these shape-local
// vertices during prep, leaving the detector independent of PxConvexMesh
// lifetime and geomutils internals. Faces own the closed-hull signed-distance
// test, while boundary vertices/edges/triangles own reverse OGC features.
struct AvbdRigidConvex
{
	PxVec3 center;
	PxQuat rotation;
	PxVec3 previousCenter;
	PxQuat previousRotation;
	PxReal localRadius;
	PxReal friction;
	PxU8 frictionCombineMode;
	PxU64 primitiveKey;
	AvbdSoftContactTargetKind targetKind;
	PxU32 targetIndex;
	PxVec3 predictedCenter;
	PxQuat predictedRotation;
	bool predictedPoseValid;
	PxTransform shapeToRigidBody;
	PxArray<PxVec3> vertices;
	PxArray<PxVec3> vertexNormals;
	PxArray<AvbdRigidConvexFace> faces;
	PxArray<AvbdRigidConvexEdge> edges;
	PxArray<AvbdRigidConvexTriangle> triangles;
	// Parent-owned feature suffixes rebuild this from the current pose once
	// per redetection epoch. Keeping it on the convex avoids charging scenes
	// without convex geometry for cache state or workspace layout changes.
	mutable PxArray<AvbdRigidConvexEdgeBounds> edgeBoundsScratch;

	AvbdRigidConvex()
		: center(0.0f), rotation(PxIdentity),
		  previousCenter(0.0f), previousRotation(PxIdentity),
		  localRadius(0.0f), friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE)),
		  primitiveKey(0),
		  targetKind(AvbdSoftContactTargetKind::eWORLD_STATIC),
		  targetIndex(PX_MAX_U32), predictedCenter(0.0f),
		  predictedRotation(PxIdentity), predictedPoseValid(false),
		  shapeToRigidBody(PxIdentity)
	{
	}
};

struct AvbdRigidTriangleSurfaceVertex
{
	PxVec3 point;
	PxVec3 outward;
	PxReal friction;
	PxU8 frictionCombineMode;
	// First source face which owns this feature's material.  Scene topology
	// caching refreshes that material without rebuilding vertex adjacency.
	PxU32 sourceTriangleIndex;
	bool active;

	AvbdRigidTriangleSurfaceVertex()
		: point(0.0f), outward(0.0f, 1.0f, 0.0f),
		  friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE)),
		  sourceTriangleIndex(PX_MAX_U32),
		  active(false)
	{
	}
};

struct AvbdRigidTriangleSurfaceEdge
{
	PxU32 p0;
	PxU32 p1;
	PxU32 triangle0;
	PxU32 triangle1;
	PxU32 adjacentCount;
	PxVec3 outward;
	PxReal friction;
	PxU8 frictionCombineMode;
	// First source face which created the undirected edge.  This is the
	// canonical material owner retained by the original topology builder.
	PxU32 sourceTriangleIndex;
	bool active;

	AvbdRigidTriangleSurfaceEdge()
		: p0(PX_MAX_U32), p1(PX_MAX_U32),
		  triangle0(PX_MAX_U32), triangle1(PX_MAX_U32),
		  adjacentCount(0),
		  outward(0.0f, 1.0f, 0.0f), friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE)),
		  sourceTriangleIndex(PX_MAX_U32),
		  active(false)
	{
	}
};

struct AvbdRigidTriangleSurfaceTriangle
{
	PxU32 p0;
	PxU32 p1;
	PxU32 p2;
	PxU32 edge0;
	PxU32 edge1;
	PxU32 edge2;
	PxU32 sourceTriangleIndex;
	PxVec3 normal;
	PxReal friction;
	PxU8 frictionCombineMode;

	AvbdRigidTriangleSurfaceTriangle()
		: p0(PX_MAX_U32), p1(PX_MAX_U32),
		  p2(PX_MAX_U32), edge0(PX_MAX_U32),
		  edge1(PX_MAX_U32), edge2(PX_MAX_U32),
		  sourceTriangleIndex(PX_MAX_U32),
		  normal(0.0f, 1.0f, 0.0f), friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE))
	{
	}
};

// Immutable local-space hierarchy over a persistent rigid triangle surface.
// Unlike deformable surface BVHs, these bounds never refit: topology and
// local vertices are rebuilt only on the Scene cache invalidation boundary.
struct AvbdRigidTriangleSurfaceBvhNode
{
	PxVec3 minimum;
	PxVec3 maximum;
	PxU32 leftChild;
	PxU32 rightChild;
	PxU32 firstPrimitive;
	PxU32 primitiveCount;

	PX_FORCE_INLINE bool isLeaf() const
	{
		return leftChild == PX_MAX_U32;
	}
};

// Scene-owned open triangle surface shared by triangle-mesh and heightfield
// shapes. PxMeshQuery bakes scale and negative-determinant winding during
// prep, so detection neither retains public mesh objects nor depends on Gu
// internals. Unlike a closed convex, this descriptor intentionally preserves
// one-sided simulation semantics and only exposes boundary/convex active
// edges for reverse OGC features.
struct AvbdRigidTriangleSurface
{
	PxVec3 center;
	PxQuat rotation;
	PxVec3 previousCenter;
	PxQuat previousRotation;
	PxBounds3 localBounds;
	PxReal localRadius;
	PxU64 primitiveKey;
	AvbdSoftContactTargetKind targetKind;
	PxU32 targetIndex;
	PxTransform shapeToRigidBody;
	// Scene-owned immutable-topology cache identity.  Pose fields above are
	// refreshed every detection; these fields only invalidate when the mesh,
	// scale, or heightfield geometry actually changes.
	const void* topologySource;
	PxU8 topologyGeometryType;
	PxVec3 topologyScale;
	PxQuat topologyScaleRotation;
	PxReal topologyHeightScale;
	PxReal topologyRowScale;
	PxReal topologyColumnScale;
	// Heightfields can edit samples in place.  Their public timestamp is part
	// of the topology identity so that mutation invalidates the cache while
	// triangle meshes (which are immutable) remain allocation-free.
	PxU32 topologyContentTimestamp;
	PxU32 sceneCompileStamp;
	PxU32 sceneCompileOrder;
	PxArray<AvbdRigidTriangleSurfaceVertex> vertices;
	PxArray<AvbdRigidTriangleSurfaceEdge> edges;
	PxArray<AvbdRigidTriangleSurfaceTriangle> triangles;
	PxArray<PxU32> triangleBvhTriangleIndices;
	PxArray<AvbdRigidTriangleSurfaceBvhNode> triangleBvhNodes;
	// Detection is serial through P1. These pre-reserved query candidates avoid
	// per-query allocation while their mutability is deliberately confined to
	// the current serial reference/BVH comparison stage.
	mutable PxArray<PxU32> triangleBvhQueryCandidates;
	// Active reverse features are recovered from the immutable triangle leaves
	// which own them. Stamps deduplicate shared features without a per-query
	// clear; sorted ids restore the legacy edge/vertex traversal order.
	mutable PxArray<PxU32> edgeBvhQueryCandidates;
	mutable PxArray<PxU32> vertexBvhQueryCandidates;
	mutable PxArray<PxU32> edgeBvhCandidateStamps;
	mutable PxArray<PxU32> vertexBvhCandidateStamps;
	mutable PxU32 featureBvhCandidateStamp;

	AvbdRigidTriangleSurface()
		: center(0.0f), rotation(PxIdentity),
		  previousCenter(0.0f), previousRotation(PxIdentity),
		  localBounds(PxBounds3::empty()), localRadius(0.0f),
		  primitiveKey(0),
		  targetKind(AvbdSoftContactTargetKind::eWORLD_STATIC),
		  targetIndex(PX_MAX_U32), shapeToRigidBody(PxIdentity),
		  topologySource(NULL), topologyGeometryType(PX_MAX_U8),
		  topologyScale(1.0f), topologyScaleRotation(PxIdentity),
		  topologyHeightScale(0.0f), topologyRowScale(0.0f),
		  topologyColumnScale(0.0f), topologyContentTimestamp(0),
		  sceneCompileStamp(0),
		  sceneCompileOrder(0), featureBvhCandidateStamp(0)
	{
	}
};

// Caller-owned query state for a particle-range triangle-surface swept leaf.
// The baked descriptor remains immutable while this object owns every BVH
// candidate and feature-dedup write that legacy serial detection stored on the
// descriptor itself. A task may reuse one instance across its range, but never
// shares it with another task.
struct AvbdRigidTriangleSurfaceQueryScratch
{
	PxArray<PxU32> triangleBvhQueryCandidates;
	PxArray<PxU32> edgeBvhQueryCandidates;
	PxArray<PxU32> vertexBvhQueryCandidates;
	PxArray<PxU32> edgeBvhCandidateStamps;
	PxArray<PxU32> vertexBvhCandidateStamps;
	PxU32 featureBvhCandidateStamp;

	AvbdRigidTriangleSurfaceQueryScratch()
		: featureBvhCandidateStamp(0)
	{
	}

	void reserve(PxU32 triangleCount, PxU32 edgeCount, PxU32 vertexCount)
	{
		triangleBvhQueryCandidates.reserve(triangleCount);
		edgeBvhQueryCandidates.reserve(edgeCount);
		vertexBvhQueryCandidates.reserve(vertexCount);
		if(edgeBvhCandidateStamps.size() < edgeCount)
		{
			const PxU32 previousCount = edgeBvhCandidateStamps.size();
			edgeBvhCandidateStamps.resize(edgeCount);
			for(PxU32 index = previousCount; index < edgeCount; ++index)
				edgeBvhCandidateStamps[index] = 0;
		}
		if(vertexBvhCandidateStamps.size() < vertexCount)
		{
			const PxU32 previousCount = vertexBvhCandidateStamps.size();
			vertexBvhCandidateStamps.resize(vertexCount);
			for(PxU32 index = previousCount; index < vertexCount; ++index)
				vertexBvhCandidateStamps[index] = 0;
		}
	}

	bool beginFeatureCandidates(PxU32 edgeCount, PxU32 vertexCount)
	{
		if(edgeBvhCandidateStamps.size() != edgeCount)
		{
			edgeBvhCandidateStamps.resize(edgeCount);
			for(PxU32 index = 0; index < edgeCount; ++index)
				edgeBvhCandidateStamps[index] = 0;
		}
		if(vertexBvhCandidateStamps.size() != vertexCount)
		{
			vertexBvhCandidateStamps.resize(vertexCount);
			for(PxU32 index = 0; index < vertexCount; ++index)
				vertexBvhCandidateStamps[index] = 0;
		}
		if(++featureBvhCandidateStamp == 0)
		{
			featureBvhCandidateStamp = 1;
			for(PxU32 index = 0;
				index < edgeBvhCandidateStamps.size(); ++index)
				edgeBvhCandidateStamps[index] = 0;
			for(PxU32 index = 0;
				index < vertexBvhCandidateStamps.size(); ++index)
				vertexBvhCandidateStamps[index] = 0;
		}
		return true;
	}
};

// P5.17b's immutable parent IR for the two triangle-surface OGC feature
// suffixes.  `primitiveBegin/End` address a soft-edge index for eSOFT_EDGE and
// a soft-triangle index (not its three-entry array offset) for eSOFT_TRIANGLE.
// Keeping phase, body, surface and feature family in one ordered stream makes
// the legacy output order explicit: all swept rows precede all discrete rows;
// within either phase, body -> surface -> soft-edge -> soft-triangle.  This is
// deliberately a plan only: no range leaf or task may consume it until its
// per-row output and scratch contracts are established.
struct AvbdRigidTriangleSurfaceFeatureWorkItem
{
	enum Phase : PxU8
	{
		eSWEPT,
		eDISCRETE
	};
	enum Family : PxU8
	{
		eSOFT_EDGE,
		eSOFT_TRIANGLE
	};

	Phase phase;
	Family family;
	PxU32 bodyIndex;
	PxU32 surfaceIndex;
	PxU32 primitiveBegin;
	PxU32 primitiveEnd;

	AvbdRigidTriangleSurfaceFeatureWorkItem(
		Phase inputPhase, Family inputFamily,
		PxU32 inputBodyIndex, PxU32 inputSurfaceIndex,
		PxU32 inputPrimitiveBegin, PxU32 inputPrimitiveEnd)
		: phase(inputPhase), family(inputFamily),
		  bodyIndex(inputBodyIndex), surfaceIndex(inputSurfaceIndex),
		  primitiveBegin(inputPrimitiveBegin), primitiveEnd(inputPrimitiveEnd)
	{
	}
};

struct AvbdRigidTriangleSurfaceFeaturePlan
{
	PxArray<AvbdRigidTriangleSurfaceFeatureWorkItem> items;

	void clear()
	{
		items.clear();
	}
};

// P5.26 keeps the canonical plan execution untouched while allowing the
// default-off Scene task leaf to attribute each immutable row by phase/family.
// The optional object owns no collision state and callers may omit it without
// paying for per-row clock reads.
struct AvbdRigidTriangleSurfaceFeaturePlanRangeTiming
{
	PxU64 sweptEdgeNanos;
	PxU64 sweptTriangleNanos;
	PxU64 discreteEdgeNanos;
	PxU64 discreteTriangleNanos;

	AvbdRigidTriangleSurfaceFeaturePlanRangeTiming()
		: sweptEdgeNanos(0), sweptTriangleNanos(0),
		  discreteEdgeNanos(0), discreteTriangleNanos(0)
	{
	}

	void record(const AvbdRigidTriangleSurfaceFeatureWorkItem& workItem,
		PxU64 elapsedNanos)
	{
		if(workItem.phase ==
			AvbdRigidTriangleSurfaceFeatureWorkItem::eSWEPT)
		{
			if(workItem.family ==
				AvbdRigidTriangleSurfaceFeatureWorkItem::eSOFT_EDGE)
				sweptEdgeNanos += elapsedNanos;
			else
				sweptTriangleNanos += elapsedNanos;
		}
		else if(workItem.family ==
			AvbdRigidTriangleSurfaceFeatureWorkItem::eSOFT_EDGE)
			discreteEdgeNanos += elapsedNanos;
		else
			discreteTriangleNanos += elapsedNanos;
	}
};

// P5.30's optional timing stays below the canonical plan-row boundary. The
// three intervals are deliberately non-exhaustive: primitive validation and
// row-loop bookkeeping remain outside them, and therefore their sum can only
// bound (never replace) P5.26's swept edge/triangle leaf totals.
struct AvbdRigidTriangleSurfaceSweptOGCFeatureSubstageTiming
{
	PxU64 sweptEdgeForwardOwnerNanos;
	PxU64 sweptEdgeBvhRecoveryNanos;
	PxU64 sweptEdgeNarrowPhaseNanos;
	PxU64 sweptTriangleForwardOwnerNanos;
	PxU64 sweptTriangleBvhRecoveryNanos;
	PxU64 sweptTriangleNarrowPhaseNanos;

	AvbdRigidTriangleSurfaceSweptOGCFeatureSubstageTiming()
		: sweptEdgeForwardOwnerNanos(0), sweptEdgeBvhRecoveryNanos(0),
		  sweptEdgeNarrowPhaseNanos(0),
		  sweptTriangleForwardOwnerNanos(0),
		  sweptTriangleBvhRecoveryNanos(0),
		  sweptTriangleNarrowPhaseNanos(0)
	{
	}
};

// P5.31's exact query identity is (rigid-surface index, soft-particle index).
// The task owns the stamp payload and advances its generation between frames,
// so this object is only an observability contract and never stores a query
// result that could alter forward-owner behavior.
struct AvbdRigidTriangleSurfaceForwardOwnerQueryStats
{
	PxArray<PxU32>* stamps;
	PxU32 numParticles;
	PxU32 numSurfaces;
	PxU32 stamp;
	PxU64 queryCalls;
	PxU64 uniqueQueries;

	AvbdRigidTriangleSurfaceForwardOwnerQueryStats()
		: stamps(NULL), numParticles(0), numSurfaces(0), stamp(0),
		  queryCalls(0), uniqueQueries(0)
	{
	}

	void configure(PxArray<PxU32>& inputStamps, PxU32 inputNumParticles,
		PxU32 inputNumSurfaces, PxU32 inputStamp)
	{
		stamps = &inputStamps;
		numParticles = inputNumParticles;
		numSurfaces = inputNumSurfaces;
		stamp = inputStamp;
		queryCalls = 0;
		uniqueQueries = 0;
		PX_ASSERT(stamp > 0 &&
			PxU64(numParticles) * numSurfaces <= inputStamps.size());
	}

	PX_FORCE_INLINE void record(PxU32 surfaceIndex, PxU32 particleIndex)
	{
		++queryCalls;
		if(!stamps || surfaceIndex >= numSurfaces ||
			particleIndex >= numParticles)
			return;
		const PxU64 index64 = PxU64(surfaceIndex) * numParticles +
			particleIndex;
		PX_ASSERT(index64 < stamps->size());
		if(index64 >= stamps->size())
			return;
		PxU32& entry = (*stamps)[PxU32(index64)];
		if(entry != stamp)
		{
			entry = stamp;
			++uniqueQueries;
		}
	}
};

// P5.38 distinguishes every per-soft-feature rigid-triangle BVH query, the
// triangle leaves recovered by that BVH and the resulting active edge/vertex
// features. It is task-local and observational: the collector, candidate
// order and narrow phase remain exactly unchanged.
struct AvbdRigidTriangleSurfaceDiscreteOGCQueryStats
{
	PxU64 edgeBvhQueries;
	PxU64 edgeBvhTriangleCandidates;
	PxU64 edgeFeatureCandidates;
	PxU64 edgeFallbackQueries;
	PxU64 triangleBvhQueries;
	PxU64 triangleBvhTriangleCandidates;
	PxU64 triangleFeatureCandidates;
	PxU64 triangleFallbackQueries;

	AvbdRigidTriangleSurfaceDiscreteOGCQueryStats()
		: edgeBvhQueries(0), edgeBvhTriangleCandidates(0),
		  edgeFeatureCandidates(0), edgeFallbackQueries(0),
		  triangleBvhQueries(0), triangleBvhTriangleCandidates(0),
		  triangleFeatureCandidates(0), triangleFallbackQueries(0)
	{
	}

	PX_FORCE_INLINE void recordEdgeQuery(bool usedBvh,
		PxU32 triangleCandidates, PxU32 featureCandidates)
	{
		if(usedBvh)
		{
			++edgeBvhQueries;
			edgeBvhTriangleCandidates += triangleCandidates;
		}
		else
			++edgeFallbackQueries;
		edgeFeatureCandidates += featureCandidates;
	}

	PX_FORCE_INLINE void recordTriangleQuery(bool usedBvh,
		PxU32 triangleCandidates, PxU32 featureCandidates)
	{
		if(usedBvh)
		{
			++triangleBvhQueries;
			triangleBvhTriangleCandidates += triangleCandidates;
		}
		else
			++triangleFallbackQueries;
		triangleFeatureCandidates += featureCandidates;
	}
};

// P5.37 packs P5.32's boolean result and generation stamp into one 32-bit
// entry. Each Scene child owns its storage and advances its 31-bit generation
// every frame, so it never shares dynamic state with sibling leaves, another
// frame or the serial route.
struct AvbdRigidTriangleSurfaceForwardOwnerResultCache
{
	PxArray<PxU32>* entries;
	const PxArray<PxU32>* surfaceSlots;
	PxU32 numParticles;
	PxU32 numSurfaces;
	PxU32 numCachedSurfaces;
	PxU32 stamp;
	PxU64 hits;
	PxU64 misses;

	AvbdRigidTriangleSurfaceForwardOwnerResultCache()
		: entries(NULL), surfaceSlots(NULL), numParticles(0),
		  numSurfaces(0), numCachedSurfaces(0), stamp(0), hits(0), misses(0)
	{
	}

	void configure(PxArray<PxU32>& inputEntries,
		const PxArray<PxU32>& inputSurfaceSlots, PxU32 inputNumParticles,
		PxU32 inputNumSurfaces, PxU32 inputNumCachedSurfaces,
		PxU32 inputStamp)
	{
		entries = &inputEntries;
		surfaceSlots = &inputSurfaceSlots;
		numParticles = inputNumParticles;
		numSurfaces = inputNumSurfaces;
		numCachedSurfaces = inputNumCachedSurfaces;
		stamp = inputStamp;
		hits = 0;
		misses = 0;
		PX_ASSERT(stamp > 0 && inputSurfaceSlots.size() >= numSurfaces &&
			PxU64(numParticles) * numCachedSurfaces <= inputEntries.size());
	}

	PX_FORCE_INLINE PxU32 getSurfaceSlot(PxU32 surfaceIndex) const
	{
		if(!surfaceSlots || surfaceIndex >= numSurfaces)
			return PX_MAX_U32;
		return (*surfaceSlots)[surfaceIndex];
	}

	PX_FORCE_INLINE bool lookup(PxU32 surfaceSlot, PxU32 particleIndex,
		bool& result)
	{
		if(!entries || surfaceSlot >= numCachedSurfaces ||
			particleIndex >= numParticles)
			return false;
		const PxU64 index64 = PxU64(surfaceSlot) * numParticles +
			particleIndex;
		PX_ASSERT(index64 < entries->size());
		if(index64 >= entries->size())
			return false;
		const PxU32 index = PxU32(index64);
		const PxU32 entry = (*entries)[index];
		if((entry >> 1) != stamp)
			return false;
		result = (entry & 1u) != 0;
		++hits;
		return true;
	}

	PX_FORCE_INLINE void store(PxU32 surfaceSlot, PxU32 particleIndex,
		bool result)
	{
		if(!entries || surfaceSlot >= numCachedSurfaces ||
			particleIndex >= numParticles)
			return;
		const PxU64 index64 = PxU64(surfaceSlot) * numParticles +
			particleIndex;
		PX_ASSERT(index64 < entries->size());
		if(index64 >= entries->size())
			return;
		const PxU32 index = PxU32(index64);
		(*entries)[index] = (stamp << 1) | (result ? 1u : 0u);
		++misses;
	}
};

// Build exactly the loop identity of the serial feature suffixes without
// inspecting pose, BVH state or contact geometry.  Those dynamic predicates
// remain inside the eventual leaf so the parent IR cannot accidentally become
// a second, behavior-changing broadphase.  Empty feature families emit no row
// because they cannot emit a contact and therefore have no merge identity.
inline void avbdBuildRigidTriangleSurfaceOGCFeaturePlan(
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxU32 numSurfaces, AvbdRigidTriangleSurfaceFeaturePlan& plan,
	bool includeSwept = true, bool includeDiscrete = true)
{
	PX_ASSERT(softBodies || numSoftBodies == 0);
	plan.clear();
	if(numSurfaces == 0 || numSoftBodies == 0)
		return;

	auto appendPhase = [&] (
		AvbdRigidTriangleSurfaceFeatureWorkItem::Phase phase,
		bool speculativeOnly)
	{
		for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
		{
			const AvbdSoftBody& body = softBodies[bodyIndex];
			if(speculativeOnly && !body.compiled.speculativeCCDEnabled)
				continue;
			const PxU32 edgeCount = body.compiled.surfaceEdges.size();
			const PxU32 triangleCount =
				body.compiled.surfaceTriangles.size() / 3;
			for(PxU32 surfaceIndex = 0;
				surfaceIndex < numSurfaces; ++surfaceIndex)
			{
				if(edgeCount > 0)
					plan.items.pushBack(
						AvbdRigidTriangleSurfaceFeatureWorkItem(
							phase,
							AvbdRigidTriangleSurfaceFeatureWorkItem::
								eSOFT_EDGE,
							bodyIndex, surfaceIndex, 0, edgeCount));
				if(triangleCount > 0)
					plan.items.pushBack(
						AvbdRigidTriangleSurfaceFeatureWorkItem(
							phase,
							AvbdRigidTriangleSurfaceFeatureWorkItem::
								eSOFT_TRIANGLE,
							bodyIndex, surfaceIndex, 0, triangleCount));
			}
		}
	};

	if(includeSwept)
		appendPhase(AvbdRigidTriangleSurfaceFeatureWorkItem::eSWEPT, true);
	if(includeDiscrete)
		appendPhase(AvbdRigidTriangleSurfaceFeatureWorkItem::eDISCRETE, false);
}

// =============================================================================
// OGC (Offset Geometric Contact) -- 4-Path Collision Detection
//
// Reference: "Offset Geometric Contact", SIGGRAPH 2025
//            Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, Cem Yuksel
//
// Path 1: Rigid-Rigid -> PhysX native broadphase/narrowphase
// Path 2: Rigid-Soft -> Analytical box SDF query
// Path 3: Soft-Soft -> OGC simplified (Sec 3.9: outward offset, pure quadratic)
// Path 4: Self-collision -> OGC full (safety bubble + two-stage C2 activation)
// =============================================================================

struct AvbdOGCParams
{
	PxReal contactRadius;     // r: offset radius
	PxReal contactStiffness;  // k_c: contact stiffness
	PxReal friction;          // mu_c
	PxReal safetyRelax;       // gamma_p: safety bound relaxation (0 < gamma_p < 0.5)
	PxReal redetectRatio;     // gamma_e: redetection trigger ratio
	PxReal tau;               // activation threshold; -1 means r/2 (auto)

	AvbdOGCParams()
		: contactRadius(0.05f)
		, contactStiffness(1e5f)
		, friction(0.3f)
		, safetyRelax(0.45f)
		, redetectRatio(0.01f)
		, tau(-1.0f) {}

	PxReal getTau() const { return (tau < 0.0f) ? contactRadius * 0.5f : tau; }
};

// Optional diagnostic counters for the component collision path.  These are
// deliberately caller-owned so production callers pay no storage cost when
// profiling is disabled.
struct AvbdSoftCollisionStats
{
	PxU64 detectionCalls;
	PxU64 bodyPairs;
	PxU64 overlappingBodyPairs;
	PxU64 particleSurfaceCandidates;
	PxU64 insideTriangleTests;
	PxU64 closestTriangleTests;
	PxU64 selfTriangleTests;
	PxU64 selfTriangleBoundsBuilt;
	PxU64 selfVertexSweepEntriesBuilt;
	PxU64 selfEdgeBoundsBuilt;
	PxU64 surfaceTriangleBvhRefitNodes;
	PxU64 surfaceTriangleBvhCandidateTriangles;
	PxU64 surfaceEdgeBvhRefitNodes;
	PxU64 surfaceEdgeBvhCandidateEdges;
	PxU64 rigidParticleBoxTests;
	PxU64 rigidParticleSphereTests;
	PxU64 rigidParticleCapsuleTests;
	PxU64 rigidParticleConvexTests;
	PxU64 rigidParticleTriangleSurfaceTests;
	PxU64 rigidTriangleSurfaceFaceCandidates;
	PxU64 rigidTriangleSurfaceFaceTests;
	PxU64 rigidTriangleSurfaceEdgeCandidates;
	PxU64 rigidTriangleSurfaceEdgeTests;
	PxU64 rigidTriangleSurfaceVertexCandidates;
	PxU64 rigidTriangleSurfaceVertexTests;
	PxU64 generatedGroundContacts;
	PxU64 generatedRigidContacts;
	PxU64 generatedSoftContacts;
	PxU64 generatedSelfContacts;

	AvbdSoftCollisionStats()
		: detectionCalls(0), bodyPairs(0), overlappingBodyPairs(0),
		  particleSurfaceCandidates(0), insideTriangleTests(0),
		  closestTriangleTests(0), selfTriangleTests(0),
		  selfTriangleBoundsBuilt(0), selfVertexSweepEntriesBuilt(0),
		  selfEdgeBoundsBuilt(0),
		  surfaceTriangleBvhRefitNodes(0),
		  surfaceTriangleBvhCandidateTriangles(0),
		  surfaceEdgeBvhRefitNodes(0), surfaceEdgeBvhCandidateEdges(0),
		  rigidParticleBoxTests(0), rigidParticleSphereTests(0),
		  rigidParticleCapsuleTests(0), rigidParticleConvexTests(0),
		  rigidParticleTriangleSurfaceTests(0),
		  rigidTriangleSurfaceFaceCandidates(0),
		  rigidTriangleSurfaceFaceTests(0),
		  rigidTriangleSurfaceEdgeCandidates(0),
		  rigidTriangleSurfaceEdgeTests(0),
		  rigidTriangleSurfaceVertexCandidates(0),
		  rigidTriangleSurfaceVertexTests(0),
		  generatedGroundContacts(0),
		  generatedRigidContacts(0), generatedSoftContacts(0),
		  generatedSelfContacts(0)
	{
	}

	void accumulate(const AvbdSoftCollisionStats& other)
	{
		detectionCalls += other.detectionCalls;
		bodyPairs += other.bodyPairs;
		overlappingBodyPairs += other.overlappingBodyPairs;
		particleSurfaceCandidates += other.particleSurfaceCandidates;
		insideTriangleTests += other.insideTriangleTests;
		closestTriangleTests += other.closestTriangleTests;
		selfTriangleTests += other.selfTriangleTests;
		selfTriangleBoundsBuilt += other.selfTriangleBoundsBuilt;
		selfVertexSweepEntriesBuilt += other.selfVertexSweepEntriesBuilt;
		selfEdgeBoundsBuilt += other.selfEdgeBoundsBuilt;
		surfaceTriangleBvhRefitNodes +=
			other.surfaceTriangleBvhRefitNodes;
		surfaceTriangleBvhCandidateTriangles +=
			other.surfaceTriangleBvhCandidateTriangles;
		surfaceEdgeBvhRefitNodes += other.surfaceEdgeBvhRefitNodes;
		surfaceEdgeBvhCandidateEdges += other.surfaceEdgeBvhCandidateEdges;
		rigidParticleBoxTests += other.rigidParticleBoxTests;
		rigidParticleSphereTests += other.rigidParticleSphereTests;
		rigidParticleCapsuleTests += other.rigidParticleCapsuleTests;
		rigidParticleConvexTests += other.rigidParticleConvexTests;
		rigidParticleTriangleSurfaceTests +=
			other.rigidParticleTriangleSurfaceTests;
		rigidTriangleSurfaceFaceCandidates +=
			other.rigidTriangleSurfaceFaceCandidates;
		rigidTriangleSurfaceFaceTests +=
			other.rigidTriangleSurfaceFaceTests;
		rigidTriangleSurfaceEdgeCandidates +=
			other.rigidTriangleSurfaceEdgeCandidates;
		rigidTriangleSurfaceEdgeTests +=
			other.rigidTriangleSurfaceEdgeTests;
		rigidTriangleSurfaceVertexCandidates +=
			other.rigidTriangleSurfaceVertexCandidates;
		rigidTriangleSurfaceVertexTests +=
			other.rigidTriangleSurfaceVertexTests;
		generatedGroundContacts += other.generatedGroundContacts;
		generatedRigidContacts += other.generatedRigidContacts;
		generatedSoftContacts += other.generatedSoftContacts;
		generatedSelfContacts += other.generatedSelfContacts;
	}
};

struct AvbdClosestPointResult
{
	PxVec3              point;     // closest point on triangle
	PxVec3              barycentric;
	PxVec3              normal;    // direction from closest point to query point
	PxReal              distance;  // unsigned distance
	AvbdClosestFeature  feature;
	PxU32               featureIndex; // face=0, edges AB/AC/BC, vertices A/B/C
};

// Enhanced closest-point-on-triangle with feature classification
inline AvbdClosestPointResult avbdClosestPointOnTriangleOGC(
	const PxVec3& p, const PxVec3& a, const PxVec3& b, const PxVec3& c)
{
	AvbdClosestPointResult result;
	PxVec3 ab = b - a, ac = c - a, ap = p - a;
	PxReal d1 = ab.dot(ap), d2 = ac.dot(ap);
	if (d1 <= 0.0f && d2 <= 0.0f) {
		result.point = a; result.feature = AVBD_FEATURE_VERTEX; result.featureIndex = 0;
		result.barycentric = PxVec3(1.0f, 0.0f, 0.0f);
		PxVec3 diff = p - a; result.distance = diff.magnitude();
		result.normal = result.distance > 1e-10f ? diff * (1.0f / result.distance) : PxVec3(0,1,0);
		return result;
	}
	PxVec3 bp = p - b;
	PxReal d3 = ab.dot(bp), d4 = ac.dot(bp);
	if (d3 >= 0.0f && d4 <= d3) {
		result.point = b; result.feature = AVBD_FEATURE_VERTEX; result.featureIndex = 1;
		result.barycentric = PxVec3(0.0f, 1.0f, 0.0f);
		PxVec3 diff = p - b; result.distance = diff.magnitude();
		result.normal = result.distance > 1e-10f ? diff * (1.0f / result.distance) : PxVec3(0,1,0);
		return result;
	}
	PxReal vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
		PxReal v = d1 / (d1 - d3);
		result.point = a + ab * v; result.feature = AVBD_FEATURE_EDGE; result.featureIndex = 0;
		result.barycentric = PxVec3(1.0f - v, v, 0.0f);
		PxVec3 diff = p - result.point; result.distance = diff.magnitude();
		result.normal = result.distance > 1e-10f ? diff * (1.0f / result.distance) : PxVec3(0,1,0);
		return result;
	}
	PxVec3 cp = p - c;
	PxReal d5 = ab.dot(cp), d6 = ac.dot(cp);
	if (d6 >= 0.0f && d5 <= d6) {
		result.point = c; result.feature = AVBD_FEATURE_VERTEX; result.featureIndex = 2;
		result.barycentric = PxVec3(0.0f, 0.0f, 1.0f);
		PxVec3 diff = p - c; result.distance = diff.magnitude();
		result.normal = result.distance > 1e-10f ? diff * (1.0f / result.distance) : PxVec3(0,1,0);
		return result;
	}
	PxReal vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
		PxReal w = d2 / (d2 - d6);
		result.point = a + ac * w; result.feature = AVBD_FEATURE_EDGE; result.featureIndex = 1;
		result.barycentric = PxVec3(1.0f - w, 0.0f, w);
		PxVec3 diff = p - result.point; result.distance = diff.magnitude();
		result.normal = result.distance > 1e-10f ? diff * (1.0f / result.distance) : PxVec3(0,1,0);
		return result;
	}
	PxReal va = d3 * d6 - d5 * d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
		PxReal w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		result.point = b + (c - b) * w; result.feature = AVBD_FEATURE_EDGE; result.featureIndex = 2;
		result.barycentric = PxVec3(0.0f, 1.0f - w, w);
		PxVec3 diff = p - result.point; result.distance = diff.magnitude();
		result.normal = result.distance > 1e-10f ? diff * (1.0f / result.distance) : PxVec3(0,1,0);
		return result;
	}
	// Inside triangle
	PxReal denom = 1.0f / (va + vb + vc);
	PxReal v = vb * denom;
	PxReal w = vc * denom;
	result.point = a + ab * v + ac * w;
	result.barycentric = PxVec3(1.0f - v - w, v, w);
	result.feature = AVBD_FEATURE_FACE;
	result.featureIndex = 0;
	PxVec3 diff = p - result.point;
	result.distance = diff.magnitude();
	if (result.distance > 1e-10f)
		result.normal = diff * (1.0f / result.distance);
	else {
		PxVec3 faceN = ab.cross(ac);
		PxReal fLen = faceN.magnitude();
		result.normal = fLen > 1e-10f ? faceN * (1.0f / fLen) : PxVec3(0,1,0);
	}
	return result;
}

PX_FORCE_INLINE PxReal avbdGetRestPointTriangleDistance(
	const PxVec3& point, const PxVec3& vertex0,
	const PxVec3& vertex1, const PxVec3& vertex2)
{
	return avbdClosestPointOnTriangleOGC(
		point, vertex0, vertex1, vertex2).distance;
}

// =============================================================================
// Two-stage C2 activation function (OGC Eq. 18-20)
// =============================================================================

struct AvbdActivationResult
{
	PxReal energy;
	PxReal force;    // -dg/dd (positive = repulsive)
	PxReal hessian;  // d2g/dd2
};

PX_FORCE_INLINE AvbdActivationResult avbdOGCActivationQuadratic(PxReal d, PxReal r, PxReal kc)
{
	PxReal pen = r - d;
	AvbdActivationResult res;
	res.energy  = 0.5f * kc * pen * pen;
	res.force   = kc * pen;
	res.hessian = kc;
	return res;
}

PX_FORCE_INLINE AvbdActivationResult avbdOGCActivationFull(PxReal d, PxReal r, PxReal kc, PxReal tau)
{
	AvbdActivationResult res;
	if (d >= r) {
		res.energy = 0.0f; res.force = 0.0f; res.hessian = 0.0f;
	} else if (d >= tau) {
		PxReal pen = r - d;
		res.energy  = 0.5f * kc * pen * pen;
		res.force   = kc * pen;
		res.hessian = kc;
	} else if (d > 1e-10f) {
		PxReal rmt = r - tau;
		PxReal kc_prime = tau * kc * rmt * rmt;
		PxReal b = 0.5f * kc * rmt * rmt + kc_prime * PxLog(tau);
		res.energy  = -kc_prime * PxLog(d) + b;
		res.force   = kc_prime / d;
		res.hessian = kc_prime / (d * d);
	} else {
		PxReal rmt = r - tau;
		PxReal kc_prime = tau * kc * rmt * rmt;
		PxReal d_clamp = 1e-10f;
		res.energy  = kc_prime * 10.0f;
		res.force   = kc_prime / d_clamp;
		res.hessian = kc_prime / (d_clamp * d_clamp);
	}
	return res;
}

// =============================================================================
// Helper: point-inside-tet-mesh via Moller-Trumbore ray casting (parity)
// =============================================================================

inline bool avbdIsPointInsideTetMesh(
	const PxVec3& point,
	const PxArray<PxU32>& surfaceTriangles,
	const AvbdSoftParticle* particles,
	AvbdSoftCollisionStats* stats = NULL)
{
	int crossings = 0;
	PxVec3 rayDir(0.0f, 1.0f, 0.0f);
	for (PxU32 ti = 0; ti + 2 < surfaceTriangles.size(); ti += 3)
	{
		if(stats)
			stats->insideTriangleTests++;
		const PxVec3& a = particles[surfaceTriangles[ti]].position;
		const PxVec3& b = particles[surfaceTriangles[ti+1]].position;
		const PxVec3& c = particles[surfaceTriangles[ti+2]].position;
		PxVec3 e1 = b - a, e2 = c - a;
		PxVec3 h = rayDir.cross(e2);
		PxReal det = e1.dot(h);
		if (PxAbs(det) < 1e-10f) continue;
		PxReal invDet = 1.0f / det;
		PxVec3 s = point - a;
		PxReal u = invDet * s.dot(h);
		if (u < 0.0f || u > 1.0f) continue;
		PxVec3 q = s.cross(e1);
		PxReal v = invDet * rayDir.dot(q);
		if (v < 0.0f || u + v > 1.0f) continue;
		PxReal t = invDet * e2.dot(q);
		if (t > 1e-6f) crossings++;
	}
	return (crossings & 1) != 0;
}

// =============================================================================
// PATH 2 (OGC): Analytical SDF Rigid-Soft Contact
// =============================================================================

PX_FORCE_INLINE PxVec3 avbdGetRigidBoxFaceNormal(
	const PxVec3& localNormal)
{
	const PxVec3 absNormal(
		PxAbs(localNormal.x),
		PxAbs(localNormal.y),
		PxAbs(localNormal.z));
	if(absNormal.x >= absNormal.y && absNormal.x >= absNormal.z)
		return PxVec3(
			localNormal.x >= 0.0f ? 1.0f : -1.0f, 0.0f, 0.0f);
	if(absNormal.y >= absNormal.z)
		return PxVec3(
			0.0f, localNormal.y >= 0.0f ? 1.0f : -1.0f, 0.0f);
	return PxVec3(
		0.0f, 0.0f, localNormal.z >= 0.0f ? 1.0f : -1.0f);
}

PX_FORCE_INLINE PxU64 avbdGetRigidBoxFaceFeatureKey(
	const PxVec3& faceNormal)
{
	if(PxAbs(faceNormal.x) > 0.5f)
		return faceNormal.x > 0.0f ? 1u : 2u;
	if(PxAbs(faceNormal.y) > 0.5f)
		return faceNormal.y > 0.0f ? 3u : 4u;
	return faceNormal.z > 0.0f ? 5u : 6u;
}

PX_FORCE_INLINE PxU64 avbdGetRigidSoftFeatureKey(
	PxU32 tag, PxU32 vertex0, PxU32 vertex1,
	PxU32 vertex2, PxU32 rigidFeatureIndex)
{
	PxU64 hash = 1469598103934665603ull;
	hash = avbdSoftContactHashValue(hash, tag);
	hash = avbdSoftContactHashValue(hash, vertex0);
	hash = avbdSoftContactHashValue(hash, vertex1);
	hash = avbdSoftContactHashValue(hash, vertex2);
	return avbdSoftContactHashValue(hash, rigidFeatureIndex);
}

PX_FORCE_INLINE void avbdClosestPointsOnSegments(
	const PxVec3& p0, const PxVec3& p1,
	const PxVec3& q0, const PxVec3& q1,
	PxReal& pWeight1, PxReal& qWeight1,
	PxVec3& pClosest, PxVec3& qClosest)
{
	const PxVec3 dP = p1 - p0;
	const PxVec3 dQ = q1 - q0;
	const PxVec3 r = p0 - q0;
	const PxReal a = dP.dot(dP);
	const PxReal e = dQ.dot(dQ);
	const PxReal epsilon = 1e-12f;

	if(a <= epsilon && e <= epsilon)
	{
		pWeight1 = qWeight1 = 0.0f;
		pClosest = p0;
		qClosest = q0;
		return;
	}
	if(a <= epsilon)
	{
		pWeight1 = 0.0f;
		qWeight1 = PxClamp(dQ.dot(-r) / e, 0.0f, 1.0f);
	}
	else
	{
		const PxReal c = dP.dot(r);
		if(e <= epsilon)
		{
			qWeight1 = 0.0f;
			pWeight1 = PxClamp(-c / a, 0.0f, 1.0f);
		}
		else
		{
			const PxReal b = dP.dot(dQ);
			const PxReal f = dQ.dot(r);
			const PxReal denominator = a * e - b * b;
			pWeight1 = denominator > epsilon
				? PxClamp((b * f - c * e) / denominator, 0.0f, 1.0f)
				: 0.0f;
			qWeight1 = (b * pWeight1 + f) / e;
			if(qWeight1 < 0.0f)
			{
				qWeight1 = 0.0f;
				pWeight1 = PxClamp(-c / a, 0.0f, 1.0f);
			}
			else if(qWeight1 > 1.0f)
			{
				qWeight1 = 1.0f;
				pWeight1 = PxClamp((b - c) / a, 0.0f, 1.0f);
			}
		}
	}
	pClosest = p0 + dP * pWeight1;
	qClosest = q0 + dQ * qWeight1;
}

struct AvbdClosestSegmentTriangleResult
{
	PxVec3 segmentPoint;
	PxVec3 trianglePoint;
	PxVec3 barycentric;
	PxReal segmentWeight1;
	PxReal distance;
	AvbdClosestFeature feature;
	PxU32 featureIndex;
};

PX_FORCE_INLINE void avbdUpdateClosestSegmentTriangle(
	AvbdClosestSegmentTriangleResult& result,
	const PxVec3& segmentPoint, const PxVec3& trianglePoint,
	const PxVec3& barycentric, PxReal segmentWeight1,
	AvbdClosestFeature feature, PxU32 featureIndex)
{
	const PxReal distance =
		(segmentPoint - trianglePoint).magnitude();
	if(PxIsFinite(distance) && distance < result.distance)
	{
		result.segmentPoint = segmentPoint;
		result.trianglePoint = trianglePoint;
		result.barycentric = barycentric;
		result.segmentWeight1 = segmentWeight1;
		result.distance = distance;
		result.feature = feature;
		result.featureIndex = featureIndex;
	}
}

// Complete segment/triangle closest query used by the capsule reverse OGC
// path. Endpoint/triangle candidates own cap features, segment/edge candidates
// own side features, and the explicit plane crossing covers a medial segment
// passing through a triangle interior.
PX_FORCE_INLINE AvbdClosestSegmentTriangleResult
avbdClosestSegmentTriangleOGC(
	const PxVec3& segment0, const PxVec3& segment1,
	const PxVec3& a, const PxVec3& b, const PxVec3& c)
{
	AvbdClosestSegmentTriangleResult result;
	result.segmentPoint = segment0;
	result.trianglePoint = a;
	result.barycentric = PxVec3(1.0f, 0.0f, 0.0f);
	result.segmentWeight1 = 0.0f;
	result.distance = PX_MAX_F32;
	result.feature = AVBD_FEATURE_UNKNOWN;
	result.featureIndex = 0;

	const PxVec3 segmentDirection = segment1 - segment0;
	const PxVec3 triangleNormal = (b - a).cross(c - a);
	const PxReal normalMagnitudeSq =
		triangleNormal.magnitudeSquared();
	const PxReal planeDenominator =
		triangleNormal.dot(segmentDirection);
	if(normalMagnitudeSq > 1.0e-16f &&
		PxAbs(planeDenominator) > 1.0e-12f)
	{
		const PxReal segmentWeight =
			triangleNormal.dot(a - segment0) / planeDenominator;
		if(segmentWeight >= 0.0f && segmentWeight <= 1.0f)
		{
			const PxVec3 planePoint =
				segment0 + segmentDirection * segmentWeight;
			const AvbdClosestPointResult planeClosest =
				avbdClosestPointOnTriangleOGC(
					planePoint, a, b, c);
			if(planeClosest.distance <= 1.0e-6f)
				avbdUpdateClosestSegmentTriangle(
					result, planePoint, planeClosest.point,
					planeClosest.barycentric, segmentWeight,
					planeClosest.feature,
					planeClosest.featureIndex);
		}
	}

	const PxVec3 endpoints[2] = {segment0, segment1};
	for(PxU32 endpoint = 0; endpoint < 2; ++endpoint)
	{
		const AvbdClosestPointResult closest =
			avbdClosestPointOnTriangleOGC(
				endpoints[endpoint], a, b, c);
		avbdUpdateClosestSegmentTriangle(
			result, endpoints[endpoint], closest.point,
			closest.barycentric, PxReal(endpoint),
			closest.feature, closest.featureIndex);
	}

	const PxVec3 edge0[3] = {a, a, b};
	const PxVec3 edge1[3] = {b, c, c};
	for(PxU32 edge = 0; edge < 3; ++edge)
	{
		PxReal segmentWeight = 0.0f;
		PxReal edgeWeight = 0.0f;
		PxVec3 segmentClosest;
		PxVec3 edgeClosest;
		avbdClosestPointsOnSegments(
			segment0, segment1, edge0[edge], edge1[edge],
			segmentWeight, edgeWeight,
			segmentClosest, edgeClosest);
		PxVec3 barycentric(0.0f);
		if(edge == 0)
			barycentric = PxVec3(
				1.0f - edgeWeight, edgeWeight, 0.0f);
		else if(edge == 1)
			barycentric = PxVec3(
				1.0f - edgeWeight, 0.0f, edgeWeight);
		else
			barycentric = PxVec3(
				0.0f, 1.0f - edgeWeight, edgeWeight);
		AvbdClosestFeature feature = AVBD_FEATURE_EDGE;
		PxU32 featureIndex = edge;
		if(edgeWeight <= 1.0e-5f ||
			edgeWeight >= 1.0f - 1.0e-5f)
		{
			feature = AVBD_FEATURE_VERTEX;
			featureIndex = edgeWeight <= 1.0e-5f
				? (edge == 2 ? 1u : 0u)
				: (edge == 0 ? 1u : 2u);
		}
		avbdUpdateClosestSegmentTriangle(
			result, segmentClosest, edgeClosest,
			barycentric, segmentWeight,
			feature, featureIndex);
	}
	return result;
}

PX_FORCE_INLINE void avbdGetRigidBoxEdgeLocal(
	const PxVec3& halfExtent, PxU32 edgeIndex,
	PxVec3& endpoint0, PxVec3& endpoint1,
	PxVec3& outwardNormal)
{
	const PxU32 axis = edgeIndex / 4;
	const PxU32 variant = edgeIndex & 3;
	const PxReal sign0 = (variant & 1) ? 1.0f : -1.0f;
	const PxReal sign1 = (variant & 2) ? 1.0f : -1.0f;
	endpoint0 = endpoint1 = PxVec3(0.0f);
	outwardNormal = PxVec3(0.0f);

	if(axis == 0)
	{
		endpoint0 = PxVec3(
			-halfExtent.x, sign0 * halfExtent.y,
			sign1 * halfExtent.z);
		endpoint1 = PxVec3(
			halfExtent.x, sign0 * halfExtent.y,
			sign1 * halfExtent.z);
		outwardNormal = PxVec3(0.0f, sign0, sign1);
	}
	else if(axis == 1)
	{
		endpoint0 = PxVec3(
			sign0 * halfExtent.x, -halfExtent.y,
			sign1 * halfExtent.z);
		endpoint1 = PxVec3(
			sign0 * halfExtent.x, halfExtent.y,
			sign1 * halfExtent.z);
		outwardNormal = PxVec3(sign0, 0.0f, sign1);
	}
	else
	{
		endpoint0 = PxVec3(
			sign0 * halfExtent.x, sign1 * halfExtent.y,
			-halfExtent.z);
		endpoint1 = PxVec3(
			sign0 * halfExtent.x, sign1 * halfExtent.y,
			halfExtent.z);
		outwardNormal = PxVec3(sign0, sign1, 0.0f);
	}
	outwardNormal.normalize();
}

PX_FORCE_INLINE PxVec3 avbdGetRigidBoxVertexLocal(
	const PxVec3& halfExtent, PxU32 vertexIndex)
{
	return PxVec3(
		(vertexIndex & 1) ? halfExtent.x : -halfExtent.x,
		(vertexIndex & 2) ? halfExtent.y : -halfExtent.y,
		(vertexIndex & 4) ? halfExtent.z : -halfExtent.z);
}

PX_FORCE_INLINE bool avbdSegmentEnterExpandedBox(
	const PxVec3& segmentStart, const PxVec3& segmentEnd,
	const PxVec3& expandedHalfExtent,
	PxReal& entryTime, PxVec3& entryNormal)
{
	const PxVec3 direction = segmentEnd - segmentStart;
	entryTime = 0.0f;
	PxReal exitTime = 1.0f;
	entryNormal = PxVec3(0.0f);
	for(PxU32 axis = 0; axis < 3; axis++)
	{
		if(PxAbs(direction[axis]) <= 1e-12f)
		{
			if(segmentStart[axis] < -expandedHalfExtent[axis] ||
				segmentStart[axis] > expandedHalfExtent[axis])
				return false;
			continue;
		}
		const PxReal inverseDirection = 1.0f / direction[axis];
		PxReal nearTime =
			(-expandedHalfExtent[axis] - segmentStart[axis]) *
			inverseDirection;
		PxReal farTime =
			(expandedHalfExtent[axis] - segmentStart[axis]) *
			inverseDirection;
		PxReal nearNormalSign = -1.0f;
		if(nearTime > farTime)
		{
			const PxReal swapTime = nearTime;
			nearTime = farTime;
			farTime = swapTime;
			nearNormalSign = 1.0f;
		}
		if(nearTime > entryTime)
		{
			entryTime = nearTime;
			entryNormal = PxVec3(0.0f);
			entryNormal[axis] = nearNormalSign;
		}
		exitTime = PxMin(exitTime, farTime);
		if(entryTime > exitTime)
			return false;
	}
	return entryTime >= 0.0f && entryTime <= 1.0f &&
		entryNormal.magnitudeSquared() > 0.5f;
}

PX_FORCE_INLINE bool avbdFindPreviousRigidBoxFace(
	const AvbdSoftContact* previousContacts,
	PxU32 numPreviousContacts,
	PxU32 particleIndex,
	const AvbdRigidBox& box,
	PxVec3& localFaceNormal)
{
	for(PxU32 contactIndex = 0;
		contactIndex < numPreviousContacts; contactIndex++)
	{
		const AvbdSoftContactGeometry& geometry =
			previousContacts[contactIndex].geometry;
		if(geometry.particleIdx != particleIndex ||
			geometry.targetKind != box.targetKind ||
			geometry.source.type !=
				AvbdSoftContactSource::eRIGID_SDF ||
			geometry.source.primitiveKey != box.primitiveKey ||
			geometry.source.featureKey < 1u ||
			geometry.source.featureKey > 6u)
			continue;
		if(box.targetKind ==
				AvbdSoftContactTargetKind::eRIGID_BODY &&
			geometry.targetIndex != box.targetIndex)
			continue;
		const PxVec3 candidate =
			box.rotation.getConjugate().rotate(geometry.normal);
		if(!candidate.isFinite() ||
			candidate.magnitudeSquared() < 0.25f)
			continue;
		localFaceNormal = avbdGetRigidBoxFaceNormal(candidate);
		return true;
	}
	return false;
}

// P5.4a candidate leaf: a worker owns one canonical particle interval and
// only appends the current-pose box SDF contacts to its private stream. It
// reads the immutable previous-contact snapshot for inside-face continuity;
// swept SDF and OGC feature passes retain their existing parent order.
inline void avbdDetectSoftRigidSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidBox* boxes, PxU32 numBoxes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftContact* previousContacts = NULL,
	PxU32 numPreviousContacts = 0,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	PX_ASSERT(particleBegin <= particleEnd && particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for (PxU32 pi = particleBegin; pi < particleEnd; pi++)
	{
		if (particles[pi].invMass <= 0.0f) continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies, pi);
		if(sourceBody &&
			!avbdIsSoftBodySurfaceVertex(*sourceBody, pi))
			continue;
		const PxVec3& pp = particles[pi].position;

		for (PxU32 bi = 0; bi < numBoxes; bi++)
		{
			const AvbdRigidBox& box = boxes[bi];
			PxVec3 he = box.halfExtent;
			if (he.x <= 0.0f && he.y <= 0.0f && he.z <= 0.0f) continue;

			// Broadphase AABB
			PxReal maxExt = PxSqrt(he.x*he.x + he.y*he.y + he.z*he.z) + margin;
			PxVec3 bMin = box.center - PxVec3(maxExt);
			PxVec3 bMax = box.center + PxVec3(maxExt);
			if (pp.x < bMin.x || pp.x > bMax.x ||
				pp.y < bMin.y || pp.y > bMax.y ||
				pp.z < bMin.z || pp.z > bMax.z) continue;

			PxVec3 localP = box.rotation.getConjugate().rotate(pp - box.center);

			// Analytical box SDF
			PxVec3 q(PxAbs(localP.x) - he.x,
			         PxAbs(localP.y) - he.y,
			         PxAbs(localP.z) - he.z);

			bool inside = (q.x <= 0.0f && q.y <= 0.0f && q.z <= 0.0f);
			PxReal sdf;
			PxVec3 localNormal;
			PxU64 featureKey = 0;

			if (inside) {
				if(avbdFindPreviousRigidBoxFace(
					previousContacts, numPreviousContacts,
					pi, box, localNormal))
				{
					const PxVec3 signedPosition(
						localNormal.x * localP.x,
						localNormal.y * localP.y,
						localNormal.z * localP.z);
					const PxVec3 selectedExtent(
						PxAbs(localNormal.x) * he.x,
						PxAbs(localNormal.y) * he.y,
						PxAbs(localNormal.z) * he.z);
					sdf =
						signedPosition.x + signedPosition.y +
						signedPosition.z -
						selectedExtent.x - selectedExtent.y -
						selectedExtent.z;
				}
				else
				{
					sdf = PxMax(q.x, PxMax(q.y, q.z));
					if (q.x > q.y && q.x > q.z)
						localNormal = PxVec3(
							localP.x > 0 ? 1.0f : -1.0f, 0, 0);
					else if (q.y > q.z)
						localNormal = PxVec3(
							0, localP.y > 0 ? 1.0f : -1.0f, 0);
					else
						localNormal = PxVec3(
							0, 0, localP.z > 0 ? 1.0f : -1.0f);
				}
				featureKey =
					avbdGetRigidBoxFaceFeatureKey(localNormal);
			} else {
				PxVec3 clamped(PxMax(q.x, 0.0f), PxMax(q.y, 0.0f), PxMax(q.z, 0.0f));
				sdf = clamped.magnitude();
				if (sdf > 1e-10f)
				{
					localNormal = PxVec3(
						(localP.x >= 0.0f ? 1.0f : -1.0f) * clamped.x,
						(localP.y >= 0.0f ? 1.0f : -1.0f) * clamped.y,
						(localP.z >= 0.0f ? 1.0f : -1.0f) * clamped.z) * (1.0f / sdf);
				}
				else
					localNormal = PxVec3(0, 1, 0);
				featureKey = avbdGetRigidBoxFaceFeatureKey(
					avbdGetRigidBoxFaceNormal(localNormal));
			}

			if (sdf >= margin) continue;

			PxReal depth = inside ? -sdf : PxMax(0.0f, margin - sdf);
			PxVec3 worldNormal = box.rotation.rotate(localNormal).getNormalized();

			// Surface point on box
			PxVec3 surfaceLocal = localP;
			if (inside)
				surfaceLocal = localP - localNormal * sdf;
			else
			{
				surfaceLocal.x = PxClamp(localP.x, -he.x, he.x);
				surfaceLocal.y = PxClamp(localP.y, -he.y, he.y);
				surfaceLocal.z = PxClamp(localP.z, -he.z, he.z);
			}
			PxVec3 worldSurf = box.center + box.rotation.rotate(surfaceLocal);

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF, PX_MAX_U32,
				box.primitiveKey, featureKey);
			geometry.particleIdx  = pi;
			geometry.targetKind = box.targetKind;
			geometry.velocityOwner =
				box.targetKind ==
					AvbdSoftContactTargetKind::eKINEMATIC_RIGID
					? AvbdVelocityObjectiveOwner::
						ComponentFinalize
					: box.targetKind ==
						AvbdSoftContactTargetKind::eRIGID_BODY
						? AvbdVelocityObjectiveOwner::
							ManifoldFinalize
						: AvbdVelocityObjectiveOwner::
							PositionAL;
			geometry.targetIndex =
				box.targetKind ==
					AvbdSoftContactTargetKind::eRIGID_BODY
				? box.targetIndex : bi;
			geometry.normal       = worldNormal;
			geometry.projNormal   = worldNormal;
			geometry.depth        = depth;
			geometry.margin       = margin;
			geometry.surfacePoint = worldSurf;
			geometry.hasRigidBoxSdf = true;
			geometry.rigidBoxHalfExtent = he;
			geometry.rigidBoxPose =
				box.targetKind ==
					AvbdSoftContactTargetKind::eRIGID_BODY
					? box.shapeToRigidBody
					: PxTransform(box.center, box.rotation);
			geometry.kinematicSurfacePointPrevious =
				box.targetKind ==
					AvbdSoftContactTargetKind::eKINEMATIC_RIGID
				? box.previousCenter +
					box.previousRotation.rotate(surfaceLocal)
				: worldSurf;
			geometry.friction = sourceBody
				? avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction,
					box.friction, box.frictionCombineMode)
				: PxMax(box.friction, 0.0f);
			if(box.targetKind ==
				AvbdSoftContactTargetKind::eRIGID_BODY)
			{
				geometry.rigidLocalPoint =
					box.shapeToRigidBody.transform(surfaceLocal);
			}
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1e5f, 1e6f, particles, contacts);
		}
	}
}

inline void avbdDetectSoftRigidSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidBox* boxes, PxU32 numBoxes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftContact* previousContacts = NULL,
	PxU32 numPreviousContacts = 0,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	avbdDetectSoftRigidSDFRange(
		particles, numParticles, 0, numParticles, boxes, numBoxes,
		contacts, margin, previousContacts, numPreviousContacts,
		softBodies, numSoftBodies);
}

// P5.12a candidate leaf: swept box SDF is particle-major and has no mutable
// query state. A child owns one particle interval and appends only to its
// private stream. The caller must complete the current-SDF family before it
// stable-merges any swept-SDF family ranges.
inline void avbdDetectSoftRigidSweptSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidBox* boxes, PxU32 numBoxes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	PX_ASSERT(particleBegin <= particleEnd && particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for(PxU32 particleIndex = particleBegin;
		particleIndex < particleEnd; particleIndex++)
	{
		const AvbdSoftParticle& particle = particles[particleIndex];
		if(particle.invMass <= 0.0f ||
			!particle.predictedPosition.isFinite())
			continue;
		const PxVec3 displacement =
			particle.predictedPosition - particle.position;
		if(displacement.magnitudeSquared() <= 1e-12f)
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies, particleIndex);
		if(sourceBody &&
			!sourceBody->compiled.speculativeCCDEnabled)
			continue;
		if(sourceBody &&
			!avbdIsSoftBodySurfaceVertex(
				*sourceBody, particleIndex))
			continue;

		for(PxU32 boxIndex = 0; boxIndex < numBoxes; boxIndex++)
		{
			const AvbdRigidBox& box = boxes[boxIndex];
			const PxQuat inverseRotation =
				box.rotation.getConjugate();
			const PxVec3 startLocal = inverseRotation.rotate(
				particle.position - box.center);
			const PxVec3 predictedLocal = inverseRotation.rotate(
				particle.predictedPosition - box.center);

			const PxVec3 currentQ(
				PxAbs(startLocal.x) - box.halfExtent.x,
				PxAbs(startLocal.y) - box.halfExtent.y,
				PxAbs(startLocal.z) - box.halfExtent.z);
			const bool currentInside =
				currentQ.x <= 0.0f &&
				currentQ.y <= 0.0f &&
				currentQ.z <= 0.0f;
			const PxReal currentSdf = currentInside
				? PxMax(currentQ.x, PxMax(currentQ.y, currentQ.z))
				: PxVec3(
					PxMax(currentQ.x, 0.0f),
					PxMax(currentQ.y, 0.0f),
					PxMax(currentQ.z, 0.0f)).magnitude();
			if(currentSdf < margin)
				continue;

			PxReal entryTime = 0.0f;
			PxVec3 entryNormalLocal(0.0f);
			if(!avbdSegmentEnterExpandedBox(
					startLocal, predictedLocal,
					box.halfExtent + PxVec3(margin),
					entryTime, entryNormalLocal))
				continue;

			const PxVec3 expandedEntryLocal =
				startLocal +
				(predictedLocal - startLocal) * entryTime;
			PxVec3 surfaceLocal =
				expandedEntryLocal - entryNormalLocal * margin;
			surfaceLocal.x = PxClamp(
				surfaceLocal.x,
				-box.halfExtent.x, box.halfExtent.x);
			surfaceLocal.y = PxClamp(
				surfaceLocal.y,
				-box.halfExtent.y, box.halfExtent.y);
			surfaceLocal.z = PxClamp(
				surfaceLocal.z,
				-box.halfExtent.z, box.halfExtent.z);

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF,
				PX_MAX_U32, box.primitiveKey,
				avbdGetRigidBoxFaceFeatureKey(entryNormalLocal));
			geometry.particleIdx = particleIndex;
			geometry.targetKind = box.targetKind;
			geometry.velocityOwner =
				box.targetKind ==
					AvbdSoftContactTargetKind::eKINEMATIC_RIGID
					? AvbdVelocityObjectiveOwner::ComponentFinalize
					: box.targetKind ==
						AvbdSoftContactTargetKind::eRIGID_BODY
						? AvbdVelocityObjectiveOwner::ManifoldFinalize
						: AvbdVelocityObjectiveOwner::PositionAL;
			geometry.targetIndex =
				box.targetKind ==
					AvbdSoftContactTargetKind::eRIGID_BODY
					? box.targetIndex : boxIndex;
			geometry.normal =
				box.rotation.rotate(entryNormalLocal).getNormalized();
			geometry.projNormal = geometry.normal;
			geometry.depth = 0.0f;
			geometry.margin = margin;
			geometry.surfacePoint =
				box.center + box.rotation.rotate(surfaceLocal);
			geometry.kinematicSurfacePointPrevious =
				box.targetKind ==
					AvbdSoftContactTargetKind::eKINEMATIC_RIGID
					? box.previousCenter +
						box.previousRotation.rotate(surfaceLocal)
					: geometry.surfacePoint;
			geometry.friction = sourceBody
				? avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction,
					box.friction, box.frictionCombineMode)
				: PxMax(box.friction, 0.0f);
			if(box.targetKind ==
				AvbdSoftContactTargetKind::eRIGID_BODY)
				geometry.rigidLocalPoint =
					box.shapeToRigidBody.transform(surfaceLocal);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1e5f, 1e6f,
				particles, contacts);
		}
	}
}

inline void avbdDetectSoftRigidSweptSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidBox* boxes, PxU32 numBoxes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	avbdDetectSoftRigidSweptSDFRange(
		particles, numParticles, 0, numParticles, boxes, numBoxes, contacts,
		margin, softBodies, numSoftBodies);
}

PX_FORCE_INLINE PxVec3 avbdGetRigidSphereNormal(
	const PxVec3& offset, const AvbdSoftParticle& particle)
{
	const PxReal offsetMagnitudeSq = offset.magnitudeSquared();
	if(offsetMagnitudeSq > 1e-12f && PxIsFinite(offsetMagnitudeSq))
		return offset * PxRecipSqrt(offsetMagnitudeSq);
	const PxVec3 initialOffset =
		particle.initialPosition - particle.position;
	const PxReal initialMagnitudeSq =
		initialOffset.magnitudeSquared();
	if(initialMagnitudeSq > 1e-12f &&
		PxIsFinite(initialMagnitudeSq))
		return initialOffset * PxRecipSqrt(initialMagnitudeSq);
	return PxVec3(0.0f, 1.0f, 0.0f);
}

PX_FORCE_INLINE void avbdConfigureRigidSphereTarget(
	AvbdSoftContactGeometry& geometry,
	const AvbdRigidSphere& sphere, PxU32 sphereIndex,
	const PxVec3& surfaceLocal)
{
	geometry.targetKind = sphere.targetKind;
	geometry.velocityOwner =
		sphere.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID
			? AvbdVelocityObjectiveOwner::ComponentFinalize
			: sphere.targetKind ==
				AvbdSoftContactTargetKind::eRIGID_BODY
				? AvbdVelocityObjectiveOwner::ManifoldFinalize
				: AvbdVelocityObjectiveOwner::PositionAL;
	geometry.targetIndex =
		sphere.targetKind ==
			AvbdSoftContactTargetKind::eRIGID_BODY
			? sphere.targetIndex : sphereIndex;
	geometry.surfacePoint =
		sphere.center + sphere.rotation.rotate(surfaceLocal);
	geometry.kinematicSurfacePointPrevious =
		sphere.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID
			? sphere.previousCenter +
				sphere.previousRotation.rotate(surfaceLocal)
			: geometry.surfacePoint;
	if(sphere.targetKind ==
		AvbdSoftContactTargetKind::eRIGID_BODY)
		geometry.rigidLocalPoint =
			sphere.shapeToRigidBody.transform(surfaceLocal);
}

// P5.5a candidate leaf: one worker owns a canonical particle interval and
// appends only the current-pose sphere SDF contacts to its private stream.
// Swept SDF and feature passes retain their parent order, exactly as the
// static-box transaction does.
inline void avbdDetectSoftRigidSphereSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	PX_ASSERT(particleBegin <= particleEnd && particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for(PxU32 particleIndex = particleBegin;
		particleIndex < particleEnd; ++particleIndex)
	{
		const AvbdSoftParticle& particle = particles[particleIndex];
		if(particle.invMass <= 0.0f)
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies, particleIndex);
		if(sourceBody &&
			!avbdIsSoftBodySurfaceVertex(
				*sourceBody, particleIndex))
			continue;

		for(PxU32 sphereIndex = 0;
			sphereIndex < numSpheres; ++sphereIndex)
		{
			const AvbdRigidSphere& sphere = spheres[sphereIndex];
			if(sphere.radius <= 0.0f ||
				!PxIsFinite(sphere.radius) ||
				!sphere.center.isFinite())
				continue;
			const PxVec3 offset =
				particle.position - sphere.center;
			const PxReal distanceSq = offset.magnitudeSquared();
			const PxReal queryRadius = sphere.radius + margin;
			if(!PxIsFinite(distanceSq) ||
				distanceSq >= queryRadius * queryRadius)
				continue;
			const PxReal distance = PxSqrt(PxMax(distanceSq, 0.0f));
			const PxVec3 normal =
				avbdGetRigidSphereNormal(offset, particle);
			const PxReal sdf = distance - sphere.radius;

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF,
				PX_MAX_U32, sphere.primitiveKey, 1u);
			geometry.particleIdx = particleIndex;
			geometry.normal = normal;
			geometry.projNormal = normal;
			geometry.depth = sdf < 0.0f
				? -sdf : PxMax(0.0f, margin - sdf);
			geometry.margin = margin;
			const PxVec3 surfaceLocal =
				sphere.rotation.getConjugate().rotate(
					normal * sphere.radius);
			avbdConfigureRigidSphereTarget(
				geometry, sphere, sphereIndex, surfaceLocal);
			geometry.friction = sourceBody
				? avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction,
					sphere.friction,
					sphere.frictionCombineMode)
				: PxMax(sphere.friction, 0.0f);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1e5f, 1e6f,
				particles, contacts);
		}
	}
}

inline void avbdDetectSoftRigidSphereSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	avbdDetectSoftRigidSphereSDFRange(
		particles, numParticles, 0, numParticles,
		spheres, numSpheres, contacts, margin,
		softBodies, numSoftBodies);
}

PX_FORCE_INLINE bool avbdSegmentEnterExpandedSphere(
	const PxVec3& segmentStart,
	const PxVec3& segmentEnd,
	const PxVec3& sphereCenter,
	PxReal expandedRadius,
	PxReal& entryTime,
	PxVec3& entryNormal)
{
	const PxVec3 direction = segmentEnd - segmentStart;
	const PxReal directionMagnitudeSq =
		direction.magnitudeSquared();
	if(directionMagnitudeSq <= 1e-12f ||
		!PxIsFinite(directionMagnitudeSq) ||
		expandedRadius <= 0.0f ||
		!PxIsFinite(expandedRadius))
		return false;
	const PxVec3 startOffset = segmentStart - sphereCenter;
	const PxReal halfB = startOffset.dot(direction);
	const PxReal c =
		startOffset.magnitudeSquared() -
			expandedRadius * expandedRadius;
	if(c < 0.0f)
		return false;
	const PxReal discriminant =
		halfB * halfB - directionMagnitudeSq * c;
	if(discriminant < 0.0f || !PxIsFinite(discriminant))
		return false;
	entryTime =
		(-halfB - PxSqrt(discriminant)) /
			directionMagnitudeSq;
	if(entryTime < 0.0f || entryTime > 1.0f)
		return false;
	const PxVec3 entryOffset =
		startOffset + direction * entryTime;
	const PxReal entryMagnitudeSq =
		entryOffset.magnitudeSquared();
	if(entryMagnitudeSq <= 1e-12f ||
		!PxIsFinite(entryMagnitudeSq))
		return false;
	entryNormal =
		entryOffset * PxRecipSqrt(entryMagnitudeSq);
	return true;
}

// P5.13a candidate leaf: swept sphere SDF is particle-major and retains no
// mutable query state. A caller that partitions it must complete the entire
// current-SDF family before stable-merging any swept-family private ranges.
inline void avbdDetectSoftRigidSphereSweptSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	PX_ASSERT(particleBegin <= particleEnd && particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for(PxU32 particleIndex = particleBegin;
		particleIndex < particleEnd; ++particleIndex)
	{
		const AvbdSoftParticle& particle = particles[particleIndex];
		if(particle.invMass <= 0.0f ||
			!particle.predictedPosition.isFinite())
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies, particleIndex);
		if(!sourceBody ||
			!sourceBody->compiled.speculativeCCDEnabled ||
			!avbdIsSoftBodySurfaceVertex(
				*sourceBody, particleIndex))
			continue;
		for(PxU32 sphereIndex = 0;
			sphereIndex < numSpheres; ++sphereIndex)
		{
			const AvbdRigidSphere& sphere = spheres[sphereIndex];
			const bool dynamicTarget =
				sphere.targetKind ==
					AvbdSoftContactTargetKind::eRIGID_BODY;
			if(sphere.radius <= 0.0f ||
				!PxIsFinite(sphere.radius) ||
				!sphere.center.isFinite() ||
				!sphere.rotation.isFinite() ||
				(sphere.targetKind !=
						AvbdSoftContactTargetKind::eWORLD_STATIC &&
				 sphere.targetKind !=
						AvbdSoftContactTargetKind::eKINEMATIC_RIGID &&
				 !dynamicTarget) ||
				(dynamicTarget &&
				 (!sphere.predictedPoseValid ||
				  !sphere.predictedCenter.isFinite() ||
				  !sphere.predictedRotation.isFinite())))
				continue;
			const PxVec3 sphereCenterStart =
				sphere.targetKind ==
					AvbdSoftContactTargetKind::eKINEMATIC_RIGID
					? sphere.previousCenter : sphere.center;
			const PxVec3 sphereCenterEnd =
				dynamicTarget
					? sphere.predictedCenter : sphere.center;
			if(!sphereCenterStart.isFinite() ||
				(sphere.targetKind ==
						AvbdSoftContactTargetKind::eKINEMATIC_RIGID &&
				 !sphere.previousRotation.isFinite()))
				continue;
			// A moving sphere is swept in relative coordinates.  This keeps
			// a prescribed target that crosses a stationary soft vertex from
			// being discarded merely because the soft displacement is zero.
			const PxVec3 relativeStart =
				particle.position - sphereCenterStart;
			const PxVec3 relativeEnd =
				particle.predictedPosition - sphereCenterEnd;
			const PxVec3 relativeDisplacement =
				relativeEnd - relativeStart;
			if(relativeDisplacement.magnitudeSquared() <= 1e-12f)
				continue;
			const PxReal currentSdf =
				relativeStart.magnitude() - sphere.radius;
			if(!PxIsFinite(currentSdf) || currentSdf < margin)
				continue;

			PxReal entryTime = 0.0f;
			PxVec3 entryNormal(0.0f);
			if(!avbdSegmentEnterExpandedSphere(
					relativeStart,
					relativeEnd,
					PxVec3(0.0f),
					sphere.radius + margin,
					entryTime, entryNormal))
				continue;

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF,
				PX_MAX_U32, sphere.primitiveKey, 1u);
			geometry.particleIdx = particleIndex;
			geometry.normal = entryNormal;
			geometry.projNormal = entryNormal;
			geometry.depth = 0.0f;
			geometry.margin = margin;
			const PxVec3 surfaceLocal =
				sphere.rotation.getConjugate().rotate(
					entryNormal * sphere.radius);
			avbdConfigureRigidSphereTarget(
				geometry, sphere, sphereIndex, surfaceLocal);
			geometry.friction =
				avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction,
					sphere.friction,
					sphere.frictionCombineMode);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1e6f, 1e6f,
				particles, contacts);
		}
	}
}

inline void avbdDetectSoftRigidSphereSweptSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	avbdDetectSoftRigidSphereSweptSDFRange(
		particles, numParticles, 0, numParticles,
		spheres, numSpheres, contacts, margin,
		softBodies, numSoftBodies);
}

struct AvbdSweptTriangleEntry
{
	PxReal entryTime;
	PxVec3 normal;
	PxVec3 barycentric;
	AvbdClosestFeature feature;
	PxU32 featureIndex;

	AvbdSweptTriangleEntry()
		: entryTime(PX_MAX_F32), normal(0.0f),
		  barycentric(1.0f, 0.0f, 0.0f),
		  feature(AVBD_FEATURE_UNKNOWN), featureIndex(0)
	{
	}
};

PX_FORCE_INLINE void avbdUpdateSweptTriangleEntry(
	AvbdSweptTriangleEntry& result,
	PxReal entryTime, const PxVec3& normal,
	const PxVec3& barycentric,
	AvbdClosestFeature feature, PxU32 featureIndex)
{
	if(entryTime < 0.0f || entryTime > 1.0f ||
		entryTime >= result.entryTime ||
		!normal.isFinite() ||
		normal.magnitudeSquared() <= 1.0e-12f ||
		!barycentric.isFinite())
		return;
	result.entryTime = entryTime;
	result.normal = normal.getNormalized();
	result.barycentric = barycentric;
	result.feature = feature;
	result.featureIndex = featureIndex;
}

// Exact entry of a moving point into the face/edge-owned part of a static
// triangle expanded by a radius. The rounded vertex caps are intentionally
// excluded: soft-vertex/sphere swept SDF is their unique owner.
PX_FORCE_INLINE bool avbdSegmentEnterExpandedTriangleNonVertex(
	const PxVec3& segmentStart, const PxVec3& segmentEnd,
	const PxVec3& a, const PxVec3& b, const PxVec3& c,
	PxReal expandedRadius, AvbdSweptTriangleEntry& result)
{
	if(!segmentStart.isFinite() || !segmentEnd.isFinite() ||
		!a.isFinite() || !b.isFinite() || !c.isFinite() ||
		expandedRadius <= 0.0f || !PxIsFinite(expandedRadius))
		return false;
	const PxVec3 direction = segmentEnd - segmentStart;
	const PxReal directionMagnitudeSq = direction.magnitudeSquared();
	if(directionMagnitudeSq <= 1.0e-12f ||
		!PxIsFinite(directionMagnitudeSq))
		return false;
	const AvbdClosestPointResult currentClosest =
		avbdClosestPointOnTriangleOGC(segmentStart, a, b, c);
	if(!PxIsFinite(currentClosest.distance) ||
		currentClosest.distance < expandedRadius)
		return false;

	const PxVec3 unnormalizedNormal = (b - a).cross(c - a);
	const PxReal normalMagnitudeSq =
		unnormalizedNormal.magnitudeSquared();
	if(normalMagnitudeSq <= 1.0e-16f ||
		!PxIsFinite(normalMagnitudeSq))
		return false;
	const PxVec3 triangleNormal =
		unnormalizedNormal * PxRecipSqrt(normalMagnitudeSq);
	const PxReal startPlaneDistance =
		triangleNormal.dot(segmentStart - a);
	const PxReal planeDirection =
		triangleNormal.dot(direction);

	if(PxAbs(planeDirection) > 1.0e-12f)
	{
		PxReal side = 0.0f;
		if(startPlaneDistance >= expandedRadius)
			side = 1.0f;
		else if(startPlaneDistance <= -expandedRadius)
			side = -1.0f;
		if(side != 0.0f)
		{
			const PxReal entryTime =
				(side * expandedRadius - startPlaneDistance) /
					planeDirection;
			if(entryTime >= 0.0f && entryTime <= 1.0f)
			{
				const PxVec3 centerAtEntry =
					segmentStart + direction * entryTime;
				const PxVec3 trianglePoint =
					centerAtEntry -
						triangleNormal *
							(side * expandedRadius);
				const AvbdClosestPointResult faceClosest =
					avbdClosestPointOnTriangleOGC(
						trianglePoint, a, b, c);
				if(faceClosest.feature == AVBD_FEATURE_FACE &&
					faceClosest.distance <= 1.0e-5f)
					avbdUpdateSweptTriangleEntry(
						result, entryTime,
						-triangleNormal * side,
						faceClosest.barycentric,
						AVBD_FEATURE_FACE, 0);
			}
		}
	}

	const PxVec3 edgeStart[3] = {a, a, b};
	const PxVec3 edgeEnd[3] = {b, c, c};
	for(PxU32 edgeIndex = 0; edgeIndex < 3; ++edgeIndex)
	{
		const PxVec3 edge = edgeEnd[edgeIndex] - edgeStart[edgeIndex];
		const PxReal edgeLengthSq = edge.magnitudeSquared();
		if(edgeLengthSq <= 1.0e-16f ||
			!PxIsFinite(edgeLengthSq))
			continue;
		const PxReal edgeLength = PxSqrt(edgeLengthSq);
		const PxVec3 edgeDirection = edge / edgeLength;
		const PxVec3 startOffset =
			segmentStart - edgeStart[edgeIndex];
		const PxReal startAxial =
			startOffset.dot(edgeDirection);
		const PxReal directionAxial =
			direction.dot(edgeDirection);
		const PxVec3 startRadial =
			startOffset - edgeDirection * startAxial;
		const PxVec3 directionRadial =
			direction - edgeDirection * directionAxial;
		const PxReal quadraticA =
			directionRadial.magnitudeSquared();
		const PxReal quadraticHalfB =
			startRadial.dot(directionRadial);
		const PxReal quadraticC =
			startRadial.magnitudeSquared() -
				expandedRadius * expandedRadius;
		if(quadraticA <= 1.0e-12f || quadraticC < 0.0f)
			continue;
		const PxReal discriminant =
			quadraticHalfB * quadraticHalfB -
				quadraticA * quadraticC;
		if(discriminant < 0.0f || !PxIsFinite(discriminant))
			continue;
		const PxReal entryTime =
			(-quadraticHalfB - PxSqrt(discriminant)) /
				quadraticA;
		if(entryTime < 0.0f || entryTime > 1.0f)
			continue;
		const PxReal axial =
			startAxial + directionAxial * entryTime;
		const PxReal endpointEpsilon =
			PxMax(1.0e-5f, edgeLength * 1.0e-5f);
		if(axial <= endpointEpsilon ||
			axial >= edgeLength - endpointEpsilon)
			continue;
		const PxVec3 centerAtEntry =
			segmentStart + direction * entryTime;
		const PxVec3 edgePoint =
			edgeStart[edgeIndex] + edgeDirection * axial;
		const PxVec3 centerToEdge = edgePoint - centerAtEntry;
		PxVec3 barycentric(0.0f);
		const PxReal edgeWeight = axial / edgeLength;
		if(edgeIndex == 0)
			barycentric =
				PxVec3(1.0f - edgeWeight, edgeWeight, 0.0f);
		else if(edgeIndex == 1)
			barycentric =
				PxVec3(1.0f - edgeWeight, 0.0f, edgeWeight);
		else
			barycentric =
				PxVec3(0.0f, 1.0f - edgeWeight, edgeWeight);
		avbdUpdateSweptTriangleEntry(
			result, entryTime, centerToEdge, barycentric,
			AVBD_FEATURE_EDGE, edgeIndex);
	}
	return result.entryTime < PX_MAX_F32 &&
		(result.feature == AVBD_FEATURE_FACE ||
		 result.feature == AVBD_FEATURE_EDGE);
}

// Continuous entry of a linearly moving point into the face/edge-owned
// portion of a linearly deforming triangle expanded by a radius. The
// point speed plus the maximum triangle-vertex speed bounds the Hausdorff
// speed of the two features, so conservative advancement cannot step over
// first contact. Rounded triangle vertices remain forward-SDF owned.
PX_FORCE_INLINE bool
avbdLinearPointEnterExpandedDeformingTriangleNonVertex(
	const PxVec3& pointStart, const PxVec3& pointEnd,
	const PxVec3& aStart, const PxVec3& bStart,
	const PxVec3& cStart, const PxVec3& aEnd,
	const PxVec3& bEnd, const PxVec3& cEnd,
	PxReal expandedRadius, AvbdSweptTriangleEntry& result)
{
	if(!pointStart.isFinite() || !pointEnd.isFinite() ||
		!aStart.isFinite() || !bStart.isFinite() ||
		!cStart.isFinite() || !aEnd.isFinite() ||
		!bEnd.isFinite() || !cEnd.isFinite() ||
		expandedRadius <= 0.0f || !PxIsFinite(expandedRadius))
		return false;
	const PxVec3 pointDisplacement = pointEnd - pointStart;
	const PxVec3 displacementA = aEnd - aStart;
	const PxVec3 displacementB = bEnd - bStart;
	const PxVec3 displacementC = cEnd - cStart;
	const PxReal triangleSpeed = PxMax(
		displacementA.magnitude(),
		PxMax(displacementB.magnitude(),
			displacementC.magnitude()));
	const PxReal speed =
		pointDisplacement.magnitude() + triangleSpeed;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, expandedRadius * 1.0e-5f);

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 64; ++iteration)
	{
		const PxVec3 point =
			pointStart + pointDisplacement * time;
		const PxVec3 a = aStart + displacementA * time;
		const PxVec3 b = bStart + displacementB * time;
		const PxVec3 c = cStart + displacementC * time;
		const PxVec3 triangleNormal = (b - a).cross(c - a);
		if(!point.isFinite() || !a.isFinite() ||
			!b.isFinite() || !c.isFinite() ||
			!triangleNormal.isFinite() ||
			triangleNormal.magnitudeSquared() <= 1.0e-16f)
			return false;
		const AvbdClosestPointResult closest =
			avbdClosestPointOnTriangleOGC(point, a, b, c);
		if(!PxIsFinite(closest.distance))
			return false;
		if(iteration == 0 &&
			closest.distance < expandedRadius)
			return false;
		const PxReal gap = closest.distance - expandedRadius;
		if(gap <= distanceTolerance)
		{
			if(closest.feature != AVBD_FEATURE_FACE &&
				closest.feature != AVBD_FEATURE_EDGE)
				return false;
			const PxVec3 normal = closest.point - point;
			const PxReal normalMagnitudeSq =
				normal.magnitudeSquared();
			if(!normal.isFinite() ||
				normalMagnitudeSq <= 1.0e-12f)
				return false;
			result.entryTime = time;
			result.normal =
				normal * PxRecipSqrt(normalMagnitudeSq);
			result.barycentric = closest.barycentric;
			result.feature = closest.feature;
			result.featureIndex = closest.featureIndex;
			return true;
		}
		const PxReal nextTime = time + gap / speed;
		if(!PxIsFinite(nextTime) || nextTime > 1.0f ||
			nextTime <= time)
			return false;
		time = nextTime;
	}
	return false;
}

PX_FORCE_INLINE bool avbdRigidSphereForwardVertexOwnsSweptFeature(
	const AvbdSoftParticle& particle,
	const AvbdRigidSphere& sphere,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	PxReal margin)
{
	if(particle.invMass <= 0.0f)
		return false;
	const PxVec3 vertexRelativeStart =
		particle.initialPosition - centerStart;
	const PxVec3 vertexRelativeEnd =
		particle.predictedPosition - centerEnd;
	const PxReal currentSdf =
		vertexRelativeStart.magnitude() - sphere.radius;
	if(!PxIsFinite(currentSdf))
		return false;
	if(currentSdf < margin)
		return true;
	PxReal vertexEntryTime = 0.0f;
	PxVec3 vertexEntryNormal(0.0f);
	return avbdSegmentEnterExpandedSphere(
		vertexRelativeStart, vertexRelativeEnd,
		PxVec3(0.0f), sphere.radius + margin,
		vertexEntryTime, vertexEntryNormal);
}

inline void avbdDetectSoftRigidSphereSweptOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	PxArray<PxU8>* persistentForwardOwnerScratch = NULL)
{
	const PxReal translationToleranceSq = 1.0e-10f;
	PxArray<PxU8> localForwardOwnerScratch;
	PxArray<PxU8>& forwardOwnerScratch =
		persistentForwardOwnerScratch
			? *persistentForwardOwnerScratch
			: localForwardOwnerScratch;
	if(forwardOwnerScratch.capacity() < numParticles)
		forwardOwnerScratch.reserve(numParticles);
	forwardOwnerScratch.resize(numParticles);
	for(PxU32 bodyIndex = 0;
		bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		if(!body.compiled.speculativeCCDEnabled)
			continue;
		for(PxU32 sphereIndex = 0;
			sphereIndex < numSpheres; ++sphereIndex)
		{
			const AvbdRigidSphere& sphere = spheres[sphereIndex];
			const bool kinematicTarget =
				sphere.targetKind ==
					AvbdSoftContactTargetKind::eKINEMATIC_RIGID;
			const bool dynamicTarget =
				sphere.targetKind ==
					AvbdSoftContactTargetKind::eRIGID_BODY;
			if(sphere.radius <= 0.0f ||
				!PxIsFinite(sphere.radius) ||
				!sphere.center.isFinite() ||
				!sphere.rotation.isFinite() ||
				(sphere.targetKind !=
						AvbdSoftContactTargetKind::eWORLD_STATIC &&
				 !kinematicTarget && !dynamicTarget) ||
				(kinematicTarget &&
				 (!sphere.previousCenter.isFinite() ||
				  !sphere.previousRotation.isFinite())) ||
				(dynamicTarget &&
				 (!sphere.predictedPoseValid ||
				  !sphere.predictedCenter.isFinite() ||
				  !sphere.predictedRotation.isFinite())))
				continue;
			const PxVec3 centerStart =
				kinematicTarget
					? sphere.previousCenter : sphere.center;
			const PxVec3 centerEnd =
				dynamicTarget
					? sphere.predictedCenter : sphere.center;
			const PxU32 particleStart = body.compiled.particleStart;
			const PxU32 particleCount = body.compiled.particleCount;
			if(particleStart <= numParticles)
			{
				const PxU32 boundedParticleCount = PxMin(
					particleCount, numParticles - particleStart);
				for(PxU32 localParticle = 0;
					localParticle < boundedParticleCount; ++localParticle)
					forwardOwnerScratch[
						particleStart + localParticle] = 0;
			}
			for(PxU32 surfaceVertexIndex = 0;
				surfaceVertexIndex < body.compiled.surfaceVertices.size();
				++surfaceVertexIndex)
			{
				const PxU32 vertexIndex =
					body.compiled.surfaceVertices[surfaceVertexIndex];
				if(vertexIndex >= numParticles)
					continue;
				forwardOwnerScratch[vertexIndex] = PxU8(
					avbdRigidSphereForwardVertexOwnsSweptFeature(
						particles[vertexIndex], sphere,
						centerStart, centerEnd, margin));
			}
			PxArray<PxU64> emittedFeatureKeys;
			for(PxU32 triangleOffset = 0;
				triangleOffset + 2 <
					body.compiled.surfaceTriangles.size();
				triangleOffset += 3)
			{
				const PxU32 v0 =
					body.compiled.surfaceTriangles[triangleOffset];
				const PxU32 v1 =
					body.compiled.surfaceTriangles[
						triangleOffset + 1];
				const PxU32 v2 =
					body.compiled.surfaceTriangles[
						triangleOffset + 2];
				if(v0 >= numParticles || v1 >= numParticles ||
					v2 >= numParticles ||
					(particles[v0].invMass <= 0.0f &&
					 particles[v1].invMass <= 0.0f &&
					 particles[v2].invMass <= 0.0f))
					continue;
				// Outer contact redetection must reconstruct the same
				// frame-level sweep.  Using the iterated position here makes
				// a first-impact row disappear as soon as the first primal
				// sweep advances through the obstacle.
				const PxVec3 p0 = particles[v0].initialPosition;
				const PxVec3 p1 = particles[v1].initialPosition;
				const PxVec3 p2 = particles[v2].initialPosition;
				const PxVec3 displacement0 =
					particles[v0].predictedPosition - p0;
				const PxVec3 displacement1 =
					particles[v1].predictedPosition - p1;
				const PxVec3 displacement2 =
					particles[v2].predictedPosition - p2;
				if(!p0.isFinite() || !p1.isFinite() ||
					!p2.isFinite() ||
					!displacement0.isFinite() ||
					!displacement1.isFinite() ||
					!displacement2.isFinite())
					continue;
				const PxVec3 relativeDisplacement1 =
					(displacement1 - displacement0);
				const PxVec3 relativeDisplacement2 =
					(displacement2 - displacement0);
				const bool softTriangleTranslationOnly =
					relativeDisplacement1.magnitudeSquared() <=
						translationToleranceSq &&
					relativeDisplacement2.magnitudeSquared() <=
						translationToleranceSq;
				const PxReal expandedRadius =
					sphere.radius + margin;
				const bool forwardVertexOwns =
					forwardOwnerScratch[v0] != 0 ||
					forwardOwnerScratch[v1] != 0 ||
					forwardOwnerScratch[v2] != 0;
				if(forwardVertexOwns)
					continue;
				const PxVec3 relativeEnd =
					centerEnd - displacement0;
				AvbdSweptTriangleEntry entry;
				const bool entered =
					softTriangleTranslationOnly
						? avbdSegmentEnterExpandedTriangleNonVertex(
							centerStart, relativeEnd,
							p0, p1, p2,
							expandedRadius, entry)
						: avbdLinearPointEnterExpandedDeformingTriangleNonVertex(
							centerStart, relativeEnd,
							p0, p1, p2, p0,
							p1 + relativeDisplacement1,
							p2 + relativeDisplacement2,
							expandedRadius, entry);
				if(!entered)
					continue;

				const PxU64 softFeatureKey =
					avbdSoftTriangleFeatureKey(
						v0, v1, v2,
						entry.feature, entry.featureIndex);
				PxU64 featureKey = 1469598103934665603ull;
				featureKey = avbdSoftContactHashValue(
					featureKey, 0x53505357u);
				featureKey = avbdSoftContactHashValue(
					featureKey, PxU32(softFeatureKey));
				featureKey = avbdSoftContactHashValue(
					featureKey, PxU32(softFeatureKey >> 32));
				bool duplicate = false;
				for(PxU32 emittedIndex = 0;
					emittedIndex < emittedFeatureKeys.size();
					++emittedIndex)
				{
					if(emittedFeatureKeys[emittedIndex] ==
						featureKey)
					{
						duplicate = true;
						break;
					}
				}
				if(duplicate)
					continue;
				emittedFeatureKeys.pushBack(featureKey);

				AvbdSoftContactGeometry geometry;
				geometry.source = AvbdSoftContactSource(
					AvbdSoftContactSource::eRIGID_SDF,
					PX_MAX_U32, sphere.primitiveKey,
					featureKey);
				geometry.particleIdx =
					particles[v0].invMass > 0.0f ? v0 :
						(particles[v1].invMass > 0.0f
							? v1 : v2);
				geometry.queryParticleIndices[0] = v0;
				geometry.queryParticleIndices[1] = v1;
				geometry.queryParticleIndices[2] = v2;
				geometry.queryWeights[0] = entry.barycentric.x;
				geometry.queryWeights[1] = entry.barycentric.y;
				geometry.queryWeights[2] = entry.barycentric.z;
				geometry.normal = entry.normal;
				geometry.projNormal = entry.normal;
				geometry.depth = 0.0f;
				geometry.margin = margin;
				const PxVec3 surfaceLocal =
					sphere.rotation.getConjugate().rotate(
						entry.normal * sphere.radius);
				avbdConfigureRigidSphereTarget(
					geometry, sphere, sphereIndex,
					surfaceLocal);
				geometry.friction =
					avbdCombineDeformableRigidFriction(
						body.material.dynamicFriction,
						sphere.friction,
						sphere.frictionCombineMode);
				avbdBuildSoftContactTangents(geometry);
				avbdAppendPreparedSoftContact(
					geometry, 1.0e6f, 1.0e6f,
					particles, contacts);
			}
		}
	}
}

inline void avbdDetectSoftRigidSphereOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f)
{
	const PxReal normalEpsilon = 1.0e-12f;
	for(PxU32 bodyIndex = 0;
		bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		PxVec3 bodyMinimum(PX_MAX_F32);
		PxVec3 bodyMaximum(-PX_MAX_F32);
		for(PxU32 localParticle = 0;
			localParticle < body.compiled.particleCount;
			++localParticle)
		{
			const PxU32 particleIndex =
				body.compiled.particleStart + localParticle;
			if(particleIndex >= numParticles)
				continue;
			const PxVec3& position =
				particles[particleIndex].position;
			bodyMinimum = bodyMinimum.minimum(position);
			bodyMaximum = bodyMaximum.maximum(position);
		}

		for(PxU32 sphereIndex = 0;
			sphereIndex < numSpheres; ++sphereIndex)
		{
			const AvbdRigidSphere& sphere = spheres[sphereIndex];
			if(sphere.radius <= 0.0f ||
				!PxIsFinite(sphere.radius) ||
				!sphere.center.isFinite() ||
				!sphere.rotation.isFinite())
				continue;
			const PxReal queryRadius =
				sphere.radius + margin;
			if(bodyMinimum.x > sphere.center.x + queryRadius ||
				bodyMaximum.x < sphere.center.x - queryRadius ||
				bodyMinimum.y > sphere.center.y + queryRadius ||
				bodyMaximum.y < sphere.center.y - queryRadius ||
				bodyMinimum.z > sphere.center.z + queryRadius ||
				bodyMaximum.z < sphere.center.z - queryRadius)
				continue;

			PxArray<PxU64> emittedFeatureKeys;
			for(PxU32 triangleOffset = 0;
				triangleOffset + 2 <
					body.compiled.surfaceTriangles.size();
				triangleOffset += 3)
			{
				const PxU32 v0 =
					body.compiled.surfaceTriangles[triangleOffset];
				const PxU32 v1 =
					body.compiled.surfaceTriangles[
						triangleOffset + 1];
				const PxU32 v2 =
					body.compiled.surfaceTriangles[
						triangleOffset + 2];
				if(v0 >= numParticles || v1 >= numParticles ||
					v2 >= numParticles ||
					(particles[v0].invMass <= 0.0f &&
					 particles[v1].invMass <= 0.0f &&
					 particles[v2].invMass <= 0.0f))
					continue;
				const PxVec3& p0 = particles[v0].position;
				const PxVec3& p1 = particles[v1].position;
				const PxVec3& p2 = particles[v2].position;
				const PxVec3 triangleMinimum =
					p0.minimum(p1).minimum(p2) -
						PxVec3(queryRadius);
				const PxVec3 triangleMaximum =
					p0.maximum(p1).maximum(p2) +
						PxVec3(queryRadius);
				if(sphere.center.x < triangleMinimum.x ||
					sphere.center.x > triangleMaximum.x ||
					sphere.center.y < triangleMinimum.y ||
					sphere.center.y > triangleMaximum.y ||
					sphere.center.z < triangleMinimum.z ||
					sphere.center.z > triangleMaximum.z)
					continue;

				const AvbdClosestPointResult closest =
					avbdClosestPointOnTriangleOGC(
						sphere.center, p0, p1, p2);
				// Vertex ownership already belongs to the forward
				// soft-vertex/sphere SDF. Reverse smooth ownership adds only
				// edge and face features, preventing duplicate AL states.
				if(closest.feature == AVBD_FEATURE_VERTEX ||
					closest.feature == AVBD_FEATURE_UNKNOWN ||
					!PxIsFinite(closest.distance) ||
					closest.distance >= queryRadius)
					continue;

				const PxU64 softFeatureKey =
					avbdSoftTriangleFeatureKey(
						v0, v1, v2,
						closest.feature, closest.featureIndex);
				PxU64 featureKey = 1469598103934665603ull;
				featureKey = avbdSoftContactHashValue(
					featureKey, 0x53504852u);
				featureKey = avbdSoftContactHashValue(
					featureKey, PxU32(softFeatureKey));
				featureKey = avbdSoftContactHashValue(
					featureKey, PxU32(softFeatureKey >> 32));
				bool duplicate = false;
				for(PxU32 emittedIndex = 0;
					emittedIndex < emittedFeatureKeys.size();
					++emittedIndex)
				{
					if(emittedFeatureKeys[emittedIndex] ==
						featureKey)
					{
						duplicate = true;
						break;
					}
				}
				if(duplicate)
					continue;
				emittedFeatureKeys.pushBack(featureKey);

				PxVec3 normal = -closest.normal;
				const PxReal normalMagnitudeSq =
					normal.magnitudeSquared();
				if(!normal.isFinite() ||
					normalMagnitudeSq <= normalEpsilon)
					continue;
				normal *= PxRecipSqrt(normalMagnitudeSq);

				AvbdSoftContactGeometry geometry;
				geometry.source = AvbdSoftContactSource(
					AvbdSoftContactSource::eRIGID_SDF,
					PX_MAX_U32, sphere.primitiveKey,
					featureKey);
				geometry.particleIdx =
					particles[v0].invMass > 0.0f ? v0 :
						(particles[v1].invMass > 0.0f
							? v1 : v2);
				geometry.queryParticleIndices[0] = v0;
				geometry.queryParticleIndices[1] = v1;
				geometry.queryParticleIndices[2] = v2;
				geometry.queryWeights[0] =
					closest.barycentric.x;
				geometry.queryWeights[1] =
					closest.barycentric.y;
				geometry.queryWeights[2] =
					closest.barycentric.z;
				geometry.normal = normal;
				geometry.projNormal = normal;
				geometry.depth =
					queryRadius - closest.distance;
				geometry.margin = margin;
				const PxVec3 surfaceLocal =
					sphere.rotation.getConjugate().rotate(
						normal * sphere.radius);
				avbdConfigureRigidSphereTarget(
					geometry, sphere, sphereIndex,
					surfaceLocal);
				geometry.friction =
					avbdCombineDeformableRigidFriction(
						body.material.dynamicFriction,
						sphere.friction,
						sphere.frictionCombineMode);
				avbdBuildSoftContactTangents(geometry);
				avbdAppendPreparedSoftContact(
					geometry, 1e5f, 1e6f,
					particles, contacts);
			}
		}
	}
}

PX_FORCE_INLINE void avbdConfigureRigidCapsuleTarget(
	AvbdSoftContactGeometry& geometry,
	const AvbdRigidCapsule& capsule, PxU32 capsuleIndex,
	const PxVec3& surfaceLocal)
{
	geometry.targetKind = capsule.targetKind;
	geometry.velocityOwner =
		capsule.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID
			? AvbdVelocityObjectiveOwner::ComponentFinalize
			: capsule.targetKind ==
				AvbdSoftContactTargetKind::eRIGID_BODY
				? AvbdVelocityObjectiveOwner::ManifoldFinalize
				: AvbdVelocityObjectiveOwner::PositionAL;
	geometry.targetIndex =
		capsule.targetKind ==
			AvbdSoftContactTargetKind::eRIGID_BODY
			? capsule.targetIndex : capsuleIndex;
	geometry.surfacePoint =
		capsule.center + capsule.rotation.rotate(surfaceLocal);
	geometry.kinematicSurfacePointPrevious =
		capsule.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID
			? capsule.previousCenter +
				capsule.previousRotation.rotate(surfaceLocal)
			: geometry.surfacePoint;
	if(capsule.targetKind ==
		AvbdSoftContactTargetKind::eRIGID_BODY)
		geometry.rigidLocalPoint =
			capsule.shapeToRigidBody.transform(surfaceLocal);
}

// P5.6a candidate leaf: current-pose capsule SDF is particle-major and uses
// only immutable primitive/body inputs plus a caller-owned output stream.
// Swept and feature suffixes remain parent-owned until separately proven.
inline void avbdDetectSoftRigidCapsuleSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	PX_ASSERT(particles && particleBegin <= particleEnd);
	PX_ASSERT(particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for(PxU32 particleIndex = particleBegin;
		particleIndex < particleEnd; ++particleIndex)
	{
		const AvbdSoftParticle& particle = particles[particleIndex];
		if(particle.invMass <= 0.0f)
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies, particleIndex);
		if(sourceBody &&
			!avbdIsSoftBodySurfaceVertex(
				*sourceBody, particleIndex))
			continue;

		for(PxU32 capsuleIndex = 0;
			capsuleIndex < numCapsules; ++capsuleIndex)
		{
			const AvbdRigidCapsule& capsule =
				capsules[capsuleIndex];
			if(capsule.radius <= 0.0f ||
				capsule.halfHeight < 0.0f ||
				!PxIsFinite(capsule.radius) ||
				!PxIsFinite(capsule.halfHeight) ||
				!capsule.center.isFinite() ||
				!capsule.rotation.isFinite())
				continue;
			const PxReal broadphaseRadius =
				capsule.radius + capsule.halfHeight + margin;
			const PxVec3 worldOffset =
				particle.position - capsule.center;
			if(worldOffset.magnitudeSquared() >=
				broadphaseRadius * broadphaseRadius)
				continue;
			const PxQuat inverseRotation =
				capsule.rotation.getConjugate();
			const PxVec3 particleLocal =
				inverseRotation.rotate(worldOffset);
			const PxVec3 axisLocal(
				PxClamp(particleLocal.x,
					-capsule.halfHeight,
					capsule.halfHeight),
				0.0f, 0.0f);
			const PxVec3 radialLocal =
				particleLocal - axisLocal;
			const PxReal distanceSq =
				radialLocal.magnitudeSquared();
			const PxReal queryRadius =
				capsule.radius + margin;
			if(!PxIsFinite(distanceSq) ||
				distanceSq >= queryRadius * queryRadius)
				continue;
			const PxReal distance =
				PxSqrt(PxMax(distanceSq, 0.0f));
			PxVec3 normalLocal(0.0f, 1.0f, 0.0f);
			if(distance > 1.0e-6f)
				normalLocal = radialLocal * (1.0f / distance);
			else
			{
				const PxVec3 initialLocal =
					inverseRotation.rotate(
						particle.initialPosition -
							capsule.center);
				const PxVec3 initialAxis(
					PxClamp(initialLocal.x,
						-capsule.halfHeight,
						capsule.halfHeight),
					0.0f, 0.0f);
				const PxVec3 initialRadial =
					initialLocal - initialAxis;
				const PxReal initialMagnitudeSq =
					initialRadial.magnitudeSquared();
				if(initialMagnitudeSq > 1.0e-12f &&
					PxIsFinite(initialMagnitudeSq))
					normalLocal = initialRadial *
						PxRecipSqrt(initialMagnitudeSq);
			}
			const PxVec3 normal =
				capsule.rotation.rotate(normalLocal).
					getNormalized();
			const PxReal sdf = distance - capsule.radius;

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF,
				PX_MAX_U32, capsule.primitiveKey, 1u);
			geometry.particleIdx = particleIndex;
			geometry.normal = normal;
			geometry.projNormal = normal;
			geometry.depth = sdf < 0.0f
				? -sdf : PxMax(0.0f, margin - sdf);
			geometry.margin = margin;
			const PxVec3 surfaceLocal =
				axisLocal + normalLocal * capsule.radius;
			avbdConfigureRigidCapsuleTarget(
				geometry, capsule, capsuleIndex,
				surfaceLocal);
			geometry.friction = sourceBody
				? avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction,
					capsule.friction,
					capsule.frictionCombineMode)
				: PxMax(capsule.friction, 0.0f);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1e5f, 1e6f,
				particles, contacts);
		}
	}
}

inline void avbdDetectSoftRigidCapsuleSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	avbdDetectSoftRigidCapsuleSDFRange(
		particles, numParticles, 0, numParticles,
		capsules, numCapsules, contacts, margin,
		softBodies, numSoftBodies);
}

PX_FORCE_INLINE bool avbdAreSweepRotationsEquivalent(
	const PxQuat& startRotation, const PxQuat& endRotation,
	PxReal tolerance = 0.0f)
{
	if(!startRotation.isFinite() || !endRotation.isFinite())
		return false;
	const PxReal alignment =
		PxAbs(startRotation.dot(endRotation));
	return PxIsFinite(alignment) &&
		alignment >= 1.0f - tolerance;
}

PX_FORCE_INLINE bool avbdGetSweepAngularDistance(
	const PxQuat& startRotation, const PxQuat& endRotation,
	PxReal& angularDistance)
{
	if(!startRotation.isFinite() || !endRotation.isFinite())
		return false;
	const PxReal startMagnitudeSq = startRotation.magnitudeSquared();
	const PxReal endMagnitudeSq = endRotation.magnitudeSquared();
	if(startMagnitudeSq <= 1.0e-12f ||
		endMagnitudeSq <= 1.0e-12f ||
		!PxIsFinite(startMagnitudeSq) ||
		!PxIsFinite(endMagnitudeSq))
		return false;
	const PxQuat normalizedStart = startRotation.getNormalized();
	const PxQuat normalizedEnd = endRotation.getNormalized();
	const PxReal alignment = PxClamp(
		PxAbs(normalizedStart.dot(normalizedEnd)), 0.0f, 1.0f);
	angularDistance = 2.0f * PxAcos(alignment);
	return PxIsFinite(angularDistance);
}

PX_FORCE_INLINE bool avbdGetRigidCapsuleSweepPose(
	const AvbdRigidCapsule& capsule,
	PxVec3& centerStart, PxVec3& centerEnd,
	PxQuat& rotationStart, PxQuat& rotationEnd,
	bool& rotationsEquivalent)
{
	const bool kinematicTarget =
		capsule.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID;
	const bool dynamicTarget =
		capsule.targetKind ==
			AvbdSoftContactTargetKind::eRIGID_BODY;
	if(capsule.radius <= 0.0f ||
		capsule.halfHeight < 0.0f ||
		!PxIsFinite(capsule.radius) ||
		!PxIsFinite(capsule.halfHeight) ||
		!capsule.center.isFinite() ||
		!capsule.rotation.isFinite() ||
		(capsule.targetKind !=
				AvbdSoftContactTargetKind::eWORLD_STATIC &&
		 !kinematicTarget && !dynamicTarget) ||
		(kinematicTarget &&
		 (!capsule.previousCenter.isFinite() ||
		  !capsule.previousRotation.isFinite())) ||
		(dynamicTarget &&
		 (!capsule.predictedPoseValid ||
		  !capsule.predictedCenter.isFinite() ||
		  !capsule.predictedRotation.isFinite())))
		return false;

	centerStart =
		kinematicTarget ? capsule.previousCenter : capsule.center;
	centerEnd =
		dynamicTarget ? capsule.predictedCenter : capsule.center;
	rotationStart =
		kinematicTarget ? capsule.previousRotation : capsule.rotation;
	rotationEnd =
		dynamicTarget ? capsule.predictedRotation : capsule.rotation;
	if(!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite())
		return false;
	rotationsEquivalent = avbdAreSweepRotationsEquivalent(
		rotationStart, rotationEnd);
	return true;
}

struct AvbdSweptRotatingCapsulePointEntry
{
	PxReal entryTime;
	PxVec3 normal;
	PxVec3 surfaceLocal;

	AvbdSweptRotatingCapsulePointEntry()
		: entryTime(PX_MAX_F32), normal(0.0f),
		  surfaceLocal(0.0f)
	{
	}
};

// Conservative point entry against a capsule whose center translates
// linearly and whose orientation follows shortest-path quaternion slerp.
// Point/center relative translation plus halfHeight*angularDistance bounds
// the Hausdorff speed of the moving medial segment, so gap/speed cannot step
// across first contact. The returned shape-local point is the material point
// at entry and remains valid for the prescribed end pose.
PX_FORCE_INLINE bool avbdSegmentEnterExpandedRotatingCapsule(
	const PxVec3& pointStart, const PxVec3& pointEnd,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	PxReal halfHeight, PxReal capsuleRadius, PxReal margin,
	AvbdSweptRotatingCapsulePointEntry& result)
{
	if(!pointStart.isFinite() || !pointEnd.isFinite() ||
		!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite() ||
		halfHeight < 0.0f || capsuleRadius <= 0.0f ||
		margin <= 0.0f || !PxIsFinite(halfHeight) ||
		!PxIsFinite(capsuleRadius) || !PxIsFinite(margin))
		return false;

	PxReal angularDistance = 0.0f;
	if(!avbdGetSweepAngularDistance(
			rotationStart, rotationEnd, angularDistance) ||
		angularDistance <= 0.0f)
		return false;
	const PxQuat normalizedStart = rotationStart.getNormalized();
	const PxQuat normalizedEnd = rotationEnd.getNormalized();
	const PxVec3 relativeTranslation =
		(pointEnd - pointStart) - (centerEnd - centerStart);
	const PxReal speed =
		relativeTranslation.magnitude() +
		halfHeight * angularDistance;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;

	const PxReal expandedRadius = capsuleRadius + margin;
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, expandedRadius * 1.0e-5f);
	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 64; ++iteration)
	{
		const PxVec3 point =
			pointStart + (pointEnd - pointStart) * time;
		const PxVec3 center =
			centerStart + (centerEnd - centerStart) * time;
		const PxQuat rotation =
			PxSlerp(time, normalizedStart, normalizedEnd).
				getNormalized();
		if(!point.isFinite() || !center.isFinite() ||
			!rotation.isFinite())
			return false;
		const PxVec3 axis = rotation.getBasisVector0();
		const PxReal axisCoordinate = PxClamp(
			(point - center).dot(axis),
			-halfHeight, halfHeight);
		const PxVec3 medialPoint =
			center + axis * axisCoordinate;
		const PxVec3 delta = point - medialPoint;
		const PxReal distance = delta.magnitude();
		if(!PxIsFinite(distance))
			return false;
		if(iteration == 0 && distance < expandedRadius)
			return false;
		const PxReal gap = distance - expandedRadius;
		if(gap <= distanceTolerance)
		{
			const PxReal normalMagnitudeSq =
				delta.magnitudeSquared();
			if(normalMagnitudeSq <= 1.0e-12f ||
				!PxIsFinite(normalMagnitudeSq))
				return false;
			result.entryTime = time;
			result.normal =
				delta * PxRecipSqrt(normalMagnitudeSq);
			const PxVec3 normalLocal =
				rotation.getConjugate().rotate(result.normal);
			result.surfaceLocal =
				PxVec3(axisCoordinate, 0.0f, 0.0f) +
				normalLocal * capsuleRadius;
			return result.normal.isFinite() &&
				result.surfaceLocal.isFinite();
		}
		const PxReal nextTime = time + gap / speed;
		if(!PxIsFinite(nextTime) || nextTime > 1.0f ||
			nextTime <= time)
			return false;
		time = nextTime;
	}
	return false;
}

// Exact segment entry against a static capsule whose medial segment is the
// shape-local X interval [-halfHeight, halfHeight]. The query segment is
// already expressed in the common capsule frame; callers must fail closed
// when the capsule rotates during the timestep.
PX_FORCE_INLINE bool avbdSegmentEnterExpandedCapsule(
	const PxVec3& segmentStart, const PxVec3& segmentEnd,
	PxReal halfHeight, PxReal expandedRadius,
	PxReal& entryTime, PxVec3& entryNormalLocal,
	PxVec3& medialPointLocal)
{
	if(!segmentStart.isFinite() || !segmentEnd.isFinite() ||
		halfHeight < 0.0f || expandedRadius <= 0.0f ||
		!PxIsFinite(halfHeight) || !PxIsFinite(expandedRadius))
		return false;

	const PxVec3 direction = segmentEnd - segmentStart;
	const PxReal directionMagnitudeSq =
		direction.magnitudeSquared();
	if(directionMagnitudeSq <= 1.0e-12f ||
		!PxIsFinite(directionMagnitudeSq))
		return false;

	PxReal bestTime = PX_MAX_F32;
	PxVec3 bestNormal(0.0f);
	PxVec3 bestMedial(0.0f);

	// Infinite-cylinder entry, restricted to the finite medial interval.
	const PxReal cylinderA =
		direction.y * direction.y +
		direction.z * direction.z;
	const PxReal cylinderHalfB =
		segmentStart.y * direction.y +
		segmentStart.z * direction.z;
	const PxReal cylinderC =
		segmentStart.y * segmentStart.y +
		segmentStart.z * segmentStart.z -
			expandedRadius * expandedRadius;
	if(cylinderA > 1.0e-12f && cylinderC >= 0.0f)
	{
		const PxReal discriminant =
			cylinderHalfB * cylinderHalfB -
				cylinderA * cylinderC;
		if(discriminant >= 0.0f &&
			PxIsFinite(discriminant))
		{
			const PxReal candidateTime =
				(-cylinderHalfB - PxSqrt(discriminant)) /
					cylinderA;
			if(candidateTime >= 0.0f &&
				candidateTime <= 1.0f)
			{
				const PxVec3 candidate =
					segmentStart + direction * candidateTime;
				if(candidate.x >= -halfHeight &&
					candidate.x <= halfHeight)
				{
					const PxVec3 radial(
						0.0f, candidate.y, candidate.z);
					const PxReal radialMagnitudeSq =
						radial.magnitudeSquared();
					if(radialMagnitudeSq > 1.0e-12f &&
						PxIsFinite(radialMagnitudeSq))
					{
						bestTime = candidateTime;
						bestNormal = radial *
							PxRecipSqrt(radialMagnitudeSq);
						bestMedial =
							PxVec3(candidate.x, 0.0f, 0.0f);
					}
				}
			}
		}
	}

	// Full endpoint spheres plus the finite cylinder form the capsule union.
	for(PxU32 endpoint = 0; endpoint < 2; ++endpoint)
	{
		const PxVec3 capCenter(
			endpoint == 0 ? -halfHeight : halfHeight,
			0.0f, 0.0f);
		PxReal candidateTime = 0.0f;
		PxVec3 candidateNormal(0.0f);
		if(avbdSegmentEnterExpandedSphere(
				segmentStart, segmentEnd, capCenter,
				expandedRadius, candidateTime,
				candidateNormal) &&
			candidateTime < bestTime)
		{
			bestTime = candidateTime;
			bestNormal = candidateNormal;
			bestMedial = capCenter;
		}
	}

	if(bestTime == PX_MAX_F32 ||
		!bestNormal.isFinite() ||
		bestNormal.magnitudeSquared() <= 1.0e-12f)
		return false;
	entryTime = bestTime;
	entryNormalLocal = bestNormal.getNormalized();
	medialPointLocal = bestMedial;
	return true;
}

// P5.14a candidate leaf: swept capsule SDF is particle-major and carries all
// conservative-advancement state on the stack. A caller that partitions it
// must merge the full current-SDF family before its swept-family ranges.
inline void avbdDetectSoftRigidCapsuleSweptSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	PX_ASSERT(particleBegin <= particleEnd && particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for(PxU32 particleIndex = particleBegin;
		particleIndex < particleEnd; ++particleIndex)
	{
		const AvbdSoftParticle& particle =
			particles[particleIndex];
		if(particle.invMass <= 0.0f ||
			!particle.position.isFinite() ||
			!particle.predictedPosition.isFinite())
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies, particleIndex);
		if(!sourceBody ||
			!sourceBody->compiled.speculativeCCDEnabled ||
			!avbdIsSoftBodySurfaceVertex(
				*sourceBody, particleIndex))
			continue;

		for(PxU32 capsuleIndex = 0;
			capsuleIndex < numCapsules; ++capsuleIndex)
		{
			const AvbdRigidCapsule& capsule =
				capsules[capsuleIndex];
			const bool kinematicTarget =
				capsule.targetKind ==
					AvbdSoftContactTargetKind::eKINEMATIC_RIGID;
			const bool dynamicTarget =
				capsule.targetKind ==
					AvbdSoftContactTargetKind::eRIGID_BODY;
			PxVec3 centerStart(0.0f);
			PxVec3 centerEnd(0.0f);
			PxQuat rotationStart(PxIdentity);
			PxQuat rotationEnd(PxIdentity);
			bool rotationsEquivalent = false;
			if(!avbdGetRigidCapsuleSweepPose(
					capsule, centerStart, centerEnd,
					rotationStart, rotationEnd,
					rotationsEquivalent) ||
				(!rotationsEquivalent &&
				 !kinematicTarget && !dynamicTarget))
				continue;

			PxVec3 contactNormal(0.0f);
			PxVec3 surfaceLocal(0.0f);
			if(rotationsEquivalent)
			{
				// With a fixed orientation, both moving endpoints share one
				// exact capsule-local frame. Translation of either object is
				// represented by the relative point segment.
				const PxQuat inverseRotation =
					rotationEnd.getConjugate();
				const PxVec3 relativeStart =
					inverseRotation.rotate(
						particle.position - centerStart);
				const PxVec3 relativeEnd =
					inverseRotation.rotate(
						particle.predictedPosition - centerEnd);
				const PxVec3 currentAxis(
					PxClamp(
						relativeStart.x,
						-capsule.halfHeight,
						capsule.halfHeight),
					0.0f, 0.0f);
				const PxReal currentSdf =
					(relativeStart - currentAxis).magnitude() -
						capsule.radius;
				if(!PxIsFinite(currentSdf) || currentSdf < margin)
					continue;

				PxReal entryTime = 0.0f;
				PxVec3 entryNormalLocal(0.0f);
				PxVec3 medialPointLocal(0.0f);
				if(!avbdSegmentEnterExpandedCapsule(
						relativeStart, relativeEnd,
						capsule.halfHeight,
						capsule.radius + margin,
						entryTime, entryNormalLocal,
						medialPointLocal))
					continue;
				contactNormal =
					rotationEnd.rotate(entryNormalLocal).
						getNormalized();
				surfaceLocal =
					medialPointLocal +
						entryNormalLocal * capsule.radius;
			}
			else
			{
				AvbdSweptRotatingCapsulePointEntry entry;
				if(!avbdSegmentEnterExpandedRotatingCapsule(
						particle.position,
						particle.predictedPosition,
						centerStart, centerEnd,
						rotationStart, rotationEnd,
						capsule.halfHeight, capsule.radius,
						margin, entry))
					continue;
				contactNormal = entry.normal;
				surfaceLocal = entry.surfaceLocal;
			}

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF,
				PX_MAX_U32, capsule.primitiveKey, 1u);
			geometry.particleIdx = particleIndex;
			geometry.normal = contactNormal;
			geometry.projNormal = geometry.normal;
			geometry.depth = 0.0f;
			geometry.margin = margin;
			avbdConfigureRigidCapsuleTarget(
				geometry, capsule, capsuleIndex,
				surfaceLocal);
			geometry.friction =
				avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction,
					capsule.friction,
					capsule.frictionCombineMode);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1e6f, 1e6f,
				particles, contacts);
		}
	}
}

inline void avbdDetectSoftRigidCapsuleSweptSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	avbdDetectSoftRigidCapsuleSweptSDFRange(
		particles, numParticles, 0, numParticles,
		capsules, numCapsules, contacts, margin,
		softBodies, numSoftBodies);
}

struct AvbdSweptCapsuleTriangleEntry
{
	PxReal entryTime;
	PxVec3 normal;
	PxVec3 barycentric;
	PxReal segmentWeight1;
	AvbdClosestFeature feature;
	PxU32 featureIndex;

	AvbdSweptCapsuleTriangleEntry()
		: entryTime(PX_MAX_F32), normal(0.0f),
		  barycentric(1.0f, 0.0f, 0.0f), segmentWeight1(0.0f),
		  feature(AVBD_FEATURE_UNKNOWN), featureIndex(0)
	{
	}
};

// Continuous entry of a translated finite segment into a triangle expanded
// by a radius. Exact segment/triangle distance queries drive conservative
// advancement, so every step is bounded by the relative translation speed.
// Soft-triangle vertices are deliberately excluded: the forward
// soft-vertex/capsule swept SDF is their unique owner.
PX_FORCE_INLINE bool
avbdTranslatedSegmentEnterExpandedTriangleNonVertex(
	const PxVec3& segment0, const PxVec3& segment1,
	const PxVec3& translation,
	const PxVec3& a, const PxVec3& b, const PxVec3& c,
	PxReal expandedRadius,
	AvbdSweptCapsuleTriangleEntry& result)
{
	if(!segment0.isFinite() || !segment1.isFinite() ||
		!translation.isFinite() || !a.isFinite() ||
		!b.isFinite() || !c.isFinite() ||
		expandedRadius <= 0.0f || !PxIsFinite(expandedRadius))
		return false;
	const PxReal speedSq = translation.magnitudeSquared();
	if(speedSq <= 1.0e-12f || !PxIsFinite(speedSq))
		return false;
	const PxReal speed = PxSqrt(speedSq);
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, expandedRadius * 1.0e-5f);

	const AvbdClosestSegmentTriangleResult currentClosest =
		avbdClosestSegmentTriangleOGC(
			segment0, segment1, a, b, c);
	if(!PxIsFinite(currentClosest.distance) ||
		currentClosest.distance < expandedRadius)
		return false;

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 48; ++iteration)
	{
		const PxVec3 offset = translation * time;
		const AvbdClosestSegmentTriangleResult closest =
			avbdClosestSegmentTriangleOGC(
				segment0 + offset, segment1 + offset,
				a, b, c);
		if(!PxIsFinite(closest.distance))
			return false;
		const PxReal gap = closest.distance - expandedRadius;
		if(gap <= distanceTolerance)
		{
			if(closest.feature != AVBD_FEATURE_FACE &&
				closest.feature != AVBD_FEATURE_EDGE)
				return false;
			PxVec3 normal =
				closest.trianglePoint - closest.segmentPoint;
			const PxReal normalMagnitudeSq =
				normal.magnitudeSquared();
			if(!normal.isFinite() ||
				normalMagnitudeSq <= 1.0e-12f)
				return false;
			result.entryTime = time;
			result.normal =
				normal * PxRecipSqrt(normalMagnitudeSq);
			result.barycentric = closest.barycentric;
			result.segmentWeight1 =
				PxClamp(closest.segmentWeight1, 0.0f, 1.0f);
			result.feature = closest.feature;
			result.featureIndex = closest.featureIndex;
			return true;
		}

		const PxReal nextTime = time + gap / speed;
		if(!PxIsFinite(nextTime) || nextTime > 1.0f)
			return false;
		if(nextTime <= time)
			return false;
		time = nextTime;
	}
	return false;
}

// Continuous entry of a rotating/translating finite segment into a triangle
// expanded by a radius. The triangle is expressed in a frame with its common
// translation removed. Relative center translation plus
// halfHeight*angularDistance bounds the Hausdorff speed of the medial
// segment. Soft-triangle vertices remain owned by the forward capsule SDF.
PX_FORCE_INLINE bool
avbdRotatingSegmentEnterExpandedTriangleNonVertex(
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	PxReal halfHeight,
	const PxVec3& a, const PxVec3& b, const PxVec3& c,
	PxReal expandedRadius,
	AvbdSweptCapsuleTriangleEntry& result)
{
	if(!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite() ||
		halfHeight < 0.0f || !PxIsFinite(halfHeight) ||
		!a.isFinite() || !b.isFinite() || !c.isFinite() ||
		expandedRadius <= 0.0f || !PxIsFinite(expandedRadius))
		return false;

	PxReal angularDistance = 0.0f;
	if(!avbdGetSweepAngularDistance(
			rotationStart, rotationEnd, angularDistance))
		return false;
	const PxQuat normalizedStart = rotationStart.getNormalized();
	const PxQuat normalizedEnd = rotationEnd.getNormalized();
	const PxReal speed =
		(centerEnd - centerStart).magnitude() +
		halfHeight * angularDistance;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, expandedRadius * 1.0e-5f);

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 64; ++iteration)
	{
		const PxVec3 center =
			centerStart + (centerEnd - centerStart) * time;
		const PxQuat rotation =
			PxSlerp(time, normalizedStart, normalizedEnd).
				getNormalized();
		if(!center.isFinite() || !rotation.isFinite())
			return false;
		const PxVec3 axisOffset =
			rotation.getBasisVector0() * halfHeight;
		const AvbdClosestSegmentTriangleResult closest =
			avbdClosestSegmentTriangleOGC(
				center - axisOffset, center + axisOffset,
				a, b, c);
		if(!PxIsFinite(closest.distance))
			return false;
		if(iteration == 0 && closest.distance < expandedRadius)
			return false;
		const PxReal gap = closest.distance - expandedRadius;
		if(gap <= distanceTolerance)
		{
			if(closest.feature != AVBD_FEATURE_FACE &&
				closest.feature != AVBD_FEATURE_EDGE)
				return false;
			PxVec3 normal =
				closest.trianglePoint - closest.segmentPoint;
			const PxReal normalMagnitudeSq =
				normal.magnitudeSquared();
			if(!normal.isFinite() ||
				normalMagnitudeSq <= 1.0e-12f)
				return false;
			result.entryTime = time;
			result.normal =
				normal * PxRecipSqrt(normalMagnitudeSq);
			result.barycentric = closest.barycentric;
			result.segmentWeight1 =
				PxClamp(closest.segmentWeight1, 0.0f, 1.0f);
			result.feature = closest.feature;
			result.featureIndex = closest.featureIndex;
			return true;
		}

		const PxReal nextTime = time + gap / speed;
		if(!PxIsFinite(nextTime) || nextTime > 1.0f ||
			nextTime <= time)
			return false;
		time = nextTime;
	}
	return false;
}

// Continuous rotating/translating finite segment entry into a linearly
// deforming triangle. A common soft displacement may be removed by the
// caller. Center translation plus endpoint angular speed plus the maximum
// residual soft-vertex speed is a conservative Hausdorff speed bound.
// Triangle vertex caps remain owned by forward rigid-SDF sweeps.
PX_FORCE_INLINE bool
avbdRotatingSegmentEnterExpandedDeformingTriangleNonVertex(
	const PxVec3& rigidLocal0, const PxVec3& rigidLocal1,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	const PxVec3& aStart, const PxVec3& bStart,
	const PxVec3& cStart, const PxVec3& aEnd,
	const PxVec3& bEnd, const PxVec3& cEnd,
	PxReal expandedRadius,
	AvbdSweptCapsuleTriangleEntry& result)
{
	if(!rigidLocal0.isFinite() || !rigidLocal1.isFinite() ||
		!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite() ||
		!aStart.isFinite() || !bStart.isFinite() ||
		!cStart.isFinite() || !aEnd.isFinite() ||
		!bEnd.isFinite() || !cEnd.isFinite() ||
		expandedRadius <= 0.0f || !PxIsFinite(expandedRadius))
		return false;
	PxReal angularDistance = 0.0f;
	if(!avbdGetSweepAngularDistance(
			rotationStart, rotationEnd, angularDistance))
		return false;
	const PxQuat normalizedStart = rotationStart.getNormalized();
	const PxQuat normalizedEnd = rotationEnd.getNormalized();
	const PxVec3 displacementA = aEnd - aStart;
	const PxVec3 displacementB = bEnd - bStart;
	const PxVec3 displacementC = cEnd - cStart;
	const PxReal triangleSpeed = PxMax(
		displacementA.magnitude(),
		PxMax(displacementB.magnitude(),
			displacementC.magnitude()));
	const PxReal segmentRadius = PxMax(
		rigidLocal0.magnitude(), rigidLocal1.magnitude());
	const PxReal speed =
		(centerEnd - centerStart).magnitude() +
		segmentRadius * angularDistance + triangleSpeed;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, expandedRadius * 1.0e-5f);

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 64; ++iteration)
	{
		const PxVec3 center =
			centerStart + (centerEnd - centerStart) * time;
		const PxQuat rotation =
			PxSlerp(time, normalizedStart, normalizedEnd).
				getNormalized();
		const PxVec3 a = aStart + displacementA * time;
		const PxVec3 b = bStart + displacementB * time;
		const PxVec3 c = cStart + displacementC * time;
		const PxVec3 triangleNormal = (b - a).cross(c - a);
		if(!center.isFinite() || !rotation.isFinite() ||
			!a.isFinite() || !b.isFinite() || !c.isFinite() ||
			!triangleNormal.isFinite() ||
			triangleNormal.magnitudeSquared() <= 1.0e-16f)
			return false;
		const PxVec3 rigid0 =
			center + rotation.rotate(rigidLocal0);
		const PxVec3 rigid1 =
			center + rotation.rotate(rigidLocal1);
		const AvbdClosestSegmentTriangleResult closest =
			avbdClosestSegmentTriangleOGC(
				rigid0, rigid1, a, b, c);
		if(!PxIsFinite(closest.distance))
			return false;
		if(iteration == 0 &&
			closest.distance < expandedRadius)
			return false;
		const PxReal gap = closest.distance - expandedRadius;
		if(gap <= distanceTolerance)
		{
			if(closest.feature != AVBD_FEATURE_FACE &&
				closest.feature != AVBD_FEATURE_EDGE)
				return false;
			const PxVec3 normal =
				closest.trianglePoint - closest.segmentPoint;
			const PxReal normalMagnitudeSq =
				normal.magnitudeSquared();
			if(!normal.isFinite() ||
				normalMagnitudeSq <= 1.0e-12f)
				return false;
			result.entryTime = time;
			result.normal =
				normal * PxRecipSqrt(normalMagnitudeSq);
			result.barycentric = closest.barycentric;
			result.segmentWeight1 =
				PxClamp(closest.segmentWeight1, 0.0f, 1.0f);
			result.feature = closest.feature;
			result.featureIndex = closest.featureIndex;
			return true;
		}
		const PxReal nextTime = time + gap / speed;
		if(!PxIsFinite(nextTime) || nextTime > 1.0f ||
			nextTime <= time)
			return false;
		time = nextTime;
	}
	return false;
}

PX_NOINLINE inline bool avbdRigidCapsuleForwardVertexOwnsSweptFeature(
	const AvbdSoftParticle& particle,
	const AvbdRigidCapsule& capsule,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	const PxQuat& inverseRotationEnd,
	bool rotationsEquivalent, PxReal margin)
{
	if(particle.invMass <= 0.0f)
		return false;
	const PxVec3 pointStart = particle.initialPosition;
	const PxVec3 pointEnd = particle.predictedPosition;
	if(rotationsEquivalent)
	{
		const PxVec3 relativeStart =
			inverseRotationEnd.rotate(pointStart - centerStart);
		const PxVec3 relativeEnd =
			inverseRotationEnd.rotate(pointEnd - centerEnd);
		const PxVec3 currentAxis(
			PxClamp(relativeStart.x,
				-capsule.halfHeight, capsule.halfHeight),
			0.0f, 0.0f);
		const PxReal currentSdf =
			(relativeStart - currentAxis).magnitude() -
				capsule.radius;
		if(!PxIsFinite(currentSdf))
			return false;
		if(currentSdf < margin)
			return true;
		PxReal vertexEntryTime = 0.0f;
		PxVec3 vertexEntryNormal(0.0f);
		PxVec3 vertexMedialPoint(0.0f);
		return avbdSegmentEnterExpandedCapsule(
			relativeStart, relativeEnd,
			capsule.halfHeight, capsule.radius + margin,
			vertexEntryTime, vertexEntryNormal,
			vertexMedialPoint);
	}

	const PxVec3 startAxis = rotationStart.getBasisVector0();
	const PxReal axisCoordinate = PxClamp(
		(pointStart - centerStart).dot(startAxis),
		-capsule.halfHeight, capsule.halfHeight);
	const PxReal currentSdf =
		(pointStart - (centerStart + startAxis * axisCoordinate)).
			magnitude() - capsule.radius;
	if(!PxIsFinite(currentSdf))
		return false;
	if(currentSdf < margin)
		return true;
	AvbdSweptRotatingCapsulePointEntry vertexEntry;
	return avbdSegmentEnterExpandedRotatingCapsule(
		pointStart, pointEnd,
		centerStart, centerEnd,
		rotationStart, rotationEnd,
		capsule.halfHeight, capsule.radius, margin,
		vertexEntry);
}

inline void avbdDetectSoftRigidCapsuleSweptOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	PxArray<PxU8>* persistentForwardOwnerScratch = NULL)
{
	const PxReal translationToleranceSq = 1.0e-10f;
	PxArray<PxU8> localForwardOwnerScratch;
	PxArray<PxU8>& forwardOwnerScratch =
		persistentForwardOwnerScratch
			? *persistentForwardOwnerScratch
			: localForwardOwnerScratch;
	if(forwardOwnerScratch.capacity() < numParticles)
		forwardOwnerScratch.reserve(numParticles);
	forwardOwnerScratch.resize(numParticles);
	for(PxU32 bodyIndex = 0;
		bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		if(!body.compiled.speculativeCCDEnabled)
			continue;
		for(PxU32 capsuleIndex = 0;
			capsuleIndex < numCapsules; ++capsuleIndex)
		{
			const AvbdRigidCapsule& capsule =
				capsules[capsuleIndex];
			const bool kinematicTarget =
				capsule.targetKind ==
					AvbdSoftContactTargetKind::eKINEMATIC_RIGID;
			const bool dynamicTarget =
				capsule.targetKind ==
					AvbdSoftContactTargetKind::eRIGID_BODY;
			if(capsule.radius <= 0.0f ||
				capsule.halfHeight < 0.0f ||
				!PxIsFinite(capsule.radius) ||
				!PxIsFinite(capsule.halfHeight) ||
				!capsule.center.isFinite() ||
				!capsule.rotation.isFinite() ||
				(capsule.targetKind !=
						AvbdSoftContactTargetKind::eWORLD_STATIC &&
				 !kinematicTarget && !dynamicTarget) ||
				(kinematicTarget &&
				 (!capsule.previousCenter.isFinite() ||
				  !capsule.previousRotation.isFinite())) ||
				(dynamicTarget &&
				 (!capsule.predictedPoseValid ||
				  !capsule.predictedCenter.isFinite() ||
				  !capsule.predictedRotation.isFinite())))
				continue;

			const PxVec3 centerStart =
				kinematicTarget
					? capsule.previousCenter : capsule.center;
			const PxVec3 centerEnd =
				dynamicTarget
					? capsule.predictedCenter : capsule.center;
			const PxQuat rotationStart =
				kinematicTarget
					? capsule.previousRotation : capsule.rotation;
			const PxQuat rotationEnd =
				dynamicTarget
					? capsule.predictedRotation : capsule.rotation;
			if(!centerStart.isFinite() || !centerEnd.isFinite() ||
				!rotationStart.isFinite() || !rotationEnd.isFinite())
				continue;

			const bool rotationsEquivalent =
				avbdAreSweepRotationsEquivalent(
					rotationStart, rotationEnd);
			const PxQuat inverseRotation =
				rotationEnd.getConjugate();
			const PxVec3 axisOffset =
				rotationEnd.getBasisVector0() *
					capsule.halfHeight;
			const PxVec3 segment0 = centerStart - axisOffset;
			const PxVec3 segment1 = centerStart + axisOffset;
			const PxReal expandedRadius =
				capsule.radius + margin;
			const PxU32 particleStart = body.compiled.particleStart;
			const PxU32 particleCount = body.compiled.particleCount;
			if(particleStart <= numParticles)
			{
				const PxU32 boundedParticleCount = PxMin(
					particleCount, numParticles - particleStart);
				for(PxU32 localParticle = 0;
					localParticle < boundedParticleCount; ++localParticle)
					forwardOwnerScratch[
						particleStart + localParticle] = 0;
			}
			for(PxU32 surfaceVertexIndex = 0;
				surfaceVertexIndex < body.compiled.surfaceVertices.size();
				++surfaceVertexIndex)
			{
				const PxU32 vertexIndex =
					body.compiled.surfaceVertices[surfaceVertexIndex];
				if(vertexIndex >= numParticles)
					continue;
				forwardOwnerScratch[vertexIndex] = PxU8(
					avbdRigidCapsuleForwardVertexOwnsSweptFeature(
						particles[vertexIndex], capsule,
						centerStart, centerEnd,
						rotationStart, rotationEnd,
						inverseRotation, rotationsEquivalent,
						margin));
			}
			PxArray<PxU64> emittedFeatureKeys;
			for(PxU32 triangleOffset = 0;
				triangleOffset + 2 <
					body.compiled.surfaceTriangles.size();
				triangleOffset += 3)
			{
				const PxU32 v0 =
					body.compiled.surfaceTriangles[triangleOffset];
				const PxU32 v1 =
					body.compiled.surfaceTriangles[
						triangleOffset + 1];
				const PxU32 v2 =
					body.compiled.surfaceTriangles[
						triangleOffset + 2];
				if(v0 >= numParticles || v1 >= numParticles ||
					v2 >= numParticles ||
					(particles[v0].invMass <= 0.0f &&
					 particles[v1].invMass <= 0.0f &&
					 particles[v2].invMass <= 0.0f))
					continue;

				const PxVec3 p0 = particles[v0].initialPosition;
				const PxVec3 p1 = particles[v1].initialPosition;
				const PxVec3 p2 = particles[v2].initialPosition;
				const PxVec3 displacement0 =
					particles[v0].predictedPosition - p0;
				const PxVec3 displacement1 =
					particles[v1].predictedPosition - p1;
				const PxVec3 displacement2 =
					particles[v2].predictedPosition - p2;
				if(!p0.isFinite() || !p1.isFinite() ||
					!p2.isFinite() ||
					!displacement0.isFinite() ||
					!displacement1.isFinite() ||
					!displacement2.isFinite())
					continue;
				const PxVec3 relativeDisplacement1 =
					(displacement1 - displacement0);
				const PxVec3 relativeDisplacement2 =
					(displacement2 - displacement0);
				const bool softTriangleTranslationOnly =
					relativeDisplacement1.magnitudeSquared() <=
						translationToleranceSq &&
					relativeDisplacement2.magnitudeSquared() <=
						translationToleranceSq;

				const bool forwardVertexOwns =
					forwardOwnerScratch[v0] != 0 ||
					forwardOwnerScratch[v1] != 0 ||
					forwardOwnerScratch[v2] != 0;
				if(forwardVertexOwns)
					continue;

				const PxVec3 relativeTranslation =
					centerEnd - centerStart - displacement0;
				const PxVec3 relativeCenterEnd =
					centerEnd - displacement0;
				PxVec3 sweptMinimum(0.0f);
				PxVec3 sweptMaximum(0.0f);
				if(rotationsEquivalent)
				{
					const PxVec3 segment0End =
						segment0 + relativeTranslation;
					const PxVec3 segment1End =
						segment1 + relativeTranslation;
					sweptMinimum =
						segment0.minimum(segment1).
							minimum(segment0End).minimum(segment1End) -
								PxVec3(expandedRadius);
					sweptMaximum =
						segment0.maximum(segment1).
							maximum(segment0End).maximum(segment1End) +
								PxVec3(expandedRadius);
				}
				else
				{
					const PxReal rotationExtent =
						capsule.halfHeight + expandedRadius;
					sweptMinimum =
						centerStart.minimum(relativeCenterEnd) -
							PxVec3(rotationExtent);
					sweptMaximum =
						centerStart.maximum(relativeCenterEnd) +
							PxVec3(rotationExtent);
				}
				const PxVec3 triangleMinimum =
					p0.minimum(p1).minimum(p2).
						minimum(
							p1 + relativeDisplacement1).
						minimum(
							p2 + relativeDisplacement2);
				const PxVec3 triangleMaximum =
					p0.maximum(p1).maximum(p2).
						maximum(
							p1 + relativeDisplacement1).
						maximum(
							p2 + relativeDisplacement2);
				if(sweptMinimum.x > triangleMaximum.x ||
					sweptMaximum.x < triangleMinimum.x ||
					sweptMinimum.y > triangleMaximum.y ||
					sweptMaximum.y < triangleMinimum.y ||
					sweptMinimum.z > triangleMaximum.z ||
					sweptMaximum.z < triangleMinimum.z)
					continue;

				AvbdSweptCapsuleTriangleEntry entry;
				const bool entered =
					softTriangleTranslationOnly &&
						rotationsEquivalent
						? avbdTranslatedSegmentEnterExpandedTriangleNonVertex(
							segment0, segment1, relativeTranslation,
							p0, p1, p2, expandedRadius, entry)
						: softTriangleTranslationOnly
						? avbdRotatingSegmentEnterExpandedTriangleNonVertex(
							centerStart, relativeCenterEnd,
							rotationStart, rotationEnd,
							capsule.halfHeight,
							p0, p1, p2, expandedRadius, entry)
						: avbdRotatingSegmentEnterExpandedDeformingTriangleNonVertex(
							PxVec3(-capsule.halfHeight, 0.0f, 0.0f),
							PxVec3(capsule.halfHeight, 0.0f, 0.0f),
							centerStart, relativeCenterEnd,
							rotationStart, rotationEnd,
							p0, p1, p2, p0,
							p1 + relativeDisplacement1,
							p2 + relativeDisplacement2,
							expandedRadius, entry);
				if(!entered)
					continue;

				const PxU64 softFeatureKey =
					avbdSoftTriangleFeatureKey(
						v0, v1, v2,
						entry.feature, entry.featureIndex);
				PxU64 featureKey = 1469598103934665603ull;
				featureKey = avbdSoftContactHashValue(
					featureKey, 0x43505257u);
				featureKey = avbdSoftContactHashValue(
					featureKey, PxU32(softFeatureKey));
				featureKey = avbdSoftContactHashValue(
					featureKey, PxU32(softFeatureKey >> 32));
				bool duplicate = false;
				for(PxU32 emittedIndex = 0;
					emittedIndex < emittedFeatureKeys.size();
					++emittedIndex)
				{
					if(emittedFeatureKeys[emittedIndex] ==
						featureKey)
					{
						duplicate = true;
						break;
					}
				}
				if(duplicate)
					continue;
				emittedFeatureKeys.pushBack(featureKey);

				AvbdSoftContactGeometry geometry;
				geometry.source = AvbdSoftContactSource(
					AvbdSoftContactSource::eRIGID_SDF,
					PX_MAX_U32, capsule.primitiveKey,
					featureKey);
				geometry.particleIdx =
					particles[v0].invMass > 0.0f ? v0 :
						(particles[v1].invMass > 0.0f
							? v1 : v2);
				geometry.queryParticleIndices[0] = v0;
				geometry.queryParticleIndices[1] = v1;
				geometry.queryParticleIndices[2] = v2;
				geometry.queryWeights[0] = entry.barycentric.x;
				geometry.queryWeights[1] = entry.barycentric.y;
				geometry.queryWeights[2] = entry.barycentric.z;
				geometry.normal = entry.normal;
				geometry.projNormal = entry.normal;
				geometry.depth = 0.0f;
				geometry.margin = margin;
				const PxVec3 medialPointLocal(
					-capsule.halfHeight +
						2.0f * capsule.halfHeight *
							entry.segmentWeight1,
					0.0f, 0.0f);
				const PxQuat entryRotation =
					rotationsEquivalent
						? rotationEnd.getNormalized()
						: PxSlerp(
							entry.entryTime,
							rotationStart.getNormalized(),
							rotationEnd.getNormalized()).
								getNormalized();
				const PxVec3 surfaceLocal =
					medialPointLocal +
						entryRotation.getConjugate().
							rotate(entry.normal) *
							capsule.radius;
				avbdConfigureRigidCapsuleTarget(
					geometry, capsule, capsuleIndex,
					surfaceLocal);
				geometry.friction =
					avbdCombineDeformableRigidFriction(
						body.material.dynamicFriction,
						capsule.friction,
						capsule.frictionCombineMode);
				avbdBuildSoftContactTangents(geometry);
				avbdAppendPreparedSoftContact(
					geometry, 1.0e6f, 1.0e6f,
					particles, contacts);
			}
		}
	}
}

inline void avbdDetectSoftRigidCapsuleOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f)
{
	const PxReal normalEpsilon = 1.0e-12f;
	for(PxU32 bodyIndex = 0;
		bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		PxVec3 bodyMinimum(PX_MAX_F32);
		PxVec3 bodyMaximum(-PX_MAX_F32);
		for(PxU32 localParticle = 0;
			localParticle < body.compiled.particleCount;
			++localParticle)
		{
			const PxU32 particleIndex =
				body.compiled.particleStart + localParticle;
			if(particleIndex >= numParticles)
				continue;
			bodyMinimum = bodyMinimum.minimum(
				particles[particleIndex].position);
			bodyMaximum = bodyMaximum.maximum(
				particles[particleIndex].position);
		}

		for(PxU32 capsuleIndex = 0;
			capsuleIndex < numCapsules; ++capsuleIndex)
		{
			const AvbdRigidCapsule& capsule =
				capsules[capsuleIndex];
			if(capsule.radius <= 0.0f ||
				capsule.halfHeight < 0.0f ||
				!PxIsFinite(capsule.radius) ||
				!PxIsFinite(capsule.halfHeight) ||
				!capsule.center.isFinite() ||
				!capsule.rotation.isFinite())
				continue;
			const PxVec3 axisOffset =
				capsule.rotation.getBasisVector0() *
					capsule.halfHeight;
			const PxVec3 segment0 =
				capsule.center - axisOffset;
			const PxVec3 segment1 =
				capsule.center + axisOffset;
			const PxReal queryRadius =
				capsule.radius + margin;
			const PxVec3 capsuleMinimum =
				segment0.minimum(segment1) -
					PxVec3(queryRadius);
			const PxVec3 capsuleMaximum =
				segment0.maximum(segment1) +
					PxVec3(queryRadius);
			if(bodyMinimum.x > capsuleMaximum.x ||
				bodyMaximum.x < capsuleMinimum.x ||
				bodyMinimum.y > capsuleMaximum.y ||
				bodyMaximum.y < capsuleMinimum.y ||
				bodyMinimum.z > capsuleMaximum.z ||
				bodyMaximum.z < capsuleMinimum.z)
				continue;

			PxArray<PxU64> emittedFeatureKeys;
			for(PxU32 triangleOffset = 0;
				triangleOffset + 2 <
					body.compiled.surfaceTriangles.size();
				triangleOffset += 3)
			{
				const PxU32 v0 =
					body.compiled.surfaceTriangles[triangleOffset];
				const PxU32 v1 =
					body.compiled.surfaceTriangles[
						triangleOffset + 1];
				const PxU32 v2 =
					body.compiled.surfaceTriangles[
						triangleOffset + 2];
				if(v0 >= numParticles || v1 >= numParticles ||
					v2 >= numParticles ||
					(particles[v0].invMass <= 0.0f &&
					 particles[v1].invMass <= 0.0f &&
					 particles[v2].invMass <= 0.0f))
					continue;
				const PxVec3& p0 = particles[v0].position;
				const PxVec3& p1 = particles[v1].position;
				const PxVec3& p2 = particles[v2].position;
				const PxVec3 triangleMinimum =
					p0.minimum(p1).minimum(p2) -
						PxVec3(queryRadius);
				const PxVec3 triangleMaximum =
					p0.maximum(p1).maximum(p2) +
						PxVec3(queryRadius);
				if(segment0.minimum(segment1).x >
						triangleMaximum.x ||
					segment0.maximum(segment1).x <
						triangleMinimum.x ||
					segment0.minimum(segment1).y >
						triangleMaximum.y ||
					segment0.maximum(segment1).y <
						triangleMinimum.y ||
					segment0.minimum(segment1).z >
						triangleMaximum.z ||
					segment0.maximum(segment1).z <
						triangleMinimum.z)
					continue;

				const AvbdClosestSegmentTriangleResult closest =
					avbdClosestSegmentTriangleOGC(
						segment0, segment1, p0, p1, p2);
				// Soft vertices are exclusively owned by the forward
				// vertex/capsule SDF. Reverse ownership fills only edge/face
				// gaps, including a medial segment under a coarse face.
				if(closest.feature == AVBD_FEATURE_VERTEX ||
					closest.feature == AVBD_FEATURE_UNKNOWN ||
					!PxIsFinite(closest.distance) ||
					closest.distance >= queryRadius)
					continue;

				const PxU64 softFeatureKey =
					avbdSoftTriangleFeatureKey(
						v0, v1, v2,
						closest.feature,
						closest.featureIndex);
				PxU64 featureKey = 1469598103934665603ull;
				featureKey = avbdSoftContactHashValue(
					featureKey, 0x4350534cu);
				featureKey = avbdSoftContactHashValue(
					featureKey, PxU32(softFeatureKey));
				featureKey = avbdSoftContactHashValue(
					featureKey, PxU32(softFeatureKey >> 32));
				bool duplicate = false;
				for(PxU32 emittedIndex = 0;
					emittedIndex < emittedFeatureKeys.size();
					++emittedIndex)
				{
					if(emittedFeatureKeys[emittedIndex] ==
						featureKey)
					{
						duplicate = true;
						break;
					}
				}
				if(duplicate)
					continue;
				emittedFeatureKeys.pushBack(featureKey);

				PxVec3 normal =
					closest.trianglePoint -
						closest.segmentPoint;
				PxReal normalMagnitudeSq =
					normal.magnitudeSquared();
				if(normalMagnitudeSq <= normalEpsilon ||
					!PxIsFinite(normalMagnitudeSq))
				{
					normal = (p1 - p0).cross(p2 - p0);
					normalMagnitudeSq =
						normal.magnitudeSquared();
					if(normalMagnitudeSq <= normalEpsilon ||
						!PxIsFinite(normalMagnitudeSq))
						continue;
					normal *= PxRecipSqrt(normalMagnitudeSq);
					const PxVec3 triangleCentroid =
						(p0 + p1 + p2) * (1.0f / 3.0f);
					if(normal.dot(
						triangleCentroid - capsule.center) < 0.0f)
						normal = -normal;
				}
				else
					normal *= PxRecipSqrt(normalMagnitudeSq);

				AvbdSoftContactGeometry geometry;
				geometry.source = AvbdSoftContactSource(
					AvbdSoftContactSource::eRIGID_SDF,
					PX_MAX_U32, capsule.primitiveKey,
					featureKey);
				geometry.particleIdx =
					particles[v0].invMass > 0.0f ? v0 :
						(particles[v1].invMass > 0.0f
							? v1 : v2);
				geometry.queryParticleIndices[0] = v0;
				geometry.queryParticleIndices[1] = v1;
				geometry.queryParticleIndices[2] = v2;
				geometry.queryWeights[0] =
					closest.barycentric.x;
				geometry.queryWeights[1] =
					closest.barycentric.y;
				geometry.queryWeights[2] =
					closest.barycentric.z;
				geometry.normal = normal;
				geometry.projNormal = normal;
				geometry.depth =
					queryRadius - closest.distance;
				geometry.margin = margin;
				const PxVec3 surfaceWorld =
					closest.segmentPoint +
						normal * capsule.radius;
				const PxVec3 surfaceLocal =
					capsule.rotation.getConjugate().rotate(
						surfaceWorld - capsule.center);
				avbdConfigureRigidCapsuleTarget(
					geometry, capsule, capsuleIndex,
					surfaceLocal);
				geometry.friction =
					avbdCombineDeformableRigidFriction(
						body.material.dynamicFriction,
						capsule.friction,
						capsule.frictionCombineMode);
				avbdBuildSoftContactTangents(geometry);
				avbdAppendPreparedSoftContact(
					geometry, 1e5f, 1e6f,
					particles, contacts);
			}
		}
	}
}

PX_FORCE_INLINE bool avbdIsRigidConvexValid(
	const AvbdRigidConvex& convex)
{
	return convex.center.isFinite() &&
		convex.rotation.isFinite() &&
		PxIsFinite(convex.localRadius) &&
		convex.localRadius > 0.0f &&
		convex.vertices.size() >= 4 &&
		!convex.faces.empty() &&
		!convex.triangles.empty();
}

PX_FORCE_INLINE void avbdConfigureRigidConvexTarget(
	AvbdSoftContactGeometry& geometry,
	const AvbdRigidConvex& convex, PxU32 convexIndex,
	const PxVec3& surfaceLocal)
{
	geometry.targetKind = convex.targetKind;
	geometry.velocityOwner =
		convex.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID
			? AvbdVelocityObjectiveOwner::ComponentFinalize
			: convex.targetKind ==
				AvbdSoftContactTargetKind::eRIGID_BODY
				? AvbdVelocityObjectiveOwner::ManifoldFinalize
				: AvbdVelocityObjectiveOwner::PositionAL;
	geometry.targetIndex =
		convex.targetKind ==
			AvbdSoftContactTargetKind::eRIGID_BODY
			? convex.targetIndex : convexIndex;
	geometry.surfacePoint =
		convex.center + convex.rotation.rotate(surfaceLocal);
	geometry.kinematicSurfacePointPrevious =
		convex.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID
			? convex.previousCenter +
				convex.previousRotation.rotate(surfaceLocal)
			: geometry.surfacePoint;
	if(convex.targetKind ==
		AvbdSoftContactTargetKind::eRIGID_BODY)
		geometry.rigidLocalPoint =
			convex.shapeToRigidBody.transform(surfaceLocal);
}

PX_FORCE_INLINE PxU64 avbdRigidConvexFeatureKey(
	PxU32 tag, PxU32 triangleOrFaceIndex,
	AvbdClosestFeature feature, PxU32 featureIndex)
{
	PxU64 hash = 1469598103934665603ull;
	hash = avbdSoftContactHashValue(hash, tag);
	hash = avbdSoftContactHashValue(hash, triangleOrFaceIndex);
	hash = avbdSoftContactHashValue(hash, PxU32(feature));
	return avbdSoftContactHashValue(hash, featureIndex);
}

struct AvbdRigidConvexPointQuery
{
	PxReal signedDistance;
	PxVec3 surfaceLocal;
	PxVec3 normalLocal;
	PxU64 featureKey;

	AvbdRigidConvexPointQuery()
		: signedDistance(PX_MAX_F32), surfaceLocal(0.0f),
		  normalLocal(0.0f, 1.0f, 0.0f), featureKey(0)
	{
	}
};

// Exact closed-hull point query shared by discrete and swept vertex owners.
// Negative distance denotes a point inside the convex; outside distance and
// material point come from the closest baked hull triangle.
PX_FORCE_INLINE bool avbdQueryRigidConvexLocal(
	const AvbdRigidConvex& convex, const PxVec3& localPoint,
	AvbdRigidConvexPointQuery& result)
{
	if(!avbdIsRigidConvexValid(convex) || !localPoint.isFinite())
		return false;

	PxReal maximumPlaneDistance = -PX_MAX_F32;
	PxU32 maximumFace = PX_MAX_U32;
	for(PxU32 faceIndex = 0;
		faceIndex < convex.faces.size(); ++faceIndex)
	{
		const AvbdRigidConvexFace& face =
			convex.faces[faceIndex];
		const PxReal planeDistance =
			face.normal.dot(localPoint) - face.offset;
		if(planeDistance > maximumPlaneDistance)
		{
			maximumPlaneDistance = planeDistance;
			maximumFace = faceIndex;
		}
	}
	if(maximumFace == PX_MAX_U32 ||
		!PxIsFinite(maximumPlaneDistance))
		return false;

	if(maximumPlaneDistance <= 0.0f)
	{
		result.signedDistance = maximumPlaneDistance;
		result.normalLocal = convex.faces[maximumFace].normal;
		result.surfaceLocal =
			localPoint -
				result.normalLocal * result.signedDistance;
		result.featureKey = avbdRigidConvexFeatureKey(
			0x43465846u, maximumFace,
			AVBD_FEATURE_FACE, 0u);
	}
	else
	{
		AvbdClosestPointResult bestClosest = {};
		PxReal bestDistance = PX_MAX_F32;
		PxU32 bestTriangle = PX_MAX_U32;
		for(PxU32 triangleIndex = 0;
			triangleIndex < convex.triangles.size();
			++triangleIndex)
		{
			const AvbdRigidConvexTriangle& triangle =
				convex.triangles[triangleIndex];
			if(triangle.p0 >= convex.vertices.size() ||
				triangle.p1 >= convex.vertices.size() ||
				triangle.p2 >= convex.vertices.size())
				continue;
			const AvbdClosestPointResult closest =
				avbdClosestPointOnTriangleOGC(
					localPoint,
					convex.vertices[triangle.p0],
					convex.vertices[triangle.p1],
					convex.vertices[triangle.p2]);
			if(closest.distance < bestDistance)
			{
				bestDistance = closest.distance;
				bestClosest = closest;
				bestTriangle = triangleIndex;
			}
		}
		if(bestTriangle == PX_MAX_U32 ||
			!PxIsFinite(bestDistance))
			return false;
		result.signedDistance = bestDistance;
		result.surfaceLocal = bestClosest.point;
		result.normalLocal = bestClosest.normal;
		const PxReal normalMagnitudeSq =
			result.normalLocal.magnitudeSquared();
		if(!result.normalLocal.isFinite() ||
			normalMagnitudeSq <= 1.0e-12f)
		{
			const PxU32 faceIndex =
				convex.triangles[bestTriangle].faceIndex;
			if(faceIndex >= convex.faces.size())
				return false;
			result.normalLocal =
				convex.faces[faceIndex].normal;
		}
		result.featureKey = avbdRigidConvexFeatureKey(
			0x43465854u, bestTriangle,
			bestClosest.feature,
			bestClosest.featureIndex);
	}

	const PxReal normalMagnitudeSq =
		result.normalLocal.magnitudeSquared();
	if(!result.surfaceLocal.isFinite() ||
		!result.normalLocal.isFinite() ||
		normalMagnitudeSq <= 1.0e-12f ||
		!PxIsFinite(result.signedDistance))
		return false;
	result.normalLocal *= PxRecipSqrt(normalMagnitudeSq);
	return true;
}

// P5.7a candidate leaf: current-pose convex SDF is particle-major, reads
// immutable baked hull topology, and appends only to caller-owned output.
inline void avbdDetectSoftRigidConvexSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	PX_ASSERT(particles && particleBegin <= particleEnd);
	PX_ASSERT(particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for(PxU32 particleIndex = particleBegin;
		particleIndex < particleEnd; ++particleIndex)
	{
		const AvbdSoftParticle& particle = particles[particleIndex];
		if(particle.invMass <= 0.0f ||
			!particle.position.isFinite())
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies, particleIndex);
		if(sourceBody &&
			!avbdIsSoftBodySurfaceVertex(
				*sourceBody, particleIndex))
			continue;

		for(PxU32 convexIndex = 0;
			convexIndex < numConvexes; ++convexIndex)
		{
			const AvbdRigidConvex& convex =
				convexes[convexIndex];
			if(!avbdIsRigidConvexValid(convex))
				continue;
			const PxVec3 worldOffset =
				particle.position - convex.center;
			const PxReal queryRadius =
				convex.localRadius + margin;
			if(worldOffset.magnitudeSquared() >
				queryRadius * queryRadius)
				continue;
			const PxVec3 localPoint =
				convex.rotation.getConjugate().rotate(
					worldOffset);
			AvbdRigidConvexPointQuery query;
			if(!avbdQueryRigidConvexLocal(
					convex, localPoint, query) ||
				query.signedDistance >= margin)
				continue;
			const PxVec3 normal =
				convex.rotation.rotate(query.normalLocal).
					getNormalized();

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF,
				PX_MAX_U32, convex.primitiveKey,
				query.featureKey);
			geometry.particleIdx = particleIndex;
			geometry.normal = normal;
			geometry.projNormal = normal;
			geometry.depth = query.signedDistance < 0.0f
				? -query.signedDistance
				: PxMax(
					0.0f, margin - query.signedDistance);
			geometry.margin = margin;
			avbdConfigureRigidConvexTarget(
				geometry, convex, convexIndex,
				query.surfaceLocal);
			geometry.friction = sourceBody
				? avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction,
					convex.friction,
					convex.frictionCombineMode)
				: PxMax(convex.friction, 0.0f);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1e5f, 1e6f,
				particles, contacts);
		}
	}
}

inline void avbdDetectSoftRigidConvexSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	avbdDetectSoftRigidConvexSDFRange(
		particles, numParticles, 0, numParticles,
		convexes, numConvexes, contacts, margin,
		softBodies, numSoftBodies);
}

PX_FORCE_INLINE bool avbdGetRigidConvexSweepPose(
	const AvbdRigidConvex& convex,
	PxVec3& centerStart, PxVec3& centerEnd,
	PxQuat& rotationStart, PxQuat& rotationEnd,
	bool& rotationsEquivalent)
{
	if(!avbdIsRigidConvexValid(convex))
		return false;
	const bool kinematicTarget =
		convex.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID;
	const bool dynamicTarget =
		convex.targetKind ==
			AvbdSoftContactTargetKind::eRIGID_BODY;
	if(convex.targetKind !=
			AvbdSoftContactTargetKind::eWORLD_STATIC &&
		!kinematicTarget && !dynamicTarget)
		return false;
	if(kinematicTarget &&
		(!convex.previousCenter.isFinite() ||
		 !convex.previousRotation.isFinite()))
		return false;
	if(dynamicTarget &&
		(!convex.predictedPoseValid ||
		 !convex.predictedCenter.isFinite() ||
		 !convex.predictedRotation.isFinite()))
		return false;

	centerStart =
		kinematicTarget ? convex.previousCenter : convex.center;
	centerEnd =
		dynamicTarget ? convex.predictedCenter : convex.center;
	rotationStart =
		kinematicTarget ? convex.previousRotation : convex.rotation;
	rotationEnd =
		dynamicTarget ? convex.predictedRotation : convex.rotation;
	if(!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite())
		return false;
	rotationsEquivalent = avbdAreSweepRotationsEquivalent(
		rotationStart, rotationEnd);
	return true;
}

// The convex remains inside localRadius about its swept center at every pose.
// Reject only when the complete relative point segment misses that sphere's
// axis-aligned outer bound, leaving exact SDF/TOI ownership unchanged.
PX_FORCE_INLINE bool avbdSweptPointMayReachRigidConvexBound(
	const PxVec3& pointStart, const PxVec3& pointEnd,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	PxReal expandedRadius)
{
	if(!pointStart.isFinite() || !pointEnd.isFinite() ||
		!centerStart.isFinite() || !centerEnd.isFinite())
		return false;
	// An invalid bound cannot authorize rejection; the exact query keeps its
	// existing input validation and remains the conservative fallback.
	if(expandedRadius <= 0.0f || !PxIsFinite(expandedRadius))
		return true;
	const PxVec3 relativeStart = pointStart - centerStart;
	const PxVec3 relativeEnd = pointEnd - centerEnd;
	const PxVec3 relativeMinimum =
		relativeStart.minimum(relativeEnd);
	const PxVec3 relativeMaximum =
		relativeStart.maximum(relativeEnd);
	return relativeMinimum.x <= expandedRadius &&
		relativeMaximum.x >= -expandedRadius &&
		relativeMinimum.y <= expandedRadius &&
		relativeMaximum.y >= -expandedRadius &&
		relativeMinimum.z <= expandedRadius &&
		relativeMaximum.z >= -expandedRadius;
}

struct AvbdSweptConvexPointEntry
{
	PxReal entryTime;
	PxVec3 normalLocal;
	PxVec3 surfaceLocal;
	PxU64 featureKey;

	AvbdSweptConvexPointEntry()
		: entryTime(PX_MAX_F32),
		  normalLocal(0.0f, 1.0f, 0.0f),
		  surfaceLocal(0.0f), featureKey(0)
	{
	}
};

// Continuous point entry into a fixed-orientation convex expanded by margin.
// The exact closed-hull point distance is 1-Lipschitz, so gap/speed is a
// conservative advancement step and cannot cross first contact.
PX_FORCE_INLINE bool avbdSegmentEnterExpandedConvex(
	const AvbdRigidConvex& convex,
	const PxVec3& segmentStartLocal,
	const PxVec3& segmentEndLocal,
	PxReal margin, AvbdSweptConvexPointEntry& result,
	const AvbdRigidConvexPointQuery* initialQuery = NULL)
{
	if(!segmentStartLocal.isFinite() ||
		!segmentEndLocal.isFinite() ||
		margin <= 0.0f || !PxIsFinite(margin))
		return false;
	const PxVec3 direction =
		segmentEndLocal - segmentStartLocal;
	const PxReal speedSq = direction.magnitudeSquared();
	if(speedSq <= 1.0e-12f || !PxIsFinite(speedSq))
		return false;
	const PxReal speed = PxSqrt(speedSq);
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, margin * 1.0e-5f);
	AvbdRigidConvexPointQuery currentQuery;
	if(initialQuery)
		currentQuery = *initialQuery;
	else if(!avbdQueryRigidConvexLocal(
			convex, segmentStartLocal, currentQuery))
		return false;
	if(currentQuery.signedDistance < margin)
		return false;

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 48; ++iteration)
	{
		AvbdRigidConvexPointQuery query;
		if(iteration == 0)
			query = currentQuery;
		else if(!avbdQueryRigidConvexLocal(
				convex,
				segmentStartLocal + direction * time,
				query))
			return false;
		const PxReal gap = query.signedDistance - margin;
		if(gap <= distanceTolerance)
		{
			result.entryTime = time;
			result.normalLocal = query.normalLocal;
			result.surfaceLocal = query.surfaceLocal;
			result.featureKey = query.featureKey;
			return true;
		}
		const PxReal nextTime = time + gap / speed;
		if(!PxIsFinite(nextTime) || nextTime > 1.0f ||
			nextTime <= time)
			return false;
		time = nextTime;
	}
	return false;
}

// Continuous point entry against a translating/rotating convex. The exact
// local closed-hull point distance is sampled at the shortest-path slerped
// pose. Relative point/center translation plus localRadius*angularDistance
// bounds the Hausdorff speed of the hull, so gap/speed cannot step across
// first contact.
PX_FORCE_INLINE bool avbdSegmentEnterExpandedRotatingConvex(
	const AvbdRigidConvex& convex,
	const PxVec3& pointStart, const PxVec3& pointEnd,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	PxReal margin, AvbdSweptConvexPointEntry& result,
	const AvbdRigidConvexPointQuery* initialQuery = NULL)
{
	if(!avbdIsRigidConvexValid(convex) ||
		!pointStart.isFinite() || !pointEnd.isFinite() ||
		!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite() ||
		margin <= 0.0f || !PxIsFinite(margin))
		return false;

	PxReal angularDistance = 0.0f;
	if(!avbdGetSweepAngularDistance(
			rotationStart, rotationEnd, angularDistance) ||
		angularDistance <= 0.0f)
		return false;
	const PxQuat normalizedStart = rotationStart.getNormalized();
	const PxQuat normalizedEnd = rotationEnd.getNormalized();
	const PxVec3 relativeTranslation =
		(pointEnd - pointStart) - (centerEnd - centerStart);
	const PxReal speed =
		relativeTranslation.magnitude() +
		convex.localRadius * angularDistance;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, margin * 1.0e-5f);

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 64; ++iteration)
	{
		const PxVec3 point =
			pointStart + (pointEnd - pointStart) * time;
		const PxVec3 center =
			centerStart + (centerEnd - centerStart) * time;
		const PxQuat rotation =
			PxSlerp(time, normalizedStart, normalizedEnd).
				getNormalized();
		if(!point.isFinite() || !center.isFinite() ||
			!rotation.isFinite())
			return false;
		const PxVec3 localPoint =
			rotation.getConjugate().rotate(point - center);
		AvbdRigidConvexPointQuery query;
		if(iteration == 0 && initialQuery)
			query = *initialQuery;
		else if(!avbdQueryRigidConvexLocal(
				convex, localPoint, query))
			return false;
		if(iteration == 0 && query.signedDistance < margin)
			return false;
		const PxReal gap = query.signedDistance - margin;
		if(gap <= distanceTolerance)
		{
			result.entryTime = time;
			result.normalLocal = query.normalLocal;
			result.surfaceLocal = query.surfaceLocal;
			result.featureKey = query.featureKey;
			return true;
		}
		const PxReal nextTime = time + gap / speed;
		if(!PxIsFinite(nextTime) || nextTime > 1.0f ||
			nextTime <= time)
			return false;
		time = nextTime;
	}
	return false;
}

// P5.15a candidate leaf: swept convex SDF is particle-major. Convex topology
// is immutable here and every conservative-advancement/query object is local
// to one particle/convex evaluation. A partitioned caller must merge the full
// current-SDF family before stable-merging any swept-family range.
inline void avbdDetectSoftRigidConvexSweptSDFRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	PX_ASSERT(particleBegin <= particleEnd && particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for(PxU32 particleIndex = particleBegin;
		particleIndex < particleEnd; ++particleIndex)
	{
		const AvbdSoftParticle& particle =
			particles[particleIndex];
		if(particle.invMass <= 0.0f ||
			!particle.position.isFinite() ||
			!particle.predictedPosition.isFinite())
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies, particleIndex);
		if(!sourceBody ||
			!sourceBody->compiled.speculativeCCDEnabled ||
			!avbdIsSoftBodySurfaceVertex(
				*sourceBody, particleIndex))
			continue;

		for(PxU32 convexIndex = 0;
			convexIndex < numConvexes; ++convexIndex)
		{
			const AvbdRigidConvex& convex =
				convexes[convexIndex];
			PxVec3 centerStart(0.0f);
			PxVec3 centerEnd(0.0f);
			PxQuat rotationStart(PxIdentity);
			PxQuat rotationEnd(PxIdentity);
			bool rotationsEquivalent = false;
			if(!avbdGetRigidConvexSweepPose(
					convex, centerStart, centerEnd,
					rotationStart, rotationEnd,
					rotationsEquivalent))
				continue;
			if(!avbdSweptPointMayReachRigidConvexBound(
					particle.position,
					particle.predictedPosition,
					centerStart, centerEnd,
					convex.localRadius + margin))
				continue;
			AvbdSweptConvexPointEntry entry;
			PxQuat entryRotation(PxIdentity);
			if(rotationsEquivalent)
			{
				const PxQuat inverseRotation =
					rotationEnd.getConjugate();
				const PxVec3 relativeStart =
					inverseRotation.rotate(
						particle.position - centerStart);
				const PxVec3 relativeEnd =
					inverseRotation.rotate(
						particle.predictedPosition - centerEnd);
				if(!avbdSegmentEnterExpandedConvex(
						convex, relativeStart, relativeEnd,
						margin, entry))
					continue;
				entryRotation = rotationEnd.getNormalized();
			}
			else
			{
				if(!avbdSegmentEnterExpandedRotatingConvex(
						convex,
						particle.position,
						particle.predictedPosition,
						centerStart, centerEnd,
						rotationStart, rotationEnd,
						margin, entry))
					continue;
				entryRotation = PxSlerp(
					entry.entryTime,
					rotationStart.getNormalized(),
					rotationEnd.getNormalized()).getNormalized();
			}

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF,
				PX_MAX_U32, convex.primitiveKey,
				entry.featureKey);
			geometry.particleIdx = particleIndex;
			geometry.normal =
				entryRotation.rotate(entry.normalLocal).
					getNormalized();
			geometry.projNormal = geometry.normal;
			geometry.depth = 0.0f;
			geometry.margin = margin;
			avbdConfigureRigidConvexTarget(
				geometry, convex, convexIndex,
				entry.surfaceLocal);
			geometry.friction =
				avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction,
					convex.friction,
					convex.frictionCombineMode);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1.0e6f, 1.0e6f,
				particles, contacts);
		}
	}
}

inline void avbdDetectSoftRigidConvexSweptSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	avbdDetectSoftRigidConvexSweptSDFRange(
		particles, numParticles, 0, numParticles,
		convexes, numConvexes, contacts, margin,
		softBodies, numSoftBodies);
}

struct AvbdSweptConvexEdgeEntry
{
	PxReal entryTime;
	PxVec3 normal;
	PxReal softWeight1;
	PxReal rigidWeight1;

	AvbdSweptConvexEdgeEntry()
		: entryTime(PX_MAX_F32), normal(0.0f),
		  softWeight1(0.0f), rigidWeight1(0.0f)
	{
	}
};

// Continuous translated rigid-edge/soft-edge entry. Endpoint ownership is
// excluded on both segments so soft vertices remain forward-SDF owned and
// convex vertices remain reverse vertex/face owned.
PX_FORCE_INLINE bool
avbdTranslatedSegmentEnterExpandedSegmentInteriors(
	const PxVec3& rigid0, const PxVec3& rigid1,
	const PxVec3& rigidTranslation,
	const PxVec3& soft0, const PxVec3& soft1,
	PxReal margin, AvbdSweptConvexEdgeEntry& result)
{
	if(!rigid0.isFinite() || !rigid1.isFinite() ||
		!rigidTranslation.isFinite() || !soft0.isFinite() ||
		!soft1.isFinite() || margin <= 0.0f ||
		!PxIsFinite(margin))
		return false;
	const PxReal speedSq =
		rigidTranslation.magnitudeSquared();
	if(speedSq <= 1.0e-12f || !PxIsFinite(speedSq))
		return false;
	const PxReal speed = PxSqrt(speedSq);
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, margin * 1.0e-5f);
	const PxReal featureEpsilon = 1.0e-4f;
	PxReal currentSoftWeight1 = 0.0f;
	PxReal currentRigidWeight1 = 0.0f;
	PxVec3 currentSoftClosest(0.0f);
	PxVec3 currentRigidClosest(0.0f);
	avbdClosestPointsOnSegments(
		soft0, soft1, rigid0, rigid1,
		currentSoftWeight1, currentRigidWeight1,
		currentSoftClosest, currentRigidClosest);
	const PxReal currentDistance =
		(currentSoftClosest - currentRigidClosest).magnitude();
	if(!PxIsFinite(currentDistance) ||
		currentDistance < margin)
		return false;

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 48; ++iteration)
	{
		const PxVec3 offset = rigidTranslation * time;
		PxReal softWeight1 = 0.0f;
		PxReal rigidWeight1 = 0.0f;
		PxVec3 softClosest(0.0f);
		PxVec3 rigidClosest(0.0f);
		avbdClosestPointsOnSegments(
			soft0, soft1,
			rigid0 + offset, rigid1 + offset,
			softWeight1, rigidWeight1,
			softClosest, rigidClosest);
		const PxVec3 delta = softClosest - rigidClosest;
		const PxReal distance = delta.magnitude();
		if(!PxIsFinite(distance))
			return false;
		const PxReal gap = distance - margin;
		if(gap <= distanceTolerance)
		{
			if(softWeight1 <= featureEpsilon ||
				softWeight1 >= 1.0f - featureEpsilon ||
				rigidWeight1 <= featureEpsilon ||
				rigidWeight1 >= 1.0f - featureEpsilon ||
				!delta.isFinite() ||
				delta.magnitudeSquared() <= 1.0e-12f)
				return false;
			result.entryTime = time;
			result.normal = delta.getNormalized();
			result.softWeight1 = softWeight1;
			result.rigidWeight1 = rigidWeight1;
			return true;
		}
		const PxReal nextTime = time + gap / speed;
		if(!PxIsFinite(nextTime) || nextTime > 1.0f ||
			nextTime <= time)
			return false;
		time = nextTime;
	}
	return false;
}

// Continuous rotating/translating rigid-edge versus a translation-only soft
// edge. The common soft displacement is removed by the caller. Endpoint
// ownership stays excluded on both segments.
PX_FORCE_INLINE bool
avbdRotatingSegmentEnterExpandedSegmentInteriors(
	const PxVec3& rigidLocal0, const PxVec3& rigidLocal1,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	const PxVec3& soft0, const PxVec3& soft1,
	PxReal margin, AvbdSweptConvexEdgeEntry& result)
{
	if(!rigidLocal0.isFinite() || !rigidLocal1.isFinite() ||
		!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite() ||
		!soft0.isFinite() || !soft1.isFinite() ||
		margin <= 0.0f || !PxIsFinite(margin))
		return false;
	PxReal angularDistance = 0.0f;
	if(!avbdGetSweepAngularDistance(
			rotationStart, rotationEnd, angularDistance))
		return false;
	const PxQuat normalizedStart = rotationStart.getNormalized();
	const PxQuat normalizedEnd = rotationEnd.getNormalized();
	const PxReal edgeRadius = PxMax(
		rigidLocal0.magnitude(), rigidLocal1.magnitude());
	const PxReal speed =
		(centerEnd - centerStart).magnitude() +
		edgeRadius * angularDistance;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, margin * 1.0e-5f);
	const PxReal featureEpsilon = 1.0e-4f;

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 64; ++iteration)
	{
		const PxVec3 center =
			centerStart + (centerEnd - centerStart) * time;
		const PxQuat rotation =
			PxSlerp(time, normalizedStart, normalizedEnd).
				getNormalized();
		if(!center.isFinite() || !rotation.isFinite())
			return false;
		const PxVec3 rigid0 =
			center + rotation.rotate(rigidLocal0);
		const PxVec3 rigid1 =
			center + rotation.rotate(rigidLocal1);
		PxReal softWeight1 = 0.0f;
		PxReal rigidWeight1 = 0.0f;
		PxVec3 softClosest(0.0f);
		PxVec3 rigidClosest(0.0f);
		avbdClosestPointsOnSegments(
			soft0, soft1, rigid0, rigid1,
			softWeight1, rigidWeight1,
			softClosest, rigidClosest);
		const PxVec3 delta = softClosest - rigidClosest;
		const PxReal distance = delta.magnitude();
		if(!PxIsFinite(distance))
			return false;
		if(iteration == 0 && distance < margin)
			return false;
		const PxReal gap = distance - margin;
		if(gap <= distanceTolerance)
		{
			if(softWeight1 <= featureEpsilon ||
				softWeight1 >= 1.0f - featureEpsilon ||
				rigidWeight1 <= featureEpsilon ||
				rigidWeight1 >= 1.0f - featureEpsilon ||
				!delta.isFinite() ||
				delta.magnitudeSquared() <= 1.0e-12f)
				return false;
			result.entryTime = time;
			result.normal = delta.getNormalized();
			result.softWeight1 = softWeight1;
			result.rigidWeight1 = rigidWeight1;
			return true;
		}
		const PxReal nextTime = time + gap / speed;
		if(!PxIsFinite(nextTime) || nextTime > 1.0f ||
			nextTime <= time)
			return false;
		time = nextTime;
	}
	return false;
}

// Continuous rotating/translating rigid edge versus a linearly deforming
// soft edge. The maximum endpoint speeds bound each segment's Hausdorff
// motion. Endpoint-owned pairs remain excluded on both features.
PX_FORCE_INLINE bool
avbdRotatingSegmentEnterExpandedDeformingSegmentInteriors(
	const PxVec3& rigidLocal0, const PxVec3& rigidLocal1,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	const PxVec3& soft0Start, const PxVec3& soft1Start,
	const PxVec3& soft0End, const PxVec3& soft1End,
	PxReal margin, AvbdSweptConvexEdgeEntry& result)
{
	if(!rigidLocal0.isFinite() || !rigidLocal1.isFinite() ||
		!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite() ||
		!soft0Start.isFinite() || !soft1Start.isFinite() ||
		!soft0End.isFinite() || !soft1End.isFinite() ||
		margin <= 0.0f || !PxIsFinite(margin))
		return false;
	PxReal angularDistance = 0.0f;
	if(!avbdGetSweepAngularDistance(
			rotationStart, rotationEnd, angularDistance))
		return false;
	const PxQuat normalizedStart = rotationStart.getNormalized();
	const PxQuat normalizedEnd = rotationEnd.getNormalized();
	const PxVec3 softDisplacement0 = soft0End - soft0Start;
	const PxVec3 softDisplacement1 = soft1End - soft1Start;
	const PxReal softSpeed = PxMax(
		softDisplacement0.magnitude(),
		softDisplacement1.magnitude());
	const PxReal rigidRadius = PxMax(
		rigidLocal0.magnitude(), rigidLocal1.magnitude());
	const PxReal speed =
		(centerEnd - centerStart).magnitude() +
		rigidRadius * angularDistance + softSpeed;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, margin * 1.0e-5f);
	const PxReal featureEpsilon = 1.0e-4f;

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 64; ++iteration)
	{
		const PxVec3 center =
			centerStart + (centerEnd - centerStart) * time;
		const PxQuat rotation =
			PxSlerp(time, normalizedStart, normalizedEnd).
				getNormalized();
		const PxVec3 rigid0 =
			center + rotation.rotate(rigidLocal0);
		const PxVec3 rigid1 =
			center + rotation.rotate(rigidLocal1);
		const PxVec3 soft0 =
			soft0Start + softDisplacement0 * time;
		const PxVec3 soft1 =
			soft1Start + softDisplacement1 * time;
		if(!center.isFinite() || !rotation.isFinite() ||
			!rigid0.isFinite() || !rigid1.isFinite() ||
			!soft0.isFinite() || !soft1.isFinite() ||
			(rigid1 - rigid0).magnitudeSquared() <= 1.0e-16f ||
			(soft1 - soft0).magnitudeSquared() <= 1.0e-16f)
			return false;
		PxReal softWeight1 = 0.0f;
		PxReal rigidWeight1 = 0.0f;
		PxVec3 softClosest(0.0f);
		PxVec3 rigidClosest(0.0f);
		avbdClosestPointsOnSegments(
			soft0, soft1, rigid0, rigid1,
			softWeight1, rigidWeight1,
			softClosest, rigidClosest);
		const PxVec3 delta = softClosest - rigidClosest;
		const PxReal distance = delta.magnitude();
		if(!PxIsFinite(distance))
			return false;
		if(iteration == 0 && distance < margin)
			return false;
		const PxReal gap = distance - margin;
		if(gap <= distanceTolerance)
		{
			if(softWeight1 <= featureEpsilon ||
				softWeight1 >= 1.0f - featureEpsilon ||
				rigidWeight1 <= featureEpsilon ||
				rigidWeight1 >= 1.0f - featureEpsilon ||
				!delta.isFinite() ||
				delta.magnitudeSquared() <= 1.0e-12f)
				return false;
			result.entryTime = time;
			result.normal = delta.getNormalized();
			result.softWeight1 = softWeight1;
			result.rigidWeight1 = rigidWeight1;
			return true;
		}
		const PxReal nextTime = time + gap / speed;
		if(!PxIsFinite(nextTime) || nextTime > 1.0f ||
			nextTime <= time)
			return false;
		time = nextTime;
	}
	return false;
}

// Continuous linearly deforming edge versus linearly deforming edge.  The sum
// of the two maximum endpoint displacements is a conservative relative-speed
// bound for the segment distance.  Only the two edge interiors are owned here;
// vertex-edge and vertex-vertex contacts retain their existing owners.
PX_FORCE_INLINE bool
avbdDeformingSegmentsEnterExpandedInteriors(
	const PxVec3& query0Start, const PxVec3& query1Start,
	const PxVec3& query0End, const PxVec3& query1End,
	const PxVec3& target0Start, const PxVec3& target1Start,
	const PxVec3& target0End, const PxVec3& target1End,
	PxReal margin, AvbdSweptConvexEdgeEntry& result)
{
	if(!query0Start.isFinite() || !query1Start.isFinite() ||
		!query0End.isFinite() || !query1End.isFinite() ||
		!target0Start.isFinite() || !target1Start.isFinite() ||
		!target0End.isFinite() || !target1End.isFinite() ||
		margin <= 0.0f || !PxIsFinite(margin))
		return false;

	const PxVec3 queryDisplacement0 = query0End - query0Start;
	const PxVec3 queryDisplacement1 = query1End - query1Start;
	const PxVec3 targetDisplacement0 = target0End - target0Start;
	const PxVec3 targetDisplacement1 = target1End - target1Start;
	const PxReal querySpeed = PxMax(
		queryDisplacement0.magnitude(),
		queryDisplacement1.magnitude());
	const PxReal targetSpeed = PxMax(
		targetDisplacement0.magnitude(),
		targetDisplacement1.magnitude());
	const PxReal speed = querySpeed + targetSpeed;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;

	const PxReal distanceTolerance =
		PxMax(1.0e-5f, margin * 1.0e-5f);
	const PxReal featureEpsilon = 1.0e-4f;
	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 64; ++iteration)
	{
		const PxVec3 query0 =
			query0Start + queryDisplacement0 * time;
		const PxVec3 query1 =
			query1Start + queryDisplacement1 * time;
		const PxVec3 target0 =
			target0Start + targetDisplacement0 * time;
		const PxVec3 target1 =
			target1Start + targetDisplacement1 * time;
		if(!query0.isFinite() || !query1.isFinite() ||
			!target0.isFinite() || !target1.isFinite() ||
			(query1 - query0).magnitudeSquared() <= 1.0e-16f ||
			(target1 - target0).magnitudeSquared() <= 1.0e-16f)
			return false;

		PxReal queryWeight1 = 0.0f;
		PxReal targetWeight1 = 0.0f;
		PxVec3 queryClosest(0.0f);
		PxVec3 targetClosest(0.0f);
		avbdClosestPointsOnSegments(
			query0, query1, target0, target1,
			queryWeight1, targetWeight1,
			queryClosest, targetClosest);
		const PxVec3 delta = queryClosest - targetClosest;
		const PxReal distance = delta.magnitude();
		if(!PxIsFinite(distance))
			return false;
		// A contact already active at the beginning of the step is owned by
		// the discrete path.  This also prevents a second swept owner.
		if(iteration == 0 && distance < margin)
			return false;
		const PxReal gap = distance - margin;
		if(gap <= distanceTolerance)
		{
			if(queryWeight1 <= featureEpsilon ||
				queryWeight1 >= 1.0f - featureEpsilon ||
				targetWeight1 <= featureEpsilon ||
				targetWeight1 >= 1.0f - featureEpsilon ||
				!delta.isFinite() ||
				delta.magnitudeSquared() <= 1.0e-12f)
				return false;
			result.entryTime = time;
			result.normal = delta.getNormalized();
			result.softWeight1 = queryWeight1;
			result.rigidWeight1 = targetWeight1;
			return true;
		}
		const PxReal nextTime = time + gap / speed;
		if(!PxIsFinite(nextTime) || nextTime > 1.0f ||
			nextTime <= time)
			return false;
		time = nextTime;
	}
	return false;
}

// Continuous rotating convex-vertex entry into the face-owned portion of a
// translation-only soft triangle. Soft edges/vertices and convex-edge cases
// retain their separate owners.
PX_FORCE_INLINE bool avbdRotatingPointEnterExpandedTriangleFace(
	const PxVec3& rigidLocalPoint,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	const PxVec3& a, const PxVec3& b, const PxVec3& c,
	PxReal margin, AvbdSweptTriangleEntry& result)
{
	if(!rigidLocalPoint.isFinite() ||
		!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite() ||
		!a.isFinite() || !b.isFinite() || !c.isFinite() ||
		margin <= 0.0f || !PxIsFinite(margin))
		return false;
	PxReal angularDistance = 0.0f;
	if(!avbdGetSweepAngularDistance(
			rotationStart, rotationEnd, angularDistance))
		return false;
	const PxQuat normalizedStart = rotationStart.getNormalized();
	const PxQuat normalizedEnd = rotationEnd.getNormalized();
	const PxReal speed =
		(centerEnd - centerStart).magnitude() +
		rigidLocalPoint.magnitude() * angularDistance;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, margin * 1.0e-5f);

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 64; ++iteration)
	{
		const PxVec3 center =
			centerStart + (centerEnd - centerStart) * time;
		const PxQuat rotation =
			PxSlerp(time, normalizedStart, normalizedEnd).
				getNormalized();
		if(!center.isFinite() || !rotation.isFinite())
			return false;
		const PxVec3 rigidPoint =
			center + rotation.rotate(rigidLocalPoint);
		const AvbdClosestPointResult closest =
			avbdClosestPointOnTriangleOGC(
				rigidPoint, a, b, c);
		if(!PxIsFinite(closest.distance))
			return false;
		if(iteration == 0 && closest.distance < margin)
			return false;
		const PxReal gap = closest.distance - margin;
		if(gap <= distanceTolerance)
		{
			if(closest.feature != AVBD_FEATURE_FACE)
				return false;
			const PxVec3 normal = closest.point - rigidPoint;
			const PxReal normalMagnitudeSq =
				normal.magnitudeSquared();
			if(!normal.isFinite() ||
				normalMagnitudeSq <= 1.0e-12f)
				return false;
			result.entryTime = time;
			result.normal =
				normal * PxRecipSqrt(normalMagnitudeSq);
			result.barycentric = closest.barycentric;
			result.feature = AVBD_FEATURE_FACE;
			result.featureIndex = 0;
			return true;
		}
		const PxReal nextTime = time + gap / speed;
		if(!PxIsFinite(nextTime) || nextTime > 1.0f ||
			nextTime <= time)
			return false;
		time = nextTime;
	}
	return false;
}

// Continuous rotating/translating rigid vertex entry into the face-owned
// portion of a linearly deforming soft triangle. The maximum residual
// soft-vertex speed augments the rigid point speed bound. Soft edges and
// vertices retain their existing unique owners.
PX_FORCE_INLINE bool
avbdRotatingPointEnterExpandedDeformingTriangleFace(
	const PxVec3& rigidLocalPoint,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	const PxVec3& aStart, const PxVec3& bStart,
	const PxVec3& cStart, const PxVec3& aEnd,
	const PxVec3& bEnd, const PxVec3& cEnd,
	PxReal margin, AvbdSweptTriangleEntry& result)
{
	if(!rigidLocalPoint.isFinite() ||
		!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite() ||
		!aStart.isFinite() || !bStart.isFinite() ||
		!cStart.isFinite() || !aEnd.isFinite() ||
		!bEnd.isFinite() || !cEnd.isFinite() ||
		margin <= 0.0f || !PxIsFinite(margin))
		return false;
	PxReal angularDistance = 0.0f;
	if(!avbdGetSweepAngularDistance(
			rotationStart, rotationEnd, angularDistance))
		return false;
	const PxQuat normalizedStart = rotationStart.getNormalized();
	const PxQuat normalizedEnd = rotationEnd.getNormalized();
	const PxVec3 displacementA = aEnd - aStart;
	const PxVec3 displacementB = bEnd - bStart;
	const PxVec3 displacementC = cEnd - cStart;
	const PxReal triangleSpeed = PxMax(
		displacementA.magnitude(),
		PxMax(displacementB.magnitude(),
			displacementC.magnitude()));
	const PxReal speed =
		(centerEnd - centerStart).magnitude() +
		rigidLocalPoint.magnitude() * angularDistance +
		triangleSpeed;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, margin * 1.0e-5f);

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 64; ++iteration)
	{
		const PxVec3 center =
			centerStart + (centerEnd - centerStart) * time;
		const PxQuat rotation =
			PxSlerp(time, normalizedStart, normalizedEnd).
				getNormalized();
		const PxVec3 rigidPoint =
			center + rotation.rotate(rigidLocalPoint);
		const PxVec3 a = aStart + displacementA * time;
		const PxVec3 b = bStart + displacementB * time;
		const PxVec3 c = cStart + displacementC * time;
		const PxVec3 triangleNormal = (b - a).cross(c - a);
		if(!center.isFinite() || !rotation.isFinite() ||
			!rigidPoint.isFinite() || !a.isFinite() ||
			!b.isFinite() || !c.isFinite() ||
			!triangleNormal.isFinite() ||
			triangleNormal.magnitudeSquared() <= 1.0e-16f)
			return false;
		const AvbdClosestPointResult closest =
			avbdClosestPointOnTriangleOGC(
				rigidPoint, a, b, c);
		if(!PxIsFinite(closest.distance))
			return false;
		if(iteration == 0 && closest.distance < margin)
			return false;
		const PxReal gap = closest.distance - margin;
		if(gap <= distanceTolerance)
		{
			if(closest.feature != AVBD_FEATURE_FACE)
				return false;
			const PxVec3 normal = closest.point - rigidPoint;
			const PxReal normalMagnitudeSq =
				normal.magnitudeSquared();
			if(!normal.isFinite() ||
				normalMagnitudeSq <= 1.0e-12f)
				return false;
			result.entryTime = time;
			result.normal =
				normal * PxRecipSqrt(normalMagnitudeSq);
			result.barycentric = closest.barycentric;
			result.feature = AVBD_FEATURE_FACE;
			result.featureIndex = 0;
			return true;
		}
		const PxReal nextTime = time + gap / speed;
		if(!PxIsFinite(nextTime) || nextTime > 1.0f ||
			nextTime <= time)
			return false;
		time = nextTime;
	}
	return false;
}

PX_FORCE_INLINE bool avbdRigidConvexForwardVertexOwnsSweptFeature(
	const AvbdSoftParticle& particle,
	const AvbdRigidConvex& convex,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	const PxQuat& inverseRotationEnd,
	bool rotationsEquivalent, PxReal margin)
{
	if(particle.invMass <= 0.0f)
		return false;
	AvbdRigidConvexPointQuery currentQuery;
	AvbdSweptConvexPointEntry vertexEntry;
	const PxVec3 pointStart = particle.initialPosition;
	const PxVec3 pointEnd = particle.predictedPosition;
	if(!avbdSweptPointMayReachRigidConvexBound(
			pointStart, pointEnd, centerStart, centerEnd,
			convex.localRadius + margin))
		return false;
	const PxVec3 relativeStart =
		rotationStart.getConjugate().rotate(
			pointStart - centerStart);
	if(!avbdQueryRigidConvexLocal(
			convex, relativeStart, currentQuery))
		return false;
	if(currentQuery.signedDistance < margin)
		return true;
	if(rotationsEquivalent)
	{
		const PxVec3 relativeEnd =
			inverseRotationEnd.rotate(pointEnd - centerEnd);
		return avbdSegmentEnterExpandedConvex(
			convex, relativeStart, relativeEnd,
			margin, vertexEntry, &currentQuery);
	}
	return avbdSegmentEnterExpandedRotatingConvex(
		convex, pointStart, pointEnd,
		centerStart, centerEnd,
		rotationStart, rotationEnd,
		margin, vertexEntry, &currentQuery);
}

inline void avbdDetectSoftRigidConvexSweptOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	PxArray<PxU8>* persistentForwardOwnerScratch = NULL)
{
	const PxReal translationToleranceSq = 1.0e-10f;
	PxArray<PxU8> localForwardOwnerScratch;
	PxArray<PxU8>& forwardOwnerScratch =
		persistentForwardOwnerScratch
			? *persistentForwardOwnerScratch
			: localForwardOwnerScratch;
	if(forwardOwnerScratch.capacity() < numParticles)
		forwardOwnerScratch.reserve(numParticles);
	forwardOwnerScratch.resize(numParticles);
	for(PxU32 bodyIndex = 0;
		bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		if(!body.compiled.speculativeCCDEnabled)
			continue;
		for(PxU32 convexIndex = 0;
			convexIndex < numConvexes; ++convexIndex)
		{
			const AvbdRigidConvex& convex =
				convexes[convexIndex];
			PxArray<AvbdRigidConvexEdgeBounds>& edgeBoundsScratch =
				convex.edgeBoundsScratch;
			PxVec3 centerStart(0.0f);
			PxVec3 centerEnd(0.0f);
			PxQuat rotationStart(PxIdentity);
			PxQuat rotationEnd(PxIdentity);
			bool rotationsEquivalent = false;
			if(!avbdGetRigidConvexSweepPose(
					convex, centerStart, centerEnd,
					rotationStart, rotationEnd,
					rotationsEquivalent))
				continue;
			const PxQuat inverseRotation =
				rotationEnd.getConjugate();
			// Broadphase rejection is invariant under a common frame change. Cache
			// each rigid edge's world-space swept AABB once; the exact kernels
			// below retain their original relative-motion coordinates and TOI.
			const PxVec3 rigidTranslation =
				centerEnd - centerStart;
			const PxReal sweptRotationExtent =
				convex.localRadius + margin;
			const PxVec3 rotatingMinimum =
				centerStart.minimum(centerEnd) -
					PxVec3(sweptRotationExtent);
			const PxVec3 rotatingMaximum =
				centerStart.maximum(centerEnd) +
					PxVec3(sweptRotationExtent);
			if(edgeBoundsScratch.capacity() < convex.edges.size())
				edgeBoundsScratch.reserve(convex.edges.size());
			edgeBoundsScratch.resize(convex.edges.size());
			for(PxU32 rigidEdgeIndex = 0;
				rigidEdgeIndex < convex.edges.size(); ++rigidEdgeIndex)
			{
				const AvbdRigidConvexEdge& rigidEdge =
					convex.edges[rigidEdgeIndex];
				if(rigidEdge.p0 >= convex.vertices.size() ||
					rigidEdge.p1 >= convex.vertices.size())
					continue;
				AvbdRigidConvexEdgeBounds& edgeBounds =
					edgeBoundsScratch[rigidEdgeIndex];
				edgeBounds.point0 =
					centerStart + rotationStart.rotate(
						convex.vertices[rigidEdge.p0]);
				edgeBounds.point1 =
					centerStart + rotationStart.rotate(
						convex.vertices[rigidEdge.p1]);
				const PxVec3 edgeMinimum =
					edgeBounds.point0.minimum(edgeBounds.point1);
				const PxVec3 edgeMaximum =
					edgeBounds.point0.maximum(edgeBounds.point1);
				edgeBounds.minimum = rotationsEquivalent
					? edgeMinimum.minimum(
						edgeMinimum + rigidTranslation) -
							PxVec3(margin)
					: rotatingMinimum;
				edgeBounds.maximum = rotationsEquivalent
					? edgeMaximum.maximum(
						edgeMaximum + rigidTranslation) +
							PxVec3(margin)
					: rotatingMaximum;
			}
			const PxU32 particleStart = body.compiled.particleStart;
			const PxU32 particleCount = body.compiled.particleCount;
			if(particleStart <= numParticles)
			{
				const PxU32 boundedParticleCount = PxMin(
					particleCount, numParticles - particleStart);
				for(PxU32 localParticle = 0;
					localParticle < boundedParticleCount; ++localParticle)
					forwardOwnerScratch[
						particleStart + localParticle] = 0;
			}
			for(PxU32 surfaceVertexIndex = 0;
				surfaceVertexIndex < body.compiled.surfaceVertices.size();
				++surfaceVertexIndex)
			{
				const PxU32 vertexIndex =
					body.compiled.surfaceVertices[surfaceVertexIndex];
				if(vertexIndex >= numParticles)
					continue;
				forwardOwnerScratch[vertexIndex] = PxU8(
					avbdRigidConvexForwardVertexOwnsSweptFeature(
						particles[vertexIndex], convex,
						centerStart, centerEnd,
						rotationStart, rotationEnd,
						inverseRotation, rotationsEquivalent,
						margin));
			}

			// Convex edge versus a linearly deforming soft edge. The
			// translation-only kernel remains the zero-relative-motion
			// fast path.
			for(PxU32 softEdgeIndex = 0;
				softEdgeIndex <
					body.compiled.surfaceEdges.size();
				++softEdgeIndex)
			{
				const AvbdEdgeInfo& softEdge =
					body.compiled.surfaceEdges[softEdgeIndex];
				if(softEdge.p0 >= numParticles ||
					softEdge.p1 >= numParticles ||
					(particles[softEdge.p0].invMass <= 0.0f &&
					 particles[softEdge.p1].invMass <= 0.0f))
					continue;
				const PxVec3 soft0 =
					particles[softEdge.p0].initialPosition;
				const PxVec3 soft1 =
					particles[softEdge.p1].initialPosition;
				const PxVec3 predicted0 =
					particles[softEdge.p0].predictedPosition;
				const PxVec3 predicted1 =
					particles[softEdge.p1].predictedPosition;
				const PxVec3 displacement0 =
					predicted0 - soft0;
				const PxVec3 displacement1 =
					predicted1 - soft1;
				if(!soft0.isFinite() || !soft1.isFinite() ||
					!displacement0.isFinite() ||
					!displacement1.isFinite())
					continue;
				const PxVec3 relativeSoftDisplacement1 =
					displacement1 - displacement0;
				const bool softEdgeTranslationOnly =
					relativeSoftDisplacement1.
						magnitudeSquared() <=
							translationToleranceSq;

				const bool forwardVertexOwns =
					forwardOwnerScratch[softEdge.p0] != 0 ||
					forwardOwnerScratch[softEdge.p1] != 0;
				if(forwardVertexOwns)
					continue;

				const PxVec3 relativeTranslation =
					centerEnd - centerStart - displacement0;
				if(softEdgeTranslationOnly &&
					rotationsEquivalent &&
					relativeTranslation.magnitudeSquared() <=
						translationToleranceSq)
					continue;
				const PxVec3 relativeCenterEnd =
					centerEnd - displacement0;
				const PxVec3 soft0End = soft0;
				const PxVec3 soft1End =
					soft1 + relativeSoftDisplacement1;
				const PxVec3 sweptSoftMinimum =
					soft0.minimum(soft1).
						minimum(predicted0).
						minimum(predicted1);
				const PxVec3 sweptSoftMaximum =
					soft0.maximum(soft1).
						maximum(predicted0).
						maximum(predicted1);
				for(PxU32 rigidEdgeIndex = 0;
					rigidEdgeIndex < convex.edges.size();
					++rigidEdgeIndex)
				{
					const AvbdRigidConvexEdge& rigidEdge =
						convex.edges[rigidEdgeIndex];
					if(rigidEdge.p0 >= convex.vertices.size() ||
						rigidEdge.p1 >= convex.vertices.size())
						continue;
					const AvbdRigidConvexEdgeBounds& edgeBounds =
						edgeBoundsScratch[rigidEdgeIndex];
					const PxVec3& rigid0 = edgeBounds.point0;
					const PxVec3& rigid1 = edgeBounds.point1;
					const PxVec3& rigidMinimum = edgeBounds.minimum;
					const PxVec3& rigidMaximum = edgeBounds.maximum;
					if(rigidMinimum.x > sweptSoftMaximum.x ||
						rigidMaximum.x < sweptSoftMinimum.x ||
						rigidMinimum.y > sweptSoftMaximum.y ||
						rigidMaximum.y < sweptSoftMinimum.y ||
						rigidMinimum.z > sweptSoftMaximum.z ||
						rigidMaximum.z < sweptSoftMinimum.z)
						continue;
					AvbdSweptConvexEdgeEntry entry;
					const bool entered =
						softEdgeTranslationOnly &&
							rotationsEquivalent
							? avbdTranslatedSegmentEnterExpandedSegmentInteriors(
								rigid0, rigid1,
								relativeTranslation,
								soft0, soft1, margin, entry)
							: softEdgeTranslationOnly
							? avbdRotatingSegmentEnterExpandedSegmentInteriors(
								convex.vertices[rigidEdge.p0],
								convex.vertices[rigidEdge.p1],
								centerStart, relativeCenterEnd,
								rotationStart, rotationEnd,
								soft0, soft1, margin, entry)
							: avbdRotatingSegmentEnterExpandedDeformingSegmentInteriors(
								convex.vertices[rigidEdge.p0],
								convex.vertices[rigidEdge.p1],
								centerStart, relativeCenterEnd,
								rotationStart, rotationEnd,
								soft0, soft1,
								soft0End, soft1End,
								margin, entry);
					if(!entered)
						continue;
					PxVec3 normal = entry.normal;
					const PxQuat entryRotation =
						rotationsEquivalent
							? rotationEnd.getNormalized()
							: PxSlerp(
								entry.entryTime,
								rotationStart.getNormalized(),
								rotationEnd.getNormalized()).
									getNormalized();
					const PxVec3 outward =
						entryRotation.rotate(rigidEdge.outward);
					if(normal.dot(outward) <= 0.0f)
						continue;

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, convex.primitiveKey,
						avbdGetRigidSoftFeatureKey(
							0x43564545u,
							softEdge.p0, softEdge.p1,
							0u, rigidEdgeIndex));
					geometry.particleIdx =
						particles[softEdge.p0].invMass > 0.0f
							? softEdge.p0 : softEdge.p1;
					geometry.queryParticleIndices[0] =
						softEdge.p0;
					geometry.queryParticleIndices[1] =
						softEdge.p1;
					geometry.queryWeights[0] =
						1.0f - entry.softWeight1;
					geometry.queryWeights[1] =
						entry.softWeight1;
					geometry.normal = normal.getNormalized();
					geometry.projNormal = geometry.normal;
					geometry.depth = 0.0f;
					geometry.margin = margin;
					const PxVec3 surfaceLocal =
						convex.vertices[rigidEdge.p0] *
							(1.0f - entry.rigidWeight1) +
						convex.vertices[rigidEdge.p1] *
							entry.rigidWeight1;
					avbdConfigureRigidConvexTarget(
						geometry, convex, convexIndex,
						surfaceLocal);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							convex.friction,
							convex.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1.0e7f, 1.0e6f,
						particles, contacts);
				}
			}

			// Convex vertex versus a linearly deforming soft face. The
			// translation-only kernel remains the zero-relative-motion fast
			// path. Any forward soft-vertex owner suppresses the complete
			// triangle candidate.
			for(PxU32 triangleOffset = 0;
				triangleOffset + 2 <
					body.compiled.surfaceTriangles.size();
				triangleOffset += 3)
			{
				const PxU32 v0 =
					body.compiled.surfaceTriangles[
						triangleOffset];
				const PxU32 v1 =
					body.compiled.surfaceTriangles[
						triangleOffset + 1];
				const PxU32 v2 =
					body.compiled.surfaceTriangles[
						triangleOffset + 2];
				if(v0 >= numParticles || v1 >= numParticles ||
					v2 >= numParticles ||
					(particles[v0].invMass <= 0.0f &&
					 particles[v1].invMass <= 0.0f &&
					 particles[v2].invMass <= 0.0f))
					continue;
				const PxVec3 p0 = particles[v0].initialPosition;
				const PxVec3 p1 = particles[v1].initialPosition;
				const PxVec3 p2 = particles[v2].initialPosition;
				const PxVec3 displacement0 =
					particles[v0].predictedPosition - p0;
				const PxVec3 displacement1 =
					particles[v1].predictedPosition - p1;
				const PxVec3 displacement2 =
					particles[v2].predictedPosition - p2;
				if(!p0.isFinite() || !p1.isFinite() ||
					!p2.isFinite() ||
					!displacement0.isFinite() ||
					!displacement1.isFinite() ||
					!displacement2.isFinite())
					continue;
				const PxVec3 relativeDisplacement1 =
					(displacement1 - displacement0);
				const PxVec3 relativeDisplacement2 =
					(displacement2 - displacement0);
				const bool softTriangleTranslationOnly =
					relativeDisplacement1.magnitudeSquared() <=
						translationToleranceSq &&
					relativeDisplacement2.magnitudeSquared() <=
						translationToleranceSq;

				const bool forwardVertexOwns =
					forwardOwnerScratch[v0] != 0 ||
					forwardOwnerScratch[v1] != 0 ||
					forwardOwnerScratch[v2] != 0;
				if(forwardVertexOwns)
					continue;

				const PxVec3 relativeTranslation =
					centerEnd - centerStart - displacement0;
				if(softTriangleTranslationOnly &&
					rotationsEquivalent &&
					relativeTranslation.magnitudeSquared() <=
						translationToleranceSq)
					continue;
				const PxVec3 relativeCenterEnd =
					centerEnd - displacement0;
				const PxVec3 triangleMinimum =
					p0.minimum(p1).minimum(p2).
						minimum(p1 + relativeDisplacement1).
						minimum(p2 + relativeDisplacement2) -
						PxVec3(margin);
				const PxVec3 triangleMaximum =
					p0.maximum(p1).maximum(p2).
						maximum(p1 + relativeDisplacement1).
						maximum(p2 + relativeDisplacement2) +
						PxVec3(margin);
				for(PxU32 rigidVertexIndex = 0;
					rigidVertexIndex < convex.vertices.size();
					++rigidVertexIndex)
				{
					const PxVec3 rigidVertexStart =
						centerStart + rotationStart.rotate(
							convex.vertices[
								rigidVertexIndex]);
					const PxVec3 rigidVertexEnd =
						rotationsEquivalent
							? rigidVertexStart + relativeTranslation
							: relativeCenterEnd +
								rotationEnd.rotate(
									convex.vertices[
										rigidVertexIndex]);
					PxVec3 sweptMinimum(0.0f);
					PxVec3 sweptMaximum(0.0f);
					if(rotationsEquivalent)
					{
						sweptMinimum =
							rigidVertexStart.minimum(
								rigidVertexEnd);
						sweptMaximum =
							rigidVertexStart.maximum(
								rigidVertexEnd);
					}
					else
					{
						const PxReal rotationExtent =
							convex.localRadius;
						sweptMinimum =
							centerStart.minimum(relativeCenterEnd) -
								PxVec3(rotationExtent);
						sweptMaximum =
							centerStart.maximum(relativeCenterEnd) +
								PxVec3(rotationExtent);
					}
					if(sweptMinimum.x > triangleMaximum.x ||
						sweptMaximum.x < triangleMinimum.x ||
						sweptMinimum.y > triangleMaximum.y ||
						sweptMaximum.y < triangleMinimum.y ||
						sweptMinimum.z > triangleMaximum.z ||
						sweptMaximum.z < triangleMinimum.z)
						continue;
					AvbdSweptTriangleEntry entry;
					const bool entered =
						softTriangleTranslationOnly &&
							rotationsEquivalent
							? avbdSegmentEnterExpandedTriangleNonVertex(
								rigidVertexStart, rigidVertexEnd,
								p0, p1, p2, margin, entry)
							: softTriangleTranslationOnly
							? avbdRotatingPointEnterExpandedTriangleFace(
								convex.vertices[rigidVertexIndex],
								centerStart, relativeCenterEnd,
								rotationStart, rotationEnd,
								p0, p1, p2, margin, entry)
							: avbdRotatingPointEnterExpandedDeformingTriangleFace(
								convex.vertices[rigidVertexIndex],
								centerStart, relativeCenterEnd,
								rotationStart, rotationEnd,
								p0, p1, p2, p0,
								p1 + relativeDisplacement1,
								p2 + relativeDisplacement2,
								margin, entry);
					if(!entered ||
						entry.feature != AVBD_FEATURE_FACE)
						continue;
					PxVec3 outwardLocal =
						rigidVertexIndex <
								convex.vertexNormals.size()
							? convex.vertexNormals[
								rigidVertexIndex]
							: convex.vertices[
								rigidVertexIndex];
					if(!outwardLocal.isFinite() ||
						outwardLocal.magnitudeSquared() <=
							1.0e-12f)
						outwardLocal =
							PxVec3(0.0f, 1.0f, 0.0f);
					outwardLocal.normalize();
					const PxQuat entryRotation =
						rotationsEquivalent
							? rotationEnd.getNormalized()
							: PxSlerp(
								entry.entryTime,
								rotationStart.getNormalized(),
								rotationEnd.getNormalized()).
									getNormalized();
					const PxVec3 outward =
						entryRotation.rotate(outwardLocal);
					PxVec3 normal = entry.normal;
					if(normal.dot(outward) <= 0.0f)
						continue;

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, convex.primitiveKey,
						avbdGetRigidSoftFeatureKey(
							0x43565646u,
							v0, v1, v2,
							rigidVertexIndex));
					geometry.particleIdx =
						particles[v0].invMass > 0.0f ? v0 :
							(particles[v1].invMass > 0.0f
								? v1 : v2);
					geometry.queryParticleIndices[0] = v0;
					geometry.queryParticleIndices[1] = v1;
					geometry.queryParticleIndices[2] = v2;
					geometry.queryWeights[0] =
						entry.barycentric.x;
					geometry.queryWeights[1] =
						entry.barycentric.y;
					geometry.queryWeights[2] =
						entry.barycentric.z;
					geometry.normal = normal.getNormalized();
					geometry.projNormal = geometry.normal;
					geometry.depth = 0.0f;
					geometry.margin = margin;
					avbdConfigureRigidConvexTarget(
						geometry, convex, convexIndex,
						convex.vertices[rigidVertexIndex]);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							convex.friction,
							convex.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1.0e7f, 1.0e6f,
						particles, contacts);
				}
			}
		}
	}
}

inline void avbdDetectSoftRigidConvexOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f)
{
	const PxReal featureEpsilon = 1.0e-4f;
	const PxReal distanceEpsilon = 1.0e-8f;
	for(PxU32 bodyIndex = 0;
		bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		PxVec3 bodyMinimum(PX_MAX_F32);
		PxVec3 bodyMaximum(-PX_MAX_F32);
		for(PxU32 localParticle = 0;
			localParticle < body.compiled.particleCount;
			++localParticle)
		{
			const PxU32 particleIndex =
				body.compiled.particleStart + localParticle;
			if(particleIndex >= numParticles)
				continue;
			bodyMinimum = bodyMinimum.minimum(
				particles[particleIndex].position);
			bodyMaximum = bodyMaximum.maximum(
				particles[particleIndex].position);
		}

		for(PxU32 convexIndex = 0;
			convexIndex < numConvexes; ++convexIndex)
		{
			const AvbdRigidConvex& convex =
				convexes[convexIndex];
			PxArray<AvbdRigidConvexEdgeBounds>& edgeBoundsScratch =
				convex.edgeBoundsScratch;
			if(!avbdIsRigidConvexValid(convex))
				continue;
			const PxReal broadphaseRadius =
				convex.localRadius + margin;
			if(bodyMinimum.x >
					convex.center.x + broadphaseRadius ||
				bodyMaximum.x <
					convex.center.x - broadphaseRadius ||
				bodyMinimum.y >
					convex.center.y + broadphaseRadius ||
				bodyMaximum.y <
					convex.center.y - broadphaseRadius ||
				bodyMinimum.z >
					convex.center.z + broadphaseRadius ||
				bodyMaximum.z <
					convex.center.z - broadphaseRadius)
				continue;
			const PxQuat inverseRotation =
				convex.rotation.getConjugate();
			if(edgeBoundsScratch.capacity() < convex.edges.size())
				edgeBoundsScratch.reserve(convex.edges.size());
			edgeBoundsScratch.resize(convex.edges.size());
			for(PxU32 rigidEdgeIndex = 0;
				rigidEdgeIndex < convex.edges.size(); ++rigidEdgeIndex)
			{
				const AvbdRigidConvexEdge& rigidEdge =
					convex.edges[rigidEdgeIndex];
				if(rigidEdge.p0 >= convex.vertices.size() ||
					rigidEdge.p1 >= convex.vertices.size())
					continue;
				AvbdRigidConvexEdgeBounds& edgeBounds =
					edgeBoundsScratch[rigidEdgeIndex];
				edgeBounds.point0 = convex.vertices[rigidEdge.p0];
				edgeBounds.point1 = convex.vertices[rigidEdge.p1];
				edgeBounds.minimum =
					edgeBounds.point0.minimum(edgeBounds.point1) -
						PxVec3(margin);
				edgeBounds.maximum =
					edgeBounds.point0.maximum(edgeBounds.point1) +
						PxVec3(margin);
			}

			// Convex boundary edge versus soft boundary edge. Endpoint cases
			// remain owned by forward vertex-SDF or reverse vertex-face.
			for(PxU32 softEdgeIndex = 0;
				softEdgeIndex <
					body.compiled.surfaceEdges.size();
				++softEdgeIndex)
			{
				const AvbdEdgeInfo& softEdge =
					body.compiled.surfaceEdges[softEdgeIndex];
				if(softEdge.p0 >= numParticles ||
					softEdge.p1 >= numParticles ||
					(particles[softEdge.p0].invMass <= 0.0f &&
					 particles[softEdge.p1].invMass <= 0.0f))
					continue;
				const PxVec3 soft0Local =
					inverseRotation.rotate(
						particles[softEdge.p0].position -
							convex.center);
				const PxVec3 soft1Local =
					inverseRotation.rotate(
						particles[softEdge.p1].position -
							convex.center);
				for(PxU32 rigidEdgeIndex = 0;
					rigidEdgeIndex < convex.edges.size();
					++rigidEdgeIndex)
				{
					const AvbdRigidConvexEdge& rigidEdge =
						convex.edges[rigidEdgeIndex];
					if(rigidEdge.p0 >= convex.vertices.size() ||
						rigidEdge.p1 >= convex.vertices.size())
						continue;
					const AvbdRigidConvexEdgeBounds& edgeBounds =
						edgeBoundsScratch[rigidEdgeIndex];
					const PxVec3& rigid0Local = edgeBounds.point0;
					const PxVec3& rigid1Local = edgeBounds.point1;
					const PxVec3 softMinimum =
						soft0Local.minimum(soft1Local);
					const PxVec3 softMaximum =
						soft0Local.maximum(soft1Local);
					const PxVec3& rigidMinimum = edgeBounds.minimum;
					const PxVec3& rigidMaximum = edgeBounds.maximum;
					if(softMinimum.x > rigidMaximum.x ||
						softMaximum.x < rigidMinimum.x ||
						softMinimum.y > rigidMaximum.y ||
						softMaximum.y < rigidMinimum.y ||
						softMinimum.z > rigidMaximum.z ||
						softMaximum.z < rigidMinimum.z)
						continue;
					PxReal softWeight1 = 0.0f;
					PxReal rigidWeight1 = 0.0f;
					PxVec3 softClosestLocal;
					PxVec3 rigidClosestLocal;
					avbdClosestPointsOnSegments(
						soft0Local, soft1Local,
						rigid0Local, rigid1Local,
						softWeight1, rigidWeight1,
						softClosestLocal, rigidClosestLocal);
					if(softWeight1 <= featureEpsilon ||
						softWeight1 >=
							1.0f - featureEpsilon ||
						rigidWeight1 <= featureEpsilon ||
						rigidWeight1 >=
							1.0f - featureEpsilon)
						continue;
					PxVec3 deltaLocal =
						softClosestLocal - rigidClosestLocal;
					const PxReal distance =
						deltaLocal.magnitude();
					if(!PxIsFinite(distance) ||
						distance >= margin)
						continue;
					PxVec3 normalLocal =
						distance > distanceEpsilon
							? deltaLocal * (1.0f / distance)
							: rigidEdge.outward;
					if(normalLocal.dot(rigidEdge.outward) < 0.0f)
						normalLocal = -normalLocal;
					if(!normalLocal.isFinite() ||
						normalLocal.magnitudeSquared() <=
							1.0e-12f)
						continue;

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, convex.primitiveKey,
						avbdGetRigidSoftFeatureKey(
							0x43564545u,
							softEdge.p0, softEdge.p1,
							0u, rigidEdgeIndex));
					geometry.particleIdx =
						particles[softEdge.p0].invMass > 0.0f
							? softEdge.p0 : softEdge.p1;
					geometry.queryParticleIndices[0] =
						softEdge.p0;
					geometry.queryParticleIndices[1] =
						softEdge.p1;
					geometry.queryWeights[0] =
						1.0f - softWeight1;
					geometry.queryWeights[1] =
						softWeight1;
					geometry.normal =
						convex.rotation.rotate(normalLocal).
							getNormalized();
					geometry.projNormal = geometry.normal;
					geometry.depth = margin - distance;
					geometry.margin = margin;
					avbdConfigureRigidConvexTarget(
						geometry, convex, convexIndex,
						rigidClosestLocal);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							convex.friction,
							convex.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1e5f, 1e6f,
						particles, contacts);
				}
			}

			// Convex vertex versus soft face. Soft vertex/edge closest
			// features are excluded so the forward/edge-edge paths retain
			// unique physical feature ownership.
			for(PxU32 triangleOffset = 0;
				triangleOffset + 2 <
					body.compiled.surfaceTriangles.size();
				triangleOffset += 3)
			{
				const PxU32 v0 =
					body.compiled.surfaceTriangles[
						triangleOffset];
				const PxU32 v1 =
					body.compiled.surfaceTriangles[
						triangleOffset + 1];
				const PxU32 v2 =
					body.compiled.surfaceTriangles[
						triangleOffset + 2];
				if(v0 >= numParticles || v1 >= numParticles ||
					v2 >= numParticles ||
					(particles[v0].invMass <= 0.0f &&
					 particles[v1].invMass <= 0.0f &&
					 particles[v2].invMass <= 0.0f))
					continue;
				const PxVec3& p0 = particles[v0].position;
				const PxVec3& p1 = particles[v1].position;
				const PxVec3& p2 = particles[v2].position;
				const PxVec3 p0Local =
					inverseRotation.rotate(p0 - convex.center);
				const PxVec3 p1Local =
					inverseRotation.rotate(p1 - convex.center);
				const PxVec3 p2Local =
					inverseRotation.rotate(p2 - convex.center);
				const PxVec3 triangleMinimum =
					p0Local.minimum(p1Local).minimum(p2Local) -
						PxVec3(margin);
				const PxVec3 triangleMaximum =
					p0Local.maximum(p1Local).maximum(p2Local) +
						PxVec3(margin);

				for(PxU32 rigidVertexIndex = 0;
					rigidVertexIndex < convex.vertices.size();
					++rigidVertexIndex)
				{
					const PxVec3& rigidVertexLocal =
						convex.vertices[rigidVertexIndex];
					if(rigidVertexLocal.x < triangleMinimum.x ||
						rigidVertexLocal.x > triangleMaximum.x ||
						rigidVertexLocal.y < triangleMinimum.y ||
						rigidVertexLocal.y > triangleMaximum.y ||
						rigidVertexLocal.z < triangleMinimum.z ||
						rigidVertexLocal.z > triangleMaximum.z)
						continue;
					const PxVec3 rigidVertexWorld =
						convex.center +
						convex.rotation.rotate(
							rigidVertexLocal);
					const AvbdClosestPointResult closest =
						avbdClosestPointOnTriangleOGC(
							rigidVertexWorld, p0, p1, p2);
					if(closest.feature != AVBD_FEATURE_FACE ||
						!PxIsFinite(closest.distance) ||
						closest.distance >= margin)
						continue;
					PxVec3 outwardLocal =
						rigidVertexIndex <
								convex.vertexNormals.size()
							? convex.vertexNormals[
								rigidVertexIndex]
							: rigidVertexLocal;
					if(!outwardLocal.isFinite() ||
						outwardLocal.magnitudeSquared() <=
							1.0e-12f)
						outwardLocal =
							PxVec3(0.0f, 1.0f, 0.0f);
					outwardLocal.normalize();
					const PxVec3 outwardWorld =
						convex.rotation.rotate(outwardLocal);
					PxVec3 normalWorld =
						closest.distance > distanceEpsilon
							? (closest.point -
								rigidVertexWorld) *
								(1.0f / closest.distance)
							: outwardWorld;
					if(normalWorld.dot(outwardWorld) < 0.0f)
						normalWorld = -normalWorld;
					if(!normalWorld.isFinite() ||
						normalWorld.magnitudeSquared() <=
							1.0e-12f)
						continue;

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, convex.primitiveKey,
						avbdGetRigidSoftFeatureKey(
							0x43565646u,
							v0, v1, v2,
							rigidVertexIndex));
					geometry.particleIdx =
						particles[v0].invMass > 0.0f ? v0 :
							(particles[v1].invMass > 0.0f
								? v1 : v2);
					geometry.queryParticleIndices[0] = v0;
					geometry.queryParticleIndices[1] = v1;
					geometry.queryParticleIndices[2] = v2;
					geometry.queryWeights[0] =
						closest.barycentric.x;
					geometry.queryWeights[1] =
						closest.barycentric.y;
					geometry.queryWeights[2] =
						closest.barycentric.z;
					geometry.normal =
						normalWorld.getNormalized();
					geometry.projNormal = geometry.normal;
					geometry.depth =
						margin - closest.distance;
					geometry.margin = margin;
					avbdConfigureRigidConvexTarget(
						geometry, convex, convexIndex,
						rigidVertexLocal);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							convex.friction,
							convex.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1e5f, 1e6f,
						particles, contacts);
				}
			}
		}
	}
}

PX_FORCE_INLINE bool avbdIsRigidTriangleSurfaceValid(
	const AvbdRigidTriangleSurface& surface)
{
	return surface.center.isFinite() &&
		surface.rotation.isFinite() &&
		!surface.localBounds.isEmpty() &&
		PxIsFinite(surface.localRadius) &&
		surface.localRadius > 0.0f &&
		surface.vertices.size() >= 3 &&
		!surface.triangles.empty();
}

PX_FORCE_INLINE void avbdConfigureRigidTriangleSurfaceTarget(
	AvbdSoftContactGeometry& geometry,
	const AvbdRigidTriangleSurface& surface,
	PxU32 surfaceIndex, const PxVec3& surfaceLocal)
{
	geometry.targetKind = surface.targetKind;
	geometry.velocityOwner =
		surface.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID
			? AvbdVelocityObjectiveOwner::ComponentFinalize
			: AvbdVelocityObjectiveOwner::PositionAL;
	geometry.targetIndex = surfaceIndex;
	geometry.surfacePoint =
		surface.center + surface.rotation.rotate(surfaceLocal);
	geometry.kinematicSurfacePointPrevious =
		surface.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID
			? surface.previousCenter +
				surface.previousRotation.rotate(surfaceLocal)
			: geometry.surfacePoint;
}

PX_FORCE_INLINE PxU64 avbdRigidTriangleSurfaceFeatureKey(
	PxU32 tag, PxU32 featureIndex)
{
	PxU64 hash = 1469598103934665603ull;
	hash = avbdSoftContactHashValue(hash, tag);
	return avbdSoftContactHashValue(hash, featureIndex);
}

PX_FORCE_INLINE bool avbdRigidTriangleSurfaceBvhIntersects(
	const AvbdRigidTriangleSurfaceBvhNode& node,
	const PxVec3& queryMinimum, const PxVec3& queryMaximum,
	PxReal margin)
{
	return node.minimum.x <= queryMaximum.x + margin &&
		node.maximum.x >= queryMinimum.x - margin &&
		node.minimum.y <= queryMaximum.y + margin &&
		node.maximum.y >= queryMinimum.y - margin &&
		node.minimum.z <= queryMaximum.z + margin &&
		node.maximum.z >= queryMinimum.z - margin;
}

inline void avbdCollectRigidTriangleSurfaceBvhNodeCandidates(
	const AvbdRigidTriangleSurface& surface, PxU32 nodeIndex,
	const PxVec3& queryMinimum, const PxVec3& queryMaximum,
	PxReal margin, PxArray<PxU32>& candidates)
{
	const AvbdRigidTriangleSurfaceBvhNode& node =
		surface.triangleBvhNodes[nodeIndex];
	if(!avbdRigidTriangleSurfaceBvhIntersects(
			node, queryMinimum, queryMaximum, margin))
		return;
	if(!node.isLeaf())
	{
		avbdCollectRigidTriangleSurfaceBvhNodeCandidates(
			surface, node.leftChild, queryMinimum, queryMaximum,
			margin, candidates);
		avbdCollectRigidTriangleSurfaceBvhNodeCandidates(
			surface, node.rightChild, queryMinimum, queryMaximum,
			margin, candidates);
		return;
	}
	for(PxU32 entry = node.firstPrimitive;
		entry < node.firstPrimitive + node.primitiveCount; ++entry)
		candidates.pushBack(surface.triangleBvhTriangleIndices[entry]);
}

// Candidate ids are restored to original triangle order before the retained
// exact OGC test. This makes the hierarchy an acceleration only: tie owner,
// contact source order and feature keys stay byte-identical to the linear
// reference traversal.
PX_FORCE_INLINE bool avbdCollectRigidTriangleSurfaceBvhCandidates(
	const AvbdRigidTriangleSurface& surface,
	const PxVec3& queryMinimum, const PxVec3& queryMaximum,
	PxReal margin, PxArray<PxU32>& candidates)
{
	if(!avbdUseRigidTriangleSurfaceBvh() ||
		surface.triangleBvhNodes.empty())
		return false;
	candidates.clear();
	avbdCollectRigidTriangleSurfaceBvhNodeCandidates(
		surface, 0, queryMinimum, queryMaximum, margin, candidates);
	PxSort(candidates.begin(), candidates.size());
	return true;
}

PX_FORCE_INLINE bool avbdBeginRigidTriangleSurfaceFeatureCandidates(
	const AvbdRigidTriangleSurface& surface)
{
	if(++surface.featureBvhCandidateStamp == 0)
	{
		surface.featureBvhCandidateStamp = 1;
		for(PxU32 index = 0;
			index < surface.edgeBvhCandidateStamps.size(); ++index)
			surface.edgeBvhCandidateStamps[index] = 0;
		for(PxU32 index = 0;
			index < surface.vertexBvhCandidateStamps.size(); ++index)
			surface.vertexBvhCandidateStamps[index] = 0;
	}
	return surface.edgeBvhCandidateStamps.size() ==
			surface.edges.size() &&
		surface.vertexBvhCandidateStamps.size() ==
			surface.vertices.size();
}

PX_FORCE_INLINE bool avbdCollectRigidTriangleSurfaceEdgeBvhCandidates(
	const AvbdRigidTriangleSurface& surface,
	const PxVec3& queryMinimum, const PxVec3& queryMaximum,
	PxReal margin, PxArray<PxU32>& candidates,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch = NULL)
{
	PxArray<PxU32>& triangleCandidates = queryScratch
		? queryScratch->triangleBvhQueryCandidates
		: surface.triangleBvhQueryCandidates;
	if(!avbdCollectRigidTriangleSurfaceBvhCandidates(
			surface, queryMinimum, queryMaximum, margin,
			triangleCandidates) ||
		(queryScratch
			? !queryScratch->beginFeatureCandidates(
				surface.edges.size(), surface.vertices.size())
			: !avbdBeginRigidTriangleSurfaceFeatureCandidates(surface)))
		return false;
	candidates.clear();
	const PxU32 stamp = queryScratch
		? queryScratch->featureBvhCandidateStamp
		: surface.featureBvhCandidateStamp;
	PxArray<PxU32>& edgeStamps = queryScratch
		? queryScratch->edgeBvhCandidateStamps
		: surface.edgeBvhCandidateStamps;
	for(PxU32 entry = 0;
		entry < triangleCandidates.size(); ++entry)
	{
		const PxU32 triangleIndex =
			triangleCandidates[entry];
		if(triangleIndex >= surface.triangles.size())
			continue;
		const AvbdRigidTriangleSurfaceTriangle& triangle =
			surface.triangles[triangleIndex];
		const PxU32 triangleEdges[3] =
			{triangle.edge0, triangle.edge1, triangle.edge2};
		for(PxU32 localEdge = 0; localEdge < 3; ++localEdge)
		{
			const PxU32 edgeIndex = triangleEdges[localEdge];
			if(edgeIndex >= surface.edges.size() ||
				!surface.edges[edgeIndex].active ||
				edgeStamps[edgeIndex] == stamp)
				continue;
			edgeStamps[edgeIndex] = stamp;
			candidates.pushBack(edgeIndex);
		}
	}
	PxSort(candidates.begin(), candidates.size());
	return true;
}

PX_FORCE_INLINE bool avbdCollectRigidTriangleSurfaceVertexBvhCandidates(
	const AvbdRigidTriangleSurface& surface,
	const PxVec3& queryMinimum, const PxVec3& queryMaximum,
	PxReal margin, PxArray<PxU32>& candidates,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch = NULL)
{
	PxArray<PxU32>& triangleCandidates = queryScratch
		? queryScratch->triangleBvhQueryCandidates
		: surface.triangleBvhQueryCandidates;
	if(!avbdCollectRigidTriangleSurfaceBvhCandidates(
			surface, queryMinimum, queryMaximum, margin,
			triangleCandidates) ||
		(queryScratch
			? !queryScratch->beginFeatureCandidates(
				surface.edges.size(), surface.vertices.size())
			: !avbdBeginRigidTriangleSurfaceFeatureCandidates(surface)))
		return false;
	candidates.clear();
	const PxU32 stamp = queryScratch
		? queryScratch->featureBvhCandidateStamp
		: surface.featureBvhCandidateStamp;
	PxArray<PxU32>& vertexStamps = queryScratch
		? queryScratch->vertexBvhCandidateStamps
		: surface.vertexBvhCandidateStamps;
	for(PxU32 entry = 0;
		entry < triangleCandidates.size(); ++entry)
	{
		const PxU32 triangleIndex =
			triangleCandidates[entry];
		if(triangleIndex >= surface.triangles.size())
			continue;
		const AvbdRigidTriangleSurfaceTriangle& triangle =
			surface.triangles[triangleIndex];
		const PxU32 triangleVertices[3] =
			{triangle.p0, triangle.p1, triangle.p2};
		for(PxU32 localVertex = 0; localVertex < 3; ++localVertex)
		{
			const PxU32 vertexIndex = triangleVertices[localVertex];
			if(vertexIndex >= surface.vertices.size() ||
				!surface.vertices[vertexIndex].active ||
				vertexStamps[vertexIndex] == stamp)
				continue;
			vertexStamps[vertexIndex] = stamp;
			candidates.pushBack(vertexIndex);
		}
	}
	PxSort(candidates.begin(), candidates.size());
	return true;
}

struct AvbdRigidTriangleSurfacePointQuery
{
	PxReal distance;
	PxVec3 surfaceLocal;
	PxVec3 normalLocal;
	PxReal friction;
	PxU8 frictionCombineMode;
	PxU64 featureKey;

	AvbdRigidTriangleSurfacePointQuery()
		: distance(PX_MAX_F32), surfaceLocal(0.0f),
		  normalLocal(0.0f, 1.0f, 0.0f), friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE)),
		  featureKey(0)
	{
	}
};

// Canonical one-sided point query shared by discrete and continuous triangle
// surface owners. Inactive tessellation seams only own an orthogonal
// projection from an adjacent face; rounded seam features remain excluded.
PX_FORCE_INLINE bool avbdQueryRigidTriangleSurfaceLocal(
	const AvbdRigidTriangleSurface& surface,
	const PxVec3& localPoint, PxReal maximumDistance,
	AvbdRigidTriangleSurfacePointQuery& result,
	AvbdSoftCollisionStats* stats = NULL,
	PxArray<PxU32>* triangleBvhQueryCandidatesOverride = NULL)
{
	if(!avbdIsRigidTriangleSurfaceValid(surface) ||
		!localPoint.isFinite() || maximumDistance <= 0.0f ||
		!PxIsFinite(maximumDistance))
		return false;
	const PxBounds3 expandedBounds(
		surface.localBounds.minimum - PxVec3(maximumDistance),
		surface.localBounds.maximum + PxVec3(maximumDistance));
	if(!expandedBounds.contains(localPoint))
		return false;

	const PxReal normalEpsilon = 1.0e-12f;
	const PxReal featureProjectionTolerance = 1.0e-5f;
	// The legacy serial path reuses the surface-owned candidate storage. A
	// range/task caller supplies private storage so BVH traversal remains a
	// read-only operation over the baked surface topology.
	PxArray<PxU32>& triangleCandidates =
		triangleBvhQueryCandidatesOverride
			? *triangleBvhQueryCandidatesOverride
			: surface.triangleBvhQueryCandidates;
	const bool useTriangleBvh =
		avbdCollectRigidTriangleSurfaceBvhCandidates(
			surface, localPoint, localPoint, maximumDistance,
			triangleCandidates);
	const PxU32 triangleCount = useTriangleBvh
		? triangleCandidates.size() : surface.triangles.size();
	if(stats)
	{
		stats->rigidTriangleSurfaceFaceCandidates += triangleCount;
		stats->rigidTriangleSurfaceFaceTests += triangleCount;
	}
	bool found = false;
	for(PxU32 triangleEntry = 0; triangleEntry < triangleCount;
		++triangleEntry)
	{
		const PxU32 triangleIndex = useTriangleBvh
			? triangleCandidates[triangleEntry] : triangleEntry;
		const AvbdRigidTriangleSurfaceTriangle& triangle =
			surface.triangles[triangleIndex];
		if(triangle.p0 >= surface.vertices.size() ||
			triangle.p1 >= surface.vertices.size() ||
			triangle.p2 >= surface.vertices.size())
			continue;
		const PxVec3& p0 =
			surface.vertices[triangle.p0].point;
		const PxVec3& p1 =
			surface.vertices[triangle.p1].point;
		const PxVec3& p2 =
			surface.vertices[triangle.p2].point;
		const PxReal signedPlaneDistance =
			triangle.normal.dot(localPoint - p0);
		if(!PxIsFinite(signedPlaneDistance) ||
			signedPlaneDistance < 0.0f)
			continue;
		const AvbdClosestPointResult closest =
			avbdClosestPointOnTriangleOGC(
				localPoint, p0, p1, p2);
		if(!PxIsFinite(closest.distance) ||
			closest.distance >= maximumDistance ||
			closest.distance >= result.distance)
			continue;

		PxVec3 featureOutward = triangle.normal;
		PxU64 featureKey =
			avbdRigidTriangleSurfaceFeatureKey(
				0x54534641u, triangleIndex);
		if(closest.feature == AVBD_FEATURE_EDGE)
		{
			const PxU32 edgeIndex =
				closest.featureIndex == 0
					? triangle.edge0
					: closest.featureIndex == 1
						? triangle.edge1
						: triangle.edge2;
			if(edgeIndex >= surface.edges.size())
				continue;
			const AvbdRigidTriangleSurfaceEdge& edge =
				surface.edges[edgeIndex];
			if(!edge.active &&
				closest.distance >
					signedPlaneDistance +
						featureProjectionTolerance)
				continue;
			featureOutward = edge.outward;
			featureKey =
				avbdRigidTriangleSurfaceFeatureKey(
					0x54534544u, edgeIndex);
		}
		else if(closest.feature == AVBD_FEATURE_VERTEX)
		{
			const PxU32 vertexIndex =
				closest.featureIndex == 0
					? triangle.p0
					: closest.featureIndex == 1
						? triangle.p1
						: triangle.p2;
			if(vertexIndex >= surface.vertices.size())
				continue;
			const AvbdRigidTriangleSurfaceVertex& vertex =
				surface.vertices[vertexIndex];
			if(!vertex.active &&
				closest.distance >
					signedPlaneDistance +
						featureProjectionTolerance)
				continue;
			featureOutward = vertex.outward;
			featureKey =
				avbdRigidTriangleSurfaceFeatureKey(
					0x54535654u, vertexIndex);
		}
		if(!featureOutward.isFinite() ||
			featureOutward.magnitudeSquared() <= normalEpsilon)
			continue;
		featureOutward.normalize();
		PxVec3 normalLocal =
			closest.distance > 1.0e-8f
				? closest.normal : featureOutward;
		if(normalLocal.dot(featureOutward) < -1.0e-5f)
			continue;
		if(closest.feature == AVBD_FEATURE_FACE)
			normalLocal = triangle.normal;
		if(!normalLocal.isFinite() ||
			normalLocal.magnitudeSquared() <= normalEpsilon)
			continue;

		result.distance = closest.distance;
		result.surfaceLocal = closest.point;
		result.normalLocal = normalLocal.getNormalized();
		result.friction = triangle.friction;
		result.frictionCombineMode =
			triangle.frictionCombineMode;
		result.featureKey = featureKey;
		found = true;
	}
	return found;
}

// Soft boundary vertex versus an open rigid triangle surface. One contact is
// selected per particle/surface using canonical face/edge/vertex ownership,
// avoiding duplicate contacts at tessellation seams. Back-side points are
// rejected: PxMeshGeometryFlag::eDOUBLE_SIDED does not alter PhysX simulation
// contact semantics.
inline void avbdDetectSoftRigidTriangleSurface(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidTriangleSurface* surfaces, PxU32 numSurfaces,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0,
	AvbdSoftCollisionStats* stats = NULL)
{
	for(PxU32 particleIndex = 0;
		particleIndex < numParticles; ++particleIndex)
	{
		const AvbdSoftParticle& particle = particles[particleIndex];
		if(particle.invMass <= 0.0f ||
			!particle.position.isFinite())
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies, particleIndex);
		if(sourceBody &&
			!avbdIsSoftBodySurfaceVertex(
				*sourceBody, particleIndex))
			continue;

		for(PxU32 surfaceIndex = 0;
			surfaceIndex < numSurfaces; ++surfaceIndex)
		{
			const AvbdRigidTriangleSurface& surface =
				surfaces[surfaceIndex];
			if(!avbdIsRigidTriangleSurfaceValid(surface))
				continue;
			const PxVec3 worldOffset =
				particle.position - surface.center;
			const PxReal broadphaseRadius =
				surface.localRadius + margin;
			if(worldOffset.magnitudeSquared() >
				broadphaseRadius * broadphaseRadius)
				continue;
			const PxVec3 localPoint =
				surface.rotation.getConjugate().rotate(
					worldOffset);
			AvbdRigidTriangleSurfacePointQuery query;
			if(!avbdQueryRigidTriangleSurfaceLocal(
					surface, localPoint, margin, query, stats))
				continue;

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF,
				PX_MAX_U32, surface.primitiveKey,
				query.featureKey);
			geometry.particleIdx = particleIndex;
			geometry.normal =
				surface.rotation.rotate(query.normalLocal).
					getNormalized();
			geometry.projNormal = geometry.normal;
			geometry.depth = margin - query.distance;
			geometry.margin = margin;
			avbdConfigureRigidTriangleSurfaceTarget(
				geometry, surface, surfaceIndex,
				query.surfaceLocal);
			geometry.friction = sourceBody
				? avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction,
					query.friction,
					query.frictionCombineMode)
				: PxMax(query.friction, 0.0f);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1e5f, 1e6f,
				particles, contacts);
		}
	}
}

// P5.8a candidate leaf: current-pose triangle-surface detection is
// particle-major. The surface descriptor has serial legacy BVH scratch, so a
// parallel caller must pass a caller-owned candidate array rather than writing
// that mutable cache. Baked topology and current static pose are read only.
inline void avbdDetectSoftRigidTriangleSurfaceRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidTriangleSurface* surfaces, PxU32 numSurfaces,
	PxArray<AvbdSoftContact>& contacts,
	PxArray<PxU32>& triangleBvhQueryCandidates,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0,
	AvbdSoftCollisionStats* stats = NULL)
{
	PX_ASSERT(particleBegin <= particleEnd);
	PX_ASSERT(particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for(PxU32 particleIndex = particleBegin;
		particleIndex < particleEnd; ++particleIndex)
	{
		const AvbdSoftParticle& particle = particles[particleIndex];
		if(particle.invMass <= 0.0f || !particle.position.isFinite())
			continue;
		const AvbdSoftBody* sourceBody = avbdFindSoftBodyForParticle(
			softBodies, numSoftBodies, particleIndex);
		if(sourceBody && !avbdIsSoftBodySurfaceVertex(
				*sourceBody, particleIndex))
			continue;
		for(PxU32 surfaceIndex = 0; surfaceIndex < numSurfaces;
			++surfaceIndex)
		{
			const AvbdRigidTriangleSurface& surface = surfaces[surfaceIndex];
			if(!avbdIsRigidTriangleSurfaceValid(surface))
				continue;
			const PxVec3 worldOffset = particle.position - surface.center;
			const PxReal broadphaseRadius = surface.localRadius + margin;
			if(worldOffset.magnitudeSquared() >
				broadphaseRadius * broadphaseRadius)
				continue;
			const PxVec3 localPoint =
				surface.rotation.getConjugate().rotate(worldOffset);
			AvbdRigidTriangleSurfacePointQuery query;
			if(!avbdQueryRigidTriangleSurfaceLocal(
					surface, localPoint, margin, query, stats,
					&triangleBvhQueryCandidates))
				continue;
			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF, PX_MAX_U32,
				surface.primitiveKey, query.featureKey);
			geometry.particleIdx = particleIndex;
			geometry.normal = surface.rotation.rotate(query.normalLocal).
				getNormalized();
			geometry.projNormal = geometry.normal;
			geometry.depth = margin - query.distance;
			geometry.margin = margin;
			avbdConfigureRigidTriangleSurfaceTarget(
				geometry, surface, surfaceIndex, query.surfaceLocal);
			geometry.friction = sourceBody
				? avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction, query.friction,
					query.frictionCombineMode)
				: PxMax(query.friction, 0.0f);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1e5f, 1e6f, particles, contacts);
		}
	}
}

PX_FORCE_INLINE bool avbdGetRigidTriangleSurfaceSweepPose(
	const AvbdRigidTriangleSurface& surface,
	PxVec3& centerStart, PxVec3& centerEnd,
	PxQuat& rotationStart, PxQuat& rotationEnd,
	bool& rotationsEquivalent)
{
	if(!avbdIsRigidTriangleSurfaceValid(surface))
		return false;
	const bool kinematicTarget =
		surface.targetKind ==
			AvbdSoftContactTargetKind::eKINEMATIC_RIGID;
	if(surface.targetKind !=
			AvbdSoftContactTargetKind::eWORLD_STATIC &&
		!kinematicTarget)
		return false;
	if(kinematicTarget &&
		(!surface.previousCenter.isFinite() ||
		 !surface.previousRotation.isFinite()))
		return false;

	centerStart =
		kinematicTarget ? surface.previousCenter : surface.center;
	centerEnd = surface.center;
	rotationStart =
		kinematicTarget
			? surface.previousRotation : surface.rotation;
	rotationEnd = surface.rotation;
	if(!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite())
		return false;
	rotationsEquivalent = avbdAreSweepRotationsEquivalent(
		rotationStart, rotationEnd);
	return true;
}

struct AvbdSweptTriangleSurfacePointEntry
{
	PxReal entryTime;
	PxVec3 normalLocal;
	PxVec3 surfaceLocal;
	PxReal friction;
	PxU8 frictionCombineMode;
	PxU64 featureKey;

	AvbdSweptTriangleSurfacePointEntry()
		: entryTime(PX_MAX_F32),
		  normalLocal(0.0f, 1.0f, 0.0f),
		  surfaceLocal(0.0f), friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE)),
		  featureKey(0)
	{
	}
};

PX_FORCE_INLINE void avbdUpdateSweptTriangleSurfacePointEntry(
	AvbdSweptTriangleSurfacePointEntry& result,
	PxReal entryTime, const PxVec3& normalLocal,
	const PxVec3& surfaceLocal, PxReal friction,
	PxU8 frictionCombineMode, PxU64 featureKey)
{
	if(entryTime < 0.0f || entryTime > 1.0f ||
		entryTime >= result.entryTime ||
		!normalLocal.isFinite() ||
		normalLocal.magnitudeSquared() <= 1.0e-12f ||
		!surfaceLocal.isFinite())
		return;
	result.entryTime = entryTime;
	result.normalLocal = normalLocal.getNormalized();
	result.surfaceLocal = surfaceLocal;
	result.friction = friction;
	result.frictionCombineMode = frictionCombineMode;
	result.featureKey = featureKey;
}

// Exact moving-point entry into the cylindrical interior of an expanded
// segment. Rounded endpoint caps are excluded and remain vertex-owned.
PX_FORCE_INLINE bool avbdSegmentEnterExpandedSegmentInterior(
	const PxVec3& segmentStart, const PxVec3& segmentEnd,
	const PxVec3& edge0, const PxVec3& edge1,
	PxReal expandedRadius, PxReal& entryTime,
	PxReal& edgeWeight1, PxVec3& entryNormal)
{
	if(!segmentStart.isFinite() || !segmentEnd.isFinite() ||
		!edge0.isFinite() || !edge1.isFinite() ||
		expandedRadius <= 0.0f ||
		!PxIsFinite(expandedRadius))
		return false;
	const PxVec3 direction = segmentEnd - segmentStart;
	const PxVec3 edge = edge1 - edge0;
	const PxReal edgeLengthSq = edge.magnitudeSquared();
	if(direction.magnitudeSquared() <= 1.0e-12f ||
		edgeLengthSq <= 1.0e-12f ||
		!PxIsFinite(edgeLengthSq))
		return false;

	const PxVec3 startOffset = segmentStart - edge0;
	const PxReal startWeight =
		startOffset.dot(edge) / edgeLengthSq;
	const PxReal weightDirection =
		direction.dot(edge) / edgeLengthSq;
	const PxVec3 radialStart =
		startOffset - edge * startWeight;
	const PxVec3 radialDirection =
		direction - edge * weightDirection;
	const PxReal quadraticA =
		radialDirection.magnitudeSquared();
	const PxReal halfB =
		radialStart.dot(radialDirection);
	const PxReal quadraticC =
		radialStart.magnitudeSquared() -
			expandedRadius * expandedRadius;
	if(quadraticA <= 1.0e-12f ||
		!PxIsFinite(quadraticA) || quadraticC < 0.0f)
		return false;
	const PxReal discriminant =
		halfB * halfB - quadraticA * quadraticC;
	if(discriminant < 0.0f || !PxIsFinite(discriminant))
		return false;
	entryTime =
		(-halfB - PxSqrt(discriminant)) / quadraticA;
	if(entryTime < 0.0f || entryTime > 1.0f)
		return false;
	edgeWeight1 =
		startWeight + weightDirection * entryTime;
	const PxReal featureEpsilon = 1.0e-4f;
	if(edgeWeight1 <= featureEpsilon ||
		edgeWeight1 >= 1.0f - featureEpsilon)
		return false;
	const PxVec3 radial =
		radialStart + radialDirection * entryTime;
	if(!radial.isFinite() ||
		radial.magnitudeSquared() <= 1.0e-12f)
		return false;
	entryNormal = radial.getNormalized();
	return true;
}

// Exact translation-only entry of a point into the one-sided triangle
// surface offset. Face slabs, active finite edge cylinders, and active vertex
// caps are tested independently, then reduced to the earliest canonical
// owner. The current discrete owner suppresses duplicate swept contacts.
PX_FORCE_INLINE bool avbdSegmentEnterExpandedTriangleSurface(
	const AvbdRigidTriangleSurface& surface,
	const PxVec3& segmentStartLocal,
	const PxVec3& segmentEndLocal, PxReal margin,
	AvbdSweptTriangleSurfacePointEntry& result,
	AvbdSoftCollisionStats* stats = NULL,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch = NULL)
{
	if(!segmentStartLocal.isFinite() ||
		!segmentEndLocal.isFinite() ||
		margin <= 0.0f || !PxIsFinite(margin))
		return false;
	const PxVec3 direction =
		segmentEndLocal - segmentStartLocal;
	if(direction.magnitudeSquared() <= 1.0e-12f)
		return false;
	AvbdRigidTriangleSurfacePointQuery currentQuery;
	if(avbdQueryRigidTriangleSurfaceLocal(
			surface, segmentStartLocal, margin,
			currentQuery, stats, queryScratch
				? &queryScratch->triangleBvhQueryCandidates : NULL))
		return false;
	const PxVec3 queryMinimum =
		segmentStartLocal.minimum(segmentEndLocal);
	const PxVec3 queryMaximum =
		segmentStartLocal.maximum(segmentEndLocal);
	PxArray<PxU32>& triangleCandidates = queryScratch
		? queryScratch->triangleBvhQueryCandidates
		: surface.triangleBvhQueryCandidates;
	const bool useTriangleBvh =
		avbdCollectRigidTriangleSurfaceBvhCandidates(
			surface, queryMinimum, queryMaximum, margin,
			triangleCandidates);
	const PxU32 triangleCount = useTriangleBvh
		? triangleCandidates.size() : surface.triangles.size();
	if(stats)
	{
		stats->rigidTriangleSurfaceFaceCandidates += triangleCount;
		stats->rigidTriangleSurfaceFaceTests += triangleCount;
	}

	const PxReal projectionTolerance = 1.0e-5f;
	for(PxU32 triangleEntry = 0; triangleEntry < triangleCount;
		++triangleEntry)
	{
		const PxU32 triangleIndex = useTriangleBvh
			? triangleCandidates[triangleEntry] : triangleEntry;
		const AvbdRigidTriangleSurfaceTriangle& triangle =
			surface.triangles[triangleIndex];
		if(triangle.p0 >= surface.vertices.size() ||
			triangle.p1 >= surface.vertices.size() ||
			triangle.p2 >= surface.vertices.size() ||
			!triangle.normal.isFinite())
			continue;
		const PxVec3& p0 =
			surface.vertices[triangle.p0].point;
		const PxVec3& p1 =
			surface.vertices[triangle.p1].point;
		const PxVec3& p2 =
			surface.vertices[triangle.p2].point;
		const PxReal startPlaneDistance =
			triangle.normal.dot(segmentStartLocal - p0);
		const PxReal planeDirection =
			triangle.normal.dot(direction);
		if(!PxIsFinite(startPlaneDistance) ||
			startPlaneDistance < margin ||
			planeDirection >= -1.0e-12f)
			continue;
		const PxReal entryTime =
			(margin - startPlaneDistance) /
				planeDirection;
		if(entryTime < 0.0f || entryTime > 1.0f ||
			entryTime >= result.entryTime)
			continue;
		const PxVec3 centerAtEntry =
			segmentStartLocal + direction * entryTime;
		const PxVec3 projected =
			centerAtEntry - triangle.normal * margin;
		const AvbdClosestPointResult closest =
			avbdClosestPointOnTriangleOGC(
				projected, p0, p1, p2);
		if(!PxIsFinite(closest.distance) ||
			closest.distance > projectionTolerance)
			continue;

		PxVec3 featureOutward = triangle.normal;
		PxU64 featureKey =
			avbdRigidTriangleSurfaceFeatureKey(
				0x54534641u, triangleIndex);
		if(closest.feature == AVBD_FEATURE_EDGE)
		{
			const PxU32 edgeIndex =
				closest.featureIndex == 0
					? triangle.edge0
					: closest.featureIndex == 1
						? triangle.edge1
						: triangle.edge2;
			if(edgeIndex >= surface.edges.size())
				continue;
			featureOutward =
				surface.edges[edgeIndex].outward;
			featureKey =
				avbdRigidTriangleSurfaceFeatureKey(
					0x54534544u, edgeIndex);
		}
		else if(closest.feature == AVBD_FEATURE_VERTEX)
		{
			const PxU32 vertexIndex =
				closest.featureIndex == 0
					? triangle.p0
					: closest.featureIndex == 1
						? triangle.p1
						: triangle.p2;
			if(vertexIndex >= surface.vertices.size())
				continue;
			featureOutward =
				surface.vertices[vertexIndex].outward;
			featureKey =
				avbdRigidTriangleSurfaceFeatureKey(
					0x54535654u, vertexIndex);
		}
		if(!featureOutward.isFinite() ||
			triangle.normal.dot(featureOutward) < -1.0e-5f)
			continue;
		avbdUpdateSweptTriangleSurfacePointEntry(
			result, entryTime, triangle.normal,
			closest.point, triangle.friction,
			triangle.frictionCombineMode, featureKey);
	}

	PxArray<PxU32>& edgeCandidates = queryScratch
		? queryScratch->edgeBvhQueryCandidates
		: surface.edgeBvhQueryCandidates;
	const bool useEdgeBvh =
		avbdCollectRigidTriangleSurfaceEdgeBvhCandidates(
			surface, queryMinimum, queryMaximum, margin,
			edgeCandidates, queryScratch);
	const PxU32 edgeCount = useEdgeBvh
		? edgeCandidates.size() : surface.edges.size();
	for(PxU32 edgeEntry = 0; edgeEntry < edgeCount; ++edgeEntry)
	{
		const PxU32 edgeIndex = useEdgeBvh
			? edgeCandidates[edgeEntry] : edgeEntry;
		const AvbdRigidTriangleSurfaceEdge& edge =
			surface.edges[edgeIndex];
		if(!edge.active ||
			edge.p0 >= surface.vertices.size() ||
			edge.p1 >= surface.vertices.size())
			continue;
		if(stats)
		{
			stats->rigidTriangleSurfaceEdgeCandidates++;
			stats->rigidTriangleSurfaceEdgeTests++;
		}
		const PxVec3& edge0 =
			surface.vertices[edge.p0].point;
		const PxVec3& edge1 =
			surface.vertices[edge.p1].point;
		PxReal entryTime = 0.0f;
		PxReal edgeWeight1 = 0.0f;
		PxVec3 entryNormal(0.0f);
		if(!avbdSegmentEnterExpandedSegmentInterior(
				segmentStartLocal, segmentEndLocal,
				edge0, edge1, margin, entryTime,
				edgeWeight1, entryNormal) ||
			entryTime >= result.entryTime ||
			entryNormal.dot(edge.outward) < -1.0e-5f)
			continue;
		avbdUpdateSweptTriangleSurfacePointEntry(
			result, entryTime, entryNormal,
			edge0 * (1.0f - edgeWeight1) +
				edge1 * edgeWeight1,
			edge.friction, edge.frictionCombineMode,
			avbdRigidTriangleSurfaceFeatureKey(
				0x54534544u, edgeIndex));
	}

	PxArray<PxU32>& vertexCandidates = queryScratch
		? queryScratch->vertexBvhQueryCandidates
		: surface.vertexBvhQueryCandidates;
	const bool useVertexBvh =
		avbdCollectRigidTriangleSurfaceVertexBvhCandidates(
			surface, queryMinimum, queryMaximum, margin,
			vertexCandidates, queryScratch);
	const PxU32 vertexCount = useVertexBvh
		? vertexCandidates.size() : surface.vertices.size();
	for(PxU32 vertexEntry = 0; vertexEntry < vertexCount;
		++vertexEntry)
	{
		const PxU32 vertexIndex = useVertexBvh
			? vertexCandidates[vertexEntry] : vertexEntry;
		const AvbdRigidTriangleSurfaceVertex& vertex =
			surface.vertices[vertexIndex];
		if(!vertex.active)
			continue;
		if(stats)
		{
			stats->rigidTriangleSurfaceVertexCandidates++;
			stats->rigidTriangleSurfaceVertexTests++;
		}
		PxReal entryTime = 0.0f;
		PxVec3 entryNormal(0.0f);
		if(!avbdSegmentEnterExpandedSphere(
				segmentStartLocal, segmentEndLocal,
				vertex.point, margin, entryTime,
				entryNormal) ||
			entryTime >= result.entryTime ||
			entryNormal.dot(vertex.outward) < -1.0e-5f)
			continue;
		avbdUpdateSweptTriangleSurfacePointEntry(
			result, entryTime, entryNormal,
			vertex.point, vertex.friction,
			vertex.frictionCombineMode,
			avbdRigidTriangleSurfaceFeatureKey(
				0x54535654u, vertexIndex));
	}
	return result.entryTime <= 1.0f;
}

// Continuous point entry against a translating/rotating one-sided triangle
// surface. Each shortest-path slerped pose uses the canonical exact local
// face/active-edge/active-vertex query. Relative point/center translation
// plus localRadius*angularDistance bounds the surface speed, so the
// conservative step cannot cross first contact.
PX_FORCE_INLINE bool avbdSegmentEnterExpandedRotatingTriangleSurface(
	const AvbdRigidTriangleSurface& surface,
	const PxVec3& pointStart, const PxVec3& pointEnd,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	PxReal margin, AvbdSweptTriangleSurfacePointEntry& result,
	AvbdSoftCollisionStats* stats = NULL,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch = NULL)
{
	if(!avbdIsRigidTriangleSurfaceValid(surface) ||
		!pointStart.isFinite() || !pointEnd.isFinite() ||
		!centerStart.isFinite() || !centerEnd.isFinite() ||
		!rotationStart.isFinite() || !rotationEnd.isFinite() ||
		margin <= 0.0f || !PxIsFinite(margin))
		return false;

	PxReal angularDistance = 0.0f;
	if(!avbdGetSweepAngularDistance(
			rotationStart, rotationEnd, angularDistance) ||
		angularDistance <= 0.0f)
		return false;
	const PxQuat normalizedStart = rotationStart.getNormalized();
	const PxQuat normalizedEnd = rotationEnd.getNormalized();
	const PxVec3 relativeTranslation =
		(pointEnd - pointStart) - (centerEnd - centerStart);
	const PxReal speed =
		relativeTranslation.magnitude() +
		surface.localRadius * angularDistance;
	if(speed <= 1.0e-8f || !PxIsFinite(speed))
		return false;
	const PxReal distanceTolerance =
		PxMax(1.0e-5f, margin * 1.0e-5f);

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 64; ++iteration)
	{
		const PxVec3 point =
			pointStart + (pointEnd - pointStart) * time;
		const PxVec3 center =
			centerStart + (centerEnd - centerStart) * time;
		const PxQuat rotation =
			PxSlerp(time, normalizedStart, normalizedEnd).
				getNormalized();
		if(!point.isFinite() || !center.isFinite() ||
			!rotation.isFinite())
			return false;
		const PxVec3 localPoint =
			rotation.getConjugate().rotate(point - center);
		const PxReal maximumDistance =
			localPoint.magnitude() + surface.localRadius +
				margin + 1.0f;
		if(!PxIsFinite(maximumDistance) ||
			maximumDistance <= margin)
			return false;
		AvbdRigidTriangleSurfacePointQuery query;
		if(!avbdQueryRigidTriangleSurfaceLocal(
				surface, localPoint, maximumDistance, query, stats,
				queryScratch
					? &queryScratch->triangleBvhQueryCandidates : NULL))
			return false;
		if(iteration == 0 && query.distance < margin)
			return false;
		const PxReal gap = query.distance - margin;
		if(gap <= distanceTolerance)
		{
			result.entryTime = time;
			result.normalLocal = query.normalLocal;
			result.surfaceLocal = query.surfaceLocal;
			result.friction = query.friction;
			result.frictionCombineMode =
				query.frictionCombineMode;
			result.featureKey = query.featureKey;
			return true;
		}
		const PxReal nextTime = time + gap / speed;
		if(!PxIsFinite(nextTime) || nextTime > 1.0f ||
			nextTime <= time)
			return false;
		time = nextTime;
	}
	return false;
}

inline void avbdDetectSoftRigidTriangleSurfaceSweptImpl(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidTriangleSurface* surfaces,
	PxU32 numSurfaces, PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0,
	AvbdSoftCollisionStats* stats = NULL,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch = NULL)
{
	PX_ASSERT(particleBegin <= particleEnd && particleEnd <= numParticles);
	PX_UNUSED(numParticles);
	for(PxU32 particleIndex = particleBegin;
		particleIndex < particleEnd; ++particleIndex)
	{
		const AvbdSoftParticle& particle =
			particles[particleIndex];
		if(particle.invMass <= 0.0f ||
			!particle.position.isFinite() ||
			!particle.predictedPosition.isFinite())
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies, particleIndex);
		if(!sourceBody ||
			!sourceBody->compiled.speculativeCCDEnabled ||
			!avbdIsSoftBodySurfaceVertex(
				*sourceBody, particleIndex))
			continue;

		for(PxU32 surfaceIndex = 0;
			surfaceIndex < numSurfaces; ++surfaceIndex)
		{
			const AvbdRigidTriangleSurface& surface =
				surfaces[surfaceIndex];
			PxVec3 centerStart(0.0f);
			PxVec3 centerEnd(0.0f);
			PxQuat rotationStart(PxIdentity);
			PxQuat rotationEnd(PxIdentity);
			bool rotationsEquivalent = false;
			if(!avbdGetRigidTriangleSurfaceSweepPose(
					surface, centerStart, centerEnd,
					rotationStart, rotationEnd,
					rotationsEquivalent))
				continue;

			AvbdSweptTriangleSurfacePointEntry entry;
			PxQuat entryRotation(PxIdentity);
			if(rotationsEquivalent)
			{
				const PxQuat inverseRotation =
					rotationEnd.getConjugate();
				const PxVec3 relativeStart =
					inverseRotation.rotate(
						particle.position - centerStart);
				const PxVec3 relativeEnd =
					inverseRotation.rotate(
						particle.predictedPosition - centerEnd);
				const PxBounds3 sweptBounds(
					relativeStart.minimum(relativeEnd),
					relativeStart.maximum(relativeEnd));
				const PxBounds3 expandedSurfaceBounds(
					surface.localBounds.minimum -
						PxVec3(margin),
					surface.localBounds.maximum +
						PxVec3(margin));
				if(!sweptBounds.intersects(
						expandedSurfaceBounds) ||
					!avbdSegmentEnterExpandedTriangleSurface(
						surface, relativeStart, relativeEnd,
						margin, entry, stats, queryScratch))
					continue;
				entryRotation = rotationEnd.getNormalized();
			}
			else
			{
				const PxReal rotationExtent =
					surface.localRadius + margin;
				const PxVec3 pointMinimum =
					particle.position.minimum(
						particle.predictedPosition);
				const PxVec3 pointMaximum =
					particle.position.maximum(
						particle.predictedPosition);
				const PxVec3 centerMinimum =
					centerStart.minimum(centerEnd) -
						PxVec3(rotationExtent);
				const PxVec3 centerMaximum =
					centerStart.maximum(centerEnd) +
						PxVec3(rotationExtent);
				if(pointMinimum.x > centerMaximum.x ||
					pointMaximum.x < centerMinimum.x ||
					pointMinimum.y > centerMaximum.y ||
					pointMaximum.y < centerMinimum.y ||
					pointMinimum.z > centerMaximum.z ||
					pointMaximum.z < centerMinimum.z ||
					!avbdSegmentEnterExpandedRotatingTriangleSurface(
						surface, particle.position,
						particle.predictedPosition,
						centerStart, centerEnd,
						rotationStart, rotationEnd,
						margin, entry, stats, queryScratch))
					continue;
				entryRotation = PxSlerp(
					entry.entryTime,
					rotationStart.getNormalized(),
					rotationEnd.getNormalized()).getNormalized();
			}

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eRIGID_SDF,
				PX_MAX_U32, surface.primitiveKey,
				entry.featureKey);
			geometry.particleIdx = particleIndex;
			geometry.normal =
				entryRotation.rotate(entry.normalLocal).
					getNormalized();
			geometry.projNormal = geometry.normal;
			geometry.depth = 0.0f;
			geometry.margin = margin;
			avbdConfigureRigidTriangleSurfaceTarget(
				geometry, surface, surfaceIndex,
				entry.surfaceLocal);
			geometry.friction =
				avbdCombineDeformableRigidFriction(
					sourceBody->material.dynamicFriction,
					entry.friction,
					entry.frictionCombineMode);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry, 1.0e6f, 1.0e6f,
				particles, contacts);
		}
	}
}

inline void avbdDetectSoftRigidTriangleSurfaceSwept(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidTriangleSurface* surfaces,
	PxU32 numSurfaces, PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0,
	AvbdSoftCollisionStats* stats = NULL)
{
	avbdDetectSoftRigidTriangleSurfaceSweptImpl(
		particles, numParticles, 0, numParticles,
		surfaces, numSurfaces, contacts, margin,
		softBodies, numSoftBodies, stats, NULL);
}

// P5.16a candidate leaf: every swept forward-SDF query write is supplied by
// the caller-owned scratch object. The parent must merge all current-SDF
// ranges before any swept-SDF range and retains both OGC feature suffixes.
inline void avbdDetectSoftRigidTriangleSurfaceSweptRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	PxU32 particleBegin, PxU32 particleEnd,
	const AvbdRigidTriangleSurface* surfaces,
	PxU32 numSurfaces, PxArray<AvbdSoftContact>& contacts,
	AvbdRigidTriangleSurfaceQueryScratch& queryScratch,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0,
	AvbdSoftCollisionStats* stats = NULL)
{
	avbdDetectSoftRigidTriangleSurfaceSweptImpl(
		particles, numParticles, particleBegin, particleEnd,
		surfaces, numSurfaces, contacts, margin,
		softBodies, numSoftBodies, stats, &queryScratch);
}

PX_FORCE_INLINE bool avbdTriangleSurfaceForwardVertexOwnsSweep(
	const AvbdRigidTriangleSurface& surface,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	bool rotationsEquivalent,
	const AvbdSoftParticle& particle, PxReal margin,
	AvbdSoftCollisionStats* stats = NULL,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch = NULL)
{
	const PxQuat inverseStart =
		rotationStart.getConjugate();
	const PxVec3 relativeStart =
		inverseStart.rotate(
			particle.initialPosition - centerStart);
	const PxReal maximumDistance =
		relativeStart.magnitude() + surface.localRadius +
			margin + 1.0f;
	if(!PxIsFinite(maximumDistance))
		return false;
	AvbdRigidTriangleSurfacePointQuery currentQuery;
	if(avbdQueryRigidTriangleSurfaceLocal(
			surface, relativeStart, maximumDistance,
			currentQuery, stats, queryScratch
				? &queryScratch->triangleBvhQueryCandidates : NULL) &&
		currentQuery.distance < margin)
		return true;
	AvbdSweptTriangleSurfacePointEntry entry;
	if(!rotationsEquivalent)
		return avbdSegmentEnterExpandedRotatingTriangleSurface(
			surface, particle.initialPosition,
			particle.predictedPosition,
			centerStart, centerEnd,
			rotationStart, rotationEnd, margin, entry, stats, queryScratch);
	const PxVec3 relativeEnd =
		rotationEnd.getConjugate().rotate(
			particle.predictedPosition - centerEnd);
	return avbdSegmentEnterExpandedTriangleSurface(
		surface, relativeStart, relativeEnd, margin, entry, stats, queryScratch);
}

PX_FORCE_INLINE bool avbdTriangleSurfaceForwardVertexOwnsSweepCached(
	const AvbdRigidTriangleSurface& surface, PxU32 surfaceSlot,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	bool rotationsEquivalent, const AvbdSoftParticle& particle,
	PxU32 particleIndex, PxReal margin, AvbdSoftCollisionStats* stats,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch,
	AvbdRigidTriangleSurfaceForwardOwnerResultCache& resultCache)
{
	bool cachedResult = false;
	if(resultCache.lookup(surfaceSlot, particleIndex, cachedResult))
		return cachedResult;
	const bool result = avbdTriangleSurfaceForwardVertexOwnsSweep(
		surface, centerStart, centerEnd, rotationStart, rotationEnd,
		rotationsEquivalent, particle, margin, stats, queryScratch);
	resultCache.store(surfaceSlot, particleIndex, result);
	return result;
}

PX_FORCE_INLINE bool avbdTriangleSurfaceForwardVertexOwnsSweepParentCached(
	const AvbdRigidTriangleSurface& surface,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	bool rotationsEquivalent, const AvbdSoftParticle& particle,
	PxU32 particleIndex, PxReal margin, AvbdSoftCollisionStats* stats,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch,
	PxArray<PxU8>& resultCache)
{
	PX_ASSERT(particleIndex < resultCache.size());
	PxU8& state = resultCache[particleIndex];
	if(state != 0)
		return state == 2;
	const bool result = avbdTriangleSurfaceForwardVertexOwnsSweep(
		surface, centerStart, centerEnd, rotationStart, rotationEnd,
		rotationsEquivalent, particle, margin, stats, queryScratch);
	state = result ? PxU8(2) : PxU8(1);
	return result;
}

// P5.17a permits a caller-owned override for every query write in both
// triangle OGC feature suffixes. It is intentionally only a serial-equivalence
// contract: a future range leaf must still preserve body/surface/edge/face
// family order rather than treating the two feature loops as one flat stream.
// Reverse OGC completion for translating/rotating triangle surfaces. Active
// rigid edges sweep against linearly deforming soft edge interiors and active
// rigid vertices sweep against linearly deforming soft face interiors. The
// translation-only kernels remain zero-relative-motion fast paths. A current
// or swept forward soft-vertex owner suppresses each candidate.
inline void avbdDetectSoftRigidTriangleSurfaceSweptOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidTriangleSurface* surfaces,
	PxU32 numSurfaces, const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies, PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	AvbdSoftCollisionStats* stats = NULL,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch = NULL,
	PxArray<PxU8>* persistentForwardOwnerScratch = NULL,
	const AvbdRigidTriangleSurfaceFeatureWorkItem* workItem = NULL,
	AvbdRigidTriangleSurfaceSweptOGCFeatureSubstageTiming*
		sweptSubstageTiming = NULL,
	AvbdRigidTriangleSurfaceForwardOwnerQueryStats*
		forwardOwnerQueryStats = NULL,
	AvbdRigidTriangleSurfaceForwardOwnerResultCache*
		forwardOwnerResultCache = NULL)
{
	typedef AvbdRigidTriangleSurfaceFeatureWorkItem WorkItem;
	if(workItem && workItem->phase != WorkItem::eSWEPT)
		return;
	const PxReal translationToleranceSq = 1.0e-10f;
	PxArray<PxU8>* parentForwardOwnerScratch = workItem ?
		NULL : persistentForwardOwnerScratch;
	if(parentForwardOwnerScratch)
	{
		if(parentForwardOwnerScratch->capacity() < numParticles)
			parentForwardOwnerScratch->reserve(numParticles);
		parentForwardOwnerScratch->resize(numParticles);
	}
	for(PxU32 bodyIndex = 0;
		bodyIndex < numSoftBodies; ++bodyIndex)
	{
		if(workItem && bodyIndex != workItem->bodyIndex)
			continue;
		const AvbdSoftBody& body = softBodies[bodyIndex];
		if(!body.compiled.speculativeCCDEnabled)
			continue;
		for(PxU32 surfaceIndex = 0;
			surfaceIndex < numSurfaces; ++surfaceIndex)
		{
			if(workItem && surfaceIndex != workItem->surfaceIndex)
				continue;
			const AvbdRigidTriangleSurface& surface =
				surfaces[surfaceIndex];
			const PxU32 forwardOwnerResultCacheSurfaceSlot =
				forwardOwnerResultCache ?
					forwardOwnerResultCache->getSurfaceSlot(surfaceIndex) :
					PX_MAX_U32;
			PxVec3 centerStart(0.0f);
			PxVec3 centerEnd(0.0f);
			PxQuat rotationStart(PxIdentity);
			PxQuat rotationEnd(PxIdentity);
			bool rotationsEquivalent = false;
			if(!avbdGetRigidTriangleSurfaceSweepPose(
					surface, centerStart, centerEnd,
					rotationStart, rotationEnd,
					rotationsEquivalent))
				continue;
			if(parentForwardOwnerScratch)
			{
				const PxU32 particleStart = body.compiled.particleStart;
				if(particleStart <= numParticles)
				{
					const PxU32 particleCount = PxMin(
						body.compiled.particleCount,
						numParticles - particleStart);
					for(PxU32 localParticle = 0;
						localParticle < particleCount; ++localParticle)
						(*parentForwardOwnerScratch)[
							particleStart + localParticle] = 0;
				}
			}

			PxU32 softEdgeBegin = 0;
			PxU32 softEdgeEnd = 0;
			if(!workItem)
				softEdgeEnd = body.compiled.surfaceEdges.size();
			else if(workItem->family == WorkItem::eSOFT_EDGE)
			{
				softEdgeBegin = PxMin(workItem->primitiveBegin,
					body.compiled.surfaceEdges.size());
				softEdgeEnd = PxMin(
					PxMax(workItem->primitiveEnd, softEdgeBegin),
					body.compiled.surfaceEdges.size());
			}
			for(PxU32 softEdgeIndex = softEdgeBegin;
				softEdgeIndex < softEdgeEnd;
				++softEdgeIndex)
			{
				const AvbdEdgeInfo& softEdge =
					body.compiled.surfaceEdges[softEdgeIndex];
				if(softEdge.p0 >= numParticles ||
					softEdge.p1 >= numParticles ||
					(particles[softEdge.p0].invMass <= 0.0f &&
					 particles[softEdge.p1].invMass <= 0.0f))
					continue;
				const PxVec3 soft0 =
					particles[softEdge.p0].initialPosition;
				const PxVec3 soft1 =
					particles[softEdge.p1].initialPosition;
				const PxVec3 displacement0 =
					particles[softEdge.p0].predictedPosition -
						soft0;
				const PxVec3 displacement1 =
					particles[softEdge.p1].predictedPosition -
						soft1;
				if(!soft0.isFinite() || !soft1.isFinite() ||
					!displacement0.isFinite() ||
					!displacement1.isFinite())
					continue;
				const PxVec3 relativeSoftDisplacement1 =
					displacement1 - displacement0;
				const bool softEdgeTranslationOnly =
					relativeSoftDisplacement1.
						magnitudeSquared() <=
							translationToleranceSq;

				const PxU64 forwardOwnerStartNanos = sweptSubstageTiming ?
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
				const PxU32 softVertices[2] =
					{softEdge.p0, softEdge.p1};
				bool forwardVertexOwns = false;
				for(PxU32 endpoint = 0;
					endpoint < 2 && !forwardVertexOwns;
					++endpoint)
				{
					const PxU32 vertexIndex =
						softVertices[endpoint];
					if(particles[vertexIndex].invMass <= 0.0f)
						continue;
					if(forwardOwnerQueryStats)
						forwardOwnerQueryStats->record(surfaceIndex, vertexIndex);
					forwardVertexOwns = parentForwardOwnerScratch ?
						avbdTriangleSurfaceForwardVertexOwnsSweepParentCached(
							surface, centerStart, centerEnd,
							rotationStart, rotationEnd, rotationsEquivalent,
							particles[vertexIndex], vertexIndex, margin, stats,
							queryScratch, *parentForwardOwnerScratch) :
						forwardOwnerResultCacheSurfaceSlot !=
						PX_MAX_U32 ?
						avbdTriangleSurfaceForwardVertexOwnsSweepCached(
							surface, forwardOwnerResultCacheSurfaceSlot,
							centerStart, centerEnd,
							rotationStart, rotationEnd, rotationsEquivalent,
							particles[vertexIndex], vertexIndex, margin, stats,
							queryScratch, *forwardOwnerResultCache) :
						avbdTriangleSurfaceForwardVertexOwnsSweep(
							surface, centerStart, centerEnd, rotationStart, rotationEnd,
							rotationsEquivalent, particles[vertexIndex], margin,
							stats, queryScratch);
				}
				if(sweptSubstageTiming)
					sweptSubstageTiming->sweptEdgeForwardOwnerNanos +=
						PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
						forwardOwnerStartNanos;
				if(forwardVertexOwns)
					continue;

				const PxVec3 relativeTranslation =
					centerEnd - centerStart - displacement0;
				if(softEdgeTranslationOnly &&
					rotationsEquivalent &&
					relativeTranslation.magnitudeSquared() <=
						translationToleranceSq)
					continue;
				const PxVec3 relativeCenterEnd =
					centerEnd - displacement0;
				const PxVec3 soft0End = soft0;
				const PxVec3 soft1End =
					soft1 + relativeSoftDisplacement1;
				PxArray<PxU32>& rigidEdgeCandidates = queryScratch
					? queryScratch->edgeBvhQueryCandidates
					: surface.edgeBvhQueryCandidates;
				const PxU64 bvhRecoveryStartNanos = sweptSubstageTiming ?
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
				bool useRigidEdgeBvh = false;
				if(softEdgeTranslationOnly && rotationsEquivalent)
				{
					const PxQuat inverseRotation =
						rotationStart.getConjugate();
					const PxVec3 localSoft0 = inverseRotation.rotate(
						soft0 - centerStart);
					const PxVec3 localSoft1 = inverseRotation.rotate(
						soft1 - centerStart);
					const PxVec3 localRelativeTranslation =
						inverseRotation.rotate(relativeTranslation);
					const PxVec3 localSoftMinimum =
						localSoft0.minimum(localSoft1);
					const PxVec3 localSoftMaximum =
						localSoft0.maximum(localSoft1);
					// A rigid local edge E can enter the static local soft edge
					// only when E overlaps the soft edge swept backwards by the
					// relative translation. This conservative local AABB can
					// therefore recover owning triangle leaves.
					useRigidEdgeBvh =
						avbdCollectRigidTriangleSurfaceEdgeBvhCandidates(
							surface,
							localSoftMinimum.minimum(
								localSoftMinimum -
									localRelativeTranslation),
							localSoftMaximum.maximum(
								localSoftMaximum -
									localRelativeTranslation),
							margin, rigidEdgeCandidates, queryScratch);
				}
				if(sweptSubstageTiming)
					sweptSubstageTiming->sweptEdgeBvhRecoveryNanos +=
						PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
						bvhRecoveryStartNanos;
				// Rotation or a deforming soft edge has no independently
				// validated local translation envelope. Keep the legacy full
				// traversal as the conservative authority in those branches.
				const PxU32 rigidEdgeCount = useRigidEdgeBvh
					? rigidEdgeCandidates.size() : surface.edges.size();
				const PxU64 narrowPhaseStartNanos = sweptSubstageTiming ?
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
				for(PxU32 rigidEdgeEntry = 0;
					rigidEdgeEntry < rigidEdgeCount;
					++rigidEdgeEntry)
				{
					const PxU32 rigidEdgeIndex = useRigidEdgeBvh
						? rigidEdgeCandidates[rigidEdgeEntry]
						: rigidEdgeEntry;
					const AvbdRigidTriangleSurfaceEdge& rigidEdge =
						surface.edges[rigidEdgeIndex];
					if(!rigidEdge.active ||
						rigidEdge.p0 >= surface.vertices.size() ||
						rigidEdge.p1 >= surface.vertices.size())
						continue;
					const PxVec3 rigid0 =
						centerStart + rotationStart.rotate(
							surface.vertices[
								rigidEdge.p0].point);
					const PxVec3 rigid1 =
						centerStart + rotationStart.rotate(
							surface.vertices[
								rigidEdge.p1].point);
					PxVec3 rigidMinimum(0.0f);
					PxVec3 rigidMaximum(0.0f);
					if(rotationsEquivalent)
					{
						rigidMinimum =
							rigid0.minimum(rigid1).
								minimum(
									rigid0 + relativeTranslation).
								minimum(
									rigid1 + relativeTranslation) -
									PxVec3(margin);
						rigidMaximum =
							rigid0.maximum(rigid1).
								maximum(
									rigid0 + relativeTranslation).
								maximum(
									rigid1 + relativeTranslation) +
									PxVec3(margin);
					}
					else
					{
						const PxReal rotationExtent =
							surface.localRadius + margin;
						rigidMinimum =
							centerStart.minimum(
								relativeCenterEnd) -
								PxVec3(rotationExtent);
						rigidMaximum =
							centerStart.maximum(
								relativeCenterEnd) +
								PxVec3(rotationExtent);
					}
					const PxVec3 softMinimum =
						soft0.minimum(soft1).
							minimum(soft0End).
							minimum(soft1End);
					const PxVec3 softMaximum =
						soft0.maximum(soft1).
							maximum(soft0End).
							maximum(soft1End);
					if(rigidMinimum.x > softMaximum.x ||
						rigidMaximum.x < softMinimum.x ||
						rigidMinimum.y > softMaximum.y ||
						rigidMaximum.y < softMinimum.y ||
						rigidMinimum.z > softMaximum.z ||
						rigidMaximum.z < softMinimum.z)
						continue;
					if(stats)
					{
						stats->rigidTriangleSurfaceEdgeCandidates++;
						stats->rigidTriangleSurfaceEdgeTests++;
					}

					AvbdSweptConvexEdgeEntry entry;
					const bool entered =
						softEdgeTranslationOnly &&
							rotationsEquivalent
							? avbdTranslatedSegmentEnterExpandedSegmentInteriors(
								rigid0, rigid1,
								relativeTranslation,
								soft0, soft1, margin, entry)
							: softEdgeTranslationOnly
							? avbdRotatingSegmentEnterExpandedSegmentInteriors(
								surface.vertices[
									rigidEdge.p0].point,
								surface.vertices[
									rigidEdge.p1].point,
								centerStart, relativeCenterEnd,
								rotationStart, rotationEnd,
								soft0, soft1, margin, entry)
							: avbdRotatingSegmentEnterExpandedDeformingSegmentInteriors(
								surface.vertices[
									rigidEdge.p0].point,
								surface.vertices[
									rigidEdge.p1].point,
								centerStart, relativeCenterEnd,
								rotationStart, rotationEnd,
								soft0, soft1,
								soft0End, soft1End,
								margin, entry);
					if(!entered)
						continue;
					const PxQuat entryRotation =
						rotationsEquivalent
							? rotationEnd.getNormalized()
							: PxSlerp(
								entry.entryTime,
								rotationStart.getNormalized(),
								rotationEnd.getNormalized()).
									getNormalized();
					const PxVec3 outward =
						entryRotation.rotate(rigidEdge.outward);
					if(entry.normal.dot(outward) <= 0.0f)
						continue;

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, surface.primitiveKey,
						avbdGetRigidSoftFeatureKey(
							0x54534553u,
							softEdge.p0, softEdge.p1,
							0u, rigidEdgeIndex));
					geometry.particleIdx =
						particles[softEdge.p0].invMass > 0.0f
							? softEdge.p0 : softEdge.p1;
					geometry.queryParticleIndices[0] =
						softEdge.p0;
					geometry.queryParticleIndices[1] =
						softEdge.p1;
					geometry.queryWeights[0] =
						1.0f - entry.softWeight1;
					geometry.queryWeights[1] =
						entry.softWeight1;
					geometry.normal =
						entry.normal.getNormalized();
					geometry.projNormal = geometry.normal;
					geometry.depth = 0.0f;
					geometry.margin = margin;
					const PxVec3 surfaceLocal =
						surface.vertices[rigidEdge.p0].point *
							(1.0f - entry.rigidWeight1) +
						surface.vertices[rigidEdge.p1].point *
							entry.rigidWeight1;
					avbdConfigureRigidTriangleSurfaceTarget(
						geometry, surface, surfaceIndex,
						surfaceLocal);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							rigidEdge.friction,
							rigidEdge.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1.0e7f, 1.0e6f,
						particles, contacts);
				}
				if(sweptSubstageTiming)
					sweptSubstageTiming->sweptEdgeNarrowPhaseNanos +=
						PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
						narrowPhaseStartNanos;
			}

			PxU32 softTriangleBegin = 0;
			PxU32 softTriangleEnd = 0;
			const PxU32 softTriangleCount =
				body.compiled.surfaceTriangles.size() / 3;
			if(!workItem)
				softTriangleEnd = softTriangleCount;
			else if(workItem->family == WorkItem::eSOFT_TRIANGLE)
			{
				softTriangleBegin = PxMin(workItem->primitiveBegin,
					softTriangleCount);
				softTriangleEnd = PxMin(
					PxMax(workItem->primitiveEnd, softTriangleBegin),
					softTriangleCount);
			}
			for(PxU32 softTriangleIndex = softTriangleBegin;
				softTriangleIndex < softTriangleEnd;
				++softTriangleIndex)
			{
				const PxU32 triangleOffset = softTriangleIndex * 3;
				const PxU32 v0 =
					body.compiled.surfaceTriangles[
						triangleOffset];
				const PxU32 v1 =
					body.compiled.surfaceTriangles[
						triangleOffset + 1];
				const PxU32 v2 =
					body.compiled.surfaceTriangles[
						triangleOffset + 2];
				if(v0 >= numParticles || v1 >= numParticles ||
					v2 >= numParticles ||
					(particles[v0].invMass <= 0.0f &&
					 particles[v1].invMass <= 0.0f &&
					 particles[v2].invMass <= 0.0f))
					continue;
				const PxVec3 p0 = particles[v0].initialPosition;
				const PxVec3 p1 = particles[v1].initialPosition;
				const PxVec3 p2 = particles[v2].initialPosition;
				const PxVec3 displacement0 =
					particles[v0].predictedPosition - p0;
				const PxVec3 displacement1 =
					particles[v1].predictedPosition - p1;
				const PxVec3 displacement2 =
					particles[v2].predictedPosition - p2;
				if(!p0.isFinite() || !p1.isFinite() ||
					!p2.isFinite() ||
					!displacement0.isFinite() ||
					!displacement1.isFinite() ||
					!displacement2.isFinite())
					continue;
				const PxVec3 relativeDisplacement1 =
					(displacement1 - displacement0);
				const PxVec3 relativeDisplacement2 =
					(displacement2 - displacement0);
				const bool softTriangleTranslationOnly =
					relativeDisplacement1.magnitudeSquared() <=
						translationToleranceSq &&
					relativeDisplacement2.magnitudeSquared() <=
						translationToleranceSq;

				const PxU64 forwardOwnerStartNanos = sweptSubstageTiming ?
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
				const PxU32 triangleVertices[3] =
					{v0, v1, v2};
				bool forwardVertexOwns = false;
				for(PxU32 vertexIndex = 0;
					vertexIndex < 3 && !forwardVertexOwns;
					++vertexIndex)
				{
					const PxU32 particleIndex =
						triangleVertices[vertexIndex];
					if(particles[particleIndex].invMass <= 0.0f)
						continue;
					if(forwardOwnerQueryStats)
						forwardOwnerQueryStats->record(surfaceIndex, particleIndex);
					forwardVertexOwns = parentForwardOwnerScratch ?
						avbdTriangleSurfaceForwardVertexOwnsSweepParentCached(
							surface, centerStart, centerEnd,
							rotationStart, rotationEnd, rotationsEquivalent,
							particles[particleIndex], particleIndex, margin, stats,
							queryScratch, *parentForwardOwnerScratch) :
						forwardOwnerResultCacheSurfaceSlot !=
						PX_MAX_U32 ?
						avbdTriangleSurfaceForwardVertexOwnsSweepCached(
							surface, forwardOwnerResultCacheSurfaceSlot,
							centerStart, centerEnd,
							rotationStart, rotationEnd, rotationsEquivalent,
							particles[particleIndex], particleIndex, margin, stats,
							queryScratch, *forwardOwnerResultCache) :
						avbdTriangleSurfaceForwardVertexOwnsSweep(
							surface, centerStart, centerEnd, rotationStart, rotationEnd,
							rotationsEquivalent, particles[particleIndex], margin,
							stats, queryScratch);
				}
				if(sweptSubstageTiming)
					sweptSubstageTiming->sweptTriangleForwardOwnerNanos +=
						PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
						forwardOwnerStartNanos;
				if(forwardVertexOwns)
					continue;

				const PxVec3 relativeTranslation =
					centerEnd - centerStart - displacement0;
				if(softTriangleTranslationOnly &&
					rotationsEquivalent &&
					relativeTranslation.magnitudeSquared() <=
						translationToleranceSq)
					continue;
				const PxVec3 relativeCenterEnd =
					centerEnd - displacement0;
				const PxVec3 triangleMinimum =
					p0.minimum(p1).minimum(p2).
						minimum(p1 + relativeDisplacement1).
						minimum(p2 + relativeDisplacement2) -
						PxVec3(margin);
				const PxVec3 triangleMaximum =
					p0.maximum(p1).maximum(p2).
						maximum(p1 + relativeDisplacement1).
						maximum(p2 + relativeDisplacement2) +
						PxVec3(margin);
				PxArray<PxU32>& rigidVertexCandidates = queryScratch
					? queryScratch->vertexBvhQueryCandidates
					: surface.vertexBvhQueryCandidates;
				const PxU64 bvhRecoveryStartNanos = sweptSubstageTiming ?
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
				bool useRigidVertexBvh = false;
				if(softTriangleTranslationOnly && rotationsEquivalent)
				{
					const PxQuat inverseRotation =
						rotationStart.getConjugate();
					const PxVec3 localP0 = inverseRotation.rotate(
						p0 - centerStart);
					const PxVec3 localP1 = inverseRotation.rotate(
						p1 - centerStart);
					const PxVec3 localP2 = inverseRotation.rotate(
						p2 - centerStart);
					const PxVec3 localRelativeTranslation =
						inverseRotation.rotate(relativeTranslation);
					const PxVec3 localTriangleMinimum =
						localP0.minimum(localP1).minimum(localP2);
					const PxVec3 localTriangleMaximum =
						localP0.maximum(localP1).maximum(localP2);
					// Transform the relative translation to the stationary
					// surface-local frame before triangle-leaf traversal.
					useRigidVertexBvh =
						avbdCollectRigidTriangleSurfaceVertexBvhCandidates(
							surface,
							localTriangleMinimum.minimum(
								localTriangleMinimum -
									localRelativeTranslation),
							localTriangleMaximum.maximum(
								localTriangleMaximum -
									localRelativeTranslation),
							margin, rigidVertexCandidates, queryScratch);
				}
				if(sweptSubstageTiming)
					sweptSubstageTiming->sweptTriangleBvhRecoveryNanos +=
						PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
						bvhRecoveryStartNanos;
				// Rotation or soft-face deformation retains the exact legacy
				// scan until a separately proven swept local envelope exists.
				const PxU32 rigidVertexCount = useRigidVertexBvh
					? rigidVertexCandidates.size() : surface.vertices.size();
				const PxU64 narrowPhaseStartNanos = sweptSubstageTiming ?
					PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
				for(PxU32 rigidVertexEntry = 0;
					rigidVertexEntry < rigidVertexCount;
					++rigidVertexEntry)
				{
					const PxU32 rigidVertexIndex = useRigidVertexBvh
						? rigidVertexCandidates[rigidVertexEntry]
						: rigidVertexEntry;
					const AvbdRigidTriangleSurfaceVertex& vertex =
						surface.vertices[rigidVertexIndex];
					if(!vertex.active)
						continue;
					const PxVec3 rigidVertexStart =
						centerStart +
							rotationStart.rotate(vertex.point);
					const PxVec3 rigidVertexEnd =
						rotationsEquivalent
							? rigidVertexStart +
								relativeTranslation
							: relativeCenterEnd +
								rotationEnd.rotate(vertex.point);
					PxVec3 sweptMinimum(0.0f);
					PxVec3 sweptMaximum(0.0f);
					if(rotationsEquivalent)
					{
						sweptMinimum =
							rigidVertexStart.minimum(
								rigidVertexEnd);
						sweptMaximum =
							rigidVertexStart.maximum(
								rigidVertexEnd);
					}
					else
					{
						const PxReal rotationExtent =
							surface.localRadius;
						sweptMinimum =
							centerStart.minimum(
								relativeCenterEnd) -
								PxVec3(rotationExtent);
						sweptMaximum =
							centerStart.maximum(
								relativeCenterEnd) +
								PxVec3(rotationExtent);
					}
					if(sweptMinimum.x > triangleMaximum.x ||
						sweptMaximum.x < triangleMinimum.x ||
						sweptMinimum.y > triangleMaximum.y ||
						sweptMaximum.y < triangleMinimum.y ||
						sweptMinimum.z > triangleMaximum.z ||
						sweptMaximum.z < triangleMinimum.z)
						continue;
					if(stats)
					{
						stats->rigidTriangleSurfaceVertexCandidates++;
						stats->rigidTriangleSurfaceVertexTests++;
					}

					AvbdSweptTriangleEntry entry;
					const bool entered =
						softTriangleTranslationOnly &&
							rotationsEquivalent
							? avbdSegmentEnterExpandedTriangleNonVertex(
								rigidVertexStart, rigidVertexEnd,
								p0, p1, p2, margin, entry)
							: softTriangleTranslationOnly
							? avbdRotatingPointEnterExpandedTriangleFace(
								vertex.point,
								centerStart, relativeCenterEnd,
								rotationStart, rotationEnd,
								p0, p1, p2, margin, entry)
							: avbdRotatingPointEnterExpandedDeformingTriangleFace(
								vertex.point,
								centerStart, relativeCenterEnd,
								rotationStart, rotationEnd,
								p0, p1, p2, p0,
								p1 + relativeDisplacement1,
								p2 + relativeDisplacement2,
								margin, entry);
					if(!entered ||
						entry.feature != AVBD_FEATURE_FACE)
						continue;
					const PxQuat entryRotation =
						rotationsEquivalent
							? rotationEnd.getNormalized()
							: PxSlerp(
								entry.entryTime,
								rotationStart.getNormalized(),
								rotationEnd.getNormalized()).
									getNormalized();
					const PxVec3 outward =
						entryRotation.rotate(vertex.outward);
					if(entry.normal.dot(outward) <= 0.0f)
						continue;

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, surface.primitiveKey,
						avbdGetRigidSoftFeatureKey(
							0x54535653u,
							v0, v1, v2,
							rigidVertexIndex));
					geometry.particleIdx =
						particles[v0].invMass > 0.0f ? v0 :
							(particles[v1].invMass > 0.0f
								? v1 : v2);
					geometry.queryParticleIndices[0] = v0;
					geometry.queryParticleIndices[1] = v1;
					geometry.queryParticleIndices[2] = v2;
					geometry.queryWeights[0] =
						entry.barycentric.x;
					geometry.queryWeights[1] =
						entry.barycentric.y;
					geometry.queryWeights[2] =
						entry.barycentric.z;
					geometry.normal =
						entry.normal.getNormalized();
					geometry.projNormal = geometry.normal;
					geometry.depth = 0.0f;
					geometry.margin = margin;
					avbdConfigureRigidTriangleSurfaceTarget(
						geometry, surface, surfaceIndex,
						vertex.point);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							vertex.friction,
							vertex.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1.0e7f, 1.0e6f,
						particles, contacts);
				}
				if(sweptSubstageTiming)
					sweptSubstageTiming->sweptTriangleNarrowPhaseNanos +=
						PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
						narrowPhaseStartNanos;
			}
		}
	}
}

// The local bounds of the eight transformed corners conservatively contain
// every transformed point of the world-axis-aligned body box. A miss against
// the expanded immutable triangle-surface bounds therefore proves that none
// of this row's soft edges/triangles can recover a rigid feature candidate.
PX_FORCE_INLINE bool avbdRigidTriangleSurfaceBodyMayReachLocalBounds(
	const PxVec3& bodyMinimum, const PxVec3& bodyMaximum,
	const AvbdRigidTriangleSurface& surface, const PxQuat& inverseRotation,
	PxReal margin)
{
	if(!bodyMinimum.isFinite() || !bodyMaximum.isFinite() ||
		bodyMinimum.x > bodyMaximum.x || bodyMinimum.y > bodyMaximum.y ||
		bodyMinimum.z > bodyMaximum.z || margin < 0.0f || !PxIsFinite(margin))
		return true;
	PxVec3 localMinimum(PX_MAX_F32);
	PxVec3 localMaximum(-PX_MAX_F32);
	for(PxU32 cornerIndex = 0; cornerIndex < 8; ++cornerIndex)
	{
		const PxVec3 worldPoint(
			(cornerIndex & 1u) ? bodyMaximum.x : bodyMinimum.x,
			(cornerIndex & 2u) ? bodyMaximum.y : bodyMinimum.y,
			(cornerIndex & 4u) ? bodyMaximum.z : bodyMinimum.z);
		const PxVec3 localPoint = inverseRotation.rotate(
			worldPoint - surface.center);
		if(!localPoint.isFinite())
			return true;
		localMinimum = localMinimum.minimum(localPoint);
		localMaximum = localMaximum.maximum(localPoint);
	}
	const PxVec3 expandedMinimum =
		surface.localBounds.minimum - PxVec3(margin);
	const PxVec3 expandedMaximum =
		surface.localBounds.maximum + PxVec3(margin);
	return !(localMinimum.x > expandedMaximum.x ||
		localMaximum.x < expandedMinimum.x ||
		localMinimum.y > expandedMaximum.y ||
		localMaximum.y < expandedMinimum.y ||
		localMinimum.z > expandedMaximum.z ||
		localMaximum.z < expandedMinimum.z);
}

// Reverse OGC completeness for an open triangle surface: active rigid edge
// versus soft boundary edge, and active rigid vertex versus soft face. Feature
// endpoint cases are excluded so forward vertex-triangle owns them.
inline void avbdDetectSoftRigidTriangleSurfaceOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidTriangleSurface* surfaces, PxU32 numSurfaces,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	AvbdSoftCollisionStats* stats = NULL,
	AvbdRigidTriangleSurfaceQueryScratch* queryScratch = NULL,
	const AvbdRigidTriangleSurfaceFeatureWorkItem* workItem = NULL,
	AvbdRigidTriangleSurfaceDiscreteOGCQueryStats*
		discreteQueryStats = NULL,
	bool useBodyLocalBoundsCull = false)
{
	typedef AvbdRigidTriangleSurfaceFeatureWorkItem WorkItem;
	if(workItem && workItem->phase != WorkItem::eDISCRETE)
		return;
	const PxReal featureEpsilon = 1.0e-4f;
	const PxReal distanceEpsilon = 1.0e-8f;
	for(PxU32 bodyIndex = 0;
		bodyIndex < numSoftBodies; ++bodyIndex)
	{
		if(workItem && bodyIndex != workItem->bodyIndex)
			continue;
		const AvbdSoftBody& body = softBodies[bodyIndex];
		PxVec3 bodyMinimum(PX_MAX_F32);
		PxVec3 bodyMaximum(-PX_MAX_F32);
		for(PxU32 localParticle = 0;
			localParticle < body.compiled.particleCount;
			++localParticle)
		{
			const PxU32 particleIndex =
				body.compiled.particleStart + localParticle;
			if(particleIndex >= numParticles)
				continue;
			bodyMinimum = bodyMinimum.minimum(
				particles[particleIndex].position);
			bodyMaximum = bodyMaximum.maximum(
				particles[particleIndex].position);
		}

		for(PxU32 surfaceIndex = 0;
			surfaceIndex < numSurfaces; ++surfaceIndex)
		{
			if(workItem && surfaceIndex != workItem->surfaceIndex)
				continue;
			const AvbdRigidTriangleSurface& surface =
				surfaces[surfaceIndex];
			if(!avbdIsRigidTriangleSurfaceValid(surface))
				continue;
			const PxReal broadphaseRadius =
				surface.localRadius + margin;
			if(bodyMinimum.x >
					surface.center.x + broadphaseRadius ||
				bodyMaximum.x <
					surface.center.x - broadphaseRadius ||
				bodyMinimum.y >
					surface.center.y + broadphaseRadius ||
				bodyMaximum.y <
					surface.center.y - broadphaseRadius ||
				bodyMinimum.z >
					surface.center.z + broadphaseRadius ||
				bodyMaximum.z <
					surface.center.z - broadphaseRadius)
				continue;
			const PxQuat inverseRotation =
				surface.rotation.getConjugate();
			if(useBodyLocalBoundsCull &&
				!avbdRigidTriangleSurfaceBodyMayReachLocalBounds(
					bodyMinimum, bodyMaximum, surface, inverseRotation, margin))
				continue;

			PxU32 softEdgeBegin = 0;
			PxU32 softEdgeEnd = 0;
			if(!workItem)
				softEdgeEnd = body.compiled.surfaceEdges.size();
			else if(workItem->family == WorkItem::eSOFT_EDGE)
			{
				softEdgeBegin = PxMin(workItem->primitiveBegin,
					body.compiled.surfaceEdges.size());
				softEdgeEnd = PxMin(
					PxMax(workItem->primitiveEnd, softEdgeBegin),
					body.compiled.surfaceEdges.size());
			}
			for(PxU32 softEdgeIndex = softEdgeBegin;
				softEdgeIndex < softEdgeEnd;
				++softEdgeIndex)
			{
				const AvbdEdgeInfo& softEdge =
					body.compiled.surfaceEdges[softEdgeIndex];
				if(softEdge.p0 >= numParticles ||
					softEdge.p1 >= numParticles ||
					(particles[softEdge.p0].invMass <= 0.0f &&
					 particles[softEdge.p1].invMass <= 0.0f))
					continue;
				const PxVec3 soft0Local =
					inverseRotation.rotate(
						particles[softEdge.p0].position -
							surface.center);
				const PxVec3 soft1Local =
					inverseRotation.rotate(
						particles[softEdge.p1].position -
							surface.center);
				const PxVec3 softMinimum =
					soft0Local.minimum(soft1Local);
				const PxVec3 softMaximum =
					soft0Local.maximum(soft1Local);
				PxArray<PxU32>& rigidEdgeCandidates = queryScratch
					? queryScratch->edgeBvhQueryCandidates
					: surface.edgeBvhQueryCandidates;
				const bool useEdgeBvh =
					avbdCollectRigidTriangleSurfaceEdgeBvhCandidates(
						surface, softMinimum, softMaximum, margin,
						rigidEdgeCandidates, queryScratch);
				const PxU32 rigidEdgeCount = useEdgeBvh
					? rigidEdgeCandidates.size() : surface.edges.size();
				if(discreteQueryStats)
				{
					const PxU32 triangleCandidateCount = useEdgeBvh
						? (queryScratch
							? queryScratch->triangleBvhQueryCandidates.size()
							: surface.triangleBvhQueryCandidates.size()) : 0;
					discreteQueryStats->recordEdgeQuery(
						useEdgeBvh, triangleCandidateCount, rigidEdgeCount);
				}
				for(PxU32 rigidEdgeEntry = 0;
					rigidEdgeEntry < rigidEdgeCount;
					++rigidEdgeEntry)
				{
					const PxU32 rigidEdgeIndex = useEdgeBvh
						? rigidEdgeCandidates[rigidEdgeEntry] :
							rigidEdgeEntry;
					const AvbdRigidTriangleSurfaceEdge& rigidEdge =
						surface.edges[rigidEdgeIndex];
					if(!rigidEdge.active ||
						rigidEdge.p0 >= surface.vertices.size() ||
						rigidEdge.p1 >= surface.vertices.size())
						continue;
					if(stats)
					{
						stats->rigidTriangleSurfaceEdgeCandidates++;
						stats->rigidTriangleSurfaceEdgeTests++;
					}
					const PxVec3& rigid0Local =
						surface.vertices[rigidEdge.p0].point;
					const PxVec3& rigid1Local =
						surface.vertices[rigidEdge.p1].point;
					const PxVec3 rigidMinimum =
						rigid0Local.minimum(rigid1Local) -
							PxVec3(margin);
					const PxVec3 rigidMaximum =
						rigid0Local.maximum(rigid1Local) +
							PxVec3(margin);
					if(softMinimum.x > rigidMaximum.x ||
						softMaximum.x < rigidMinimum.x ||
						softMinimum.y > rigidMaximum.y ||
						softMaximum.y < rigidMinimum.y ||
						softMinimum.z > rigidMaximum.z ||
						softMaximum.z < rigidMinimum.z)
						continue;
					PxReal softWeight1 = 0.0f;
					PxReal rigidWeight1 = 0.0f;
					PxVec3 softClosestLocal;
					PxVec3 rigidClosestLocal;
					avbdClosestPointsOnSegments(
						soft0Local, soft1Local,
						rigid0Local, rigid1Local,
						softWeight1, rigidWeight1,
						softClosestLocal, rigidClosestLocal);
					if(softWeight1 <= featureEpsilon ||
						softWeight1 >= 1.0f - featureEpsilon ||
						rigidWeight1 <= featureEpsilon ||
						rigidWeight1 >= 1.0f - featureEpsilon)
						continue;
					const PxVec3 deltaLocal =
						softClosestLocal - rigidClosestLocal;
					const PxReal distance =
						deltaLocal.magnitude();
					if(!PxIsFinite(distance) ||
						distance >= margin ||
						deltaLocal.dot(rigidEdge.outward) <
							-1.0e-5f)
						continue;
					PxVec3 normalLocal =
						distance > distanceEpsilon
							? deltaLocal * (1.0f / distance)
							: rigidEdge.outward;
					if(!normalLocal.isFinite() ||
						normalLocal.magnitudeSquared() <=
							1.0e-12f)
						continue;

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, surface.primitiveKey,
						avbdGetRigidSoftFeatureKey(
							0x54534545u,
							softEdge.p0, softEdge.p1,
							0u, rigidEdgeIndex));
					geometry.particleIdx =
						particles[softEdge.p0].invMass > 0.0f
							? softEdge.p0 : softEdge.p1;
					geometry.queryParticleIndices[0] =
						softEdge.p0;
					geometry.queryParticleIndices[1] =
						softEdge.p1;
					geometry.queryWeights[0] =
						1.0f - softWeight1;
					geometry.queryWeights[1] =
						softWeight1;
					geometry.normal =
						surface.rotation.rotate(normalLocal).
							getNormalized();
					geometry.projNormal = geometry.normal;
					geometry.depth = margin - distance;
					geometry.margin = margin;
					avbdConfigureRigidTriangleSurfaceTarget(
						geometry, surface, surfaceIndex,
						rigidClosestLocal);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							rigidEdge.friction,
							rigidEdge.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1e5f, 1e6f,
						particles, contacts);
				}
			}

			PxU32 softTriangleBegin = 0;
			PxU32 softTriangleEnd = 0;
			const PxU32 softTriangleCount =
				body.compiled.surfaceTriangles.size() / 3;
			if(!workItem)
				softTriangleEnd = softTriangleCount;
			else if(workItem->family == WorkItem::eSOFT_TRIANGLE)
			{
				softTriangleBegin = PxMin(workItem->primitiveBegin,
					softTriangleCount);
				softTriangleEnd = PxMin(
					PxMax(workItem->primitiveEnd, softTriangleBegin),
					softTriangleCount);
			}
			for(PxU32 softTriangleIndex = softTriangleBegin;
				softTriangleIndex < softTriangleEnd;
				++softTriangleIndex)
			{
				const PxU32 triangleOffset = softTriangleIndex * 3;
				const PxU32 v0 =
					body.compiled.surfaceTriangles[
						triangleOffset];
				const PxU32 v1 =
					body.compiled.surfaceTriangles[
						triangleOffset + 1];
				const PxU32 v2 =
					body.compiled.surfaceTriangles[
						triangleOffset + 2];
				if(v0 >= numParticles || v1 >= numParticles ||
					v2 >= numParticles ||
					(particles[v0].invMass <= 0.0f &&
					 particles[v1].invMass <= 0.0f &&
					 particles[v2].invMass <= 0.0f))
					continue;
				const PxVec3& p0 = particles[v0].position;
				const PxVec3& p1 = particles[v1].position;
				const PxVec3& p2 = particles[v2].position;
				const PxVec3 p0Local =
					inverseRotation.rotate(p0 - surface.center);
				const PxVec3 p1Local =
					inverseRotation.rotate(p1 - surface.center);
				const PxVec3 p2Local =
					inverseRotation.rotate(p2 - surface.center);
				const PxVec3 triangleMinimum =
					p0Local.minimum(p1Local).minimum(p2Local);
				const PxVec3 triangleMaximum =
					p0Local.maximum(p1Local).maximum(p2Local);
				PxArray<PxU32>& rigidVertexCandidates = queryScratch
					? queryScratch->vertexBvhQueryCandidates
					: surface.vertexBvhQueryCandidates;
				const bool useVertexBvh =
					avbdCollectRigidTriangleSurfaceVertexBvhCandidates(
						surface, triangleMinimum, triangleMaximum, margin,
						rigidVertexCandidates, queryScratch);
				const PxU32 rigidVertexCount = useVertexBvh
					? rigidVertexCandidates.size() :
						surface.vertices.size();
				if(discreteQueryStats)
				{
					const PxU32 triangleCandidateCount = useVertexBvh
						? (queryScratch
							? queryScratch->triangleBvhQueryCandidates.size()
							: surface.triangleBvhQueryCandidates.size()) : 0;
					discreteQueryStats->recordTriangleQuery(
						useVertexBvh, triangleCandidateCount, rigidVertexCount);
				}

				for(PxU32 rigidVertexEntry = 0;
					rigidVertexEntry < rigidVertexCount;
					++rigidVertexEntry)
				{
					const PxU32 rigidVertexIndex = useVertexBvh
						? rigidVertexCandidates[rigidVertexEntry] :
							rigidVertexEntry;
					const AvbdRigidTriangleSurfaceVertex& vertex =
						surface.vertices[rigidVertexIndex];
					if(!vertex.active ||
						vertex.point.x < triangleMinimum.x - margin ||
						vertex.point.x > triangleMaximum.x + margin ||
						vertex.point.y < triangleMinimum.y - margin ||
						vertex.point.y > triangleMaximum.y + margin ||
						vertex.point.z < triangleMinimum.z - margin ||
						vertex.point.z > triangleMaximum.z + margin)
						continue;
					if(stats)
					{
						stats->rigidTriangleSurfaceVertexCandidates++;
						stats->rigidTriangleSurfaceVertexTests++;
					}
					const PxVec3 rigidVertexWorld =
						surface.center +
							surface.rotation.rotate(vertex.point);
					const AvbdClosestPointResult closest =
						avbdClosestPointOnTriangleOGC(
							rigidVertexWorld, p0, p1, p2);
					if(closest.feature != AVBD_FEATURE_FACE ||
						!PxIsFinite(closest.distance) ||
						closest.distance >= margin)
						continue;
					const PxVec3 outwardWorld =
						surface.rotation.rotate(vertex.outward);
					const PxVec3 deltaWorld =
						closest.point - rigidVertexWorld;
					if(deltaWorld.dot(outwardWorld) < -1.0e-5f)
						continue;
					PxVec3 normalWorld =
						closest.distance > distanceEpsilon
							? deltaWorld *
								(1.0f / closest.distance)
							: outwardWorld;
					if(!normalWorld.isFinite() ||
						normalWorld.magnitudeSquared() <=
							1.0e-12f)
						continue;

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, surface.primitiveKey,
						avbdGetRigidSoftFeatureKey(
							0x54535646u,
							v0, v1, v2,
							rigidVertexIndex));
					geometry.particleIdx =
						particles[v0].invMass > 0.0f ? v0 :
							(particles[v1].invMass > 0.0f
								? v1 : v2);
					geometry.queryParticleIndices[0] = v0;
					geometry.queryParticleIndices[1] = v1;
					geometry.queryParticleIndices[2] = v2;
					geometry.queryWeights[0] =
						closest.barycentric.x;
					geometry.queryWeights[1] =
						closest.barycentric.y;
					geometry.queryWeights[2] =
						closest.barycentric.z;
					geometry.normal = normalWorld.getNormalized();
					geometry.projNormal = geometry.normal;
					geometry.depth =
						margin - closest.distance;
					geometry.margin = margin;
					avbdConfigureRigidTriangleSurfaceTarget(
						geometry, surface, surfaceIndex,
						vertex.point);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							vertex.friction,
							vertex.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1e5f, 1e6f,
						particles, contacts);
				}
			}
		}
	}
}

// P5.17c candidate leaf: consume a contiguous interval of the canonical
// feature plan. Each call owns its contact output and complete query scratch;
// parent code must stable-merge outputs by plan index. The serial suffixes use
// the same row filter internally, so the leaf cannot introduce a second
// feature predicate, traversal order or BVH ownership model.
inline void avbdDetectSoftRigidTriangleSurfaceOGCFeaturePlanRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidTriangleSurface* surfaces, PxU32 numSurfaces,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdRigidTriangleSurfaceFeaturePlan& plan,
	PxU32 planBegin, PxU32 planEnd,
	PxArray<AvbdSoftContact>& contacts,
	AvbdRigidTriangleSurfaceQueryScratch& queryScratch,
	PxReal margin = 0.05f,
	AvbdSoftCollisionStats* stats = NULL,
	AvbdRigidTriangleSurfaceFeaturePlanRangeTiming* timing = NULL,
	AvbdRigidTriangleSurfaceSweptOGCFeatureSubstageTiming*
		sweptSubstageTiming = NULL,
	AvbdRigidTriangleSurfaceForwardOwnerQueryStats*
		forwardOwnerQueryStats = NULL,
	AvbdRigidTriangleSurfaceForwardOwnerResultCache*
		forwardOwnerResultCache = NULL,
	AvbdRigidTriangleSurfaceDiscreteOGCQueryStats*
		discreteQueryStats = NULL,
	bool useDiscreteBodyLocalBoundsCull = false)
{
	PX_ASSERT(planBegin <= planEnd && planEnd <= plan.items.size());
	const PxU32 clampedBegin = PxMin(planBegin, plan.items.size());
	const PxU32 clampedEnd = PxMin(PxMax(planEnd, clampedBegin),
		plan.items.size());
	for(PxU32 planIndex = clampedBegin;
		planIndex < clampedEnd; ++planIndex)
	{
		const AvbdRigidTriangleSurfaceFeatureWorkItem& workItem =
			plan.items[planIndex];
		PX_ASSERT(workItem.bodyIndex < numSoftBodies);
		PX_ASSERT(workItem.surfaceIndex < numSurfaces);
		if(workItem.bodyIndex >= numSoftBodies ||
			workItem.surfaceIndex >= numSurfaces)
			continue;
		const PxU64 workItemStartNanos = timing ?
			PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u : 0;
		if(workItem.phase ==
			AvbdRigidTriangleSurfaceFeatureWorkItem::eSWEPT)
		{
			avbdDetectSoftRigidTriangleSurfaceSweptOGCFeatures(
				particles, numParticles, surfaces, numSurfaces,
				softBodies, numSoftBodies, contacts, margin, stats,
				&queryScratch, NULL, &workItem, sweptSubstageTiming,
				forwardOwnerQueryStats, forwardOwnerResultCache);
		}
		else
		{
			avbdDetectSoftRigidTriangleSurfaceOGCFeatures(
				particles, numParticles, surfaces, numSurfaces,
				softBodies, numSoftBodies, contacts, margin, stats,
				&queryScratch, &workItem, discreteQueryStats,
				useDiscreteBodyLocalBoundsCull);
		}
		if(timing)
			timing->record(workItem,
				PxTime::getCurrentTimeInTensOfNanoSeconds() * 10u -
				workItemStartNanos);
	}
}

inline void avbdDetectSoftRigidOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidBox* boxes, PxU32 numBoxes,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f)
{
	const PxReal featureEpsilon = 1e-4f;
	const PxReal distanceEpsilon = 1e-8f;

	auto configureRigidTarget = [](
		AvbdSoftContactGeometry& geometry,
		const AvbdRigidBox& box, PxU32 boxIndex,
		const PxVec3& surfaceLocal)
	{
		geometry.targetKind = box.targetKind;
		geometry.velocityOwner =
			box.targetKind == AvbdSoftContactTargetKind::eKINEMATIC_RIGID
				? AvbdVelocityObjectiveOwner::ComponentFinalize
				: box.targetKind ==
					AvbdSoftContactTargetKind::eRIGID_BODY
					? AvbdVelocityObjectiveOwner::ManifoldFinalize
					: AvbdVelocityObjectiveOwner::PositionAL;
		geometry.targetIndex =
			box.targetKind == AvbdSoftContactTargetKind::eRIGID_BODY
				? box.targetIndex : boxIndex;
		geometry.surfacePoint =
			box.center + box.rotation.rotate(surfaceLocal);
		geometry.hasRigidBoxSdf = true;
		geometry.rigidBoxHalfExtent = box.halfExtent;
		geometry.rigidBoxPose =
			box.targetKind == AvbdSoftContactTargetKind::eRIGID_BODY
				? box.shapeToRigidBody
				: PxTransform(box.center, box.rotation);
		geometry.kinematicSurfacePointPrevious =
			box.targetKind == AvbdSoftContactTargetKind::eKINEMATIC_RIGID
				? box.previousCenter +
					box.previousRotation.rotate(surfaceLocal)
				: geometry.surfacePoint;
		if(box.targetKind == AvbdSoftContactTargetKind::eRIGID_BODY)
			geometry.rigidLocalPoint =
				box.shapeToRigidBody.transform(surfaceLocal);
	};

	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		PxVec3 bodyMinimum(PX_MAX_F32);
		PxVec3 bodyMaximum(-PX_MAX_F32);
		for(PxU32 localParticle = 0;
			localParticle < body.compiled.particleCount;
			localParticle++)
		{
			const PxVec3& position =
				particles[
					body.compiled.particleStart + localParticle].
					position;
			bodyMinimum = bodyMinimum.minimum(position);
			bodyMaximum = bodyMaximum.maximum(position);
		}
		for(PxU32 boxIndex = 0; boxIndex < numBoxes; boxIndex++)
		{
			const AvbdRigidBox& box = boxes[boxIndex];
			if(box.halfExtent.x <= 0.0f &&
				box.halfExtent.y <= 0.0f &&
				box.halfExtent.z <= 0.0f)
				continue;
			const PxReal boxRadius =
				box.halfExtent.magnitude() + margin;
			const PxVec3 broadphaseExtent(boxRadius);
			if(bodyMinimum.x > box.center.x + broadphaseExtent.x ||
				bodyMaximum.x < box.center.x - broadphaseExtent.x ||
				bodyMinimum.y > box.center.y + broadphaseExtent.y ||
				bodyMaximum.y < box.center.y - broadphaseExtent.y ||
				bodyMinimum.z > box.center.z + broadphaseExtent.z ||
				bodyMaximum.z < box.center.z - broadphaseExtent.z)
				continue;
			const PxQuat inverseRotation = box.rotation.getConjugate();

			// OGC edge-edge blocks: closest points must lie in the interiors
			// of both edges. Endpoint cases are owned by the adjacent
			// vertex/face blocks and are intentionally excluded.
			for(PxU32 softEdgeIndex = 0;
				softEdgeIndex < body.compiled.surfaceEdges.size();
				softEdgeIndex++)
			{
				const AvbdEdgeInfo& softEdge =
					body.compiled.surfaceEdges[softEdgeIndex];
				if(softEdge.p0 >= numParticles ||
					softEdge.p1 >= numParticles ||
					(particles[softEdge.p0].invMass <= 0.0f &&
					 particles[softEdge.p1].invMass <= 0.0f))
					continue;
				const PxVec3 soft0Local = inverseRotation.rotate(
					particles[softEdge.p0].position - box.center);
				const PxVec3 soft1Local = inverseRotation.rotate(
					particles[softEdge.p1].position - box.center);
				const PxVec3 softMinimum =
					soft0Local.minimum(soft1Local);
				const PxVec3 softMaximum =
					soft0Local.maximum(soft1Local);
				const PxVec3 expandedHalfExtent =
					box.halfExtent + PxVec3(margin);
				if(softMinimum.x > expandedHalfExtent.x ||
					softMaximum.x < -expandedHalfExtent.x ||
					softMinimum.y > expandedHalfExtent.y ||
					softMaximum.y < -expandedHalfExtent.y ||
					softMinimum.z > expandedHalfExtent.z ||
					softMaximum.z < -expandedHalfExtent.z)
					continue;

				for(PxU32 rigidEdgeIndex = 0;
					rigidEdgeIndex < 12; rigidEdgeIndex++)
				{
					PxVec3 rigid0Local, rigid1Local, outwardLocal;
					avbdGetRigidBoxEdgeLocal(
						box.halfExtent, rigidEdgeIndex,
						rigid0Local, rigid1Local, outwardLocal);
					const PxVec3 rigidMinimum =
						rigid0Local.minimum(rigid1Local) -
						PxVec3(margin);
					const PxVec3 rigidMaximum =
						rigid0Local.maximum(rigid1Local) +
						PxVec3(margin);
					if(softMinimum.x > rigidMaximum.x ||
						softMaximum.x < rigidMinimum.x ||
						softMinimum.y > rigidMaximum.y ||
						softMaximum.y < rigidMinimum.y ||
						softMinimum.z > rigidMaximum.z ||
						softMaximum.z < rigidMinimum.z)
						continue;
					PxReal softWeight1 = 0.0f;
					PxReal rigidWeight1 = 0.0f;
					PxVec3 softClosestLocal, rigidClosestLocal;
					avbdClosestPointsOnSegments(
						soft0Local, soft1Local,
						rigid0Local, rigid1Local,
						softWeight1, rigidWeight1,
						softClosestLocal, rigidClosestLocal);
					if(softWeight1 <= featureEpsilon ||
						softWeight1 >= 1.0f - featureEpsilon ||
						rigidWeight1 <= featureEpsilon ||
						rigidWeight1 >= 1.0f - featureEpsilon)
						continue;

					PxVec3 deltaLocal =
						softClosestLocal - rigidClosestLocal;
					const PxReal distance = deltaLocal.magnitude();
					if(distance >= margin)
						continue;
					PxVec3 normalLocal = distance > distanceEpsilon
						? deltaLocal * (1.0f / distance)
						: outwardLocal;
					if(normalLocal.dot(outwardLocal) < 0.0f)
						normalLocal = -normalLocal;

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, box.primitiveKey,
						avbdGetRigidSoftFeatureKey(
							0x45444745u,
							softEdge.p0, softEdge.p1,
							0u, rigidEdgeIndex));
					// particleIdx remains the representative used by the
					// rigid/soft routing code.  Prefer a movable endpoint so
					// an edge incident to a pinned cloth vertex is not
					// mistaken for a wholly kinematic shell contact.
					geometry.particleIdx =
						particles[softEdge.p0].invMass > 0.0f
							? softEdge.p0 : softEdge.p1;
					geometry.queryParticleIndices[0] = softEdge.p0;
					geometry.queryParticleIndices[1] = softEdge.p1;
					geometry.queryWeights[0] = 1.0f - softWeight1;
					geometry.queryWeights[1] = softWeight1;
					geometry.normal =
						box.rotation.rotate(normalLocal).getNormalized();
					geometry.projNormal = geometry.normal;
					geometry.depth = margin - distance;
					geometry.margin = margin;
					configureRigidTarget(
						geometry, box, boxIndex, rigidClosestLocal);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							box.friction, box.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1e5f, 1e6f,
						particles, contacts);
				}
			}

			// Reverse vertex-facet blocks: a rigid box vertex can approach or
			// cross the interior of a cloth triangle while every cloth vertex
			// remains outside the box SDF. Store the closest cloth point as a
			// barycentric query so its response is distributed to the face.
			for(PxU32 triangleOffset = 0;
				triangleOffset + 2 <
					body.compiled.surfaceTriangles.size();
				triangleOffset += 3)
			{
				const PxU32 v0 =
					body.compiled.surfaceTriangles[triangleOffset];
				const PxU32 v1 =
					body.compiled.surfaceTriangles[triangleOffset + 1];
				const PxU32 v2 =
					body.compiled.surfaceTriangles[triangleOffset + 2];
				if(v0 >= numParticles || v1 >= numParticles ||
					v2 >= numParticles ||
					(particles[v0].invMass <= 0.0f &&
					 particles[v1].invMass <= 0.0f &&
					 particles[v2].invMass <= 0.0f))
					continue;
				const PxVec3& p0 = particles[v0].position;
				const PxVec3& p1 = particles[v1].position;
				const PxVec3& p2 = particles[v2].position;
				const PxVec3 p0Local =
					inverseRotation.rotate(p0 - box.center);
				const PxVec3 p1Local =
					inverseRotation.rotate(p1 - box.center);
				const PxVec3 p2Local =
					inverseRotation.rotate(p2 - box.center);
				const PxVec3 triangleMinimum =
					p0Local.minimum(p1Local).minimum(p2Local) -
					PxVec3(margin);
				const PxVec3 triangleMaximum =
					p0Local.maximum(p1Local).maximum(p2Local) +
					PxVec3(margin);

				for(PxU32 rigidVertexIndex = 0;
					rigidVertexIndex < 8; rigidVertexIndex++)
				{
					const PxVec3 rigidVertexLocal =
						avbdGetRigidBoxVertexLocal(
							box.halfExtent, rigidVertexIndex);
					if(rigidVertexLocal.x < triangleMinimum.x ||
						rigidVertexLocal.x > triangleMaximum.x ||
						rigidVertexLocal.y < triangleMinimum.y ||
						rigidVertexLocal.y > triangleMaximum.y ||
						rigidVertexLocal.z < triangleMinimum.z ||
						rigidVertexLocal.z > triangleMaximum.z)
						continue;
					const PxVec3 rigidVertexWorld =
						box.center +
						box.rotation.rotate(rigidVertexLocal);
					const AvbdClosestPointResult closest =
						avbdClosestPointOnTriangleOGC(
							rigidVertexWorld, p0, p1, p2);
					if(closest.feature != AVBD_FEATURE_FACE ||
						closest.distance >= margin)
						continue;

					const PxVec3 outwardLocal =
						rigidVertexLocal.getNormalized();
					PxVec3 normalWorld;
					if(closest.distance > distanceEpsilon)
						normalWorld =
							(closest.point - rigidVertexWorld) *
							(1.0f / closest.distance);
					else
						normalWorld =
							box.rotation.rotate(outwardLocal);
					const PxVec3 outwardWorld =
						box.rotation.rotate(outwardLocal);
					if(normalWorld.dot(outwardWorld) < 0.0f)
						normalWorld = -normalWorld;

					AvbdSoftContactGeometry geometry;
					geometry.source = AvbdSoftContactSource(
						AvbdSoftContactSource::eRIGID_SDF,
						PX_MAX_U32, box.primitiveKey,
						avbdGetRigidSoftFeatureKey(
							0x56464143u, v0, v1, v2,
							rigidVertexIndex));
					// As above, keep the representative on a movable query
					// vertex whenever the triangle is not fully prescribed.
					geometry.particleIdx =
						particles[v0].invMass > 0.0f ? v0 :
						(particles[v1].invMass > 0.0f ? v1 : v2);
					geometry.queryParticleIndices[0] = v0;
					geometry.queryParticleIndices[1] = v1;
					geometry.queryParticleIndices[2] = v2;
					geometry.queryWeights[0] =
						closest.barycentric.x;
					geometry.queryWeights[1] =
						closest.barycentric.y;
					geometry.queryWeights[2] =
						closest.barycentric.z;
					geometry.normal = normalWorld.getNormalized();
					geometry.projNormal = geometry.normal;
					geometry.depth = margin - closest.distance;
					geometry.margin = margin;
					configureRigidTarget(
						geometry, box, boxIndex,
						rigidVertexLocal);
					geometry.friction =
						avbdCombineDeformableRigidFriction(
							body.material.dynamicFriction,
							box.friction, box.frictionCombineMode);
					avbdBuildSoftContactTangents(geometry);
					avbdAppendPreparedSoftContact(
						geometry, 1e5f, 1e6f,
						particles, contacts);
				}

				// Triangle/box core block.  The vertex-SDF, edge-edge and
				// reverse vertex-face blocks above deliberately own their normal
				// cases.  They still leave one discrete blind spot: a soft
				// collision triangle can pass through an OBB while all three soft
				// vertices and all eight OBB vertices remain outside the opposing
				// primitive.  Clip the triangle against the current box and emit
				// exactly one barycentric face row for a genuinely interior patch.
				// This is current-pose OGC/DCD, not a swept test.
				// Do not limit the clipper to the historical all-vertices-outside
				// blind spot.  A triangle with one vertex inside is still a complete
				// triangle/OBB core overlap, and a single vertex SDF row cannot prove
				// that the rest of its face exits the box in this same DCD step.
				{
					// The terminal verifier accepts up to 1 mm of numerical overlap.
					// Build the core witness against the same eroded OBB so a triangle
					// which merely touches the exterior does not create a repair row,
					// while every visible core penetration has a robust interior
					// barycentric witness rather than a boundary-only centroid.
					const PxReal coreInset = PxMin(1.0e-3f,
						0.25f * PxMin(box.halfExtent.x,
							PxMin(box.halfExtent.y, box.halfExtent.z)));
					const PxVec3 coreHalfExtent = box.halfExtent -
						PxVec3(PxMax(coreInset, 0.0f));
					if(coreHalfExtent.x <= 0.0f || coreHalfExtent.y <= 0.0f ||
						coreHalfExtent.z <= 0.0f)
						continue;
					struct TriangleBoxClipVertex
					{
						PxVec3 point;
						PxVec3 barycentric;
					};
					TriangleBoxClipVertex input[16] =
					{
						{p0Local, PxVec3(1.0f, 0.0f, 0.0f)},
						{p1Local, PxVec3(0.0f, 1.0f, 0.0f)},
						{p2Local, PxVec3(0.0f, 0.0f, 1.0f)}
					};
					TriangleBoxClipVertex output[16];
					PxU32 inputCount = 3;
					bool validClip = true;
					for(PxU32 axis = 0; axis < 3 && validClip;
						++axis)
					{
						for(PxU32 side = 0; side < 2 && validClip;
							++side)
						{
							const PxReal bound =
								side == 0 ? coreHalfExtent[axis] :
									-coreHalfExtent[axis];
							const bool upperBound = side == 0;
							PxU32 outputCount = 0;
							TriangleBoxClipVertex previous =
								input[inputCount - 1];
							bool previousInside = upperBound ?
								previous.point[axis] <= bound :
								previous.point[axis] >= bound;
							for(PxU32 vertex = 0; vertex < inputCount; ++vertex)
							{
								const TriangleBoxClipVertex current = input[vertex];
								const bool currentInside = upperBound ?
									current.point[axis] <= bound :
									current.point[axis] >= bound;
								if(currentInside != previousInside)
								{
									const PxReal denominator =
										current.point[axis] - previous.point[axis];
									if(PxAbs(denominator) <= 1.0e-12f ||
										outputCount >= 16)
									{
										validClip = false;
										break;
									}
									const PxReal t = PxClamp(
										(bound - previous.point[axis]) / denominator,
										0.0f, 1.0f);
									output[outputCount].point =
										previous.point +
										(current.point - previous.point) * t;
									output[outputCount].barycentric =
										previous.barycentric +
										(current.barycentric - previous.barycentric) * t;
									++outputCount;
								}
								if(currentInside)
								{
									if(outputCount >= 16)
									{
										validClip = false;
										break;
									}
									output[outputCount++] = current;
								}
								previous = current;
								previousInside = currentInside;
							}
							if(outputCount == 0)
							{
								validClip = false;
								break;
							}
							inputCount = outputCount;
							for(PxU32 vertex = 0; vertex < inputCount; ++vertex)
								input[vertex] = output[vertex];
						}
					}
					if(validClip && inputCount >= 3)
					{
						PxVec3 clippedPoint(0.0f);
						PxVec3 clippedBarycentric(0.0f);
						for(PxU32 vertex = 0; vertex < inputCount; ++vertex)
						{
							clippedPoint += input[vertex].point;
							clippedBarycentric += input[vertex].barycentric;
						}
						const PxReal reciprocalCount = 1.0f / inputCount;
						clippedPoint *= reciprocalCount;
						clippedBarycentric *= reciprocalCount;
						const PxReal barycentricSum =
							clippedBarycentric.x + clippedBarycentric.y +
							clippedBarycentric.z;
						if(PxIsFinite(barycentricSum) &&
							PxAbs(barycentricSum) > 1.0e-6f)
							clippedBarycentric *= 1.0f / barycentricSum;

						const PxVec3 coreQ(
							PxAbs(clippedPoint.x) - box.halfExtent.x,
							PxAbs(clippedPoint.y) - box.halfExtent.y,
							PxAbs(clippedPoint.z) - box.halfExtent.z);
						const PxReal coreSdf =
							PxMax(coreQ.x, PxMax(coreQ.y, coreQ.z));
						if(clippedPoint.isFinite() &&
							clippedBarycentric.isFinite() &&
							PxIsFinite(coreSdf) && coreSdf < -1.0e-5f)
						{
							// The centroid is the right compact AL query, but it is
							// not a sufficient escape witness for a triangle that
							// crosses the box.  Pick the shortest whole-triangle
							// translation through one OBB face. Once all three source
							// vertices lie beyond that face, their convex hull (the
							// complete triangle) is rigorously separated from the box.
							// Keep this certificate separate from the centroid normal so
							// ordinary Position-AL rows preserve their existing semantics.
							const PxVec3 triangleMin =
								p0Local.minimum(p1Local).minimum(p2Local);
							const PxVec3 triangleMax =
								p0Local.maximum(p1Local).maximum(p2Local);
							const PxReal exitDistances[6] =
							{
								box.halfExtent.x - triangleMin.x,
								triangleMax.x + box.halfExtent.x,
								box.halfExtent.y - triangleMin.y,
								triangleMax.y + box.halfExtent.y,
								box.halfExtent.z - triangleMin.z,
								triangleMax.z + box.halfExtent.z
							};
							const PxVec3 exitNormals[6] =
							{
								PxVec3(1.0f, 0.0f, 0.0f),
								PxVec3(-1.0f, 0.0f, 0.0f),
								PxVec3(0.0f, 1.0f, 0.0f),
								PxVec3(0.0f, -1.0f, 0.0f),
								PxVec3(0.0f, 0.0f, 1.0f),
								PxVec3(0.0f, 0.0f, -1.0f)
							};
							PxReal coreExitDistance = PX_MAX_F32;
							PxVec3 coreExitNormalLocal(0.0f);
							for(PxU32 exitIndex = 0; exitIndex < 6;
								++exitIndex)
							{
								const PxReal candidate = exitDistances[exitIndex];
								if(PxIsFinite(candidate) && candidate >= 0.0f &&
									candidate < coreExitDistance)
								{
									coreExitDistance = candidate;
									coreExitNormalLocal = exitNormals[exitIndex];
								}
							}
							if(!PxIsFinite(coreExitDistance) ||
								!coreExitNormalLocal.isFinite())
								continue;
							PxVec3 normalLocal;
							if(coreQ.x > coreQ.y && coreQ.x > coreQ.z)
								normalLocal = PxVec3(
									clippedPoint.x >= 0.0f ? 1.0f : -1.0f,
									0.0f, 0.0f);
							else if(coreQ.y > coreQ.z)
								normalLocal = PxVec3(0.0f,
									clippedPoint.y >= 0.0f ? 1.0f : -1.0f,
									0.0f);
							else
								normalLocal = PxVec3(0.0f, 0.0f,
									clippedPoint.z >= 0.0f ? 1.0f : -1.0f);
							const PxVec3 surfaceLocal =
								clippedPoint - normalLocal * coreSdf;
							AvbdSoftContactGeometry geometry;
							geometry.source = AvbdSoftContactSource(
								AvbdSoftContactSource::eRIGID_SDF,
								PX_MAX_U32, box.primitiveKey,
								avbdGetRigidSoftFeatureKey(
									0x54424958u, v0, v1, v2, 0u));
							geometry.particleIdx =
								particles[v0].invMass > 0.0f ? v0 :
								(particles[v1].invMass > 0.0f ? v1 : v2);
							geometry.queryParticleIndices[0] = v0;
							geometry.queryParticleIndices[1] = v1;
							geometry.queryParticleIndices[2] = v2;
							geometry.queryWeights[0] = clippedBarycentric.x;
							geometry.queryWeights[1] = clippedBarycentric.y;
							geometry.queryWeights[2] = clippedBarycentric.z;
							geometry.normal =
								box.rotation.rotate(normalLocal).getNormalized();
							geometry.projNormal = geometry.normal;
							geometry.depth = -coreSdf;
							geometry.margin = margin;
							geometry.rigidBoxTriangleCoreExitNormalLocal =
								coreExitNormalLocal;
							geometry.rigidBoxTriangleCoreExitDistance =
								coreExitDistance + PxMax(1.0e-5f, margin * 0.02f);
							geometry.hasRigidBoxTriangleCoreExit =
								geometry.rigidBoxTriangleCoreExitNormalLocal.isFinite() &&
								PxIsFinite(geometry.rigidBoxTriangleCoreExitDistance);
							geometry.rigidBoxTriangleCoreMinimumLocal = triangleMin;
							geometry.rigidBoxTriangleCoreMaximumLocal = triangleMax;
							geometry.rigidBoxTriangleCoreCollisionParticleIndices[0] = v0;
							geometry.rigidBoxTriangleCoreCollisionParticleIndices[1] = v1;
							geometry.rigidBoxTriangleCoreCollisionParticleIndices[2] = v2;
							configureRigidTarget(
								geometry, box, boxIndex, surfaceLocal);
							geometry.friction = avbdCombineDeformableRigidFriction(
								body.material.dynamicFriction, box.friction,
								box.frictionCombineMode);
							avbdBuildSoftContactTangents(geometry);
							avbdAppendPreparedSoftContact(
								geometry, 1e5f, 1e6f, particles, contacts);
						}
					}
				}
		}
	}
}
}

// =============================================================================
// PATH 3 (OGC): Simplified Soft-Soft Contact (Sec 3.9)
//
// Outward-only offset, pure quadratic energy, DCD for penetration.
// =============================================================================

// P5.2b makes the existing serial body-pair broadphase an explicit immutable
// input to later refit/query work.  The plan's lexicographic pair order and
// current-versus-swept choice are the canonical merge order for any future
// private candidate tasks.
inline void avbdBuildSoftSoftOGCDetectionPlan(
	const AvbdSoftParticle* particles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdOGCParams& params,
	AvbdSoftCollisionStats* stats,
	AvbdSoftContactWorkspace& workspace)
{
	const PxReal r = params.contactRadius;
	const bool hasPreparedBodyBounds =
		workspace.softBodyBoundsValid &&
		workspace.softBodyBounds.size() == numSoftBodies;

	// P3 Slice 5 planning boundary. This is intentionally serial: it produces
	// the same lexicographic overlapping-pair stream the legacy loop would
	// visit, while making pair-specific swept-mode ownership explicit.
	workspace.beginSoftPairDetectionPlan();
	for(PxU32 sA = 0; sA < numSoftBodies; sA++)
	{
		for(PxU32 sB = sA + 1; sB < numSoftBodies; sB++)
		{
			if(stats)
				stats->bodyPairs++;
			const AvbdSoftBody& bodyA = softBodies[sA];
			const AvbdSoftBody& bodyB = softBodies[sB];
			const bool pairSpeculative =
				bodyA.compiled.speculativeCCDEnabled ||
				bodyB.compiled.speculativeCCDEnabled;

			// AABB broadphase per body pair. A P3 prediction fan-in may provide
			// one current/swept result per body. The direct path retains the
			// identical legacy traversal, and no cache survives this detection.
			PxVec3 minA(PX_MAX_F32), maxA(-PX_MAX_F32);
			PxVec3 minB(PX_MAX_F32), maxB(-PX_MAX_F32);
			if(hasPreparedBodyBounds)
			{
				const AvbdSoftBodyBounds& boundsA =
					workspace.softBodyBounds[sA];
				const AvbdSoftBodyBounds& boundsB =
					workspace.softBodyBounds[sB];
				minA = pairSpeculative ? boundsA.sweptMinimum :
					boundsA.currentMinimum;
				maxA = pairSpeculative ? boundsA.sweptMaximum :
					boundsA.currentMaximum;
				minB = pairSpeculative ? boundsB.sweptMinimum :
					boundsB.currentMinimum;
				maxB = pairSpeculative ? boundsB.sweptMaximum :
					boundsB.currentMaximum;
			}
			else
			{
				for (PxU32 i = 0; i < bodyA.compiled.particleCount; i++) {
					const AvbdSoftParticle& particle =
						particles[bodyA.compiled.particleStart + i];
					const PxVec3& p = particle.position;
					minA = minA.minimum(p); maxA = maxA.maximum(p);
					if(pairSpeculative)
					{
						minA = minA.minimum(particle.initialPosition);
						maxA = maxA.maximum(particle.initialPosition);
					}
				}
				for (PxU32 i = 0; i < bodyB.compiled.particleCount; i++) {
					const AvbdSoftParticle& particle =
						particles[bodyB.compiled.particleStart + i];
					const PxVec3& p = particle.position;
					minB = minB.minimum(p); maxB = maxB.maximum(p);
					if(pairSpeculative)
					{
						minB = minB.minimum(particle.initialPosition);
						maxB = maxB.maximum(particle.initialPosition);
					}
				}
			}
			if (minA.x > maxB.x + r || maxA.x < minB.x - r ||
				minA.y > maxB.y + r || maxA.y < minB.y - r ||
				minA.z > maxB.z + r || maxA.z < minB.z - r)
				continue;
			if(stats)
				stats->overlappingBodyPairs++;
			AvbdSoftPairDetectionPlan plan;
			plan.bodyA = sA;
			plan.bodyB = sB;
			plan.swept = pairSpeculative;
			plan.minimumA = minA;
			plan.maximumA = maxA;
			plan.minimumB = minB;
			plan.maximumB = maxB;
			workspace.appendSoftPairDetectionPlan(plan);
		}
	}
	if(avbdValidateRedetectionPhasePlan())
		PX_ASSERT(workspace.validateSoftPairDetectionPlan(numSoftBodies));
}

// P5.2c owns the shared body/mode refit barrier independently from the
// pair-query loop. The workspace epoch spans are still parent-owned; this
// seam merely makes every required write/read dependency explicit.
inline bool avbdRefitSoftSoftOGCDetectionPlan(
	const AvbdSoftParticle* particles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdSoftCollisionStats* stats,
	AvbdSoftContactWorkspace& workspace)
{
	const bool useSurfaceTriangleBvh = avbdUseSurfaceTriangleBvh();
	if(!useSurfaceTriangleBvh)
		return false;
	workspace.beginSoftPairTriangleBvhEpoch(numSoftBodies);
	for(PxU32 planIndex = 0;
		planIndex < workspace.softPairDetectionPlan.size(); ++planIndex)
	{
		const AvbdSoftPairDetectionPlan& plan =
			workspace.softPairDetectionPlan[planIndex];
		workspace.requireSoftPairTriangleBvhBounds(
			plan.bodyA, plan.swept,
			softBodies[plan.bodyA].compiled.
				surfaceTriangleBvhNodes.size());
		workspace.requireSoftPairTriangleBvhBounds(
			plan.bodyB, plan.swept,
			softBodies[plan.bodyB].compiled.
				surfaceTriangleBvhNodes.size());
	}
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		for(PxU32 mode = 0; mode < 2; ++mode)
		{
			const bool swept = mode != 0;
			if(!workspace.isSoftPairTriangleBvhBoundsRequired(
					bodyIndex, swept))
				continue;
			body.compiled.refitSurfaceTriangleBvh(
				particles, swept,
				workspace.getSoftPairTriangleBvhBoundsForRefit(
					bodyIndex, swept));
			workspace.markSoftPairTriangleBvhBoundsRefit(
				bodyIndex, swept);
			if(stats)
				stats->surfaceTriangleBvhRefitNodes +=
					body.compiled.surfaceTriangleBvhNodes.size();
		}
	}
	return true;
}

// P5.9c's post-refit work unit. The parent has already frozen the canonical
// pair plan and refitted every required body/mode BVH span. This leaf reads
// only that immutable epoch and writes only its output stream, statistics and
// caller-owned pair-query scratch.
inline void avbdDetectSoftSoftOGCPlanRange(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftContactWorkspace& refitWorkspace,
	AvbdSoftContactWorkspace* serialScratchWorkspace,
	AvbdSoftSoftPairQueryScratch& queryScratch,
	bool useSurfaceTriangleBvh,
	PxU32 planBegin, PxU32 planEnd,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params,
	AvbdSoftCollisionStats* stats = NULL)
{
	PX_UNUSED(numParticles);
	PX_UNUSED(numSoftBodies);
	auto reserveSoftPairQueryScratch = [&](
		PxU32 edgeCountA, PxU32 edgeCountB,
		PxU32 triangleCandidateCapacity)
	{
		if(serialScratchWorkspace)
			serialScratchWorkspace->reserveSoftPairSweep(
				edgeCountA, edgeCountB, triangleCandidateCapacity);
		else
			queryScratch.reserve(
				edgeCountA, edgeCountB, triangleCandidateCapacity);
	};
	const PxReal r = params.contactRadius;
	const PxU32 clampedPlanEnd = PxMin(
		planEnd, refitWorkspace.softPairDetectionPlan.size());
	PX_ASSERT(planBegin <= clampedPlanEnd);
	PxArray<AvbdSurfaceBvhNodeBounds> emptySoftPairTriangleBvhBounds;

	for(PxU32 planIndex = planBegin;
		planIndex < clampedPlanEnd; ++planIndex)
	{
		const AvbdSoftPairDetectionPlan& plan =
			refitWorkspace.softPairDetectionPlan[planIndex];
		PX_ASSERT(plan.bodyA < numSoftBodies &&
			plan.bodyB < numSoftBodies && plan.bodyA < plan.bodyB);
		const PxU32 sA = plan.bodyA;
		const PxU32 sB = plan.bodyB;
		const AvbdSoftBody& bodyA = softBodies[sA];
		const AvbdSoftBody& bodyB = softBodies[sB];
		const bool pairSpeculative = plan.swept;
		const PxVec3& minA = plan.minimumA;
		const PxVec3& maxA = plan.maximumA;
		const PxVec3& minB = plan.minimumB;
		const PxVec3& maxB = plan.maximumB;
		const PxReal pairFriction = 0.5f * (
			PxMax(bodyA.material.dynamicFriction, 0.0f) +
			PxMax(bodyB.material.dynamicFriction, 0.0f));
		const PxArray<AvbdSurfaceBvhNodeBounds>& triangleBvhBoundsA =
			useSurfaceTriangleBvh
				? refitWorkspace.getSoftPairTriangleBvhBounds(
					sA, pairSpeculative)
				: emptySoftPairTriangleBvhBounds;
		const PxArray<AvbdSurfaceBvhNodeBounds>& triangleBvhBoundsB =
			useSurfaceTriangleBvh
				? refitWorkspace.getSoftPairTriangleBvhBounds(
					sB, pairSpeculative)
				: emptySoftPairTriangleBvhBounds;
			reserveSoftPairQueryScratch(
				bodyA.compiled.surfaceEdges.size(),
				bodyB.compiled.surfaceEdges.size(),
				PxMax(bodyA.compiled.surfaceTriangles.size() / 3,
					bodyB.compiled.surfaceTriangles.size() / 3));

			// Lambda: test particles of testBody against surface of surfBody
			auto testParticlesVsSurface = [&](
				const AvbdSoftBody& testBody, const AvbdSoftBody& surfBody,
				PxU32 surfBodyIdx,
				const PxVec3& aabbLo, const PxVec3& aabbHi,
				const PxArray<AvbdSurfaceBvhNodeBounds>&
					surfaceBvhBounds)
			{
				const bool targetIsShell =
					surfBody.compiled.tetrahedra.empty();
				for(PxU32 queryVertexIndex = 0;
					queryVertexIndex <
						testBody.compiled.surfaceVertices.size();
					queryVertexIndex++)
				{
					const PxU32 pi =
						testBody.compiled.surfaceVertices[
							queryVertexIndex];
					if(pi < testBody.compiled.particleStart ||
						pi - testBody.compiled.particleStart >=
							testBody.compiled.particleCount)
						continue;
					if (particles[pi].invMass <= 0.0f) continue;
					const PxVec3& pp = particles[pi].position;

					// Per-particle AABB cull
					const PxVec3 queryMinimum = pairSpeculative
						? pp.minimum(particles[pi].initialPosition) : pp;
					const PxVec3 queryMaximum = pairSpeculative
						? pp.maximum(particles[pi].initialPosition) : pp;
					if (queryMaximum.x < aabbLo.x - r ||
						queryMinimum.x > aabbHi.x + r ||
						queryMaximum.y < aabbLo.y - r ||
						queryMinimum.y > aabbHi.y + r ||
						queryMaximum.z < aabbLo.z - r ||
						queryMinimum.z > aabbHi.z + r)
						continue;
					if(stats)
						stats->particleSurfaceCandidates++;
					const bool useSurfaceTriangleBvhForBody =
						useSurfaceTriangleBvh &&
						!surfBody.compiled.surfaceTriangleBvhNodes.empty();
					PxArray<PxU32>& triangleCandidates =
						queryScratch.triangleCandidates;
					if(useSurfaceTriangleBvhForBody)
					{
						surfBody.compiled.collectSurfaceTriangleBvhCandidates(
							queryMinimum, queryMaximum, r,
							surfaceBvhBounds,
							triangleCandidates);
						if(stats)
							stats->surfaceTriangleBvhCandidateTriangles +=
								triangleCandidates.size();
					}
					const PxU32 candidateTriangleCount =
						useSurfaceTriangleBvhForBody
							? triangleCandidates.size()
							: surfBody.compiled.surfaceTriangles.size() / 3;

					// The discrete owner below cannot see a vertex that
					// crossed the complete target face during this step and
					// finished outside the contact shell.  Select the first
					// face entry over the same initial->predicted interval
					// used by rigid-soft speculative OGC.  Selecting only the
					// earliest face gives redetection a stable, unique owner.
					if(pairSpeculative)
					{
						PxReal bestEntryTime = PX_MAX_F32;
						PxU32 bestTriangleOffset = PX_MAX_U32;
						AvbdSweptTriangleEntry bestEntry;
						for(PxU32 candidateIndex = 0;
							candidateIndex < candidateTriangleCount;
							candidateIndex++)
						{
							const PxU32 ti = useSurfaceTriangleBvhForBody
								? triangleCandidates[candidateIndex] * 3
								: candidateIndex * 3;
							const PxU32 source0 =
								surfBody.compiled.surfaceTriangles[ti];
							const PxU32 source1 =
								surfBody.compiled.surfaceTriangles[ti + 1];
							const PxU32 source2 =
								surfBody.compiled.surfaceTriangles[ti + 2];
							const PxVec3 targetMinimum =
								particles[source0].initialPosition.minimum(
									particles[source1].initialPosition).
								minimum(
									particles[source2].initialPosition).
								minimum(particles[source0].position).
								minimum(particles[source1].position).
								minimum(particles[source2].position);
							const PxVec3 targetMaximum =
								particles[source0].initialPosition.maximum(
									particles[source1].initialPosition).
								maximum(
									particles[source2].initialPosition).
								maximum(particles[source0].position).
								maximum(particles[source1].position).
								maximum(particles[source2].position);
							if(queryMaximum.x < targetMinimum.x - r ||
								queryMinimum.x > targetMaximum.x + r ||
								queryMaximum.y < targetMinimum.y - r ||
								queryMinimum.y > targetMaximum.y + r ||
								queryMaximum.z < targetMinimum.z - r ||
								queryMinimum.z > targetMaximum.z + r)
								continue;
							if(stats)
								stats->closestTriangleTests++;
							AvbdSweptTriangleEntry entry;
							if(avbdRotatingPointEnterExpandedDeformingTriangleFace(
									PxVec3(0.0f),
									particles[pi].initialPosition, pp,
									PxQuat(PxIdentity), PxQuat(PxIdentity),
									particles[source0].initialPosition,
									particles[source1].initialPosition,
									particles[source2].initialPosition,
									particles[source0].position,
									particles[source1].position,
									particles[source2].position,
									r, entry) &&
								entry.entryTime < bestEntryTime)
							{
								bestEntryTime = entry.entryTime;
								bestTriangleOffset = ti;
								bestEntry = entry;
							}
						}
						if(bestTriangleOffset != PX_MAX_U32)
						{
							const PxU32 source0 =
								surfBody.compiled.surfaceTriangles[
									bestTriangleOffset];
							const PxU32 source1 =
								surfBody.compiled.surfaceTriangles[
									bestTriangleOffset + 1];
							const PxU32 source2 =
								surfBody.compiled.surfaceTriangles[
									bestTriangleOffset + 2];
							PxVec3 contactNormal = -bestEntry.normal;
							const AvbdClosestPointResult initialClosest =
								avbdClosestPointOnTriangleOGC(
									particles[pi].initialPosition,
									particles[source0].initialPosition,
									particles[source1].initialPosition,
									particles[source2].initialPosition);
							if(contactNormal.dot(initialClosest.normal) < 0.0f)
								contactNormal = -contactNormal;
							const PxVec3 entry0 =
								particles[source0].initialPosition +
								(particles[source0].position -
								 particles[source0].initialPosition) *
									bestEntry.entryTime;
							const PxVec3 entry1 =
								particles[source1].initialPosition +
								(particles[source1].position -
								 particles[source1].initialPosition) *
									bestEntry.entryTime;
							const PxVec3 entry2 =
								particles[source2].initialPosition +
								(particles[source2].position -
								 particles[source2].initialPosition) *
									bestEntry.entryTime;
							AvbdSoftContactGeometry geometry;
							geometry.source = AvbdSoftContactSource(
								AvbdSoftContactSource::eSOFT_SURFACE,
								surfBodyIdx,
								avbdSoftTrianglePrimitiveKey(
									source0, source1, source2),
								avbdSoftTriangleFeatureKey(
									source0, source1, source2,
									AVBD_FEATURE_FACE, 0));
							geometry.particleIdx = pi;
							geometry.targetKind =
								AvbdSoftContactTargetKind::
									eDEFORMABLE_SURFACE;
							geometry.velocityOwner =
								AvbdVelocityObjectiveOwner::PositionAL;
							geometry.targetIndex = surfBodyIdx;
							const PxU32 triangleIndex =
								bestTriangleOffset / 3;
							geometry.targetSourceElementIndex =
								triangleIndex <
									surfBody.compiled.
										surfaceTriangleElementIndices.size()
								? surfBody.compiled.
									surfaceTriangleElementIndices[
										triangleIndex]
								: PX_MAX_U32;
							geometry.normal = contactNormal;
							geometry.projNormal = contactNormal;
							geometry.depth = 0.0f;
							geometry.margin = r;
							geometry.surfacePoint =
								entry0 * bestEntry.barycentric.x +
								entry1 * bestEntry.barycentric.y +
								entry2 * bestEntry.barycentric.z;
							geometry.surfaceParticleIndices[0] = source0;
							geometry.surfaceParticleIndices[1] = source1;
							geometry.surfaceParticleIndices[2] = source2;
							geometry.surfaceWeights[0] =
								bestEntry.barycentric.x;
							geometry.surfaceWeights[1] =
								bestEntry.barycentric.y;
							geometry.surfaceWeights[2] =
								bestEntry.barycentric.z;
							geometry.friction = pairFriction;
							avbdBuildSoftContactTangents(geometry);
							avbdAppendPreparedSoftContact(
								geometry,
								params.contactStiffness,
								params.contactStiffness * 10.0f,
								particles, contacts);
							continue;
						}
					}

					// DCD: check if particle is inside the other body
					const bool isInside = !targetIsShell &&
						avbdIsPointInsideTetMesh(
							pp, surfBody.compiled.surfaceTriangles,
							particles, stats);
					if (isInside)
					{
						// Find closest surface triangle for direction
						PxReal minDist = PX_MAX_F32;
						PxVec3 bestNormal(0.0f, 1.0f, 0.0f);
						PxVec3 bestClosest(0.0f);
						PxVec3 bestBarycentric(1.0f, 0.0f, 0.0f);
						PxU32 bestTriangle = PX_MAX_U32;
						AvbdClosestFeature bestFeature = AVBD_FEATURE_UNKNOWN;
						PxU32 bestFeatureIndex = 0;
						for (PxU32 ti = 0; ti + 2 < surfBody.compiled.surfaceTriangles.size(); ti += 3)
						{
							if(stats)
								stats->closestTriangleTests++;
							const PxVec3& va = particles[surfBody.compiled.surfaceTriangles[ti]].position;
							const PxVec3& vb = particles[surfBody.compiled.surfaceTriangles[ti+1]].position;
							const PxVec3& vc = particles[surfBody.compiled.surfaceTriangles[ti+2]].position;
							AvbdClosestPointResult cp = avbdClosestPointOnTriangleOGC(pp, va, vb, vc);
							if (cp.distance < minDist) {
								minDist = cp.distance;
								bestClosest = cp.point;
								bestBarycentric = cp.barycentric;
								bestTriangle = ti / 3;
								bestFeature = cp.feature;
								bestFeatureIndex = cp.featureIndex;
								PxVec3 faceN = (vb - va).cross(vc - va);
								PxReal fLen = faceN.magnitude();
								bestNormal = fLen > 1e-10f ? faceN * (1.0f / fLen) : cp.normal;
							}
						}

						PxReal depth = minDist + r;
						AvbdSoftContactGeometry geometry;
						const PxU32 source0 =
							surfBody.compiled.surfaceTriangles[bestTriangle * 3];
						const PxU32 source1 =
							surfBody.compiled.surfaceTriangles[bestTriangle * 3 + 1];
						const PxU32 source2 =
							surfBody.compiled.surfaceTriangles[bestTriangle * 3 + 2];
						geometry.source = AvbdSoftContactSource(
							AvbdSoftContactSource::eSOFT_SURFACE,
							surfBodyIdx,
							avbdSoftTrianglePrimitiveKey(
								source0, source1, source2),
							avbdSoftTriangleFeatureKey(
								source0, source1, source2,
								bestFeature, bestFeatureIndex));
						geometry.particleIdx  = pi;
						geometry.targetKind =
							AvbdSoftContactTargetKind::
								eDEFORMABLE_SURFACE;
						geometry.velocityOwner =
							AvbdVelocityObjectiveOwner::
								PositionAL;
						geometry.targetIndex = surfBodyIdx;
						geometry.targetSourceElementIndex =
							bestTriangle <
								surfBody.compiled.
									surfaceTriangleElementIndices.size()
							? surfBody.compiled.
								surfaceTriangleElementIndices[
									bestTriangle]
							: PX_MAX_U32;
						geometry.normal       = bestNormal;
						geometry.projNormal   = bestNormal;
						geometry.depth        = depth;
						geometry.margin       = r;
						geometry.surfacePoint = bestClosest;
						geometry.surfaceParticleIndices[0] = source0;
						geometry.surfaceParticleIndices[1] = source1;
						geometry.surfaceParticleIndices[2] = source2;
						geometry.surfaceWeights[0] = bestBarycentric.x;
						geometry.surfaceWeights[1] = bestBarycentric.y;
						geometry.surfaceWeights[2] = bestBarycentric.z;
						geometry.friction = pairFriction;
						avbdBuildSoftContactTangents(geometry);
						avbdAppendPreparedSoftContact(
							geometry,
							params.contactStiffness,
							params.contactStiffness * 10.0f,
							particles, contacts);
						continue;
					}

					auto appendOutwardContact = [&](
						PxU32 ti,
						const AvbdClosestPointResult& cp,
						const PxVec3& contactNormal)
					{
						const PxReal depth = r - cp.distance;
						AvbdSoftContactGeometry geometry;
						const PxU32 source0 =
							surfBody.compiled.surfaceTriangles[ti];
						const PxU32 source1 =
							surfBody.compiled.surfaceTriangles[ti + 1];
						const PxU32 source2 =
							surfBody.compiled.surfaceTriangles[ti + 2];
						geometry.source = AvbdSoftContactSource(
							AvbdSoftContactSource::eSOFT_SURFACE,
							surfBodyIdx,
							avbdSoftTrianglePrimitiveKey(
								source0, source1, source2),
							avbdSoftTriangleFeatureKey(
								source0, source1, source2,
								cp.feature, cp.featureIndex));
						geometry.particleIdx  = pi;
						geometry.targetKind =
							AvbdSoftContactTargetKind::
								eDEFORMABLE_SURFACE;
						geometry.velocityOwner =
							AvbdVelocityObjectiveOwner::
								PositionAL;
						geometry.targetIndex = surfBodyIdx;
						geometry.targetSourceElementIndex =
							ti / 3 <
								surfBody.compiled.
									surfaceTriangleElementIndices.size()
							? surfBody.compiled.
								surfaceTriangleElementIndices[
									ti / 3]
							: PX_MAX_U32;
						geometry.normal       = contactNormal;
						geometry.projNormal   = contactNormal;
						geometry.depth        = depth;
						geometry.margin       = r;
						geometry.surfacePoint = cp.point;
						geometry.surfaceParticleIndices[0] = source0;
						geometry.surfaceParticleIndices[1] = source1;
						geometry.surfaceParticleIndices[2] = source2;
						geometry.surfaceWeights[0] =
							cp.barycentric.x;
						geometry.surfaceWeights[1] =
							cp.barycentric.y;
						geometry.surfaceWeights[2] =
							cp.barycentric.z;
						geometry.friction = pairFriction;
						avbdBuildSoftContactTangents(geometry);
						avbdAppendPreparedSoftContact(
							geometry,
							params.contactStiffness,
							params.contactStiffness * 10.0f,
							particles, contacts);
					};

					// A shared vertex or edge can be represented by several
					// surface triangles. Compiling all of them duplicates
					// one physical signed-distance objective and injects
					// energy. Keep one deterministic closest feature per
					// particle/body pair, matching the penetration branch.
					PxReal bestDistance = r;
					PxU32 bestTriangle = PX_MAX_U32;
					AvbdClosestPointResult bestClosest = {};
					PxVec3 bestContactNormal(0.0f);

					// Not inside: OGC outward offset blocks on surface. Iterate
					// candidates in ascending compiled triangle order so a rare
					// exact closest-distance tie keeps the old traversal owner.
					for(PxU32 candidateIndex = 0;
						candidateIndex < candidateTriangleCount;
						candidateIndex++)
					{
						const PxU32 ti = useSurfaceTriangleBvhForBody
							? triangleCandidates[candidateIndex] * 3
							: candidateIndex * 3;
						if(stats)
							stats->closestTriangleTests++;
						const PxVec3& va = particles[surfBody.compiled.surfaceTriangles[ti]].position;
						const PxVec3& vb = particles[surfBody.compiled.surfaceTriangles[ti+1]].position;
						const PxVec3& vc = particles[surfBody.compiled.surfaceTriangles[ti+2]].position;

						AvbdClosestPointResult cp = avbdClosestPointOnTriangleOGC(pp, va, vb, vc);
						if (cp.distance >= r) continue;

						// Face normal for outward check
						PxVec3 faceN = (vb - va).cross(vc - va);
						PxReal fLen = faceN.magnitude();
						if (fLen < 1e-10f) continue;
						faceN = faceN * (1.0f / fLen);

						// Sec 3.9: outward-only offset
						PxVec3 toPoint = pp - cp.point;
						if (toPoint.dot(faceN) < 0.0f) continue;

						// OGC contact normal per feature type
						PxVec3 contactNormal = (cp.feature == AVBD_FEATURE_FACE) ? faceN : cp.normal;

						if(cp.distance < bestDistance)
						{
							bestDistance = cp.distance;
							bestTriangle = ti;
							bestClosest = cp;
							bestContactNormal = contactNormal;
						}
					}
					if(bestTriangle != PX_MAX_U32)
						appendOutwardContact(
							bestTriangle,
							bestClosest,
							bestContactNormal);
				}
			};

			// Test A particles vs B surface, then B particles vs A surface
			testParticlesVsSurface(
				bodyA, bodyB, sB, minB, maxB,
				triangleBvhBoundsB);
			testParticlesVsSurface(
				bodyB, bodyA, sA, minA, maxA,
				triangleBvhBoundsA);

			// Vertex-face features alone do not own a crossing between two
			// edge interiors.  Compile one canonical A-edge/B-edge row for
			// that missing OGC feature, with the swept owner taking
			// precedence over the end-of-step discrete owner.
			auto buildEdgeBounds =
				[&](const AvbdSoftBody& body,
					PxArray<AvbdSoftPairEdgeBounds>& bounds)
			{
				bounds.clear();
				for(PxU32 edgeIndex = 0;
					edgeIndex < body.compiled.surfaceEdges.size();
					edgeIndex++)
				{
					const AvbdEdgeInfo& edge =
						body.compiled.surfaceEdges[edgeIndex];
					if(!edge.collisionFeature)
						continue;
					if(edge.p0 >= numParticles ||
						edge.p1 >= numParticles)
						continue;
					AvbdSoftPairEdgeBounds edgeBounds;
					edgeBounds.edgeIndex = edgeIndex;
					edgeBounds.adjacentNormal0 = PxVec3(0.0f);
					edgeBounds.adjacentNormal1 = PxVec3(0.0f);
					edgeBounds.hasExteriorNormalCone = false;
					if(edge.adjacentSurfaceFace0 != PX_MAX_U32 &&
						edge.adjacentSurfaceFace1 != PX_MAX_U32)
					{
						const PxU32 face0 =
							edge.adjacentSurfaceFace0 * 3;
						const PxU32 face1 =
							edge.adjacentSurfaceFace1 * 3;
						if(face0 + 2 <
								body.compiled.surfaceTriangles.size() &&
							face1 + 2 <
								body.compiled.surfaceTriangles.size())
						{
							const PxU32 f00 =
								body.compiled.surfaceTriangles[face0];
							const PxU32 f01 =
								body.compiled.surfaceTriangles[face0 + 1];
							const PxU32 f02 =
								body.compiled.surfaceTriangles[face0 + 2];
							const PxU32 f10 =
								body.compiled.surfaceTriangles[face1];
							const PxU32 f11 =
								body.compiled.surfaceTriangles[face1 + 1];
							const PxU32 f12 =
								body.compiled.surfaceTriangles[face1 + 2];
							if(f00 < numParticles && f01 < numParticles &&
								f02 < numParticles && f10 < numParticles &&
								f11 < numParticles && f12 < numParticles)
							{
								edgeBounds.adjacentNormal0 =
									(particles[f01].position -
									 particles[f00].position).cross(
										particles[f02].position -
										particles[f00].position);
								edgeBounds.adjacentNormal1 =
									(particles[f11].position -
									 particles[f10].position).cross(
										particles[f12].position -
										particles[f10].position);
								edgeBounds.hasExteriorNormalCone =
									edgeBounds.adjacentNormal0.
										magnitudeSquared() > 1.0e-12f &&
									edgeBounds.adjacentNormal1.
										magnitudeSquared() > 1.0e-12f;
							}
						}
					}
					edgeBounds.minimum =
						particles[edge.p0].position.minimum(
							particles[edge.p1].position);
					edgeBounds.maximum =
						particles[edge.p0].position.maximum(
							particles[edge.p1].position);
					if(pairSpeculative)
					{
						edgeBounds.minimum =
							edgeBounds.minimum.minimum(
								particles[edge.p0].initialPosition).
							minimum(
								particles[edge.p1].initialPosition);
						edgeBounds.maximum =
							edgeBounds.maximum.maximum(
								particles[edge.p0].initialPosition).
							maximum(
								particles[edge.p1].initialPosition);
					}
					bounds.pushBack(edgeBounds);
				}
				PxSort(
					bounds.begin(), bounds.size(),
					[](const AvbdSoftPairEdgeBounds& a,
					   const AvbdSoftPairEdgeBounds& b)
					{
						return a.minimum.x < b.minimum.x;
					});
			};
			reserveSoftPairQueryScratch(
				bodyA.compiled.surfaceEdges.size(),
				bodyB.compiled.surfaceEdges.size(),
				PxMax(bodyA.compiled.surfaceTriangles.size() / 3,
					bodyB.compiled.surfaceTriangles.size() / 3));
			PxArray<AvbdSoftPairEdgeBounds>& edgeBoundsA =
				queryScratch.edgeBoundsA;
			PxArray<AvbdSoftPairEdgeBounds>& edgeBoundsB =
				queryScratch.edgeBoundsB;
			buildEdgeBounds(bodyA, edgeBoundsA);
			buildEdgeBounds(bodyB, edgeBoundsB);
			const PxReal edgeFeatureEpsilon = 1.0e-4f;
			const PxReal edgeDistanceEpsilon = 1.0e-8f;
			auto ownsEdgeContactDirection = [](
				const AvbdSoftPairEdgeBounds& edgeBounds,
				const PxVec3& outwardDirection) -> bool
			{
				return !edgeBounds.hasExteriorNormalCone ||
					avbdIsDirectionInSurfaceEdgeNormalCone(
						outwardDirection,
						edgeBounds.adjacentNormal0,
						edgeBounds.adjacentNormal1);
			};
			auto ownsSweptEdgeContactDirection = [&particles, numParticles](
				const AvbdSoftBody& body,
				const AvbdEdgeInfo& edge,
				PxReal time,
				const PxVec3& outwardDirection) -> bool
			{
				if(edge.adjacentSurfaceFace0 == PX_MAX_U32 ||
					edge.adjacentSurfaceFace1 == PX_MAX_U32)
					return true;
				const PxU32 face0 = edge.adjacentSurfaceFace0 * 3;
				const PxU32 face1 = edge.adjacentSurfaceFace1 * 3;
				if(face0 + 2 >= body.compiled.surfaceTriangles.size() ||
					face1 + 2 >= body.compiled.surfaceTriangles.size())
					return true;
				const PxU32 f00 = body.compiled.surfaceTriangles[face0];
				const PxU32 f01 = body.compiled.surfaceTriangles[face0 + 1];
				const PxU32 f02 = body.compiled.surfaceTriangles[face0 + 2];
				const PxU32 f10 = body.compiled.surfaceTriangles[face1];
				const PxU32 f11 = body.compiled.surfaceTriangles[face1 + 1];
				const PxU32 f12 = body.compiled.surfaceTriangles[face1 + 2];
				if(f00 >= numParticles || f01 >= numParticles ||
					f02 >= numParticles || f10 >= numParticles ||
					f11 >= numParticles || f12 >= numParticles)
					return true;
				auto positionAtTime = [&particles, time](PxU32 index)
				{
					return particles[index].initialPosition +
						(particles[index].position -
						 particles[index].initialPosition) * time;
				};
				const PxVec3 p00 = positionAtTime(f00);
				const PxVec3 p10 = positionAtTime(f10);
				const PxVec3 normal0 =
					(positionAtTime(f01) - p00).cross(
						positionAtTime(f02) - p00);
				const PxVec3 normal1 =
					(positionAtTime(f11) - p10).cross(
						positionAtTime(f12) - p10);
				return avbdIsDirectionInSurfaceEdgeNormalCone(
					outwardDirection, normal0, normal1);
			};

			auto findTargetEdgeElement =
				[&](const AvbdSoftBody& target,
					PxU32 edge0, PxU32 edge1) -> PxU32
			{
				for(PxU32 triangleOffset = 0;
					triangleOffset + 2 <
						target.compiled.surfaceTriangles.size();
					triangleOffset += 3)
				{
					const PxU32 v0 =
						target.compiled.surfaceTriangles[
							triangleOffset];
					const PxU32 v1 =
						target.compiled.surfaceTriangles[
							triangleOffset + 1];
					const PxU32 v2 =
						target.compiled.surfaceTriangles[
							triangleOffset + 2];
					const bool has0 =
						v0 == edge0 || v1 == edge0 || v2 == edge0;
					const bool has1 =
						v0 == edge1 || v1 == edge1 || v2 == edge1;
					if(has0 && has1)
					{
						const PxU32 triangleIndex =
							triangleOffset / 3;
						return triangleIndex <
								target.compiled.
									surfaceTriangleElementIndices.size()
							? target.compiled.
								surfaceTriangleElementIndices[
									triangleIndex]
							: PX_MAX_U32;
					}
				}
				return PX_MAX_U32;
			};

			for(PxU32 sortedEdgeA = 0;
				sortedEdgeA < edgeBoundsA.size();
				sortedEdgeA++)
			{
				const AvbdSoftPairEdgeBounds& boundsA =
					edgeBoundsA[sortedEdgeA];
				for(PxU32 sortedEdgeB = 0;
					sortedEdgeB < edgeBoundsB.size();
					sortedEdgeB++)
				{
					const AvbdSoftPairEdgeBounds& boundsB =
						edgeBoundsB[sortedEdgeB];
					if(boundsB.minimum.x > boundsA.maximum.x + r)
						break;
					if(boundsB.maximum.x < boundsA.minimum.x - r ||
						boundsA.minimum.y > boundsB.maximum.y + r ||
						boundsA.maximum.y < boundsB.minimum.y - r ||
						boundsA.minimum.z > boundsB.maximum.z + r ||
						boundsA.maximum.z < boundsB.minimum.z - r)
						continue;
					const AvbdEdgeInfo& queryEdge =
						bodyA.compiled.surfaceEdges[
							boundsA.edgeIndex];
					const AvbdEdgeInfo& targetEdge =
						bodyB.compiled.surfaceEdges[
							boundsB.edgeIndex];
					const PxU32 q0 = queryEdge.p0;
					const PxU32 q1 = queryEdge.p1;
					const PxU32 t0 = targetEdge.p0;
					const PxU32 t1 = targetEdge.p1;
					if(particles[q0].invMass <= 0.0f &&
						particles[q1].invMass <= 0.0f &&
						particles[t0].invMass <= 0.0f &&
						particles[t1].invMass <= 0.0f)
						continue;

					PxReal previousQueryWeight1 = 0.0f;
					PxReal previousTargetWeight1 = 0.0f;
					PxVec3 previousQueryClosest;
					PxVec3 previousTargetClosest;
					avbdClosestPointsOnSegments(
						particles[q0].initialPosition,
						particles[q1].initialPosition,
						particles[t0].initialPosition,
						particles[t1].initialPosition,
						previousQueryWeight1,
						previousTargetWeight1,
						previousQueryClosest,
						previousTargetClosest);
					const PxVec3 previousDelta =
						previousQueryClosest -
						previousTargetClosest;
					auto stabilizeNormal =
						[&](PxVec3 normal) -> PxVec3
					{
						if(previousDelta.magnitudeSquared() >
							edgeDistanceEpsilon *
								edgeDistanceEpsilon)
						{
							if(normal.dot(previousDelta) < 0.0f)
								normal = -normal;
						}
						else
						{
							const PxVec3 previousCross =
								(particles[q1].initialPosition -
								 particles[q0].initialPosition).cross(
									particles[t1].initialPosition -
									particles[t0].initialPosition);
							if(previousCross.magnitudeSquared() >
									edgeDistanceEpsilon *
										edgeDistanceEpsilon &&
								normal.dot(previousCross) < 0.0f)
								normal = -normal;
						}
						return normal;
					};
					auto appendEdgeContact =
						[&](PxReal queryWeight1,
							PxReal targetWeight1,
							const PxVec3& normal,
							PxReal depth,
							const PxVec3& surfacePoint)
					{
						AvbdSoftContactGeometry geometry;
						geometry.source = AvbdSoftContactSource(
							AvbdSoftContactSource::eSOFT_SURFACE,
							sB,
							avbdGetRigidSoftFeatureKey(
								0x53504530u, t0, t1, 0u, 0u),
							avbdGetRigidSoftFeatureKey(
								0x53504531u, q0, q1, t0, t1));
						geometry.particleIdx =
							particles[q0].invMass > 0.0f ? q0 :
							(particles[q1].invMass > 0.0f ? q1 :
							(particles[t0].invMass > 0.0f ? t0 : t1));
						geometry.queryParticleIndices[0] = q0;
						geometry.queryParticleIndices[1] = q1;
						geometry.queryWeights[0] =
							1.0f - queryWeight1;
						geometry.queryWeights[1] = queryWeight1;
						geometry.targetKind =
							AvbdSoftContactTargetKind::
								eDEFORMABLE_SURFACE;
						geometry.velocityOwner =
							AvbdVelocityObjectiveOwner::PositionAL;
						geometry.targetIndex = sB;
						geometry.targetSourceElementIndex =
							findTargetEdgeElement(bodyB, t0, t1);
						geometry.surfaceParticleIndices[0] = t0;
						geometry.surfaceParticleIndices[1] = t1;
						geometry.surfaceWeights[0] =
							1.0f - targetWeight1;
						geometry.surfaceWeights[1] = targetWeight1;
						geometry.normal = normal;
						geometry.projNormal = normal;
						geometry.depth = depth;
						geometry.margin = r;
						geometry.surfacePoint = surfacePoint;
						geometry.friction = pairFriction;
						avbdBuildSoftContactTangents(geometry);
						avbdAppendPreparedSoftContact(
							geometry,
							params.contactStiffness,
							params.contactStiffness * 10.0f,
							particles, contacts);
					};

					if(pairSpeculative)
					{
						AvbdSweptConvexEdgeEntry entry;
						if(avbdDeformingSegmentsEnterExpandedInteriors(
								particles[q0].initialPosition,
								particles[q1].initialPosition,
								particles[q0].position,
								particles[q1].position,
								particles[t0].initialPosition,
								particles[t1].initialPosition,
								particles[t0].position,
								particles[t1].position,
								r, entry))
						{
							const PxVec3 sweptNormal =
								stabilizeNormal(entry.normal);
							// A swept edge row is valid only when the entry
							// direction is owned by both exterior edge cones.
							// Evaluate those cones at the exact entry time so a
							// rotating crease cannot be rejected by its end pose.
							if(ownsSweptEdgeContactDirection(
									bodyA, queryEdge, entry.entryTime,
									-sweptNormal) &&
								ownsSweptEdgeContactDirection(
									bodyB, targetEdge, entry.entryTime,
									sweptNormal))
							{
								const PxVec3 target0AtEntry =
									particles[t0].initialPosition +
									(particles[t0].position -
									 particles[t0].initialPosition) *
										entry.entryTime;
								const PxVec3 target1AtEntry =
									particles[t1].initialPosition +
									(particles[t1].position -
									 particles[t1].initialPosition) *
										entry.entryTime;
								appendEdgeContact(
									entry.softWeight1,
									entry.rigidWeight1,
									sweptNormal,
									0.0f,
									target0AtEntry *
										(1.0f - entry.rigidWeight1) +
									target1AtEntry *
										entry.rigidWeight1);
								continue;
							}
						}
					}

					PxReal queryWeight1 = 0.0f;
					PxReal targetWeight1 = 0.0f;
					PxVec3 queryClosest;
					PxVec3 targetClosest;
					avbdClosestPointsOnSegments(
						particles[q0].position,
						particles[q1].position,
						particles[t0].position,
						particles[t1].position,
						queryWeight1, targetWeight1,
						queryClosest, targetClosest);
					if(queryWeight1 <= edgeFeatureEpsilon ||
						queryWeight1 >=
							1.0f - edgeFeatureEpsilon ||
						targetWeight1 <= edgeFeatureEpsilon ||
						targetWeight1 >=
							1.0f - edgeFeatureEpsilon)
						continue;
					const PxVec3 delta =
						queryClosest - targetClosest;
					const PxReal distance = delta.magnitude();
					if(distance >= r)
						continue;
					PxVec3 normal;
					if(distance > edgeDistanceEpsilon)
						normal = delta * (1.0f / distance);
					else
					{
						normal =
							(particles[q1].position -
							 particles[q0].position).cross(
								particles[t1].position -
								particles[t0].position);
						if(normal.magnitudeSquared() <=
							edgeDistanceEpsilon *
								edgeDistanceEpsilon)
							continue;
						normal.normalize();
					}
					normal = stabilizeNormal(normal);
					if(!ownsEdgeContactDirection(boundsA, -normal) ||
						!ownsEdgeContactDirection(boundsB, normal))
						continue;
					appendEdgeContact(
						queryWeight1, targetWeight1,
						normal,
						r - distance, targetClosest);
				}
		}
	}
}

// Legacy serial entry. It owns the mutable plan/refit epoch, then consumes the
// entire canonical stream through the same P5.9c range leaf used by a future
// task transaction.
inline void avbdDetectSoftSoftOGC(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params = AvbdOGCParams(),
	AvbdSoftCollisionStats* stats = NULL,
	AvbdSoftContactWorkspace* persistentWorkspace = NULL,
	AvbdSoftSoftPairQueryScratch* queryScratchOverride = NULL)
{
	AvbdSoftContactWorkspace localWorkspace;
	AvbdSoftContactWorkspace& workspace =
		persistentWorkspace ? *persistentWorkspace : localWorkspace;
	avbdBuildSoftSoftOGCDetectionPlan(
		particles, softBodies, numSoftBodies, params, stats, workspace);
	const bool useSurfaceTriangleBvh =
		avbdRefitSoftSoftOGCDetectionPlan(
			particles, softBodies, numSoftBodies, stats, workspace);
	AvbdSoftSoftPairQueryScratch& queryScratch =
		queryScratchOverride ? *queryScratchOverride :
		workspace.softPairQueryScratch;
	avbdDetectSoftSoftOGCPlanRange(
		particles, numParticles, softBodies, numSoftBodies, workspace,
		queryScratchOverride ? NULL : &workspace, queryScratch,
		useSurfaceTriangleBvh, 0, workspace.softPairDetectionPlan.size(),
		contacts, params, stats);
}

// =============================================================================
// PATH 4 (OGC): Full Self-Collision Detection
//
// Two-stage C2 activation, topological adjacency filtering, safety bubble.
// =============================================================================

// Build topological adjacency for self-collision filtering.
// Returns per-particle sorted list of connected local particle indices.
inline void avbdBuildSelfCollisionAdjacency(
	const AvbdSoftBody& sb,
	PxArray<PxArray<PxU32> >& adj)
{
	adj.resize(sb.compiled.particleCount);
	for (PxU32 i = 0; i < sb.compiled.particleCount; i++)
		adj[i].clear();

	auto addAdj = [&](PxU32 la, PxU32 lb) {
		adj[la].pushBack(lb);
		adj[lb].pushBack(la);
	};

	for (PxU32 i = 0; i + 3 < sb.compiled.tetrahedra.size(); i += 4) {
		PxU32 v[4];
		for (int j = 0; j < 4; j++) v[j] = sb.compiled.tetrahedra[i + PxU32(j)];
		for (int a = 0; a < 4; a++)
			for (int b = a + 1; b < 4; b++)
				addAdj(v[a], v[b]);
	}
	for (PxU32 i = 0; i + 2 < sb.compiled.triangles.size(); i += 3) {
		PxU32 v[3];
		for (int j = 0; j < 3; j++) v[j] = sb.compiled.triangles[i + PxU32(j)];
		for (int a = 0; a < 3; a++)
			for (int b = a + 1; b < 3; b++)
				addAdj(v[a], v[b]);
	}

	// Sort and deduplicate
	for (PxU32 i = 0; i < sb.compiled.particleCount; i++) {
		PxArray<PxU32>& a = adj[i];
		if (a.size() > 1) {
			PxSort(a.begin(), a.size());
			PxU32 writeIdx = 1;
			for (PxU32 k = 1; k < a.size(); k++)
				if (a[k] != a[k-1])
					a[writeIdx++] = a[k];
			a.resize(writeIdx);
		}
	}
}

PX_FORCE_INLINE bool avbdIsAdjacentSelfCollision(
	PxU32 localA, PxU32 localB,
	const PxArray<PxArray<PxU32> >& adj)
{
	if (localA >= adj.size()) return false;
	const PxArray<PxU32>& a = adj[localA];
	// Binary search in sorted array
	PxU32 lo = 0, hi = a.size();
	while (lo < hi) {
		PxU32 mid = (lo + hi) / 2;
		if (a[mid] < localB) lo = mid + 1;
		else if (a[mid] > localB) hi = mid;
		else return true;
	}
	return false;
}

// Per-vertex conservative displacement bound (Eq. 21)
inline void avbdComputeSafetyBounds(
	const AvbdSoftBody& sb,
	const AvbdSoftParticle* particles,
	const PxArray<PxArray<PxU32> >& adj,
	PxReal queryRadius,
	PxReal gammaP,
	PxArray<PxReal>& bounds,
	AvbdSoftContactWorkspace& workspace)
{
	PX_UNUSED(adj);
	const PxU32 particleCount = sb.compiled.particleCount;
	const PxU32 particleStart = sb.compiled.particleStart;
	const PxReal rq = PxMax(queryRadius, 1.0e-6f);
	const PxReal gamma = PxClamp(gammaP, 1.0e-4f, 0.499f);
	const PxReal filterDistance =
		PxMax(sb.compiled.selfCollisionFilterDistance, 0.0f);
	const bool hasRestFilter =
		filterDistance > 0.0f &&
		sb.compiled.selfCollisionRestPositions.size() == particleCount;

	bounds.resize(particleCount);
	const PxU32 triangleCount = sb.compiled.surfaceTriangles.size() / 3;
	const PxU32 edgeCount = sb.compiled.surfaceEdges.size();
	// Detection and safety execute in separate outer-epoch phases.  Reuse the
	// same caller-owned sweep buffers here, but preserve the serial traversal,
	// sort predicates and floating-point reduction order below.
	workspace.reserveSweepScratch(
		workspace.selfSafetyTriangleMinimums, triangleCount);
	workspace.reserveSweepScratch(
		workspace.selfSafetyEdgeMinimums, edgeCount);
	workspace.reserveSweepScratch(
		workspace.selfTriangleBounds, triangleCount);
	workspace.reserveSweepScratch(
		workspace.selfSortedVertices,
		sb.compiled.surfaceVertices.size());
	workspace.reserveSweepScratch(
		workspace.selfActiveTriangles, triangleCount);
	workspace.reserveSweepScratch(workspace.selfEdgeBounds, edgeCount);
	PxArray<PxReal>& triangleMinimums =
		workspace.selfSafetyTriangleMinimums;
	PxArray<PxReal>& edgeMinimums =
		workspace.selfSafetyEdgeMinimums;
	triangleMinimums.resize(triangleCount);
	edgeMinimums.resize(edgeCount);
	for(PxU32 vertexIndex = 0;
		vertexIndex < particleCount; vertexIndex++)
		bounds[vertexIndex] = rq;
	for(PxU32 triangleIndex = 0;
		triangleIndex < triangleMinimums.size(); triangleIndex++)
		triangleMinimums[triangleIndex] = rq;
	for(PxU32 edgeIndex = 0;
		edgeIndex < edgeMinimums.size(); edgeIndex++)
		edgeMinimums[edgeIndex] = rq;

	PxArray<AvbdSelfCollisionTriangleBounds>& triangleBounds =
		workspace.selfTriangleBounds;
	triangleBounds.clear();
	for(PxU32 triangleOffset = 0;
		triangleOffset + 2 <
			sb.compiled.surfaceTriangles.size();
		triangleOffset += 3)
	{
		const PxU32 vertex0 =
			sb.compiled.surfaceTriangles[triangleOffset];
		const PxU32 vertex1 =
			sb.compiled.surfaceTriangles[triangleOffset + 1];
		const PxU32 vertex2 =
			sb.compiled.surfaceTriangles[triangleOffset + 2];
		if(vertex0 < particleStart ||
			vertex1 < particleStart ||
			vertex2 < particleStart ||
			vertex0 - particleStart >= particleCount ||
			vertex1 - particleStart >= particleCount ||
			vertex2 - particleStart >= particleCount)
			continue;
		AvbdSelfCollisionTriangleBounds triangle;
		triangle.triangleOffset = triangleOffset;
		triangle.minimum =
			particles[vertex0].position.minimum(
				particles[vertex1].position).minimum(
				particles[vertex2].position);
		triangle.maximum =
			particles[vertex0].position.maximum(
				particles[vertex1].position).maximum(
				particles[vertex2].position);
		triangleBounds.pushBack(triangle);
	}
	PxSort(
		triangleBounds.begin(), triangleBounds.size(),
		[](const AvbdSelfCollisionTriangleBounds& a,
		   const AvbdSelfCollisionTriangleBounds& b)
		{
			return a.minimum.x < b.minimum.x;
		});
	PxArray<AvbdSelfCollisionVertexSweepEntry>& sortedVertices =
		workspace.selfSortedVertices;
	sortedVertices.clear();
	for(PxU32 surfaceVertexIndex = 0;
		surfaceVertexIndex <
			sb.compiled.surfaceVertices.size();
		surfaceVertexIndex++)
	{
		const PxU32 globalIndex =
			sb.compiled.surfaceVertices[surfaceVertexIndex];
		if(globalIndex < particleStart ||
			globalIndex - particleStart >= particleCount)
			continue;
		AvbdSelfCollisionVertexSweepEntry vertex;
		vertex.localIndex = globalIndex - particleStart;
		vertex.minimumX =
			particles[globalIndex].position.x;
		vertex.maximumX = vertex.minimumX;
		sortedVertices.pushBack(vertex);
	}
	PxSort(
		sortedVertices.begin(), sortedVertices.size(),
		[](const AvbdSelfCollisionVertexSweepEntry& a,
		   const AvbdSelfCollisionVertexSweepEntry& b)
		{
			return a.minimumX < b.minimumX;
		});

	// OGC Eq. 22 and Eq. 26.  The sweep-and-prune list is the CPU
	// equivalent of the paper's facet-BVH radius query.  Values are
	// initialized to rq, so pairs outside the query shell cannot reduce the
	// conservative bound.
	PxArray<PxU32>& activeTriangles = workspace.selfActiveTriangles;
	activeTriangles.clear();
	PxU32 triangleCursor = 0;
	for(PxU32 sortedVertexIndex = 0;
		sortedVertexIndex < sortedVertices.size();
		sortedVertexIndex++)
	{
		const PxU32 localIndex =
			sortedVertices[sortedVertexIndex].localIndex;
		const PxU32 globalIndex = particleStart + localIndex;
		const PxVec3& point = particles[globalIndex].position;
		while(triangleCursor < triangleBounds.size() &&
			triangleBounds[triangleCursor].minimum.x <=
				point.x + rq)
			activeTriangles.pushBack(triangleCursor++);

		for(PxU32 activeIndex = 0;
			activeIndex < activeTriangles.size();)
		{
			const AvbdSelfCollisionTriangleBounds& triangle =
				triangleBounds[activeTriangles[activeIndex]];
			if(triangle.maximum.x < point.x - rq)
			{
				activeTriangles[activeIndex] =
					activeTriangles.back();
				activeTriangles.popBack();
				continue;
			}
			activeIndex++;
			if(triangle.minimum.y > point.y + rq ||
				triangle.maximum.y < point.y - rq ||
				triangle.minimum.z > point.z + rq ||
				triangle.maximum.z < point.z - rq)
				continue;

			const PxU32 triangleOffset =
				triangle.triangleOffset;
			const PxU32 vertex0 =
				sb.compiled.surfaceTriangles[triangleOffset];
			const PxU32 vertex1 =
				sb.compiled.surfaceTriangles[
					triangleOffset + 1];
			const PxU32 vertex2 =
				sb.compiled.surfaceTriangles[
					triangleOffset + 2];
			if(globalIndex == vertex0 ||
				globalIndex == vertex1 ||
				globalIndex == vertex2)
				continue;
			if(hasRestFilter)
			{
				if(sb.compiled.selfCollisionRestFilterCacheValid)
				{
					if(avbdIsSelfRestVertexTriangleFiltered(
						sb, localIndex, triangleOffset / 3))
						continue;
				}
				else
				{
					const PxArray<PxVec3>& restPositions =
						sb.compiled.selfCollisionRestPositions;
					const AvbdClosestPointResult restClosest =
						avbdClosestPointOnTriangleOGC(
							restPositions[localIndex],
							restPositions[vertex0 - particleStart],
							restPositions[vertex1 - particleStart],
							restPositions[vertex2 - particleStart]);
					if(restClosest.distance <= filterDistance)
						continue;
				}
			}
			const AvbdClosestPointResult closest =
				avbdClosestPointOnTriangleOGC(
					point,
					particles[vertex0].position,
					particles[vertex1].position,
					particles[vertex2].position);
			if(closest.distance >= rq)
				continue;
			bounds[localIndex] = PxMin(
				bounds[localIndex],
				closest.distance);
			const PxU32 triangleIndex = triangleOffset / 3;
			triangleMinimums[triangleIndex] = PxMin(
				triangleMinimums[triangleIndex],
				closest.distance);
		}
	}

	PxArray<AvbdSelfCollisionEdgeBounds>& edgeBounds =
		workspace.selfEdgeBounds;
	edgeBounds.clear();
	for(PxU32 edgeIndex = 0;
		edgeIndex < sb.compiled.surfaceEdges.size(); edgeIndex++)
	{
		const AvbdEdgeInfo& edge =
			sb.compiled.surfaceEdges[edgeIndex];
		if(edge.p0 < particleStart ||
			edge.p1 < particleStart ||
			edge.p0 - particleStart >= particleCount ||
			edge.p1 - particleStart >= particleCount)
			continue;
		AvbdSelfCollisionEdgeBounds edgeBound;
		edgeBound.edgeIndex = edgeIndex;
		edgeBound.minimum =
			particles[edge.p0].position.minimum(
				particles[edge.p1].position);
		edgeBound.maximum =
			particles[edge.p0].position.maximum(
				particles[edge.p1].position);
		edgeBounds.pushBack(edgeBound);
	}
	PxSort(
		edgeBounds.begin(), edgeBounds.size(),
		[](const AvbdSelfCollisionEdgeBounds& a,
		   const AvbdSelfCollisionEdgeBounds& b)
		{
			return a.minimum.x < b.minimum.x;
		});

	// OGC Eq. 24.  Every unordered non-incident edge pair contributes its
	// distance to both edge minima.
	for(PxU32 sortedEdge0 = 0;
		sortedEdge0 < edgeBounds.size(); sortedEdge0++)
	{
		const AvbdSelfCollisionEdgeBounds& bounds0 =
			edgeBounds[sortedEdge0];
		for(PxU32 sortedEdge1 = sortedEdge0 + 1;
			sortedEdge1 < edgeBounds.size(); sortedEdge1++)
		{
			const AvbdSelfCollisionEdgeBounds& bounds1 =
				edgeBounds[sortedEdge1];
			if(bounds1.minimum.x > bounds0.maximum.x + rq)
				break;
			if(bounds0.minimum.y > bounds1.maximum.y + rq ||
				bounds0.maximum.y < bounds1.minimum.y - rq ||
				bounds0.minimum.z > bounds1.maximum.z + rq ||
				bounds0.maximum.z < bounds1.minimum.z - rq)
				continue;
			const AvbdEdgeInfo& edge0 =
				sb.compiled.surfaceEdges[bounds0.edgeIndex];
			const AvbdEdgeInfo& edge1 =
				sb.compiled.surfaceEdges[bounds1.edgeIndex];
			if(edge0.p0 == edge1.p0 ||
				edge0.p0 == edge1.p1 ||
				edge0.p1 == edge1.p0 ||
				edge0.p1 == edge1.p1)
				continue;
			if(hasRestFilter)
			{
				const PxArray<PxVec3>& restPositions =
					sb.compiled.selfCollisionRestPositions;
				PxReal restWeight0 = 0.0f;
				PxReal restWeight1 = 0.0f;
				PxVec3 restClosest0, restClosest1;
				avbdClosestPointsOnSegments(
					restPositions[edge0.p0 - particleStart],
					restPositions[edge0.p1 - particleStart],
					restPositions[edge1.p0 - particleStart],
					restPositions[edge1.p1 - particleStart],
					restWeight0, restWeight1,
					restClosest0, restClosest1);
				if((restClosest0 - restClosest1).
						magnitude() <= filterDistance)
					continue;
			}
			PxReal weight0 = 0.0f;
			PxReal weight1 = 0.0f;
			PxVec3 closest0, closest1;
			avbdClosestPointsOnSegments(
				particles[edge0.p0].position,
				particles[edge0.p1].position,
				particles[edge1.p0].position,
				particles[edge1.p1].position,
				weight0, weight1, closest0, closest1);
			const PxReal distance =
				(closest0 - closest1).magnitude();
			if(distance >= rq)
				continue;
			edgeMinimums[bounds0.edgeIndex] = PxMin(
				edgeMinimums[bounds0.edgeIndex], distance);
			edgeMinimums[bounds1.edgeIndex] = PxMin(
				edgeMinimums[bounds1.edgeIndex], distance);
		}
	}

	// OGC Eq. 21, 23, and 25: gather the incident edge and triangle
	// minima onto each vertex, then apply gamma_p.
	for(PxU32 triangleOffset = 0;
		triangleOffset + 2 <
			sb.compiled.surfaceTriangles.size();
		triangleOffset += 3)
	{
		const PxReal triangleMinimum =
			triangleMinimums[triangleOffset / 3];
		for(PxU32 corner = 0; corner < 3; corner++)
		{
			const PxU32 globalIndex =
				sb.compiled.surfaceTriangles[
					triangleOffset + corner];
			if(globalIndex >= particleStart &&
				globalIndex - particleStart < particleCount)
				bounds[globalIndex - particleStart] = PxMin(
					bounds[globalIndex - particleStart],
					triangleMinimum);
		}
	}
	for(PxU32 edgeIndex = 0;
		edgeIndex < sb.compiled.surfaceEdges.size(); edgeIndex++)
	{
		const AvbdEdgeInfo& edge =
			sb.compiled.surfaceEdges[edgeIndex];
		const PxReal edgeMinimum = edgeMinimums[edgeIndex];
		if(edge.p0 >= particleStart &&
			edge.p0 - particleStart < particleCount)
			bounds[edge.p0 - particleStart] = PxMin(
				bounds[edge.p0 - particleStart], edgeMinimum);
		if(edge.p1 >= particleStart &&
			edge.p1 - particleStart < particleCount)
			bounds[edge.p1 - particleStart] = PxMin(
				bounds[edge.p1 - particleStart], edgeMinimum);
	}
	for(PxU32 localIndex = 0;
		localIndex < particleCount; localIndex++)
		bounds[localIndex] =
			gamma * PxMax(bounds[localIndex], 1.0e-6f);
}

// Truncate displacement to safety bound
PX_FORCE_INLINE void avbdTruncateDisplacement(
	AvbdSoftParticle& sp,
	const PxVec3& prevPosition,
	PxReal bound)
{
	PxVec3 disp = sp.position - prevPosition;
	PxReal dispMag = disp.magnitude();
	if (dispMag > bound && dispMag > 1e-10f)
		sp.position = prevPosition + disp * (bound / dispMag);
}

// Detect self-collision contacts within a single soft body
inline void avbdDetectSelfCollisionOGC(
	const AvbdSoftParticle* particles,
	const AvbdSoftBody& sb,
	PxU32 softBodyIdx,
	const PxArray<PxArray<PxU32> >& adj,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params = AvbdOGCParams(),
	AvbdSoftCollisionStats* stats = NULL,
	AvbdSoftContactWorkspace* persistentWorkspace = NULL,
	// When supplied, the immutable stress coefficients and both BVH refits
	// belong to a parent transaction.  The caller must provide a distinct
	// persistentWorkspace for this range leaf: candidates, emitted-feature
	// keys and all output remain private to that leaf.
	const AvbdSoftContactWorkspace* preparedBvhWorkspace = NULL,
	PxU32 vertexLoopBegin = 0,
	PxU32 vertexLoopEnd = PX_MAX_U32,
	PxU32 edgeLoopBegin = 0,
	PxU32 edgeLoopEnd = PX_MAX_U32)
{
	AvbdSoftContactWorkspace localWorkspace;
	AvbdSoftContactWorkspace& workspace =
		persistentWorkspace ? *persistentWorkspace : localWorkspace;
	const bool usePreparedBvhWorkspace = preparedBvhWorkspace != NULL;
	workspace.reserveSelfCollisionSweep(
		sb.compiled.tetElements.size(),
		sb.compiled.surfaceTriangles.size() / 3,
		sb.compiled.surfaceVertices.size(),
		sb.compiled.surfaceEdges.size());
	if(!usePreparedBvhWorkspace)
		workspace.prepareSelfBvhBounds(
			sb.compiled.surfaceTriangleBvhNodes.size(),
			sb.compiled.surfaceEdgeBvhNodes.size());
	PxReal r   = params.contactRadius;
	PxReal tau = params.getTau();
	PX_UNUSED(adj);
	const bool sweepEnabled =
		sb.compiled.speculativeCCDEnabled;
	const PxReal filterDistance =
		PxMax(sb.compiled.selfCollisionFilterDistance, 0.0f);
	const bool hasRestFilter =
		filterDistance > 0.0f &&
		sb.compiled.selfCollisionRestPositions.size() ==
			sb.compiled.particleCount;
	PX_ASSERT(
		filterDistance == 0.0f ||
		sb.compiled.selfCollisionRestPositions.size() ==
			sb.compiled.particleCount);
	PxArray<PxReal>& localTetStressCoefficients =
		workspace.selfTetStressCoefficients;
	const PxArray<PxReal>* tetStressCoefficients =
		usePreparedBvhWorkspace
			? &preparedBvhWorkspace->selfTetStressCoefficients
			: &localTetStressCoefficients;
	if(!usePreparedBvhWorkspace)
	{
		localTetStressCoefficients.clear();
		if(!sb.compiled.tetElements.empty())
		{
			localTetStressCoefficients.resize(
				sb.compiled.tetElements.size());
			for(PxU32 tetIndex = 0;
				tetIndex < sb.compiled.tetElements.size();
				tetIndex++)
			{
				localTetStressCoefficients[tetIndex] =
					avbdComputeTetStressCoefficient(
						sb.compiled.tetElements[tetIndex],
						particles);
			}
		}
	}
	auto targetStressAllowsTriangle =
		[&](PxU32 triangleOffset) -> bool
	{
		if(tetStressCoefficients->empty())
			return true;
		const PxU32 triangleIndex = triangleOffset / 3;
		if(triangleIndex >=
			sb.compiled.surfaceTriangleTetElementIndices.size())
			return true;
		const PxU32 tetElementIndex =
			sb.compiled.surfaceTriangleTetElementIndices[
				triangleIndex];
		if(tetElementIndex < tetStressCoefficients->size())
			return (*tetStressCoefficients)[tetElementIndex] <=
				sb.compiled.selfCollisionStressTolerance;
		// Preserve the previous behavior when a source tetrahedron was skipped
		// during element compilation: no known compiled stress owner means the
		// boundary triangle remains eligible for self collision.
		return true;
	};
	// Keep the topology-stable hierarchy refitted from the authoritative
	// current/swept particle positions.  The legacy x-sweep remains the exact
	// fallback for bodies with no compiled hierarchy and for the benchmark
	// switch above; no contact-owner policy changes with this choice.
	const bool useSurfaceTriangleBvh = usePreparedBvhWorkspace ||
		(avbdUseSurfaceTriangleBvh() &&
		 !sb.compiled.surfaceTriangleBvhNodes.empty());
	PX_ASSERT(!usePreparedBvhWorkspace ||
		!sb.compiled.surfaceTriangleBvhNodes.empty());
	if(useSurfaceTriangleBvh && !usePreparedBvhWorkspace)
	{
		sb.compiled.refitSurfaceTriangleBvh(
			particles, sweepEnabled, workspace.selfTriangleBvhBounds);
		if(stats)
			stats->surfaceTriangleBvhRefitNodes +=
				sb.compiled.surfaceTriangleBvhNodes.size();
	}
	const PxArray<AvbdSurfaceBvhNodeBounds>& triangleBvhBounds =
		usePreparedBvhWorkspace
			? preparedBvhWorkspace->selfTriangleBvhBounds
			: workspace.selfTriangleBvhBounds;
	PX_ASSERT(!usePreparedBvhWorkspace ||
		triangleBvhBounds.size() ==
			sb.compiled.surfaceTriangleBvhNodes.size());

	PxArray<AvbdSelfCollisionTriangleBounds>& triangleBounds =
		workspace.selfTriangleBounds;
	triangleBounds.clear();
	if(!useSurfaceTriangleBvh)
	{
		for(PxU32 triangleOffset = 0;
			triangleOffset + 2 <
				sb.compiled.surfaceTriangles.size();
			triangleOffset += 3)
		{
			const PxU32 source0 =
				sb.compiled.surfaceTriangles[triangleOffset];
			const PxU32 source1 =
				sb.compiled.surfaceTriangles[triangleOffset + 1];
			const PxU32 source2 =
				sb.compiled.surfaceTriangles[triangleOffset + 2];
			AvbdSelfCollisionTriangleBounds triangle;
			triangle.triangleOffset = triangleOffset;
			triangle.minimum =
				particles[source0].position.minimum(
					particles[source1].position).minimum(
					particles[source2].position);
			triangle.maximum =
				particles[source0].position.maximum(
					particles[source1].position).maximum(
					particles[source2].position);
			if(sweepEnabled)
			{
				triangle.minimum = triangle.minimum.minimum(
					particles[source0].initialPosition).minimum(
					particles[source1].initialPosition).minimum(
					particles[source2].initialPosition);
				triangle.maximum = triangle.maximum.maximum(
					particles[source0].initialPosition).maximum(
					particles[source1].initialPosition).maximum(
					particles[source2].initialPosition);
			}
			triangleBounds.pushBack(triangle);
			if(stats)
				stats->selfTriangleBoundsBuilt++;
		}
	}
	if(!useSurfaceTriangleBvh)
		PxSort(
			triangleBounds.begin(), triangleBounds.size(),
			[](const AvbdSelfCollisionTriangleBounds& a,
			   const AvbdSelfCollisionTriangleBounds& b)
			{
				return a.minimum.x < b.minimum.x;
			});
	PxArray<AvbdSelfCollisionVertexSweepEntry>& sortedVertices =
		workspace.selfSortedVertices;
	sortedVertices.clear();
	if(!useSurfaceTriangleBvh)
	{
		for(PxU32 surfaceVertexIndex = 0;
			surfaceVertexIndex <
				sb.compiled.surfaceVertices.size();
			surfaceVertexIndex++)
		{
			const PxU32 globalIndex =
				sb.compiled.surfaceVertices[surfaceVertexIndex];
			if(globalIndex < sb.compiled.particleStart ||
				globalIndex - sb.compiled.particleStart >=
				sb.compiled.particleCount)
				continue;
			AvbdSelfCollisionVertexSweepEntry vertex;
			vertex.localIndex =
				globalIndex - sb.compiled.particleStart;
			vertex.minimumX = particles[globalIndex].position.x;
			vertex.maximumX = particles[globalIndex].position.x;
			if(sweepEnabled)
			{
				vertex.minimumX = PxMin(
					vertex.minimumX,
					particles[globalIndex].initialPosition.x);
				vertex.maximumX = PxMax(
					vertex.maximumX,
					particles[globalIndex].initialPosition.x);
			}
			sortedVertices.pushBack(vertex);
			if(stats)
				stats->selfVertexSweepEntriesBuilt++;
		}
	}
	if(!useSurfaceTriangleBvh)
		PxSort(
			sortedVertices.begin(), sortedVertices.size(),
			[](const AvbdSelfCollisionVertexSweepEntry& a,
			   const AvbdSelfCollisionVertexSweepEntry& b)
			{
				return a.minimumX < b.minimumX;
			});

	// Radius-query broadphase.  The previous all-vertices by all-triangles
	// traversal made each OGC redetection O(V*T).
	PxArray<PxU32>& activeTriangles = workspace.selfActiveTriangles;
	activeTriangles.clear();
	PxArray<PxU32>& triangleCandidates =
		workspace.selfTriangleCandidates;
	triangleCandidates.clear();
	PxArray<PxU64>& emittedFeatureKeys = workspace.selfEmittedFeatureKeys;
	emittedFeatureKeys.clear();
	auto triangleOverlapsQuery =
		[&](PxU32 triangleOffset, const PxVec3& queryMinimum,
			const PxVec3& queryMaximum) -> bool
	{
		const PxU32 source0 =
			sb.compiled.surfaceTriangles[triangleOffset];
		const PxU32 source1 =
			sb.compiled.surfaceTriangles[triangleOffset + 1];
		const PxU32 source2 =
			sb.compiled.surfaceTriangles[triangleOffset + 2];
		PxVec3 minimum = particles[source0].position.minimum(
			particles[source1].position).minimum(
			particles[source2].position);
		PxVec3 maximum = particles[source0].position.maximum(
			particles[source1].position).maximum(
			particles[source2].position);
		if(sweepEnabled)
		{
			minimum = minimum.minimum(
				particles[source0].initialPosition).minimum(
				particles[source1].initialPosition).minimum(
				particles[source2].initialPosition);
			maximum = maximum.maximum(
				particles[source0].initialPosition).maximum(
				particles[source1].initialPosition).maximum(
				particles[source2].initialPosition);
		}
		return !(minimum.x > queryMaximum.x + r ||
			maximum.x < queryMinimum.x - r ||
			minimum.y > queryMaximum.y + r ||
			maximum.y < queryMinimum.y - r ||
			minimum.z > queryMaximum.z + r ||
			maximum.z < queryMinimum.z - r);
	};
	PxU32 triangleCursor = 0;
	const PxU32 vertexLoopCount = useSurfaceTriangleBvh
		? sb.compiled.surfaceVertices.size() : sortedVertices.size();
	const PxU32 clampedVertexLoopBegin =
		PxMin(vertexLoopBegin, vertexLoopCount);
	const PxU32 clampedVertexLoopEnd =
		PxMin(PxMax(vertexLoopEnd, clampedVertexLoopBegin),
			vertexLoopCount);
	for(PxU32 vertexLoopIndex = clampedVertexLoopBegin;
		vertexLoopIndex < clampedVertexLoopEnd;
		vertexLoopIndex++)
	{
		const PxU32 gi = useSurfaceTriangleBvh
			? sb.compiled.surfaceVertices[vertexLoopIndex]
			: sb.compiled.particleStart +
				sortedVertices[vertexLoopIndex].localIndex;
		if(gi < sb.compiled.particleStart ||
			gi - sb.compiled.particleStart >=
				sb.compiled.particleCount)
			continue;
		const PxU32 li = gi - sb.compiled.particleStart;
		const PxVec3& pp = particles[gi].position;
		const PxVec3 vertexMinimum = sweepEnabled
			? particles[gi].initialPosition.minimum(pp) : pp;
		const PxVec3 vertexMaximum = sweepEnabled
			? particles[gi].initialPosition.maximum(pp) : pp;
		if(useSurfaceTriangleBvh)
		{
			sb.compiled.collectSurfaceTriangleBvhCandidates(
				vertexMinimum, vertexMaximum, r,
				triangleBvhBounds, triangleCandidates);
			if(stats)
				stats->surfaceTriangleBvhCandidateTriangles +=
					triangleCandidates.size();
		}
		if(!useSurfaceTriangleBvh)
		{
		const PxReal vertexMinimumX =
			sortedVertices[vertexLoopIndex].minimumX;
		const PxReal vertexMaximumX =
			sortedVertices[vertexLoopIndex].maximumX;
		while(triangleCursor < triangleBounds.size() &&
			triangleBounds[triangleCursor].minimum.x <=
			vertexMaximumX + r)
			activeTriangles.pushBack(triangleCursor++);
		for(PxU32 activeIndex = 0;
			activeIndex < activeTriangles.size();)
		{
			const AvbdSelfCollisionTriangleBounds& triangle =
				triangleBounds[activeTriangles[activeIndex]];
			if(triangle.maximum.x < vertexMinimumX - r)
			{
				activeTriangles[activeIndex] =
					activeTriangles.back();
				activeTriangles.popBack();
				continue;
			}
			activeIndex++;
		}
		}
		const PxU32 activeTriangleCount = useSurfaceTriangleBvh
			? triangleCandidates.size() : activeTriangles.size();

		// Select a single first face crossing for this vertex.  This prevents
		// adjacent triangles from compiling several speculative rows for one
		// physical crossing and keeps every outer redetection deterministic.
		if(sweepEnabled)
		{
			PxReal bestEntryTime = PX_MAX_F32;
			PxU32 bestTriangleOffset = PX_MAX_U32;
			AvbdSweptTriangleEntry bestEntry;
			for(PxU32 activeIndex = 0;
				activeIndex < activeTriangleCount;
				activeIndex++)
			{
				const PxU32 ti = useSurfaceTriangleBvh
					? triangleCandidates[activeIndex] * 3
					: triangleBounds[
						activeTriangles[activeIndex]].triangleOffset;
				if(useSurfaceTriangleBvh)
				{
					if(!triangleOverlapsQuery(
						ti, vertexMinimum, vertexMaximum))
						continue;
				}
				else
				{
					const AvbdSelfCollisionTriangleBounds& triangle =
						triangleBounds[activeTriangles[activeIndex]];
					if(triangle.minimum.y > vertexMaximum.y + r ||
						triangle.maximum.y < vertexMinimum.y - r ||
						triangle.minimum.z > vertexMaximum.z + r ||
						triangle.maximum.z < vertexMinimum.z - r)
						continue;
				}
				if(stats)
					stats->selfTriangleTests++;
				const PxU32 source0 =
					sb.compiled.surfaceTriangles[ti];
				const PxU32 source1 =
					sb.compiled.surfaceTriangles[ti + 1];
				const PxU32 source2 =
					sb.compiled.surfaceTriangles[ti + 2];
				const PxU32 lv0 =
					source0 - sb.compiled.particleStart;
				const PxU32 lv1 =
					source1 - sb.compiled.particleStart;
				const PxU32 lv2 =
					source2 - sb.compiled.particleStart;
				if(lv0 == li || lv1 == li || lv2 == li)
					continue;
				if(particles[gi].invMass <= 0.0f &&
					particles[source0].invMass <= 0.0f &&
					particles[source1].invMass <= 0.0f &&
					particles[source2].invMass <= 0.0f)
					continue;
				if(!targetStressAllowsTriangle(ti))
					continue;
				if(hasRestFilter)
				{
					if(sb.compiled.selfCollisionRestFilterCacheValid)
					{
						if(avbdIsSelfRestVertexTriangleFiltered(
							sb, li, ti / 3))
							continue;
					}
					else
					{
						const PxArray<PxVec3>& restPositions =
							sb.compiled.selfCollisionRestPositions;
						const AvbdClosestPointResult restClosest =
							avbdClosestPointOnTriangleOGC(
								restPositions[li],
								restPositions[lv0],
								restPositions[lv1],
								restPositions[lv2]);
						if(restClosest.distance <= filterDistance)
							continue;
					}
				}
				AvbdSweptTriangleEntry entry;
				if(avbdRotatingPointEnterExpandedDeformingTriangleFace(
						PxVec3(0.0f),
						particles[gi].initialPosition, pp,
						PxQuat(PxIdentity), PxQuat(PxIdentity),
						particles[source0].initialPosition,
						particles[source1].initialPosition,
						particles[source2].initialPosition,
						particles[source0].position,
						particles[source1].position,
						particles[source2].position,
						r, entry) &&
					entry.entryTime < bestEntryTime)
				{
					bestEntryTime = entry.entryTime;
					bestTriangleOffset = ti;
					bestEntry = entry;
				}
			}
			if(bestTriangleOffset != PX_MAX_U32)
			{
				const PxU32 source0 =
					sb.compiled.surfaceTriangles[
						bestTriangleOffset];
				const PxU32 source1 =
					sb.compiled.surfaceTriangles[
						bestTriangleOffset + 1];
				const PxU32 source2 =
					sb.compiled.surfaceTriangles[
						bestTriangleOffset + 2];
				PxVec3 contactNormal = -bestEntry.normal;
				const AvbdClosestPointResult initialClosest =
					avbdClosestPointOnTriangleOGC(
						particles[gi].initialPosition,
						particles[source0].initialPosition,
						particles[source1].initialPosition,
						particles[source2].initialPosition);
				if(contactNormal.dot(initialClosest.normal) < 0.0f)
					contactNormal = -contactNormal;
				const PxVec3 entry0 =
					particles[source0].initialPosition +
					(particles[source0].position -
					 particles[source0].initialPosition) *
						bestEntry.entryTime;
				const PxVec3 entry1 =
					particles[source1].initialPosition +
					(particles[source1].position -
					 particles[source1].initialPosition) *
						bestEntry.entryTime;
				const PxVec3 entry2 =
					particles[source2].initialPosition +
					(particles[source2].position -
					 particles[source2].initialPosition) *
						bestEntry.entryTime;
				AvbdSoftContactGeometry geometry;
				geometry.source = AvbdSoftContactSource(
					AvbdSoftContactSource::eSELF_SURFACE,
					softBodyIdx,
					avbdSoftTrianglePrimitiveKey(
						source0, source1, source2),
					avbdSoftTriangleFeatureKey(
						source0, source1, source2,
						AVBD_FEATURE_FACE, 0));
				geometry.particleIdx = gi;
				geometry.targetKind =
					AvbdSoftContactTargetKind::
						eDEFORMABLE_SURFACE;
				geometry.velocityOwner =
					AvbdVelocityObjectiveOwner::PositionAL;
				geometry.targetIndex = softBodyIdx;
				geometry.normal = contactNormal;
				geometry.projNormal = contactNormal;
				geometry.depth = 0.0f;
				geometry.margin = r;
				geometry.surfacePoint =
					entry0 * bestEntry.barycentric.x +
					entry1 * bestEntry.barycentric.y +
					entry2 * bestEntry.barycentric.z;
				geometry.surfaceParticleIndices[0] = source0;
				geometry.surfaceParticleIndices[1] = source1;
				geometry.surfaceParticleIndices[2] = source2;
				geometry.surfaceWeights[0] =
					bestEntry.barycentric.x;
				geometry.surfaceWeights[1] =
					bestEntry.barycentric.y;
				geometry.surfaceWeights[2] =
					bestEntry.barycentric.z;
				geometry.friction =
					PxMax(sb.material.dynamicFriction, 0.0f);
				avbdBuildSoftContactTangents(geometry);
				avbdAppendPreparedSoftContact(
					geometry,
					params.contactStiffness,
					params.contactStiffness * 10.0f,
					particles, contacts);
				continue;
			}
		}
		emittedFeatureKeys.clear();
		for(PxU32 activeIndex = 0;
			activeIndex < activeTriangleCount;
			activeIndex++)
		{
			const PxU32 ti = useSurfaceTriangleBvh
				? triangleCandidates[activeIndex] * 3
				: triangleBounds[
					activeTriangles[activeIndex]].triangleOffset;
			if(useSurfaceTriangleBvh)
			{
				if(!triangleOverlapsQuery(ti, pp, pp))
					continue;
			}
			else
			{
				const AvbdSelfCollisionTriangleBounds& triangle =
					triangleBounds[activeTriangles[activeIndex]];
				if(triangle.minimum.y > pp.y + r ||
					triangle.maximum.y < pp.y - r ||
					triangle.minimum.z > pp.z + r ||
					triangle.maximum.z < pp.z - r)
					continue;
			}
			if(stats)
				stats->selfTriangleTests++;

			const PxU32 source0 =
				sb.compiled.surfaceTriangles[ti];
			const PxU32 source1 =
				sb.compiled.surfaceTriangles[ti + 1];
			const PxU32 source2 =
				sb.compiled.surfaceTriangles[ti + 2];
			const PxU32 lv0 =
				source0 - sb.compiled.particleStart;
			const PxU32 lv1 =
				source1 - sb.compiled.particleStart;
			const PxU32 lv2 =
				source2 - sb.compiled.particleStart;

			// The OGC conservative proof excludes incident facets.  A
			// caller-requested rest-distance filter handles any wider
			// topological exclusion explicitly.
			if(lv0 == li || lv1 == li || lv2 == li)
				continue;
			if(particles[gi].invMass <= 0.0f &&
				particles[source0].invMass <= 0.0f &&
				particles[source1].invMass <= 0.0f &&
				particles[source2].invMass <= 0.0f)
				continue;
			if(!targetStressAllowsTriangle(ti))
				continue;
			if(hasRestFilter)
			{
				if(sb.compiled.selfCollisionRestFilterCacheValid)
				{
					if(avbdIsSelfRestVertexTriangleFiltered(
						sb, li, ti / 3))
						continue;
				}
				else
				{
					const PxArray<PxVec3>& restPositions =
						sb.compiled.selfCollisionRestPositions;
					const AvbdClosestPointResult restClosest =
						avbdClosestPointOnTriangleOGC(
							restPositions[li],
							restPositions[lv0],
							restPositions[lv1],
							restPositions[lv2]);
					if(restClosest.distance <= filterDistance)
						continue;
				}
			}

			const PxVec3& va = particles[source0].position;
			const PxVec3& vb = particles[source1].position;
			const PxVec3& vc = particles[source2].position;
			const AvbdClosestPointResult cp =
				avbdClosestPointOnTriangleOGC(pp, va, vb, vc);
			if(cp.distance >= r)
				continue;

			PxVec3 faceNormal = (vb - va).cross(vc - va);
			const PxReal faceNormalLength =
				faceNormal.magnitude();
			if(faceNormalLength < 1.0e-10f)
				continue;
			faceNormal *= 1.0f / faceNormalLength;
			PxVec3 contactNormal =
				cp.feature == AVBD_FEATURE_FACE
					? (cp.normal.dot(faceNormal) >= 0.0f
						? faceNormal : -faceNormal)
					: cp.normal;

			// Keep the normal on the penetration-free side recorded at the
			// beginning of this time step.  Choosing the current nearest side
			// after a crossing makes the normal flip every redetection and is
			// the source of the observed self-twitching.
			const AvbdClosestPointResult previousClosest =
				avbdClosestPointOnTriangleOGC(
					particles[gi].initialPosition,
					particles[source0].initialPosition,
					particles[source1].initialPosition,
					particles[source2].initialPosition);
			PxVec3 previousNormal = previousClosest.normal;
			if(previousNormal.magnitudeSquared() <= 1.0e-16f)
			{
				previousNormal =
					(particles[source1].initialPosition -
					 particles[source0].initialPosition).cross(
						particles[source2].initialPosition -
						particles[source0].initialPosition);
				if(previousNormal.magnitudeSquared() <= 1.0e-16f)
					continue;
				previousNormal.normalize();
			}
			if(contactNormal.dot(previousNormal) < 0.0f)
				contactNormal = -contactNormal;

			const AvbdActivationResult activation =
				avbdOGCActivationFull(
					cp.distance, r,
					params.contactStiffness, tau);
			if(activation.force <= 0.0f)
				continue;
			const PxU64 featureKey =
				avbdSoftTriangleFeatureKey(
					source0, source1, source2,
					cp.feature, cp.featureIndex);
			bool duplicateFeature = false;
			for(PxU32 emittedIndex = 0;
				emittedIndex < emittedFeatureKeys.size();
				emittedIndex++)
			{
				if(emittedFeatureKeys[emittedIndex] ==
					featureKey)
				{
					duplicateFeature = true;
					break;
				}
			}
			if(duplicateFeature)
				continue;
			emittedFeatureKeys.pushBack(featureKey);

			AvbdSoftContactGeometry geometry;
			geometry.source = AvbdSoftContactSource(
				AvbdSoftContactSource::eSELF_SURFACE,
				softBodyIdx,
				cp.feature == AVBD_FEATURE_FACE
					? avbdSoftTrianglePrimitiveKey(
						source0, source1, source2)
					: featureKey,
				featureKey);
			geometry.particleIdx = gi;
			geometry.targetKind =
				AvbdSoftContactTargetKind::
					eDEFORMABLE_SURFACE;
			geometry.velocityOwner =
				AvbdVelocityObjectiveOwner::PositionAL;
			geometry.targetIndex = softBodyIdx;
			geometry.normal = contactNormal;
			geometry.projNormal = contactNormal;
			geometry.depth = r - cp.distance;
			geometry.margin = r;
			geometry.surfacePoint = cp.point;
			geometry.surfaceParticleIndices[0] = source0;
			geometry.surfaceParticleIndices[1] = source1;
			geometry.surfaceParticleIndices[2] = source2;
			geometry.surfaceWeights[0] = cp.barycentric.x;
			geometry.surfaceWeights[1] = cp.barycentric.y;
			geometry.surfaceWeights[2] = cp.barycentric.z;
			geometry.friction =
				PxMax(sb.material.dynamicFriction, 0.0f);
			avbdBuildSoftContactTangents(geometry);
			avbdAppendPreparedSoftContact(
				geometry,
				params.contactStiffness,
				params.contactStiffness * 10.0f,
				particles, contacts);
		}
	}

	// Vertex-face alone does not preserve the topology of two crossing
	// triangle interiors: both endpoints of a cloth edge can remain outside
	// the opposing triangles while the edges pass through one another.
	// Complete the self-collision feature set with one barycentric edge-edge
	// objective for every non-adjacent edge pair in the contact shell.
	const bool useSurfaceEdgeBvh = usePreparedBvhWorkspace ||
		(avbdUseSurfaceEdgeBvh() &&
		 !sb.compiled.surfaceEdgeBvhNodes.empty());
	PX_ASSERT(!usePreparedBvhWorkspace ||
		(sb.compiled.surfaceEdges.empty() ||
		 !sb.compiled.surfaceEdgeBvhNodes.empty()));
	if(useSurfaceEdgeBvh && !usePreparedBvhWorkspace)
	{
		sb.compiled.refitSurfaceEdgeBvh(
			particles, sweepEnabled, workspace.selfEdgeBvhBounds);
		if(stats)
			stats->surfaceEdgeBvhRefitNodes +=
				sb.compiled.surfaceEdgeBvhNodes.size();
	}
	const PxArray<AvbdSurfaceBvhNodeBounds>& edgeBvhBounds =
		usePreparedBvhWorkspace
			? preparedBvhWorkspace->selfEdgeBvhBounds
			: workspace.selfEdgeBvhBounds;
	PX_ASSERT(!usePreparedBvhWorkspace ||
		edgeBvhBounds.size() == sb.compiled.surfaceEdgeBvhNodes.size());
	auto getEdgeBounds =
		[&](PxU32 edgeIndex, PxVec3& minimum, PxVec3& maximum)
	{
		const AvbdEdgeInfo& edge = sb.compiled.surfaceEdges[edgeIndex];
		minimum = particles[edge.p0].position.minimum(
			particles[edge.p1].position);
		maximum = particles[edge.p0].position.maximum(
			particles[edge.p1].position);
		if(sweepEnabled)
		{
			minimum = minimum.minimum(particles[edge.p0].initialPosition).
				minimum(particles[edge.p1].initialPosition);
			maximum = maximum.maximum(particles[edge.p0].initialPosition).
				maximum(particles[edge.p1].initialPosition);
		}
	};
	PxArray<AvbdSelfCollisionEdgeBounds>& edgeBounds =
		workspace.selfEdgeBounds;
	edgeBounds.clear();
	if(!useSurfaceEdgeBvh)
	{
		for(PxU32 edgeIndex = 0;
			edgeIndex < sb.compiled.surfaceEdges.size(); edgeIndex++)
		{
			const AvbdEdgeInfo& edge =
				sb.compiled.surfaceEdges[edgeIndex];
			if(edge.p0 >= sb.compiled.particleStart +
					sb.compiled.particleCount ||
				edge.p1 >= sb.compiled.particleStart +
					sb.compiled.particleCount)
				continue;
			AvbdSelfCollisionEdgeBounds bounds;
			bounds.edgeIndex = edgeIndex;
			getEdgeBounds(edgeIndex, bounds.minimum, bounds.maximum);
			edgeBounds.pushBack(bounds);
			if(stats)
				stats->selfEdgeBoundsBuilt++;
		}
	}
	if(!useSurfaceEdgeBvh)
		PxSort(
			edgeBounds.begin(), edgeBounds.size(),
			[](const AvbdSelfCollisionEdgeBounds& a,
			   const AvbdSelfCollisionEdgeBounds& b)
			{
				return a.minimum.x < b.minimum.x;
			});

	const PxReal edgeFeatureEpsilon = 1.0e-4f;
	const PxReal edgeDistanceEpsilon = 1.0e-8f;
	auto targetStressAllowsEdge =
		[&](PxU32 edge0, PxU32 edge1) -> bool
	{
		if(tetStressCoefficients->empty())
			return true;
		for(PxU32 triangleOffset = 0;
			triangleOffset + 2 <
				sb.compiled.surfaceTriangles.size();
			triangleOffset += 3)
		{
			const PxU32 v0 =
				sb.compiled.surfaceTriangles[triangleOffset];
			const PxU32 v1 =
				sb.compiled.surfaceTriangles[
					triangleOffset + 1];
			const PxU32 v2 =
				sb.compiled.surfaceTriangles[
					triangleOffset + 2];
			const bool has0 =
				v0 == edge0 || v1 == edge0 || v2 == edge0;
			const bool has1 =
				v0 == edge1 || v1 == edge1 || v2 == edge1;
			if(has0 && has1 &&
				!targetStressAllowsTriangle(triangleOffset))
				return false;
		}
		return true;
	};
	PxArray<PxU32>& edgeCandidates = workspace.selfEdgeCandidates;
	edgeCandidates.clear();
	const PxU32 outerEdgeCount = useSurfaceEdgeBvh
		? sb.compiled.surfaceEdges.size() : edgeBounds.size();
	const PxU32 clampedEdgeLoopBegin =
		PxMin(edgeLoopBegin, outerEdgeCount);
	const PxU32 clampedEdgeLoopEnd =
		PxMin(PxMax(edgeLoopEnd, clampedEdgeLoopBegin), outerEdgeCount);
	for(PxU32 outerEdgeIndex = clampedEdgeLoopBegin;
		outerEdgeIndex < clampedEdgeLoopEnd; outerEdgeIndex++)
	{
		const PxU32 sourceEdgeIndex = useSurfaceEdgeBvh
			? outerEdgeIndex : edgeBounds[outerEdgeIndex].edgeIndex;
		if(sourceEdgeIndex >= sb.compiled.surfaceEdges.size())
			continue;
		PxVec3 sourceMinimum, sourceMaximum;
		if(useSurfaceEdgeBvh)
		{
			getEdgeBounds(sourceEdgeIndex, sourceMinimum, sourceMaximum);
			sb.compiled.collectSurfaceEdgeBvhCandidates(
				sourceMinimum, sourceMaximum, r,
				edgeBvhBounds, edgeCandidates);
			if(stats)
				stats->surfaceEdgeBvhCandidateEdges += edgeCandidates.size();
		}
		else
		{
			sourceMinimum = edgeBounds[outerEdgeIndex].minimum;
			sourceMaximum = edgeBounds[outerEdgeIndex].maximum;
		}
		const PxU32 innerFirst = useSurfaceEdgeBvh
			? 0u : outerEdgeIndex + 1;
		const PxU32 innerEdgeCount = useSurfaceEdgeBvh
			? edgeCandidates.size() : edgeBounds.size();
		for(PxU32 innerEdgeIndex = innerFirst;
			innerEdgeIndex < innerEdgeCount; innerEdgeIndex++)
		{
			const PxU32 candidateEdgeIndex = useSurfaceEdgeBvh
				? edgeCandidates[innerEdgeIndex]
				: edgeBounds[innerEdgeIndex].edgeIndex;
			if(candidateEdgeIndex >= sb.compiled.surfaceEdges.size())
				continue;
			PxVec3 targetMinimum, targetMaximum;
			if(useSurfaceEdgeBvh)
			{
				if(candidateEdgeIndex <= sourceEdgeIndex)
					continue;
				getEdgeBounds(
					candidateEdgeIndex, targetMinimum, targetMaximum);
			}
			else
			{
				targetMinimum = edgeBounds[innerEdgeIndex].minimum;
				targetMaximum = edgeBounds[innerEdgeIndex].maximum;
				if(targetMinimum.x > sourceMaximum.x + r)
					break;
			}
			if(sourceMinimum.y > targetMaximum.y + r ||
				sourceMaximum.y < targetMinimum.y - r ||
				sourceMinimum.z > targetMaximum.z + r ||
				sourceMaximum.z < targetMinimum.z - r ||
				(sourceMinimum.x > targetMaximum.x + r ||
				 sourceMaximum.x < targetMinimum.x - r))
				continue;

			const PxU32 queryEdgeIndex =
				PxMin(sourceEdgeIndex, candidateEdgeIndex);
			const PxU32 targetEdgeIndex =
				PxMax(sourceEdgeIndex, candidateEdgeIndex);
			const AvbdEdgeInfo& queryEdge =
				sb.compiled.surfaceEdges[queryEdgeIndex];
			const AvbdEdgeInfo& targetEdge =
				sb.compiled.surfaceEdges[targetEdgeIndex];
			const PxU32 q0 = queryEdge.p0;
			const PxU32 q1 = queryEdge.p1;
			const PxU32 t0 = targetEdge.p0;
			const PxU32 t1 = targetEdge.p1;
			if(q0 == t0 || q0 == t1 ||
				q1 == t0 || q1 == t1)
				continue;

			const PxU32 lq0 = q0 - sb.compiled.particleStart;
			const PxU32 lq1 = q1 - sb.compiled.particleStart;
			const PxU32 lt0 = t0 - sb.compiled.particleStart;
			const PxU32 lt1 = t1 - sb.compiled.particleStart;
			if(particles[q0].invMass <= 0.0f &&
				particles[q1].invMass <= 0.0f &&
				particles[t0].invMass <= 0.0f &&
				particles[t1].invMass <= 0.0f)
				continue;
			if(!targetStressAllowsEdge(t0, t1))
				continue;

			if(hasRestFilter)
			{
				const PxArray<PxVec3>& restPositions =
					sb.compiled.selfCollisionRestPositions;
				PxReal restQueryWeight = 0.0f;
				PxReal restTargetWeight = 0.0f;
				PxVec3 restQueryClosest, restTargetClosest;
				avbdClosestPointsOnSegments(
					restPositions[lq0], restPositions[lq1],
					restPositions[lt0], restPositions[lt1],
					restQueryWeight, restTargetWeight,
					restQueryClosest, restTargetClosest);
				if((restQueryClosest - restTargetClosest).
						magnitude() <= filterDistance)
					continue;
			}

			PxReal previousQueryWeight1 = 0.0f;
			PxReal previousTargetWeight1 = 0.0f;
			PxVec3 previousQueryClosest;
			PxVec3 previousTargetClosest;
			avbdClosestPointsOnSegments(
				particles[q0].initialPosition,
				particles[q1].initialPosition,
				particles[t0].initialPosition,
				particles[t1].initialPosition,
				previousQueryWeight1,
				previousTargetWeight1,
				previousQueryClosest,
				previousTargetClosest);
			const PxVec3 previousDelta =
				previousQueryClosest - previousTargetClosest;

			auto stabilizeEdgeNormal =
				[&](PxVec3 contactNormal) -> PxVec3
			{
				if(previousDelta.magnitudeSquared() >
					edgeDistanceEpsilon *
						edgeDistanceEpsilon)
				{
					if(contactNormal.dot(previousDelta) < 0.0f)
						contactNormal = -contactNormal;
				}
				else
				{
					const PxVec3 previousCross =
						(particles[q1].initialPosition -
						 particles[q0].initialPosition).cross(
							particles[t1].initialPosition -
							particles[t0].initialPosition);
					if(previousCross.magnitudeSquared() >
							edgeDistanceEpsilon *
								edgeDistanceEpsilon &&
						contactNormal.dot(previousCross) < 0.0f)
						contactNormal = -contactNormal;
				}
				return contactNormal;
			};

			auto appendEdgeContact =
				[&](PxReal queryWeight1,
					PxReal targetWeight1,
					const PxVec3& contactNormal,
					PxReal depth,
					const PxVec3& targetClosest)
			{
				AvbdSoftContactGeometry geometry;
				geometry.source = AvbdSoftContactSource(
					AvbdSoftContactSource::eSELF_SURFACE,
					softBodyIdx,
					avbdGetRigidSoftFeatureKey(
						0x53454530u, q0, q1, 0u, 0u),
					avbdGetRigidSoftFeatureKey(
						0x53454531u, q0, q1, t0, t1));
				geometry.particleIdx =
					particles[q0].invMass > 0.0f ? q0 :
					(particles[q1].invMass > 0.0f ? q1 :
					(particles[t0].invMass > 0.0f ? t0 : t1));
				geometry.queryParticleIndices[0] = q0;
				geometry.queryParticleIndices[1] = q1;
				geometry.queryWeights[0] = 1.0f - queryWeight1;
				geometry.queryWeights[1] = queryWeight1;
				geometry.targetKind =
					AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE;
				geometry.velocityOwner =
					AvbdVelocityObjectiveOwner::PositionAL;
				geometry.targetIndex = softBodyIdx;
				geometry.surfaceParticleIndices[0] = t0;
				geometry.surfaceParticleIndices[1] = t1;
				geometry.surfaceWeights[0] = 1.0f - targetWeight1;
				geometry.surfaceWeights[1] = targetWeight1;
				geometry.normal = contactNormal;
				geometry.projNormal = contactNormal;
				geometry.depth = depth;
				geometry.margin = r;
				geometry.surfacePoint = targetClosest;
				geometry.friction =
					PxMax(sb.material.dynamicFriction, 0.0f);
				avbdBuildSoftContactTangents(geometry);
				avbdAppendPreparedSoftContact(
					geometry,
					params.contactStiffness,
					params.contactStiffness * 10.0f,
					particles, contacts);
			};

			if(sweepEnabled)
			{
				AvbdSweptConvexEdgeEntry entry;
				if(avbdDeformingSegmentsEnterExpandedInteriors(
						particles[q0].initialPosition,
						particles[q1].initialPosition,
						particles[q0].position,
						particles[q1].position,
						particles[t0].initialPosition,
						particles[t1].initialPosition,
						particles[t0].position,
						particles[t1].position,
						r, entry))
				{
					const PxVec3 target0AtEntry =
						particles[t0].initialPosition +
						(particles[t0].position -
						 particles[t0].initialPosition) *
							entry.entryTime;
					const PxVec3 target1AtEntry =
						particles[t1].initialPosition +
						(particles[t1].position -
						 particles[t1].initialPosition) *
							entry.entryTime;
					const PxVec3 contactNormal =
						stabilizeEdgeNormal(entry.normal);
					appendEdgeContact(
						entry.softWeight1,
						entry.rigidWeight1,
						contactNormal,
						0.0f,
						target0AtEntry *
							(1.0f - entry.rigidWeight1) +
						target1AtEntry * entry.rigidWeight1);
					continue;
				}
			}

			PxReal queryWeight1 = 0.0f;
			PxReal targetWeight1 = 0.0f;
			PxVec3 queryClosest, targetClosest;
			avbdClosestPointsOnSegments(
				particles[q0].position,
				particles[q1].position,
				particles[t0].position,
				particles[t1].position,
				queryWeight1, targetWeight1,
				queryClosest, targetClosest);
			if(queryWeight1 <= edgeFeatureEpsilon ||
				queryWeight1 >= 1.0f - edgeFeatureEpsilon ||
				targetWeight1 <= edgeFeatureEpsilon ||
				targetWeight1 >= 1.0f - edgeFeatureEpsilon)
				continue;
			const PxVec3 delta = queryClosest - targetClosest;
			const PxReal distance = delta.magnitude();
			if(distance >= r)
				continue;

			PxVec3 contactNormal;
			if(distance > edgeDistanceEpsilon)
				contactNormal = delta * (1.0f / distance);
			else
			{
				contactNormal =
					(particles[q1].position -
					 particles[q0].position).cross(
						particles[t1].position -
						particles[t0].position);
				if(contactNormal.magnitudeSquared() <=
					edgeDistanceEpsilon *
						edgeDistanceEpsilon)
					continue;
				contactNormal.normalize();
			}
			contactNormal = stabilizeEdgeNormal(contactNormal);
			const AvbdActivationResult activation =
				avbdOGCActivationFull(
					distance, r,
					params.contactStiffness, tau);
			if(activation.force <= 0.0f)
				continue;
			appendEdgeContact(
				queryWeight1, targetWeight1,
				contactNormal, r - distance,
				targetClosest);
		}
	}
}

// A self-contact range is only safe after one parent has produced immutable
// BVH bounds for both feature families.  The retained linear sweep has a
// cross-range active-triangle cursor and deliberately remains serial.
PX_FORCE_INLINE bool avbdCanUseSelfCollisionOGCBvhRanges(
	const AvbdSoftBody& sb)
{
	return avbdUseSurfaceTriangleBvh() &&
		!sb.compiled.surfaceTriangleBvhNodes.empty() &&
		(sb.compiled.surfaceEdges.empty() ||
			(avbdUseSurfaceEdgeBvh() &&
			 !sb.compiled.surfaceEdgeBvhNodes.empty()));
}

// This parent-only preparation performs the mutable work exactly once.  The
// empty vertex/edge ranges intentionally avoid creating contacts while the
// detector computes stress coefficients and refits the two compiled BVHs.
// Range leaves below may then consume these arrays read-only from independent
// worker workspaces.
inline bool avbdPrepareSelfCollisionOGCBvhRanges(
	const AvbdSoftParticle* particles,
	const AvbdSoftBody& sb,
	PxU32 softBodyIdx,
	const PxArray<PxArray<PxU32> >& adj,
	const AvbdOGCParams& params,
	AvbdSoftContactWorkspace& parentWorkspace,
	AvbdSoftCollisionStats* stats = NULL)
{
	if(!avbdCanUseSelfCollisionOGCBvhRanges(sb))
		return false;
	PxArray<AvbdSoftContact> noContacts;
	avbdDetectSelfCollisionOGC(
		particles, sb, softBodyIdx, adj, noContacts, params, stats,
		&parentWorkspace, NULL, 0, 0, 0, 0);
	return true;
}

// Run a contiguous VF and/or EE outer range against the immutable parent
// snapshot.  Callers that split a body must stable-merge every VF range before
// every EE range, matching the serial detector's canonical feature order.
inline void avbdDetectSelfCollisionOGCBvhRange(
	const AvbdSoftParticle* particles,
	const AvbdSoftBody& sb,
	PxU32 softBodyIdx,
	const PxArray<PxArray<PxU32> >& adj,
	const AvbdSoftContactWorkspace& parentWorkspace,
	AvbdSoftContactWorkspace& rangeWorkspace,
	PxU32 vertexLoopBegin,
	PxU32 vertexLoopEnd,
	PxU32 edgeLoopBegin,
	PxU32 edgeLoopEnd,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params = AvbdOGCParams(),
	AvbdSoftCollisionStats* stats = NULL)
{
	PX_ASSERT(&parentWorkspace != &rangeWorkspace);
	PX_ASSERT(avbdCanUseSelfCollisionOGCBvhRanges(sb));
	avbdDetectSelfCollisionOGC(
		particles, sb, softBodyIdx, adj, contacts, params, stats,
		&rangeWorkspace, &parentWorkspace,
		vertexLoopBegin, vertexLoopEnd, edgeLoopBegin, edgeLoopEnd);
}

// =============================================================================
// Convenience: detect all OGC contacts (ground + soft-rigid + soft-soft + self)
// =============================================================================

PX_FORCE_INLINE bool avbdGetSoftContactDetectionQueryPoint(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftParticle* detectionParticles,
	PxVec3& point)
{
	point = PxVec3(0.0f);
	if(geometry.queryParticleIndices[0] != PX_MAX_U32)
	{
		for(PxU32 i = 0; i < 3; ++i)
		{
			const PxU32 particleIndex = geometry.queryParticleIndices[i];
			if(particleIndex == PX_MAX_U32)
				break;
			point += detectionParticles[particleIndex].position *
				geometry.queryWeights[i];
		}
		return point.isFinite();
	}
	const PxU32 particleIndex =
		geometry.collisionFeatureParticleIdx != PX_MAX_U32
			? geometry.collisionFeatureParticleIdx
			: geometry.particleIdx;
	if(particleIndex == PX_MAX_U32)
		return false;
	point = detectionParticles[particleIndex].position;
	return point.isFinite();
}

PX_FORCE_INLINE bool avbdGetSoftContactDetectionSurfacePoint(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftParticle* detectionParticles,
	PxVec3& point)
{
	point = PxVec3(0.0f);
	if(!geometry.hasDeformableSurfaceTarget())
		return false;
	for(PxU32 i = 0; i < 3; ++i)
	{
		const PxU32 particleIndex = geometry.surfaceParticleIndices[i];
		if(particleIndex == PX_MAX_U32)
			break;
		point += detectionParticles[particleIndex].position *
			geometry.surfaceWeights[i];
	}
	return point.isFinite();
}

PX_FORCE_INLINE bool avbdCanTransferSoftContactFrictionAnchors(
	const AvbdSoftContactGeometry& previousGeometry,
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftParticle* detectionParticles)
{
	if(!previousGeometry.hasDeformableSurfaceTarget() ||
		!geometry.hasDeformableSurfaceTarget())
		return true;

	PxVec3 previousQueryPoint, queryPoint;
	PxVec3 previousSurfacePoint, surfacePoint;
	if(!avbdGetSoftContactDetectionQueryPoint(
			previousGeometry, detectionParticles, previousQueryPoint) ||
		!avbdGetSoftContactDetectionQueryPoint(
			geometry, detectionParticles, queryPoint) ||
		!avbdGetSoftContactDetectionSurfacePoint(
			previousGeometry, detectionParticles, previousSurfacePoint) ||
		!avbdGetSoftContactDetectionSurfacePoint(
			geometry, detectionParticles, surfacePoint))
		return false;

	// A collision feature can remain identical while the closest point moves
	// along a long edge or across a face. The normal multiplier still belongs
	// to that feature, but carrying its static-friction anchor to the new
	// material points creates an artificial tangential tether. One contact-shell
	// radius is the largest migration for which the old friction patch remains
	// local. Detection-domain supports are used deliberately: public Volume
	// contacts have already-expanded simulation points in queryPoint/targetPoint,
	// while these legacy support arrays retain the authoritative proxy feature.
	const PxReal anchorRadius = PxMax(
		PxMax(previousGeometry.margin, geometry.margin), 1.0e-4f);
	const PxReal anchorRadiusSq = anchorRadius * anchorRadius;
	return (previousQueryPoint - queryPoint).magnitudeSquared() <=
			anchorRadiusSq &&
		(previousSurfacePoint - surfacePoint).magnitudeSquared() <=
			anchorRadiusSq;
}

PX_FORCE_INLINE bool avbdCanTransferSoftContactNormalState(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftParticle* particles)
{
	const PxVec3 queryPoint =
		avbdGetSoftContactQueryPoint(geometry, particles);
	const PxVec3 surfacePoint =
		avbdGetSoftContactSurfacePoint(geometry, particles);
	if(!queryPoint.isFinite() || !surfacePoint.isFinite() ||
		!geometry.normal.isFinite())
		return false;

	const PxReal constraint = avbdEvaluateSoftContactNormalConstraint(
		geometry, queryPoint, surfacePoint);
	if(!PxIsFinite(constraint))
		return false;

	// Contact detection deliberately retains rows in the proximity shell.
	// A unilateral row that has separated inside that shell is no longer the
	// same active normal objective, even when its feature identity is stable.
	// Carrying its negative multiplier and elevated AL penalty across the
	// flight phase preloads the next impact and can freeze or launch the body.
	// Keep a tiny tolerance at the boundary to preserve useful warm-starting
	// for a genuinely active resting contact.
	const PxReal activeTolerance = PxMax(
		1.0e-5f, 1.0e-4f * PxMax(geometry.margin, 0.0f));
	return constraint <= activeTolerance;
}

PX_FORCE_INLINE void avbdTransferSoftContactState(
	const AvbdSoftContact* previousContacts, PxU32 numPreviousContacts,
	const AvbdSoftParticle* particles,
	PxArray<AvbdSoftContact>& contacts,
	AvbdSoftContactWorkspace* persistentWorkspace = NULL)
{
	const PxReal normalMatch = 0.8f;
	const PxReal pointMatchSq = 0.05f * 0.05f;
	AvbdSoftContactWorkspace localWorkspace;
	AvbdSoftContactWorkspace& workspace =
		persistentWorkspace ? *persistentWorkspace : localWorkspace;
	workspace.resizePreviousUsed(numPreviousContacts);
	PxArray<PxU8>& previousUsed = workspace.previousUsed;
	for(PxU32 oldIdx = 0; oldIdx < numPreviousContacts; ++oldIdx)
		previousUsed[oldIdx] = 0;
	for(PxU32 contactIdx = 0; contactIdx < contacts.size(); ++contactIdx)
	{
		AvbdSoftContact& contact = contacts[contactIdx];
		const AvbdSoftContactGeometry& geometry = contact.geometry;
		AvbdSoftContactAugmentedState& state = contact.state;
		avbdInitializeSoftContactAnchors(
			geometry, state, particles);

		const AvbdSoftContact* best = NULL;
		PxU32 bestIdx = PX_MAX_U32;
		PxReal bestDistanceSq = PX_MAX_F32;
		for(PxU32 oldIdx = 0; oldIdx < numPreviousContacts; ++oldIdx)
		{
			if(previousUsed[oldIdx])
				continue;
			const AvbdSoftContact& old = previousContacts[oldIdx];
			const AvbdSoftContactGeometry& oldGeometry = old.geometry;
			const PxU32 oldFeatureParticle =
				oldGeometry.collisionFeatureParticleIdx != PX_MAX_U32
					? oldGeometry.collisionFeatureParticleIdx
					: oldGeometry.particleIdx;
			const PxU32 newFeatureParticle =
				geometry.collisionFeatureParticleIdx != PX_MAX_U32
					? geometry.collisionFeatureParticleIdx
					: geometry.particleIdx;
			if(oldFeatureParticle != newFeatureParticle ||
				oldGeometry.normal.dot(geometry.normal) < normalMatch)
				continue;

			const PxReal distanceSq =
				(oldGeometry.surfacePoint -
				 geometry.surfacePoint).magnitudeSquared();
			if(geometry.source.isValid() || oldGeometry.source.isValid())
			{
				if(!geometry.source.isValid() ||
					!oldGeometry.source.isValid() ||
					!(geometry.source == oldGeometry.source))
					continue;
			}
			else
			{
				// Compatibility path for manually authored legacy contacts.
				if(oldGeometry.targetKind != geometry.targetKind ||
					oldGeometry.targetIndex != geometry.targetIndex)
					continue;
				if(!geometry.hasWorldStaticTarget() &&
					distanceSq > pointMatchSq)
					continue;
			}
			if(distanceSq < bestDistanceSq)
			{
				best = &old;
				bestIdx = oldIdx;
				bestDistanceSq = distanceSq;
			}
		}
		if(!best)
			continue;
		if(!avbdCanTransferSoftContactNormalState(geometry, particles))
			continue;
		previousUsed[bestIdx] = 1;

		const PxReal dualDecay = 0.99f;
		const PxReal penaltyDecay = 0.999f;
		const AvbdSoftContactGeometry& bestGeometry = best->geometry;
		const AvbdSoftContactAugmentedState& bestState = best->state;
		state.alLambda = bestState.alLambda * dualDecay;
		state.k = PxClamp(
			bestState.k * penaltyDecay, state.k, state.ke);
		state.depenetrationConstraintOffset =
			bestState.depenetrationConstraintOffset;
		state.depenetrationLimitInitialized =
			bestState.depenetrationLimitInitialized;
		if(bestState.frictionStick &&
			!avbdCanTransferSoftContactFrictionAnchors(
				bestGeometry, geometry, particles))
			continue;

		state.penTangent[0] = PxClamp(
			bestState.penTangent[0] * penaltyDecay,
			1000.0f, state.ke);
		state.penTangent[1] = PxClamp(
			bestState.penTangent[1] * penaltyDecay,
			1000.0f, state.ke);

		const PxVec3 oldTangentForce =
			bestGeometry.tangent1 * bestState.alLambdaTangent[0] +
			bestGeometry.tangent2 * bestState.alLambdaTangent[1];
		state.alLambdaTangent[0] =
			oldTangentForce.dot(geometry.tangent1) * dualDecay;
		state.alLambdaTangent[1] =
			oldTangentForce.dot(geometry.tangent2) * dualDecay;
		state.frictionStick = bestState.frictionStick;
		if(bestState.frictionStick)
		{
			state.particlePointPrev = bestState.particlePointPrev;
			state.surfacePointPrev = bestState.surfacePointPrev;
		}
	}
}

// Per-body self-collision adjacency array type
typedef PxArray<PxArray<PxU32> > AvbdSelfCollisionAdjacency;

inline void avbdBuildSoftContactRedetectionPhasePlan(
	AvbdSoftContactWorkspace& workspace,
	PxU32 numWorldPlanes, bool includeLegacyGround,
	PxU32 numRigidBoxes, PxU32 numRigidSpheres,
	PxU32 numRigidCapsules, PxU32 numRigidConvexes,
	PxU32 numRigidTriangleSurfaces, PxU32 numSoftBodies,
	const AvbdSelfCollisionAdjacency* perBodyAdj, PxU32 numAdj,
	const PxU8* selfCollisionEnabled)
{
	workspace.beginRedetectionPhasePlan();
	if(numWorldPlanes > 0)
	{
		workspace.appendRedetectionPhasePlan(
			AvbdSoftContactRedetectionPhase::eWORLD_PLANES,
			0, numWorldPlanes);
	}
	else if(includeLegacyGround)
	{
		workspace.appendRedetectionPhasePlan(
			AvbdSoftContactRedetectionPhase::eLEGACY_GROUND, 0, 1);
	}
	if(numRigidBoxes > 0)
		workspace.appendRedetectionPhasePlan(
			AvbdSoftContactRedetectionPhase::eRIGID_BOXES,
			0, numRigidBoxes);
	if(numRigidSpheres > 0)
		workspace.appendRedetectionPhasePlan(
			AvbdSoftContactRedetectionPhase::eRIGID_SPHERES,
			0, numRigidSpheres);
	if(numRigidCapsules > 0)
		workspace.appendRedetectionPhasePlan(
			AvbdSoftContactRedetectionPhase::eRIGID_CAPSULES,
			0, numRigidCapsules);
	if(numRigidConvexes > 0)
		workspace.appendRedetectionPhasePlan(
			AvbdSoftContactRedetectionPhase::eRIGID_CONVEXES,
			0, numRigidConvexes);
	if(numRigidTriangleSurfaces > 0)
		workspace.appendRedetectionPhasePlan(
			AvbdSoftContactRedetectionPhase::eRIGID_TRIANGLE_SURFACES,
			0, numRigidTriangleSurfaces);
	if(numSoftBodies > 1)
		workspace.appendRedetectionPhasePlan(
			AvbdSoftContactRedetectionPhase::eSOFT_SOFT,
			0, numSoftBodies);
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
	{
		if(bodyIndex < numAdj && perBodyAdj &&
			(!selfCollisionEnabled || selfCollisionEnabled[bodyIndex]))
		{
			workspace.appendRedetectionPhasePlan(
				AvbdSoftContactRedetectionPhase::eSELF_BODY,
				bodyIndex, bodyIndex + 1);
		}
	}
	if(avbdValidateRedetectionPhasePlan())
		PX_ASSERT(workspace.validateRedetectionPhasePlan());
}

// Parent-owned boundaries for one complete contact-redetection transaction.
// Detection leaves may populate private streams between these calls, but only
// this parent is allowed to snapshot prior state, mutate the canonical stream
// and transfer persistent contact state. This is deliberately independent of
// any particular detection source so a future Scene task fan-in can preserve
// the serial stream without exposing workspace mutation to workers.
inline void avbdBeginSoftContactRedetection(
	PxArray<AvbdSoftContact>& contacts,
	AvbdSoftContactWorkspace& workspace,
	AvbdSoftCollisionStats* stats = NULL)
{
	workspace.copyPreviousContacts(contacts);
	contacts.clear();
	workspace.redetectionOutputCapacityBefore = contacts.capacity();
	if(stats)
		stats->detectionCalls++;
}

inline void avbdCompleteSoftContactRedetection(
	AvbdSoftParticle* particles,
	PxArray<AvbdSoftContact>& contacts,
	AvbdSoftContactWorkspace& workspace)
{
	workspace.recordOutputCapacityGrowth(
		workspace.redetectionOutputCapacityBefore, contacts.capacity());
	workspace.recordOutputWatermark(contacts.size(), contacts.capacity());
	avbdTransferSoftContactState(
		workspace.previousContacts.begin(), workspace.previousContacts.size(),
		particles, contacts, &workspace);
	// A later outer iteration observes mutated primal positions, so it must not
	// reuse the post-prediction bounds prepared for this single redetection.
	workspace.invalidateSoftBodyBounds();
}

inline void avbdDetectAllOGCContacts(
	AvbdSoftParticle* particles, PxU32 numParticles,
	AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdRigidBox* rigidBoxes, PxU32 numRigidBoxes,
	const AvbdSelfCollisionAdjacency* perBodyAdj, PxU32 numAdj,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params = AvbdOGCParams(),
	PxReal groundY = 0.0f,
	AvbdSoftCollisionStats* stats = NULL,
	AvbdSoftContactWorkspace* persistentWorkspace = NULL,
	const AvbdWorldPlane* worldPlanes = NULL,
	PxU32 numWorldPlanes = 0,
	bool includeLegacyGround = true,
	const PxU8* selfCollisionEnabled = NULL,
	const AvbdRigidSphere* rigidSpheres = NULL,
	PxU32 numRigidSpheres = 0,
	const AvbdRigidCapsule* rigidCapsules = NULL,
	PxU32 numRigidCapsules = 0,
	const AvbdRigidConvex* rigidConvexes = NULL,
	PxU32 numRigidConvexes = 0,
	const AvbdRigidTriangleSurface* rigidTriangleSurfaces = NULL,
	PxU32 numRigidTriangleSurfaces = 0)
{
	AvbdSoftContactWorkspace localWorkspace;
	AvbdSoftContactWorkspace& workspace =
		persistentWorkspace ? *persistentWorkspace : localWorkspace;
	PxArray<AvbdSoftContact>& previousContacts = workspace.previousContacts;
	// Keep the authoritative serial transaction local.  The begin/complete
	// helpers are ownership seams for Scene task fan-in; routing this path
	// through them prevents the compiler from seeing the complete serial
	// contact epoch and charges task-graph structure to every redetection.
	workspace.copyPreviousContacts(contacts);
	contacts.clear();
	const PxU32 outputCapacityBefore = contacts.capacity();
	if(stats)
		stats->detectionCalls++;

	// Ground
	const PxU32 groundStart = contacts.size();
	if(numWorldPlanes > 0 && worldPlanes)
	{
		avbdDetectSoftWorldPlaneContacts(
			particles, numParticles,
			worldPlanes, numWorldPlanes,
			contacts, params.contactRadius,
			softBodies, numSoftBodies);
	}
	else if(includeLegacyGround)
	{
		avbdDetectSoftGroundContacts(
			particles, numParticles, contacts,
			groundY, params.contactRadius, params.friction,
			softBodies, numSoftBodies);
	}
	if(stats)
		stats->generatedGroundContacts += contacts.size() - groundStart;

	// Path 2: Rigid-soft SDF
	if (numRigidBoxes > 0)
	{
		const PxU32 rigidStart = contacts.size();
		if(stats)
			stats->rigidParticleBoxTests += PxU64(numParticles) * numRigidBoxes;
		avbdDetectSoftRigidSDF(particles, numParticles,
		                       rigidBoxes, numRigidBoxes,
		                       contacts, params.contactRadius,
		                       previousContacts.begin(),
		                       previousContacts.size(),
		                       softBodies, numSoftBodies);
		avbdDetectSoftRigidSweptSDF(
			particles, numParticles,
			rigidBoxes, numRigidBoxes,
			contacts, params.contactRadius,
			softBodies, numSoftBodies);
		avbdDetectSoftRigidOGCFeatures(
			particles, numParticles,
			rigidBoxes, numRigidBoxes,
			softBodies, numSoftBodies,
			contacts, params.contactRadius);
		if(stats)
			stats->generatedRigidContacts += contacts.size() - rigidStart;
	}
	if(numRigidSpheres > 0 && rigidSpheres)
	{
		const PxU32 rigidStart = contacts.size();
		if(stats)
			stats->rigidParticleSphereTests +=
				PxU64(numParticles) * numRigidSpheres;
		avbdDetectSoftRigidSphereSDF(
			particles, numParticles,
			rigidSpheres, numRigidSpheres,
			contacts, params.contactRadius,
			softBodies, numSoftBodies);
		avbdDetectSoftRigidSphereSweptSDF(
			particles, numParticles,
			rigidSpheres, numRigidSpheres,
			contacts, params.contactRadius,
			softBodies, numSoftBodies);
		avbdDetectSoftRigidSphereSweptOGCFeatures(
			particles, numParticles,
			rigidSpheres, numRigidSpheres,
			softBodies, numSoftBodies,
			contacts, params.contactRadius,
			&workspace.rigidConvexForwardOwnerScratch);
		avbdDetectSoftRigidSphereOGCFeatures(
			particles, numParticles,
			rigidSpheres, numRigidSpheres,
			softBodies, numSoftBodies,
			contacts, params.contactRadius);
		if(stats)
			stats->generatedRigidContacts +=
				contacts.size() - rigidStart;
	}
	if(numRigidCapsules > 0 && rigidCapsules)
	{
		const PxU32 rigidStart = contacts.size();
		if(stats)
			stats->rigidParticleCapsuleTests +=
				PxU64(numParticles) * numRigidCapsules;
		avbdDetectSoftRigidCapsuleSDF(
			particles, numParticles,
			rigidCapsules, numRigidCapsules,
			contacts, params.contactRadius,
			softBodies, numSoftBodies);
		avbdDetectSoftRigidCapsuleSweptSDF(
			particles, numParticles,
			rigidCapsules, numRigidCapsules,
			contacts, params.contactRadius,
			softBodies, numSoftBodies);
		avbdDetectSoftRigidCapsuleSweptOGCFeatures(
			particles, numParticles,
			rigidCapsules, numRigidCapsules,
			softBodies, numSoftBodies,
			contacts, params.contactRadius,
			&workspace.rigidConvexForwardOwnerScratch);
		avbdDetectSoftRigidCapsuleOGCFeatures(
			particles, numParticles,
			rigidCapsules, numRigidCapsules,
			softBodies, numSoftBodies,
			contacts, params.contactRadius);
		if(stats)
			stats->generatedRigidContacts +=
				contacts.size() - rigidStart;
	}
	if(numRigidConvexes > 0 && rigidConvexes)
	{
		const PxU32 rigidStart = contacts.size();
		if(stats)
			stats->rigidParticleConvexTests +=
				PxU64(numParticles) * numRigidConvexes;
		avbdDetectSoftRigidConvexSDF(
			particles, numParticles,
			rigidConvexes, numRigidConvexes,
			contacts, params.contactRadius,
			softBodies, numSoftBodies);
		avbdDetectSoftRigidConvexSweptSDF(
			particles, numParticles,
			rigidConvexes, numRigidConvexes,
			contacts, params.contactRadius,
			softBodies, numSoftBodies);
		avbdDetectSoftRigidConvexSweptOGCFeatures(
			particles, numParticles,
			rigidConvexes, numRigidConvexes,
			softBodies, numSoftBodies,
			contacts, params.contactRadius,
			&workspace.rigidConvexForwardOwnerScratch);
		avbdDetectSoftRigidConvexOGCFeatures(
			particles, numParticles,
			rigidConvexes, numRigidConvexes,
			softBodies, numSoftBodies,
			contacts, params.contactRadius);
		if(stats)
			stats->generatedRigidContacts +=
				contacts.size() - rigidStart;
	}
	if(numRigidTriangleSurfaces > 0 && rigidTriangleSurfaces)
	{
		const PxU32 rigidStart = contacts.size();
		if(stats)
			stats->rigidParticleTriangleSurfaceTests +=
				PxU64(numParticles) *
					numRigidTriangleSurfaces;
		avbdDetectSoftRigidTriangleSurface(
			particles, numParticles,
			rigidTriangleSurfaces,
			numRigidTriangleSurfaces,
			contacts, params.contactRadius,
			softBodies, numSoftBodies, stats);
		avbdDetectSoftRigidTriangleSurfaceSwept(
			particles, numParticles,
			rigidTriangleSurfaces,
			numRigidTriangleSurfaces,
			contacts, params.contactRadius,
			softBodies, numSoftBodies, stats);
		avbdDetectSoftRigidTriangleSurfaceSweptOGCFeatures(
			particles, numParticles,
			rigidTriangleSurfaces,
			numRigidTriangleSurfaces,
			softBodies, numSoftBodies,
			contacts, params.contactRadius, stats, NULL,
			&workspace.rigidTriangleSurfaceForwardOwnerScratch);
		avbdDetectSoftRigidTriangleSurfaceOGCFeatures(
			particles, numParticles,
			rigidTriangleSurfaces,
			numRigidTriangleSurfaces,
			softBodies, numSoftBodies,
			contacts, params.contactRadius, stats);
		if(stats)
			stats->generatedRigidContacts +=
				contacts.size() - rigidStart;
	}

	// Path 3: Soft-soft OGC simplified
	if (numSoftBodies > 1)
	{
		const PxU32 softStart = contacts.size();
		avbdDetectSoftSoftOGC(particles, numParticles,
		                      softBodies, numSoftBodies,
		                      contacts, params, stats, &workspace);
		if(stats)
			stats->generatedSoftContacts += contacts.size() - softStart;
	}

	// Path 4: Self-collision OGC full
	for (PxU32 si = 0; si < numSoftBodies; si++)
	{
		if (si < numAdj && perBodyAdj &&
			(!selfCollisionEnabled || selfCollisionEnabled[si]))
		{
			const PxU32 selfStart = contacts.size();
			avbdDetectSelfCollisionOGC(particles, softBodies[si], si,
			                           perBodyAdj[si], contacts, params, stats,
			                           &workspace);
			if(stats)
				stats->generatedSelfContacts += contacts.size() - selfStart;
		}
	}

	workspace.recordOutputCapacityGrowth(
		outputCapacityBefore, contacts.capacity());
	workspace.recordOutputWatermark(contacts.size(), contacts.capacity());
	avbdTransferSoftContactState(
		previousContacts.begin(), previousContacts.size(),
		particles, contacts, &workspace);
	// A later outer iteration observes mutated primal positions, so it must not
	// reuse the post-prediction bounds prepared for this single redetection.
	workspace.invalidateSoftBodyBounds();
}

// Build all per-body self-collision adjacencies
inline void avbdBuildAllSelfCollisionAdjacencies(
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSelfCollisionAdjacency>& outAdj)
{
	outAdj.resize(numSoftBodies);
	for (PxU32 si = 0; si < numSoftBodies; si++)
		avbdBuildSelfCollisionAdjacency(softBodies[si], outAdj[si]);
}

// =============================================================================
// Mesh generators
// =============================================================================

inline void avbdGenerateCubeTets(
	PxVec3 center, PxReal halfSize,
	PxArray<PxVec3>& outVerts,
	PxArray<PxU32>& outTets)
{
	PxReal h = halfSize;
	outVerts.clear();
	outVerts.pushBack(center + PxVec3(-h, -h, -h));
	outVerts.pushBack(center + PxVec3( h, -h, -h));
	outVerts.pushBack(center + PxVec3( h,  h, -h));
	outVerts.pushBack(center + PxVec3(-h,  h, -h));
	outVerts.pushBack(center + PxVec3(-h, -h,  h));
	outVerts.pushBack(center + PxVec3( h, -h,  h));
	outVerts.pushBack(center + PxVec3( h,  h,  h));
	outVerts.pushBack(center + PxVec3(-h,  h,  h));

	outTets.clear();
	PxU32 tets[] = { 0,1,3,4, 1,2,3,6, 3,4,6,7, 1,4,5,6, 1,3,4,6 };
	for (PxU32 i = 0; i < 20; i++)
		outTets.pushBack(tets[i]);
}

inline void avbdGenerateSubdividedCubeTets(
	PxVec3 center, PxReal halfSize, int N,
	PxArray<PxVec3>& outVerts,
	PxArray<PxU32>& outTets)
{
	outVerts.clear();
	outTets.clear();
	PxReal cellSize = 2.0f * halfSize / PxReal(N);
	PxVec3 origin = center - PxVec3(halfSize, halfSize, halfSize);

	for (int iz = 0; iz <= N; iz++)
		for (int iy = 0; iy <= N; iy++)
			for (int ix = 0; ix <= N; ix++)
				outVerts.pushBack(origin + PxVec3(PxReal(ix) * cellSize,
				                                  PxReal(iy) * cellSize,
				                                  PxReal(iz) * cellSize));

	for (int iz = 0; iz < N; iz++)
		for (int iy = 0; iy < N; iy++)
			for (int ix = 0; ix < N; ix++)
			{
				PxU32 v[8];
				v[0] = PxU32(iz * (N+1) * (N+1) + iy * (N+1) + ix);
				v[1] = v[0] + 1;
				v[2] = v[0] + PxU32(N+1) + 1;
				v[3] = v[0] + PxU32(N+1);
				v[4] = v[0] + PxU32((N+1) * (N+1));
				v[5] = v[4] + 1;
				v[6] = v[4] + PxU32(N+1) + 1;
				v[7] = v[4] + PxU32(N+1);

				PxU32 t[] = {
					v[0],v[1],v[3],v[4], v[1],v[2],v[3],v[6],
					v[3],v[4],v[6],v[7], v[1],v[4],v[5],v[6],
					v[1],v[3],v[4],v[6]
				};
				for (PxU32 i = 0; i < 20; i++)
					outTets.pushBack(t[i]);
			}
}

inline void avbdGenerateClothGrid(
	PxVec3 center, PxReal sizeX, PxReal sizeZ,
	int M, int N,
	PxArray<PxVec3>& outVerts,
	PxArray<PxU32>& outTris)
{
	outVerts.clear();
	outTris.clear();
	PxReal dx = sizeX / PxReal(M - 1);
	PxReal dz = sizeZ / PxReal(N - 1);
	PxVec3 origin = center - PxVec3(sizeX * 0.5f, 0.0f, sizeZ * 0.5f);

	for (int j = 0; j < N; j++)
		for (int i = 0; i < M; i++)
			outVerts.pushBack(origin + PxVec3(PxReal(i) * dx, 0.0f, PxReal(j) * dz));

	for (int j = 0; j < N - 1; j++)
		for (int i = 0; i < M - 1; i++)
		{
			PxU32 v00 = PxU32(j * M + i);
			PxU32 v10 = v00 + 1;
			PxU32 v01 = v00 + PxU32(M);
			PxU32 v11 = v01 + 1;
			outTris.pushBack(v00); outTris.pushBack(v10); outTris.pushBack(v01);
			outTris.pushBack(v10); outTris.pushBack(v11); outTris.pushBack(v01);
		}
}

inline void avbdGenerateSubdividedSphereTets(
	PxVec3 center, PxReal radius, int N,
	PxArray<PxVec3>& outVerts,
	PxArray<PxU32>& outTets)
{
	// Generate a subdivided cube, then map vertices proportionally onto a sphere.
	// Each vertex keeps its fractional distance from center to cube surface,
	// but the direction is spherically normalized.  This avoids collapsing
	// multiple interior vertices onto the same surface point (which would
	// create degenerate zero-volume tetrahedra).
	avbdGenerateSubdividedCubeTets(center, radius, N, outVerts, outTets);

	for (PxU32 i = 0; i < outVerts.size(); i++)
	{
		PxVec3 d = outVerts[i] - center;
		PxReal len = d.magnitude();
		if (len > 1e-8f)
		{
			// Distance from center to cube surface in direction d:
			//   cubeSurfR = halfSize * len / max(|dx|,|dy|,|dz|)
			// Fraction of the way from center to cube surface:
			//   frac = len / cubeSurfR = max(|dx|,|dy|,|dz|) / halfSize
			// Map to sphere: new distance = frac * radius
			PxReal maxAbs = PxMax(PxAbs(d.x), PxMax(PxAbs(d.y), PxAbs(d.z)));
			PxReal frac = maxAbs / radius;  // 0 at center, 1 at cube face
			outVerts[i] = center + d * (1.0f / len) * (frac * radius);
		}
	}

	// Fix tet orientation after the non-linear mapping (some tets may invert)
	for (PxU32 t = 0; t + 3 < outTets.size(); t += 4)
	{
		PxVec3 e1 = outVerts[outTets[t+1]] - outVerts[outTets[t]];
		PxVec3 e2 = outVerts[outTets[t+2]] - outVerts[outTets[t]];
		PxVec3 e3 = outVerts[outTets[t+3]] - outVerts[outTets[t]];
		if (e1.dot(e2.cross(e3)) < 0.0f)
		{
			PxU32 tmp = outTets[t+1]; outTets[t+1] = outTets[t+2]; outTets[t+2] = tmp;
		}
	}
}

// Generate a cone-shaped tet mesh directly from layered rings + apex.
// Base center at `center`, base radius `radius`, height along +Y.
inline void avbdGenerateConeTets(
	PxVec3 center, PxReal radius, PxReal height, int N,
	PxArray<PxVec3>& outVerts,
	PxArray<PxU32>& outTets)
{
	outVerts.clear();
	outTets.clear();

	const int nLayers = PxMax(N, 2);
	const int nRing   = PxMax(4 * N, 8);
	const PxReal pi2  = 2.0f * 3.14159265358979f;

	// --- vertices ---
	// Each layer i (0..nLayers-1): 1 center + nRing ring vertices
	// Final vertex: apex
	for (int i = 0; i < nLayers; i++)
	{
		PxReal t = PxReal(i) / PxReal(nLayers); // 0 = base, approaches 1 near tip
		PxReal h = t * height;
		PxReal r = radius * (1.0f - t);

		// Center of this layer
		outVerts.pushBack(center + PxVec3(0.0f, h, 0.0f));

		// Ring vertices
		for (int j = 0; j < nRing; j++)
		{
			PxReal angle = pi2 * PxReal(j) / PxReal(nRing);
			outVerts.pushBack(center + PxVec3(r * PxCos(angle), h, r * PxSin(angle)));
		}
	}

	PxU32 apexIdx = outVerts.size();
	outVerts.pushBack(center + PxVec3(0.0f, height, 0.0f));

	// --- helper lambdas ---
	const int stride = 1 + nRing;
	// center vertex of layer i
	auto ci = [stride](int layer) -> PxU32 { return PxU32(layer * stride); };
	// ring vertex j of layer i (j wraps around)
	auto ri = [stride, nRing](int layer, int j) -> PxU32
	{
		return PxU32(layer * stride + 1 + ((j % nRing + nRing) % nRing));
	};

	// --- tets between adjacent layers (prism decomposition) ---
	for (int i = 0; i + 1 < nLayers; i++)
	{
		for (int j = 0; j < nRing; j++)
		{
			// 3 tets per wedge (triangular prism between two ring segments)
			outTets.pushBack(ci(i));   outTets.pushBack(ri(i, j+1)); outTets.pushBack(ri(i, j));   outTets.pushBack(ci(i+1));
			outTets.pushBack(ri(i,j)); outTets.pushBack(ri(i,j+1)); outTets.pushBack(ci(i+1));     outTets.pushBack(ri(i+1,j));
			outTets.pushBack(ri(i,j+1)); outTets.pushBack(ci(i+1)); outTets.pushBack(ri(i+1,j));   outTets.pushBack(ri(i+1,j+1));
		}
	}

	// --- apex cap (connect top layer to apex) ---
	{
		int top = nLayers - 1;
		for (int j = 0; j < nRing; j++)
		{
			outTets.pushBack(ci(top)); outTets.pushBack(ri(top, j+1)); outTets.pushBack(ri(top, j)); outTets.pushBack(apexIdx);
		}
	}

	// --- fix orientation: ensure positive tet volume ---
	for (PxU32 t = 0; t + 3 < outTets.size(); t += 4)
	{
		PxVec3 e1 = outVerts[outTets[t+1]] - outVerts[outTets[t]];
		PxVec3 e2 = outVerts[outTets[t+2]] - outVerts[outTets[t]];
		PxVec3 e3 = outVerts[outTets[t+3]] - outVerts[outTets[t]];
		if (e1.dot(e2.cross(e3)) < 0.0f)
		{
			PxU32 tmp = outTets[t+1]; outTets[t+1] = outTets[t+2]; outTets[t+2] = tmp;
		}
	}
}

// =============================================================================
// NOTE: For production tet mesh generation from arbitrary shapes, use the
// PhysX TetMaker API (PxTetMaker::createConformingTetrahedronMesh +
// PxTetMaker::createVoxelTetrahedronMesh) which provides BVH-based surface
// projection, boundary cell subdivision, and iterative relaxation.
// See extensions/PxTetMakerExt.h.
// =============================================================================

// =============================================================================
// Soft body creation helper
// =============================================================================

inline PxU32 avbdCreateSoftBody(
	const PxVec3* vertices, PxU32 numVertices,
	const PxU32* tets, PxU32 numTetIndices,
	const PxU32* tris, PxU32 numTriIndices,
	PxReal youngsModulus, PxReal poissonsRatio,
	PxReal density, PxReal damping,
	PxReal bendingStiffness, PxReal thickness,
	PxArray<AvbdSoftParticle>& outParticles,
	PxArray<AvbdSoftBody>& outSoftBodies,
	bool flatteningEnabled = false,
	PxReal selfCollisionFilterDistance = 0.0f,
	PxReal dynamicFriction = 0.5f,
	bool coRotationalVolumeModel = true)
{
	PxU32 particleStart = outParticles.size();

	PxArray<PxReal> vertexMass;
	vertexMass.resize(numVertices, 0.0f);

	if (numTetIndices > 0)
	{
		for (PxU32 i = 0; i + 3 < numTetIndices; i += 4)
		{
			PxVec3 e1 = vertices[tets[i+1]] - vertices[tets[i]];
			PxVec3 e2 = vertices[tets[i+2]] - vertices[tets[i]];
			PxVec3 e3 = vertices[tets[i+3]] - vertices[tets[i]];
			PxReal vol = PxAbs(e1.dot(e2.cross(e3)) / 6.0f);
			PxReal tetMass = vol * density;
			PxReal perVertex = tetMass / 4.0f;
			vertexMass[tets[i]]   += perVertex;
			vertexMass[tets[i+1]] += perVertex;
			vertexMass[tets[i+2]] += perVertex;
			vertexMass[tets[i+3]] += perVertex;
		}
	}
	else if (numTriIndices > 0)
	{
		for (PxU32 i = 0; i + 2 < numTriIndices; i += 3)
		{
			PxVec3 e1 = vertices[tris[i+1]] - vertices[tris[i]];
			PxVec3 e2 = vertices[tris[i+2]] - vertices[tris[i]];
			PxReal area = e1.cross(e2).magnitude() * 0.5f;
			PxReal triMass = area * thickness * density;
			PxReal perVertex = triMass / 3.0f;
			vertexMass[tris[i]]   += perVertex;
			vertexMass[tris[i+1]] += perVertex;
			vertexMass[tris[i+2]] += perVertex;
		}
	}

	PxReal minMass = 1e-4f;
	for (PxU32 i = 0; i < numVertices; i++)
		vertexMass[i] = PxMax(vertexMass[i], minMass);

	// Mass uniformization (matches PhysX GPU's maxInvMassRatio=50):
	// Clamp minimum mass so that max/min mass ratio <= 50.
	// Prevents tiny tets from creating particles with extreme inv-mass
	// that blow up under elastic forces.
	{
		PxReal maxMass = 0.0f;
		for (PxU32 i = 0; i < numVertices; i++)
			maxMass = PxMax(maxMass, vertexMass[i]);
		const PxReal maxInvMassRatio = 50.0f;
		PxReal massFloor = maxMass / maxInvMassRatio;
		for (PxU32 i = 0; i < numVertices; i++)
			vertexMass[i] = PxMax(vertexMass[i], massFloor);
	}

	for (PxU32 i = 0; i < numVertices; i++)
	{
		AvbdSoftParticle sp;
		sp.position = vertices[i];
		sp.velocity = PxVec3(0.0f);
		sp.prevVelocity = PxVec3(0.0f);
		sp.initialPosition = vertices[i];
		sp.predictedPosition = vertices[i];
		sp.mass = vertexMass[i];
		sp.invMass = 1.0f / sp.mass;
		sp.damping = damping;
		outParticles.pushBack(sp);
	}

	AvbdSoftBody sb;
	sb.compiled.particleStart = particleStart;
	sb.compiled.particleCount = numVertices;
	sb.compiled.selfCollisionRestPositions.resize(numVertices);
	for(PxU32 i = 0; i < numVertices; i++)
		sb.compiled.selfCollisionRestPositions[i] = vertices[i];
	sb.compiled.selfCollisionFilterDistance =
		PxMax(selfCollisionFilterDistance, 0.0f);

	if (numTetIndices > 0)
		for (PxU32 i = 0; i < numTetIndices; i++)
			sb.compiled.tetrahedra.pushBack(tets[i]);

	if (numTriIndices > 0)
		for (PxU32 i = 0; i < numTriIndices; i++)
			sb.compiled.triangles.pushBack(tris[i]);

	sb.material.youngsModulus = youngsModulus;
	sb.material.poissonsRatio = poissonsRatio;
	sb.material.density = density;
	sb.material.damping = damping;
	sb.material.bendingStiffness = bendingStiffness;
	sb.material.thickness = thickness;
	sb.material.dynamicFriction =
		PxMax(dynamicFriction, 0.0f);
	sb.material.coRotationalVolumeModel = coRotationalVolumeModel;

	sb.buildElements(outParticles);
	sb.compiled.compileBendingRestAngles(flatteningEnabled);

	outSoftBodies.pushBack(sb);
	return particleStart;
}

// =============================================================================
// AVBD unified solver step -- self-contained soft body simulation step
// =============================================================================

// Callback type for contact re-detection between outer iterations.
// Called with current particles/bodies + the contacts array to refill.
typedef void (*AvbdContactRedetectFn)(
	AvbdSoftParticle* particles, PxU32 numParticles,
	AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts, void* userData);

struct AvbdSoftBodyStepStats
{
	PxF64 predictionMs;
	PxF64 contactIndexMs;
	PxF64 bodyPrecomputeMs;
	PxF64 bodySolveMs;
	PxF64 particleSolveMs;
	PxF64 projectionMs;
	PxF64 dualMs;
	PxF64 redetectMs;
	PxF64 velocityMs;
	PxF64 frictionMs;
	PxU64 requestedOuterIterations;
	PxU64 requestedInnerIterations;
	PxU64 executedOuterIterations;
	PxU64 executedInnerIterations;
	PxU64 particleSweeps;
	// Read-only qualification telemetry for the default-off one-tet ground
	// patch experiment. These count prepared rows, never extra solver work.
	PxU64 groundTetPatchGroundPositionAlRows;
	PxU64 groundTetPatchFourSupportRows;
	PxU64 groundTetPatchSingleTetRows;
	PxU64 groundTetPatchActiveRows;
	// Split world-static tangent ownership counters.  These count final
	// velocity rows, not Position-AL normal detection or primal work.
	PxU64 worldStaticVelocityTangentOwnerRows;
	PxU64 worldStaticVelocityTangentAppliedRows;
	PxU64 workspaceGrowthEvents;
	PxU64 workspaceGrowthBytes;
	PxU64 contactWorkspaceGrowthEvents;
	PxU64 contactWorkspaceGrowthBytes;
	PxU64 contactSweepScratchGrowthEvents;
	PxU64 contactSweepScratchGrowthBytes;
	PxU64 contactOutputGrowthEvents;
	PxU64 contactOutputGrowthBytes;
	PxU32 peakContactOutputCount;
	PxU32 peakContactOutputCapacity;
	PxU32 peakContactIncidenceCount;
	PxU32 peakContactIncidenceCapacity;
	PxU32 peakStateTransferContactCount;
	PxU32 peakStateTransferContactCapacity;
	PxU32 peakStateTransferUsedCapacity;
	// P4 validation telemetry. A zero color count means no complete dynamic
	// plan was published for this component step; it is not a license to run
	// an incomplete colored schedule.
	PxU32 particlePrimalColorCount;
	PxU32 particlePrimalDynamicAccessGroupCount;
	// Historical public field name.  This counts complete colored primal
	// sweeps for both ordered-reference and relaxed production schedules.
	PxU64 particlePrimalColoredSerialSweeps;
	PxU64 particlePrimalColoredSerialFallbackSweeps;
	PxU64 trustRegionLimitedParticleSteps;
	PxU64 positiveJLimitedParticleSteps;
	PxU64 positiveJRejectedParticleSteps;
	PxU64 nonFiniteRejectedParticleSteps;
	PxU64 tetLinearizationCacheFallbackParticleSteps;
	PxU64 legacyAppliedConvergedOuterIterations;
	PxU64 residualConvergedOuterIterations;
	PxU64 unsafeAppliedConvergenceCandidates;
	PxU64 budgetExhaustedOuterIterations;
	PxU64 shadowResidual1e5ConvergedOuterIterations;
	PxU64 shadowResidual1e5SavedInnerIterations;
	PxU64 shadowResidual1e4ConvergedOuterIterations;
	PxU64 shadowResidual1e4SavedInnerIterations;
	// P8.1 opt-in census of work actually reached by non-static particle
	// primal solves. These counters identify packet capacity; they do not
	// imply that a SIMD material kernel was executed.
	PxU64 particlePrimalCensusDynamicParticleSolves;
	PxU64 particlePrimalCensusTriangleEvaluations;
	PxU64 particlePrimalCensusCorotationalTetEvaluations;
	PxU64 particlePrimalCensusNeoHookeanTetEvaluations;
	PxU64 particlePrimalCensusBendingEvaluations;
	PxU64 particlePrimalCensusContactEvaluations;
	PxU64 particlePrimalCensusTetPacket8FullPackets;
	PxU64 particlePrimalCensusTetPacket8TailLanes;
	// P8.2 topology-owned packet IR telemetry. These report compiled metadata,
	// not executed SIMD work.
	PxU64 particlePrimalTetPacketIrBodies;
	PxU64 particlePrimalTetPacketIrPackets;
	PxU64 particlePrimalTetPacketIrActiveLanes;
	PxU64 particlePrimalTetPacketIrTailLanes;
	PxU64 particlePrimalTetPacketIrActiveTailLanes;
	PxU64 particlePrimalTetPacketIrInvalidBodies;
	PxReal finalMaxLocalSolveDisplacement;
	PxReal finalMaxAppliedDisplacement;
	PxReal finalMaxDisplacement;

	AvbdSoftBodyStepStats()
		: predictionMs(0.0), contactIndexMs(0.0), bodyPrecomputeMs(0.0),
		  bodySolveMs(0.0), particleSolveMs(0.0), projectionMs(0.0),
		  dualMs(0.0), redetectMs(0.0), velocityMs(0.0), frictionMs(0.0),
		  requestedOuterIterations(0), requestedInnerIterations(0),
		  executedOuterIterations(0), executedInnerIterations(0),
		  particleSweeps(0), groundTetPatchGroundPositionAlRows(0),
		  groundTetPatchFourSupportRows(0),
		  groundTetPatchSingleTetRows(0), groundTetPatchActiveRows(0),
		  worldStaticVelocityTangentOwnerRows(0),
		  worldStaticVelocityTangentAppliedRows(0),
		  workspaceGrowthEvents(0),
		  workspaceGrowthBytes(0), contactWorkspaceGrowthEvents(0),
		  contactWorkspaceGrowthBytes(0), contactSweepScratchGrowthEvents(0),
		  contactSweepScratchGrowthBytes(0), contactOutputGrowthEvents(0),
		  contactOutputGrowthBytes(0), peakContactOutputCount(0),
		  peakContactOutputCapacity(0), peakContactIncidenceCount(0),
		  peakContactIncidenceCapacity(0), peakStateTransferContactCount(0),
		  peakStateTransferContactCapacity(0), peakStateTransferUsedCapacity(0),
		  particlePrimalColorCount(0),
		  particlePrimalDynamicAccessGroupCount(0),
		  particlePrimalColoredSerialSweeps(0),
		  particlePrimalColoredSerialFallbackSweeps(0),
		  trustRegionLimitedParticleSteps(0),
		  positiveJLimitedParticleSteps(0),
		  positiveJRejectedParticleSteps(0),
		  nonFiniteRejectedParticleSteps(0),
		  tetLinearizationCacheFallbackParticleSteps(0),
		  legacyAppliedConvergedOuterIterations(0),
		  residualConvergedOuterIterations(0),
		  unsafeAppliedConvergenceCandidates(0),
		  budgetExhaustedOuterIterations(0),
		  shadowResidual1e5ConvergedOuterIterations(0),
		  shadowResidual1e5SavedInnerIterations(0),
		  shadowResidual1e4ConvergedOuterIterations(0),
		  shadowResidual1e4SavedInnerIterations(0),
		  particlePrimalCensusDynamicParticleSolves(0),
		  particlePrimalCensusTriangleEvaluations(0),
		  particlePrimalCensusCorotationalTetEvaluations(0),
		  particlePrimalCensusNeoHookeanTetEvaluations(0),
		  particlePrimalCensusBendingEvaluations(0),
		  particlePrimalCensusContactEvaluations(0),
		  particlePrimalCensusTetPacket8FullPackets(0),
		  particlePrimalCensusTetPacket8TailLanes(0),
		  particlePrimalTetPacketIrBodies(0),
		  particlePrimalTetPacketIrPackets(0),
		  particlePrimalTetPacketIrActiveLanes(0),
		  particlePrimalTetPacketIrTailLanes(0),
		  particlePrimalTetPacketIrActiveTailLanes(0),
		  particlePrimalTetPacketIrInvalidBodies(0),
		  finalMaxLocalSolveDisplacement(0.0f),
		  finalMaxAppliedDisplacement(0.0f),
		  finalMaxDisplacement(0.0f)
	{
	}

	// All members are numeric counters or timings whose constructor value is
	// zero.  Reset in place rather than assigning a value-initialized temporary:
	// the latter makes MSVC reserve a full telemetry-sized object in the scalar
	// step frame even though the temporary has no solver role.
	PX_FORCE_INLINE void reset()
	{
		std::memset(this, 0, sizeof(*this));
	}
};

// Publish topology-owned IR facts once per component step. This remains out
// of every particle range and does not imply that a packet evaluator runs.
inline void avbdPublishCorotationalTetPacketIrStats(
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdSoftBodyStepStats* stepStats)
{
	if(!stepStats || !avbdUseCorotationalTetPacketIr())
		return;
	stepStats->particlePrimalTetPacketIrBodies = 0;
	stepStats->particlePrimalTetPacketIrPackets = 0;
	stepStats->particlePrimalTetPacketIrActiveLanes = 0;
	stepStats->particlePrimalTetPacketIrTailLanes = 0;
	stepStats->particlePrimalTetPacketIrActiveTailLanes = 0;
	stepStats->particlePrimalTetPacketIrInvalidBodies = 0;
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		const AvbdSoftBodyCompiledData& compiled =
			softBodies[bodyIndex].compiled;
		if(compiled.tetElements.empty())
			continue;
		// Topology compilation already validated the full packet mapping. A
		// step must only read that immutable result; rescanning every tet ref
		// here would distort the very material-stage timing P8 is preparing.
		if(!compiled.tetIncidencePacketProgramValid)
		{
			stepStats->particlePrimalTetPacketIrInvalidBodies++;
			continue;
		}
		stepStats->particlePrimalTetPacketIrBodies++;
		stepStats->particlePrimalTetPacketIrPackets +=
			compiled.tetIncidencePackets.size();
		for(PxU32 localParticleIndex = 0;
			localParticleIndex < compiled.particleCount;
			localParticleIndex++)
		{
			const PxU32 activeLanes =
				compiled.elementAdjacency[localParticleIndex].tetRefs.size();
			const PxU32 packets =
				compiled.tetIncidencePacketRanges[localParticleIndex].packetCount;
			stepStats->particlePrimalTetPacketIrActiveLanes += activeLanes;
			stepStats->particlePrimalTetPacketIrTailLanes +=
				packets * eAVBD_TET_INCIDENCE_PACKET_WIDTH - activeLanes;
			stepStats->particlePrimalTetPacketIrActiveTailLanes +=
				activeLanes % eAVBD_TET_INCIDENCE_PACKET_WIDTH;
		}
	}
}

struct AvbdSoftContactParticleRef
{
	PxU32 contactIndex;
	PxReal jacobianScale;

	AvbdSoftContactParticleRef()
		: contactIndex(PX_MAX_U32), jacobianScale(0.0f)
	{
	}

	AvbdSoftContactParticleRef(PxU32 index, PxReal scale)
		: contactIndex(index), jacobianScale(scale)
	{
	}
};

// P4.5's unit of task-local primal accounting.  The particle block has no
// license to update a shared step statistic: a future Scene task owns one of
// these records and its parent performs the canonical reduction.
struct AvbdParticlePrimalRangeObservation
{
	AvbdSoftSweepConvergenceObservation sweepObservation;
	PxU64 tetLinearizationCacheFallbackParticleSteps;

	AvbdParticlePrimalRangeObservation()
		: tetLinearizationCacheFallbackParticleSteps(0)
	{
	}

	PX_FORCE_INLINE void merge(
		const AvbdParticlePrimalRangeObservation& other)
	{
		sweepObservation.merge(other.sweepObservation);
		tetLinearizationCacheFallbackParticleSteps +=
			other.tetLinearizationCacheFallbackParticleSteps;
	}
};

// P8.1 is a diagnostic transaction, not a particle-range dependency.  Keep
// its counters physically separate from the task-local convergence record so
// a disabled census cannot change the scalar primal's stack/register layout.
struct AvbdParticlePrimalWorkCensus
{
	PxU64 dynamicParticleSolves;
	PxU64 triangleEvaluations;
	PxU64 corotationalTetEvaluations;
	PxU64 neoHookeanTetEvaluations;
	PxU64 bendingEvaluations;
	PxU64 contactEvaluations;
	PxU64 tetPacket8FullPackets;
	PxU64 tetPacket8TailLanes;

	AvbdParticlePrimalWorkCensus()
		: dynamicParticleSolves(0), triangleEvaluations(0),
		  corotationalTetEvaluations(0), neoHookeanTetEvaluations(0),
		  bendingEvaluations(0), contactEvaluations(0),
		  tetPacket8FullPackets(0), tetPacket8TailLanes(0)
	{
	}
};

// This is deliberately separate from convergence telemetry. P8.1 uses the
// count-only result to choose a packet boundary, and does not change any
// convergence, limiter or early-out decision.
PX_FORCE_INLINE void avbdAccumulateParticlePrimalWorkCensus(
	AvbdSoftBodyStepStats& stepStats,
	const AvbdParticlePrimalWorkCensus& census)
{
	stepStats.particlePrimalCensusDynamicParticleSolves +=
		census.dynamicParticleSolves;
	stepStats.particlePrimalCensusTriangleEvaluations +=
		census.triangleEvaluations;
	stepStats.particlePrimalCensusCorotationalTetEvaluations +=
		census.corotationalTetEvaluations;
	stepStats.particlePrimalCensusNeoHookeanTetEvaluations +=
		census.neoHookeanTetEvaluations;
	stepStats.particlePrimalCensusBendingEvaluations +=
		census.bendingEvaluations;
	stepStats.particlePrimalCensusContactEvaluations +=
		census.contactEvaluations;
	stepStats.particlePrimalCensusTetPacket8FullPackets +=
		census.tetPacket8FullPackets;
	stepStats.particlePrimalCensusTetPacket8TailLanes +=
		census.tetPacket8TailLanes;
}

// P8.1 instrumentation is intentionally outside the scalar solve kernel.
// The count is a diagnostic-only replay of immutable sweep inputs.  The
// topology and contact index are fixed for one outer epoch, so one replay per
// epoch scaled by its executed sweep count is exactly equivalent to a replay
// after every inner sweep and leaves no census control in the scalar loop.
inline void avbdRecordParticlePrimalWorkCensusForSweep(
	const AvbdSoftParticle* particles, const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies, const PxU32* contactStarts,
	AvbdParticlePrimalWorkCensus& census)
{
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		for(PxU32 localIndex = 0;
			localIndex < body.compiled.particleCount; ++localIndex)
		{
			const PxU32 particleIndex =
				body.compiled.particleStart + localIndex;
			if(particles[particleIndex].isStatic())
				continue;
			const AvbdParticleElementAdjacency& elementAdjacency =
				body.compiled.elementAdjacency[localIndex];
			const PxU32 tetIncidenceCount =
				elementAdjacency.tetRefs.size();
			census.dynamicParticleSolves++;
			census.triangleEvaluations +=
				elementAdjacency.triRefs.size();
			if(body.material.coRotationalVolumeModel)
				census.corotationalTetEvaluations +=
					tetIncidenceCount;
			else
				census.neoHookeanTetEvaluations +=
					tetIncidenceCount;
			census.bendingEvaluations +=
				elementAdjacency.bendRefs.size();
			census.contactEvaluations +=
				contactStarts[particleIndex + 1] -
				contactStarts[particleIndex];
			census.tetPacket8FullPackets +=
				tetIncidenceCount / 8;
			census.tetPacket8TailLanes +=
				tetIncidenceCount % 8;
		}
	}
}

// Keep the diagnostic replay entirely outside the scalar sweep's generated
// code.  The enabled path is intentionally one cold transaction per outer
// epoch; the default path has no inner-loop diagnostic control or call edge.
PX_NOINLINE inline void avbdAccumulateParticlePrimalWorkCensusForOuterEpoch(
	AvbdSoftBodyStepStats& stepStats,
	const AvbdSoftParticle* particles, const AvbdSoftBody* softBodies,
	PxU32 numSoftBodies, const PxU32* contactStarts, PxU64 sweepCount)
{
	AvbdParticlePrimalWorkCensus workCensus;
	avbdRecordParticlePrimalWorkCensusForSweep(
		particles, softBodies, numSoftBodies, contactStarts, workCensus);
	workCensus.dynamicParticleSolves *= sweepCount;
	workCensus.triangleEvaluations *= sweepCount;
	workCensus.corotationalTetEvaluations *= sweepCount;
	workCensus.neoHookeanTetEvaluations *= sweepCount;
	workCensus.bendingEvaluations *= sweepCount;
	workCensus.contactEvaluations *= sweepCount;
	workCensus.tetPacket8FullPackets *= sweepCount;
	workCensus.tetPacket8TailLanes *= sweepCount;
	avbdAccumulateParticlePrimalWorkCensus(stepStats, workCensus);
}

// Admitted P8.3 material backend.  Its stack-resident gather/output
// packets and control flow stay out of the scalar solve body.  Complete
// canonical incidence packets always use SIMD.  P17 additionally admits a
// final canonical tail when at least half of its lanes are active; shorter
// tails and every exceptional active lane call the scalar authority before
// any lane result is reduced.
PX_NOINLINE inline void avbdAccumulateCorotationalTetPacketContributions(
	const AvbdSoftBody& softBody, PxU32 localParticleIndex,
	const AvbdSoftParticle* particles,
	AvbdCpuIsaCorotationalTetPacket8Fn packetKernel,
	bool cacheTetLinearizations,
	AvbdTetVertexLinearization* tetLinearizations,
	PxVec3& force, PxMat33& hessian)
{
	PX_ASSERT(packetKernel);
	const AvbdSoftBodyCompiledData& compiled = softBody.compiled;
	const AvbdParticleElementAdjacency& adjacency =
		compiled.elementAdjacency[localParticleIndex];
	const PxU32 tetIncidenceCount = adjacency.tetRefs.size();
	PX_ASSERT(compiled.tetIncidencePacketProgramValid);
	PX_ASSERT(localParticleIndex <
		compiled.tetIncidencePacketRanges.size());
	const AvbdTetIncidencePacketRange& packetRange =
		compiled.tetIncidencePacketRanges[localParticleIndex];
	const PxU32 fullPacketCount = tetIncidenceCount /
		eAVBD_TET_INCIDENCE_PACKET_WIDTH;
	const PxU32 tailLaneCount = tetIncidenceCount %
		eAVBD_TET_INCIDENCE_PACKET_WIDTH;
	const bool useTailPacket = tailLaneCount >= 4;
	const PxU32 vectorPacketCount =
		fullPacketCount + PxU32(useTailPacket);
	const PxU32 vectorIncidenceCount =
		fullPacketCount * eAVBD_TET_INCIDENCE_PACKET_WIDTH +
		(useTailPacket ? tailLaneCount : 0);
	PX_ASSERT(packetRange.packetCount >= vectorPacketCount);

	for(PxU32 packetOrdinal = 0;
		packetOrdinal < vectorPacketCount; packetOrdinal++)
	{
		const AvbdTetIncidencePacket8& incidencePacket =
			compiled.tetIncidencePackets[
				packetRange.packetStart + packetOrdinal];
		const PxU32 firstTetRefIndex = packetOrdinal *
			eAVBD_TET_INCIDENCE_PACKET_WIDTH;
		const PxU32 packetLaneCount = PxMin(
			eAVBD_TET_INCIDENCE_PACKET_WIDTH,
			tetIncidenceCount - firstTetRefIndex);
		PX_ASSERT(incidencePacket.validMask ==
			(packetLaneCount == eAVBD_TET_INCIDENCE_PACKET_WIDTH
				? PxU8(0xffu)
				: PxU8((1u << packetLaneCount) - 1u)));
		AvbdCorotationalTetPacket8Input input = {};
		AvbdCorotationalTetPacket8Output output = {};
		for(PxU32 lane = 0;
			lane < packetLaneCount; lane++)
		{
			const PxU32 tetRefIndex = firstTetRefIndex + lane;
			PX_ASSERT(adjacency.tetRefs[tetRefIndex].index ==
				incidencePacket.tetIndices[lane]);
			PX_ASSERT(adjacency.tetRefs[tetRefIndex].vOrder ==
				incidencePacket.vertexOrders[lane]);
			const AvbdTetElement& tet =
				compiled.tetElements[incidencePacket.tetIndices[lane]];
			const PxU32 vertexOrder =
				incidencePacket.vertexOrders[lane];
			const PxVec3 p0 = particles[tet.p0].position;
			const PxVec3 e1 = particles[tet.p1].position - p0;
			const PxVec3 e2 = particles[tet.p2].position - p0;
			const PxVec3 e3 = particles[tet.p3].position - p0;
			if(cacheTetLinearizations)
				avbdEvaluateTetDeterminantAndGradient(
					tet, vertexOrder, e1, e2, e3,
					tetLinearizations[tetRefIndex].determinant,
					tetLinearizations[tetRefIndex].determinantGradient);

			input.e1X[lane] = e1.x;
			input.e1Y[lane] = e1.y;
			input.e1Z[lane] = e1.z;
			input.e2X[lane] = e2.x;
			input.e2Y[lane] = e2.y;
			input.e2Z[lane] = e2.z;
			input.e3X[lane] = e3.x;
			input.e3Y[lane] = e3.y;
			input.e3Z[lane] = e3.z;
			input.dm0X[lane] = tet.DmInv.column0.x;
			input.dm0Y[lane] = tet.DmInv.column0.y;
			input.dm0Z[lane] = tet.DmInv.column0.z;
			input.dm1X[lane] = tet.DmInv.column1.x;
			input.dm1Y[lane] = tet.DmInv.column1.y;
			input.dm1Z[lane] = tet.DmInv.column1.z;
			input.dm2X[lane] = tet.DmInv.column2.x;
			input.dm2Y[lane] = tet.DmInv.column2.y;
			input.dm2Z[lane] = tet.DmInv.column2.z;
			input.shapeX[lane] = tet.shapeGradients[vertexOrder].x;
			input.shapeY[lane] = tet.shapeGradients[vertexOrder].y;
			input.shapeZ[lane] = tet.shapeGradients[vertexOrder].z;
			input.shapeNormSq[lane] =
				tet.shapeGradientNormSq[vertexOrder];
			input.restVolume[lane] = tet.restVolume;
		}

		packetKernel(
			input, softBody.material.mu, softBody.material.lambda,
			output);
		for(PxU32 lane = 0;
			lane < packetLaneCount; lane++)
		{
			PxVec3 elementForce;
			PxMat33 elementHessian;
			if((output.validMask & PxU8(1u << lane)) != 0u)
			{
				elementForce = PxVec3(
					output.forceX[lane], output.forceY[lane],
					output.forceZ[lane]);
				elementHessian = PxMat33(
					PxVec3(output.hessianXX[lane],
						output.hessianXY[lane], output.hessianXZ[lane]),
					PxVec3(output.hessianXY[lane],
						output.hessianYY[lane], output.hessianYZ[lane]),
					PxVec3(output.hessianXZ[lane],
						output.hessianYZ[lane], output.hessianZZ[lane]));
			}
			else
			{
				const PxU32 tetRefIndex = firstTetRefIndex + lane;
				const AvbdParticleElementRef& ref =
					adjacency.tetRefs[tetRefIndex];
				avbdEvaluateCorotationalForceHessianPrepared(
					compiled.tetElements[ref.index], int(ref.vOrder),
					softBody.material.mu, softBody.material.lambda,
					particles, elementForce, elementHessian, NULL);
			}
			force = force + elementForce;
			hessian = hessian + elementHessian;
		}
	}

	for(PxU32 tetRefIndex = vectorIncidenceCount;
		tetRefIndex < tetIncidenceCount; tetRefIndex++)
	{
		const AvbdParticleElementRef& ref =
			adjacency.tetRefs[tetRefIndex];
		PxVec3 elementForce;
		PxMat33 elementHessian;
		avbdEvaluateCorotationalForceHessianPrepared(
			compiled.tetElements[ref.index], int(ref.vOrder),
			softBody.material.mu, softBody.material.lambda,
			particles, elementForce, elementHessian,
			cacheTetLinearizations
				? &tetLinearizations[tetRefIndex] : NULL);
		force = force + elementForce;
		hessian = hessian + elementHessian;
	}
}

// Explicit immutable-for-one-sweep input bundle for the component particle
// primal.  It is intentionally independent of Scene/task types: P4.5 can
// hand the same bundle and a private observation to a causal-layer range
// task, while the reference and P4.4 paths continue to invoke it serially.
// The only solver-state write is particles[pi].position.
struct AvbdParticlePrimalSolveContext
{
	AvbdSoftParticle* particles;
	const AvbdSoftContact* contacts;
	const PxU32* contactStarts;
	const AvbdSoftContactParticleRef* contactIndices;
	const PxReal* selfCollisionSafetyBounds;
	PxReal invDt;
	PxReal invDtSq;
	AvbdCpuIsaCorotationalTetPacket8Fn corotationalTetPacketKernel;

	PX_FORCE_INLINE bool canUseCorotationalTetPackets(
		const AvbdSoftBody& sb, PxU32 localParticleIndex) const
	{
		return corotationalTetPacketKernel &&
			sb.material.coRotationalVolumeModel &&
			sb.compiled.tetIncidencePacketProgramValid &&
			localParticleIndex <
				sb.compiled.tetIncidencePacketRanges.size() &&
			sb.compiled.elementAdjacency[localParticleIndex].tetRefs.size() >=
				eAVBD_TET_INCIDENCE_PACKET_WIDTH;
	}

	// Keep the candidate instantiation behind a real call boundary.  In
	// particular, do not let its packet eligibility and fallback control flow
	// enter the canonical scalar step section.
	PX_NOINLINE void solveWithCorotationalTetPackets(
		const AvbdSoftBody& sb, PxU32 localParticleIndex,
		AvbdParticlePrimalRangeObservation& observation) const
	{
		PX_ASSERT(corotationalTetPacketKernel);
		PX_ASSERT(canUseCorotationalTetPackets(sb, localParticleIndex));
		solve<true, true>(sb, localParticleIndex, observation);
	}

	template<
		bool enableCorotationalTetPackets = false,
		bool corotationalTetPacketEligibilityProven = false>
	PX_FORCE_INLINE void solve(
		const AvbdSoftBody& sb, PxU32 localParticleIndex,
		AvbdParticlePrimalRangeObservation& observation) const
	{
		const PxU32 particleIndex =
			sb.compiled.particleStart + localParticleIndex;
		AvbdSoftParticle& particle = particles[particleIndex];
		if(particle.isStatic())
			return;

		// Inertial term
		const PxReal massDtSq = particle.mass * invDtSq;
		PxMat33 H = PxMat33::createDiagonal(PxVec3(massDtSq));
		PxVec3 f = (particle.predictedPosition - particle.position) *
			massDtSq;

		const AvbdParticleElementAdjacency& elementAdjacency =
			sb.compiled.elementAdjacency[localParticleIndex];
		const AvbdParticleObjectiveAdjacency& objectiveAdjacency =
			sb.runtime.objectiveAdjacency[localParticleIndex];
		static const PxU32 eMAX_CACHED_TET_INCIDENCE = 64;
		AvbdTetVertexLinearization tetLinearizations[
			eMAX_CACHED_TET_INCIDENCE];
		const PxU32 tetIncidenceCount = elementAdjacency.tetRefs.size();
		const bool cacheTetLinearizations = tetIncidenceCount <=
			eMAX_CACHED_TET_INCIDENCE;
		if(!cacheTetLinearizations)
			observation.tetLinearizationCacheFallbackParticleSteps++;

		// Triangle (StVK) contributions
		for(PxU32 triangleRefIndex = 0;
			triangleRefIndex < elementAdjacency.triRefs.size();
			triangleRefIndex++)
		{
			const AvbdParticleElementRef& ref =
				elementAdjacency.triRefs[triangleRefIndex];
			PxVec3 elementForce;
			PxMat33 elementHessian;
			avbdEvaluateStVKForceHessian(
				sb.compiled.triElements[ref.index], int(ref.vOrder),
				sb.material.mu, sb.material.lambda, particles,
				elementForce, elementHessian);
			f = f + elementForce;
			H = H + elementHessian;
		}

		// Tetrahedral material-model contributions.  The scalar loop remains
		// the authority and the default route.  P8.3's candidate is admitted
		// only for a valid canonical program with at least one full packet.
		const bool useCorotationalTetPackets =
			enableCorotationalTetPackets &&
			(corotationalTetPacketEligibilityProven ||
			 canUseCorotationalTetPackets(sb, localParticleIndex));
		if(useCorotationalTetPackets)
			avbdAccumulateCorotationalTetPacketContributions(
				sb, localParticleIndex, particles,
				corotationalTetPacketKernel, cacheTetLinearizations,
				tetLinearizations, f, H);
		else
		{
			for(PxU32 tetRefIndex = 0;
				tetRefIndex < elementAdjacency.tetRefs.size(); tetRefIndex++)
			{
				const AvbdParticleElementRef& ref =
					elementAdjacency.tetRefs[tetRefIndex];
				PxVec3 elementForce;
				PxMat33 elementHessian;
				if(sb.material.coRotationalVolumeModel)
					avbdEvaluateCorotationalForceHessianPrepared(
						sb.compiled.tetElements[ref.index], int(ref.vOrder),
						sb.material.mu, sb.material.lambda, particles,
						elementForce, elementHessian,
						cacheTetLinearizations
							? &tetLinearizations[tetRefIndex] : NULL);
				else
					avbdEvaluateNeoHookeanForceHessianPrepared(
						sb.compiled.tetElements[ref.index], int(ref.vOrder),
						sb.material.mu, sb.material.lambda,
						sb.material.neoHookeanAlpha, particles,
						elementForce, elementHessian,
						cacheTetLinearizations
							? &tetLinearizations[tetRefIndex] : NULL);
				f = f + elementForce;
				H = H + elementHessian;
			}
		}

		// Bending contributions
		for(PxU32 bendRefIndex = 0;
			bendRefIndex < elementAdjacency.bendRefs.size(); bendRefIndex++)
		{
			const AvbdParticleElementRef& ref =
				elementAdjacency.bendRefs[bendRefIndex];
			PxVec3 elementForce;
			PxMat33 elementHessian;
			avbdEvaluateBendingForceHessian(
				sb.compiled.bendElements[ref.index], int(ref.vOrder),
				sb.material.bendingStiffness, particles,
				elementForce, elementHessian);
			f = f + elementForce;
			H = H + elementHessian;
		}

		// Contact contributions (indexed lookup)
		for(PxU32 contactRefIndex = contactStarts[particleIndex];
			contactRefIndex < contactStarts[particleIndex + 1];
			contactRefIndex++)
		{
			PxVec3 contactForce;
			PxMat33 contactHessian;
			const AvbdSoftContactParticleRef& contactRef =
				contactIndices[contactRefIndex];
			const AvbdSoftContact& contact = contacts[contactRef.contactIndex];
			avbdEvaluateContactParticleBlock(
				contact.geometry, contact.state, particles,
				contactRef.jacobianScale, contactForce, contactHessian);
			f = f + contactForce;
			H = H + contactHessian;
		}

		// Scene-external component supports only compiled one-way pin owners.
		// Rigid attachments require the low-level rigid-body block and must
		// never be consumed as a one-way particle-only objective here.
		for(PxU32 objectiveRefIndex = 0;
			objectiveRefIndex < objectiveAdjacency.objectiveIndices.size();
			objectiveRefIndex++)
		{
			const PxU32 objectiveIndex =
				objectiveAdjacency.objectiveIndices[objectiveRefIndex];
			const AvbdCompiledSoftObjective& objective =
				sb.runtime.compiledObjectives[objectiveIndex];
			if(!avbdIsPinPositionOwner(objective.owner))
			{
				PX_ASSERT(avbdIsPinPositionOwner(objective.owner));
				continue;
			}
			PxVec3 pinForce;
			PxMat33 pinHessian;
			avbdEvaluatePinForceHessian(
				objective.point,
				sb.runtime.pins[objective.runtimeStateIndex], particles,
				particleIndex, pinForce, pinHessian);
			f = f + pinForce;
			H = H + pinHessian;
		}

		// Stiffness-proportional Rayleigh damping (Newton VBD style):
		// Per-axis damping is proportional to elastic stiffness and clamped
		// so no axis receives less damping than the mass-proportional floor.
		if(particle.damping > 0.0f)
		{
			const PxReal dampingCoefficient =
				particle.damping * particle.mass * invDt;
			const PxReal elasticHxx = PxMax(H.column0.x - massDtSq, 0.0f);
			const PxReal elasticHyy = PxMax(H.column1.y - massDtSq, 0.0f);
			const PxReal elasticHzz = PxMax(H.column2.z - massDtSq, 0.0f);
			const PxReal traceElasticH =
				elasticHxx + elasticHyy + elasticHzz;
			PxReal dampingX;
			PxReal dampingY;
			PxReal dampingZ;
			if(traceElasticH > 1e-10f)
			{
				const PxReal scale = dampingCoefficient * 3.0f /
					traceElasticH;
				dampingX = PxMax(elasticHxx * scale, dampingCoefficient);
				dampingY = PxMax(elasticHyy * scale, dampingCoefficient);
				dampingZ = PxMax(elasticHzz * scale, dampingCoefficient);
			}
			else
				dampingX = dampingY = dampingZ = dampingCoefficient;
			const PxVec3 dampingDisplacement =
				particle.position - particle.initialPosition;
			f.x -= dampingX * dampingDisplacement.x;
			f.y -= dampingY * dampingDisplacement.y;
			f.z -= dampingZ * dampingDisplacement.z;
			H.column0.x += dampingX;
			H.column1.y += dampingY;
			H.column2.z += dampingZ;
		}

		// AVBD elastic proximal term: pulls toward the outer-iteration anchor.
		if(particle.elasticK > 0.0f)
		{
			H.column0.x += particle.elasticK;
			H.column1.y += particle.elasticK;
			H.column2.z += particle.elasticK;
			f = f + (particle.outerPosition - particle.position) *
				particle.elasticK;
		}

		const PxVec3 localSolveDisplacement = avbdSolveSymmetric33(H, f);
		const PxReal localSolveDisplacementSq =
			localSolveDisplacement.magnitudeSquared();
		PxVec3 proposedDisplacement = localSolveDisplacement;
		bool trustRegionLimited = false;
		const PxReal maxDisplacement = 1.0f;
		AvbdSoftTetDisplacementLimitResult limitResult;
		if(localSolveDisplacement.isFinite() &&
			PxIsFinite(localSolveDisplacementSq))
		{
			if(localSolveDisplacementSq >
				maxDisplacement * maxDisplacement)
			{
				proposedDisplacement *= maxDisplacement /
					PxSqrt(localSolveDisplacementSq);
				trustRegionLimited = true;
			}
			limitResult = cacheTetLinearizations
				? avbdLimitTetDisplacementFromLinearizations(
					proposedDisplacement, tetLinearizations, tetIncidenceCount)
				: avbdLimitTetDisplacementObserved(
					sb, particleIndex, particles, proposedDisplacement);
		}
		else
		{
			limitResult = AvbdSoftTetDisplacementLimitResult(
				PxVec3(0.0f), 0.0f,
				AvbdSoftTetDisplacementLimitReason::eNONFINITE_REJECTED);
		}
		const PxVec3 positionBeforeStep = particle.position;
		if(limitResult.appliedDisplacement.isFinite())
		{
			particle.position += limitResult.appliedDisplacement;
			const PxVec3 positionBeforeOgc = particle.position;
			avbdTruncateDisplacement(
				particle, particle.outerPosition,
				selfCollisionSafetyBounds[particleIndex]);
			if((particle.position - positionBeforeOgc).magnitudeSquared() >
				1.0e-20f)
				trustRegionLimited = true;
			limitResult.appliedDisplacement =
				particle.position - positionBeforeStep;
		}
		observation.sweepObservation.observe(
			localSolveDisplacement, trustRegionLimited, limitResult);
	}
};

// One cold candidate call per sweep keeps both the candidate traversal and
// its solve instantiation out of the default scalar caller.  Eligibility is
// still decided per particle inside the candidate because mixed surface,
// Neo-Hookean and short-incidence bodies must retain scalar authority.
PX_NOINLINE inline void avbdSolveParticlePrimalCorotationalTetPacketBodyRange(
	const AvbdParticlePrimalSolveContext& solveContext,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdParticlePrimalRangeObservation& observation)
{
	PX_ASSERT(solveContext.corotationalTetPacketKernel);
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		const bool useBodyPackets =
			solveContext.corotationalTetPacketKernel &&
			body.material.coRotationalVolumeModel &&
			body.compiled.tetIncidencePacketProgramValid;
		for(PxU32 localIndex = 0;
			localIndex < body.compiled.particleCount; localIndex++)
		{
			if(useBodyPackets &&
				body.compiled.elementAdjacency[localIndex].tetRefs.size() >=
					eAVBD_TET_INCIDENCE_PACKET_WIDTH)
				solveContext.solveWithCorotationalTetPackets(
					body, localIndex, observation);
			else
				solveContext.solve(body, localIndex, observation);
		}
	}
}

// P6 coarse task work owns complete bodies, never particle fragments.  The
// caller proves that the published sweep has no cross-body contact/objective
// write dependency, so every child writes a disjoint particle range.  Keep
// this candidate out of the scalar caller: it is reached only from the
// Scene-owned task continuation and reduces through a private observation.
PX_NOINLINE inline void avbdSolveParticlePrimalIndependentBodyRange(
	const AvbdParticlePrimalSolveContext& solveContext,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxU32 bodyBegin, PxU32 bodyEnd,
	AvbdParticlePrimalRangeObservation& observation)
{
	PX_UNUSED(numSoftBodies);
	PX_ASSERT(softBodies && bodyBegin < bodyEnd &&
		bodyEnd <= numSoftBodies);
	for(PxU32 bodyIndex = bodyBegin; bodyIndex < bodyEnd; bodyIndex++)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		const bool useBodyPackets =
			solveContext.corotationalTetPacketKernel &&
			body.material.coRotationalVolumeModel &&
			body.compiled.tetIncidencePacketProgramValid;
		for(PxU32 localIndex = 0;
			localIndex < body.compiled.particleCount; localIndex++)
		{
			if(useBodyPackets &&
				body.compiled.elementAdjacency[localIndex].tetRefs.size() >=
					eAVBD_TET_INCIDENCE_PACKET_WIDTH)
				solveContext.solveWithCorotationalTetPackets(
					body, localIndex, observation);
			else
				solveContext.solve(body, localIndex, observation);
		}
	}
}

// This is the exact work unit a future P4.5 task may own. The caller supplies
// a stable subrange of one already-published causal layer and a private
// observation. It neither changes planning state nor reduces into a shared
// step statistic.
inline void avbdSolveParticlePrimalPackedRange(
	const AvbdParticlePrimalSolveContext& solveContext,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const PxU32* particleBodyIndices, PxU32 numParticles,
	const PxU32* packedParticleIndices,
	PxU32 packedBegin, PxU32 packedEnd,
	AvbdParticlePrimalRangeObservation& observation)
{
	PX_UNUSED(numSoftBodies);
	PX_UNUSED(numParticles);
	PX_ASSERT(packedBegin <= packedEnd);
	if(solveContext.corotationalTetPacketKernel)
	{
		for(PxU32 packedIndex = packedBegin;
			packedIndex < packedEnd; packedIndex++)
		{
			const PxU32 particleIndex = packedParticleIndices[packedIndex];
			PX_ASSERT(particleIndex < numParticles);
			const PxU32 bodyIndex = particleBodyIndices[particleIndex];
			PX_ASSERT(bodyIndex < numSoftBodies);
			const AvbdSoftBody& body = softBodies[bodyIndex];
			PX_ASSERT(particleIndex >= body.compiled.particleStart &&
				particleIndex - body.compiled.particleStart <
					body.compiled.particleCount);
			const PxU32 localParticleIndex =
				particleIndex - body.compiled.particleStart;
			if(solveContext.canUseCorotationalTetPackets(
				body, localParticleIndex))
				solveContext.solveWithCorotationalTetPackets(
					body, localParticleIndex, observation);
			else
				solveContext.solve(
					body, localParticleIndex, observation);
		}
	}
	else
	{
		for(PxU32 packedIndex = packedBegin;
			packedIndex < packedEnd; packedIndex++)
		{
			const PxU32 particleIndex = packedParticleIndices[packedIndex];
			PX_ASSERT(particleIndex < numParticles);
			const PxU32 bodyIndex = particleBodyIndices[particleIndex];
			PX_ASSERT(bodyIndex < numSoftBodies);
			const AvbdSoftBody& body = softBodies[bodyIndex];
			PX_ASSERT(particleIndex >= body.compiled.particleStart &&
				particleIndex - body.compiled.particleStart <
					body.compiled.particleCount);
			solveContext.solve(
				body, particleIndex - body.compiled.particleStart,
				observation);
		}
	}
}

// P4.5.2a owns the parent-side lifetime of one published causal sweep.  It
// deliberately contains no dispatcher or Scene type: a caller may publish a
// layer to one or more children, then merge their private observations here
// before the next causal layer becomes visible.  The state is also used by
// the present one-worker path, so fixed-order reduction is exercised before
// task submission is introduced.
//
// This is intentionally narrower than the eventual AvbdSoftBodyStepState.
// It captures the non-reentrant particle-primal portion of an inner sweep;
// outer preparation, redetection, Chebyshev and dual/finalizer transitions
// remain serial until their enclosing state machine is extracted.  In
// particular, no child receives a reference to this object.
struct AvbdParticlePrimalCausalLayerState
{
	AvbdParticlePrimalCausalLayerState()
		: solveContext(NULL), softBodies(NULL), numSoftBodies(0),
		  particleBodyIndices(NULL), numParticles(0),
		  packedParticleIndices(NULL), layerOffsets(NULL),
		  layerCount(0), currentLayer(0), active(false)
	{
	}

	bool begin(
		const AvbdParticlePrimalSolveContext& inputSolveContext,
		const AvbdSoftBody* inputSoftBodies, PxU32 inputNumSoftBodies,
		const PxU32* inputParticleBodyIndices, PxU32 inputNumParticles,
		const PxU32* inputPackedParticleIndices,
		const PxU32* inputLayerOffsets, PxU32 inputLayerCount)
	{
		if(!inputSoftBodies || inputNumSoftBodies == 0 ||
			!inputParticleBodyIndices || inputNumParticles == 0 ||
			!inputPackedParticleIndices || !inputLayerOffsets ||
			inputLayerCount == 0)
			return false;
		if(inputLayerOffsets[0] != 0 ||
			inputLayerOffsets[inputLayerCount] != inputNumParticles)
			return false;
		for(PxU32 layer = 0; layer < inputLayerCount; layer++)
		{
			if(inputLayerOffsets[layer] > inputLayerOffsets[layer + 1] ||
				inputLayerOffsets[layer + 1] > inputNumParticles)
				return false;
		}
		solveContext = &inputSolveContext;
		softBodies = inputSoftBodies;
		numSoftBodies = inputNumSoftBodies;
		particleBodyIndices = inputParticleBodyIndices;
		numParticles = inputNumParticles;
		packedParticleIndices = inputPackedParticleIndices;
		layerOffsets = inputLayerOffsets;
		layerCount = inputLayerCount;
		currentLayer = 0;
		sweepObservation = AvbdParticlePrimalRangeObservation();
		active = true;
		return true;
	}

	PX_FORCE_INLINE bool hasPublishedLayer() const
	{
		return active && currentLayer < layerCount;
	}

	PX_FORCE_INLINE PxU32 getPublishedLayerIndex() const
	{
		PX_ASSERT(hasPublishedLayer());
		return currentLayer;
	}

	PX_FORCE_INLINE void getPublishedPackedRange(
		PxU32& packedBegin, PxU32& packedEnd) const
	{
		PX_ASSERT(hasPublishedLayer());
		packedBegin = layerOffsets[currentLayer];
		packedEnd = layerOffsets[currentLayer + 1];
	}

	// This is the one-worker reference consumer for a published layer.  The
	// future Scene task route performs this same range operation in children,
	// then calls completePublishedLayer() once on its fan-in parent.
	void solvePublishedLayerSerial()
	{
		PX_ASSERT(hasPublishedLayer());
		PxU32 packedBegin = 0;
		PxU32 packedEnd = 0;
		getPublishedPackedRange(packedBegin, packedEnd);
		AvbdParticlePrimalRangeObservation observation;
		avbdSolveParticlePrimalPackedRange(
			*solveContext, softBodies, numSoftBodies,
			particleBodyIndices, numParticles, packedParticleIndices,
			packedBegin, packedEnd, observation);
		completePublishedLayer(&observation, 1);
	}

	// Parent-only deterministic reduction.  Observation order is the stable
	// child-range order constructed by Scene, never task completion order.
	bool completePublishedLayer(
		const AvbdParticlePrimalRangeObservation* observations,
		PxU32 observationCount)
	{
		if(!hasPublishedLayer() || !observations || observationCount == 0)
			return false;
		for(PxU32 observationIndex = 0;
			observationIndex < observationCount; observationIndex++)
			sweepObservation.merge(observations[observationIndex]);
		currentLayer++;
		if(currentLayer == layerCount)
			active = false;
		return true;
	}

	PX_FORCE_INLINE const AvbdParticlePrimalRangeObservation&
	getSweepObservation() const
	{
		PX_ASSERT(!active);
		return sweepObservation;
	}

private:
	const AvbdParticlePrimalSolveContext* solveContext;
	const AvbdSoftBody* softBodies;
	PxU32 numSoftBodies;
	const PxU32* particleBodyIndices;
	PxU32 numParticles;
	const PxU32* packedParticleIndices;
	const PxU32* layerOffsets;
	PxU32 layerCount;
	PxU32 currentLayer;
	AvbdParticlePrimalRangeObservation sweepObservation;
	bool active;
};

enum class AvbdParticlePrimalDynamicAccessSource : PxU8
{
	eCONTACT,
	ePIN_OBJECTIVE
};

// One dynamically prepared objective support. Contact groups contain the
// complete query/surface support (not merely the nonzero Jacobian owner), so
// the eventual colored primal cannot race a live geometric read whose weight
// happens to be zero. Pin groups contain their complete point support.
struct AvbdParticlePrimalDynamicAccessGroup
{
	PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
	PxU8 particleCount;
	AvbdParticlePrimalDynamicAccessSource source;
	PxU16 padding;

	AvbdParticlePrimalDynamicAccessGroup()
		: particleCount(0),
		  source(AvbdParticlePrimalDynamicAccessSource::eCONTACT),
		  padding(0)
	{
		for(PxU32 i = 0; i < AVBD_CONTACT_MAX_PARTICLES; ++i)
			particleIndices[i] = PX_MAX_U32;
	}
};

struct AvbdSoftBodyWorkspace
{
	AvbdSoftContactWorkspace contact;
	PxArray<AvbdSoftContactParticleRef> contactIndices;
	PxArray<PxU32> contactStarts;
	PxArray<PxU32> contactCounts;
	PxArray<AvbdParticlePrimalDynamicAccessGroup>
		particlePrimalDynamicAccessGroups;
	PxArray<PxU32> particlePrimalDynamicConflictOffsets;
	PxArray<PxU32> particlePrimalDynamicConflictIndices;
	PxArray<PxU32> particlePrimalDynamicConflictCounts;
	PxArray<PxU32> particlePrimalBodyIndices;
	PxArray<PxU32> particlePrimalColors;
	PxArray<PxU32> particlePrimalColorCounts;
	PxArray<PxU32> particlePrimalColorOffsets;
	PxArray<PxU32> particlePrimalColorParticles;
	PxU32 particlePrimalColorCount;
	bool particlePrimalDynamicConflictValid;
	bool particlePrimalColorPlanValid;
	PxArray<PxVec3> chebyPrevPos;
	PxArray<PxVec3> chebyPrevPrevPos;
	PxArray<PxReal> selfCollisionSafetyBounds;
	PxArray<PxReal> bodySelfCollisionSafetyBounds;
	// Component fallback consumes the same pair records as native mixed
	// islands.  The indices remain aligned with the prepared contact stream;
	// the body mask is only a compact fan-out for per-particle bounds.
	PxArray<AvbdOgcPairState> componentOgcPairStates;
	PxArray<PxU32> componentOgcPairIndices;
	PxArray<PxU8> componentOgcSafetyBodyMask;
	PxArray<AvbdCompiledSoftVelocityObjective>
		compiledVelocityObjectives;
	PxArray<AvbdSoftComponentMomentumTarget>
		componentMomentumTargets;
	PxArray<AvbdSoftComponentFinalizeMode>
		componentFinalizeModes;
	// Marks component-fallback bodies with local endpoint-DCD recovery, so the
	// following velocity phase can clamp only those recovered contact supports.
	PxArray<PxU8> worldStaticEndpointRecoveredBodies;
	PxU64 growthEvents;
	PxU64 growthBytes;
	PxU32 peakContactIncidenceCount;
	PxU32 peakContactIncidenceCapacity;

	AvbdSoftBodyWorkspace()
		: particlePrimalColorCount(0),
		  particlePrimalDynamicConflictValid(false),
		  particlePrimalColorPlanValid(false),
		  growthEvents(0), growthBytes(0), peakContactIncidenceCount(0),
		  peakContactIncidenceCapacity(0)
	{
	}

	void reserve(PxU32 numParticles, PxU32 contactCapacity,
		AvbdParticlePrimalSchedule particlePrimalSchedule =
			AvbdParticlePrimalSchedule::eDEFAULT)
	{
		contact.reserve(contactCapacity);
		contact.reserveSweepScratch(
			contact.rigidConvexForwardOwnerScratch, numParticles);
		contact.reserveSweepScratch(
			contact.rigidTriangleSurfaceForwardOwnerScratch, numParticles);
		// A contact can contribute up to six unique particle incidences
		// (three query and three deformable-surface target vertices).
		const PxU32 contactIndexCapacity =
			contactCapacity <= PX_MAX_U32 / 6
				? contactCapacity * AVBD_CONTACT_MAX_PARTICLES : PX_MAX_U32;
		contactIndices.reserve(contactIndexCapacity);
		contactStarts.reserve(numParticles + 1);
		contactCounts.reserve(numParticles);
		// Avoid charging the authoritative serial default for graph scratch, but
		// when a colored schedule is requested reserve one complete dynamic
		// support clique per lifecycle-budgeted contact or particle-owned group.
		// Embedded query/target contacts can use every one of the declared
		// AVBD_CONTACT_MAX_PARTICLES slots, not just a triangle's six vertices.
		// A denser unexpected epoch is rejected before publication and takes the
		// serial fallback instead of growing worker-visible storage.
		const AvbdParticlePrimalSchedule resolvedParticlePrimalSchedule =
			particlePrimalSchedule == AvbdParticlePrimalSchedule::eDEFAULT
				? avbdGetParticlePrimalSchedule() : particlePrimalSchedule;
		if(avbdValidateParticlePrimalAccessPlan() ||
			avbdUsesColoredParticlePrimalSchedule(
				resolvedParticlePrimalSchedule))
		{
			const PxU64 dynamicGroupCapacity =
				PxU64(contactCapacity) + PxU64(numParticles);
			particlePrimalDynamicAccessGroups.reserve(
				dynamicGroupCapacity > PX_MAX_U32
					? PX_MAX_U32 : PxU32(dynamicGroupCapacity));
			particlePrimalDynamicConflictOffsets.reserve(numParticles + 1);
			particlePrimalDynamicConflictCounts.reserve(numParticles);
			// A contact support may span all query/target slots, whereas a
			// particle-owned pin group has at most three vertices.  Do not charge
			// every particle for a 24-way contact clique: that would make a large
			// contact-free body exceed the fixed graph budget before it could
			// publish its structural color plan.
			const PxU64 dynamicConflictCapacity =
				PxU64(contactCapacity) *
					PxU64(AVBD_CONTACT_MAX_PARTICLES) *
					PxU64(AVBD_CONTACT_MAX_PARTICLES - 1u) +
				PxU64(numParticles) * 3u * 2u;
			const PxU32 maxDynamicConflictIndices =
				(2u * 1024u * 1024u) / sizeof(PxU32);
			particlePrimalDynamicConflictIndices.reserve(
				dynamicConflictCapacity >= maxDynamicConflictIndices
					? maxDynamicConflictIndices :
						PxU32(dynamicConflictCapacity));
			particlePrimalBodyIndices.reserve(numParticles);
			particlePrimalColors.reserve(numParticles);
			particlePrimalColorCounts.reserve(numParticles);
			particlePrimalColorOffsets.reserve(numParticles + 1);
			particlePrimalColorParticles.reserve(numParticles);
		}
		chebyPrevPos.reserve(numParticles);
		chebyPrevPrevPos.reserve(numParticles);
		selfCollisionSafetyBounds.reserve(numParticles);
		bodySelfCollisionSafetyBounds.reserve(numParticles);
		componentOgcPairStates.reserve(contactCapacity);
		componentOgcPairIndices.reserve(contactCapacity);
		componentOgcSafetyBodyMask.reserve(numParticles);
		compiledVelocityObjectives.reserve(contactCapacity);
		componentMomentumTargets.reserve(numParticles);
		componentFinalizeModes.reserve(numParticles);
	}

	template<typename T, typename Alloc>
	void resize(PxArray<T, Alloc>& array, PxU32 size)
	{
		if(size > array.capacity())
		{
			growthEvents++;
			growthBytes +=
				PxU64(size - array.capacity()) * sizeof(T);
		}
		array.resize(size);
	}

	void beginStep()
	{
		growthEvents = 0;
		growthBytes = 0;
		peakContactIncidenceCount = 0;
		peakContactIncidenceCapacity = 0;
		particlePrimalColorCount = 0;
		particlePrimalDynamicConflictValid = false;
		particlePrimalColorPlanValid = false;
		contact.beginStep();
	}

	void reset()
	{
		contact.reset();
		contactIndices.reset();
		contactStarts.reset();
		contactCounts.reset();
		particlePrimalDynamicAccessGroups.reset();
		particlePrimalDynamicConflictOffsets.reset();
		particlePrimalDynamicConflictIndices.reset();
		particlePrimalDynamicConflictCounts.reset();
		particlePrimalBodyIndices.reset();
		particlePrimalColors.reset();
		particlePrimalColorCounts.reset();
		particlePrimalColorOffsets.reset();
		particlePrimalColorParticles.reset();
		chebyPrevPos.reset();
		chebyPrevPrevPos.reset();
		selfCollisionSafetyBounds.reset();
		bodySelfCollisionSafetyBounds.reset();
		componentOgcPairStates.reset();
		componentOgcPairIndices.reset();
		componentOgcSafetyBodyMask.reset();
		compiledVelocityObjectives.reset();
		componentMomentumTargets.reset();
		componentFinalizeModes.reset();
		worldStaticEndpointRecoveredBodies.reset();
		beginStep();
	}
};

PX_FORCE_INLINE bool avbdSoftBodyContainsParticle(
	const AvbdSoftBody& body, PxU32 particleIndex, PxU32 numParticles)
{
	return body.compiled.particleStart <= numParticles &&
		body.compiled.particleCount <=
			numParticles - body.compiled.particleStart &&
		particleIndex >= body.compiled.particleStart &&
		particleIndex - body.compiled.particleStart <
			body.compiled.particleCount;
}

inline void avbdSnapshotOuterPositionsScalar(
	AvbdSoftParticle* particles, PxU32 numParticles,
	PxReal* selfCollisionSafetyBounds)
{
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
	{
		particles[particleIndex].outerPosition =
			particles[particleIndex].position;
		selfCollisionSafetyBounds[particleIndex] = PX_MAX_F32;
	}
}

PX_FORCE_INLINE bool avbdCanReuseComponentOgcEpoch(
	const AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftParticle* particles)
{
	if(!contacts || numContacts == 0 || !softBodies ||
		numSoftBodies == 0 || !particles)
		return false;
	for(PxU32 contactIndex = 0; contactIndex < numContacts; ++contactIndex)
	{
		const AvbdSoftContactGeometry& geometry =
			contacts[contactIndex].geometry;
		if(geometry.queryBodyIndex >= numSoftBodies)
			return false;
		const bool softPair =
			geometry.source.type == AvbdSoftContactSource::eSOFT_SURFACE &&
			geometry.targetKind ==
				AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE &&
			geometry.targetIndex < numSoftBodies &&
			geometry.queryBodyIndex != geometry.targetIndex;
		const bool worldStatic =
			geometry.targetKind ==
				AvbdSoftContactTargetKind::eWORLD_STATIC &&
			(geometry.source.type == AvbdSoftContactSource::eGROUND ||
			 geometry.source.type == AvbdSoftContactSource::eRIGID_SDF);
		if(!softPair && !worldStatic)
			return false;
		const PxReal normalLengthSq = geometry.normal.magnitudeSquared();
		if(!PxIsFinite(normalLengthSq) || normalLengthSq <= 1.0e-12f)
			return false;
		const PxVec3 queryPoint = avbdGetSoftContactQueryPoint(
			geometry, particles);
		const PxVec3 surfacePoint = softPair
			? avbdGetSoftContactSurfacePoint(geometry, particles)
			: geometry.surfacePoint;
		const PxReal physicalGap = (queryPoint - surfacePoint).dot(
			geometry.normal * PxRecipSqrt(normalLengthSq));
		if(!queryPoint.isFinite() || !surfacePoint.isFinite() ||
			!PxIsFinite(physicalGap) || physicalGap < -1.0e-5f)
			return false;
	}
	return true;
}

PX_FORCE_INLINE void avbdBuildComponentOgcPairStates(
	const AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdSoftParticle* particles, AvbdSoftBodyWorkspace& workspace)
{
	workspace.componentOgcPairStates.clear();
	workspace.componentOgcPairIndices.resize(numContacts);
	for(PxU32 contactIndex = 0; contactIndex < numContacts; ++contactIndex)
	{
		workspace.componentOgcPairIndices[contactIndex] = PX_MAX_U32;
		const AvbdSoftContactGeometry& geometry =
			contacts[contactIndex].geometry;
		PxU32 pairIndex = PX_MAX_U32;
		for(PxU32 candidateIndex = 0;
			candidateIndex < workspace.componentOgcPairStates.size();
			++candidateIndex)
		{
			const AvbdOgcPairState& candidate =
				workspace.componentOgcPairStates[candidateIndex];
			if(candidate.sourceType == geometry.source.type &&
				candidate.targetKind == geometry.targetKind &&
				candidate.sourceBodyIndex == geometry.queryBodyIndex &&
				candidate.targetBodyIndex == geometry.targetIndex &&
				candidate.primitiveKey == geometry.source.primitiveKey)
			{
				pairIndex = candidateIndex;
				break;
			}
		}
		if(pairIndex == PX_MAX_U32)
		{
			pairIndex = workspace.componentOgcPairStates.size();
			AvbdOgcPairState pair;
			pair.sourceType = geometry.source.type;
			pair.targetKind = geometry.targetKind;
			pair.sourceBodyIndex = geometry.queryBodyIndex;
			pair.targetBodyIndex = geometry.targetIndex;
			pair.primitiveKey = geometry.source.primitiveKey;
			pair.epoch = 1u;
			pair.active = true;
			workspace.componentOgcPairStates.pushBack(pair);
		}

		AvbdOgcPairState& pair =
			workspace.componentOgcPairStates[pairIndex];
		++pair.contactCount;
		const PxReal normalLengthSq = geometry.normal.magnitudeSquared();
		const PxVec3 queryPoint = avbdGetSoftContactQueryPoint(
			geometry, particles);
		const PxVec3 surfacePoint =
			geometry.targetKind ==
				AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE
				? avbdGetSoftContactSurfacePoint(geometry, particles)
				: geometry.surfacePoint;
		if(normalLengthSq > 1.0e-12f && PxIsFinite(normalLengthSq) &&
			queryPoint.isFinite() && surfacePoint.isFinite())
		{
			const PxVec3 normal = geometry.normal *
				PxRecipSqrt(normalLengthSq);
			const PxReal gap = (queryPoint - surfacePoint).dot(normal);
			if(PxIsFinite(gap) &&
				(pair.representativeContact == PX_MAX_U32 ||
				 gap < pair.referenceGap))
			{
				pair.referenceGap = gap;
				pair.safetyGap = gap;
				pair.minimumGap = gap;
				pair.representativeContact = contactIndex;
				pair.representativeNormal = normal;
				pair.representativeGap = gap;
			}
		}
		workspace.componentOgcPairIndices[contactIndex] = pairIndex;
	}
}

// A component OGC epoch starts from one complete current-pose DCD manifold.
// It may reuse that manifold only when each prepared row has a static target
// or another deformable surface, and all moving support bodies remain inside
// a strict fraction of their *current physical separation*.  Collision
// proxies are convex embeddings of simulation particles, therefore this is a
// conservative pair-wise bound for every prepared support point.  Dynamic
// rigid targets deliberately stay out of this path: their 6DOF target motion
// is owned by the native shared OgcPairState solver.
//
// This is the OGC scheduling rule, not a microstep: candidate motion is
// clipped inside one same-time epoch, and touching the bound publishes a new
// DCD epoch before another outer solve may proceed.
PX_FORCE_INLINE bool avbdApplyComponentOgcEpochSafetyBounds(
	const AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftParticle* particles,
	PxReal contactRadius, PxReal safetyRelax,
	PxReal* particleSafetyBounds, PxU32 numParticles,
	AvbdSoftBodyWorkspace& workspace)
{
	if(!contacts || numContacts == 0 || !softBodies ||
		numSoftBodies == 0 || !particles || !particleSafetyBounds)
		return false;
	if(!avbdCanReuseComponentOgcEpoch(
		contacts, numContacts, softBodies, numSoftBodies, particles))
		return false;
	workspace.resize(workspace.componentOgcSafetyBodyMask, numSoftBodies);
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
		workspace.componentOgcSafetyBodyMask[bodyIndex] = 0u;

	avbdBuildComponentOgcPairStates(
		contacts, numContacts, particles, workspace);
	if(workspace.componentOgcPairStates.empty() ||
		workspace.componentOgcPairIndices.size() != numContacts)
		return false;
	const PxReal maximumSafetyFraction =
		PxClamp(safetyRelax, 1.0e-4f, 0.499f);
	const PxReal maximumSafetyDistance =
		PxMax(contactRadius, 1.0e-6f);
	bool appliedProximitySafetyBound = false;
	const auto applyBodySafetyDistance = [&](PxU32 bodyIndex,
		PxReal safetyDistance)
	{
		if(bodyIndex >= numSoftBodies)
			return false;
		const AvbdSoftBody& body = softBodies[bodyIndex];
		if(body.compiled.particleStart > numParticles ||
			body.compiled.particleCount >
				numParticles - body.compiled.particleStart)
			return false;
		for(PxU32 localIndex = 0;
			localIndex < body.compiled.particleCount; ++localIndex)
		{
			const PxU32 particleIndex = body.compiled.particleStart + localIndex;
			particleSafetyBounds[particleIndex] = PxMin(
				particleSafetyBounds[particleIndex], safetyDistance);
		}
		workspace.componentOgcSafetyBodyMask[bodyIndex] = 1u;
		return true;
	};
	for(PxU32 pairIndex = 0;
		pairIndex < workspace.componentOgcPairStates.size(); ++pairIndex)
	{
		AvbdOgcPairState& pair = workspace.componentOgcPairStates[pairIndex];
		if(!pair.active || pair.sourceBodyIndex >= numSoftBodies ||
			!PxIsFinite(pair.safetyGap) || pair.safetyGap < -1.0e-5f)
			return false;
		if(pair.representativeContact >= numContacts)
			return false;
		const AvbdSoftContactGeometry& representativeGeometry =
			contacts[pair.representativeContact].geometry;
		const PxVec3 representativeQueryPoint =
			avbdGetSoftContactQueryPoint(
				representativeGeometry, particles);
		const PxVec3 representativeSurfacePoint =
			avbdGetSoftContactSurfacePoint(
				representativeGeometry, particles);
		if(!representativeQueryPoint.isFinite() ||
			!representativeSurfacePoint.isFinite())
			return false;
		const PxReal normalConstraint =
			avbdEvaluateSoftContactNormalConstraint(
				representativeGeometry, representativeQueryPoint,
				representativeSurfacePoint);
		if(!PxIsFinite(normalConstraint))
			return false;

		// An active unilateral manifold already owns motion toward its contact
		// plane.  Turning its zero normal gap into an isotropic per-particle
		// trust region clamps tangential motion, rotation, and deformation to
		// roughly one micron and makes a resting soft body appear asleep.  OGC
		// safety bounds are needed only while a separated pair is approaching;
		// once active, the prepared contact rows and terminal current-pose DCD
		// own non-penetration without freezing the other directions.
		const PxReal activeTolerance = PxMax(
			1.0e-5f,
			1.0e-4f * PxMax(representativeGeometry.margin, 0.0f));
		if(normalConstraint <= activeTolerance)
		{
			pair.remainingSafeDisplacement = 0.0f;
			pair.accumulatedRelativeDisplacement = 0.0f;
			pair.refreshRequested = false;
			continue;
		}
		const PxReal safetyDistance = PxMax(1.0e-6f,
			maximumSafetyFraction * PxMin(maximumSafetyDistance,
				normalConstraint));
		pair.remainingSafeDisplacement = safetyDistance;
		pair.accumulatedRelativeDisplacement = 0.0f;
		pair.refreshRequested = false;
		if(!applyBodySafetyDistance(pair.sourceBodyIndex, safetyDistance))
			return false;
		if(pair.targetKind ==
			AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE &&
			!applyBodySafetyDistance(pair.targetBodyIndex, safetyDistance))
			return false;
		appliedProximitySafetyBound = true;
	}
	return appliedProximitySafetyBound;
}

// P4.5.2b state-machine seam.  Contact redetection may change component
// finalization ownership and the compiled kinematic velocity objectives.  Keep
// that operation independent of avbdStepSoftBodies()'s stack lambdas so a
// future persistent step state invokes exactly the same canonical update at
// its initial and between-outer redetection transitions.
PX_FORCE_INLINE PxU32 avbdFindSoftComponentBodyIndex(
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxU32 particleIndex)
{
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		const PxU32 particleStart =
			softBodies[bodyIndex].compiled.particleStart;
		const PxU32 particleCount =
			softBodies[bodyIndex].compiled.particleCount;
		if(particleIndex >= particleStart &&
			particleIndex - particleStart < particleCount)
			return bodyIndex;
	}
	return PX_MAX_U32;
}

PX_FORCE_INLINE void avbdMergeSoftComponentFinalizeMode(
	PxArray<AvbdSoftComponentFinalizeMode>& componentFinalizeModes,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxU32 particleIndex, AvbdSoftComponentFinalizeMode incoming)
{
	const PxU32 bodyIndex = avbdFindSoftComponentBodyIndex(
		softBodies, numSoftBodies, particleIndex);
	if(bodyIndex == PX_MAX_U32)
		return;
	AvbdSoftComponentFinalizeMode& current =
		componentFinalizeModes[bodyIndex];
	if(current == incoming ||
		current == AvbdSoftComponentFinalizeMode::eUNSUPPORTED)
		return;
	if(current == AvbdSoftComponentFinalizeMode::eMOMENTUM)
		current = incoming;
	else
		current = AvbdSoftComponentFinalizeMode::eUNSUPPORTED;
}

inline void avbdCompileSoftVelocityObjectives(
	PxArray<AvbdCompiledSoftVelocityObjective>& compiledVelocityObjectives,
	PxArray<AvbdSoftComponentFinalizeMode>& componentFinalizeModes,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftContact* sourceContacts, PxU32 sourceContactCount)
{
	for(PxU32 sourceIndex = 0;
		sourceIndex < sourceContactCount; sourceIndex++)
	{
		const AvbdSoftContact& source = sourceContacts[sourceIndex];
		const AvbdSoftContactGeometry& geometry = source.geometry;
		AvbdSoftComponentFinalizeMode incoming =
			AvbdSoftComponentFinalizeMode::eUNSUPPORTED;
		if(geometry.velocityOwner ==
			AvbdVelocityObjectiveOwner::PositionAL)
		{
			incoming = geometry.hasKinematicRigidTarget()
				? AvbdSoftComponentFinalizeMode::eUNSUPPORTED
				: AvbdSoftComponentFinalizeMode::ePOSITION_OWNED;
		}
		else if(geometry.velocityOwner ==
			AvbdVelocityObjectiveOwner::ComponentFinalize)
		{
			incoming = geometry.hasKinematicRigidTarget()
				? AvbdSoftComponentFinalizeMode::eKINEMATIC_CONTACT
				: AvbdSoftComponentFinalizeMode::eUNSUPPORTED;
		}
		if(geometry.hasWeightedQueryPoint())
		{
			for(PxU32 pointIndex = 0;
				pointIndex < geometry.queryPoint.count; pointIndex++)
				avbdMergeSoftComponentFinalizeMode(
					componentFinalizeModes, softBodies, numSoftBodies,
					geometry.queryPoint.particleIndices[pointIndex], incoming);
		}
		else if(geometry.hasBarycentricQueryPoint())
		{
			for(PxU32 vertexIndex = 0; vertexIndex < 3; vertexIndex++)
			{
				if(geometry.queryParticleIndices[vertexIndex] ==
					PX_MAX_U32)
					break;
				avbdMergeSoftComponentFinalizeMode(
					componentFinalizeModes, softBodies, numSoftBodies,
					geometry.queryParticleIndices[vertexIndex], incoming);
			}
		}
		else
			avbdMergeSoftComponentFinalizeMode(
				componentFinalizeModes, softBodies, numSoftBodies,
				geometry.particleIdx, incoming);
		if(geometry.hasWeightedTargetPoint())
		{
			for(PxU32 pointIndex = 0;
				pointIndex < geometry.targetPoint.count; pointIndex++)
				avbdMergeSoftComponentFinalizeMode(
					componentFinalizeModes, softBodies, numSoftBodies,
					geometry.targetPoint.particleIndices[pointIndex],
					geometry.velocityOwner ==
						AvbdVelocityObjectiveOwner::PositionAL
						? AvbdSoftComponentFinalizeMode::ePOSITION_OWNED
						: AvbdSoftComponentFinalizeMode::eUNSUPPORTED);
		}
		else if(geometry.hasDeformableSurfaceTarget())
		{
			for(PxU32 vertexIndex = 0; vertexIndex < 3; vertexIndex++)
			{
				avbdMergeSoftComponentFinalizeMode(
					componentFinalizeModes, softBodies, numSoftBodies,
					geometry.surfaceParticleIndices[vertexIndex],
					geometry.velocityOwner ==
						AvbdVelocityObjectiveOwner::PositionAL
						? AvbdSoftComponentFinalizeMode::ePOSITION_OWNED
						: AvbdSoftComponentFinalizeMode::eUNSUPPORTED);
			}
		}
		if(geometry.velocityOwner !=
				AvbdVelocityObjectiveOwner::ComponentFinalize ||
			!geometry.hasKinematicRigidTarget())
			continue;
		const PxU32 representativeParticle = geometry.hasWeightedQueryPoint()
			? geometry.queryPoint.particleIndices[0]
			: geometry.particleIdx;
		const PxU32 bodyIndex = geometry.queryBodyIndex < numSoftBodies
			? geometry.queryBodyIndex
			: avbdFindSoftComponentBodyIndex(
				softBodies, numSoftBodies, representativeParticle);
		if(bodyIndex == PX_MAX_U32)
			continue;
		AvbdCompiledSoftVelocityObjective objective;
		objective.owner = geometry.velocityOwner;
		objective.source = geometry.source;
		objective.bodyIndex = bodyIndex;
		objective.particleIndex = representativeParticle;
		if(geometry.hasWeightedQueryPoint())
			objective.queryPoint = geometry.queryPoint;
		else if(geometry.hasBarycentricQueryPoint())
		{
			for(PxU32 queryVertex = 0; queryVertex < 3; queryVertex++)
			{
				if(geometry.queryParticleIndices[queryVertex] == PX_MAX_U32)
					break;
				objective.queryPoint.appendMerged(
					geometry.queryParticleIndices[queryVertex],
					geometry.queryWeights[queryVertex]);
			}
		}
		else
			objective.queryPoint.setVertex(geometry.particleIdx);
		objective.normal = geometry.normal;
		objective.surfacePoint = geometry.surfacePoint;
		objective.previousSurfacePoint =
			geometry.kinematicSurfacePointPrevious;
		bool replaced = false;
		for(PxU32 compiledIndex = 0;
			compiledIndex < compiledVelocityObjectives.size();
			compiledIndex++)
		{
			AvbdCompiledSoftVelocityObjective& compiled =
				compiledVelocityObjectives[compiledIndex];
			if(compiled.particleIndex == objective.particleIndex &&
				compiled.source == objective.source)
			{
				compiled = objective;
				replaced = true;
				break;
			}
		}
		if(!replaced)
			compiledVelocityObjectives.pushBack(objective);
	}
}

PX_FORCE_INLINE void avbdAssignVelocityTangentOwners(
	AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftParticle* particles, PxU32 numParticles);

// A component solve can move the simulation particles after its last outer
// redetection.  A cached contact row is not a valid final-pose OGC manifold:
// it cannot see a newly entered ground/static-box feature and it may carry a
// normal from a pose that the material solve has already left.  Refresh the
// contact stream once at the same simulation time, immediately before the
// terminal current-pose recovery and velocity reconstruction.
//
// This is deliberately a DCD-only epoch.  The callback owns proxy expansion
// and does not advance time or invoke any swept/CCD query.  Replacing the
// velocity objectives is essential: they are compiled from contact geometry,
// so retaining entries from the superseded manifold would let stale normals
// affect the final velocity phase.
PX_FORCE_INLINE bool avbdRefreshComponentTerminalOgcEpoch(
	AvbdSoftParticle* particles, PxU32 numParticles,
	AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdContactRedetectFn redetectFn,
	PxArray<AvbdSoftContact>* contactsArray,
	void* redetectUserData,
	AvbdSoftContact*& contacts, PxU32& numContacts,
	AvbdSoftBodyWorkspace& workspace)
{
	if(!particles || !softBodies || numParticles == 0 ||
		numSoftBodies == 0 || !redetectFn || !contactsArray)
		return false;

	redetectFn(particles, numParticles, softBodies, numSoftBodies,
		*contactsArray, redetectUserData);
	contacts = contactsArray->begin();
	numContacts = contactsArray->size();
	// A terminal current-pose epoch replaces the contact stream.  Re-apply
	// typed tangent ownership before compiling its velocity objectives; otherwise
	// eligible world-static rows silently fall back to a positional sticking
	// spring for the last solve phase.
	avbdAssignVelocityTangentOwners(
		contacts, numContacts, softBodies, numSoftBodies,
		particles, numParticles);
	workspace.compiledVelocityObjectives.clear();
	workspace.resize(workspace.componentFinalizeModes, numSoftBodies);
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
		workspace.componentFinalizeModes[bodyIndex] =
			AvbdSoftComponentFinalizeMode::eMOMENTUM;
	avbdCompileSoftVelocityObjectives(
		workspace.compiledVelocityObjectives,
		workspace.componentFinalizeModes, softBodies, numSoftBodies,
		contacts, numContacts);
	return true;
}

// P4.2 planning path. It composes P4.1's per-body structural CSR with the
// contact/objective supports prepared for this redetection epoch. The ordered
// schedule preserves the serial authority; the relaxed schedule publishes the
// same complete conflict proof to the Scene's per-color fan-in.
inline void avbdBuildParticlePrimalColorPlan(
	AvbdSoftBodyWorkspace& workspace,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftContact* contacts, PxU32 numContacts,
	PxU32 numParticles,
	AvbdParticlePrimalSchedule particlePrimalSchedule)
{
	static const PxU32 eMAX_DYNAMIC_CONFLICT_INDICES =
		(2u * 1024u * 1024u) / sizeof(PxU32);
	workspace.particlePrimalDynamicConflictValid = false;
	workspace.particlePrimalColorPlanValid = false;
	workspace.particlePrimalColorCount = 0;
	// A future color task must never grow worker-visible scratch while it is
	// preparing an epoch.  The Scene reserves these arrays at lifecycle
	// boundaries; a smaller dynamic budget simply leaves this optional plan
	// unpublished so the authoritative serial primal remains the fallback.
	if(workspace.particlePrimalBodyIndices.capacity() < numParticles ||
		workspace.particlePrimalColors.capacity() < numParticles ||
		workspace.particlePrimalColorCounts.capacity() < numParticles ||
		workspace.particlePrimalColorOffsets.capacity() <
			numParticles + 1 ||
		workspace.particlePrimalColorParticles.capacity() < numParticles ||
		workspace.particlePrimalDynamicConflictOffsets.capacity() <
			numParticles + 1 ||
		workspace.particlePrimalDynamicConflictCounts.capacity() <
			numParticles)
		return;
	workspace.resize(workspace.particlePrimalBodyIndices, numParticles);
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
		workspace.particlePrimalBodyIndices[particleIndex] = PX_MAX_U32;

	bool valid = true;
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		const AvbdSoftBodyCompiledData& compiled =
			softBodies[bodyIndex].compiled;
		if(!compiled.validateParticlePrimalStructuralAccessDescriptor() ||
			compiled.particleStart > numParticles ||
			compiled.particleCount >
				numParticles - compiled.particleStart)
		{
			valid = false;
			break;
		}
		for(PxU32 localIndex = 0;
			localIndex < compiled.particleCount; localIndex++)
		{
			const PxU32 particleIndex =
				compiled.particleStart + localIndex;
			if(workspace.particlePrimalBodyIndices[particleIndex] !=
				PX_MAX_U32)
			{
				valid = false;
				break;
			}
			workspace.particlePrimalBodyIndices[particleIndex] = bodyIndex;
		}
		if(!valid)
			break;
	}
	if(valid)
	{
		for(PxU32 particleIndex = 0; particleIndex < numParticles;
			particleIndex++)
		{
			if(workspace.particlePrimalBodyIndices[particleIndex] ==
				PX_MAX_U32)
			{
				valid = false;
				break;
			}
		}
	}
	if(!valid)
		return;

	PxU64 groupCount64 = numContacts;
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		const PxArray<AvbdCompiledSoftObjective>& objectives =
			softBodies[bodyIndex].runtime.compiledObjectives;
		for(PxU32 objectiveIndex = 0;
			objectiveIndex < objectives.size(); objectiveIndex++)
		{
			if(avbdIsPinPositionOwner(objectives[objectiveIndex].owner))
				groupCount64++;
		}
	}
	if(groupCount64 > PX_MAX_U32)
		return;
	if(workspace.particlePrimalDynamicAccessGroups.capacity() <
		PxU32(groupCount64))
		return;
	workspace.resize(workspace.particlePrimalDynamicAccessGroups,
		PxU32(groupCount64));

	auto writeGroup = [&workspace, numParticles](
		PxU32 groupIndex,
		AvbdParticlePrimalDynamicAccessSource source,
		const PxU32* inputIndices, PxU32 inputCount) -> bool
	{
		if(groupIndex >=
			workspace.particlePrimalDynamicAccessGroups.size() ||
			inputCount > AVBD_CONTACT_MAX_PARTICLES)
			return false;
		AvbdParticlePrimalDynamicAccessGroup& group =
			workspace.particlePrimalDynamicAccessGroups[groupIndex];
		group = AvbdParticlePrimalDynamicAccessGroup();
		group.source = source;
		for(PxU32 inputIndex = 0; inputIndex < inputCount;
			inputIndex++)
		{
			const PxU32 particleIndex = inputIndices[inputIndex];
			if(particleIndex >= numParticles)
				return false;
			bool unique = true;
			for(PxU32 previous = 0;
				previous < group.particleCount; previous++)
			{
				if(group.particleIndices[previous] == particleIndex)
				{
					unique = false;
					break;
				}
			}
			if(unique)
				group.particleIndices[group.particleCount++] = particleIndex;
		}
		PxSort(group.particleIndices, group.particleCount);
		return true;
	};

	PxU32 groupCursor = 0;
	for(PxU32 contactIndex = 0; contactIndex < numContacts;
		contactIndex++)
	{
		PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
		const PxU32 particleCount =
			avbdCollectSoftContactParticleIndices(
				contacts[contactIndex].geometry, particleIndices);
		if(!writeGroup(groupCursor++,
			AvbdParticlePrimalDynamicAccessSource::eCONTACT,
			particleIndices, particleCount))
			return;
	}
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		const PxArray<AvbdCompiledSoftObjective>& objectives =
			softBodies[bodyIndex].runtime.compiledObjectives;
		for(PxU32 objectiveIndex = 0;
			objectiveIndex < objectives.size(); objectiveIndex++)
		{
			const AvbdCompiledSoftObjective& objective =
				objectives[objectiveIndex];
			if(!avbdIsPinPositionOwner(objective.owner))
				continue;
			PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
			const PxU32 particleCount = objective.point.particleCount;
			if(particleCount > 3)
				return;
			for(PxU32 pointIndex = 0;
				pointIndex < particleCount; pointIndex++)
				particleIndices[pointIndex] =
					objective.point.particleIndices[pointIndex];
			if(!writeGroup(groupCursor++,
				AvbdParticlePrimalDynamicAccessSource::ePIN_OBJECTIVE,
				particleIndices, particleCount))
				return;
		}
	}
	PX_ASSERT(groupCursor ==
		workspace.particlePrimalDynamicAccessGroups.size());

	workspace.resize(workspace.particlePrimalDynamicConflictCounts,
		numParticles);
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
		workspace.particlePrimalDynamicConflictCounts[particleIndex] = 0;
	for(PxU32 groupIndex = 0;
		groupIndex < workspace.particlePrimalDynamicAccessGroups.size();
		groupIndex++)
	{
		const AvbdParticlePrimalDynamicAccessGroup& group =
			workspace.particlePrimalDynamicAccessGroups[groupIndex];
		if(group.particleCount < 2)
			continue;
		const PxU32 additions = group.particleCount - 1;
		for(PxU32 participant = 0;
			participant < group.particleCount; participant++)
		{
			const PxU32 particleIndex =
				group.particleIndices[participant];
			if(workspace.particlePrimalDynamicConflictCounts[
				particleIndex] > PX_MAX_U32 - additions)
				return;
			workspace.particlePrimalDynamicConflictCounts[
				particleIndex] += additions;
		}
	}
	workspace.resize(workspace.particlePrimalDynamicConflictOffsets,
		numParticles + 1);
	workspace.particlePrimalDynamicConflictOffsets[0] = 0;
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
	{
		const PxU32 begin =
			workspace.particlePrimalDynamicConflictOffsets[particleIndex];
		const PxU32 count =
			workspace.particlePrimalDynamicConflictCounts[particleIndex];
		if(begin > eMAX_DYNAMIC_CONFLICT_INDICES ||
			count > eMAX_DYNAMIC_CONFLICT_INDICES - begin)
			return;
		workspace.particlePrimalDynamicConflictOffsets[particleIndex + 1] =
			begin + count;
	}
	const PxU32 dynamicConflictCount =
		workspace.particlePrimalDynamicConflictOffsets[numParticles];
	if(workspace.particlePrimalDynamicConflictIndices.capacity() <
		dynamicConflictCount)
		return;
	workspace.resize(workspace.particlePrimalDynamicConflictIndices,
		dynamicConflictCount);
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
		workspace.particlePrimalDynamicConflictCounts[particleIndex] = 0;
	for(PxU32 groupIndex = 0;
		groupIndex < workspace.particlePrimalDynamicAccessGroups.size();
		groupIndex++)
	{
		const AvbdParticlePrimalDynamicAccessGroup& group =
			workspace.particlePrimalDynamicAccessGroups[groupIndex];
		for(PxU32 source = 0; source < group.particleCount; source++)
		{
			const PxU32 particleIndex = group.particleIndices[source];
			for(PxU32 target = 0; target < group.particleCount; target++)
			{
				if(source == target)
					continue;
				const PxU32 writeIndex =
					workspace.particlePrimalDynamicConflictOffsets[
						particleIndex] +
					workspace.particlePrimalDynamicConflictCounts[
						particleIndex]++;
				PX_ASSERT(writeIndex <
					workspace.particlePrimalDynamicConflictIndices.size());
				workspace.particlePrimalDynamicConflictIndices[writeIndex] =
					group.particleIndices[target];
			}
		}
	}

	// Sort and compact each local range in place. The write cursor never passes
	// an unread source range because compaction only removes duplicates.
	PxU32 compactWrite = 0;
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
	{
		const PxU32 begin =
			workspace.particlePrimalDynamicConflictOffsets[particleIndex];
		const PxU32 end =
			workspace.particlePrimalDynamicConflictOffsets[
				particleIndex + 1];
		PxSort(
			workspace.particlePrimalDynamicConflictIndices.begin() + begin,
			end - begin);
		workspace.particlePrimalDynamicConflictOffsets[particleIndex] =
			compactWrite;
		for(PxU32 conflictIndex = begin; conflictIndex < end;
			conflictIndex++)
		{
			const PxU32 conflict =
				workspace.particlePrimalDynamicConflictIndices[conflictIndex];
			if(conflict == particleIndex ||
				(compactWrite >
					workspace.particlePrimalDynamicConflictOffsets[
						particleIndex] &&
					workspace.particlePrimalDynamicConflictIndices[
						compactWrite - 1] == conflict))
				continue;
			workspace.particlePrimalDynamicConflictIndices[compactWrite++] =
				conflict;
		}
	}
	workspace.particlePrimalDynamicConflictOffsets[numParticles] =
		compactWrite;
	workspace.particlePrimalDynamicConflictIndices.resize(compactWrite);

	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
	{
		const PxU32 begin =
			workspace.particlePrimalDynamicConflictOffsets[particleIndex];
		const PxU32 end =
			workspace.particlePrimalDynamicConflictOffsets[
				particleIndex + 1];
		if(begin > end || end >
			workspace.particlePrimalDynamicConflictIndices.size())
			return;
		for(PxU32 conflictIndex = begin; conflictIndex < end;
			conflictIndex++)
		{
			const PxU32 conflict =
				workspace.particlePrimalDynamicConflictIndices[conflictIndex];
			if(conflict >= numParticles || conflict == particleIndex ||
				(conflictIndex > begin &&
					workspace.particlePrimalDynamicConflictIndices[
						conflictIndex - 1] >= conflict))
				return;
			bool reverseFound = false;
			for(PxU32 reverseIndex =
				workspace.particlePrimalDynamicConflictOffsets[conflict];
				reverseIndex <
					workspace.particlePrimalDynamicConflictOffsets[
						conflict + 1]; reverseIndex++)
			{
				if(workspace.particlePrimalDynamicConflictIndices[
					reverseIndex] == particleIndex)
				{
					reverseFound = true;
					break;
				}
			}
			if(!reverseFound)
				return;
		}
	}
	workspace.particlePrimalDynamicConflictValid = true;

	workspace.resize(workspace.particlePrimalColors, numParticles);
	// The ordered reference schedule orients every conflict by the legacy
	// particle order.  The relaxed fast schedule instead uses a standard
	// greedy coloring: it preserves the no-conflict invariant inside a color,
	// but deliberately permits a different nonlinear-GS trajectory across
	// colors.  It is therefore never selected by the ordered reference mode.
	const bool preserveLegacyCausalOrder =
		particlePrimalSchedule !=
			AvbdParticlePrimalSchedule::eRELAXED_COLOR;
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
		workspace.particlePrimalColors[particleIndex] = PX_MAX_U32;
	PxU32 colorCount = 0;
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
	{
		PxU32 color = 0;
		const PxU32 bodyIndex =
			workspace.particlePrimalBodyIndices[particleIndex];
		const AvbdSoftBodyCompiledData& compiled =
			softBodies[bodyIndex].compiled;
		const PxU32 localIndex = particleIndex - compiled.particleStart;
		auto hasEarlierNeighborWithColor = [&workspace, &compiled,
			particleIndex, localIndex](
			PxU32 candidateColor)
		{
			for(PxU32 conflictIndex =
				compiled.particlePrimalStructuralConflictOffsets[localIndex];
				conflictIndex <
				compiled.particlePrimalStructuralConflictOffsets[
					localIndex + 1]; conflictIndex++)
			{
				const PxU32 neighbor = compiled.particleStart +
					compiled.particlePrimalStructuralConflictIndices[
						conflictIndex];
				if(neighbor < particleIndex &&
					workspace.particlePrimalColors[neighbor] == candidateColor)
					return true;
			}
			for(PxU32 conflictIndex =
				workspace.particlePrimalDynamicConflictOffsets[particleIndex];
				conflictIndex <
				workspace.particlePrimalDynamicConflictOffsets[
					particleIndex + 1]; conflictIndex++)
			{
				const PxU32 neighbor =
					workspace.particlePrimalDynamicConflictIndices[
						conflictIndex];
				if(neighbor < particleIndex &&
					workspace.particlePrimalColors[neighbor] == candidateColor)
					return true;
			}
			return false;
		};
		if(preserveLegacyCausalOrder)
		{
			auto observeEarlierConflict = [&workspace, particleIndex, &color](
				PxU32 neighbor)
			{
				if(neighbor >= particleIndex)
					return true;
				const PxU32 neighborColor =
					workspace.particlePrimalColors[neighbor];
				if(neighborColor == PX_MAX_U32 || neighborColor >=
					particleIndex)
					return false;
				color = PxMax(color, neighborColor + 1);
				return true;
			};
			for(PxU32 conflictIndex =
				compiled.particlePrimalStructuralConflictOffsets[localIndex];
				conflictIndex <
					compiled.particlePrimalStructuralConflictOffsets[
						localIndex + 1]; conflictIndex++)
			{
				if(!observeEarlierConflict(compiled.particleStart +
					compiled.particlePrimalStructuralConflictIndices[conflictIndex]))
					return;
			}
			for(PxU32 conflictIndex =
				workspace.particlePrimalDynamicConflictOffsets[particleIndex];
				conflictIndex <
					workspace.particlePrimalDynamicConflictOffsets[
						particleIndex + 1]; conflictIndex++)
			{
				if(!observeEarlierConflict(
					workspace.particlePrimalDynamicConflictIndices[conflictIndex]))
					return;
			}
		}
		else
		{
			while(hasEarlierNeighborWithColor(color))
			{
				if(++color >= numParticles)
					return;
			}
		}
		if(color >= numParticles)
			return;
		workspace.particlePrimalColors[particleIndex] = color;
		colorCount = PxMax(colorCount, color + 1);
	}

	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
	{
		const PxU32 color = workspace.particlePrimalColors[particleIndex];
		const PxU32 bodyIndex =
			workspace.particlePrimalBodyIndices[particleIndex];
		const AvbdSoftBodyCompiledData& compiled =
			softBodies[bodyIndex].compiled;
		const PxU32 localIndex = particleIndex - compiled.particleStart;
		for(PxU32 conflictIndex =
			compiled.particlePrimalStructuralConflictOffsets[localIndex];
			conflictIndex <
				compiled.particlePrimalStructuralConflictOffsets[
					localIndex + 1]; conflictIndex++)
		{
			const PxU32 neighbor = compiled.particleStart +
				compiled.particlePrimalStructuralConflictIndices[conflictIndex];
			if(workspace.particlePrimalColors[neighbor] == color)
				return;
		}
		for(PxU32 conflictIndex =
			workspace.particlePrimalDynamicConflictOffsets[particleIndex];
			conflictIndex <
				workspace.particlePrimalDynamicConflictOffsets[
					particleIndex + 1]; conflictIndex++)
		{
			const PxU32 neighbor =
				workspace.particlePrimalDynamicConflictIndices[conflictIndex];
			if(workspace.particlePrimalColors[neighbor] == color)
				return;
		}
	}

	workspace.resize(workspace.particlePrimalColorCounts, colorCount);
	workspace.resize(workspace.particlePrimalColorOffsets, colorCount + 1);
	for(PxU32 color = 0; color < colorCount; color++)
		workspace.particlePrimalColorCounts[color] = 0;
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
		workspace.particlePrimalColorCounts[
			workspace.particlePrimalColors[particleIndex]]++;
	workspace.particlePrimalColorOffsets[0] = 0;
	for(PxU32 color = 0; color < colorCount; color++)
		workspace.particlePrimalColorOffsets[color + 1] =
			workspace.particlePrimalColorOffsets[color] +
			workspace.particlePrimalColorCounts[color];
	workspace.resize(workspace.particlePrimalColorParticles, numParticles);
	for(PxU32 color = 0; color < colorCount; color++)
		workspace.particlePrimalColorCounts[color] = 0;
	for(PxU32 particleIndex = 0; particleIndex < numParticles;
		particleIndex++)
	{
		const PxU32 color = workspace.particlePrimalColors[particleIndex];
		workspace.particlePrimalColorParticles[
			workspace.particlePrimalColorOffsets[color] +
			workspace.particlePrimalColorCounts[color]++] = particleIndex;
	}
	for(PxU32 color = 0; color < colorCount; color++)
	{
		const PxU32 begin = workspace.particlePrimalColorOffsets[color];
		const PxU32 end = workspace.particlePrimalColorOffsets[color + 1];
		for(PxU32 packedIndex = begin; packedIndex < end; packedIndex++)
		{
			if(workspace.particlePrimalColorParticles[packedIndex] >=
				numParticles ||
				workspace.particlePrimalColors[
					workspace.particlePrimalColorParticles[packedIndex]] != color ||
				(packedIndex > begin &&
					workspace.particlePrimalColorParticles[
						packedIndex - 1] >=
					workspace.particlePrimalColorParticles[packedIndex]))
				return;
		}
	}
	workspace.particlePrimalColorCount = colorCount;
	workspace.particlePrimalColorPlanValid = true;
}

// True-boundary collision vertices are normally expanded by Scene before they
// reach this component.  The eventual ground-patch experiment is meaningful
// only when one such vertex maps strictly inside one simulation tet.  Keep
// this qualification independent from solver ownership: it reads immutable
// contact IR and never changes a contact, particle, plan, or workspace.
PX_FORCE_INLINE bool avbdCollectGroundTetPatchFourSupport(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftBody& body, const AvbdSoftParticle* particles,
	PxU32 numParticles, PxU32 outSupport[4])
{
	if(!particles || geometry.queryPoint.count != 4 ||
		body.compiled.particleStart > numParticles ||
		body.compiled.particleCount >
			numParticles - body.compiled.particleStart)
		return false;

	PxReal weightSum = 0.0f;
	for(PxU32 supportIndex = 0; supportIndex < 4; ++supportIndex)
	{
		const PxU32 particleIndex =
			geometry.queryPoint.particleIndices[supportIndex];
		const PxReal weight = geometry.queryPoint.weights[supportIndex];
		if(particleIndex < body.compiled.particleStart ||
			particleIndex >= body.compiled.particleStart +
				body.compiled.particleCount ||
			!PxIsFinite(weight) || weight <= 1.0e-8f ||
			particles[particleIndex].invMass <= 0.0f ||
			!PxIsFinite(particles[particleIndex].invMass) ||
			!particles[particleIndex].position.isFinite())
			return false;
		for(PxU32 earlier = 0; earlier < supportIndex; ++earlier)
			if(outSupport[earlier] == particleIndex)
				return false;
		outSupport[supportIndex] = particleIndex;
		weightSum += weight;
	}
	return PxIsFinite(weightSum) && PxAbs(weightSum - 1.0f) <= 1.0e-3f;
}

PX_FORCE_INLINE bool avbdFindGroundTetPatchSingleTet(
	const AvbdSoftBody& body, const PxU32 support[4], PxU32& outTetIndex)
{
	outTetIndex = PX_MAX_U32;
	if(body.compiled.particleCount == 0)
		return false;

	const PxU32 firstLocalIndex =
		support[0] - body.compiled.particleStart;
	if(firstLocalIndex >= body.compiled.elementAdjacency.size())
		return false;
	const PxArray<AvbdParticleElementRef>& tetRefs =
		body.compiled.elementAdjacency[firstLocalIndex].tetRefs;
	for(PxU32 tetRefOffset = 0; tetRefOffset < tetRefs.size();
		++tetRefOffset)
	{
		const PxU32 tetIndex = tetRefs[tetRefOffset].index;
		if(tetIndex >= body.compiled.tetElements.size())
			continue;
		const AvbdTetElement& tet = body.compiled.tetElements[tetIndex];
		const PxU32 tetVertices[4] = {tet.p0, tet.p1, tet.p2, tet.p3};
		bool sameSet = true;
		for(PxU32 supportIndex = 0; supportIndex < 4 && sameSet;
			supportIndex++)
		{
			bool found = false;
			for(PxU32 vertexIndex = 0; vertexIndex < 4; ++vertexIndex)
				if(tetVertices[vertexIndex] == support[supportIndex])
				{
					found = true;
					break;
				}
			if(!found)
				sameSet = false;
		}
		if(sameSet)
		{
			outTetIndex = tetIndex;
			return true;
		}
	}
	return false;
}

PX_NOINLINE inline void avbdAccumulateGroundTetPatchProbe(
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdSoftParticle* particles, PxU32 numParticles,
	AvbdSoftBodyStepStats& stepStats)
{
	if(!softBodies || !contacts || !particles)
		return;
	for(PxU32 contactIndex = 0; contactIndex < numContacts;
		++contactIndex)
	{
		const AvbdSoftContact& contact = contacts[contactIndex];
		const AvbdSoftContactGeometry& geometry = contact.geometry;
		if(geometry.source.type != AvbdSoftContactSource::eGROUND ||
			!geometry.hasWorldStaticTarget() ||
			geometry.velocityOwner !=
				AvbdVelocityObjectiveOwner::PositionAL)
			continue;
		stepStats.groundTetPatchGroundPositionAlRows++;
		if(geometry.queryBodyIndex >= numSoftBodies ||
			geometry.queryPoint.count != 4)
			continue;
		const AvbdSoftBody& body = softBodies[geometry.queryBodyIndex];
		if(body.compiled.speculativeCCDEnabled)
			continue;
		PxU32 support[4];
		if(!avbdCollectGroundTetPatchFourSupport(
				geometry, body, particles, numParticles, support))
			continue;
		stepStats.groundTetPatchFourSupportRows++;
		PxU32 tetIndex = PX_MAX_U32;
		if(!avbdFindGroundTetPatchSingleTet(body, support, tetIndex))
			continue;
		stepStats.groundTetPatchSingleTetRows++;
		const PxVec3 surfacePoint =
			avbdGetSoftContactSurfacePoint(geometry, particles);
		if(!surfacePoint.isFinite())
			continue;
		const AvbdSoftContactRowForces rowForces =
			avbdEvaluateSoftContactRowForces(
				geometry, contact.state, particles, surfacePoint);
		if(PxIsFinite(rowForces.normal) && rowForces.normal < 0.0f)
			stepStats.groundTetPatchActiveRows++;
	}
}

// The velocity tangent path is intentionally narrower than the normal
// Position-AL path.  It is admitted only after Scene has expanded the
// collision proxy into the final simulation-particle support.  That keeps the
// impulse lever arm, mass response and ownership checks in the same domain as
// the particle solver.
PX_FORCE_INLINE bool avbdCanUseVelocityTangentOwner(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftParticle* particles, PxU32 numParticles)
{
	const bool supportedWorldStaticSource =
		geometry.source.type == AvbdSoftContactSource::eGROUND ||
		(geometry.source.type == AvbdSoftContactSource::eRIGID_SDF &&
		 geometry.hasRigidBoxSdf);
	const bool supportedWorldStatic = supportedWorldStaticSource &&
		geometry.hasWorldStaticTarget();
	const bool supportedSoftSoft =
		geometry.source.type == AvbdSoftContactSource::eSOFT_SURFACE &&
		geometry.targetKind ==
			AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE &&
		(geometry.hasWeightedTargetPoint() ||
		 geometry.hasDeformableSurfaceTarget());
	const bool supportedDynamicRigid =
		geometry.source.type == AvbdSoftContactSource::eRIGID_SDF &&
		geometry.hasRigidBodyTarget();
	if(!softBodies || !particles ||
		(!supportedWorldStatic && !supportedSoftSoft &&
		 !supportedDynamicRigid) ||
		geometry.velocityOwner !=
			AvbdVelocityObjectiveOwner::PositionAL ||
		!PxIsFinite(geometry.friction) || geometry.friction <= 0.0f)
		return false;

	const PxReal normalMagnitudeSq = geometry.normal.magnitudeSquared();
	const PxReal tangent1MagnitudeSq = geometry.tangent1.magnitudeSquared();
	const PxReal tangent2MagnitudeSq = geometry.tangent2.magnitudeSquared();
	if(!geometry.normal.isFinite() || !geometry.tangent1.isFinite() ||
		!geometry.tangent2.isFinite() || !PxIsFinite(normalMagnitudeSq) ||
		!PxIsFinite(tangent1MagnitudeSq) ||
		!PxIsFinite(tangent2MagnitudeSq) ||
		PxAbs(normalMagnitudeSq - 1.0f) > 1.0e-3f ||
		PxAbs(tangent1MagnitudeSq - 1.0f) > 1.0e-3f ||
		PxAbs(tangent2MagnitudeSq - 1.0f) > 1.0e-3f ||
		PxAbs(geometry.normal.dot(geometry.tangent1)) > 1.0e-3f ||
		PxAbs(geometry.normal.dot(geometry.tangent2)) > 1.0e-3f ||
		PxAbs(geometry.tangent1.dot(geometry.tangent2)) > 1.0e-3f)
		return false;

	PxU32 supportIndices[AVBD_CONTACT_MAX_PARTICLES];
	const PxU32 supportCount = avbdCollectSoftContactParticleIndices(
		geometry, supportIndices);
	if(supportCount == 0 || supportCount > AVBD_CONTACT_MAX_PARTICLES)
		return false;
	PxReal response = 0.0f;
	for(PxU32 supportIndex = 0; supportIndex < supportCount; ++supportIndex)
	{
		const PxU32 particleIndex = supportIndices[supportIndex];
		if(particleIndex >= numParticles)
			return false;
		const AvbdSoftBody* body = avbdFindSoftBodyForParticle(
			softBodies, numSoftBodies, particleIndex);
		if(!body || body->compiled.speculativeCCDEnabled ||
			!PxIsFinite(body->compiled.maxDepenetrationVelocity) ||
			body->compiled.maxDepenetrationVelocity < 1.0e20f)
			return false;
		const AvbdSoftParticle& particle = particles[particleIndex];
		const PxReal weight = avbdGetSoftContactParticleJacobianScale(
			geometry, particleIndex);
		if(!PxIsFinite(weight) || PxAbs(weight) <= 1.0e-8f ||
			particle.invMass < 0.0f || !PxIsFinite(particle.invMass) ||
			!particle.position.isFinite() || !particle.velocity.isFinite())
			return false;
		response += weight * weight * particle.invMass;
	}
	return PxIsFinite(response) && response > 1.0e-12f;
}

PX_FORCE_INLINE void avbdAssignVelocityTangentOwners(
	AvbdSoftContact* contacts, PxU32 numContacts,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftParticle* particles, PxU32 numParticles)
{
	if(!contacts || !particles ||
		!avbdUseVelocityTangentOwner())
		return;
	for(PxU32 contactIndex = 0; contactIndex < numContacts;
		++contactIndex)
	{
		AvbdSoftContact& contact = contacts[contactIndex];
		AvbdSoftContactGeometry& geometry = contact.geometry;
		const AvbdSoftContactTangentOwner previousOwner =
			geometry.tangentOwner;
		const bool useVelocityOwner =
			avbdCanUseVelocityTangentOwner(
				geometry, softBodies, numSoftBodies, particles,
				numParticles);
		geometry.tangentOwner = useVelocityOwner
			? AvbdSoftContactTangentOwner::eVELOCITY
			: AvbdSoftContactTangentOwner::ePOSITION_AL;
		// Any owner transition invalidates the old tangent state.  In
		// particular, a row that was velocity-owned in a prior epoch must not
		// later resurrect a stale Position-AL spring when its eligibility drops.
		if(useVelocityOwner || previousOwner != geometry.tangentOwner)
			avbdResetSoftContactTangentState(
				geometry, contact.state, particles);
	}
}

// Terminal tangent projection for rows whose Position-AL normal was retained
// but whose tangent was deliberately removed from the primal and dual paths.
// It uses the final simulation-particle support, applies no normal impulse,
// and does not write any AL state.  For one row the disk projection is
// non-energy-injecting in the particle inv-mass metric.
PX_NOINLINE inline void avbdProjectSoftContactVelocityTangents(
	AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftContact* contacts, PxU32 numContacts,
	PxReal dt, AvbdSoftBodyStepStats* stepStats = NULL)
{
	// The policy is process-static, so this is one cached branch per step; an
	// exact-zero diagnostic rollback keeps the legacy path free of this scan.
	if(!avbdUseVelocityTangentOwner() ||
		!particles || !contacts || !softBodies || dt <= 0.0f ||
		!PxIsFinite(dt))
		return;

	for(PxU32 contactIndex = 0; contactIndex < numContacts;
		++contactIndex)
	{
		const AvbdSoftContact& contact = contacts[contactIndex];
		const AvbdSoftContactGeometry& geometry = contact.geometry;
		const AvbdSoftContactAugmentedState& state = contact.state;
		if(geometry.tangentOwner !=
			AvbdSoftContactTangentOwner::eVELOCITY ||
			geometry.hasRigidBodyTarget() ||
			!avbdCanUseVelocityTangentOwner(
				geometry, softBodies, numSoftBodies, particles,
				numParticles) ||
			!PxIsFinite(state.alLambda) || state.alLambda >= 0.0f)
			continue;

		if(stepStats)
			stepStats->worldStaticVelocityTangentOwnerRows++;
		PxU32 supportIndices[AVBD_CONTACT_MAX_PARTICLES];
		const PxU32 supportCount = avbdCollectSoftContactParticleIndices(
			geometry, supportIndices);
		PxReal response = 0.0f;
		PxVec3 relativeVelocity(0.0f);
		bool valid = true;
		for(PxU32 supportIndex = 0; supportIndex < supportCount;
			++supportIndex)
		{
			const PxU32 particleIndex = supportIndices[supportIndex];
			const PxReal weight = avbdGetSoftContactParticleJacobianScale(
				geometry, particleIndex);
			const AvbdSoftParticle& particle = particles[particleIndex];
			if(!particle.velocity.isFinite())
			{
				valid = false;
				break;
			}
			response += weight * weight * particle.invMass;
			relativeVelocity += particle.velocity * weight;
		}
		if(!valid || !PxIsFinite(response) || response <= 1.0e-12f ||
			!relativeVelocity.isFinite())
			continue;

		const PxReal tangentVelocity0 =
			relativeVelocity.dot(geometry.tangent1);
		const PxReal tangentVelocity1 =
			relativeVelocity.dot(geometry.tangent2);
		if(!PxIsFinite(tangentVelocity0) || !PxIsFinite(tangentVelocity1))
			continue;
		PxReal tangentImpulse0 = -tangentVelocity0 / response;
		PxReal tangentImpulse1 = -tangentVelocity1 / response;
		const PxReal normalImpulseBudget =
			PxMax(-state.alLambda, 0.0f) * dt;
		const PxReal tangentImpulseLimit =
			geometry.friction * normalImpulseBudget;
		if(!PxIsFinite(normalImpulseBudget) ||
			!PxIsFinite(tangentImpulseLimit) || tangentImpulseLimit < 0.0f)
			continue;
		const PxReal tangentImpulseMagnitude = PxSqrt(
			tangentImpulse0 * tangentImpulse0 +
			tangentImpulse1 * tangentImpulse1);
		if(!PxIsFinite(tangentImpulseMagnitude))
			continue;
		if(tangentImpulseMagnitude > tangentImpulseLimit &&
			tangentImpulseMagnitude > 1.0e-12f)
		{
			const PxReal scale = tangentImpulseLimit /
				tangentImpulseMagnitude;
			tangentImpulse0 *= scale;
			tangentImpulse1 *= scale;
		}
		if(PxAbs(tangentImpulse0) <= 1.0e-12f &&
			PxAbs(tangentImpulse1) <= 1.0e-12f)
			continue;
		const PxVec3 tangentImpulse =
			geometry.tangent1 * tangentImpulse0 +
			geometry.tangent2 * tangentImpulse1;
		if(!tangentImpulse.isFinite())
			continue;

		PxVec3 updatedVelocities[AVBD_CONTACT_MAX_PARTICLES];
		for(PxU32 supportIndex = 0; supportIndex < supportCount;
			++supportIndex)
		{
			const PxU32 particleIndex = supportIndices[supportIndex];
			const PxReal weight = avbdGetSoftContactParticleJacobianScale(
				geometry, particleIndex);
			updatedVelocities[supportIndex] = particles[particleIndex].velocity +
				tangentImpulse * (particles[particleIndex].invMass * weight);
			// Match the finite-speed envelope used by
			// updateVelocityFromPosition().  This pass runs after that rebuild,
			// so it must fail closed rather than reintroduce an unbounded terminal
			// velocity through an otherwise finite multiplier.
			if(!updatedVelocities[supportIndex].isFinite() ||
				PxAbs(updatedVelocities[supportIndex].x) > 1.0e6f ||
				PxAbs(updatedVelocities[supportIndex].y) > 1.0e6f ||
				PxAbs(updatedVelocities[supportIndex].z) > 1.0e6f)
			{
				valid = false;
				break;
			}
		}
		if(!valid)
			continue;
		for(PxU32 supportIndex = 0; supportIndex < supportCount;
			++supportIndex)
		{
			const PxU32 particleIndex = supportIndices[supportIndex];
			particles[particleIndex].velocity = updatedVelocities[supportIndex];
		}
		if(stepStats)
			stepStats->worldStaticVelocityTangentAppliedRows++;
	}
}

// Re-query one world-static contact against its authoritative current shape.
// This is intentionally a discrete endpoint query: unlike the Position-AL
// normal constraint it does not include the OGC shell margin, and it never
// tests an old-to-new segment.  Box rows retain immutable shape metadata;
// planes and legacy static rows retain their world-space surface point.
PX_FORCE_INLINE bool avbdGetCurrentWorldStaticEndpointDcdGeometry(
	const AvbdSoftContactGeometry& geometry, const PxVec3& queryPoint,
	PxVec3& normal, PxReal& trueGap)
{
	if(!geometry.hasWorldStaticTarget() || !queryPoint.isFinite())
		return false;

	if(!geometry.hasRigidBoxSdf)
	{
		const PxReal normalLengthSq = geometry.normal.magnitudeSquared();
		if(!geometry.normal.isFinite() || !PxIsFinite(normalLengthSq) ||
			normalLengthSq <= 1.0e-12f || !geometry.surfacePoint.isFinite())
			return false;
		normal = geometry.normal * PxRecipSqrt(normalLengthSq);
		trueGap = (queryPoint - geometry.surfacePoint).dot(normal);
		return PxIsFinite(trueGap);
	}

	const PxVec3 halfExtent = geometry.rigidBoxHalfExtent;
	const PxTransform& shapeToWorld = geometry.rigidBoxPose;
	const PxReal rotationLengthSq =
		shapeToWorld.q.magnitudeSquared();
	if(!halfExtent.isFinite() || halfExtent.x <= 0.0f ||
		halfExtent.y <= 0.0f || halfExtent.z <= 0.0f ||
		!shapeToWorld.p.isFinite() || !shapeToWorld.q.isFinite() ||
		!PxIsFinite(rotationLengthSq) || rotationLengthSq <= 1.0e-12f)
		return false;

	const PxVec3 localPoint = shapeToWorld.transformInv(queryPoint);
	if(!localPoint.isFinite())
		return false;
	const PxVec3 q(
		PxAbs(localPoint.x) - halfExtent.x,
		PxAbs(localPoint.y) - halfExtent.y,
		PxAbs(localPoint.z) - halfExtent.z);
	const bool inside = q.x <= 0.0f && q.y <= 0.0f && q.z <= 0.0f;
	PxVec3 localNormal(0.0f);
	if(inside)
	{
		trueGap = PxMax(q.x, PxMax(q.y, q.z));
		if(q.x > q.y && q.x > q.z)
			localNormal = PxVec3(localPoint.x >= 0.0f ? 1.0f : -1.0f,
				0.0f, 0.0f);
		else if(q.y > q.z)
			localNormal = PxVec3(0.0f,
				localPoint.y >= 0.0f ? 1.0f : -1.0f, 0.0f);
		else
			localNormal = PxVec3(0.0f, 0.0f,
				localPoint.z >= 0.0f ? 1.0f : -1.0f);
	}
	else
	{
		const PxVec3 outside(
			PxMax(q.x, 0.0f), PxMax(q.y, 0.0f), PxMax(q.z, 0.0f));
		trueGap = outside.magnitude();
		if(!PxIsFinite(trueGap) || trueGap <= 1.0e-12f)
			return false;
		localNormal = PxVec3(
			(localPoint.x >= 0.0f ? 1.0f : -1.0f) * outside.x,
			(localPoint.y >= 0.0f ? 1.0f : -1.0f) * outside.y,
			(localPoint.z >= 0.0f ? 1.0f : -1.0f) * outside.z) /
			trueGap;
	}

	normal = shapeToWorld.q.rotate(localNormal);
	const PxReal normalLengthSq = normal.magnitudeSquared();
	if(!normal.isFinite() || !PxIsFinite(normalLengthSq) ||
		normalLengthSq <= 1.0e-12f || !PxIsFinite(trueGap))
		return false;
	normal *= PxRecipSqrt(normalLengthSq);
	return true;
}

PX_FORCE_INLINE const AvbdSoftBody*
avbdFindWorldStaticEndpointDcdSourceBody(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxU32 numParticles)
{
	if(!softBodies || numSoftBodies == 0)
		return NULL;
	const PxU32 representative = geometry.hasWeightedQueryPoint()
		? geometry.queryPoint.particleIndices[0]
		: geometry.hasBarycentricQueryPoint()
			? geometry.queryParticleIndices[0] : geometry.particleIdx;
	if(representative >= numParticles)
		return NULL;
	if(geometry.queryBodyIndex < numSoftBodies &&
		avbdSoftBodyContainsParticle(
			softBodies[geometry.queryBodyIndex], representative,
			numParticles))
		return &softBodies[geometry.queryBodyIndex];
	return avbdFindSoftBodyForParticle(
		softBodies, numSoftBodies, representative);
}

// Component fallback does not own a movable rigid endpoint, but it can still
// receive current-pose ground and world-static box contacts.  This must use
// the same *local* support recovery as the native mixed path.  Translating a
// complete soft body out of a static target makes a falling volume appear
// kinematic (and, after the velocity anchor is translated with it, removes
// the normal component of gravity).  Recover only the weighted query support
// instead, then let the ordinary material rows distribute that load through
// the tetrahedra.
PX_NOINLINE inline void avbdApplyWorldStaticComponentEndpointDcdRecovery(
	AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftContact* contacts, PxU32 numContacts,
	AvbdSoftBodyWorkspace& workspace, PxU32 sweeps = 4u)
{
	workspace.resize(workspace.worldStaticEndpointRecoveredBodies,
		numSoftBodies);
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
		workspace.worldStaticEndpointRecoveredBodies[bodyIndex] = 0u;
	if(!particles || !softBodies || !contacts || numParticles == 0 ||
		numSoftBodies == 0 || numContacts == 0 || sweeps == 0)
		return;

	for(PxU32 sweep = 0; sweep < sweeps; ++sweep)
	{
		bool appliedAny = false;
		for(PxU32 contactIndex = 0; contactIndex < numContacts;
			++contactIndex)
		{
			const AvbdSoftContactGeometry& geometry =
				contacts[contactIndex].geometry;
			if((geometry.source.type != AvbdSoftContactSource::eGROUND &&
				(geometry.source.type != AvbdSoftContactSource::eRIGID_SDF ||
					!geometry.hasRigidBoxSdf)) ||
				!geometry.hasWorldStaticTarget() ||
				geometry.velocityOwner !=
					AvbdVelocityObjectiveOwner::PositionAL ||
				!avbdHasSoftContactDynamicQuerySupport(
					geometry, particles, numParticles))
				continue;

			const AvbdSoftBody* sourceBody =
				avbdFindWorldStaticEndpointDcdSourceBody(
					geometry, softBodies, numSoftBodies, numParticles);
			if(!sourceBody || sourceBody->compiled.speculativeCCDEnabled ||
				!PxIsFinite(sourceBody->compiled.maxDepenetrationVelocity) ||
				sourceBody->compiled.maxDepenetrationVelocity < 1.0e20f)
				continue;
			const PxU32 bodyIndex = PxU32(sourceBody - softBodies);
			if(bodyIndex >= numSoftBodies)
				continue;

			const PxVec3 queryPoint =
				avbdGetSoftContactQueryPoint(geometry, particles);
			PxVec3 normal(0.0f);
			PxReal trueGap = 0.0f;
			if(!queryPoint.isFinite() ||
				!avbdGetCurrentWorldStaticEndpointDcdGeometry(
					geometry, queryPoint, normal, trueGap) ||
				!(trueGap < 0.0f) || !PxIsFinite(trueGap))
				continue;

			PxU32 supportIndices[AVBD_CONTACT_MAX_PARTICLES];
			const PxU32 supportCount = avbdCollectSoftContactParticleIndices(
				geometry, supportIndices);
			if(supportCount == 0 || supportCount > AVBD_CONTACT_MAX_PARTICLES)
				continue;

			PxReal response = 0.0f;
			PxReal weights[AVBD_CONTACT_MAX_PARTICLES];
			PxVec3 deltas[AVBD_CONTACT_MAX_PARTICLES];
			bool validSupport = true;
			for(PxU32 supportIndex = 0; supportIndex < supportCount;
				++supportIndex)
			{
				const PxU32 particleIndex = supportIndices[supportIndex];
				if(particleIndex >= numParticles ||
					!avbdSoftBodyContainsParticle(
						*sourceBody, particleIndex, numParticles))
				{
					validSupport = false;
					break;
				}
				const AvbdSoftParticle& particle = particles[particleIndex];
				const PxReal weight = avbdGetSoftContactParticleJacobianScale(
					geometry, particleIndex);
				if(!PxIsFinite(weight) || !PxIsFinite(particle.invMass) ||
					particle.invMass < 0.0f || !particle.position.isFinite() ||
					!particle.initialPosition.isFinite())
				{
					validSupport = false;
					break;
				}
				weights[supportIndex] = weight;
				response += particle.invMass * weight * weight;
			}
			if(!validSupport || !PxIsFinite(response) || response <= 1.0e-12f)
				continue;

			const PxReal lambda = -trueGap / response;
			if(!PxIsFinite(lambda) || lambda <= 0.0f)
				continue;
			for(PxU32 supportIndex = 0; supportIndex < supportCount;
				++supportIndex)
			{
				const AvbdSoftParticle& particle =
					particles[supportIndices[supportIndex]];
				deltas[supportIndex] = normal *
					(particle.invMass * weights[supportIndex] * lambda);
				if(!deltas[supportIndex].isFinite() ||
					!(particle.position + deltas[supportIndex]).isFinite() ||
					!(particle.initialPosition + deltas[supportIndex]).isFinite())
				{
					validSupport = false;
					break;
				}
			}
			if(!validSupport)
				continue;

			// The support vertices move as one mass-weighted contact block.  A
			// single alpha preserves that relation and lets the exact incident-tet
			// test reject a correction before it can turn a resting contact into
			// an inversion.
			PxReal commonAlpha = 1.0f;
			bool accepted = false;
			for(PxU32 attempt = 0; attempt < 8u && !accepted; ++attempt)
			{
				bool candidateValid = true;
				auto candidatePositionFor = [&supportIndices, &deltas,
					particles, supportCount, commonAlpha](PxU32 particleIndex)
					-> PxVec3
				{
					for(PxU32 i = 0; i < supportCount; ++i)
						if(supportIndices[i] == particleIndex)
							return particles[particleIndex].position +
								deltas[i] * commonAlpha;
					return particles[particleIndex].position;
				};
				bool hasSubthresholdTet = false;
				bool improvesSubthresholdTet = false;
				for(PxU32 supportIndex = 0;
					supportIndex < supportCount && candidateValid; ++supportIndex)
				{
					const PxU32 particleIndex = supportIndices[supportIndex];
					const PxU32 localIndex = particleIndex -
						sourceBody->compiled.particleStart;
					if(localIndex >= sourceBody->compiled.elementAdjacency.size())
					{
						candidateValid = false;
						break;
					}
					const AvbdParticleElementAdjacency& adjacency =
						sourceBody->compiled.elementAdjacency[localIndex];
					for(PxU32 refIndex = 0;
						refIndex < adjacency.tetRefs.size(); ++refIndex)
					{
						const AvbdParticleElementRef& ref =
							adjacency.tetRefs[refIndex];
						if(ref.index >= sourceBody->compiled.tetElements.size())
						{
							candidateValid = false;
							break;
						}
						const AvbdTetElement& tet =
							sourceBody->compiled.tetElements[ref.index];
						if(tet.p0 >= numParticles || tet.p1 >= numParticles ||
							tet.p2 >= numParticles || tet.p3 >= numParticles)
						{
							candidateValid = false;
							break;
						}
						const PxVec3 currentP0 = particles[tet.p0].position;
						const PxVec3 currentE1 = particles[tet.p1].position - currentP0;
						const PxVec3 currentE2 = particles[tet.p2].position - currentP0;
						const PxVec3 currentE3 = particles[tet.p3].position - currentP0;
						PxReal currentDeterminant;
						PxVec3 unusedGradient;
						avbdEvaluateTetDeterminantAndGradient(
							tet, 0u, currentE1, currentE2, currentE3,
							currentDeterminant, unusedGradient);
						const PxVec3 candidateP0 = candidatePositionFor(tet.p0);
						const PxVec3 candidateE1 =
							candidatePositionFor(tet.p1) - candidateP0;
						const PxVec3 candidateE2 =
							candidatePositionFor(tet.p2) - candidateP0;
						const PxVec3 candidateE3 =
							candidatePositionFor(tet.p3) - candidateP0;
						PxReal candidateDeterminant;
						avbdEvaluateTetDeterminantAndGradient(
							tet, 0u, candidateE1, candidateE2, candidateE3,
							candidateDeterminant, unusedGradient);
						if(!PxIsFinite(currentDeterminant) ||
							!PxIsFinite(candidateDeterminant))
						{
							candidateValid = false;
							break;
						}
						if(currentDeterminant >= 0.05f)
						{
							if(candidateDeterminant < 0.05f)
							{
								candidateValid = false;
								break;
							}
						}
						else
						{
							hasSubthresholdTet = true;
							if(candidateDeterminant + 1.0e-6f <
								currentDeterminant)
							{
								candidateValid = false;
								break;
							}
							if(candidateDeterminant >
								currentDeterminant + 1.0e-6f)
								improvesSubthresholdTet = true;
						}
					}
				}
				if(candidateValid && hasSubthresholdTet &&
					!improvesSubthresholdTet)
					candidateValid = false;

				PxVec3 candidateQuery = queryPoint;
				for(PxU32 supportIndex = 0;
					supportIndex < supportCount && candidateValid; ++supportIndex)
					candidateQuery += deltas[supportIndex] *
						(commonAlpha * weights[supportIndex]);
				PxVec3 candidateNormal(0.0f);
				PxReal candidateGap = 0.0f;
				if(!candidateQuery.isFinite() ||
					!avbdGetCurrentWorldStaticEndpointDcdGeometry(
						geometry, candidateQuery, candidateNormal, candidateGap) ||
					!PxIsFinite(candidateGap) ||
					candidateGap <= trueGap + 1.0e-6f)
					candidateValid = false;
				if(candidateValid)
					accepted = true;
				else
					commonAlpha *= 0.5f;
			}
			if(!accepted)
				continue;

			for(PxU32 supportIndex = 0; supportIndex < supportCount;
				++supportIndex)
			{
				const PxVec3 delta = deltas[supportIndex] * commonAlpha;
				if(delta.magnitudeSquared() <= 0.0f)
					continue;
				AvbdSoftParticle& particle =
					particles[supportIndices[supportIndex]];
				particle.position += delta;
				// Geometric recovery is not a rebound impulse.  Moving this
				// support's reconstruction anchor prevents a one-frame launch,
				// while untouched body particles keep their gravity motion.
				particle.initialPosition += delta;
			}
			workspace.worldStaticEndpointRecoveredBodies[bodyIndex] = 1u;
			appliedAny = true;
		}
		if(!appliedAny)
			break;
	}
}

// Position recovery moves the velocity anchor along with the particles, so it
// deliberately does not manufacture a separating bounce.  Remove only the
// remaining inward normal velocity at recovered current-pose rows after the
// ordinary position-to-velocity rebuild and component finalization.
PX_NOINLINE inline void avbdClampWorldStaticComponentEndpointDcdVelocities(
	AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftContact* contacts, PxU32 numContacts,
	const PxArray<PxU8>& recoveredBodies)
{
	if(!particles || !softBodies || !contacts ||
		recoveredBodies.size() != numSoftBodies)
		return;

	for(PxU32 contactIndex = 0; contactIndex < numContacts;
		++contactIndex)
	{
		const AvbdSoftContactGeometry& geometry = contacts[contactIndex].geometry;
		if((geometry.source.type != AvbdSoftContactSource::eGROUND &&
			(geometry.source.type != AvbdSoftContactSource::eRIGID_SDF ||
				!geometry.hasRigidBoxSdf)) ||
			!geometry.hasWorldStaticTarget() ||
			geometry.velocityOwner != AvbdVelocityObjectiveOwner::PositionAL ||
			!avbdHasSoftContactDynamicQuerySupport(
				geometry, particles, numParticles))
			continue;
		const AvbdSoftBody* sourceBody =
			avbdFindWorldStaticEndpointDcdSourceBody(
				geometry, softBodies, numSoftBodies, numParticles);
		if(!sourceBody || sourceBody->compiled.speculativeCCDEnabled)
			continue;
		const PxU32 bodyIndex = PxU32(sourceBody - softBodies);
		if(bodyIndex >= numSoftBodies || recoveredBodies[bodyIndex] == 0u)
			continue;

		const PxVec3 queryPoint =
			avbdGetSoftContactQueryPoint(geometry, particles);
		PxVec3 normal(0.0f);
		PxReal trueGap = 0.0f;
		if(!queryPoint.isFinite() ||
			!avbdGetCurrentWorldStaticEndpointDcdGeometry(
				geometry, queryPoint, normal, trueGap) ||
			!PxIsFinite(trueGap) || trueGap > 1.0e-3f)
			continue;

		PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
		const PxU32 supportCount = avbdCollectSoftContactParticleIndices(
			geometry, particleIndices);
		if(supportCount == 0 || supportCount > AVBD_CONTACT_MAX_PARTICLES)
			continue;
		PxReal response = 0.0f;
		PxVec3 queryVelocity(0.0f);
		bool valid = true;
		for(PxU32 supportIndex = 0; supportIndex < supportCount;
			++supportIndex)
		{
			const PxU32 particleIndex = particleIndices[supportIndex];
			if(particleIndex >= numParticles)
			{
				valid = false;
				break;
			}
			const PxReal weight = avbdGetSoftContactParticleJacobianScale(
				geometry, particleIndex);
			const AvbdSoftParticle& particle = particles[particleIndex];
			if(!PxIsFinite(weight) || !PxIsFinite(particle.invMass) ||
				!particle.velocity.isFinite())
			{
				valid = false;
				break;
			}
			response += particle.invMass * weight * weight;
			queryVelocity += particle.velocity * weight;
		}
		if(!valid || !PxIsFinite(response) || response <= 1.0e-12f ||
			!queryVelocity.isFinite())
			continue;
		const PxReal normalVelocity = queryVelocity.dot(normal);
		if(!PxIsFinite(normalVelocity) || normalVelocity >= -1.0e-6f)
			continue;
		const PxReal impulse = -normalVelocity / response;
		if(!PxIsFinite(impulse) || impulse <= 0.0f)
			continue;

		PxVec3 candidateVelocities[AVBD_CONTACT_MAX_PARTICLES];
		for(PxU32 supportIndex = 0; supportIndex < supportCount;
			++supportIndex)
		{
			const PxU32 particleIndex = particleIndices[supportIndex];
			const PxReal weight = avbdGetSoftContactParticleJacobianScale(
				geometry, particleIndex);
			const AvbdSoftParticle& particle = particles[particleIndex];
			candidateVelocities[supportIndex] = particle.velocity +
				normal * (particle.invMass * weight * impulse);
			if(!candidateVelocities[supportIndex].isFinite() ||
				PxAbs(candidateVelocities[supportIndex].x) > 1.0e6f ||
				PxAbs(candidateVelocities[supportIndex].y) > 1.0e6f ||
				PxAbs(candidateVelocities[supportIndex].z) > 1.0e6f)
			{
				valid = false;
				break;
			}
		}
		if(!valid)
			continue;
		for(PxU32 supportIndex = 0; supportIndex < supportCount;
			++supportIndex)
			particles[particleIndices[supportIndex]].velocity =
				candidateVelocities[supportIndex];
	}
}

// Rebuild the redetection-epoch particle incidence and, when explicitly
// requested, its P4 causal access plan.  This has no caller-stack capture, so
// the persistent component step state can invoke it only at its serial
// redetection barriers.  It must never be called by a particle range task.
inline void avbdBuildSoftParticleContactIndex(
	AvbdSoftBodyWorkspace& workspace,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdSoftContact* contacts, PxU32 numContacts,
	PxU32 numParticles, AvbdSoftBodyStepStats* stepStats,
	AvbdParticlePrimalSchedule particlePrimalSchedule =
		AvbdParticlePrimalSchedule::eSERIAL_LINEAR,
	bool validateP4AccessPlan = false,
	const AvbdSoftParticle* probeParticles = NULL)
{
	if(contacts && probeParticles)
		avbdAssignVelocityTangentOwners(
			contacts, numContacts, softBodies, numSoftBodies,
			probeParticles, numParticles);
	if(stepStats && probeParticles && avbdUseGroundTetPatchProbe())
		avbdAccumulateGroundTetPatchProbe(
			softBodies, numSoftBodies, contacts, numContacts,
			probeParticles, numParticles, *stepStats);
	workspace.resize(workspace.contactStarts, numParticles + 1);
	workspace.resize(workspace.contactCounts, numParticles);
	PxArray<AvbdSoftContactParticleRef>& contactIdxBuf =
		workspace.contactIndices;
	PxArray<PxU32>& contactStart = workspace.contactStarts;
	PxArray<PxU32>& contactCount = workspace.contactCounts;
	for(PxU32 particleIndex = 0;
		particleIndex < numParticles; particleIndex++)
		contactCount[particleIndex] = 0;
	for(PxU32 contactIndex = 0; contactIndex < numContacts; contactIndex++)
	{
		const AvbdSoftContactGeometry& geometry =
			contacts[contactIndex].geometry;
		PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
		const PxU32 particleIndexCount =
			avbdCollectSoftContactParticleIndices(
				geometry, particleIndices);
		for(PxU32 particleOffset = 0;
			particleOffset < particleIndexCount; particleOffset++)
		{
			const PxU32 particleIndex = particleIndices[particleOffset];
			if(particleIndex >= numParticles)
				continue;
			if(PxAbs(avbdGetSoftContactParticleJacobianScale(
				geometry, particleIndex)) > 1e-12f)
				contactCount[particleIndex]++;
		}
	}
	contactStart[0] = 0;
	for(PxU32 particleIndex = 0;
		particleIndex < numParticles; particleIndex++)
		contactStart[particleIndex + 1] =
			contactStart[particleIndex] + contactCount[particleIndex];
	workspace.resize(contactIdxBuf, contactStart[numParticles]);
	workspace.peakContactIncidenceCount = PxMax(
		workspace.peakContactIncidenceCount, contactIdxBuf.size());
	workspace.peakContactIncidenceCapacity = PxMax(
		workspace.peakContactIncidenceCapacity, contactIdxBuf.capacity());
	for(PxU32 particleIndex = 0;
		particleIndex < numParticles; particleIndex++)
		contactCount[particleIndex] = 0;
	for(PxU32 contactIndex = 0; contactIndex < numContacts; contactIndex++)
	{
		const AvbdSoftContactGeometry& geometry =
			contacts[contactIndex].geometry;
		PxU32 particleIndices[AVBD_CONTACT_MAX_PARTICLES];
		const PxU32 particleIndexCount =
			avbdCollectSoftContactParticleIndices(
				geometry, particleIndices);
		for(PxU32 particleOffset = 0;
			particleOffset < particleIndexCount; particleOffset++)
		{
			const PxU32 particleIndex = particleIndices[particleOffset];
			if(particleIndex >= numParticles)
				continue;
			const PxReal jacobianScale =
				avbdGetSoftContactParticleJacobianScale(
					geometry, particleIndex);
			if(PxAbs(jacobianScale) <= 1e-12f)
				continue;
			contactIdxBuf[
				contactStart[particleIndex] +
				contactCount[particleIndex]++] =
				AvbdSoftContactParticleRef(contactIndex, jacobianScale);
		}
	}
	const bool buildP4AccessPlan = validateP4AccessPlan ||
		avbdUsesColoredParticlePrimalSchedule(particlePrimalSchedule);
	if(buildP4AccessPlan)
	{
		avbdBuildParticlePrimalColorPlan(
			workspace, softBodies, numSoftBodies,
			contacts, numContacts, numParticles,
			particlePrimalSchedule);
		// The explicit validation mode turns a missing descriptor into a
		// Checked failure. The experimental colored schedule deliberately
		// does not: capacity or graph rejection must retain serial GS.
		if(validateP4AccessPlan)
		{
			PX_ASSERT(workspace.particlePrimalDynamicConflictValid);
			PX_ASSERT(workspace.particlePrimalColorPlanValid);
		}
	}
	if(stepStats && workspace.particlePrimalColorPlanValid)
	{
		stepStats->particlePrimalColorCount = PxMax(
			stepStats->particlePrimalColorCount,
			workspace.particlePrimalColorCount);
		stepStats->particlePrimalDynamicAccessGroupCount = PxMax(
			stepStats->particlePrimalDynamicAccessGroupCount,
			workspace.particlePrimalDynamicAccessGroups.size());
	}
}

// Rejected research prototype: the pre-sweep support takeover replaced whole
// vertex subproblems rather than the one contact row.  Keep it out of the
// build while the row-owned post-sweep formulation is developed separately.
#if 0
// Default-off, one-row PositionAL block experiment.  It intentionally takes
// over only the first stable eligible row in a sweep; that makes every
// endpoint write serial and avoids claiming that a contact-row coloring scheme
// already exists.  The owning caller skips the marked particles in the normal
// scalar traversal only after this function succeeds.
PX_NOINLINE inline bool avbdTryApplySoftSoftPositionRowBlock(
	const AvbdParticlePrimalSolveContext& solveContext,
	AvbdSoftBodyWorkspace& workspace,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftContact* contacts, PxU32 numContacts,
	PxU32 numParticles,
	AvbdParticlePrimalRangeObservation& observation)
{
	if(!solveContext.particles || !solveContext.selfCollisionSafetyBounds ||
		!softBodies || !contacts || numParticles == 0)
		return false;
	workspace.resize(workspace.softSoftPositionRowBlockOwnedParticles,
		numParticles);
	std::memset(workspace.softSoftPositionRowBlockOwnedParticles.begin(),
		0, sizeof(PxU8) * numParticles);

	for(PxU32 contactIndex = 0; contactIndex < numContacts; ++contactIndex)
	{
		const AvbdSoftContact& contact = contacts[contactIndex];
		const AvbdSoftContactGeometry& geometry = contact.geometry;
		const AvbdSoftContactAugmentedState& state = contact.state;
		if(geometry.source.type != AvbdSoftContactSource::eSOFT_SURFACE ||
			geometry.targetKind !=
				AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE ||
			geometry.velocityOwner !=
				AvbdVelocityObjectiveOwner::PositionAL)
			continue;

		const PxReal normalMagnitudeSq = geometry.normal.magnitudeSquared();
		const PxReal tangent1MagnitudeSq = geometry.tangent1.magnitudeSquared();
		const PxReal tangent2MagnitudeSq = geometry.tangent2.magnitudeSquared();
		if(!geometry.normal.isFinite() || !geometry.tangent1.isFinite() ||
			!geometry.tangent2.isFinite() || !PxIsFinite(normalMagnitudeSq) ||
			!PxIsFinite(tangent1MagnitudeSq) ||
			!PxIsFinite(tangent2MagnitudeSq) ||
			PxAbs(normalMagnitudeSq - 1.0f) > 1.0e-3f ||
			PxAbs(tangent1MagnitudeSq - 1.0f) > 1.0e-3f ||
			PxAbs(tangent2MagnitudeSq - 1.0f) > 1.0e-3f ||
			PxAbs(geometry.normal.dot(geometry.tangent1)) > 1.0e-3f ||
			PxAbs(geometry.normal.dot(geometry.tangent2)) > 1.0e-3f ||
			PxAbs(geometry.tangent1.dot(geometry.tangent2)) > 1.0e-3f ||
			!PxIsFinite(geometry.friction) || geometry.friction <= 0.0f ||
			!PxIsFinite(state.k) || state.k <= 0.0f ||
			!PxIsFinite(state.penTangent[0]) ||
			!PxIsFinite(state.penTangent[1]) ||
			state.penTangent[0] < 0.0f || state.penTangent[1] < 0.0f)
			continue;

		PxU32 queryEndpoint[AVBD_CONTACT_POINT_MAX_SUPPORT];
		PxU32 targetEndpoint[AVBD_CONTACT_POINT_MAX_SUPPORT];
		PxU32 queryEndpointCount = 0;
		PxU32 targetEndpointCount = 0;
		if(!avbdCollectSoftContactEndpointIndices(
				geometry, true, queryEndpoint, queryEndpointCount) ||
			!avbdCollectSoftContactEndpointIndices(
				geometry, false, targetEndpoint, targetEndpointCount) ||
			queryEndpointCount == 0 || targetEndpointCount == 0)
			continue;
		const AvbdSoftBody* queryBody =
			geometry.queryBodyIndex < numSoftBodies
				? &softBodies[geometry.queryBodyIndex]
				: avbdFindSoftBodyForParticle(
					softBodies, numSoftBodies, queryEndpoint[0]);
		const AvbdSoftBody* targetBody =
			geometry.source.targetBodyIndex < numSoftBodies
				? &softBodies[geometry.source.targetBodyIndex]
				: avbdFindSoftBodyForParticle(
					softBodies, numSoftBodies, targetEndpoint[0]);
		if(!queryBody || !targetBody || queryBody == targetBody ||
			queryBody->compiled.speculativeCCDEnabled ||
			targetBody->compiled.speculativeCCDEnabled)
			continue;
		bool endpointsValid = true;
		for(PxU32 i = 0; i < queryEndpointCount; ++i)
			if(!avbdSoftBodyContainsParticle(
					*queryBody, queryEndpoint[i], numParticles))
			{
				endpointsValid = false;
				break;
			}
		for(PxU32 i = 0; i < targetEndpointCount; ++i)
			if(!avbdSoftBodyContainsParticle(
					*targetBody, targetEndpoint[i], numParticles))
			{
				endpointsValid = false;
				break;
			}
		if(!endpointsValid)
			continue;
		const PxU32 queryEnd = queryBody->compiled.particleStart +
			queryBody->compiled.particleCount;
		const PxU32 targetEnd = targetBody->compiled.particleStart +
			targetBody->compiled.particleCount;
		if(queryBody->compiled.particleStart < targetEnd &&
			targetBody->compiled.particleStart < queryEnd)
			continue;

		PxU32 supportIndices[AVBD_CONTACT_MAX_PARTICLES];
		const PxU32 supportCount = avbdCollectSoftContactParticleIndices(
			geometry, supportIndices);
		if(supportCount < 2 || supportCount > AVBD_CONTACT_MAX_PARTICLES)
			continue;
		// The first diagnostic must not silently turn every other row touching
		// an endpoint into a stale Jacobi contribution.  It only takes over a
		// contact whose dynamic support has no other indexed contact row in
		// this epoch.  That keeps the scalar fallback mathematically local and
		// makes a successful A/B attributable to this one PositionAL row.
		if(!solveContext.contactStarts || !solveContext.contactIndices)
			continue;
		bool isolatedDynamicSupport = true;
		for(PxU32 supportIndex = 0; supportIndex < supportCount &&
			isolatedDynamicSupport; ++supportIndex)
		{
			const PxU32 particleIndex = supportIndices[supportIndex];
			if(particleIndex >= numParticles)
			{
				isolatedDynamicSupport = false;
				break;
			}
			for(PxU32 refIndex = solveContext.contactStarts[particleIndex];
				refIndex < solveContext.contactStarts[particleIndex + 1];
				++refIndex)
			{
				const AvbdSoftContactParticleRef& ref =
					solveContext.contactIndices[refIndex];
				if(ref.contactIndex != contactIndex &&
					PxAbs(ref.jacobianScale) > 1.0e-12f)
				{
					isolatedDynamicSupport = false;
					break;
				}
			}
		}
		if(!isolatedDynamicSupport)
			continue;

		PxMat33 inverseBaseHessian[AVBD_CONTACT_MAX_PARTICLES];
		PxMat33 responseJacobian[AVBD_CONTACT_MAX_PARTICLES];
		PxVec3 baseDisplacement[AVBD_CONTACT_MAX_PARTICLES];
		PxVec3 displacement[AVBD_CONTACT_MAX_PARTICLES];
		const AvbdSoftBody* supportBodies[AVBD_CONTACT_MAX_PARTICLES];
		PxMat33 response(PxZero);
		PxVec3 baseRowDisplacement(0.0f);
		bool valid = true;
		for(PxU32 supportIndex = 0; supportIndex < supportCount;
			++supportIndex)
		{
			const PxU32 particleIndex = supportIndices[supportIndex];
			if(particleIndex >= numParticles)
			{
				valid = false;
				break;
			}
			const PxReal signedScale =
				avbdGetSoftContactParticleJacobianScale(
					geometry, particleIndex);
			const AvbdSoftBody* body =
				avbdSoftBodyContainsParticle(*queryBody, particleIndex,
					numParticles) ? queryBody :
				(avbdSoftBodyContainsParticle(*targetBody, particleIndex,
					numParticles) ? targetBody : NULL);
			AvbdSoftParticle& particle = solveContext.particles[particleIndex];
			if(!body || !PxIsFinite(signedScale) ||
				PxAbs(signedScale) <= 1.0e-12f ||
				particle.invMass <= 0.0f || !PxIsFinite(particle.invMass) ||
				!PxIsFinite(particle.mass) || particle.mass <= 0.0f ||
				!particle.position.isFinite() ||
				!particle.predictedPosition.isFinite() ||
				!particle.outerPosition.isFinite())
			{
				valid = false;
				break;
			}
			const PxU32 localParticleIndex = particleIndex -
				body->compiled.particleStart;
			PxVec3 baseForce;
			PxMat33 baseHessian;
			if(!avbdAssembleParticlePrimalLocalSystem(
					solveContext, *body,
					localParticleIndex,
					contactIndex, baseForce, baseHessian) ||
				!avbdInvertPositiveDefiniteSymmetric33(
					baseHessian, inverseBaseHessian[supportIndex]))
			{
				valid = false;
				break;
			}
			const PxMat33 jacobian(
				geometry.normal * signedScale,
				geometry.tangent1 * signedScale,
				geometry.tangent2 * signedScale);
			responseJacobian[supportIndex] =
				inverseBaseHessian[supportIndex] * jacobian;
			baseDisplacement[supportIndex] =
				inverseBaseHessian[supportIndex] * baseForce;
			if(!responseJacobian[supportIndex].column0.isFinite() ||
				!responseJacobian[supportIndex].column1.isFinite() ||
				!responseJacobian[supportIndex].column2.isFinite() ||
				!baseDisplacement[supportIndex].isFinite())
			{
				valid = false;
				break;
			}
			response += jacobian.getTranspose() *
				responseJacobian[supportIndex];
			baseRowDisplacement += jacobian.getTranspose() *
				baseDisplacement[supportIndex];
			supportBodies[supportIndex] = body;
		}
		if(!valid || !response.column0.isFinite() ||
			!response.column1.isFinite() || !response.column2.isFinite() ||
			!baseRowDisplacement.isFinite())
			continue;

		const PxVec3 surfacePoint = avbdGetSoftContactSurfacePoint(
			geometry, solveContext.particles);
		if(!surfacePoint.isFinite())
			continue;
		const AvbdSoftContactRowForces rowForces =
			avbdEvaluateSoftContactRowForces(
				geometry, state, solveContext.particles, surfacePoint);
		// This is a rotation-quality experiment, not an alternative way to
		// advance an inactive proximity row.  A negative normal row force is
		// the same active unilateral condition consumed by the scalar contact
		// evaluator, and makes the selected row observable in the A/B.
		if(!PxIsFinite(rowForces.normal) || rowForces.normal >= 0.0f)
			continue;
		const PxReal tangentK0 = geometry.friction > 0.0f &&
			rowForces.normal < 0.0f && !rowForces.tangentClamped
				? state.penTangent[0] : 0.0f;
		const PxReal tangentK1 = geometry.friction > 0.0f &&
			rowForces.normal < 0.0f && !rowForces.tangentClamped
				? state.penTangent[1] : 0.0f;
		const PxMat33 penalty = PxMat33::createDiagonal(
			PxVec3(state.k, tangentK0, tangentK1));
		const PxVec3 rowForce(
			rowForces.normal, rowForces.tangent[0], rowForces.tangent[1]);
		const PxMat33 system = PxMat33(PxIdentity) + penalty * response;
		PxVec3 rowSolve;
		if(!rowForce.isFinite() ||
			!avbdSolveGeneral33Checked(
				system, rowForce + penalty * baseRowDisplacement, rowSolve))
			continue;

		for(PxU32 supportIndex = 0; supportIndex < supportCount;
			++supportIndex)
		{
			displacement[supportIndex] = baseDisplacement[supportIndex] -
				responseJacobian[supportIndex] * rowSolve;
			const PxReal displacementSq =
				displacement[supportIndex].magnitudeSquared();
			if(!displacement[supportIndex].isFinite() ||
				!PxIsFinite(displacementSq) || displacementSq > 1.0f)
			{
				valid = false;
				break;
			}
		}
		if(!valid)
			continue;

		// The support moves form one coupled block.  A per-vertex limiter would
		// destroy its Schur relation, so use one shared feasibility fraction for
		// both OGC and positive-J.  This is the multi-vertex analogue of the
		// scalar limiter: every accepted endpoint preserves the same row update.
		PxReal commonAlpha = 1.0f;
		bool acceptedCandidate = false;
		for(PxU32 attempt = 0; attempt < 8 && !acceptedCandidate;
			++attempt)
		{
			bool candidateValid = true;
			for(PxU32 supportIndex = 0; supportIndex < supportCount;
				++supportIndex)
			{
				const PxU32 particleIndex = supportIndices[supportIndex];
				const AvbdSoftParticle& particle =
					solveContext.particles[particleIndex];
				const PxReal safetyBound =
					solveContext.selfCollisionSafetyBounds[particleIndex];
				const PxVec3 candidatePosition = particle.position +
					displacement[supportIndex] * commonAlpha;
				if(!candidatePosition.isFinite() || !PxIsFinite(safetyBound) ||
					(safetyBound < 1.0e20f &&
						(candidatePosition - particle.outerPosition).magnitude() >
							safetyBound + 1.0e-6f))
				{
					candidateValid = false;
					break;
				}
			}
			if(!candidateValid)
			{
				commonAlpha *= 0.5f;
				continue;
			}

			auto candidatePositionFor = [&supportIndices, &displacement,
				&solveContext, supportCount, commonAlpha](
					PxU32 particleIndex) -> PxVec3
			{
				for(PxU32 i = 0; i < supportCount; ++i)
					if(supportIndices[i] == particleIndex)
						return solveContext.particles[particleIndex].position +
							displacement[i] * commonAlpha;
				return solveContext.particles[particleIndex].position;
			};
			for(PxU32 supportIndex = 0; supportIndex < supportCount &&
				candidateValid; ++supportIndex)
			{
				const AvbdSoftBody& body = *supportBodies[supportIndex];
				const PxU32 particleIndex = supportIndices[supportIndex];
				const PxU32 localIndex = particleIndex -
					body.compiled.particleStart;
				const AvbdParticleElementAdjacency& adjacency =
					body.compiled.elementAdjacency[localIndex];
				for(PxU32 refIndex = 0; refIndex < adjacency.tetRefs.size();
					++refIndex)
				{
					const AvbdTetElement& tet = body.compiled.tetElements[
						adjacency.tetRefs[refIndex].index];
					const PxVec3 p0 = candidatePositionFor(tet.p0);
					const PxVec3 e1 = candidatePositionFor(tet.p1) - p0;
					const PxVec3 e2 = candidatePositionFor(tet.p2) - p0;
					const PxVec3 e3 = candidatePositionFor(tet.p3) - p0;
					PxReal determinant;
					PxVec3 unusedGradient;
					avbdEvaluateTetDeterminantAndGradient(
						tet, 0, e1, e2, e3, determinant, unusedGradient);
					if(!PxIsFinite(determinant) || determinant < 0.05f)
					{
						candidateValid = false;
						break;
					}
				}
			}
			if(candidateValid)
				acceptedCandidate = true;
			else
				commonAlpha *= 0.5f;
		}
		if(!acceptedCandidate)
			continue;

		for(PxU32 supportIndex = 0; supportIndex < supportCount;
			++supportIndex)
		{
			const PxU32 particleIndex = supportIndices[supportIndex];
			const PxVec3 appliedDisplacement =
				displacement[supportIndex] * commonAlpha;
			solveContext.particles[particleIndex].position += appliedDisplacement;
			workspace.softSoftPositionRowBlockOwnedParticles[particleIndex] = 1;
			observation.sweepObservation.observe(
				displacement[supportIndex], commonAlpha < 1.0f,
				AvbdSoftTetDisplacementLimitResult(
					appliedDisplacement, commonAlpha,
					commonAlpha < 1.0f
						? AvbdSoftTetDisplacementLimitReason::ePOSITIVE_J_LIMITED
						: AvbdSoftTetDisplacementLimitReason::eNONE));
		}
		return true;
	}
	return false;
}

// Keep the default particle solve unchanged.  Only an admitted row-block
// sweep calls this cold serial traversal, which skips the support particles
// already solved as one coupled local system.
PX_NOINLINE inline void avbdSolveParticlePrimalExcludingOwnedParticles(
	const AvbdParticlePrimalSolveContext& solveContext,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const PxU8* ownedParticles,
	AvbdParticlePrimalRangeObservation& observation)
{
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		for(PxU32 localIndex = 0; localIndex < body.compiled.particleCount;
			++localIndex)
		{
			const PxU32 particleIndex = body.compiled.particleStart + localIndex;
			if(ownedParticles[particleIndex])
				continue;
			if(solveContext.canUseCorotationalTetPackets(body, localIndex))
				solveContext.solveWithCorotationalTetPackets(
					body, localIndex, observation);
			else
				solveContext.solve(body, localIndex, observation);
		}
	}
}
#endif

// P3 stage-boundary contract.  ePREPARE performs every operation that must
// precede prediction (including initial-contact state setup); eRESUME consumes
// particles whose predictedPosition/elasticK have already been updated and
// immediately performs the existing predicted-position OGC redetection.
// Split execution requires caller-owned persistent workspace and one stable
// AvbdSoftBodyStepStats instance across both calls.
enum class AvbdSoftBodyStepExecutionMode : PxU8
{
	eFULL,
	ePREPARE,
	eRESUME
};

inline void avbdPredictSoftBodyParticles(
	AvbdSoftParticle* particles, PxU32 numParticles,
	PxReal dt, const PxVec3& gravity, bool useAdaptiveInitialGuess)
{
	if(!avbdUseSoftElasticProximal())
	{
		if(useAdaptiveInitialGuess)
		{
			for(PxU32 i = 0; i < numParticles; i++)
			{
				particles[i].computePredictionWithAdaptiveInitialGuess(
					dt, gravity);
				// A disabled proximal must not retain warm-start state from a
				// preceding legacy run or timestep.
				particles[i].elasticK = 0.0f;
			}
		}
		else
		{
			for(PxU32 i = 0; i < numParticles; i++)
			{
				particles[i].computePrediction(dt, gravity);
				// A disabled proximal must not retain warm-start state from a
				// preceding legacy run or timestep.
				particles[i].elasticK = 0.0f;
			}
		}
		return;
	}
	if(useAdaptiveInitialGuess)
	{
		for(PxU32 i = 0; i < numParticles; i++)
		{
			particles[i].computePredictionWithAdaptiveInitialGuess(dt, gravity);
			// Reset elastic proximal weight for new timestep
			// (warmstart: retain a fraction from prior timestep for stability)
			particles[i].elasticK = particles[i].elasticK * 0.5f;
		}
	}
	else
	{
		for(PxU32 i = 0; i < numParticles; i++)
		{
			particles[i].computePrediction(dt, gravity);
			// Reset elastic proximal weight for new timestep
			// (warmstart: retain a fraction from prior timestep for stability)
			particles[i].elasticK = particles[i].elasticK * 0.5f;
		}
	}
}

// P4.5.2c uses this reference-only switch while the resumable lifecycle state
// is compared against the retained direct call-stack authority.  It does not
// create a worker or alter the production serial-linear default.
PX_FORCE_INLINE bool avbdUsePersistentStepStateSerial()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P4_STEP_STATE_SERIAL");
	return enabled;
}

// P4.5 task-fanin validation is deliberately opt-in.  It is consumed only
// by the Scene-owned continuation after prediction; the low-level state never
// creates a task or waits for one.  The production default remains the legacy
// serial-linear Gauss-Seidel path.
PX_FORCE_INLINE bool avbdUseCausalLayerTaskFanIn()
{
	static const bool enabled = []()
	{
		const char* value =
			std::getenv("PHYSX_AVBD_P4_CAUSAL_LAYER_TASK_FANIN");
		return value && value[0] == '1' && value[1] == '\0';
	}();
	return enabled;
}

// P6 coarse primal policy. Unlike P4's particle-color fan-in, this route
// publishes only complete, mutually independent soft bodies. Scene performs
// the collision/objective ownership proof before enabling the state, and the
// state additionally requires an empty contact epoch before every sweep. The
// rollback is sampled once and never reaches a particle loop.
PX_FORCE_INLINE bool avbdDisableIndependentBodySweepTaskFanIn()
{
	static const bool disabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P6_DISABLE_INDEPENDENT_BODY_SWEEP_TASK_FANIN");
	return disabled;
}

// Test-only eligibility override. It never changes the default P2/P3
// threshold, and is useful solely to exercise the P4.5 continuation on the
// small canonical-contact fixtures that intentionally fall below it.
PX_FORCE_INLINE bool avbdForceCausalLayerTaskFanIn()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P4_FORCE_CAUSAL_LAYER_TASK_FANIN");
	return enabled;
}

// This separate test-only switch forces only the P2/P3 Scene continuation for
// a small fixture.  Unlike avbdForceCausalLayerTaskFanIn(), it publishes no
// causal-layer children; its purpose is to retain the same Scene-owned serial
// oracle when comparing that continuation with a forced task fan-in route.
PX_FORCE_INLINE bool avbdForceCausalLayerTaskGraphReference()
{
	static const bool enabled = []()
	{
		const char* value = std::getenv(
			"PHYSX_AVBD_P4_FORCE_CAUSAL_LAYER_TASKGRAPH_REFERENCE");
		return value && value[0] == '1' && value[1] == '\0';
	}();
	return enabled;
}

// P4.5.3b validation control.  P4.5.3a intentionally submits one whole
// published layer; splitting it into several dispatcher children remains
// opt-in until the N-worker canonical-contact and determinism matrix accepts
// it.  This is independent from the task-fanin switch so the one-range
// reference continues to be directly selectable.
PX_FORCE_INLINE bool avbdUseCausalLayerTaskPartition()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P4_CAUSAL_LAYER_TASK_PARTITION");
	return enabled;
}

// Small self/soft-contact fixtures often have causal layers with only two
// non-conflicting particles. Keep those on the one-range reference by default,
// but permit an explicit test-only override to exercise their multi-child
// parent fan-in. This must never become a production scheduling heuristic.
PX_FORCE_INLINE bool avbdForceCausalLayerTaskPartition()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P4_FORCE_CAUSAL_LAYER_TASK_PARTITION");
	return enabled;
}

// P5.1 is a validation-only ownership seam.  It makes the Scene continuation
// execute redetection and hand the rebuilt contact array back to the low-level
// state; it creates no collision worker task and leaves the default callback
// route untouched.
PX_FORCE_INLINE bool avbdUseSceneRedetectionBridge()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_SCENE_REDETECTION_BRIDGE");
	return enabled;
}

// P5.3b explicitly gates the first candidate task transaction. It is useful
// only together with the Scene redetection bridge, which owns the canonical
// stream and the state-transfer fan-in.
PX_FORCE_INLINE bool avbdUseWorldPlaneContactTaskFanIn()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_WORLD_PLANE_TASK_FANIN");
	return enabled;
}

PX_FORCE_INLINE bool avbdForceWorldPlaneContactTaskFanIn()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_FORCE_WORLD_PLANE_TASK_FANIN");
	return enabled;
}

// P5.4b uses the same parent-owned redetection transaction for the discrete
// static-box SDF leaf.  The swept and OGC feature passes intentionally remain
// parent-serial after the stable range merge; AVX/task work must not reorder
// the legacy box contact stream.
PX_FORCE_INLINE bool avbdUseRigidBoxSdfContactTaskFanIn()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_RIGID_BOX_SDF_TASK_FANIN");
	return enabled;
}

PX_FORCE_INLINE bool avbdForceRigidBoxSdfContactTaskFanIn()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_FORCE_RIGID_BOX_SDF_TASK_FANIN");
	return enabled;
}

// P5.5b has an independently gated sphere transaction.  AVBD must not infer
// that a box opt-in authorizes a sphere route, even though both retain their
// swept and OGC feature suffixes in the parent.
PX_FORCE_INLINE bool avbdUseRigidSphereSdfContactTaskFanIn()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_RIGID_SPHERE_SDF_TASK_FANIN");
	return enabled;
}

PX_FORCE_INLINE bool avbdForceRigidSphereSdfContactTaskFanIn()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_FORCE_RIGID_SPHERE_SDF_TASK_FANIN");
	return enabled;
}

// P5.6b keeps static capsule discrete SDF independently opt-in.  Capsule and
// sphere transactions share only a mutually exclusive Scene continuation slot;
// their task pools, outputs, telemetry, and geometry suffixes stay separate.
PX_FORCE_INLINE bool avbdUseRigidCapsuleSdfContactTaskFanIn()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_RIGID_CAPSULE_SDF_TASK_FANIN");
	return enabled;
}

PX_FORCE_INLINE bool avbdForceRigidCapsuleSdfContactTaskFanIn()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_FORCE_RIGID_CAPSULE_SDF_TASK_FANIN");
	return enabled;
}

// P5.7b gives static convex SDF its own opt-in transaction. It can reuse the
// smooth-rigid continuation readiness slot only because strict eligibility
// makes sphere/capsule/convex routes mutually exclusive.
PX_FORCE_INLINE bool avbdUseRigidConvexSdfContactTaskFanIn()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_RIGID_CONVEX_SDF_TASK_FANIN");
	return enabled;
}

PX_FORCE_INLINE bool avbdForceRigidConvexSdfContactTaskFanIn()
{
	static const bool enabled = avbdReadProcessExactOneFlag(
		"PHYSX_AVBD_P5_FORCE_RIGID_CONVEX_SDF_TASK_FANIN");
	return enabled;
}

enum class AvbdSoftBodyStepAdvanceResult : PxU8
{
	eREDETECTION_READY,
	eCAUSAL_LAYER_READY,
	eCOMPLETE,
	eINVALID
};

// Persistent, parent-owned continuation for the component step after
// prediction.  It owns every mutable loop/control value that must survive a
// future causal-layer fan-in; particle children receive only the frozen solve
// context and one published packed range through accessors below.
struct AvbdSoftBodyStepState
{
	AvbdSoftBodyStepState();

	bool beginAfterPrediction(
		AvbdSoftParticle* inputParticles, PxU32 inputNumParticles,
		AvbdSoftBody* inputSoftBodies, PxU32 inputNumSoftBodies,
		AvbdSoftContact* inputContacts, PxU32 inputNumContacts,
		PxReal inputDt, PxU32 inputOuterIterations,
		PxU32 inputInnerIterations, PxU32 inputRequestedInnerBudget,
		PxReal inputAvbdBeta, AvbdContactRedetectFn inputRedetectFn,
		PxArray<AvbdSoftContact>* inputContactsArray,
		void* inputRedetectUserData, PxReal inputChebyshevRho,
		AvbdSoftBodyStepStats* inputStepStats,
		AvbdSoftBodyWorkspace& inputWorkspace,
		const AvbdSelfCollisionAdjacency* inputSelfCollisionAdjacencies,
		PxU32 inputNumSelfCollisionAdjacencies,
		const PxU8* inputSelfCollisionEnabled,
		const AvbdOGCParams* inputOgcParams,
		AvbdParticlePrimalSchedule inputParticlePrimalSchedule,
		bool inputDeferRedetectionToParent = false,
		bool inputPublishIndependentBodySweeps = false);

	AvbdSoftBodyStepAdvanceResult advance();

	// P5 ownership seam: a Scene parent runs the pending redetection, then
	// publishes its rebuilt contact array back to this resumable state. This
	// method deliberately does not invoke the callback or access Scene state.
	bool completePendingRedetection();

	bool getPublishedCausalLayer(
		PxU32& layerIndex, PxU32& packedBegin, PxU32& packedEnd,
		const AvbdParticlePrimalSolveContext*& solveContext,
		const AvbdSoftBody*& bodies, PxU32& bodyCount,
		const PxU32*& particleBodyIndices,
		const PxU32*& packedParticleIndices) const;

	bool completePublishedCausalLayer(
		const AvbdParticlePrimalRangeObservation* observations,
		PxU32 observationCount);

	bool getPublishedIndependentBodySweep(
		const AvbdParticlePrimalSolveContext*& solveContext,
		const AvbdSoftBody*& bodies, PxU32& bodyCount) const;

	bool completePublishedIndependentBodySweep(
		const AvbdParticlePrimalRangeObservation* observations,
		PxU32 observationCount);

	void runToCompletionSerial();

	PX_FORCE_INLINE bool isComplete() const
	{
		return phase == Phase::eCOMPLETE;
	}

private:
	enum class Phase : PxU8
	{
		eIDLE,
		eOUTER_PREPARE,
		eINNER_BEGIN,
		eCAUSAL_LAYER,
		eDUAL,
		eREDETECTION,
		eCOMPLETE,
		eINVALID
	};

	void prepareOuterIteration();
	bool beginInnerSweep();
	void finishParticlePrimalSweep();
	void updateDualAndRedetect();
	void finishInitialRedetection();
	void finalizeStep();

	AvbdSoftParticle* particles;
	PxU32 numParticles;
	AvbdSoftBody* softBodies;
	PxU32 numSoftBodies;
	AvbdSoftContact* contacts;
	PxU32 numContacts;
	PxReal dt;
	PxReal invDt;
	PxReal invDtSq;
	PxU32 outerIterations;
	PxU32 requestedInnerIterationBudget;
	PxU32 remainingInnerIterationBudget;
	PxU32 outerIt;
	PxU32 currentInnerIterations;
	PxU32 innerIt;
	PxReal avbdBeta;
	AvbdContactRedetectFn redetectFn;
	PxArray<AvbdSoftContact>* contactsArray;
	void* redetectUserData;
	PxReal chebyshevRho;
	bool useChebyshev;
	PxReal chebyOmega;
	PxReal adaptiveRho;
	PxReal prevMaxDxSq;
	PxU32 shadowResidual1e5ConsecutiveSweeps;
	PxU32 shadowResidual1e4ConsecutiveSweeps;
	bool shadowResidual1e5Recorded;
	bool shadowResidual1e4Recorded;
	bool legacyAppliedConvergenceRecorded;
	PxU32 residualConsecutiveSweeps;
	AvbdSoftBodyStepStats* stepStats;
	AvbdSoftBodyWorkspace* workspace;
	const AvbdSelfCollisionAdjacency* selfCollisionAdjacencies;
	PxU32 numSelfCollisionAdjacencies;
	const PxU8* selfCollisionEnabled;
	const AvbdOGCParams* ogcParams;
	bool deferRedetectionToParent;
	bool pendingInitialRedetection;
	bool reuseComponentOgcSafetyEpoch;
	bool componentOgcSafetyEpochActive;
	bool componentOgcSafetyEpochLimited;
	bool publishIndependentBodySweeps;
	bool independentBodySweepPublished;
	AvbdParticlePrimalSchedule particlePrimalSchedule;
	bool validateParticlePrimalAccessPlan;
	AvbdParticlePrimalSolveContext particlePrimalSolveContext;
	AvbdParticlePrimalRangeObservation particlePrimalObservation;
	AvbdParticlePrimalCausalLayerState causalLayerState;
	PxTime stageTimer;
	Phase phase;
};

#if defined(DY_AVBD_SOFT_BODY_STEP_STATE_IMPLEMENTATION)

AvbdSoftBodyStepState::AvbdSoftBodyStepState()
	: particles(NULL), numParticles(0), softBodies(NULL), numSoftBodies(0),
	  contacts(NULL), numContacts(0), dt(0.0f), invDt(0.0f),
	  invDtSq(0.0f), outerIterations(0),
	  requestedInnerIterationBudget(0), remainingInnerIterationBudget(0),
	  outerIt(0), currentInnerIterations(0), innerIt(0), avbdBeta(0.0f),
	  redetectFn(NULL), contactsArray(NULL), redetectUserData(NULL),
	  chebyshevRho(0.0f), useChebyshev(false), chebyOmega(1.0f),
	  adaptiveRho(0.0f), prevMaxDxSq(0.0f),
	  shadowResidual1e5ConsecutiveSweeps(0),
	  shadowResidual1e4ConsecutiveSweeps(0),
	  shadowResidual1e5Recorded(false), shadowResidual1e4Recorded(false),
	  legacyAppliedConvergenceRecorded(false), residualConsecutiveSweeps(0),
	  stepStats(NULL), workspace(NULL), selfCollisionAdjacencies(NULL),
	  numSelfCollisionAdjacencies(0), selfCollisionEnabled(NULL),
	  ogcParams(NULL), deferRedetectionToParent(false),
	  pendingInitialRedetection(false),
	  reuseComponentOgcSafetyEpoch(false),
	  componentOgcSafetyEpochActive(false),
	  componentOgcSafetyEpochLimited(false),
	  publishIndependentBodySweeps(false),
	  independentBodySweepPublished(false),
	  particlePrimalSchedule(AvbdParticlePrimalSchedule::eSERIAL_LINEAR),
	  validateParticlePrimalAccessPlan(false),
	  phase(Phase::eIDLE)
{
}

bool AvbdSoftBodyStepState::beginAfterPrediction(
	AvbdSoftParticle* inputParticles, PxU32 inputNumParticles,
	AvbdSoftBody* inputSoftBodies, PxU32 inputNumSoftBodies,
	AvbdSoftContact* inputContacts, PxU32 inputNumContacts,
	PxReal inputDt, PxU32 inputOuterIterations,
	PxU32 inputInnerIterations, PxU32 inputRequestedInnerBudget,
	PxReal inputAvbdBeta, AvbdContactRedetectFn inputRedetectFn,
	PxArray<AvbdSoftContact>* inputContactsArray,
	void* inputRedetectUserData, PxReal inputChebyshevRho,
	AvbdSoftBodyStepStats* inputStepStats,
	AvbdSoftBodyWorkspace& inputWorkspace,
	const AvbdSelfCollisionAdjacency* inputSelfCollisionAdjacencies,
	PxU32 inputNumSelfCollisionAdjacencies,
	const PxU8* inputSelfCollisionEnabled,
	const AvbdOGCParams* inputOgcParams,
	AvbdParticlePrimalSchedule inputParticlePrimalSchedule,
	bool inputDeferRedetectionToParent,
	bool inputPublishIndependentBodySweeps)
{
	PX_UNUSED(inputInnerIterations);
	if(!inputParticles || inputNumParticles == 0 ||
		!inputSoftBodies || inputNumSoftBodies == 0 ||
		(phase != Phase::eIDLE && phase != Phase::eCOMPLETE))
	{
		phase = Phase::eINVALID;
		return false;
	}
	particles = inputParticles;
	numParticles = inputNumParticles;
	softBodies = inputSoftBodies;
	numSoftBodies = inputNumSoftBodies;
	contacts = inputContacts;
	numContacts = inputNumContacts;
	dt = inputDt;
	invDt = dt > 0.0f ? 1.0f / dt : 0.0f;
	invDtSq = invDt * invDt;
	outerIterations = inputOuterIterations;
	requestedInnerIterationBudget = inputRequestedInnerBudget;
	remainingInnerIterationBudget = requestedInnerIterationBudget;
	avbdBeta = inputAvbdBeta;
	redetectFn = inputRedetectFn;
	contactsArray = inputContactsArray;
	redetectUserData = inputRedetectUserData;
	chebyshevRho = inputChebyshevRho;
	useChebyshev = chebyshevRho > 0.0f && chebyshevRho < 1.0f;
	chebyOmega = 1.0f;
	adaptiveRho = chebyshevRho;
	stepStats = inputStepStats;
	avbdPublishCorotationalTetPacketIrStats(
		softBodies, numSoftBodies, stepStats);
	workspace = &inputWorkspace;
	selfCollisionAdjacencies = inputSelfCollisionAdjacencies;
	numSelfCollisionAdjacencies = inputNumSelfCollisionAdjacencies;
	selfCollisionEnabled = inputSelfCollisionEnabled;
	ogcParams = inputOgcParams;
	deferRedetectionToParent = inputDeferRedetectionToParent;
	pendingInitialRedetection = false;
	reuseComponentOgcSafetyEpoch = false;
	componentOgcSafetyEpochActive = false;
	componentOgcSafetyEpochLimited = false;
	publishIndependentBodySweeps = inputPublishIndependentBodySweeps;
	independentBodySweepPublished = false;
	particlePrimalSchedule = inputParticlePrimalSchedule ==
		AvbdParticlePrimalSchedule::eDEFAULT
		? avbdGetParticlePrimalSchedule() : inputParticlePrimalSchedule;
	validateParticlePrimalAccessPlan =
		avbdValidateParticlePrimalAccessPlan();
	stageTimer = PxTime();

	// This is the original predicted-position contact epoch barrier.  It is
	// parent-owned and completes before the first outer/layer publication.
	if(redetectFn && contactsArray)
	{
		if(deferRedetectionToParent)
		{
			pendingInitialRedetection = true;
			phase = Phase::eREDETECTION;
			return true;
		}
		redetectFn(particles, numParticles, softBodies, numSoftBodies,
			*contactsArray, redetectUserData);
		contacts = contactsArray->begin();
		numContacts = contactsArray->size();
		avbdCompileSoftVelocityObjectives(
			workspace->compiledVelocityObjectives,
			workspace->componentFinalizeModes,
			softBodies, numSoftBodies, contacts, numContacts);
	}
	finishInitialRedetection();
	return true;
}

void AvbdSoftBodyStepState::finishInitialRedetection()
{
	avbdInitializeSoftContactDepenetrationLimits(
		contacts, numContacts, particles,
		softBodies, numSoftBodies, dt);
	PxArray<AvbdSoftComponentMomentumTarget>& momentumTargets =
		workspace->componentMomentumTargets;
	workspace->resize(momentumTargets, numSoftBodies);
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		AvbdSoftComponentMomentumTarget& target = momentumTargets[bodyIndex];
		target = AvbdSoftComponentMomentumTarget();
		if(workspace->componentFinalizeModes[bodyIndex] ==
				AvbdSoftComponentFinalizeMode::eUNSUPPORTED)
			continue;
		PxVec3 centroid(0.0f);
		PxMat33 inertia(PxZero);
		target.valid = avbdComputeSoftComponentMomentum(
			particles, numParticles, softBodies[bodyIndex], true, invDt,
			centroid, target.linearMomentum, target.angularMomentum,
			inertia, target.mass);
		target.centroid = centroid;
		avbdApplySoftComponentDampingToMomentumTarget(
			target, softBodies[bodyIndex], dt);
		PX_UNUSED(inertia);
	}
	if(stepStats)
		stepStats->predictionMs += stageTimer.getElapsedSeconds() * 1000.0;
	avbdBuildSoftParticleContactIndex(
		*workspace, softBodies, numSoftBodies,
		contacts, numContacts, numParticles, stepStats,
		particlePrimalSchedule, validateParticlePrimalAccessPlan, particles);
	reuseComponentOgcSafetyEpoch = false;
	componentOgcSafetyEpochActive = false;
	componentOgcSafetyEpochLimited = false;
	if(stepStats)
		stepStats->contactIndexMs += stageTimer.getElapsedSeconds() * 1000.0;
	workspace->resize(workspace->selfCollisionSafetyBounds, numParticles);
	if(useChebyshev)
	{
		workspace->resize(workspace->chebyPrevPos, numParticles);
		workspace->resize(workspace->chebyPrevPrevPos, numParticles);
		for(PxU32 particleIndex = 0;
			particleIndex < numParticles; particleIndex++)
		{
			workspace->chebyPrevPos[particleIndex] =
				particles[particleIndex].position;
			workspace->chebyPrevPrevPos[particleIndex] =
				particles[particleIndex].position;
		}
	}
	outerIt = 0;
	phase = Phase::eOUTER_PREPARE;
}

bool AvbdSoftBodyStepState::completePendingRedetection()
{
	if(phase != Phase::eREDETECTION || !contactsArray)
		return false;
	contacts = contactsArray->begin();
	numContacts = contactsArray->size();
	avbdCompileSoftVelocityObjectives(
		workspace->compiledVelocityObjectives,
		workspace->componentFinalizeModes,
		softBodies, numSoftBodies, contacts, numContacts);
	if(pendingInitialRedetection)
	{
		pendingInitialRedetection = false;
		finishInitialRedetection();
		return true;
	}
	avbdInitializeSoftContactDepenetrationLimits(
		contacts, numContacts, particles,
		softBodies, numSoftBodies, dt);
	avbdBuildSoftParticleContactIndex(
		*workspace, softBodies, numSoftBodies,
		contacts, numContacts, numParticles, stepStats,
		particlePrimalSchedule, validateParticlePrimalAccessPlan, particles);
	reuseComponentOgcSafetyEpoch = false;
	componentOgcSafetyEpochActive = false;
	componentOgcSafetyEpochLimited = false;
	if(stepStats)
		stepStats->redetectMs += stageTimer.getElapsedSeconds() * 1000.0;
	outerIt++;
	phase = Phase::eOUTER_PREPARE;
	return true;
}

void AvbdSoftBodyStepState::prepareOuterIteration()
{
	PX_ASSERT(outerIt < outerIterations);
	if(stepStats)
		stepStats->executedOuterIterations++;
	const PxU32 remainingOuterIterations = outerIterations - outerIt;
	currentInnerIterations =
		(remainingInnerIterationBudget + remainingOuterIterations - 1) /
		remainingOuterIterations;
	remainingInnerIterationBudget -= currentInnerIterations;
	if(!reuseComponentOgcSafetyEpoch)
	{
		avbdSnapshotOuterPositionsScalar(
			particles, numParticles,
			workspace->selfCollisionSafetyBounds.begin());
		const AvbdOGCParams defaultOgcParams;
		const AvbdOGCParams& activeOgcParams =
			ogcParams ? *ogcParams : defaultOgcParams;
		if(selfCollisionAdjacencies)
		{
			for(PxU32 bodyIndex = 0;
				bodyIndex < numSoftBodies &&
				bodyIndex < numSelfCollisionAdjacencies; bodyIndex++)
			{
				if(selfCollisionEnabled && !selfCollisionEnabled[bodyIndex])
					continue;
				const AvbdSoftBody& body = softBodies[bodyIndex];
				avbdComputeSafetyBounds(
					body, particles, selfCollisionAdjacencies[bodyIndex],
					activeOgcParams.contactRadius, activeOgcParams.safetyRelax,
					workspace->bodySelfCollisionSafetyBounds,
					workspace->contact);
				for(PxU32 localIndex = 0;
					localIndex < body.compiled.particleCount; localIndex++)
				{
					const PxU32 particleIndex =
						body.compiled.particleStart + localIndex;
					if(particleIndex < numParticles)
						workspace->selfCollisionSafetyBounds[particleIndex] =
							workspace->bodySelfCollisionSafetyBounds[localIndex];
				}
			}
		}
		componentOgcSafetyEpochActive =
			avbdApplyComponentOgcEpochSafetyBounds(
				contacts, numContacts, softBodies, numSoftBodies, particles,
				activeOgcParams.contactRadius, activeOgcParams.safetyRelax,
				workspace->selfCollisionSafetyBounds.begin(), numParticles,
				*workspace);
	}
	// A reused epoch intentionally retains its original proximal anchor and
	// bound.  The next outer boundary will either retain it again or publish a
	// fresh DCD epoch after the limiter reports that it was spent.
	reuseComponentOgcSafetyEpoch = false;
	componentOgcSafetyEpochLimited = false;
	if(useChebyshev)
	{
		chebyOmega = 1.0f;
		for(PxU32 particleIndex = 0;
			particleIndex < numParticles; particleIndex++)
		{
			workspace->chebyPrevPos[particleIndex] =
				particles[particleIndex].position;
			workspace->chebyPrevPrevPos[particleIndex] =
				particles[particleIndex].position;
		}
	}
	innerIt = 0;
	prevMaxDxSq = 0.0f;
	shadowResidual1e5ConsecutiveSweeps = 0;
	shadowResidual1e4ConsecutiveSweeps = 0;
	shadowResidual1e5Recorded = false;
	shadowResidual1e4Recorded = false;
	legacyAppliedConvergenceRecorded = false;
	residualConsecutiveSweeps = 0;
	phase = Phase::eINNER_BEGIN;
}

bool AvbdSoftBodyStepState::beginInnerSweep()
{
	if(innerIt >= currentInnerIterations)
	{
		phase = Phase::eDUAL;
		return false;
	}
	if(stepStats)
	{
		stepStats->executedInnerIterations++;
		stepStats->particleSweeps++;
	}
	particlePrimalObservation = AvbdParticlePrimalRangeObservation();
	particlePrimalSolveContext =
	{
		particles,
		contacts,
		workspace->contactStarts.begin(),
		workspace->contactIndices.begin(),
		workspace->selfCollisionSafetyBounds.begin(),
		invDt,
		invDtSq,
		avbdSelectCorotationalTetPacketKernel(
			softBodies, numSoftBodies)
	};
	// A complete-body layer has no Gauss-Seidel dependency between children.
	// Scene freezes the structural eligibility for the whole step; the contact
	// epoch is checked here because redetection can introduce a soft pair at an
	// outer boundary.  Such an epoch immediately returns to scalar authority.
	if(publishIndependentBodySweeps && numSoftBodies > 1 && numContacts == 0)
	{
		independentBodySweepPublished = true;
		phase = Phase::eCAUSAL_LAYER;
		return true;
	}
	const bool useColoredPrimal =
		avbdUsesColoredParticlePrimalSchedule(
			particlePrimalSchedule) &&
		workspace->particlePrimalColorPlanValid;
	if(stepStats && avbdUsesColoredParticlePrimalSchedule(
		particlePrimalSchedule))
	{
		if(useColoredPrimal)
			stepStats->particlePrimalColoredSerialSweeps++;
		else
			stepStats->particlePrimalColoredSerialFallbackSweeps++;
	}
	if(useColoredPrimal)
	{
		const bool published = causalLayerState.begin(
			particlePrimalSolveContext, softBodies, numSoftBodies,
			workspace->particlePrimalBodyIndices.begin(), numParticles,
			workspace->particlePrimalColorParticles.begin(),
			workspace->particlePrimalColorOffsets.begin(),
			workspace->particlePrimalColorCount);
		if(!published)
		{
			phase = Phase::eINVALID;
			return false;
		}
		phase = Phase::eCAUSAL_LAYER;
		return true;
	}
	if(particlePrimalSolveContext.corotationalTetPacketKernel)
		avbdSolveParticlePrimalCorotationalTetPacketBodyRange(
			particlePrimalSolveContext, softBodies, numSoftBodies,
			particlePrimalObservation);
	else
	{
		for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
		{
			const AvbdSoftBody& body = softBodies[bodyIndex];
			for(PxU32 localIndex = 0;
				localIndex < body.compiled.particleCount; localIndex++)
				particlePrimalSolveContext.solve(
					body, localIndex, particlePrimalObservation);
		}
	}
	finishParticlePrimalSweep();
	return false;
}

void AvbdSoftBodyStepState::finishParticlePrimalSweep()
{
	const AvbdSoftSweepConvergenceObservation& sweepObservation =
		particlePrimalObservation.sweepObservation;
	const PxReal maxDxSq = sweepObservation.maxAppliedDisplacementSq;
	if(stepStats)
	{
		if(avbdUseParticlePrimalWorkCensus())
		{
			AvbdParticlePrimalWorkCensus workCensus;
			avbdRecordParticlePrimalWorkCensusForSweep(
				particles, softBodies, numSoftBodies,
				particlePrimalSolveContext.contactStarts, workCensus);
			avbdAccumulateParticlePrimalWorkCensus(*stepStats, workCensus);
		}
		stepStats->tetLinearizationCacheFallbackParticleSteps +=
			particlePrimalObservation.
				tetLinearizationCacheFallbackParticleSteps;
		stepStats->trustRegionLimitedParticleSteps +=
			sweepObservation.trustRegionLimitedSteps;
		stepStats->positiveJLimitedParticleSteps +=
			sweepObservation.positiveJLimitedSteps;
		stepStats->positiveJRejectedParticleSteps +=
			sweepObservation.positiveJRejectedSteps;
		stepStats->nonFiniteRejectedParticleSteps +=
			sweepObservation.nonFiniteRejectedSteps;
		stepStats->finalMaxLocalSolveDisplacement = PxSqrt(
			sweepObservation.maxLocalSolveDisplacementSq);
		stepStats->finalMaxAppliedDisplacement = PxSqrt(maxDxSq);
		stepStats->finalMaxDisplacement =
			stepStats->finalMaxAppliedDisplacement;
	}
	if(sweepObservation.trustRegionLimitedSteps > 0)
		componentOgcSafetyEpochLimited = true;
	// The OGC trust region is an epoch boundary, not a signal to spend the
	// remaining material sweeps at a clamped pose.  End this same-time epoch
	// immediately so updateDualAndRedetect() can publish a fresh manifold.
	if(componentOgcSafetyEpochActive && componentOgcSafetyEpochLimited)
	{
		innerIt = currentInnerIterations;
		phase = Phase::eDUAL;
		return;
	}

	const bool appliedDisplacementConverged =
		sweepObservation.isAppliedDisplacementConverged(1e-12f);
	const bool strictResidualCandidateConverged =
		sweepObservation.isResidualConverged(1e-12f);
	const bool residualPolicyConverged =
		sweepObservation.isResidualConverged(1e-8f)
			? ++residualConsecutiveSweeps >= 2
			: (residualConsecutiveSweeps = 0, false);
	const bool shadowResidual1e5Converged =
		sweepObservation.isResidualConverged(1e-10f);
	const bool shadowResidual1e4Converged =
		sweepObservation.isResidualConverged(1e-8f);
	shadowResidual1e5ConsecutiveSweeps = shadowResidual1e5Converged
		? shadowResidual1e5ConsecutiveSweeps + 1 : 0;
	shadowResidual1e4ConsecutiveSweeps = shadowResidual1e4Converged
		? shadowResidual1e4ConsecutiveSweeps + 1 : 0;
	if(!shadowResidual1e5Recorded &&
		shadowResidual1e5ConsecutiveSweeps >= 2)
	{
		shadowResidual1e5Recorded = true;
		if(stepStats)
		{
			stepStats->shadowResidual1e5ConvergedOuterIterations++;
			stepStats->shadowResidual1e5SavedInnerIterations +=
				currentInnerIterations - (innerIt + 1);
		}
	}
	if(!shadowResidual1e4Recorded &&
		shadowResidual1e4ConsecutiveSweeps >= 2)
	{
		shadowResidual1e4Recorded = true;
		if(stepStats)
		{
			stepStats->shadowResidual1e4ConvergedOuterIterations++;
			stepStats->shadowResidual1e4SavedInnerIterations +=
				currentInnerIterations - (innerIt + 1);
		}
	}
	if(appliedDisplacementConverged && !legacyAppliedConvergenceRecorded)
	{
		legacyAppliedConvergenceRecorded = true;
		if(stepStats)
		{
			stepStats->legacyAppliedConvergedOuterIterations++;
			if(!strictResidualCandidateConverged)
				stepStats->unsafeAppliedConvergenceCandidates++;
		}
	}
	if(residualPolicyConverged)
	{
		if(stepStats)
			stepStats->residualConvergedOuterIterations++;
		innerIt = currentInnerIterations;
		phase = Phase::eDUAL;
		return;
	}
	if(stepStats && innerIt + 1 == currentInnerIterations)
		stepStats->budgetExhaustedOuterIterations++;
	if(innerIt == 0)
		prevMaxDxSq = maxDxSq;
	else if(innerIt == 1 && useChebyshev)
	{
		if(prevMaxDxSq > 1e-20f)
		{
			PxReal measuredRho = PxSqrt(maxDxSq / prevMaxDxSq);
			adaptiveRho = PxMin(measuredRho, chebyshevRho);
			adaptiveRho = PxMin(adaptiveRho, 0.95f);
		}
		prevMaxDxSq = maxDxSq;
	}
	if(useChebyshev && innerIt >= 2)
	{
		const PxReal rhoSq = adaptiveRho * adaptiveRho;
		if(innerIt == 2)
			chebyOmega = 2.0f / (2.0f - rhoSq);
		else
			chebyOmega = 1.0f /
				(1.0f - rhoSq * chebyOmega * 0.25f);
		chebyOmega = PxMax(1.0f, PxMin(chebyOmega, 2.0f));
		if(prevMaxDxSq > 1e-20f && maxDxSq > prevMaxDxSq * 1.1f)
		{
			chebyOmega = 1.0f;
			adaptiveRho = 0.0f;
		}
		if(chebyOmega > 1.0f)
		{
			for(PxU32 particleIndex = 0;
				particleIndex < numParticles; particleIndex++)
			{
				if(particles[particleIndex].isStatic() ||
					workspace->contactStarts[particleIndex + 1] >
						workspace->contactStarts[particleIndex])
					continue;
				particles[particleIndex].position =
					workspace->chebyPrevPrevPos[particleIndex] +
					(particles[particleIndex].position -
						workspace->chebyPrevPrevPos[particleIndex]) *
						chebyOmega;
				avbdTruncateDisplacement(
					particles[particleIndex],
					particles[particleIndex].outerPosition,
					workspace->selfCollisionSafetyBounds[particleIndex]);
			}
		}
		prevMaxDxSq = maxDxSq;
	}
	if(useChebyshev)
	{
		for(PxU32 particleIndex = 0;
			particleIndex < numParticles; particleIndex++)
		{
			workspace->chebyPrevPrevPos[particleIndex] =
				workspace->chebyPrevPos[particleIndex];
			workspace->chebyPrevPos[particleIndex] =
				particles[particleIndex].position;
		}
	}
	innerIt++;
	phase = innerIt < currentInnerIterations
		? Phase::eINNER_BEGIN : Phase::eDUAL;
}

void AvbdSoftBodyStepState::updateDualAndRedetect()
{
	if(stepStats)
		stepStats->particleSolveMs +=
			stageTimer.getElapsedSeconds() * 1000.0;
	for(PxU32 contactIndex = 0; contactIndex < numContacts; contactIndex++)
	{
		AvbdSoftContact& contact = contacts[contactIndex];
		avbdUpdateSoftContactDual(
			contact.geometry, contact.state, particles, avbdBeta);
	}
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		AvbdSoftBody& body = softBodies[bodyIndex];
		for(PxU32 objectiveIndex = 0;
			objectiveIndex < body.runtime.compiledObjectives.size();
			objectiveIndex++)
		{
			const AvbdCompiledSoftObjective& objective =
				body.runtime.compiledObjectives[objectiveIndex];
			PX_ASSERT(avbdIsPinPositionOwner(objective.owner));
			if(avbdIsPinPositionOwner(objective.owner))
				avbdUpdatePinDual(
					body.runtime.pins[objective.runtimeStateIndex],
					objective.point, particles, avbdBeta);
		}
	}
	if(avbdUseSoftElasticProximal())
	{
		for(PxU32 particleIndex = 0;
			particleIndex < numParticles; particleIndex++)
		{
			AvbdSoftParticle& particle = particles[particleIndex];
			if(particle.isStatic())
				continue;
			const PxReal displacement =
				(particle.position - particle.outerPosition).magnitude();
			particle.elasticK = PxMin(
				particle.elasticK + avbdBeta * displacement,
				particle.elasticKMax);
		}
	}
	if(stepStats)
		stepStats->dualMs += stageTimer.getElapsedSeconds() * 1000.0;
	const bool mayReusePureSoftPairEpoch =
		redetectFn && contactsArray && outerIt + 1 < outerIterations &&
		componentOgcSafetyEpochActive &&
		!componentOgcSafetyEpochLimited;
	if(mayReusePureSoftPairEpoch)
	{
		// The complete DCD manifold remains valid inside the conservative
		// inter-body envelope installed in prepareOuterIteration().  Reuse the
		// contact index and AL state exactly as Jolt-style XPBD iterations reuse
		// a substep manifold; the moment a particle reaches that envelope the
		// regular redetection route below resumes.
		reuseComponentOgcSafetyEpoch = true;
	}
	else if(redetectFn && contactsArray && outerIt + 1 < outerIterations)
	{
		if(deferRedetectionToParent)
		{
			reuseComponentOgcSafetyEpoch = false;
			componentOgcSafetyEpochActive = false;
			componentOgcSafetyEpochLimited = false;
			pendingInitialRedetection = false;
			phase = Phase::eREDETECTION;
			return;
		}
		redetectFn(particles, numParticles, softBodies, numSoftBodies,
			*contactsArray, redetectUserData);
		contacts = contactsArray->begin();
		numContacts = contactsArray->size();
		avbdCompileSoftVelocityObjectives(
			workspace->compiledVelocityObjectives,
			workspace->componentFinalizeModes,
			softBodies, numSoftBodies, contacts, numContacts);
		avbdInitializeSoftContactDepenetrationLimits(
			contacts, numContacts, particles,
			softBodies, numSoftBodies, dt);
		avbdBuildSoftParticleContactIndex(
			*workspace, softBodies, numSoftBodies,
			contacts, numContacts, numParticles, stepStats,
			particlePrimalSchedule, validateParticlePrimalAccessPlan, particles);
		reuseComponentOgcSafetyEpoch = false;
		componentOgcSafetyEpochActive = false;
		componentOgcSafetyEpochLimited = false;
	}
	if(stepStats)
		stepStats->redetectMs += stageTimer.getElapsedSeconds() * 1000.0;
	outerIt++;
	phase = Phase::eOUTER_PREPARE;
}

void AvbdSoftBodyStepState::finalizeStep()
{
	// The final material sweep can consume the remaining OGC safety envelope.
	// Publish a same-time DCD epoch before terminal recovery rather than
	// applying a cached normal to the final pose.  Scene-owned deferred
	// redetection is intentionally left to its parent continuation: calling
	// the callback here would race its shared collision workspace.
	if(!deferRedetectionToParent)
		avbdRefreshComponentTerminalOgcEpoch(
			particles, numParticles, softBodies, numSoftBodies,
			redetectFn, contactsArray, redetectUserData,
			contacts, numContacts, *workspace);

	// Component fallback has no native post-AL static recovery.  Perform the
	// narrow true-gap endpoint translation before reconstructing velocity, so
	// its geometric correction cannot become a spurious separating impulse.
	avbdApplyWorldStaticComponentEndpointDcdRecovery(
		particles, numParticles, softBodies, numSoftBodies,
		contacts, numContacts, *workspace);
	for(PxU32 particleIndex = 0;
		particleIndex < numParticles; particleIndex++)
		particles[particleIndex].updateVelocityFromPosition(invDt);
	avbdApplyBendingDamping(
		particles, softBodies, numSoftBodies, dt);
	avbdFinalizeSoftComponentVelocities(
		particles, numParticles, softBodies, numSoftBodies,
		workspace->componentMomentumTargets.begin(),
		workspace->componentFinalizeModes.begin(),
		contacts, numContacts,
		workspace->compiledVelocityObjectives.begin(),
		workspace->compiledVelocityObjectives.size(), invDt);
	avbdProjectSoftContactVelocityTangents(
		particles, numParticles, softBodies, numSoftBodies,
		contacts, numContacts, dt, stepStats);
	avbdClampWorldStaticComponentEndpointDcdVelocities(
		particles, numParticles, softBodies, numSoftBodies,
		contacts, numContacts,
		workspace->worldStaticEndpointRecoveredBodies);
	workspace->contact.invalidateSoftBodyBounds();
	if(stepStats)
	{
		stepStats->velocityMs += stageTimer.getElapsedSeconds() * 1000.0;
		stepStats->workspaceGrowthEvents = workspace->growthEvents;
		stepStats->workspaceGrowthBytes = workspace->growthBytes;
		stepStats->contactWorkspaceGrowthEvents =
			workspace->contact.growthEvents;
		stepStats->contactWorkspaceGrowthBytes =
			workspace->contact.growthBytes;
		stepStats->contactSweepScratchGrowthEvents =
			workspace->contact.sweepScratchGrowthEvents;
		stepStats->contactSweepScratchGrowthBytes =
			workspace->contact.sweepScratchGrowthBytes;
		stepStats->contactOutputGrowthEvents =
			workspace->contact.outputGrowthEvents;
		stepStats->contactOutputGrowthBytes =
			workspace->contact.outputGrowthBytes;
		stepStats->peakContactOutputCount =
			workspace->contact.peakOutputContactCount;
		stepStats->peakContactOutputCapacity =
			workspace->contact.peakOutputContactCapacity;
		stepStats->peakContactIncidenceCount =
			workspace->peakContactIncidenceCount;
		stepStats->peakContactIncidenceCapacity =
			workspace->peakContactIncidenceCapacity;
		stepStats->peakStateTransferContactCount =
			workspace->contact.peakPreviousContactCount;
		stepStats->peakStateTransferContactCapacity =
			workspace->contact.peakPreviousContactCapacity;
		stepStats->peakStateTransferUsedCapacity =
			workspace->contact.peakPreviousUsedCapacity;
	}
}

AvbdSoftBodyStepAdvanceResult AvbdSoftBodyStepState::advance()
{
	for(;;)
	{
		switch(phase)
		{
		case Phase::eOUTER_PREPARE:
			if(outerIt >= outerIterations)
			{
				finalizeStep();
				phase = Phase::eCOMPLETE;
				return AvbdSoftBodyStepAdvanceResult::eCOMPLETE;
			}
			prepareOuterIteration();
			break;
		case Phase::eINNER_BEGIN:
			if(beginInnerSweep())
				return AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY;
			break;
	case Phase::eCAUSAL_LAYER:
		return AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY;
	case Phase::eDUAL:
		updateDualAndRedetect();
		break;
	case Phase::eREDETECTION:
		return AvbdSoftBodyStepAdvanceResult::eREDETECTION_READY;
		case Phase::eCOMPLETE:
			return AvbdSoftBodyStepAdvanceResult::eCOMPLETE;
		case Phase::eINVALID:
		case Phase::eIDLE:
			return AvbdSoftBodyStepAdvanceResult::eINVALID;
		}
	}
}

bool AvbdSoftBodyStepState::getPublishedCausalLayer(
	PxU32& layerIndex, PxU32& packedBegin, PxU32& packedEnd,
	const AvbdParticlePrimalSolveContext*& solveContext,
	const AvbdSoftBody*& bodies, PxU32& bodyCount,
	const PxU32*& particleBodyIndices,
	const PxU32*& packedParticleIndices) const
{
	if(independentBodySweepPublished ||
		phase != Phase::eCAUSAL_LAYER ||
		!causalLayerState.hasPublishedLayer())
		return false;
	layerIndex = causalLayerState.getPublishedLayerIndex();
	causalLayerState.getPublishedPackedRange(packedBegin, packedEnd);
	solveContext = &particlePrimalSolveContext;
	bodies = softBodies;
	bodyCount = numSoftBodies;
	particleBodyIndices = workspace->particlePrimalBodyIndices.begin();
	packedParticleIndices = workspace->particlePrimalColorParticles.begin();
	return true;
}

bool AvbdSoftBodyStepState::getPublishedIndependentBodySweep(
	const AvbdParticlePrimalSolveContext*& solveContext,
	const AvbdSoftBody*& bodies, PxU32& bodyCount) const
{
	if(phase != Phase::eCAUSAL_LAYER ||
		!independentBodySweepPublished || numSoftBodies < 2)
		return false;
	solveContext = &particlePrimalSolveContext;
	bodies = softBodies;
	bodyCount = numSoftBodies;
	return true;
}

bool AvbdSoftBodyStepState::completePublishedIndependentBodySweep(
	const AvbdParticlePrimalRangeObservation* observations,
	PxU32 observationCount)
{
	if(phase != Phase::eCAUSAL_LAYER ||
		!independentBodySweepPublished || !observations ||
		observationCount == 0)
		return false;
	particlePrimalObservation = AvbdParticlePrimalRangeObservation();
	for(PxU32 observationIndex = 0;
		observationIndex < observationCount; observationIndex++)
		particlePrimalObservation.merge(observations[observationIndex]);
	independentBodySweepPublished = false;
	finishParticlePrimalSweep();
	return true;
}

bool AvbdSoftBodyStepState::completePublishedCausalLayer(
	const AvbdParticlePrimalRangeObservation* observations,
	PxU32 observationCount)
{
	if(phase != Phase::eCAUSAL_LAYER ||
		!causalLayerState.completePublishedLayer(
			observations, observationCount))
		return false;
	if(causalLayerState.hasPublishedLayer())
		return true;
	particlePrimalObservation = causalLayerState.getSweepObservation();
	finishParticlePrimalSweep();
	return true;
}

void AvbdSoftBodyStepState::runToCompletionSerial()
{
	for(;;)
	{
		const AvbdSoftBodyStepAdvanceResult result = advance();
		if(result == AvbdSoftBodyStepAdvanceResult::eREDETECTION_READY)
		{
			if(!redetectFn || !contactsArray)
			{
				phase = Phase::eINVALID;
				return;
			}
			redetectFn(particles, numParticles, softBodies, numSoftBodies,
				*contactsArray, redetectUserData);
			if(!completePendingRedetection())
			{
				phase = Phase::eINVALID;
				return;
			}
			continue;
		}
		if(result != AvbdSoftBodyStepAdvanceResult::eCAUSAL_LAYER_READY)
			return;
		PxU32 layerIndex = 0;
		PxU32 packedBegin = 0;
		PxU32 packedEnd = 0;
		const AvbdParticlePrimalSolveContext* solveContext = NULL;
		const AvbdSoftBody* bodies = NULL;
		PxU32 bodyCount = 0;
		const PxU32* particleBodyIndices = NULL;
		const PxU32* packedParticleIndices = NULL;
		const bool published = getPublishedCausalLayer(
			layerIndex, packedBegin, packedEnd, solveContext,
			bodies, bodyCount, particleBodyIndices, packedParticleIndices);
		PX_UNUSED(layerIndex);
		if(!published)
		{
			const AvbdParticlePrimalSolveContext* bodySolveContext = NULL;
			const AvbdSoftBody* bodyRangeBodies = NULL;
			PxU32 bodyRangeCount = 0;
			if(!getPublishedIndependentBodySweep(
				bodySolveContext, bodyRangeBodies, bodyRangeCount))
			{
				phase = Phase::eINVALID;
				return;
			}
			AvbdParticlePrimalRangeObservation bodyObservation;
			avbdSolveParticlePrimalIndependentBodyRange(
				*bodySolveContext, bodyRangeBodies, bodyRangeCount,
				0, bodyRangeCount, bodyObservation);
			if(!completePublishedIndependentBodySweep(
				&bodyObservation, 1))
			{
				phase = Phase::eINVALID;
				return;
			}
			continue;
		}
		AvbdParticlePrimalRangeObservation observation;
		avbdSolveParticlePrimalPackedRange(
			*solveContext, bodies, bodyCount, particleBodyIndices,
			numParticles, packedParticleIndices, packedBegin, packedEnd,
			observation);
		if(!completePublishedCausalLayer(&observation, 1))
		{
			phase = Phase::eINVALID;
			return;
		}
	}
}

#endif // DY_AVBD_SOFT_BODY_STEP_STATE_IMPLEMENTATION

// Canonical scalar component step. Its implementation has one dedicated
// LowLevelDynamics translation unit so Scene/P4/P5 control code cannot
// perturb the reference kernel's code generation or instruction layout.
void avbdStepSoftBodies(
	AvbdSoftParticle* particles, PxU32 numParticles,
	AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdSoftContact* contacts, PxU32 numContacts,
	PxReal dt, const PxVec3& gravity,
	PxU32 outerIterations = 1, PxU32 innerIterations = 10,
	PxReal avbdBeta = 1000.0f,
	AvbdContactRedetectFn redetectFn = NULL,
	PxArray<AvbdSoftContact>* contactsArray = NULL,
	void* redetectUserData = NULL,
	PxReal chebyshevRho = 0.92f,
	AvbdSoftBodyStepStats* stepStats = NULL,
	AvbdSoftBodyWorkspace* persistentWorkspace = NULL,
	PxU32 totalInnerIterationBudget = 0,
	const AvbdSelfCollisionAdjacency*
		selfCollisionAdjacencies = NULL,
	PxU32 numSelfCollisionAdjacencies = 0,
	const PxU8* selfCollisionEnabled = NULL,
	const AvbdOGCParams* ogcParams = NULL,
	AvbdSoftBodyStepExecutionMode executionMode =
		AvbdSoftBodyStepExecutionMode::eFULL,
	AvbdParticlePrimalSchedule inputParticlePrimalSchedule =
		AvbdParticlePrimalSchedule::eDEFAULT);

#if defined(DY_AVBD_SOFT_BODY_SCALAR_STEP_IMPLEMENTATION)

void avbdStepSoftBodies(
	AvbdSoftParticle* particles, PxU32 numParticles,
	AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdSoftContact* contacts, PxU32 numContacts,
	PxReal dt, const PxVec3& gravity,
	PxU32 outerIterations, PxU32 innerIterations,
	PxReal avbdBeta,
	AvbdContactRedetectFn redetectFn,
	PxArray<AvbdSoftContact>* contactsArray,
	void* redetectUserData,
	PxReal chebyshevRho,
	AvbdSoftBodyStepStats* stepStats,
	AvbdSoftBodyWorkspace* persistentWorkspace,
	PxU32 totalInnerIterationBudget,
	const AvbdSelfCollisionAdjacency* selfCollisionAdjacencies,
	PxU32 numSelfCollisionAdjacencies,
	const PxU8* selfCollisionEnabled,
	const AvbdOGCParams* ogcParams,
	AvbdSoftBodyStepExecutionMode executionMode,
	AvbdParticlePrimalSchedule inputParticlePrimalSchedule)
{
	if (numParticles == 0 || numSoftBodies == 0) return;
	if(executionMode != AvbdSoftBodyStepExecutionMode::eFULL &&
		!persistentWorkspace)
	{
		PX_ASSERT(false);
		return;
	}
	// A total budget lets callers retain the outer contact-redetection
	// schedule without rounding every stage up to a full inner batch.
	const PxU32 requestedInnerIterationBudget =
		totalInnerIterationBudget > 0
			? PxMax(totalInnerIterationBudget, outerIterations)
			: outerIterations * innerIterations;
	AvbdSoftBodyWorkspace localWorkspace;
	AvbdSoftBodyWorkspace& workspace =
		persistentWorkspace ? *persistentWorkspace : localWorkspace;
	PxArray<AvbdCompiledSoftVelocityObjective>&
		compiledVelocityObjectives =
			workspace.compiledVelocityObjectives;
	PxArray<AvbdSoftComponentFinalizeMode>&
		componentFinalizeModes = workspace.componentFinalizeModes;
	if(executionMode != AvbdSoftBodyStepExecutionMode::eRESUME)
	{
		for(PxU32 contactIdx = 0; contactIdx < numContacts; contactIdx++)
		{
			const AvbdSoftContactGeometry& geometry =
				contacts[contactIdx].geometry;
			const AvbdSoftContactTargetKind targetKind =
				geometry.targetKind;
			if(targetKind !=
					AvbdSoftContactTargetKind::eWORLD_STATIC &&
				targetKind !=
					AvbdSoftContactTargetKind::eKINEMATIC_RIGID &&
				targetKind !=
					AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE)
			{
				// This Scene-external component has no rigid 6x6 block. Accepting
				// a rigid target here would silently turn a two-sided objective
				// into a one-way particle correction.
				PX_ASSERT(false);
				return;
			}
			const bool positionOwned =
				(targetKind ==
						AvbdSoftContactTargetKind::eWORLD_STATIC ||
				 targetKind ==
						AvbdSoftContactTargetKind::
							eDEFORMABLE_SURFACE) &&
				geometry.velocityOwner ==
					AvbdVelocityObjectiveOwner::PositionAL;
			const bool componentOwned =
				targetKind ==
					AvbdSoftContactTargetKind::eKINEMATIC_RIGID &&
				geometry.velocityOwner ==
					AvbdVelocityObjectiveOwner::
						ComponentFinalize;
			if(!positionOwned && !componentOwned)
			{
				// Prep must assign exactly one compatible owner.  No solve stage
				// is allowed to reinterpret target kind or flags later.
				PX_ASSERT(false);
				return;
			}
		}
		for (PxU32 si = 0; si < numSoftBodies; si++)
		{
			PX_ASSERT(
				softBodies[si].runtime.isObjectiveProgramCurrent(
					softBodies[si].compiled.particleStart,
					softBodies[si].compiled.particleCount));
		}
		if(stepStats)
		{
			stepStats->reset();
			stepStats->requestedOuterIterations = outerIterations;
			stepStats->requestedInnerIterations =
				requestedInnerIterationBudget;
		}
		workspace.beginStep();
		workspace.contact.prepareSoftBodyBounds(numSoftBodies);
		compiledVelocityObjectives.clear();
		workspace.resize(componentFinalizeModes, numSoftBodies);
		for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
		{
			componentFinalizeModes[bodyIndex] =
				softBodies[bodyIndex].runtime.compiledObjectives.empty()
					? AvbdSoftComponentFinalizeMode::eMOMENTUM
					: AvbdSoftComponentFinalizeMode::ePOSITION_OWNED;
			const PxU32 particleStart =
				softBodies[bodyIndex].compiled.particleStart;
			const PxU32 particleCount =
				softBodies[bodyIndex].compiled.particleCount;
			if(particleStart > numParticles ||
				particleCount > numParticles - particleStart)
			{
				componentFinalizeModes[bodyIndex] =
					AvbdSoftComponentFinalizeMode::eUNSUPPORTED;
				continue;
			}
			for(PxU32 localIndex = 0;
				localIndex < particleCount; localIndex++)
			{
				if(particles[particleStart + localIndex].invMass <= 0.0f)
				{
					componentFinalizeModes[bodyIndex] =
						AvbdSoftComponentFinalizeMode::ePOSITION_OWNED;
					break;
				}
			}
		}
	}
	PxTime stageTimer;
	if(executionMode != AvbdSoftBodyStepExecutionMode::eRESUME)
	{
		avbdCompileSoftVelocityObjectives(
			compiledVelocityObjectives, componentFinalizeModes,
			softBodies, numSoftBodies, contacts, numContacts);
		// A persistent contact carries AL/friction state across frames, but its
		// finite depenetration bias is a one-frame target.
		avbdResetSoftContactDepenetrationLimits(
			contacts, numContacts);
	}
	if(executionMode == AvbdSoftBodyStepExecutionMode::ePREPARE)
		return;

	PxReal invDt = dt > 0.0f ? 1.0f / dt : 0.0f;
	PxReal invDtSq = invDt * invDt;

	// Stage 1: prediction. eRESUME is entered only after an owner has written
	// these disjoint particle fields through the P3 continuation boundary.
	if(executionMode == AvbdSoftBodyStepExecutionMode::eFULL)
	{
		const bool useRigidInitialGuess =
			avbdCanUseSoftRigidPrimalInitialization(
				particles, numParticles, softBodies, numSoftBodies);
		const bool useAdaptiveInitialGuess = !useRigidInitialGuess &&
			avbdCanUseSoftAdaptivePrimalInitialization(
				particles, numParticles, softBodies, numSoftBodies);
		avbdPredictSoftBodyParticles(
			particles, numParticles, dt, gravity, useAdaptiveInitialGuess);
		for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; ++bodyIndex)
		{
			if(useRigidInitialGuess)
				avbdApplySoftBodyRigidPrimalInitialGuess(
					particles, numParticles, softBodies[bodyIndex]);
			avbdComputeSoftBodyBounds(
				particles, softBodies[bodyIndex],
				workspace.contact.softBodyBounds[bodyIndex]);
			workspace.contact.softBodyBoundsReady[bodyIndex] = 1;
		}
		workspace.contact.markSoftBodyBoundsReady();
	}
	avbdPublishCorotationalTetPacketIrStats(
		softBodies, numSoftBodies, stepStats);
	// Contact prep before prediction cannot see a first-impact candidate.
	// Refresh once after predictedPosition is current so speculative plane
	// and swept rigid-SDF contacts can constrain the same timestep instead of
	// recovering from an already intersecting state on the next frame.
	if(redetectFn && contactsArray)
	{
		redetectFn(
			particles, numParticles,
			softBodies, numSoftBodies,
			*contactsArray, redetectUserData);
		contacts = contactsArray->begin();
		numContacts = contactsArray->size();
		avbdCompileSoftVelocityObjectives(
			compiledVelocityObjectives, componentFinalizeModes,
			softBodies, numSoftBodies, contacts, numContacts);
	}
	avbdInitializeSoftContactDepenetrationLimits(
		contacts, numContacts, particles,
		softBodies, numSoftBodies, dt);
	PxArray<AvbdSoftComponentMomentumTarget>&
		componentMomentumTargets =
			workspace.componentMomentumTargets;
	workspace.resize(componentMomentumTargets, numSoftBodies);
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		AvbdSoftComponentMomentumTarget& target =
			componentMomentumTargets[bodyIndex];
		target = AvbdSoftComponentMomentumTarget();
		if(componentFinalizeModes[bodyIndex] ==
				AvbdSoftComponentFinalizeMode::eUNSUPPORTED)
			continue;
		PxVec3 centroid(0.0f);
		PxMat33 inertia(PxZero);
		target.valid = avbdComputeSoftComponentMomentum(
			particles, numParticles, softBodies[bodyIndex],
			true, invDt, centroid, target.linearMomentum,
			target.angularMomentum, inertia, target.mass);
		target.centroid = centroid;
		avbdApplySoftComponentDampingToMomentumTarget(
			target, softBodies[bodyIndex], dt);
		PX_UNUSED(inertia);
	}
	if(stepStats)
		stepStats->predictionMs += stageTimer.getElapsedSeconds() * 1000.0;

	// P4 policy is selected once at the step boundary.  Contact-index rebuilds
	// are part of every OGC epoch and must not perform runtime configuration
	// discovery themselves.
	const AvbdParticlePrimalSchedule particlePrimalSchedule =
		inputParticlePrimalSchedule ==
			AvbdParticlePrimalSchedule::eDEFAULT
			? avbdGetParticlePrimalSchedule() : inputParticlePrimalSchedule;
	const bool validateParticlePrimalAccessPlan =
		avbdValidateParticlePrimalAccessPlan();

	// Build per-particle contact index to avoid O(particles*contacts) scan.
	// contactStart[pi] = first index into contactIdx for particle pi.
	// contactIdx stores contact indices grouped by particle.
	avbdBuildSoftParticleContactIndex(
		workspace, softBodies, numSoftBodies,
		contacts, numContacts, numParticles, stepStats,
		particlePrimalSchedule, validateParticlePrimalAccessPlan, particles);
	PxArray<AvbdSoftContactParticleRef>& contactIdxBuf =
		workspace.contactIndices;
	const PxArray<PxU32>& contactStart = workspace.contactStarts;

	if(stepStats)
		stepStats->contactIndexMs +=
			stageTimer.getElapsedSeconds() * 1000.0;

	// Chebyshev semi-iterative acceleration state.
	// If chebyshevRho > 0, we use adaptive spectral-radius estimation:
	// measure the actual GS convergence rate from inner iterations 0-1,
	// then use min(measured, user-provided) as the Chebyshev parameter.
	// This prevents over-relaxation on meshes whose spectral radius
	// differs from the user's estimate (e.g., non-uniform voxel meshes).
	const bool useChebyshev = (chebyshevRho > 0.0f && chebyshevRho < 1.0f);
	PxReal chebyOmega = 1.0f;
	PxReal adaptiveRho = chebyshevRho;
	PxArray<PxVec3>& chebyPrevPos = workspace.chebyPrevPos;
	PxArray<PxVec3>& chebyPrevPrevPos = workspace.chebyPrevPrevPos;
	PxArray<PxReal>& selfCollisionSafetyBounds =
		workspace.selfCollisionSafetyBounds;
	PxArray<PxReal>& bodySelfCollisionSafetyBounds =
		workspace.bodySelfCollisionSafetyBounds;
	workspace.resize(selfCollisionSafetyBounds, numParticles);
	if (useChebyshev)
	{
		workspace.resize(chebyPrevPos, numParticles);
		workspace.resize(chebyPrevPrevPos, numParticles);
		for (PxU32 i = 0; i < numParticles; i++)
		{
			chebyPrevPos[i] = particles[i].position;
			chebyPrevPrevPos[i] = particles[i].position;
		}
	}
	// Main iteration loop
	PxU32 remainingInnerIterationBudget =
		requestedInnerIterationBudget;
	bool reuseComponentOgcSafetyEpoch = false;
	bool componentOgcSafetyEpochActive = false;
	for (PxU32 outerIt = 0; outerIt < outerIterations; outerIt++)
	{
		if(stepStats)
			stepStats->executedOuterIterations++;
		const PxU64 particleSweepsBeforeOuter = stepStats
			? stepStats->particleSweeps : 0;
		const PxU32 remainingOuterIterations =
			outerIterations - outerIt;
		const PxU32 currentInnerIterations =
			(remainingInnerIterationBudget +
				remainingOuterIterations - 1) /
			remainingOuterIterations;
		remainingInnerIterationBudget -= currentInnerIterations;
		if(!reuseComponentOgcSafetyEpoch)
		{
			// Snapshot positions as proximal anchor for the AVBD elastic term.
			avbdSnapshotOuterPositionsScalar(
				particles, numParticles, selfCollisionSafetyBounds.begin());

			// OGC Eq. 21-27: each fresh DCD epoch records a known
			// penetration-free anchor and a conservative displacement radius.
			const AvbdOGCParams defaultOgcParams;
			const AvbdOGCParams& activeOgcParams =
				ogcParams ? *ogcParams : defaultOgcParams;
			if(selfCollisionAdjacencies)
			{
				for(PxU32 bodyIndex = 0;
					bodyIndex < numSoftBodies &&
					bodyIndex < numSelfCollisionAdjacencies;
					bodyIndex++)
				{
					if(selfCollisionEnabled &&
						!selfCollisionEnabled[bodyIndex])
						continue;
					const AvbdSoftBody& body =
						softBodies[bodyIndex];
					avbdComputeSafetyBounds(
						body, particles,
						selfCollisionAdjacencies[bodyIndex],
						activeOgcParams.contactRadius,
						activeOgcParams.safetyRelax,
						bodySelfCollisionSafetyBounds,
						workspace.contact);
					for(PxU32 localIndex = 0;
						localIndex < body.compiled.particleCount;
						localIndex++)
					{
						const PxU32 particleIndex =
							body.compiled.particleStart +
							localIndex;
						if(particleIndex < numParticles)
							selfCollisionSafetyBounds[
								particleIndex] =
								bodySelfCollisionSafetyBounds[
									localIndex];
					}
				}
			}
			componentOgcSafetyEpochActive =
				avbdApplyComponentOgcEpochSafetyBounds(
					contacts, numContacts, softBodies, numSoftBodies, particles,
					activeOgcParams.contactRadius,
					activeOgcParams.safetyRelax,
					selfCollisionSafetyBounds.begin(), numParticles,
					workspace);
		}
		reuseComponentOgcSafetyEpoch = false;
		bool componentOgcSafetyEpochLimited = false;

		// Reset Chebyshev state each outer iteration: the system changes
		// (contacts re-detected, elasticK updated) so prior omega/positions
		// are invalid.
		if (useChebyshev)
		{
			chebyOmega = 1.0f;
			for (PxU32 i = 0; i < numParticles; i++)
			{
				chebyPrevPos[i] = particles[i].position;
				chebyPrevPrevPos[i] = particles[i].position;
			}
		}

		PxReal prevMaxDxSq = 0.0f;
		PxU32 shadowResidual1e5ConsecutiveSweeps = 0;
		PxU32 shadowResidual1e4ConsecutiveSweeps = 0;
		bool shadowResidual1e5Recorded = false;
		bool shadowResidual1e4Recorded = false;
		bool legacyAppliedConvergenceRecorded = false;
		AvbdSoftResidualConvergenceTracker residualConvergence(
			1e-8f, 2);

		for (PxU32 innerIt = 0;
			innerIt < currentInnerIterations; innerIt++)
		{
			if(stepStats)
			{
				stepStats->executedInnerIterations++;
				stepStats->particleSweeps++;
			}
			PxReal maxDxSq = 0.0f;
			AvbdParticlePrimalRangeObservation particlePrimalObservation;
			AvbdSoftSweepConvergenceObservation& sweepObservation =
				particlePrimalObservation.sweepObservation;

			// Canonical scalar reference traversal.  Causal-layer scheduling is
			// owned by the Scene continuation boundary; it must not remain as a
			// default-off branch in this scalar primal kernel.
			const AvbdParticlePrimalSolveContext particlePrimalSolveContext =
			{
				particles,
				contacts,
				contactStart.begin(),
				contactIdxBuf.begin(),
				selfCollisionSafetyBounds.begin(),
				invDt,
				invDtSq,
				avbdSelectCorotationalTetPacketKernel(
					softBodies, numSoftBodies)
			};

			if(particlePrimalSolveContext.corotationalTetPacketKernel)
				avbdSolveParticlePrimalCorotationalTetPacketBodyRange(
					particlePrimalSolveContext, softBodies, numSoftBodies,
					particlePrimalObservation);
			else
			{
				for(PxU32 bodyIndex = 0;
					bodyIndex < numSoftBodies; bodyIndex++)
				{
					const AvbdSoftBody& body = softBodies[bodyIndex];
					for(PxU32 localIndex = 0;
						localIndex < body.compiled.particleCount; localIndex++)
						particlePrimalSolveContext.solve(
							body, localIndex, particlePrimalObservation);
				}
			}
			maxDxSq =
				sweepObservation.maxAppliedDisplacementSq;
			if(stepStats)
			{
				stepStats->tetLinearizationCacheFallbackParticleSteps +=
					particlePrimalObservation.
						tetLinearizationCacheFallbackParticleSteps;
				stepStats->trustRegionLimitedParticleSteps +=
					sweepObservation.trustRegionLimitedSteps;
				stepStats->positiveJLimitedParticleSteps +=
					sweepObservation.positiveJLimitedSteps;
				stepStats->positiveJRejectedParticleSteps +=
					sweepObservation.positiveJRejectedSteps;
				stepStats->nonFiniteRejectedParticleSteps +=
					sweepObservation.nonFiniteRejectedSteps;
				stepStats->finalMaxLocalSolveDisplacement =
					PxSqrt(
						sweepObservation.
							maxLocalSolveDisplacementSq);
				stepStats->finalMaxAppliedDisplacement =
					PxSqrt(maxDxSq);
				// Compatibility alias for the original schema.
				stepStats->finalMaxDisplacement =
					stepStats->finalMaxAppliedDisplacement;
			}
			if(sweepObservation.trustRegionLimitedSteps > 0)
				componentOgcSafetyEpochLimited = true;
			if(componentOgcSafetyEpochActive &&
				componentOgcSafetyEpochLimited)
			{
				innerIt = currentInnerIterations;
				break;
			}

			// A small applied displacement is not enough to terminate: a
			// trust-region or positive-J rejection can produce a zero step
			// while the local H^-1 f stationarity residual is still active.
			// Keep the legacy candidate count as diagnostics, but only the
			// pre-limiter residual below 1e-4 for two consecutive feasible
			// sweeps owns early termination.
			const bool appliedDisplacementConverged =
				sweepObservation.isAppliedDisplacementConverged(
					1e-12f);
			const bool strictResidualCandidateConverged =
				sweepObservation.isResidualConverged(1e-12f);
			const bool residualPolicyConverged =
				residualConvergence.observe(sweepObservation);
			const bool shadowResidual1e5Converged =
				sweepObservation.isResidualConverged(1e-10f);
			const bool shadowResidual1e4Converged =
				sweepObservation.isResidualConverged(1e-8f);
			shadowResidual1e5ConsecutiveSweeps =
				shadowResidual1e5Converged
					? shadowResidual1e5ConsecutiveSweeps + 1
					: 0;
			shadowResidual1e4ConsecutiveSweeps =
				shadowResidual1e4Converged
					? shadowResidual1e4ConsecutiveSweeps + 1
					: 0;
			if(!shadowResidual1e5Recorded &&
				shadowResidual1e5ConsecutiveSweeps >= 2)
			{
				shadowResidual1e5Recorded = true;
				if(stepStats)
				{
					stepStats->
						shadowResidual1e5ConvergedOuterIterations++;
					stepStats->
						shadowResidual1e5SavedInnerIterations +=
						currentInnerIterations - (innerIt + 1);
				}
			}
			if(!shadowResidual1e4Recorded &&
				shadowResidual1e4ConsecutiveSweeps >= 2)
			{
				shadowResidual1e4Recorded = true;
				if(stepStats)
				{
					stepStats->
						shadowResidual1e4ConvergedOuterIterations++;
					stepStats->
						shadowResidual1e4SavedInnerIterations +=
						currentInnerIterations - (innerIt + 1);
				}
			}
			if(appliedDisplacementConverged &&
				!legacyAppliedConvergenceRecorded)
			{
				legacyAppliedConvergenceRecorded = true;
				if(stepStats)
				{
					stepStats->
						legacyAppliedConvergedOuterIterations++;
					if(!strictResidualCandidateConverged)
						stepStats->
							unsafeAppliedConvergenceCandidates++;
				}
			}
			if(residualPolicyConverged)
			{
				if(stepStats)
					stepStats->residualConvergedOuterIterations++;
				break;
			}
			if(stepStats &&
				innerIt + 1 == currentInnerIterations)
				stepStats->budgetExhaustedOuterIterations++;

			// Adaptive spectral-radius estimation.
			// Iterations 0-1 are pure GS (Chebyshev starts at iteration 2).
			// Measure the GS convergence ratio from these iterations and use
			// min(measured, user-provided) as the Chebyshev rho.  This makes
			// the solver adapt to any mesh density / quality automatically.
			if (innerIt == 0)
			{
				prevMaxDxSq = maxDxSq;
			}
			else if (innerIt == 1 && useChebyshev)
			{
				if (prevMaxDxSq > 1e-20f)
				{
					PxReal measuredRho = PxSqrt(maxDxSq / prevMaxDxSq);
					// Use the more conservative of measured vs user-provided,
					// and never exceed 0.95 (safety ceiling).
					adaptiveRho = PxMin(measuredRho, chebyshevRho);
					adaptiveRho = PxMin(adaptiveRho, 0.95f);
				}
				prevMaxDxSq = maxDxSq;
			}

			// Chebyshev semi-iterative position relaxation
			// x_acc = x_{k-2} + omega_k * (x_GS - x_{k-2})
			if (useChebyshev && innerIt >= 2)
			{
				PxReal rhoSq = adaptiveRho * adaptiveRho;
				if (innerIt == 2)
					chebyOmega = 2.0f / (2.0f - rhoSq);
				else
					chebyOmega = 1.0f / (1.0f - rhoSq * chebyOmega * 0.25f);
				chebyOmega = PxMax(1.0f, PxMin(chebyOmega, 2.0f));

				// Divergence guard: if displacement grew since last iteration,
				// the rho is still too high.  Disable Chebyshev for the
				// remainder of this outer iteration.
				if (prevMaxDxSq > 1e-20f && maxDxSq > prevMaxDxSq * 1.1f)
				{
					chebyOmega = 1.0f;   // effectively no acceleration
					adaptiveRho = 0.0f;  // stays disabled for remaining inner its
				}

				if (chebyOmega > 1.0f)
				{
					for (PxU32 i = 0; i < numParticles; i++)
					{
						if (particles[i].isStatic()) continue;
						// Skip Chebyshev for particles with active contacts
						// (over-relaxation can push them through surfaces)
						if (contactStart[i + 1] > contactStart[i]) continue;
						particles[i].position = chebyPrevPrevPos[i] +
							(particles[i].position - chebyPrevPrevPos[i]) * chebyOmega;
						avbdTruncateDisplacement(
							particles[i],
							particles[i].outerPosition,
							selfCollisionSafetyBounds[i]);
					}
				}
				prevMaxDxSq = maxDxSq;
			}
			if (useChebyshev)
			{
				for (PxU32 i = 0; i < numParticles; i++)
				{
					chebyPrevPrevPos[i] = chebyPrevPos[i];
					chebyPrevPos[i] = particles[i].position;
				}
			}
		}
		if(stepStats && avbdUseParticlePrimalWorkCensus())
		{
			avbdAccumulateParticlePrimalWorkCensusForOuterEpoch(
				*stepStats, particles, softBodies, numSoftBodies,
				contactStart.begin(),
				stepStats->particleSweeps - particleSweepsBeforeOuter);
		}
		if(stepStats)
			stepStats->particleSolveMs +=
				stageTimer.getElapsedSeconds() * 1000.0;
		// Dual update (contacts, pins, elastic proximal)
		for (PxU32 ci = 0; ci < numContacts; ci++)
		{
			AvbdSoftContact& contact = contacts[ci];
			avbdUpdateSoftContactDual(
				contact.geometry, contact.state,
				particles, avbdBeta);
		}

		for (PxU32 si = 0; si < numSoftBodies; si++)
		{
			AvbdSoftBody& sb = softBodies[si];
			for (PxU32 oi = 0;
				oi < sb.runtime.compiledObjectives.size(); oi++)
			{
				const AvbdCompiledSoftObjective& objective =
					sb.runtime.compiledObjectives[oi];
				if (avbdIsPinPositionOwner(objective.owner))
				{
					avbdUpdatePinDual(
						sb.runtime.pins[objective.runtimeStateIndex],
						objective.point, particles, avbdBeta);
				}
				else
				{
					PX_ASSERT(
						avbdIsPinPositionOwner(
							objective.owner));
				}
			}
		}

		// AVBD elastic proximal dual update: increase proximal weight
		// proportional to displacement from the outer-iteration anchor.
		// The A/B-off route clears this state during prediction and never
		// regrows it, while leaving every other dual update unchanged.
		if(avbdUseSoftElasticProximal())
		{
			for (PxU32 i = 0; i < numParticles; i++)
			{
				AvbdSoftParticle& sp = particles[i];
				if (sp.isStatic()) continue;
				PxReal disp = (sp.position - sp.outerPosition).magnitude();
				sp.elasticK = PxMin(
					sp.elasticK + avbdBeta * disp, sp.elasticKMax);
			}
		}
		if(stepStats)
			stepStats->dualMs +=
				stageTimer.getElapsedSeconds() * 1000.0;

		// Re-detect only after the conservative soft/soft envelope was spent.
		// Otherwise this is the same DCD epoch and the manifold/AL state remains
		// current by construction; rebuilding it per outer iteration is the
		// sustained-contact cost this OGC scheduler is designed to remove.
		const bool mayReusePureSoftPairEpoch =
			redetectFn && contactsArray && outerIt + 1 < outerIterations &&
			componentOgcSafetyEpochActive &&
			!componentOgcSafetyEpochLimited;
		if(mayReusePureSoftPairEpoch)
		{
			reuseComponentOgcSafetyEpoch = true;
		}
		else if (redetectFn && contactsArray &&
			outerIt + 1 < outerIterations)
		{
			redetectFn(particles, numParticles, softBodies, numSoftBodies,
					   *contactsArray, redetectUserData);
			contacts = contactsArray->begin();
			numContacts = contactsArray->size();
			avbdCompileSoftVelocityObjectives(
				compiledVelocityObjectives, componentFinalizeModes,
				softBodies, numSoftBodies, contacts, numContacts);
			// Matching rows retain the original frame anchor through state
			// transfer; only contacts born at this redetection are initialized.
			avbdInitializeSoftContactDepenetrationLimits(
				contacts, numContacts, particles,
				softBodies, numSoftBodies, dt);
			// Rebuild this epoch's per-particle contact index and causal plan.
			avbdBuildSoftParticleContactIndex(
				workspace, softBodies, numSoftBodies,
				contacts, numContacts, numParticles, stepStats,
				particlePrimalSchedule,
				validateParticlePrimalAccessPlan, particles);
			componentOgcSafetyEpochActive = false;
		}
		if(stepStats)
			stepStats->redetectMs +=
				stageTimer.getElapsedSeconds() * 1000.0;
	}

	// Stage 3: terminal same-time DCD, recovery, then velocity update.  This
	// is a contact-epoch refresh, not a time substep or a swept CCD pass.
	avbdRefreshComponentTerminalOgcEpoch(
		particles, numParticles, softBodies, numSoftBodies,
		redetectFn, contactsArray, redetectUserData,
		contacts, numContacts, workspace);
	avbdApplyWorldStaticComponentEndpointDcdRecovery(
		particles, numParticles, softBodies, numSoftBodies,
		contacts, numContacts, workspace);
	for (PxU32 i = 0; i < numParticles; i++)
		particles[i].updateVelocityFromPosition(invDt);
	avbdApplyBendingDamping(
		particles, softBodies, numSoftBodies, dt);
	avbdFinalizeSoftComponentVelocities(
		particles, numParticles,
		softBodies, numSoftBodies,
		componentMomentumTargets.begin(),
		componentFinalizeModes.begin(),
		contacts, numContacts,
		compiledVelocityObjectives.begin(),
		compiledVelocityObjectives.size(), invDt);
	avbdProjectSoftContactVelocityTangents(
		particles, numParticles, softBodies, numSoftBodies,
		contacts, numContacts, dt, stepStats);
	avbdClampWorldStaticComponentEndpointDcdVelocities(
		particles, numParticles, softBodies, numSoftBodies,
		contacts, numContacts,
		workspace.worldStaticEndpointRecoveredBodies);
	// A completed solve has advanced positions again; never expose the
	// one-redetection prediction cache to an unrelated later query.
	workspace.contact.invalidateSoftBodyBounds();
	if(stepStats)
		stepStats->velocityMs += stageTimer.getElapsedSeconds() * 1000.0;
	if(stepStats)
	{
		stepStats->workspaceGrowthEvents = workspace.growthEvents;
		stepStats->workspaceGrowthBytes = workspace.growthBytes;
		stepStats->contactWorkspaceGrowthEvents =
			workspace.contact.growthEvents;
		stepStats->contactWorkspaceGrowthBytes =
			workspace.contact.growthBytes;
		stepStats->contactSweepScratchGrowthEvents =
			workspace.contact.sweepScratchGrowthEvents;
		stepStats->contactSweepScratchGrowthBytes =
			workspace.contact.sweepScratchGrowthBytes;
		stepStats->contactOutputGrowthEvents =
			workspace.contact.outputGrowthEvents;
		stepStats->contactOutputGrowthBytes =
			workspace.contact.outputGrowthBytes;
		stepStats->peakContactOutputCount =
			workspace.contact.peakOutputContactCount;
		stepStats->peakContactOutputCapacity =
			workspace.contact.peakOutputContactCapacity;
		stepStats->peakContactIncidenceCount =
			workspace.peakContactIncidenceCount;
		stepStats->peakContactIncidenceCapacity =
			workspace.peakContactIncidenceCapacity;
		stepStats->peakStateTransferContactCount =
			workspace.contact.peakPreviousContactCount;
		stepStats->peakStateTransferContactCapacity =
			workspace.contact.peakPreviousContactCapacity;
		stepStats->peakStateTransferUsedCapacity =
			workspace.contact.peakPreviousUsedCapacity;
	}

}

#endif // DY_AVBD_SOFT_BODY_SCALAR_STEP_IMPLEMENTATION
} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_COMPONENT_H
