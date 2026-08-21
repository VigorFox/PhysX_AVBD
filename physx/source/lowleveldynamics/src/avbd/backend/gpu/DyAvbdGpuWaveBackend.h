// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_GPU_WAVE_BACKEND_H
#define DY_AVBD_GPU_WAVE_BACKEND_H

#include "foundation/PxSimpleTypes.h"

namespace physx
{
namespace Dy
{
class AvbdSolver;
class AvbdDynamicsContext;
struct AvbdRigidSolveContext;

// Opaque CPU producer boundary for the attached GPU owner-wave backend. The
// callback table is intentionally expressed only in terms of opaque context
// and packet pointers; the GPU side must not link against CPU producer,
// fallback, or writeback symbols.  The CPU owner supplies the table for the
// lifetime of the attached backend and may reject a callback transaction by
// returning false.
static const PxU32 AVBD_RIGID_GPU_WAVE_CALLBACK_VERSION = 1u;

struct AvbdRigidGpuWaveCallbackTable
{
	PxU32 version;
	void* userData;
	bool (*preparePacket)(
		void* userData, const void* solveContext, PxU32 waveIndex,
		PxU32 waveBodyOffset, PxU32 epoch, PxF32 avbdAlpha,
		void* opaquePacket);
	bool (*executeScalarFallback)(
		void* userData, void* scalarSolver, void* solveContext,
		const void* opaquePacket, PxU8 validMask);
	bool (*commitWriteback)(
		void* userData, void* solveContext, const void* opaquePacket,
		const void* opaqueSolution);

	AvbdRigidGpuWaveCallbackTable()
		: version(0), userData(NULL), preparePacket(NULL),
		  executeScalarFallback(NULL), commitWriteback(NULL)
	{
	}

	bool isComplete() const
	{
		return version == AVBD_RIGID_GPU_WAVE_CALLBACK_VERSION && userData &&
			preparePacket && executeScalarFallback && commitWriteback;
	}
};

// Build the canonical CPU-owned callback table.  `userData` is an opaque
// lifetime token supplied by the owner (normally its AvbdDynamicsContext);
// callbacks never dereference it, but the non-null token prevents accidental
// admission of a stateless/incomplete table.
void avbdGetRigidGpuWaveCallbackTable(
	AvbdRigidGpuWaveCallbackTable& table, void* userData);

// A narrow owner-side sink lets a scene/factory integration configure a GPU
// AVBD context without depending on its concrete GPU implementation class.
// The sink deliberately exposes only callback-table lifetime operations; it
// does not expose PxgGpuContext or any PGS/TGS controller state.
class AvbdRigidGpuWaveCallbackSink
{
public:
	virtual ~AvbdRigidGpuWaveCallbackSink() {}
	virtual bool setCpuWaveCallbacks(
		const AvbdRigidGpuWaveCallbackTable& callbacks) = 0;
	virtual void clearCpuWaveCallbacks() = 0;
	virtual bool enableOwnerWaveBackend() = 0;
	virtual bool attachToCpuAvbdContext(
		AvbdDynamicsContext& cpuContext) = 0;
	virtual bool detachFromCpuAvbdContext(
		AvbdDynamicsContext& cpuContext) = 0;
};

// Optional synchronous backend hook for one prepared dependency-wave packet.
// The CPU task remains the owner of island ordering and calls this interface
// only after the normal prepared context exists. A backend call is
// transactional for the requested chunk: returning false means that chunk's
// body state was not committed and the caller must execute the scalar range.
class AvbdRigidGpuWaveBackend
{
public:
	virtual ~AvbdRigidGpuWaveBackend() {}
	virtual bool isAvailable() const = 0;
	virtual bool solveRigidOwnerWave(
		AvbdSolver& scalarSolver, AvbdRigidSolveContext& context,
		PxU32 waveIndex, PxU32 waveBodyOffset, PxU32 epoch,
		PxF32 avbdAlpha) = 0;
};

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_GPU_WAVE_BACKEND_H
