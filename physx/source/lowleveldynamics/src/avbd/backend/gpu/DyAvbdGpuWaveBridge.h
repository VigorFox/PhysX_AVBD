// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.

#ifndef DY_AVBD_GPU_WAVE_BRIDGE_H
#define DY_AVBD_GPU_WAVE_BRIDGE_H

#include "avbd/backend/cpu/DyAvbdCpuProducer.h"
#include "avbd/solver/DyAvbdSolver.h"

namespace physx
{
namespace Dy
{

// P210-A: consume the exact owner order published by a prepared CPU island
// and build one device-neutral packet without changing the scalar authority.
// The caller may pass a non-zero waveBodyOffset to split a wide dependency
// wave into fixed-width packets.  Any malformed context or unsupported lane
// returns false; the caller must then keep that chunk on the scalar path.
template <typename DestinationPacket>
PX_FORCE_INLINE bool avbdBuildRigidOwnerWavePacket(
    const AvbdRigidSolveContext& context, PxU32 waveIndex,
    PxU32 waveBodyOffset, PxU32 waveEpoch, PxF32 avbdAlpha,
    DestinationPacket& destination, AvbdRigidLocalSystemAoSoA8& scratch)
{
	const AvbdRigidSolveIterationState& state = context.iteration;
	if(!state.bodies || state.numBodies == 0 || !state.contacts ||
		!state.contactMap || !state.contactMap->constraintOffsets ||
		!state.contactMap->constraintCounts ||
		!state.contactMap->constraintIndices ||
		state.contactMap->numBodies != state.numBodies ||
		!state.dt || !(context.invDt2 > 0.0f) || !(avbdAlpha >= 0.0f) ||
		waveEpoch == 0 || waveIndex >= context.dependencyWaveCount ||
		context.dependencyWaveOffsets.size() <
			context.dependencyWaveCount + 1u)
		return false;

	const PxU32 waveBegin = context.dependencyWaveOffsets[waveIndex];
	const PxU32 waveEnd = context.dependencyWaveOffsets[waveIndex + 1u];
	if(waveBegin > waveEnd || waveEnd > context.dependencyWaveBodies.size() ||
		waveBodyOffset >= waveEnd - waveBegin)
		return false;

	const PxU32 remaining = waveEnd - waveBegin - waveBodyOffset;
	const PxU32 bodyCount = remaining < eAVBD_RIGID_LDLT_PACKET_WIDTH
		? remaining : eAVBD_RIGID_LDLT_PACKET_WIDTH;
	if(bodyCount == 0)
		return false;

	const PxU32* ownerBodies =
		context.dependencyWaveBodies.begin() + waveBegin + waveBodyOffset;
	PxU8 activeMask = PxU8((PxU32(1u) << bodyCount) - 1u);
	PxU32 contactRefCount = 0;
	for(PxU32 lane = 0; lane < bodyCount; ++lane)
	{
		const PxU32 owner = ownerBodies[lane];
		if(owner >= state.numBodies)
			return false;
		const PxU32 offset = state.contactMap->constraintOffsets[owner];
		const PxU32 count = state.contactMap->constraintCounts[owner];
		if(offset > state.contactMap->totalConstraintRefs || count >
			state.contactMap->totalConstraintRefs - offset)
			return false;
		for(PxU32 ref = 0; ref < count; ++ref)
			if(state.contactMap->constraintIndices[offset + ref] >=
				state.numContacts)
				return false;
		contactRefCount += count;
	}

	AvbdRigidOwnerMajorWaveInput8 input;
	input.bodies = state.bodies;
	input.numBodies = state.numBodies;
	input.contacts = state.contacts;
	input.numContacts = state.numContacts;
	input.contactMap = state.contactMap;
	input.ownerBodies = ownerBodies;
	input.activeMask = activeMask;
	input.dt = state.dt;
	input.invDt2 = context.invDt2;
	input.avbdAlpha = avbdAlpha;

	const PxU8 accepted = avbdPrepareRigidOwnerMajorWave8(input, scratch);
	// The scalar authority snaps a body with no touching primal row directly to
	// its inertial pose.  A mass-only LDLT solve is algebraically equivalent but
	// leaves a small rounding residual that accumulates over iterations.  Keep
	// the packet contract exact by routing any mixed/no-touching chunk through
	// the scalar owner range instead of encoding a subtly different update.
	if(accepted != activeMask ||
		(scratch.touchingMask & activeMask) != activeMask)
		return false;
	return scratch.exportOwnerWavePacket(destination, waveEpoch, bodyCount,
		contactRefCount, state.dt, context.invDt2, avbdAlpha);
}

// Apply only lanes reported valid by the device. Invalid active lanes remain
// untouched so the caller can execute their scalar fallback before committing
// the enclosing batch. This helper owns the exact AVBD pose update used by CPU
// writeback; velocity reconstruction remains in the common post-solve stage.
template <typename Packet, typename Solution>
PX_FORCE_INLINE bool avbdApplyRigidOwnerWaveSolution(
	AvbdSolverBody* bodies, PxU32 numBodies, const Packet& packet,
	const Solution& solution)
{
	if(!bodies || numBodies == 0)
		return false;
	const PxU8 activeMask = packet.desc.activeMask;
	// Validate every writeback target before applying the first device lane.
	// The caller may otherwise replay the scalar wave after a false return.
	for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
	{
		const PxU8 bit = PxU8(1u << lane);
		if((activeMask & bit) != 0u && solution.validLane[lane] != 0u &&
			packet.ownerBodyIndex[lane] >= numBodies)
			return false;
	}
	for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
	{
		const PxU8 bit = PxU8(1u << lane);
		if((activeMask & bit) == 0 || solution.validLane[lane] == 0)
			continue;
		const PxU32 owner = packet.ownerBodyIndex[lane];
		AvbdSolverBody& body = bodies[owner];
		body.position -= PxVec3(solution.linear[0][lane],
			solution.linear[1][lane], solution.linear[2][lane]);
		const PxVec3 deltaTheta(solution.angular[0][lane],
			solution.angular[1][lane], solution.angular[2][lane]);
		if(deltaTheta.magnitudeSquared() > 1.0e-12f)
		{
			const PxQuat dq(deltaTheta.x, deltaTheta.y, deltaTheta.z, 0.0f);
			body.rotation =
				(body.rotation - dq * body.rotation * 0.5f).getNormalized();
		}
	}
	return true;
}

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_GPU_WAVE_BRIDGE_H
