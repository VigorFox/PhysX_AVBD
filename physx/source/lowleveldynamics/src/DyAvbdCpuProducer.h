// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.

#ifndef DY_AVBD_CPU_PRODUCER_H
#define DY_AVBD_CPU_PRODUCER_H

#include "DyAvbdConstraint.h"
#include "DyAvbdCpuIsa.h"
#include "DyAvbdSolverBody.h"

namespace physx {
namespace Dy {

#if !defined(PX_AVBD_EXCLUDE_EXPERIMENTAL_RIGID_SIMD)

// Cold preparation input for one eight-lane dynamic-dynamic normal range.
// The caller owns all storage and decides the range's location in the wave;
// this record contains no solver/task state and is never queried per body.
struct AvbdRigidNormalContactDynamicRangeInput8
{
	const AvbdSolverBody* bodies;
	PxU32 numBodies;
	const AvbdContactConstraint* contacts;
	PxU32 numContacts;
	const PxU32* sourceConstraints;
	PxU32 ownerBody;
	PxU32 rowBase;
	PxU32 dynamicTargetStorageIndex;
	PxU8 activeMask;
	PxU8 padding[3];
	PxF32 dt;
	PxF32 contactBoostFloor;
	PxU32 poseStamp;
	AvbdRigidNormalContactSoA* soa;
	AvbdRigidNormalContactDynamicTargetStorage8* dynamicTargetStorage;
	AvbdCpuIsaRigidNormalContactDynamicEndpoint8Fn endpointKernel;

	PX_FORCE_INLINE AvbdRigidNormalContactDynamicRangeInput8()
		: bodies(nullptr), numBodies(0), contacts(nullptr), numContacts(0),
		  sourceConstraints(nullptr),
		  ownerBody(PX_MAX_U32), rowBase(0),
		  dynamicTargetStorageIndex(PX_MAX_U32), activeMask(0), padding{0, 0, 0},
		  dt(0.0f), contactBoostFloor(0.0f), poseStamp(0), soa(nullptr),
		  dynamicTargetStorage(nullptr), endpointKernel(nullptr)
	{
	}
};

// Classify one contact without writing producer storage. This cold predicate
// lets a wave compact eligible references before forming complete eight-lane
// packets, avoiding the density loss from mixed fallback rows.
bool avbdClassifyRigidNormalContactDynamicLane(
	const AvbdContactConstraint& contact, PxU32 ownerBody, PxU32 numBodies,
	PxF32& linearScale, PxF32& angularScale, bool& ownerIsA,
	PxU32& targetBody);

// Caller-owned upstream range builder. It compacts the immutable contact map
// into homogeneous eight-lane dynamic-normal ranges and fills the persistent
// field-major SoA through avbdPrepareRigidNormalContactDynamicRange8(). No
// allocation, solver/task state, or global cursor is touched; on any capacity
// or validation failure the caller must discard the storage and use scalar
// preparation for the island.
struct AvbdRigidNormalContactWaveBuildInput
{
	const AvbdSolverBody* bodies;
	PxU32 numBodies;
	const AvbdContactConstraint* contacts;
	PxU32 numContacts;
	const AvbdBodyConstraintMap* contactMap;
	PxF32 dt;
	PxF32 invDt2;
	PxU32 poseStamp;
	AvbdRigidNormalContactWaveStorage* storage;
	AvbdCpuIsaRigidNormalContactDynamicEndpoint8Fn endpointKernel;

	PX_FORCE_INLINE AvbdRigidNormalContactWaveBuildInput()
		: bodies(nullptr), numBodies(0), contacts(nullptr), numContacts(0),
		  contactMap(nullptr), dt(0.0f), invDt2(0.0f), poseStamp(0),
		  storage(nullptr), endpointKernel(nullptr)
	{
	}
};

bool avbdBuildRigidNormalContactWave(
	const AvbdRigidNormalContactWaveBuildInput& input);

// Prepare one dynamic-dynamic normal range.  Eligible lanes are fully
// populated into the persistent SoA and target payload, then P97 transforms
// the target endpoint directly into staticContactPoint.  Ineligible lanes are
// reported in scalarFallbackMask and remain for the authoritative scalar
// producer; no partial packet lane is exposed to the solver.
PxU8 avbdPrepareRigidNormalContactDynamicRange8(
	const AvbdRigidNormalContactDynamicRangeInput8& input,
	AvbdRigidNormalContactProducerRange8& range);

// Producer-owned cold input for the homogeneous wide-D6 subset used by the
// opt-in joint fixture. Each active lane names one dynamic body and one
// prepared D6 constraint. Unsupported lanes are cleared from the returned
// mask and must remain on the scalar authority.
struct AvbdRigidD6WideRangeInput8
{
	const AvbdSolverBody* bodies;
	PxU32 numBodies;
	const AvbdD6JointConstraint* joints;
	PxU32 numJoints;
	const PxU32* ownerBodies;
	const PxU32* jointIndices;
	PxU8 activeMask;
	PxU8 padding[3];
	PxF32 dt;
	PxF32 invDt2;

	PX_FORCE_INLINE AvbdRigidD6WideRangeInput8()
		: bodies(nullptr), numBodies(0), joints(nullptr), numJoints(0),
		  ownerBodies(nullptr), jointIndices(nullptr), activeMask(0),
		  padding{0, 0, 0}, dt(0.0f), invDt2(0.0f)
	{
	}
};

// Prepare locked-linear plus ordinary SLERP-velocity rows from actual D6
// constraint/body records. The caller clears the packet once before first
// reuse and preserves packet/row storage through the selected ISA call. This
// cold producer changes no solver or task state.
PxU8 avbdPrepareRigidD6WideRange8(
	const AvbdRigidD6WideRangeInput8& input,
	AvbdRigidLocalSystemAoSoA8& target,
	AvbdRigidLocalResponsePacket8Input* rows,
	AvbdRigidD6ResponsePacket8View& view);

#endif // !PX_AVBD_EXCLUDE_EXPERIMENTAL_RIGID_SIMD

// P200 owner-major dependency-wave producer contract.  One lane represents
// one complete owner local system; the producer walks that owner's contact-map
// range in scalar-authority order and writes the inertial seed plus every
// supported dynamic-dynamic normal/tangent row directly into factorInput.
// The live GPU bridge uses this scalar producer to form its device-neutral
// packet. A lane with any unsupported contact is cleared in its entirety and
// remains on the scalar authority; no partial local system may be consumed by
// a packet factor/solve call.
struct AvbdRigidOwnerMajorWaveInput8
{
	const AvbdSolverBody* bodies;
	PxU32 numBodies;
	const AvbdContactConstraint* contacts;
	PxU32 numContacts;
	const AvbdBodyConstraintMap* contactMap;
	const PxU32* ownerBodies;
	PxU8 activeMask;
	PxU8 padding[3];
	PxF32 dt;
	PxF32 invDt2;
	PxF32 avbdAlpha;

#if !defined(PX_AVBD_EXCLUDE_EXPERIMENTAL_RIGID_SIMD)
	// Optional ISA-owned row assembly. Null keeps the scalar producer as the
	// authority. This hook exists only in standalone rejected-candidate probes;
	// the production owner-wave contract deliberately does not expose it.
	AvbdCpuIsaRigidContactBlockPacket8Fn contactBlockKernel;
#endif

	PX_FORCE_INLINE AvbdRigidOwnerMajorWaveInput8()
		: bodies(nullptr), numBodies(0), contacts(nullptr), numContacts(0),
		  contactMap(nullptr), ownerBodies(nullptr), activeMask(0),
		  padding{0, 0, 0}, dt(0.0f), invDt2(0.0f), avbdAlpha(0.0f)
#if !defined(PX_AVBD_EXCLUDE_EXPERIMENTAL_RIGID_SIMD)
		, contactBlockKernel(nullptr)
#endif
	{
	}
};

// Returns the lanes that contain a complete owner local system.  Invalid
// global input returns zero and leaves the caller on the scalar path.
PxU8 avbdPrepareRigidOwnerMajorWave8(
	const AvbdRigidOwnerMajorWaveInput8& input,
	AvbdRigidLocalSystemAoSoA8& target);

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CPU_PRODUCER_H
