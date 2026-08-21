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
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.

#ifndef DY_AVBD_CPU_ISA_H
#define DY_AVBD_CPU_ISA_H

#include "foundation/PxAllocatorCallback.h"
#include "foundation/PxAssert.h"
#include "foundation/PxSimpleTypes.h"
#include "avbd/core/DyAvbdTypes.h"

namespace physx {
namespace Dy {

// ISA dispatch is initialized once per process. Production packet consumers
// read that immutable table; hot loops never query CPUID or environment state
// per particle or element.
typedef PxF32 (*AvbdCpuIsaProbeDot8Fn)(const PxF32* lhs, const PxF32* rhs);

static const PxU32 eAVBD_TET_MATERIAL_PACKET_WIDTH = 8;
static const PxU32 eAVBD_RIGID_LDLT_PACKET_WIDTH = 8;
static const PxU32 eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH = 8;
static const PxU32 eAVBD_RIGID_NORMAL_CONTACT_UPPER_COUNT = 21;
static const PxU32 eAVBD_RIGID_D6_RESPONSE_ROW_PACKET_COUNT = 6;
static const PxU32 eAVBD_RIGID_CONTACT_BLOCK_ROW_COUNT = 3;

// Lane-major rigid/static normal-contact producer record.  The producer
// supplies only a semantically uniform normal row; friction, joints, soft,
// mixed, singular, and tail rows stay on the scalar authority.  The AVX2+FMA
// entry uses explicit multiply/add operations for this reviewed numeric
// contract, while FMA remains a required ISA gate for the wide backend.
struct AvbdRigidNormalContactPacket8Input
{
	// Non-owning field-major pointers are intentional: preparation can bind
	// producer-owned contact SoA ranges directly, avoiding an AoS row packet
	// copy before the kernel. Each pointer addresses eight contiguous lanes.
	const PxF32* bodyPosition[3];
	const PxF32* bodyRotation[4];
	const PxF32* bodyContactPoint[3];
	const PxF32* staticContactPoint[3];
	const PxF32* normal[3];
	const PxF32* penetration;
	const PxF32* penalty;
	const PxF32* lambda;
	const PxF32* maxImpulse;
	const PxF32* dt;
	const PxF32* linearResponseScale;
	const PxF32* angularResponseScale;
	// +1 when the dynamic body is A, -1 when it is B. The packet applies this
	// to the dynamic-static geometric gap and Jacobian; penetration remains an
	// authored A-B offset and is never sign-flipped.
	const PxF32* sign;
	PxU8 activeMask;
	PxU8 padding[3];
};

struct AvbdRigidNormalContactPacket8Output
{
	PxF32 rhsLinear[3][eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH];
	PxF32 rhsAngular[3][eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH];
	PxF32 hessianUpper[eAVBD_RIGID_NORMAL_CONTACT_UPPER_COUNT]
		[eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH];
	PxU8 touchingMask;
	PxU8 hessianMask;
	PxU8 forceSaturatedMask;
	PxU8 padding;
};

// Direct body-local reduction target. The packet kernel may still use private
// register/scratch state, but its public result is accumulated into the
// caller-owned local block and RHS, so the solver does not perform a second
// 27-field sidecar traversal.
struct AvbdRigidNormalContactPacket8AccumulateTarget
{
	AvbdBlock6x6* hessian;
	PxVec3* rhsLinear;
	PxVec3* rhsAngular;
	PxU32* touchingCount;

	PX_FORCE_INLINE AvbdRigidNormalContactPacket8AccumulateTarget()
		: hessian(nullptr), rhsLinear(nullptr), rhsAngular(nullptr),
		  touchingCount(nullptr)
	{
	}
};

PX_FORCE_INLINE PxU32 avbdAccumulateNormalUpperIndex(PxU32 row,
	PxU32 column)
{
	return row * 6u - row * (row - 1u) / 2u + column - row;
}

PX_FORCE_INLINE void avbdAccumulateNormalBlockEntry(AvbdBlock6x6& hessian,
	PxU32 row, PxU32 column, PxF32 value)
{
	if(row < 3u)
	{
		if(column < 3u)
			hessian.linearLinear(row, column) += value;
		else
			hessian.linearAngular(row, column - 3u) += value;
	}
	else if(column < 3u)
		hessian.angularLinear(row - 3u, column) += value;
	else
		hessian.angularAngular(row - 3u, column - 3u) += value;
}

PX_FORCE_INLINE void avbdAccumulateNormalPacketOutput(
	const AvbdRigidNormalContactPacket8Output& output,
	PxU8 activeMask, AvbdRigidNormalContactPacket8AccumulateTarget& target)
{
	if(!target.hessian || !target.rhsLinear || !target.rhsAngular ||
		!target.touchingCount)
		return;
	for(PxU32 lane = 0; lane < eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH;
		++lane)
	{
		const PxU8 bit = PxU8(1u << lane);
		if((activeMask & bit) == 0 || (output.touchingMask & bit) == 0)
			continue;
		++(*target.touchingCount);
		target.rhsLinear->x += output.rhsLinear[0][lane];
		target.rhsLinear->y += output.rhsLinear[1][lane];
		target.rhsLinear->z += output.rhsLinear[2][lane];
		target.rhsAngular->x += output.rhsAngular[0][lane];
		target.rhsAngular->y += output.rhsAngular[1][lane];
		target.rhsAngular->z += output.rhsAngular[2][lane];
		if((output.hessianMask & bit) == 0)
			continue;
		for(PxU32 row = 0; row < 6u; ++row)
			for(PxU32 column = row; column < 6u; ++column)
			{
				const PxF32 value = output.hessianUpper[
					avbdAccumulateNormalUpperIndex(row, column)][lane];
				avbdAccumulateNormalBlockEntry(*target.hessian, row, column,
					value);
				if(row == column)
					continue;
				avbdAccumulateNormalBlockEntry(*target.hessian, column, row,
					value);
			}
	}
}

// Compact preparation-owned SoA for eligible rigid/static normal rows. It is
// intentionally limited to packet fields (96 bytes per row) rather than a
// second copy of AvbdContactConstraint. bindPacket8() copies only pointers and
// an active mask; it never copies row payload into the packet ABI.
struct AvbdRigidNormalContactSoA
{
	enum FloatField : PxU32
	{
		eBodyPositionX = 0, eBodyPositionY, eBodyPositionZ,
		eBodyRotationX, eBodyRotationY, eBodyRotationZ, eBodyRotationW,
		eBodyContactPointX, eBodyContactPointY, eBodyContactPointZ,
		eStaticContactPointX, eStaticContactPointY, eStaticContactPointZ,
		eNormalX, eNormalY, eNormalZ,
		ePenetration, ePenalty, eLambda, eMaxImpulse, eDt,
		eLinearResponseScale, eAngularResponseScale, eSign,
		eFloatFieldCount
	};

	PxF32* bodyPosition[3];
	PxF32* bodyRotation[4];
	PxF32* bodyContactPoint[3];
	PxF32* staticContactPoint[3];
	PxF32* normal[3];
	PxF32* penetration;
	PxF32* penalty;
	PxF32* lambda;
	PxF32* maxImpulse;
	PxF32* dt;
	PxF32* linearResponseScale;
	PxF32* angularResponseScale;
	PxF32* sign;
	PxF32* floatStorage;
	PxU32 numRows;
	PxU32 capacity;
	PxU8 ownsStorage;
	PxU8 padding[3];

	PX_FORCE_INLINE AvbdRigidNormalContactSoA()
		: penetration(nullptr), penalty(nullptr), lambda(nullptr),
		  maxImpulse(nullptr), dt(nullptr), linearResponseScale(nullptr),
		  angularResponseScale(nullptr), sign(nullptr), floatStorage(nullptr),
		  numRows(0), capacity(0), ownsStorage(0), padding{0, 0, 0}
	{
		clearPointers();
	}

	PX_FORCE_INLINE void clearPointers()
	{
		for(PxU32 i = 0; i < 3; ++i)
		{
			bodyPosition[i] = nullptr;
			bodyContactPoint[i] = nullptr;
			staticContactPoint[i] = nullptr;
			normal[i] = nullptr;
		}
		for(PxU32 i = 0; i < 4; ++i)
			bodyRotation[i] = nullptr;
		penetration = penalty = lambda = maxImpulse = dt = nullptr;
		linearResponseScale = angularResponseScale = sign = nullptr;
	}

	PX_FORCE_INLINE void bindFieldPointers(PxF32* storage, PxU32 count)
	{
		PxU32 field = 0;
		const auto nextField = [&]() { return storage + count * field++; };
		for(PxU32 i = 0; i < 3; ++i)
			bodyPosition[i] = nextField();
		for(PxU32 i = 0; i < 4; ++i)
			bodyRotation[i] = nextField();
		for(PxU32 i = 0; i < 3; ++i)
			bodyContactPoint[i] = nextField();
		for(PxU32 i = 0; i < 3; ++i)
			staticContactPoint[i] = nextField();
		for(PxU32 i = 0; i < 3; ++i)
			normal[i] = nextField();
		penetration = nextField();
		penalty = nextField();
		lambda = nextField();
		maxImpulse = nextField();
		dt = nextField();
		linearResponseScale = nextField();
		angularResponseScale = nextField();
		sign = nextField();
		PX_ASSERT(field == eFloatFieldCount);
	}

	inline void allocateData(PxU32 count, PxAllocatorCallback& allocator)
	{
		deallocateData(allocator);
		capacity = count;
		ownsStorage = 0;
		if(count == 0)
			return;
		const size_t maxFloatElements = size_t(-1) / sizeof(PxF32);
		if(size_t(count) > maxFloatElements / eFloatFieldCount)
		{
			capacity = 0;
			return;
		}
		floatStorage = static_cast<PxF32*>(allocator.allocate(
			size_t(sizeof(PxF32)) * count * eFloatFieldCount,
			"AvbdRigidNormalContactSoA::floatStorage", __FILE__, __LINE__));
		if(!floatStorage)
		{
			capacity = 0;
			return;
		}
		ownsStorage = 1;
		bindFieldPointers(floatStorage, count);
		numRows = 0;
	}

	// Bind one caller-owned field-major block. The owner is normally frame or
	// wave scratch storage; this object never deallocates the block. This is
	// the persistent producer seam required by P92 and is not a hot-loop copy.
	PX_FORCE_INLINE bool bindExternalStorage(PxF32* storage, PxU32 count,
		PxU32 storageCapacity)
	{
		if(!storage || count > storageCapacity || floatStorage)
			return false;
		floatStorage = storage;
		capacity = storageCapacity;
		numRows = count;
		ownsStorage = 0;
		bindFieldPointers(storage, storageCapacity);
		return true;
	}

	inline void deallocateData(PxAllocatorCallback& allocator)
	{
		if(floatStorage && ownsStorage)
			allocator.deallocate(floatStorage);
		floatStorage = nullptr;
		clearPointers();
		numRows = 0;
		capacity = 0;
		ownsStorage = 0;
	}

	inline bool bindPacket8(PxU32 base,
		AvbdRigidNormalContactPacket8Input& packet, PxU8 activeMask) const
	{
		if(!floatStorage || numRows > capacity || base > numRows ||
			8u > numRows - base)
			return false;
		for(PxU32 i = 0; i < 3; ++i)
		{
			packet.bodyPosition[i] = bodyPosition[i] + base;
			packet.bodyContactPoint[i] = bodyContactPoint[i] + base;
			packet.staticContactPoint[i] = staticContactPoint[i] + base;
			packet.normal[i] = normal[i] + base;
		}
		for(PxU32 i = 0; i < 4; ++i)
			packet.bodyRotation[i] = bodyRotation[i] + base;
		packet.penetration = penetration + base;
		packet.penalty = penalty + base;
		packet.lambda = lambda + base;
		packet.maxImpulse = maxImpulse + base;
		packet.dt = dt + base;
		packet.linearResponseScale = linearResponseScale + base;
		packet.angularResponseScale = angularResponseScale + base;
		packet.sign = sign + base;
		packet.activeMask = activeMask;
		packet.padding[0] = packet.padding[1] = packet.padding[2] = 0;
		return true;
	}
};

// Preparation-owned lane contract for a future normal-contact packet range.
// The SoA payload remains the only numeric storage; this descriptor carries
// the ownership facts that cannot be inferred from a field-major row alone.
// It is deliberately a cold descriptor and is not embedded in AvbdIslandBatch
// or queried by the solver hot loop.
struct AvbdRigidNormalContactProducerRange8
{
	PxU32 rowBase;
	PxU32 ownerBody;
	PxU32 dynamicTargetStorageIndex;
	PxU32 sourceConstraint[eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH];
	PxU32 dynamicTargetBody[eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH];
	PxU32 targetPoseStamp[eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH];
	PxU8 activeMask;
	PxU8 dynamicTargetMask;
	PxU8 scalarFallbackMask;
	PxU8 padding;

	PX_FORCE_INLINE AvbdRigidNormalContactProducerRange8()
		: rowBase(0), ownerBody(PX_MAX_U32),
		  dynamicTargetStorageIndex(PX_MAX_U32),
		  activeMask(0), dynamicTargetMask(0),
		  scalarFallbackMask(0), padding(0)
	{
		for(PxU32 lane = 0;
			lane < eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH; ++lane)
		{
			sourceConstraint[lane] = PX_MAX_U32;
			dynamicTargetBody[lane] = PX_MAX_U32;
			targetPoseStamp[lane] = 0;
		}
	}

	PX_FORCE_INLINE bool validate() const
	{
		const PxU8 laneMask = PxU8(
			(1u << eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH) - 1u);
		if((activeMask & ~laneMask) != 0 ||
			(dynamicTargetMask & ~activeMask) != 0 ||
			(scalarFallbackMask & ~activeMask) != 0)
			return false;
		for(PxU32 lane = 0;
			lane < eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH; ++lane)
		{
			const PxU8 bit = PxU8(1u << lane);
			if((activeMask & bit) == 0)
				continue;
			if(sourceConstraint[lane] == PX_MAX_U32)
				return false;
			if((dynamicTargetMask & bit) != 0 &&
				(dynamicTargetBody[lane] == PX_MAX_U32 ||
				 targetPoseStamp[lane] == 0))
				return false;
		}
		return true;
	}

};

// P96 dynamic-target endpoint seam.  The existing normal packet ABI carries a
// single world-space staticContactPoint; that is sufficient for rigid/static
// rows but forces a scalar pose-refresh for dynamic-dynamic rows.  This cold
// view supplies the other body's pose and local contact point only for lanes
// selected by dynamicMask.  It is intentionally separate from the packet
// input so the accepted P93 ABI and its lane/order contract remain unchanged.
struct AvbdRigidNormalContactDynamicTarget8
{
	const PxF32* position[3];
	const PxF32* rotation[4];
	const PxF32* contactPoint[3];
	PxU8 dynamicMask;
	PxU8 padding[3];

	PX_FORCE_INLINE AvbdRigidNormalContactDynamicTarget8()
		: dynamicMask(0), padding{0, 0, 0}
	{
		for(PxU32 component = 0; component < 3; ++component)
		{
			position[component] = nullptr;
			contactPoint[component] = nullptr;
		}
		for(PxU32 component = 0; component < 4; ++component)
			rotation[component] = nullptr;
	}

	PX_FORCE_INLINE bool validate(PxU8 activeMask) const
	{
		const PxU8 validMask = PxU8(
			(1u << eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH) - 1u);
		if((dynamicMask & ~activeMask) != 0 ||
			(dynamicMask & ~validMask) != 0)
			return false;
		if(dynamicMask == 0)
			return true;
		for(PxU32 component = 0; component < 3; ++component)
			if(!position[component] || !contactPoint[component])
				return false;
		for(PxU32 component = 0; component < 4; ++component)
			if(!rotation[component])
				return false;
		return true;
	}
};

// One field-major endpoint payload for a producer range. A dynamic target
// may differ in every lane; this compact cold record is the one-time pose
// gather that feeds the P97 transform without retaining an AoS body walk in
// the packet path.
struct AvbdRigidNormalContactDynamicTargetStorage8
{
	PxF32 position[3][eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH];
	PxF32 rotation[4][eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH];
	PxF32 contactPoint[3][eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH];

	PX_FORCE_INLINE AvbdRigidNormalContactDynamicTargetStorage8()
	{
		clear();
	}

	PX_FORCE_INLINE void clear()
	{
		for(PxU32 component = 0; component < 3; ++component)
		{
			for(PxU32 lane = 0;
				lane < eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH; ++lane)
			{
				position[component][lane] = 0.0f;
				contactPoint[component][lane] = 0.0f;
			}
		}
		for(PxU32 component = 0; component < 4; ++component)
			for(PxU32 lane = 0;
				lane < eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH; ++lane)
				rotation[component][lane] = component == 3u ? 1.0f : 0.0f;
	}

	PX_FORCE_INLINE AvbdRigidNormalContactDynamicTarget8 view(
		PxU8 dynamicMask) const
	{
		AvbdRigidNormalContactDynamicTarget8 result;
		for(PxU32 component = 0; component < 3; ++component)
		{
			result.position[component] = position[component];
			result.contactPoint[component] = contactPoint[component];
		}
		for(PxU32 component = 0; component < 4; ++component)
			result.rotation[component] = rotation[component];
		result.dynamicMask = dynamicMask;
		return result;
	}
};

// Cold storage descriptor for one rigid dependency-wave producer. Numeric SoA
// storage and range arrays are owned by the caller (usually frame scratch), so
// this descriptor is pointer-sized metadata and has no per-body allocation or
// solver-path copy. A wave producer updates only the fields whose live pose or
// dual value changed before binding a packet range.
struct AvbdRigidNormalContactWaveStorage
{
	AvbdRigidNormalContactSoA soa;
	AvbdRigidNormalContactProducerRange8* ranges;
	AvbdRigidNormalContactDynamicTargetStorage8* dynamicTargets;
	PxU32* bodyRangeStarts;
	PxU32* bodyRangeCounts;
	PxU32 rangeCapacity;
	PxU32 dynamicTargetCapacity;
	PxU32 bodyCapacity;
	PxU32 rangeCount;
	PxU32 waveStamp;

	PX_FORCE_INLINE AvbdRigidNormalContactWaveStorage()
		: ranges(nullptr), dynamicTargets(nullptr),
		  bodyRangeStarts(nullptr), bodyRangeCounts(nullptr), rangeCapacity(0),
		  dynamicTargetCapacity(0), bodyCapacity(0),
		  rangeCount(0), waveStamp(0)
	{
	}

	PX_FORCE_INLINE bool bind(PxF32* floatStorage, PxU32 rowCount,
		PxU32 rowCapacity, AvbdRigidNormalContactProducerRange8* rangeData,
		PxU32 maxRanges, PxU32* starts, PxU32* counts, PxU32 maxBodies)
	{
		if(!floatStorage || !rangeData || !starts || !counts ||
			!soa.bindExternalStorage(floatStorage, rowCount, rowCapacity))
			return false;
		ranges = rangeData;
		bodyRangeStarts = starts;
		bodyRangeCounts = counts;
		rangeCapacity = maxRanges;
		dynamicTargets = nullptr;
		dynamicTargetCapacity = 0;
		bodyCapacity = maxBodies;
		rangeCount = 0;
		waveStamp = 0;
		return true;
	}

	PX_FORCE_INLINE void bindDynamicTargetStorage(
		AvbdRigidNormalContactDynamicTargetStorage8* targetData,
		PxU32 maxTargets)
	{
		dynamicTargets = targetData;
		dynamicTargetCapacity = targetData ? maxTargets : 0;
	}

	PX_FORCE_INLINE bool validate(PxU32 numBodies) const
	{
		if(!ranges || !bodyRangeStarts || !bodyRangeCounts ||
			numBodies > bodyCapacity || rangeCount > rangeCapacity ||
			soa.numRows > soa.capacity)
			return false;
		for(PxU32 body = 0; body < numBodies; ++body)
		{
			const PxU32 start = bodyRangeStarts[body];
			const PxU32 count = bodyRangeCounts[body];
			if(start > rangeCount || count > rangeCount - start)
				return false;
			for(PxU32 index = start; index < start + count; ++index)
			{
				const AvbdRigidNormalContactProducerRange8& range = ranges[index];
				if(range.ownerBody != body || !range.validate() ||
					range.rowBase > soa.numRows ||
					8u > soa.numRows - range.rowBase ||
					((range.dynamicTargetMask != 0) &&
						(!dynamicTargets ||
						 range.dynamicTargetStorageIndex >= dynamicTargetCapacity)))
					return false;
			}
		}
		return true;
	}

	PX_FORCE_INLINE void beginWave(PxU32 stamp)
	{
		waveStamp = stamp;
	}
};

// Lane-major factor/RHS view for the rigid 6x6 LDLT back-substitution.
// The scalar decomposition remains authoritative; this packet only evaluates
// the fixed triangular solve after the factors are complete.  The field order
// is deliberately explicit so the AVX2 TU never depends on AvbdLDLT layout.
struct AvbdRigidLdltPacket8Input
{
	PxF32 linearLinear[3][3][eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 angularLinear[3][3][eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 angularAngular[3][3][eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 diagonalLinear[3][eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 diagonalAngular[3][eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 rhsLinear[3][eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 rhsAngular[3][eAVBD_RIGID_LDLT_PACKET_WIDTH];
};

struct AvbdRigidLdltPacket8Output
{
	PxF32 linear[3][eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 angular[3][eAVBD_RIGID_LDLT_PACKET_WIDTH];
};

// Lane-major raw local systems.  These fields are written at the matrix/RHS
// production boundary, before factorization.  The packet kernel performs the
// regularized LDLT factorization and triangular solve in one entry; a lane is
// invalid when the wide path cannot prove the scalar condition-number
// contract and must then use the scalar authoritative path.
struct AvbdRigidLdltFactorSolvePacket8Input
{
	PxF32 linearLinear[3][3][eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 angularLinear[3][3][eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 angularAngular[3][3][eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 rhsLinear[3][eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 rhsAngular[3][eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 regularizationCoefficient;
	PxF32 singularThreshold;
	PxF32 conditionNumberThreshold;
	PxU32 maxRegularizationAttempts;
};

// Field-major producer row for the rigid local-system packet.  The producer
// owns the arrays; the ISA kernel only accumulates into the packet's existing
// matrix/RHS storage and reports the touching bits.  Keeping this as a
// read-only input record avoids another eight-lane AoS row copy at the
// assembly boundary.
struct AvbdRigidLocalResponsePacket8Input
{
	PxF32 gradPos[3][eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 gradRot[3][eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 invCompliance[eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 linearScale[eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 angularScale[eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 force[eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxU8 activeMask;
	PxU8 touchingMask;
	PxU8 padding[2];
};

// Fused six-row D6 packet input. The row-major outer dimension is fixed to
// the locked-linear plus angular-drive subset used by the wide fixture; a
// producer may reject any other row family and retain the scalar authority.
struct AvbdRigidD6ResponsePacket8Input
{
	PxF32 gradPos[eAVBD_RIGID_D6_RESPONSE_ROW_PACKET_COUNT][3]
		[eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 gradRot[eAVBD_RIGID_D6_RESPONSE_ROW_PACKET_COUNT][3]
		[eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 invCompliance[eAVBD_RIGID_D6_RESPONSE_ROW_PACKET_COUNT]
		[eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 linearScale[eAVBD_RIGID_D6_RESPONSE_ROW_PACKET_COUNT]
		[eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 angularScale[eAVBD_RIGID_D6_RESPONSE_ROW_PACKET_COUNT]
		[eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 force[eAVBD_RIGID_D6_RESPONSE_ROW_PACKET_COUNT]
		[eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxU8 activeMask;
	PxU8 touchingMask;
	PxU8 padding[2];
};

// Complete dynamic-dynamic contact block contract.  The three rows are kept
// in scalar authority order (normal, tangent0, tangent1), while all lanes
// share the response scales and active mask.  The hessian mask allows a
// saturated unilateral normal row to contribute its RHS without adding a
// derivative.  This is a cold producer/kernel seam until a live producer can
// fill all three rows without an AoS sidecar or a second contact-map walk.
struct AvbdRigidContactBlockPacket8Input
{
	PxF32 gradPos[eAVBD_RIGID_CONTACT_BLOCK_ROW_COUNT][3]
		[eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 gradRot[eAVBD_RIGID_CONTACT_BLOCK_ROW_COUNT][3]
		[eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 invCompliance[eAVBD_RIGID_CONTACT_BLOCK_ROW_COUNT]
		[eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 force[eAVBD_RIGID_CONTACT_BLOCK_ROW_COUNT]
		[eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 linearScale[eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 angularScale[eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxU8 hessianMask[eAVBD_RIGID_CONTACT_BLOCK_ROW_COUNT];
	PxU8 activeMask;
	PxU8 touchingMask;
	PxU8 padding;
};

// Non-owning six-row view for a producer that already stores each field in
// eight contiguous lanes.  The view avoids copying/zeroing a 1.9-KiB fused
// input packet at the solver seam; ownership and lifetime remain entirely on
// the producer side.  It is a cold contract until a live producer proves that
// its source ranges remain stable for the complete packet call.
struct AvbdRigidD6ResponsePacket8View
{
	const PxF32* gradPos[eAVBD_RIGID_D6_RESPONSE_ROW_PACKET_COUNT][3];
	const PxF32* gradRot[eAVBD_RIGID_D6_RESPONSE_ROW_PACKET_COUNT][3];
	const PxF32* invCompliance[eAVBD_RIGID_D6_RESPONSE_ROW_PACKET_COUNT];
	const PxF32* linearScale[eAVBD_RIGID_D6_RESPONSE_ROW_PACKET_COUNT];
	const PxF32* angularScale[eAVBD_RIGID_D6_RESPONSE_ROW_PACKET_COUNT];
	const PxF32* force[eAVBD_RIGID_D6_RESPONSE_ROW_PACKET_COUNT];
	PxU8 activeMask;
	PxU8 touchingMask;
	PxU8 padding[2];

	// Bind six producer-owned one-row records without copying their field
	// arrays.  The caller must keep all row records alive and immutable for the
	// duration of the selected ISA call; the returned view contains no owning
	// storage and no solver state.
	PX_FORCE_INLINE bool bindRows(
		const AvbdRigidLocalResponsePacket8Input* rows,
		PxU8 producerActiveMask, PxU8 producerTouchingMask)
	{
		if(!rows)
			return false;
		for(PxU32 row = 0;
			row < eAVBD_RIGID_D6_RESPONSE_ROW_PACKET_COUNT; ++row)
		{
			for(PxU32 i = 0; i < 3; ++i)
			{
				gradPos[row][i] = rows[row].gradPos[i];
				gradRot[row][i] = rows[row].gradRot[i];
			}
			invCompliance[row] = rows[row].invCompliance;
			linearScale[row] = rows[row].linearScale;
			angularScale[row] = rows[row].angularScale;
			force[row] = rows[row].force;
		}
		activeMask = producerActiveMask;
		touchingMask = PxU8(producerTouchingMask & producerActiveMask);
		padding[0] = padding[1] = 0;
		return true;
	}
};

// P103 producer-native local-system block.  This is caller-owned cold/wave
// storage, not solver context: a producer writes the matrix and RHS directly
// into the same lane-major ABI consumed by the factor packet.  The body and
// touching metadata travels with the packet so the GPU dependency-wave
// consumer does not rescan the generic contact map or carry an AoS row copy.
// The scalar setter exists only for the isolated differential gate and for a
// correctness fallback; a production producer should fill factorInput fields
// while it accumulates its local system.
struct AvbdRigidLocalSystemAoSoA8
{
	AvbdRigidLdltFactorSolvePacket8Input factorInput;
	PxU32 bodyIndex[eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxU8 activeMask;
	PxU8 touchingMask;
	PxU8 padding[2];

	PX_FORCE_INLINE AvbdRigidLocalSystemAoSoA8()
		: activeMask(0), touchingMask(0), padding{0, 0}
	{
		clear();
	}

	PX_FORCE_INLINE void clear()
	{
		for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
			bodyIndex[lane] = PX_MAX_U32;
		factorInput.regularizationCoefficient =
			AvbdConstants::AVBD_REGULARIZATION_COEFFICIENT;
		factorInput.singularThreshold =
			AvbdConstants::AVBD_LDLT_SINGULAR_THRESHOLD;
		factorInput.conditionNumberThreshold =
			AvbdConstants::AVBD_CONDITION_NUMBER_THRESHOLD;
		factorInput.maxRegularizationAttempts = 3u;
		for(PxU32 i = 0; i < 3; ++i)
		{
			for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
			{
				factorInput.rhsLinear[i][lane] = 0.0f;
				factorInput.rhsAngular[i][lane] = 0.0f;
				for(PxU32 j = 0; j < 3; ++j)
				{
					factorInput.linearLinear[i][j][lane] = 0.0f;
					factorInput.angularLinear[i][j][lane] = 0.0f;
					factorInput.angularAngular[i][j][lane] = 0.0f;
				}
			}
		}
		activeMask = 0;
		touchingMask = 0;
		padding[0] = padding[1] = 0;
	}

	// Reset only the lanes that a producer is about to reuse.  This is the
	// lifecycle boundary for persistent packet storage: a wave producer can
	// keep one packet per worker and clear the selected lane columns without
	// reconstructing the complete matrix/RHS object or rewriting the immutable
	// factorization thresholds.  Untouched lanes remain available for scalar
	// fallback or a later packet fill.  The method is deliberately branch-local
	// and allocation-free; it carries no generation/diagnostic state.
	PX_FORCE_INLINE void resetLanes(PxU8 laneMask)
	{
		const PxU8 validMask = PxU8(
			(1u << eAVBD_RIGID_LDLT_PACKET_WIDTH) - 1u);
		laneMask &= validMask;
		if(laneMask == 0)
			return;
		for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
		{
			const PxU8 bit = PxU8(1u << lane);
			if((laneMask & bit) == 0)
				continue;
			bodyIndex[lane] = PX_MAX_U32;
			for(PxU32 i = 0; i < 3; ++i)
			{
				factorInput.rhsLinear[i][lane] = 0.0f;
				factorInput.rhsAngular[i][lane] = 0.0f;
				for(PxU32 j = 0; j < 3; ++j)
				{
					factorInput.linearLinear[i][j][lane] = 0.0f;
					factorInput.angularLinear[i][j][lane] = 0.0f;
					factorInput.angularAngular[i][j][lane] = 0.0f;
				}
			}
			activeMask &= PxU8(~bit);
			touchingMask &= PxU8(~bit);
		}
	}

	// Seed one producer-native inertial local system directly into the packet
	// fields.  This is deliberately limited to diagonal linear inertia and a
	// full angular inertia tensor; response rows can then be accumulated without
	// constructing an AvbdBlock6x6 or copying an AvbdVec6 through an AoS view.
	// The helper is a cold contract boundary until a live producer supplies the
	// same field-major data and preserves the scalar fallback for rejected lanes.
	PX_FORCE_INLINE bool seedInertialSoA(PxU32 lane, PxF32 linearDiagonal,
		const PxF32 angularDiagonal[3][3], const PxF32 rhsLinear[3],
		const PxF32 rhsAngular[3], PxU32 body, bool active)
	{
		if(lane >= eAVBD_RIGID_LDLT_PACKET_WIDTH || body == PX_MAX_U32)
			return false;
		bodyIndex[lane] = body;
		for(PxU32 i = 0; i < 3; ++i)
		{
			factorInput.rhsLinear[i][lane] = rhsLinear[i];
			factorInput.rhsAngular[i][lane] = rhsAngular[i];
			for(PxU32 j = 0; j < 3; ++j)
			{
				factorInput.linearLinear[i][j][lane] =
					i == j ? linearDiagonal : 0.0f;
				factorInput.angularLinear[i][j][lane] = 0.0f;
				factorInput.angularAngular[i][j][lane] = angularDiagonal[i][j];
			}
		}
		const PxU8 bit = PxU8(1u << lane);
		if(active)
			activeMask |= bit;
		else
			activeMask &= PxU8(~bit);
		touchingMask &= PxU8(~bit);
		return true;
	}

	// Batch form for a producer that already owns field-major inertial arrays.
	// Invalid/sentinel lanes are omitted from the returned mask and remain on
	// the scalar fallback; no temporary AvbdBlock6x6 objects are materialized.
	PX_FORCE_INLINE PxU8 seedInertialBatchSoA(PxU8 laneMask,
		const PxF32 linearDiagonal[eAVBD_RIGID_LDLT_PACKET_WIDTH],
		const PxF32 angularDiagonal[3][3][eAVBD_RIGID_LDLT_PACKET_WIDTH],
		const PxF32 rhsLinear[3][eAVBD_RIGID_LDLT_PACKET_WIDTH],
		const PxF32 rhsAngular[3][eAVBD_RIGID_LDLT_PACKET_WIDTH],
		const PxU32 body[eAVBD_RIGID_LDLT_PACKET_WIDTH])
	{
		PxU8 seeded = 0;
		for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
		{
			const PxU8 bit = PxU8(1u << lane);
			if((laneMask & bit) == 0 || body[lane] == PX_MAX_U32)
			{
				activeMask &= PxU8(~bit);
				touchingMask &= PxU8(~bit);
				continue;
			}
			bodyIndex[lane] = body[lane];
			for(PxU32 i = 0; i < 3; ++i)
			{
				factorInput.rhsLinear[i][lane] = rhsLinear[i][lane];
				factorInput.rhsAngular[i][lane] = rhsAngular[i][lane];
				for(PxU32 j = 0; j < 3; ++j)
				{
					factorInput.linearLinear[i][j][lane] =
						i == j ? linearDiagonal[lane] : 0.0f;
					factorInput.angularLinear[i][j][lane] = 0.0f;
					factorInput.angularAngular[i][j][lane] =
						angularDiagonal[i][j][lane];
				}
			}
			activeMask |= bit;
			touchingMask &= PxU8(~bit);
			seeded |= bit;
		}
		return seeded;
	}

	// Pack one scalar authoritative local system into a lane.  This is a cold
	// differential/fallback helper; the hot producer is expected to write the
	// field-major factorInput directly and therefore pays no AoS round-trip.
	PX_FORCE_INLINE bool setLaneFromScalar(PxU32 lane,
		const AvbdBlock6x6& hessian, const AvbdVec6& rhs,
		PxU32 body, bool touching)
	{
		if(lane >= eAVBD_RIGID_LDLT_PACKET_WIDTH || body == PX_MAX_U32)
			return false;
		bodyIndex[lane] = body;
		for(PxU32 i = 0; i < 3; ++i)
		{
			factorInput.rhsLinear[i][lane] = rhs.linear[i];
			factorInput.rhsAngular[i][lane] = rhs.angular[i];
			for(PxU32 j = 0; j < 3; ++j)
			{
				factorInput.linearLinear[i][j][lane] =
					hessian.linearLinear(i, j);
				factorInput.angularLinear[i][j][lane] =
					hessian.angularLinear(i, j);
				factorInput.angularAngular[i][j][lane] =
					hessian.angularAngular(i, j);
			}
		}
		activeMask |= PxU8(1u << lane);
		if(touching)
			touchingMask |= PxU8(1u << lane);
		else
			touchingMask &= PxU8(~(1u << lane));
		return true;
	}

	// Add one response-scaled rank-one row directly in field-major storage.
	// This is the P103 producer primitive: it mirrors
	// AvbdBlock6x6::addResponseScaledConstraintContribution() and keeps the
	// factor ABI's lower cross block (the upper block is its transpose).  It is
	// intentionally a cold contract helper until preparation can accumulate
	// complete local systems without a generic map walk.
	//
	// A saturated unilateral row has a zero Hessian derivative but still
	// contributes its clamped RHS force. Keep that case explicit so a producer
	// does not encode it through a zero-compliance sentinel.
	PX_FORCE_INLINE bool addRhsContribution(PxU32 lane,
		const PxVec3& gradPos, const PxVec3& gradRot, PxF32 linearScale,
		PxF32 angularScale, PxF32 force, bool touching)
	{
		if(lane >= eAVBD_RIGID_LDLT_PACKET_WIDTH ||
			(activeMask & PxU8(1u << lane)) == 0)
			return false;
		const PxF32 nonnegativeLinear = PxMax(0.0f, linearScale);
		const PxF32 nonnegativeAngular = PxMax(0.0f, angularScale);
		for(PxU32 i = 0; i < 3; ++i)
		{
			factorInput.rhsLinear[i][lane] +=
				gradPos[i] * (force * nonnegativeLinear);
			factorInput.rhsAngular[i][lane] +=
				gradRot[i] * (force * nonnegativeAngular);
		}
		if(touching)
			touchingMask |= PxU8(1u << lane);
		return true;
	}

	PX_FORCE_INLINE bool addResponseContribution(PxU32 lane,
		const PxVec3& gradPos, const PxVec3& gradRot, PxF32 invCompliance,
		PxF32 linearScale, PxF32 angularScale, PxF32 force, bool touching)
	{
		if(lane >= eAVBD_RIGID_LDLT_PACKET_WIDTH ||
			(activeMask & PxU8(1u << lane)) == 0)
			return false;
		const PxF32 nonnegativeLinear = PxMax(0.0f, linearScale);
		const PxF32 nonnegativeAngular = PxMax(0.0f, angularScale);
		const PxF32 crossScale = PxSqrt(
			nonnegativeLinear * nonnegativeAngular);
		for(PxU32 i = 0; i < 3; ++i)
		{
			for(PxU32 j = 0; j < 3; ++j)
			{
				factorInput.linearLinear[i][j][lane] +=
					invCompliance * nonnegativeLinear * gradPos[i] * gradPos[j];
				factorInput.angularLinear[i][j][lane] +=
					invCompliance * crossScale * gradRot[i] * gradPos[j];
				factorInput.angularAngular[i][j][lane] +=
					invCompliance * nonnegativeAngular * gradRot[i] * gradRot[j];
			}
			factorInput.rhsLinear[i][lane] +=
				gradPos[i] * (force * nonnegativeLinear);
			factorInput.rhsAngular[i][lane] +=
				gradRot[i] * (force * nonnegativeAngular);
		}
		if(touching)
			touchingMask |= PxU8(1u << lane);
		return true;
	}

	// Producer-facing SoA row adapter.  The caller supplies field-major row
	// data and a lane mask; accumulation writes directly into factorInput and
	// never materializes an AoS local-system copy.  This is intentionally a
	// narrow assembly boundary, not a solver task or a SIMD claim: the caller
	// still owns lane eligibility and scalar fallback for cleared lanes.
	PX_FORCE_INLINE PxU8 addResponseContributionSoA(
		PxU8 laneMask, const PxF32 gradPos[3][eAVBD_RIGID_LDLT_PACKET_WIDTH],
		const PxF32 gradRot[3][eAVBD_RIGID_LDLT_PACKET_WIDTH],
		const PxF32 invCompliance[eAVBD_RIGID_LDLT_PACKET_WIDTH],
		const PxF32 linearScale[eAVBD_RIGID_LDLT_PACKET_WIDTH],
		const PxF32 angularScale[eAVBD_RIGID_LDLT_PACKET_WIDTH],
		const PxF32 force[eAVBD_RIGID_LDLT_PACKET_WIDTH], bool touching)
	{
		const PxU8 eligible = PxU8(laneMask & activeMask);
		for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
		{
			if((eligible & PxU8(1u << lane)) == 0)
				continue;
			const PxF32 nonnegativeLinear = PxMax(0.0f, linearScale[lane]);
			const PxF32 nonnegativeAngular = PxMax(0.0f, angularScale[lane]);
			const PxF32 crossScale = PxSqrt(
				nonnegativeLinear * nonnegativeAngular);
			for(PxU32 i = 0; i < 3; ++i)
			{
				for(PxU32 j = 0; j < 3; ++j)
				{
					factorInput.linearLinear[i][j][lane] +=
						invCompliance[lane] * nonnegativeLinear *
						gradPos[i][lane] * gradPos[j][lane];
					factorInput.angularLinear[i][j][lane] +=
						invCompliance[lane] * crossScale *
						gradRot[i][lane] * gradPos[j][lane];
					factorInput.angularAngular[i][j][lane] +=
						invCompliance[lane] * nonnegativeAngular *
						gradRot[i][lane] * gradRot[j][lane];
				}
				factorInput.rhsLinear[i][lane] +=
					gradPos[i][lane] * (force[lane] * nonnegativeLinear);
				factorInput.rhsAngular[i][lane] +=
					gradRot[i][lane] * (force[lane] * nonnegativeAngular);
			}
			if(touching)
				touchingMask |= PxU8(1u << lane);
		}
		return eligible;
	}

	PX_FORCE_INLINE bool validate() const
	{
		const PxU8 validMask = PxU8(
			(1u << eAVBD_RIGID_LDLT_PACKET_WIDTH) - 1u);
		if((activeMask & ~validMask) != 0 ||
			(touchingMask & ~activeMask) != 0)
			return false;
		for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
			if((activeMask & PxU8(1u << lane)) != 0 &&
				bodyIndex[lane] == PX_MAX_U32)
				return false;
		return true;
	}

	// Cold host bridge to the device-neutral owner-wave packet contract.  The
	// destination is intentionally a template so the CPU producer does not
	// depend on the GPU target; any destination with the accepted `desc`, body
	// and field-major members can consume this exact copy.  No AoS local system
	// or per-lane temporary is materialized.
	template <typename DestinationPacket>
	PX_FORCE_INLINE bool exportOwnerWavePacket(
		DestinationPacket& destination, PxU32 waveEpoch, PxU32 bodyCount,
		PxU32 contactCount, PxF32 dt, PxF32 invDt2, PxF32 avbdAlpha) const
	{
		if(!validate() || waveEpoch == 0 || bodyCount == 0 ||
			!(dt > 0.0f) || !(invDt2 > 0.0f))
			return false;

		PxU32 ownerCount = 0;
		for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
			ownerCount += (activeMask & PxU8(1u << lane)) != 0 ? 1u : 0u;

		destination.desc.waveEpoch = waveEpoch;
		destination.desc.ownerCount = ownerCount;
		destination.desc.bodyCount = bodyCount;
		destination.desc.contactCount = contactCount;
		destination.desc.dt = dt;
		destination.desc.invDt2 = invDt2;
		destination.desc.avbdAlpha = avbdAlpha;
		destination.desc.regularizationCoefficient =
			factorInput.regularizationCoefficient;
		destination.desc.singularThreshold = factorInput.singularThreshold;
		destination.desc.conditionNumberThreshold =
			factorInput.conditionNumberThreshold;
		destination.desc.maxRegularizationAttempts =
			factorInput.maxRegularizationAttempts;
		destination.desc.activeMask = activeMask;
		destination.desc.touchingMask = touchingMask;
		destination.desc.padding[0] = destination.desc.padding[1] = 0;

		for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
		{
			destination.ownerBodyIndex[lane] = bodyIndex[lane];
			for(PxU32 i = 0; i < 3; ++i)
			{
				destination.rhsLinear[i][lane] = factorInput.rhsLinear[i][lane];
				destination.rhsAngular[i][lane] = factorInput.rhsAngular[i][lane];
				for(PxU32 j = 0; j < 3; ++j)
				{
					destination.linearLinear[i][j][lane] =
						factorInput.linearLinear[i][j][lane];
					destination.angularLinear[i][j][lane] =
						factorInput.angularLinear[i][j][lane];
					destination.angularAngular[i][j][lane] =
						factorInput.angularAngular[i][j][lane];
				}
			}
		}
		return true;
	}
};

struct AvbdRigidLdltFactorSolvePacket8Output
{
	PxF32 linear[3][eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxF32 angular[3][eAVBD_RIGID_LDLT_PACKET_WIDTH];
	PxU8 validMask;
	PxU8 padding[3];
};

// Cold writeback/fallback boundary for a producer-owned local-system packet.
// A cleared output bit is never written to a body: the caller must route that
// lane through the scalar authoritative solve instead.  Keeping this helper
// outside the packet avoids adding any output or scratch fields to the hot
// producer storage while making body-index ownership explicit.
PX_FORCE_INLINE bool avbdExtractRigidLdltLaneSolution(
	const AvbdRigidLocalSystemAoSoA8& packet,
	const AvbdRigidLdltFactorSolvePacket8Output& output,
	PxU32 lane, PxU32& bodyIndex, AvbdVec6& solution)
{
	if(lane >= eAVBD_RIGID_LDLT_PACKET_WIDTH ||
		(packet.activeMask & PxU8(1u << lane)) == 0 ||
		(output.validMask & PxU8(1u << lane)) == 0 ||
		packet.bodyIndex[lane] == PX_MAX_U32)
		return false;
	bodyIndex = packet.bodyIndex[lane];
	for(PxU32 i = 0; i < 3; ++i)
	{
		solution.linear[i] = output.linear[i][lane];
		solution.angular[i] = output.angular[i][lane];
	}
	return true;
}

// SIMD material packets keep the component's soft-particle and tet-element
// AoS layout authoritative.  The caller gathers one canonical incidence
// packet into this transient SoA input; the isolated AVX2+FMA TU never
// depends on the component's private C++ layout.
struct AvbdTetMaterialPacket8Input
{
	PxF32 e1X[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 e1Y[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 e1Z[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 e2X[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 e2Y[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 e2Z[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 e3X[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 e3Y[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 e3Z[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 dm0X[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 dm0Y[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 dm0Z[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 dm1X[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 dm1Y[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 dm1Z[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 dm2X[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 dm2Y[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 dm2Z[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 shapeX[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 shapeY[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 shapeZ[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 shapeNormSq[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 restVolume[eAVBD_TET_MATERIAL_PACKET_WIDTH];
};

struct AvbdTetMaterialPacket8Output
{
	PxF32 forceX[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 forceY[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 forceZ[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 hessianXX[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 hessianYY[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 hessianZZ[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 hessianXY[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 hessianXZ[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 hessianYZ[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 determinant[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 determinantGradientX[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 determinantGradientY[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxF32 determinantGradientZ[eAVBD_TET_MATERIAL_PACKET_WIDTH];
	PxU8 validMask;
	PxU8 padding[3];
};

typedef void (*AvbdCpuIsaCorotationalTetPacket8Fn)(
	const AvbdTetMaterialPacket8Input& input,
	PxF32 mu, PxF32 lambda,
	AvbdTetMaterialPacket8Output& output);

typedef void (*AvbdCpuIsaNeoHookeanTetPacket8Fn)(
	const AvbdTetMaterialPacket8Input& input,
	PxF32 mu, PxF32 lambda, PxF32 alpha,
	AvbdTetMaterialPacket8Output& output);

struct AvbdTetMaterialPacketKernels
{
	AvbdCpuIsaCorotationalTetPacket8Fn corotational;
	AvbdCpuIsaNeoHookeanTetPacket8Fn neoHookean;

	PX_FORCE_INLINE bool hasAny() const
	{
		return corotational || neoHookean;
	}
};

typedef void (*AvbdCpuIsaRigidLdltPacket8Fn)(
	const AvbdRigidLdltPacket8Input& input,
	AvbdRigidLdltPacket8Output& output);

typedef void (*AvbdCpuIsaRigidLdltFactorSolvePacket8Fn)(
	const AvbdRigidLdltFactorSolvePacket8Input& input,
	AvbdRigidLdltFactorSolvePacket8Output& output);

typedef void (*AvbdCpuIsaRigidLocalResponsePacket8Fn)(
	const AvbdRigidLocalResponsePacket8Input& input,
	AvbdRigidLdltFactorSolvePacket8Input& target, PxU8& touchingMask);

typedef void (*AvbdCpuIsaRigidD6ResponsePacket8Fn)(
	const AvbdRigidD6ResponsePacket8Input& input,
	AvbdRigidLdltFactorSolvePacket8Input& target, PxU8& touchingMask);

typedef void (*AvbdCpuIsaRigidD6ResponsePacket8ViewFn)(
	const AvbdRigidD6ResponsePacket8View& input,
	AvbdRigidLdltFactorSolvePacket8Input& target, PxU8& touchingMask);

typedef void (*AvbdCpuIsaRigidContactBlockPacket8Fn)(
	const AvbdRigidContactBlockPacket8Input& input,
	AvbdRigidLdltFactorSolvePacket8Input& target, PxU8& touchingMask);

typedef void (*AvbdCpuIsaRigidNormalContactPacket8Fn)(
	const AvbdRigidNormalContactPacket8Input& input,
	AvbdRigidNormalContactPacket8Output& output);

typedef void (*AvbdCpuIsaRigidNormalContactPacket8AccumulateFn)(
	const AvbdRigidNormalContactPacket8Input& input,
	PxU8 activeMask,
	AvbdRigidNormalContactPacket8AccumulateTarget& target);

typedef void (*AvbdCpuIsaRigidNormalContactDynamicEndpoint8Fn)(
	const AvbdRigidNormalContactDynamicTarget8& input,
	PxF32* worldPositionX, PxF32* worldPositionY, PxF32* worldPositionZ);

struct AvbdCpuIsaFunctionTable {
	AvbdCpuIsaProbeDot8Fn probeDot8;
	// Null on the scalar SSE2 backend.  The AVX2+FMA entry evaluates eight
	// independent co-rotational tet incidences; the caller performs canonical
	// lane-order reduction and scalar fallback for every cleared valid bit.
	AvbdCpuIsaCorotationalTetPacket8Fn corotationalTetPacket8;
	// Null on the scalar SSE2 backend.  The material-neutral packet input and
	// output are shared with the co-rotational evaluator, while this entry
	// preserves Neo-Hookean's distinct constitutive law and alpha parameter.
	AvbdCpuIsaNeoHookeanTetPacket8Fn neoHookeanTetPacket8;
};

struct AvbdCpuIsaCapabilities {
	bool sse2;
	bool avx;
	bool osxsave;
	bool xmmYmmState;
	bool avx2;
	bool fma;

	PX_FORCE_INLINE bool hasAvx2FmaBackendSupport() const
	{
		return avx && osxsave && xmmYmmState && avx2 && fma;
	}
};

struct AvbdCpuIsaDispatch {
	const char* requestedIsa;
	const char* selectedIsa;
	const char* compiledIsaBackends;
	AvbdCpuIsaCapabilities capabilities;
	bool avx2FmaBackendCompiled;
	bool forceModeRejected;
	bool kernelSelfTestPassed;
	bool fmaUsed;
	PxF32 kernelSelfTestValue;
};

// Reads PX_AVBD_CPU_ISA=auto|sse2|avx2fma once.  A forced avx2fma request
// that is not executable always falls back to SSE2 and sets forceModeRejected;
// it never attempts to execute AVX/FMA instructions speculatively. The
// test-only PX_AVBD_CPU_ISA_TEST_DISABLE_AVX2_FMA=1 masks the wide capability
// before selection so a capable development machine can prove that fallback.
const AvbdCpuIsaDispatch& getAvbdCpuIsaDispatch();

// Returns the same once-selected table for the process lifetime. P6.1's
// probeDot8 remains an isolated dispatch self-test. The only production
// solver entry is the admitted corotational-tet packet kernel; rejected rigid
// kernels have a test-owned dispatch table under tools/.
const AvbdCpuIsaFunctionTable& getAvbdCpuIsaFunctionTable();

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CPU_ISA_H
