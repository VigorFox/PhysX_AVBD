// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.

#ifndef DY_AVBD_OGC_PAIR_H
#define DY_AVBD_OGC_PAIR_H

#include "foundation/PxMathUtils.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxTransform.h"
#include "foundation/PxVec3.h"

namespace physx
{
namespace Dy
{

// Stable identity assigned during contact preparation. Solver state follows
// this physical source objective, not whichever row is closest after rebuild.
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

enum class AvbdSoftContactTargetKind : PxU8
{
	eWORLD_STATIC,
	eKINEMATIC_RIGID,
	eDEFORMABLE_SURFACE,
	eRIGID_BODY,
	eUNSUPPORTED
};

// Tangential contact response can have a different owner from the geometric
// normal. This is deliberately separate from the normal objective owner.
enum class AvbdSoftContactTangentOwner : PxU8
{
	ePOSITION_AL,
	eVELOCITY
};

struct AvbdOgcPairKey
{
	AvbdSoftContactSource::Type sourceType;
	AvbdSoftContactTargetKind targetKind;
	PxU32 sourceBodyIndex;
	PxU32 targetBodyIndex;
	PxU64 primitiveKey;

	AvbdOgcPairKey()
		: sourceType(AvbdSoftContactSource::eINVALID),
		  targetKind(AvbdSoftContactTargetKind::eUNSUPPORTED),
		  sourceBodyIndex(PX_MAX_U32), targetBodyIndex(PX_MAX_U32),
		  primitiveKey(~PxU64(0))
	{
	}

	PX_FORCE_INLINE bool matches(
		AvbdSoftContactSource::Type sourceType_,
		AvbdSoftContactTargetKind targetKind_,
		PxU32 sourceBodyIndex_, PxU32 targetBodyIndex_,
		PxU64 primitiveKey_) const
	{
		return sourceType == sourceType_ && targetKind == targetKind_ &&
			sourceBodyIndex == sourceBodyIndex_ &&
			targetBodyIndex == targetBodyIndex_ &&
			primitiveKey == primitiveKey_;
	}
};

// Shape-query data is owned by one pair geometry epoch, not repeated on every
// manifold row.  During the migration from row-inline box metadata the epoch
// compiler binds this descriptor once and rejects inconsistent rows for the
// same (source, target, primitive) pair.
struct AvbdOgcRigidBoxGeometry
{
	PxVec3 halfExtent;
	PxTransform shapeToTarget;
	bool valid;

	AvbdOgcRigidBoxGeometry()
		: halfExtent(0.0f), shapeToTarget(PxIdentity), valid(false)
	{
	}

	PX_FORCE_INLINE bool bind(
		const PxVec3& candidateHalfExtent,
		const PxTransform& candidateShapeToTarget)
	{
		if(!candidateHalfExtent.isFinite() ||
			candidateHalfExtent.x <= 0.0f ||
			candidateHalfExtent.y <= 0.0f ||
			candidateHalfExtent.z <= 0.0f ||
			!candidateShapeToTarget.isValid())
			return false;
		if(!valid)
		{
			halfExtent = candidateHalfExtent;
			shapeToTarget = candidateShapeToTarget;
			valid = true;
			return true;
		}
		const bool sameRotation =
			shapeToTarget.q == candidateShapeToTarget.q ||
			shapeToTarget.q == -candidateShapeToTarget.q;
		return halfExtent == candidateHalfExtent &&
			shapeToTarget.p == candidateShapeToTarget.p && sameRotation;
	}
};

struct AvbdOgcPairGeometryState
{
	PxU32 contactCount;
	PxU32 representativeContact;
	PxU32 admissionContact;
	PxU8 triangleCoreFace;
	PxReal triangleCoreFaceExit;
	PxU32 triangleCoreGeometryEpoch;
	bool hasTriangleCoreManifold;
	PxVec3 representativeNormal;
	PxVec3 representativeRigidOffset;
	PxVec3 referenceRelativePoint;
	PxReal representativeGap;
	PxReal referenceGap;
	PxReal minimumGap;
	AvbdOgcRigidBoxGeometry rigidBox;
	PxU32 epoch;
	bool active;

	AvbdOgcPairGeometryState()
		: contactCount(0), representativeContact(PX_MAX_U32),
		  admissionContact(PX_MAX_U32),
		  triangleCoreFace(PX_MAX_U8), triangleCoreFaceExit(0.0f),
		  triangleCoreGeometryEpoch(0u),
		  hasTriangleCoreManifold(false), representativeNormal(0.0f),
		  representativeRigidOffset(0.0f), referenceRelativePoint(0.0f),
		  representativeGap(PX_MAX_F32), referenceGap(PX_MAX_F32),
		  minimumGap(PX_MAX_F32), epoch(0), active(false)
	{
	}

	PX_FORCE_INLINE void publishTriangleCoreManifold(
		PxU32 geometryEpoch_, PxU8 face, PxReal faceExit)
	{
		if(geometryEpoch_ == 0u || !PxIsFinite(faceExit) || faceExit < 0.0f)
			return;
		triangleCoreGeometryEpoch = geometryEpoch_;
		triangleCoreFace = face;
		triangleCoreFaceExit = faceExit;
		hasTriangleCoreManifold = true;
	}

	PX_FORCE_INLINE bool hasTriangleCoreManifoldForEpoch(
		PxU32 geometryEpoch_) const
	{
		return geometryEpoch_ != 0u && hasTriangleCoreManifold &&
			triangleCoreGeometryEpoch == geometryEpoch_;
	}
};

struct AvbdOgcPairTrustRegionState
{
	PxReal safetyGap;
	PxReal remainingSafeDisplacement;
	PxReal accumulatedRelativeDisplacement;
	bool refreshRequested;

	AvbdOgcPairTrustRegionState()
		: safetyGap(PX_MAX_F32), remainingSafeDisplacement(0.0f),
		  accumulatedRelativeDisplacement(0.0f), refreshRequested(false)
	{
	}
};

enum class AvbdOgcVelocityContactDomain : PxU8
{
	eNONE,
	eSELECTION,
	eTERMINAL
};

struct AvbdOgcPairSolveState
{
	PxReal accumulatedNormalLambda;
	PxReal admittedNormalDisplacement;
	PxReal admittedNormalLoad;
	PxReal localPositionCorrection;
	PxU32 localVelocityContact;
	AvbdOgcVelocityContactDomain localVelocityContactDomain;
	bool admittedAtBoundary;
	bool triangleCoreLocallyResolved;
	bool localPositionApplied;
	bool localVelocityConsumed;

	AvbdOgcPairSolveState()
		: accumulatedNormalLambda(0.0f), admittedNormalDisplacement(0.0f),
		  admittedNormalLoad(0.0f), localPositionCorrection(0.0f),
		  localVelocityContact(PX_MAX_U32),
		  localVelocityContactDomain(AvbdOgcVelocityContactDomain::eNONE),
		  admittedAtBoundary(false), triangleCoreLocallyResolved(false),
		  localPositionApplied(false), localVelocityConsumed(false)
	{
	}

	PX_FORCE_INLINE void publishLocalPositionResult(
		PxU32 contactIndex, PxReal correction,
		AvbdOgcVelocityContactDomain contactDomain)
	{
		if(contactIndex == PX_MAX_U32 || !PxIsFinite(correction) ||
			correction <= 0.0f ||
			contactDomain == AvbdOgcVelocityContactDomain::eNONE)
			return;
		// A terminal witness belongs to the final current-pose manifold and
		// therefore supersedes a selection witness even when its correction is
		// smaller. Within one domain retain the strongest committed correction.
		if(localPositionApplied &&
			PxU8(contactDomain) < PxU8(localVelocityContactDomain))
			return;
		if(localPositionApplied &&
			contactDomain == localVelocityContactDomain &&
			correction <= localPositionCorrection)
			return;
		localVelocityContact = contactIndex;
		localVelocityContactDomain = contactDomain;
		localPositionCorrection = correction;
		localPositionApplied = true;
		localVelocityConsumed = false;
	}

	PX_FORCE_INLINE bool hasPendingLocalVelocity(
		AvbdOgcVelocityContactDomain contactDomain) const
	{
		return localPositionApplied && !localVelocityConsumed &&
			localVelocityContact != PX_MAX_U32 &&
			localVelocityContactDomain == contactDomain;
	}

};

// Shared OGC epoch state. A pair has one identity across component fallback
// and the native mixed-island solver; only target response differs.
struct AvbdOgcPairState
{
	AvbdOgcPairKey key;
	AvbdOgcPairGeometryState geometry;
	AvbdOgcPairTrustRegionState trustRegion;
	AvbdOgcPairSolveState solve;

	PX_FORCE_INLINE void initializeKey(
		AvbdSoftContactSource::Type sourceType,
		AvbdSoftContactTargetKind targetKind,
		PxU32 sourceBodyIndex, PxU32 targetBodyIndex,
		PxU64 primitiveKey)
	{
		key.sourceType = sourceType;
		key.targetKind = targetKind;
		key.sourceBodyIndex = sourceBodyIndex;
		key.targetBodyIndex = targetBodyIndex;
		key.primitiveKey = primitiveKey;
	}

	PX_FORCE_INLINE void addContact()
	{
		++geometry.contactCount;
	}

	// The selection compiler owns the stable contact count. Every other
	// geometry value and all trust/solve values belong to one DCD epoch.
	PX_FORCE_INLINE void beginGeometryEpoch()
	{
		const PxU32 contactCount = geometry.contactCount;
		const PxU32 nextEpoch = geometry.epoch + 1u;
		geometry = AvbdOgcPairGeometryState();
		geometry.contactCount = contactCount;
		geometry.epoch = nextEpoch;
		trustRegion = AvbdOgcPairTrustRegionState();
		solve = AvbdOgcPairSolveState();
	}

	// Provider compilation has already published the shape descriptor for this
	// geometry epoch. Solver initialization resets derived geometry/trust/solve
	// state without discarding that immutable descriptor.
	PX_FORCE_INLINE void beginSolveEpoch()
	{
		const AvbdOgcRigidBoxGeometry rigidBox = geometry.rigidBox;
		beginGeometryEpoch();
		geometry.rigidBox = rigidBox;
	}

	PX_FORCE_INLINE bool matches(
		AvbdSoftContactSource::Type sourceType_,
		AvbdSoftContactTargetKind targetKind_,
		PxU32 sourceBodyIndex_, PxU32 targetBodyIndex_,
		PxU64 primitiveKey_) const
	{
		return key.matches(sourceType_, targetKind_, sourceBodyIndex_,
			targetBodyIndex_, primitiveKey_);
	}
};

PX_FORCE_INLINE PxReal avbdGetOgcPairNormalLoadPerContact(
	const AvbdOgcPairState& pair)
{
	if(!pair.geometry.active || !pair.solve.admittedAtBoundary ||
		!PxIsFinite(pair.solve.admittedNormalLoad) ||
		pair.solve.admittedNormalLoad <= 0.0f ||
		pair.geometry.contactCount == 0u)
		return 0.0f;
	return pair.solve.admittedNormalLoad /
		static_cast<PxReal>(PxMax(1u, pair.geometry.contactCount));
}

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_OGC_PAIR_H
