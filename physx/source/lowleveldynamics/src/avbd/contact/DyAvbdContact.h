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

#ifndef DY_AVBD_CONTACT_H
#define DY_AVBD_CONTACT_H

#include "avbd/core/DyAvbdConstraint.h"
#include "avbd/ogc/DyAvbdOgcPair.h"
#include "foundation/PxMathUtils.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxTransform.h"
#include "foundation/PxVec3.h"

namespace physx
{
namespace Dy
{

static const PxU32 AVBD_CONTACT_POINT_MAX_SUPPORT = 12;

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

struct AvbdSoftContactGeometry
{
	AvbdSoftContactSource source;
	PxU32 particleIdx;
	PxU32 collisionFeatureParticleIdx;
	PxU32 queryBodyIndex;
	PxU32 queryCollisionElementIndex;
	PxU32 targetCollisionElementIndex;
	AvbdSoftContactTargetKind targetKind;
	AvbdVelocityObjectiveOwner velocityOwner;
	AvbdSoftContactTangentOwner tangentOwner;
	PxU32 targetIndex;
	PxU32 queryParticleIndices[3];
	PxReal queryWeights[3];
	AvbdWeightedContactPoint queryPoint;
	PxU32 targetSourceElementIndex;
	PxVec3 normal;
	PxVec3 projNormal;
	PxReal depth;
	PxReal margin;
	PxReal friction;
	PxVec3 tangent1, tangent2;
	PxVec3 surfacePoint;
	PxVec3 kinematicSurfacePointPrevious;
	PxU32 surfaceParticleIndices[3];
	PxReal surfaceWeights[3];
	AvbdWeightedContactPoint targetPoint;
	PxVec3 rigidLocalPoint;

	AvbdSoftContactGeometry()
		: source(), particleIdx(0), collisionFeatureParticleIdx(PX_MAX_U32),
		  queryBodyIndex(PX_MAX_U32),
		  queryCollisionElementIndex(PX_MAX_U32),
		  targetCollisionElementIndex(PX_MAX_U32),
		  targetKind(AvbdSoftContactTargetKind::eUNSUPPORTED),
		  velocityOwner(AvbdVelocityObjectiveOwner::Unsupported),
		  tangentOwner(AvbdSoftContactTangentOwner::ePOSITION_AL),
		  targetIndex(PX_MAX_U32),
		  queryParticleIndices{PX_MAX_U32, PX_MAX_U32, PX_MAX_U32},
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
		  rigidLocalPoint(0.0f)
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
		return targetKind == AvbdSoftContactTargetKind::eDEFORMABLE_SURFACE &&
			surfaceParticleIndices[0] != PX_MAX_U32;
	}

	PX_FORCE_INLINE bool hasWorldStaticTarget() const
	{
		return targetKind == AvbdSoftContactTargetKind::eWORLD_STATIC;
	}

	PX_FORCE_INLINE bool hasKinematicRigidTarget() const
	{
		return targetKind == AvbdSoftContactTargetKind::eKINEMATIC_RIGID;
	}

	PX_FORCE_INLINE bool hasRigidBodyTarget() const
	{
		return targetKind == AvbdSoftContactTargetKind::eRIGID_BODY &&
			targetIndex != PX_MAX_U32;
	}
};

struct AvbdSoftContactAugmentedState
{
	PxVec3 surfacePointPrev;
	PxVec3 particlePointPrev;
	PxReal alLambda;
	PxReal alLambdaTangent[2];
	PxReal penTangent[2];
	bool frictionStick;
	PxReal depenetrationConstraintOffset;
	bool depenetrationLimitInitialized;
	PxReal k;
	PxReal ke;

	AvbdSoftContactAugmentedState()
		: surfacePointPrev(0.0f), particlePointPrev(0.0f),
		  alLambda(0.0f), alLambdaTangent{0.0f, 0.0f},
		  penTangent{1000.0f, 1000.0f}, frictionStick(false),
		  depenetrationConstraintOffset(0.0f),
		  depenetrationLimitInitialized(false), k(1e4f), ke(1e6f)
	{
	}
};

struct AvbdSoftContact
{
	AvbdSoftContactGeometry geometry;
	AvbdSoftContactAugmentedState state;
};

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_H
