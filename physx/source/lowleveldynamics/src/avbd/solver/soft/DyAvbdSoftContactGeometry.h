// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_SOFT_CONTACT_GEOMETRY_H
#define DY_AVBD_SOFT_CONTACT_GEOMETRY_H

#include "avbd/contact/DyAvbdContact.h"
#include "avbd/core/DyAvbdTypes.h"
#include "foundation/PxArray.h"
#include "foundation/PxMat33.h"

namespace physx
{
namespace Dy
{

// AVBD soft-contact stateless primitives and contact-object initialization.
//
// This unit owns support-point access, contact geometry/Jacobian helpers,
// attachment and pin descriptors, and prepared-contact state initialization.
// It contains no detector scheduling or solver lifecycle ownership.
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

// =============================================================================
// Per-particle element adjacency
// =============================================================================

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_CONTACT_GEOMETRY_H
