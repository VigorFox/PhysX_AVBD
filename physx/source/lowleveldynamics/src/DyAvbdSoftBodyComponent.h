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
#include "foundation/PxArray.h"
#include "foundation/PxAssert.h"
#include "foundation/PxMat33.h"
#include "foundation/PxMathUtils.h"
#include "foundation/PxQuat.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxSort.h"
#include "foundation/PxTime.h"
#include "foundation/PxVec3.h"
#include "PxMaterial.h"

#include "DyAvbdConstraint.h"

namespace physx
{
namespace Dy
{

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

// Immutable-for-the-solve output of contact prep.  This record identifies the
// physical objective and contains only geometry/material data.  Target kind
// and velocity owner are explicit prep IR; no sentinel, target kind inference,
// or repeated bit-flag tests may select a later solve stage.
struct AvbdSoftContactGeometry
{
	AvbdSoftContactSource source;
	PxU32 particleIdx;
	AvbdSoftContactTargetKind targetKind;
	AvbdVelocityObjectiveOwner velocityOwner;
	PxU32 targetIndex;
	// The deformable point on the query side. Legacy vertex contacts leave
	// queryParticleIndices[0] invalid and use particleIdx with unit weight.
	// Edge/face contacts store a barycentric point here so one geometric
	// contact owns one AL state while its block contributions are distributed
	// to every incident particle.
	PxU32 queryParticleIndices[3];
	PxReal queryWeights[3];
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
	PxVec3 rigidLocalPoint;

	AvbdSoftContactGeometry()
		: source(), particleIdx(0),
		  targetKind(AvbdSoftContactTargetKind::eUNSUPPORTED),
		  velocityOwner(
			  AvbdVelocityObjectiveOwner::Unsupported),
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
		  rigidLocalPoint(0.0f)
	{
	}

	PX_FORCE_INLINE bool hasBarycentricQueryPoint() const
	{
		return queryParticleIndices[0] != PX_MAX_U32;
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

PX_FORCE_INLINE PxVec3 avbdGetSoftContactSurfacePoint(
	const AvbdSoftContactGeometry& geometry,
	const AvbdSoftParticle* particles)
{
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
	if(geometry.hasBarycentricQueryPoint())
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
	if(geometry.hasDeformableSurfaceTarget())
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
	const AvbdSoftContactGeometry& geometry, PxU32 (&indices)[6])
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

	if(geometry.hasBarycentricQueryPoint())
	{
		for(PxU32 i = 0; i < 3; i++)
			appendUnique(geometry.queryParticleIndices[i]);
	}
	else
		appendUnique(geometry.particleIdx);

	if(geometry.hasDeformableSurfaceTarget())
	{
		for(PxU32 i = 0; i < 3; i++)
			appendUnique(geometry.surfaceParticleIndices[i]);
	}
	return count;
}

// Caller-owned scratch for contact rebuild and state transfer.  Keeping this
// separate from the contact records lets detection reuse capacity without
// making persistent solver state depend on array order.
struct AvbdSoftContactWorkspace
{
	PxArray<AvbdSoftContact> previousContacts;
	PxArray<PxU8> previousUsed;
	PxU64 growthEvents;
	PxU64 growthBytes;
	PxU64 outputGrowthEvents;
	PxU64 outputGrowthBytes;

	AvbdSoftContactWorkspace()
		: growthEvents(0), growthBytes(0),
		  outputGrowthEvents(0), outputGrowthBytes(0)
	{
	}

	void reserve(PxU32 contactCapacity)
	{
		previousContacts.reserve(contactCapacity);
		previousUsed.reserve(contactCapacity);
	}

	void beginStep()
	{
		growthEvents = 0;
		growthBytes = 0;
		outputGrowthEvents = 0;
		outputGrowthBytes = 0;
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
	}

	void reset()
	{
		previousContacts.reset();
		previousUsed.reset();
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
	bool flatteningEnabled;

	// Immutable local rest-space input used to compile self-collision
	// candidates. The public filter distance is copied beside it during
	// prep so contact detection does not reinterpret actor flags or use the
	// global current-space contact radius as a rest-neighborhood policy.
	PxArray<PxVec3> selfCollisionRestPositions;
	PxReal selfCollisionFilterDistance;
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

	AvbdSoftBodyCompiledData()
		: particleStart(0), particleCount(0),
		  flatteningEnabled(false),
		  selfCollisionFilterDistance(0.0f),
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

	void buildSurfaceTriangles()
	{
		surfaceTriangles.clear();
		surfaceVertices.clear();
		surfaceEdges.clear();
		surfaceTriangleElementIndices.clear();

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

		PxArray<PxU64> edgeKeys;
		edgeKeys.reserve(surfaceTriangles.size());
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
				edgeKeys.pushBack((PxU64(lo) << 32) | PxU64(hi));
			}
		}
		PxSort(edgeKeys.begin(), edgeKeys.size());
		for(PxU32 i = 0; i < edgeKeys.size();)
		{
			const PxU64 key = edgeKeys[i];
			AvbdEdgeInfo edge;
			edge.p0 = PxU32(key >> 32);
			edge.p1 = PxU32(key & 0xffffffffu);
			edge.restLength = 0.0f;
			surfaceEdges.pushBack(edge);
			do
				i++;
			while(i < edgeKeys.size() && edgeKeys[i] == key);
		}
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
		buildSurfaceTriangles();
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
	const AvbdSoftBody* queryBody =
		avbdFindSoftBodyForParticle(
			softBodies, numSoftBodies, geometry.particleIdx);
	PxReal maxDepenetrationVelocity = queryBody
		? queryBody->compiled.maxDepenetrationVelocity
		: PX_MAX_F32;
	if(geometry.hasDeformableSurfaceTarget())
	{
		const AvbdSoftBody* targetBody =
			avbdFindSoftBodyForParticle(
				softBodies, numSoftBodies,
				geometry.surfaceParticleIndices[0]);
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

PX_FORCE_INLINE PxMat33 avbdExtractCorotationalRotation(
	const PxMat33& deformationGradient)
{
	PxMat33 rotation = deformationGradient;
	const PxReal determinant = rotation.getDeterminant();
	if(!PxIsFinite(determinant) || PxAbs(determinant) <= 1.0e-9f)
		rotation = PxMat33(PxIdentity);
	else
	{
		for(PxU32 iteration = 0; iteration < 5; iteration++)
		{
			const PxReal det = rotation.getDeterminant();
			if(!PxIsFinite(det) || PxAbs(det) <= 1.0e-9f)
				break;
			const PxMat33 inverseTranspose =
				rotation.getInverse().getTranspose();
			if(!inverseTranspose.column0.isFinite() ||
				!inverseTranspose.column1.isFinite() ||
				!inverseTranspose.column2.isFinite())
				break;
			rotation = (rotation + inverseTranspose) * 0.5f;
		}
	}

	// Finish with an explicitly right-handed orthonormal basis.  The polar
	// iteration alone can retain a reflection for inverted configurations;
	// co-rotational elasticity requires the closest proper rotation.
	PxVec3 column0 = rotation.column0;
	if(!column0.isFinite() ||
		column0.magnitudeSquared() <= 1.0e-12f)
		column0 = deformationGradient.column0;
	if(!column0.isFinite() ||
		column0.magnitudeSquared() <= 1.0e-12f)
		column0 = PxVec3(1.0f, 0.0f, 0.0f);
	column0.normalize();

	PxVec3 column1 =
		rotation.column1 -
		column0 * rotation.column1.dot(column0);
	if(!column1.isFinite() ||
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
	PxReal Jsafe = PxMax(J, Jmin);

	outForce = (Fm * mu + cofm * (lam * (Jsafe - alpha))) * (-V0);

	const PxReal m2 = tet.shapeGradientNormSq[vertexOrder];
	outHessian = PxMat33::createDiagonal(PxVec3(mu * m2 * V0))
	           + avbdOuter(cofm, cofm) * (lam * V0);

	// Extra diagonal regularization for severely compressed / inverted tets
	// to keep the Hessian well-conditioned.
	if (J < 0.5f)
	{
		PxReal reg = (0.5f - J) * lam * V0 * m2;
		outHessian.column0.x += reg;
		outHessian.column1.y += reg;
		outHessian.column2.z += reg;
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

PX_FORCE_INLINE void avbdApplyBendingDamping(
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

	AvbdSoftContactRowForces()
		: normal(0.0f), tangent{0.0f, 0.0f}
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

	if(geometry.friction <= 0.0f || forces.normal >= 0.0f)
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

	if(geometry.friction <= 0.0f || rowForces.normal >= 0.0f)
		return;

	outForce -=
		(geometry.tangent1 * rowForces.tangent[0] +
		 geometry.tangent2 * rowForces.tangent[1]) * jacobianScale;
	outHessian = outHessian +
		avbdOuter(geometry.tangent1, geometry.tangent1) *
			(state.penTangent[0] *
			 jacobianScale * jacobianScale) +
		avbdOuter(geometry.tangent2, geometry.tangent2) *
			(state.penTangent[1] *
			 jacobianScale * jacobianScale);
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
	PxVec3 linearMomentum;
	PxVec3 angularMomentum;
	PxReal mass;
	bool valid;

	AvbdSoftComponentMomentumTarget()
		: linearMomentum(0.0f), angularMomentum(0.0f),
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
	PxU32 queryParticleIndices[3];
	PxReal queryWeights[3];
	PxVec3 normal;
	PxVec3 surfacePoint;
	PxVec3 previousSurfacePoint;

	AvbdCompiledSoftVelocityObjective()
		: owner(AvbdVelocityObjectiveOwner::Unsupported),
		  source(), bodyIndex(PX_MAX_U32),
		  particleIndex(PX_MAX_U32),
		  queryParticleIndices{
			  PX_MAX_U32, PX_MAX_U32, PX_MAX_U32},
		  queryWeights{0.0f, 0.0f, 0.0f},
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

PX_FORCE_INLINE void avbdFinalizeSoftComponentVelocities(
	AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	const AvbdSoftComponentMomentumTarget* momentumTargets,
	const AvbdSoftComponentFinalizeMode* finalizeModes,
	const AvbdCompiledSoftVelocityObjective* velocityObjectives,
	PxU32 numVelocityObjectives, PxReal invDt)
{
	if(!particles || !softBodies || !momentumTargets ||
		!finalizeModes || invDt <= 0.0f)
		return;
	for(PxU32 bodyIndex = 0; bodyIndex < numSoftBodies; bodyIndex++)
	{
		const AvbdSoftComponentMomentumTarget& target =
			momentumTargets[bodyIndex];
		const AvbdSoftComponentFinalizeMode mode =
			finalizeModes[bodyIndex];
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
		PxVec3 targetLinearMomentum = target.linearMomentum;
		PxVec3 targetAngularMomentum = target.angularMomentum;

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
				if(objective.queryParticleIndices[0] != PX_MAX_U32)
				{
					queryPoint = PxVec3(0.0f);
					for(PxU32 queryVertex = 0;
						queryVertex < 3; queryVertex++)
					{
						const PxU32 queryParticle =
							objective.queryParticleIndices[
								queryVertex];
						if(queryParticle == PX_MAX_U32)
							break;
						queryPoint +=
							particles[queryParticle].position *
							objective.queryWeights[queryVertex];
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

inline void avbdDetectSoftWorldPlaneContacts(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdWorldPlane* planes, PxU32 numPlanes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.02f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	for(PxU32 i = 0; i < numParticles; i++)
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
	bool active;

	AvbdRigidTriangleSurfaceVertex()
		: point(0.0f), outward(0.0f, 1.0f, 0.0f),
		  friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE)),
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
	bool active;

	AvbdRigidTriangleSurfaceEdge()
		: p0(PX_MAX_U32), p1(PX_MAX_U32),
		  triangle0(PX_MAX_U32), triangle1(PX_MAX_U32),
		  adjacentCount(0),
		  outward(0.0f, 1.0f, 0.0f), friction(0.5f),
		  frictionCombineMode(PxU8(PxCombineMode::eAVERAGE)),
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
	PxArray<AvbdRigidTriangleSurfaceVertex> vertices;
	PxArray<AvbdRigidTriangleSurfaceEdge> edges;
	PxArray<AvbdRigidTriangleSurfaceTriangle> triangles;

	AvbdRigidTriangleSurface()
		: center(0.0f), rotation(PxIdentity),
		  previousCenter(0.0f), previousRotation(PxIdentity),
		  localBounds(PxBounds3::empty()), localRadius(0.0f),
		  primitiveKey(0),
		  targetKind(AvbdSoftContactTargetKind::eWORLD_STATIC),
		  targetIndex(PX_MAX_U32), shapeToRigidBody(PxIdentity)
	{
	}
};

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
	PxU64 rigidParticleBoxTests;
	PxU64 rigidParticleSphereTests;
	PxU64 rigidParticleCapsuleTests;
	PxU64 rigidParticleConvexTests;
	PxU64 rigidParticleTriangleSurfaceTests;
	PxU64 generatedGroundContacts;
	PxU64 generatedRigidContacts;
	PxU64 generatedSoftContacts;
	PxU64 generatedSelfContacts;

	AvbdSoftCollisionStats()
		: detectionCalls(0), bodyPairs(0), overlappingBodyPairs(0),
		  particleSurfaceCandidates(0), insideTriangleTests(0),
		  closestTriangleTests(0), selfTriangleTests(0),
		  rigidParticleBoxTests(0), rigidParticleSphereTests(0),
		  rigidParticleCapsuleTests(0), rigidParticleConvexTests(0),
		  rigidParticleTriangleSurfaceTests(0),
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
		rigidParticleBoxTests += other.rigidParticleBoxTests;
		rigidParticleSphereTests += other.rigidParticleSphereTests;
		rigidParticleCapsuleTests += other.rigidParticleCapsuleTests;
		rigidParticleConvexTests += other.rigidParticleConvexTests;
		rigidParticleTriangleSurfaceTests +=
			other.rigidParticleTriangleSurfaceTests;
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
	for (PxU32 pi = 0; pi < numParticles; pi++)
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

inline void avbdDetectSoftRigidSweptSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidBox* boxes, PxU32 numBoxes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	for(PxU32 particleIndex = 0;
		particleIndex < numParticles; particleIndex++)
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

inline void avbdDetectSoftRigidSphereSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	for(PxU32 particleIndex = 0;
		particleIndex < numParticles; ++particleIndex)
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

inline void avbdDetectSoftRigidSphereSweptSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	for(PxU32 particleIndex = 0;
		particleIndex < numParticles; ++particleIndex)
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

inline void avbdDetectSoftRigidSphereSweptOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidSphere* spheres, PxU32 numSpheres,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f)
{
	const PxReal translationToleranceSq = 1.0e-10f;
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
				const PxU32 triangleVertices[3] =
					{v0, v1, v2};
				const PxReal expandedRadius =
					sphere.radius + margin;
				bool forwardVertexOwns = false;
				for(PxU32 endpoint = 0;
					endpoint < 3 && !forwardVertexOwns;
					++endpoint)
				{
					const PxU32 vertexIndex =
						triangleVertices[endpoint];
					if(particles[vertexIndex].invMass <= 0.0f)
						continue;
					const PxVec3 vertexRelativeStart =
						particles[vertexIndex].initialPosition -
							centerStart;
					const PxVec3 vertexRelativeEnd =
						particles[vertexIndex].
							predictedPosition - centerEnd;
					const PxReal currentSdf =
						vertexRelativeStart.magnitude() -
							sphere.radius;
					PxReal vertexEntryTime = 0.0f;
					PxVec3 vertexEntryNormal(0.0f);
					forwardVertexOwns =
						PxIsFinite(currentSdf) &&
						(currentSdf < margin ||
						 avbdSegmentEnterExpandedSphere(
							vertexRelativeStart,
							vertexRelativeEnd,
							PxVec3(0.0f),
							expandedRadius,
							vertexEntryTime,
							vertexEntryNormal));
				}
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

inline void avbdDetectSoftRigidCapsuleSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	for(PxU32 particleIndex = 0;
		particleIndex < numParticles; ++particleIndex)
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

inline void avbdDetectSoftRigidCapsuleSweptSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	for(PxU32 particleIndex = 0;
		particleIndex < numParticles; ++particleIndex)
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

inline void avbdDetectSoftRigidCapsuleSweptOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidCapsule* capsules, PxU32 numCapsules,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f)
{
	const PxReal translationToleranceSq = 1.0e-10f;
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

				const PxU32 triangleVertices[3] =
					{v0, v1, v2};
				bool forwardVertexOwns = false;
				for(PxU32 vertex = 0;
					vertex < 3 && !forwardVertexOwns; ++vertex)
				{
					const PxU32 vertexIndex =
						triangleVertices[vertex];
					if(particles[vertexIndex].invMass <= 0.0f)
						continue;
					if(rotationsEquivalent)
					{
						const PxVec3 relativeStart =
							inverseRotation.rotate(
								particles[vertexIndex].initialPosition -
									centerStart);
						const PxVec3 relativeEnd =
							inverseRotation.rotate(
								particles[vertexIndex].
									predictedPosition - centerEnd);
						const PxVec3 currentAxis(
							PxClamp(relativeStart.x,
								-capsule.halfHeight,
								capsule.halfHeight),
							0.0f, 0.0f);
						const PxReal currentSdf =
							(relativeStart - currentAxis).magnitude() -
								capsule.radius;
						PxReal vertexEntryTime = 0.0f;
						PxVec3 vertexEntryNormal(0.0f);
						PxVec3 vertexMedialPoint(0.0f);
						forwardVertexOwns =
							PxIsFinite(currentSdf) &&
							(currentSdf < margin ||
							 avbdSegmentEnterExpandedCapsule(
								relativeStart, relativeEnd,
								capsule.halfHeight,
								expandedRadius,
								vertexEntryTime,
								vertexEntryNormal,
								vertexMedialPoint));
					}
					else
					{
						const PxVec3 pointStart =
							particles[vertexIndex].initialPosition;
						const PxVec3 pointEnd =
							particles[vertexIndex].predictedPosition;
						const PxVec3 startAxis =
							rotationStart.getBasisVector0();
						const PxReal axisCoordinate =
							PxClamp(
								(pointStart - centerStart).
									dot(startAxis),
								-capsule.halfHeight,
								capsule.halfHeight);
						const PxReal currentSdf =
							(pointStart -
								(centerStart +
									startAxis * axisCoordinate)).
								magnitude() -
							capsule.radius;
						AvbdSweptRotatingCapsulePointEntry
							vertexEntry;
						forwardVertexOwns =
							PxIsFinite(currentSdf) &&
							(currentSdf < margin ||
							 avbdSegmentEnterExpandedRotatingCapsule(
								pointStart, pointEnd,
								centerStart, centerEnd,
								rotationStart, rotationEnd,
								capsule.halfHeight,
								capsule.radius, margin,
								vertexEntry));
					}
				}
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

inline void avbdDetectSoftRigidConvexSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
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
	PxReal margin, AvbdSweptConvexPointEntry& result)
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
	if(!avbdQueryRigidConvexLocal(
			convex, segmentStartLocal, currentQuery) ||
		currentQuery.signedDistance < margin)
		return false;

	PxReal time = 0.0f;
	for(PxU32 iteration = 0; iteration < 48; ++iteration)
	{
		AvbdRigidConvexPointQuery query;
		if(!avbdQueryRigidConvexLocal(
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
	PxReal margin, AvbdSweptConvexPointEntry& result)
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
		if(!avbdQueryRigidConvexLocal(
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

inline void avbdDetectSoftRigidConvexSweptSDF(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	for(PxU32 particleIndex = 0;
		particleIndex < numParticles; ++particleIndex)
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

inline void avbdDetectSoftRigidConvexSweptOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidConvex* convexes, PxU32 numConvexes,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f)
{
	const PxReal translationToleranceSq = 1.0e-10f;
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
					AvbdRigidConvexPointQuery currentQuery;
					AvbdSweptConvexPointEntry vertexEntry;
					const PxVec3 pointStart =
						particles[vertexIndex].initialPosition;
					const PxVec3 pointEnd =
						particles[vertexIndex].predictedPosition;
					const PxVec3 relativeStart =
						rotationStart.getConjugate().rotate(
							pointStart - centerStart);
					if(avbdQueryRigidConvexLocal(
							convex, relativeStart,
							currentQuery))
					{
						if(currentQuery.signedDistance < margin)
							forwardVertexOwns = true;
						else if(rotationsEquivalent)
						{
							const PxVec3 relativeEnd =
								inverseRotation.rotate(
									pointEnd - centerEnd);
							forwardVertexOwns =
								avbdSegmentEnterExpandedConvex(
									convex, relativeStart,
									relativeEnd, margin,
									vertexEntry);
						}
						else
							forwardVertexOwns =
								avbdSegmentEnterExpandedRotatingConvex(
									convex, pointStart, pointEnd,
									centerStart, centerEnd,
									rotationStart, rotationEnd,
									margin, vertexEntry);
					}
				}
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
				for(PxU32 rigidEdgeIndex = 0;
					rigidEdgeIndex < convex.edges.size();
					++rigidEdgeIndex)
				{
					const AvbdRigidConvexEdge& rigidEdge =
						convex.edges[rigidEdgeIndex];
					if(rigidEdge.p0 >= convex.vertices.size() ||
						rigidEdge.p1 >= convex.vertices.size())
						continue;
					const PxVec3 rigid0 =
						centerStart + rotationStart.rotate(
							convex.vertices[rigidEdge.p0]);
					const PxVec3 rigid1 =
						centerStart + rotationStart.rotate(
							convex.vertices[rigidEdge.p1]);
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
							convex.localRadius + margin;
						rigidMinimum =
							centerStart.minimum(relativeCenterEnd) -
								PxVec3(rotationExtent);
						rigidMaximum =
							centerStart.maximum(relativeCenterEnd) +
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

				const PxU32 triangleVertices[3] =
					{v0, v1, v2};
				bool forwardVertexOwns = false;
				for(PxU32 vertex = 0;
					vertex < 3 && !forwardVertexOwns;
					++vertex)
				{
					const PxU32 vertexIndex =
						triangleVertices[vertex];
					if(particles[vertexIndex].invMass <= 0.0f)
						continue;
					AvbdRigidConvexPointQuery currentQuery;
					AvbdSweptConvexPointEntry vertexEntry;
					const PxVec3 pointStart =
						particles[vertexIndex].initialPosition;
					const PxVec3 pointEnd =
						particles[vertexIndex].predictedPosition;
					const PxVec3 relativeStart =
						rotationStart.getConjugate().rotate(
							pointStart - centerStart);
					if(avbdQueryRigidConvexLocal(
							convex, relativeStart,
							currentQuery))
					{
						if(currentQuery.signedDistance < margin)
							forwardVertexOwns = true;
						else if(rotationsEquivalent)
						{
							const PxVec3 relativeEnd =
								inverseRotation.rotate(
									pointEnd - centerEnd);
							forwardVertexOwns =
								avbdSegmentEnterExpandedConvex(
									convex, relativeStart,
									relativeEnd, margin,
									vertexEntry);
						}
						else
							forwardVertexOwns =
								avbdSegmentEnterExpandedRotatingConvex(
									convex, pointStart, pointEnd,
									centerStart, centerEnd,
									rotationStart, rotationEnd,
									margin, vertexEntry);
					}
				}
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
					const PxVec3& rigid0Local =
						convex.vertices[rigidEdge.p0];
					const PxVec3& rigid1Local =
						convex.vertices[rigidEdge.p1];
					const PxVec3 softMinimum =
						soft0Local.minimum(soft1Local);
					const PxVec3 softMaximum =
						soft0Local.maximum(soft1Local);
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
	AvbdRigidTriangleSurfacePointQuery& result)
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
	bool found = false;
	for(PxU32 triangleIndex = 0;
		triangleIndex < surface.triangles.size();
		++triangleIndex)
	{
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
	PxU32 numSoftBodies = 0)
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
					surface, localPoint, margin, query))
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
	AvbdSweptTriangleSurfacePointEntry& result)
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
			currentQuery))
		return false;

	const PxReal projectionTolerance = 1.0e-5f;
	for(PxU32 triangleIndex = 0;
		triangleIndex < surface.triangles.size();
		++triangleIndex)
	{
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

	for(PxU32 edgeIndex = 0;
		edgeIndex < surface.edges.size(); ++edgeIndex)
	{
		const AvbdRigidTriangleSurfaceEdge& edge =
			surface.edges[edgeIndex];
		if(!edge.active ||
			edge.p0 >= surface.vertices.size() ||
			edge.p1 >= surface.vertices.size())
			continue;
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

	for(PxU32 vertexIndex = 0;
		vertexIndex < surface.vertices.size();
		++vertexIndex)
	{
		const AvbdRigidTriangleSurfaceVertex& vertex =
			surface.vertices[vertexIndex];
		if(!vertex.active)
			continue;
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
	PxReal margin, AvbdSweptTriangleSurfacePointEntry& result)
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
				surface, localPoint, maximumDistance, query))
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

inline void avbdDetectSoftRigidTriangleSurfaceSwept(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidTriangleSurface* surfaces,
	PxU32 numSurfaces, PxArray<AvbdSoftContact>& contacts,
	PxReal margin = 0.05f,
	const AvbdSoftBody* softBodies = NULL,
	PxU32 numSoftBodies = 0)
{
	for(PxU32 particleIndex = 0;
		particleIndex < numParticles; ++particleIndex)
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
						margin, entry))
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

PX_FORCE_INLINE bool avbdTriangleSurfaceForwardVertexOwnsSweep(
	const AvbdRigidTriangleSurface& surface,
	const PxVec3& centerStart, const PxVec3& centerEnd,
	const PxQuat& rotationStart, const PxQuat& rotationEnd,
	bool rotationsEquivalent,
	const AvbdSoftParticle& particle, PxReal margin)
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
			currentQuery) &&
		currentQuery.distance < margin)
		return true;
	AvbdSweptTriangleSurfacePointEntry entry;
	if(!rotationsEquivalent)
		return avbdSegmentEnterExpandedRotatingTriangleSurface(
			surface, particle.initialPosition,
			particle.predictedPosition,
			centerStart, centerEnd,
			rotationStart, rotationEnd, margin, entry);
	const PxVec3 relativeEnd =
		rotationEnd.getConjugate().rotate(
			particle.predictedPosition - centerEnd);
	return avbdSegmentEnterExpandedTriangleSurface(
		surface, relativeStart, relativeEnd, margin, entry);
}

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
	PxReal margin = 0.05f)
{
	const PxReal translationToleranceSq = 1.0e-10f;
	for(PxU32 bodyIndex = 0;
		bodyIndex < numSoftBodies; ++bodyIndex)
	{
		const AvbdSoftBody& body = softBodies[bodyIndex];
		if(!body.compiled.speculativeCCDEnabled)
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
					forwardVertexOwns =
						avbdTriangleSurfaceForwardVertexOwnsSweep(
							surface,
							centerStart, centerEnd,
							rotationStart, rotationEnd,
							rotationsEquivalent,
							particles[vertexIndex], margin);
				}
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
				for(PxU32 rigidEdgeIndex = 0;
					rigidEdgeIndex < surface.edges.size();
					++rigidEdgeIndex)
				{
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
			}

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
					forwardVertexOwns =
						avbdTriangleSurfaceForwardVertexOwnsSweep(
							surface,
							centerStart, centerEnd,
							rotationStart, rotationEnd,
							rotationsEquivalent,
							particles[particleIndex], margin);
				}
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
					rigidVertexIndex < surface.vertices.size();
					++rigidVertexIndex)
				{
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
			}
		}
	}
}

// Reverse OGC completeness for an open triangle surface: active rigid edge
// versus soft boundary edge, and active rigid vertex versus soft face. Feature
// endpoint cases are excluded so forward vertex-triangle owns them.
inline void avbdDetectSoftRigidTriangleSurfaceOGCFeatures(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdRigidTriangleSurface* surfaces, PxU32 numSurfaces,
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

		for(PxU32 surfaceIndex = 0;
			surfaceIndex < numSurfaces; ++surfaceIndex)
		{
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
							surface.center);
				const PxVec3 soft1Local =
					inverseRotation.rotate(
						particles[softEdge.p1].position -
							surface.center);
				for(PxU32 rigidEdgeIndex = 0;
					rigidEdgeIndex < surface.edges.size();
					++rigidEdgeIndex)
				{
					const AvbdRigidTriangleSurfaceEdge& rigidEdge =
						surface.edges[rigidEdgeIndex];
					if(!rigidEdge.active ||
						rigidEdge.p0 >= surface.vertices.size() ||
						rigidEdge.p1 >= surface.vertices.size())
						continue;
					const PxVec3& rigid0Local =
						surface.vertices[rigidEdge.p0].point;
					const PxVec3& rigid1Local =
						surface.vertices[rigidEdge.p1].point;
					const PxVec3 softMinimum =
						soft0Local.minimum(soft1Local);
					const PxVec3 softMaximum =
						soft0Local.maximum(soft1Local);
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
					inverseRotation.rotate(p0 - surface.center);
				const PxVec3 p1Local =
					inverseRotation.rotate(p1 - surface.center);
				const PxVec3 p2Local =
					inverseRotation.rotate(p2 - surface.center);
				const PxVec3 triangleMinimum =
					p0Local.minimum(p1Local).minimum(p2Local) -
						PxVec3(margin);
				const PxVec3 triangleMaximum =
					p0Local.maximum(p1Local).maximum(p2Local) +
						PxVec3(margin);

				for(PxU32 rigidVertexIndex = 0;
					rigidVertexIndex < surface.vertices.size();
					++rigidVertexIndex)
				{
					const AvbdRigidTriangleSurfaceVertex& vertex =
						surface.vertices[rigidVertexIndex];
					if(!vertex.active ||
						vertex.point.x < triangleMinimum.x ||
						vertex.point.x > triangleMaximum.x ||
						vertex.point.y < triangleMinimum.y ||
						vertex.point.y > triangleMaximum.y ||
						vertex.point.z < triangleMinimum.z ||
						vertex.point.z > triangleMaximum.z)
						continue;
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
			}
		}
	}
}

// =============================================================================
// PATH 3 (OGC): Simplified Soft-Soft Contact (Sec 3.9)
//
// Outward-only offset, pure quadratic energy, DCD for penetration.
// =============================================================================

inline void avbdDetectSoftSoftOGC(
	const AvbdSoftParticle* particles, PxU32 numParticles,
	const AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts,
	const AvbdOGCParams& params = AvbdOGCParams(),
	AvbdSoftCollisionStats* stats = NULL)
{
	PX_UNUSED(numParticles);
	PxReal r = params.contactRadius;

	for (PxU32 sA = 0; sA < numSoftBodies; sA++)
	{
		for (PxU32 sB = sA + 1; sB < numSoftBodies; sB++)
		{
			if(stats)
				stats->bodyPairs++;
			const AvbdSoftBody& bodyA = softBodies[sA];
			const AvbdSoftBody& bodyB = softBodies[sB];
			const bool pairSpeculative =
				bodyA.compiled.speculativeCCDEnabled ||
				bodyB.compiled.speculativeCCDEnabled;
			const PxReal pairFriction = 0.5f * (
				PxMax(bodyA.material.dynamicFriction, 0.0f) +
				PxMax(bodyB.material.dynamicFriction, 0.0f));

			// AABB broadphase per body pair
			PxVec3 minA(PX_MAX_F32), maxA(-PX_MAX_F32);
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
			PxVec3 minB(PX_MAX_F32), maxB(-PX_MAX_F32);
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
			if (minA.x > maxB.x + r || maxA.x < minB.x - r ||
				minA.y > maxB.y + r || maxA.y < minB.y - r ||
				minA.z > maxB.z + r || maxA.z < minB.z - r)
				continue;
			if(stats)
				stats->overlappingBodyPairs++;

			// Lambda: test particles of testBody against surface of surfBody
			auto testParticlesVsSurface = [&](
				const AvbdSoftBody& testBody, const AvbdSoftBody& surfBody,
				PxU32 surfBodyIdx,
				const PxVec3& aabbLo, const PxVec3& aabbHi)
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
						for(PxU32 ti = 0;
							ti + 2 <
								surfBody.compiled.surfaceTriangles.size();
							ti += 3)
						{
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

					// Not inside: OGC outward offset blocks on surface
					for (PxU32 ti = 0; ti + 2 < surfBody.compiled.surfaceTriangles.size(); ti += 3)
					{
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
			testParticlesVsSurface(bodyA, bodyB, sB, minB, maxB);
			testParticlesVsSurface(bodyB, bodyA, sA, minA, maxA);

			// Vertex-face features alone do not own a crossing between two
			// edge interiors.  Compile one canonical A-edge/B-edge row for
			// that missing OGC feature, with the swept owner taking
			// precedence over the end-of-step discrete owner.
			struct SoftPairEdgeBounds
			{
				PxU32 edgeIndex;
				PxVec3 minimum;
				PxVec3 maximum;
			};
			auto buildEdgeBounds =
				[&](const AvbdSoftBody& body,
					PxArray<SoftPairEdgeBounds>& bounds)
			{
				bounds.reserve(body.compiled.surfaceEdges.size());
				for(PxU32 edgeIndex = 0;
					edgeIndex < body.compiled.surfaceEdges.size();
					edgeIndex++)
				{
					const AvbdEdgeInfo& edge =
						body.compiled.surfaceEdges[edgeIndex];
					if(edge.p0 >= numParticles ||
						edge.p1 >= numParticles)
						continue;
					SoftPairEdgeBounds edgeBounds;
					edgeBounds.edgeIndex = edgeIndex;
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
					[](const SoftPairEdgeBounds& a,
					   const SoftPairEdgeBounds& b)
					{
						return a.minimum.x < b.minimum.x;
					});
			};
			PxArray<SoftPairEdgeBounds> edgeBoundsA;
			PxArray<SoftPairEdgeBounds> edgeBoundsB;
			buildEdgeBounds(bodyA, edgeBoundsA);
			buildEdgeBounds(bodyB, edgeBoundsB);
			const PxReal edgeFeatureEpsilon = 1.0e-4f;
			const PxReal edgeDistanceEpsilon = 1.0e-8f;

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
				const SoftPairEdgeBounds& boundsA =
					edgeBoundsA[sortedEdgeA];
				for(PxU32 sortedEdgeB = 0;
					sortedEdgeB < edgeBoundsB.size();
					sortedEdgeB++)
				{
					const SoftPairEdgeBounds& boundsB =
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
								stabilizeNormal(entry.normal),
								0.0f,
								target0AtEntry *
									(1.0f - entry.rigidWeight1) +
								target1AtEntry *
									entry.rigidWeight1);
							continue;
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
					appendEdgeContact(
						queryWeight1, targetWeight1,
						stabilizeNormal(normal),
						r - distance, targetClosest);
				}
			}
		}
	}
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
	PxArray<PxReal>& bounds)
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
	PxArray<PxReal> vertexMinimums;
	PxArray<PxReal> triangleMinimums;
	PxArray<PxReal> edgeMinimums;
	vertexMinimums.resize(particleCount);
	triangleMinimums.resize(
		sb.compiled.surfaceTriangles.size() / 3);
	edgeMinimums.resize(sb.compiled.surfaceEdges.size());
	for(PxU32 vertexIndex = 0;
		vertexIndex < particleCount; vertexIndex++)
		vertexMinimums[vertexIndex] = rq;
	for(PxU32 triangleIndex = 0;
		triangleIndex < triangleMinimums.size(); triangleIndex++)
		triangleMinimums[triangleIndex] = rq;
	for(PxU32 edgeIndex = 0;
		edgeIndex < edgeMinimums.size(); edgeIndex++)
		edgeMinimums[edgeIndex] = rq;

	struct SafetyTriangleBounds
	{
		PxU32 triangleOffset;
		PxVec3 minimum;
		PxVec3 maximum;
	};
	struct SafetyVertex
	{
		PxU32 localIndex;
		PxReal x;
	};
	PxArray<SafetyTriangleBounds> triangleBounds;
	triangleBounds.reserve(triangleMinimums.size());
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
		SafetyTriangleBounds triangle;
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
		[](const SafetyTriangleBounds& a,
		   const SafetyTriangleBounds& b)
		{
			return a.minimum.x < b.minimum.x;
		});
	PxArray<SafetyVertex> sortedVertices;
	sortedVertices.reserve(
		sb.compiled.surfaceVertices.size());
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
		SafetyVertex vertex;
		vertex.localIndex = globalIndex - particleStart;
		vertex.x =
			particles[globalIndex].position.x;
		sortedVertices.pushBack(vertex);
	}
	PxSort(
		sortedVertices.begin(), sortedVertices.size(),
		[](const SafetyVertex& a, const SafetyVertex& b)
		{
			return a.x < b.x;
		});

	// OGC Eq. 22 and Eq. 26.  The sweep-and-prune list is the CPU
	// equivalent of the paper's facet-BVH radius query.  Values are
	// initialized to rq, so pairs outside the query shell cannot reduce the
	// conservative bound.
	PxArray<PxU32> activeTriangles;
	activeTriangles.reserve(triangleBounds.size());
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
			const SafetyTriangleBounds& triangle =
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
			const AvbdClosestPointResult closest =
				avbdClosestPointOnTriangleOGC(
					point,
					particles[vertex0].position,
					particles[vertex1].position,
					particles[vertex2].position);
			if(closest.distance >= rq)
				continue;
			vertexMinimums[localIndex] = PxMin(
				vertexMinimums[localIndex],
				closest.distance);
			const PxU32 triangleIndex = triangleOffset / 3;
			triangleMinimums[triangleIndex] = PxMin(
				triangleMinimums[triangleIndex],
				closest.distance);
		}
	}

	struct SafetyEdgeBounds
	{
		PxU32 edgeIndex;
		PxVec3 minimum;
		PxVec3 maximum;
	};
	PxArray<SafetyEdgeBounds> edgeBounds;
	edgeBounds.reserve(sb.compiled.surfaceEdges.size());
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
		SafetyEdgeBounds edgeBound;
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
		[](const SafetyEdgeBounds& a,
		   const SafetyEdgeBounds& b)
		{
			return a.minimum.x < b.minimum.x;
		});

	// OGC Eq. 24.  Every unordered non-incident edge pair contributes its
	// distance to both edge minima.
	for(PxU32 sortedEdge0 = 0;
		sortedEdge0 < edgeBounds.size(); sortedEdge0++)
	{
		const SafetyEdgeBounds& bounds0 =
			edgeBounds[sortedEdge0];
		for(PxU32 sortedEdge1 = sortedEdge0 + 1;
			sortedEdge1 < edgeBounds.size(); sortedEdge1++)
		{
			const SafetyEdgeBounds& bounds1 =
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
	for(PxU32 localIndex = 0;
		localIndex < particleCount; localIndex++)
		bounds[localIndex] = vertexMinimums[localIndex];
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
	AvbdSoftCollisionStats* stats = NULL)
{
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
	PxArray<PxReal> tetStressCoefficients;
	if(!sb.compiled.tetElements.empty())
	{
		tetStressCoefficients.resize(
			sb.compiled.tetElements.size());
		for(PxU32 tetIndex = 0;
			tetIndex < sb.compiled.tetElements.size();
			tetIndex++)
		{
			tetStressCoefficients[tetIndex] =
				avbdComputeTetStressCoefficient(
					sb.compiled.tetElements[tetIndex],
					particles);
		}
	}
	auto targetStressAllowsTriangle =
		[&](PxU32 triangleOffset) -> bool
	{
		if(tetStressCoefficients.empty())
			return true;
		const PxU32 triangleIndex = triangleOffset / 3;
		if(triangleIndex >=
			sb.compiled.surfaceTriangleElementIndices.size())
			return true;
		const PxU32 sourceElementIndex =
			sb.compiled.surfaceTriangleElementIndices[
				triangleIndex];
		for(PxU32 tetIndex = 0;
			tetIndex < sb.compiled.tetElements.size();
			tetIndex++)
		{
			if(sb.compiled.tetElements[tetIndex].
					sourceElementIndex == sourceElementIndex)
			{
				return tetStressCoefficients[tetIndex] <=
					sb.compiled.selfCollisionStressTolerance;
			}
		}
		return true;
	};

	struct SelfTriangleBounds
	{
		PxU32 triangleOffset;
		PxVec3 minimum;
		PxVec3 maximum;
	};
	struct SelfVertexSweepEntry
	{
		PxU32 localIndex;
		PxReal minimumX;
		PxReal maximumX;
	};
	PxArray<SelfTriangleBounds> triangleBounds;
	triangleBounds.reserve(
		sb.compiled.surfaceTriangles.size() / 3);
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
		SelfTriangleBounds triangle;
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
	}
	PxSort(
		triangleBounds.begin(), triangleBounds.size(),
		[](const SelfTriangleBounds& a,
		   const SelfTriangleBounds& b)
		{
			return a.minimum.x < b.minimum.x;
		});
	PxArray<SelfVertexSweepEntry> sortedVertices;
	sortedVertices.reserve(
		sb.compiled.surfaceVertices.size());
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
		SelfVertexSweepEntry vertex;
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
	}
	PxSort(
		sortedVertices.begin(), sortedVertices.size(),
		[](const SelfVertexSweepEntry& a,
		   const SelfVertexSweepEntry& b)
		{
			return a.minimumX < b.minimumX;
		});

	// Radius-query broadphase.  The previous all-vertices by all-triangles
	// traversal made each OGC redetection O(V*T).
	PxArray<PxU32> activeTriangles;
	activeTriangles.reserve(triangleBounds.size());
	PxArray<PxU64> emittedFeatureKeys;
	PxU32 triangleCursor = 0;
	for(PxU32 sortedVertexIndex = 0;
		sortedVertexIndex < sortedVertices.size();
		sortedVertexIndex++)
	{
		const PxU32 li =
			sortedVertices[sortedVertexIndex].localIndex;
		const PxU32 gi = sb.compiled.particleStart + li;
		const PxVec3& pp = particles[gi].position;
		const PxReal vertexMinimumX =
			sortedVertices[sortedVertexIndex].minimumX;
		const PxReal vertexMaximumX =
			sortedVertices[sortedVertexIndex].maximumX;
		while(triangleCursor < triangleBounds.size() &&
			triangleBounds[triangleCursor].minimum.x <=
				vertexMaximumX + r)
			activeTriangles.pushBack(triangleCursor++);
		for(PxU32 activeIndex = 0;
			activeIndex < activeTriangles.size();)
		{
			const SelfTriangleBounds& triangle =
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

		// Select a single first face crossing for this vertex.  This prevents
		// adjacent triangles from compiling several speculative rows for one
		// physical crossing and keeps every outer redetection deterministic.
		if(sweepEnabled)
		{
			const PxVec3 vertexMinimum =
				particles[gi].initialPosition.minimum(pp);
			const PxVec3 vertexMaximum =
				particles[gi].initialPosition.maximum(pp);
			PxReal bestEntryTime = PX_MAX_F32;
			PxU32 bestTriangleOffset = PX_MAX_U32;
			AvbdSweptTriangleEntry bestEntry;
			for(PxU32 activeIndex = 0;
				activeIndex < activeTriangles.size();
				activeIndex++)
			{
				const SelfTriangleBounds& triangle =
					triangleBounds[activeTriangles[activeIndex]];
				if(triangle.minimum.y > vertexMaximum.y + r ||
					triangle.maximum.y < vertexMinimum.y - r ||
					triangle.minimum.z > vertexMaximum.z + r ||
					triangle.maximum.z < vertexMinimum.z - r)
					continue;
				if(stats)
					stats->selfTriangleTests++;
				const PxU32 ti = triangle.triangleOffset;
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
			activeIndex < activeTriangles.size();
			activeIndex++)
		{
			const SelfTriangleBounds& triangle =
				triangleBounds[activeTriangles[activeIndex]];
			if(triangle.minimum.y > pp.y + r ||
				triangle.maximum.y < pp.y - r ||
				triangle.minimum.z > pp.z + r ||
				triangle.maximum.z < pp.z - r)
				continue;
			if(stats)
				stats->selfTriangleTests++;

			const PxU32 ti = triangle.triangleOffset;
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
	struct SelfEdgeBounds
	{
		PxU32 edgeIndex;
		PxVec3 minimum;
		PxVec3 maximum;
	};
	PxArray<SelfEdgeBounds> edgeBounds;
	edgeBounds.reserve(sb.compiled.surfaceEdges.size());
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
		SelfEdgeBounds bounds;
		bounds.edgeIndex = edgeIndex;
		bounds.minimum =
			particles[edge.p0].position.minimum(
				particles[edge.p1].position);
		bounds.maximum =
			particles[edge.p0].position.maximum(
				particles[edge.p1].position);
		if(sweepEnabled)
		{
			bounds.minimum = bounds.minimum.minimum(
				particles[edge.p0].initialPosition).minimum(
				particles[edge.p1].initialPosition);
			bounds.maximum = bounds.maximum.maximum(
				particles[edge.p0].initialPosition).maximum(
				particles[edge.p1].initialPosition);
		}
		edgeBounds.pushBack(bounds);
	}
	PxSort(
		edgeBounds.begin(), edgeBounds.size(),
		[](const SelfEdgeBounds& a, const SelfEdgeBounds& b)
		{
			return a.minimum.x < b.minimum.x;
		});

	const PxReal edgeFeatureEpsilon = 1.0e-4f;
	const PxReal edgeDistanceEpsilon = 1.0e-8f;
	auto targetStressAllowsEdge =
		[&](PxU32 edge0, PxU32 edge1) -> bool
	{
		if(tetStressCoefficients.empty())
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
	for(PxU32 sortedEdge0 = 0;
		sortedEdge0 < edgeBounds.size(); sortedEdge0++)
	{
		const SelfEdgeBounds& bounds0 = edgeBounds[sortedEdge0];
		for(PxU32 sortedEdge1 = sortedEdge0 + 1;
			sortedEdge1 < edgeBounds.size(); sortedEdge1++)
		{
			const SelfEdgeBounds& bounds1 =
				edgeBounds[sortedEdge1];
			if(bounds1.minimum.x > bounds0.maximum.x + r)
				break;
			if(bounds0.minimum.y > bounds1.maximum.y + r ||
				bounds0.maximum.y < bounds1.minimum.y - r ||
				bounds0.minimum.z > bounds1.maximum.z + r ||
				bounds0.maximum.z < bounds1.minimum.z - r)
				continue;

			const PxU32 queryEdgeIndex =
				PxMin(bounds0.edgeIndex, bounds1.edgeIndex);
			const PxU32 targetEdgeIndex =
				PxMax(bounds0.edgeIndex, bounds1.edgeIndex);
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

// =============================================================================
// Convenience: detect all OGC contacts (ground + soft-rigid + soft-soft + self)
// =============================================================================

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
			if(oldGeometry.particleIdx != geometry.particleIdx ||
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
		previousUsed[bestIdx] = 1;

		const PxReal dualDecay = 0.99f;
		const PxReal penaltyDecay = 0.999f;
		const AvbdSoftContactGeometry& bestGeometry = best->geometry;
		const AvbdSoftContactAugmentedState& bestState = best->state;
		state.alLambda = bestState.alLambda * dualDecay;
		state.k = PxClamp(
			bestState.k * penaltyDecay, state.k, state.ke);
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
		state.depenetrationConstraintOffset =
			bestState.depenetrationConstraintOffset;
		state.depenetrationLimitInitialized =
			bestState.depenetrationLimitInitialized;
		if(bestState.frictionStick)
		{
			state.particlePointPrev = bestState.particlePointPrev;
			state.surfacePointPrev = bestState.surfacePointPrev;
		}
	}
}

// Per-body self-collision adjacency array type
typedef PxArray<PxArray<PxU32> > AvbdSelfCollisionAdjacency;

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
			contacts, params.contactRadius);
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
			contacts, params.contactRadius);
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
			contacts, params.contactRadius);
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
			softBodies, numSoftBodies);
		avbdDetectSoftRigidTriangleSurfaceSwept(
			particles, numParticles,
			rigidTriangleSurfaces,
			numRigidTriangleSurfaces,
			contacts, params.contactRadius,
			softBodies, numSoftBodies);
		avbdDetectSoftRigidTriangleSurfaceSweptOGCFeatures(
			particles, numParticles,
			rigidTriangleSurfaces,
			numRigidTriangleSurfaces,
			softBodies, numSoftBodies,
			contacts, params.contactRadius);
		avbdDetectSoftRigidTriangleSurfaceOGCFeatures(
			particles, numParticles,
			rigidTriangleSurfaces,
			numRigidTriangleSurfaces,
			softBodies, numSoftBodies,
			contacts, params.contactRadius);
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
		                      contacts, params, stats);
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
			                           perBodyAdj[si], contacts, params, stats);
			if(stats)
				stats->generatedSelfContacts += contacts.size() - selfStart;
		}
	}

	workspace.recordOutputCapacityGrowth(
		outputCapacityBefore, contacts.capacity());
	avbdTransferSoftContactState(
		previousContacts.begin(), previousContacts.size(),
		particles, contacts, &workspace);
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
	PxReal dynamicFriction = 0.5f)
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
	PxU64 workspaceGrowthEvents;
	PxU64 workspaceGrowthBytes;
	PxU64 contactWorkspaceGrowthEvents;
	PxU64 contactWorkspaceGrowthBytes;
	PxU64 contactOutputGrowthEvents;
	PxU64 contactOutputGrowthBytes;
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
	PxReal finalMaxLocalSolveDisplacement;
	PxReal finalMaxAppliedDisplacement;
	PxReal finalMaxDisplacement;

	AvbdSoftBodyStepStats()
		: predictionMs(0.0), contactIndexMs(0.0), bodyPrecomputeMs(0.0),
		  bodySolveMs(0.0), particleSolveMs(0.0), projectionMs(0.0),
		  dualMs(0.0), redetectMs(0.0), velocityMs(0.0), frictionMs(0.0),
		  requestedOuterIterations(0), requestedInnerIterations(0),
		  executedOuterIterations(0), executedInnerIterations(0),
		  particleSweeps(0), workspaceGrowthEvents(0),
		  workspaceGrowthBytes(0), contactWorkspaceGrowthEvents(0),
		  contactWorkspaceGrowthBytes(0), contactOutputGrowthEvents(0),
		  contactOutputGrowthBytes(0),
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
		  finalMaxLocalSolveDisplacement(0.0f),
		  finalMaxAppliedDisplacement(0.0f),
		  finalMaxDisplacement(0.0f)
	{
	}
};

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

struct AvbdSoftBodyWorkspace
{
	AvbdSoftContactWorkspace contact;
	PxArray<AvbdSoftContactParticleRef> contactIndices;
	PxArray<PxU32> contactStarts;
	PxArray<PxU32> contactCounts;
	PxArray<PxVec3> chebyPrevPos;
	PxArray<PxVec3> chebyPrevPrevPos;
	PxArray<PxReal> selfCollisionSafetyBounds;
	PxArray<PxReal> bodySelfCollisionSafetyBounds;
	PxArray<AvbdCompiledSoftVelocityObjective>
		compiledVelocityObjectives;
	PxArray<AvbdSoftComponentMomentumTarget>
		componentMomentumTargets;
	PxArray<AvbdSoftComponentFinalizeMode>
		componentFinalizeModes;
	PxU64 growthEvents;
	PxU64 growthBytes;

	AvbdSoftBodyWorkspace() : growthEvents(0), growthBytes(0)
	{
	}

	void reserve(PxU32 numParticles, PxU32 contactCapacity)
	{
		contact.reserve(contactCapacity);
		contactIndices.reserve(contactCapacity * 4);
		contactStarts.reserve(numParticles + 1);
		contactCounts.reserve(numParticles);
		chebyPrevPos.reserve(numParticles);
		chebyPrevPrevPos.reserve(numParticles);
		selfCollisionSafetyBounds.reserve(numParticles);
		bodySelfCollisionSafetyBounds.reserve(numParticles);
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
		contact.beginStep();
	}

	void reset()
	{
		contact.reset();
		contactIndices.reset();
		contactStarts.reset();
		contactCounts.reset();
		chebyPrevPos.reset();
		chebyPrevPrevPos.reset();
		selfCollisionSafetyBounds.reset();
		bodySelfCollisionSafetyBounds.reset();
		compiledVelocityObjectives.reset();
		componentMomentumTargets.reset();
		componentFinalizeModes.reset();
		beginStep();
	}
};

inline void avbdStepSoftBodies(
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
	const AvbdOGCParams* ogcParams = NULL)
{
	if (numParticles == 0 || numSoftBodies == 0) return;
	// A total budget lets callers retain the outer contact-redetection
	// schedule without rounding every stage up to a full inner batch.
	const PxU32 requestedInnerIterationBudget =
		totalInnerIterationBudget > 0
			? PxMax(totalInnerIterationBudget, outerIterations)
			: outerIterations * innerIterations;
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
		*stepStats = AvbdSoftBodyStepStats();
		stepStats->requestedOuterIterations = outerIterations;
		stepStats->requestedInnerIterations =
			requestedInnerIterationBudget;
	}
	PxTime stageTimer;
	AvbdSoftBodyWorkspace localWorkspace;
	AvbdSoftBodyWorkspace& workspace =
		persistentWorkspace ? *persistentWorkspace : localWorkspace;
	workspace.beginStep();
	PxArray<AvbdCompiledSoftVelocityObjective>&
		compiledVelocityObjectives =
			workspace.compiledVelocityObjectives;
	compiledVelocityObjectives.clear();
	PxArray<AvbdSoftComponentFinalizeMode>&
		componentFinalizeModes = workspace.componentFinalizeModes;
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
	const auto findComponentBodyIndex =
		[softBodies, numSoftBodies](PxU32 particleIndex)
	{
		for(PxU32 bodyIndex = 0;
			bodyIndex < numSoftBodies; bodyIndex++)
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
	};
	const auto mergeComponentFinalizeMode =
		[&componentFinalizeModes, &findComponentBodyIndex](
			PxU32 particleIndex,
			AvbdSoftComponentFinalizeMode incoming)
	{
		const PxU32 bodyIndex =
			findComponentBodyIndex(particleIndex);
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
	};
	const auto compileVelocityObjectives =
		[&compiledVelocityObjectives,
		 &mergeComponentFinalizeMode,
		 &findComponentBodyIndex](
			const AvbdSoftContact* sourceContacts,
			PxU32 sourceContactCount)
	{
		for(PxU32 sourceIndex = 0;
			sourceIndex < sourceContactCount; sourceIndex++)
		{
			const AvbdSoftContact& source =
				sourceContacts[sourceIndex];
			const AvbdSoftContactGeometry& geometry =
				source.geometry;
			AvbdSoftComponentFinalizeMode incoming =
				AvbdSoftComponentFinalizeMode::eUNSUPPORTED;
			if(geometry.velocityOwner ==
				AvbdVelocityObjectiveOwner::PositionAL)
			{
				incoming =
					geometry.hasKinematicRigidTarget()
						? AvbdSoftComponentFinalizeMode::
							eUNSUPPORTED
						: AvbdSoftComponentFinalizeMode::
							ePOSITION_OWNED;
			}
			else if(geometry.velocityOwner ==
				AvbdVelocityObjectiveOwner::ComponentFinalize)
			{
				incoming =
					geometry.hasKinematicRigidTarget()
						? AvbdSoftComponentFinalizeMode::
							eKINEMATIC_CONTACT
						: AvbdSoftComponentFinalizeMode::
							eUNSUPPORTED;
			}
			if(geometry.hasBarycentricQueryPoint())
			{
				for(PxU32 vertexIndex = 0;
					vertexIndex < 3; vertexIndex++)
				{
					if(geometry.queryParticleIndices[vertexIndex] ==
						PX_MAX_U32)
						break;
					mergeComponentFinalizeMode(
						geometry.queryParticleIndices[vertexIndex],
						incoming);
				}
			}
			else
				mergeComponentFinalizeMode(
					geometry.particleIdx, incoming);
			if(geometry.hasDeformableSurfaceTarget())
			{
				for(PxU32 vertexIndex = 0;
					vertexIndex < 3; vertexIndex++)
				{
					mergeComponentFinalizeMode(
						geometry.surfaceParticleIndices[
							vertexIndex],
						geometry.velocityOwner ==
							AvbdVelocityObjectiveOwner::
								PositionAL
							? AvbdSoftComponentFinalizeMode::
								ePOSITION_OWNED
							: AvbdSoftComponentFinalizeMode::
								eUNSUPPORTED);
				}
			}
			if(geometry.velocityOwner !=
					AvbdVelocityObjectiveOwner::
						ComponentFinalize ||
				!geometry.hasKinematicRigidTarget())
				continue;
			const PxU32 bodyIndex =
				findComponentBodyIndex(geometry.particleIdx);
			if(bodyIndex == PX_MAX_U32)
				continue;
			AvbdCompiledSoftVelocityObjective objective;
			objective.owner = geometry.velocityOwner;
			objective.source = geometry.source;
			objective.bodyIndex = bodyIndex;
			objective.particleIndex = geometry.particleIdx;
			for(PxU32 queryVertex = 0;
				queryVertex < 3; queryVertex++)
			{
				objective.queryParticleIndices[queryVertex] =
					geometry.queryParticleIndices[queryVertex];
				objective.queryWeights[queryVertex] =
					geometry.queryWeights[queryVertex];
			}
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
				if(compiled.particleIndex ==
						objective.particleIndex &&
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
	};
	compileVelocityObjectives(contacts, numContacts);
	// A persistent contact carries AL/friction state across frames, but its
	// finite depenetration bias is a one-frame target.
	avbdResetSoftContactDepenetrationLimits(
		contacts, numContacts);

	PxReal invDt = dt > 0.0f ? 1.0f / dt : 0.0f;
	PxReal invDtSq = invDt * invDt;

	// Stage 1: prediction
	for (PxU32 i = 0; i < numParticles; i++)
	{
		particles[i].computePrediction(dt, gravity);
		// Reset elastic proximal weight for new timestep
		// (warmstart: retain a fraction from prior timestep for stability)
		particles[i].elasticK = particles[i].elasticK * 0.5f;
	}
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
		compileVelocityObjectives(contacts, numContacts);
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
				AvbdSoftComponentFinalizeMode::ePOSITION_OWNED ||
			componentFinalizeModes[bodyIndex] ==
				AvbdSoftComponentFinalizeMode::eUNSUPPORTED)
			continue;
		PxVec3 centroid(0.0f);
		PxMat33 inertia(PxZero);
		target.valid = avbdComputeSoftComponentMomentum(
			particles, numParticles, softBodies[bodyIndex],
			true, invDt, centroid, target.linearMomentum,
			target.angularMomentum, inertia, target.mass);
		PX_UNUSED(centroid);
		PX_UNUSED(inertia);
	}
	if(stepStats)
		stepStats->predictionMs += stageTimer.getElapsedSeconds() * 1000.0;

	// Build per-particle contact index to avoid O(particles*contacts) scan.
	// contactStart[pi] = first index into contactIdx for particle pi.
	// contactIdx stores contact indices grouped by particle.
	workspace.resize(workspace.contactStarts, numParticles + 1);
	workspace.resize(workspace.contactCounts, numParticles);
	PxArray<AvbdSoftContactParticleRef>& contactIdxBuf =
		workspace.contactIndices;
	PxArray<PxU32>& contactStart = workspace.contactStarts;
	PxArray<PxU32>& contactCount = workspace.contactCounts;
	auto buildContactIndex = [&]()
	{
		for (PxU32 i = 0; i < numParticles; i++) contactCount[i] = 0;
		for (PxU32 ci = 0; ci < numContacts; ci++)
		{
			const AvbdSoftContactGeometry& geometry =
				contacts[ci].geometry;
			PxU32 particleIndices[6];
			const PxU32 particleIndexCount =
				avbdCollectSoftContactParticleIndices(
					geometry, particleIndices);
			for(PxU32 i = 0; i < particleIndexCount; i++)
			{
				const PxU32 particleIndex = particleIndices[i];
				if(particleIndex >= numParticles)
					continue;
				if(PxAbs(avbdGetSoftContactParticleJacobianScale(
					geometry, particleIndex)) > 1e-12f)
					contactCount[particleIndex]++;
			}
		}
		contactStart[0] = 0;
		for (PxU32 i = 0; i < numParticles; i++)
			contactStart[i + 1] = contactStart[i] + contactCount[i];
		workspace.resize(contactIdxBuf, contactStart[numParticles]);
		for (PxU32 i = 0; i < numParticles; i++) contactCount[i] = 0;
		for (PxU32 ci = 0; ci < numContacts; ci++)
		{
			const AvbdSoftContactGeometry& geometry =
				contacts[ci].geometry;
			PxU32 particleIndices[6];
			const PxU32 particleIndexCount =
				avbdCollectSoftContactParticleIndices(
					geometry, particleIndices);
			for(PxU32 i = 0; i < particleIndexCount; i++)
			{
				const PxU32 particleIndex = particleIndices[i];
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
					AvbdSoftContactParticleRef(
						ci, jacobianScale);
			}
		}
	};
	buildContactIndex();

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
	for (PxU32 outerIt = 0; outerIt < outerIterations; outerIt++)
	{
		if(stepStats)
			stepStats->executedOuterIterations++;
		const PxU32 remainingOuterIterations =
			outerIterations - outerIt;
		const PxU32 currentInnerIterations =
			(remainingInnerIterationBudget +
				remainingOuterIterations - 1) /
			remainingOuterIterations;
		remainingInnerIterationBudget -= currentInnerIterations;
		// Snapshot positions as proximal anchor for AVBD elastic term
		for (PxU32 i = 0; i < numParticles; i++)
		{
			particles[i].outerPosition = particles[i].position;
			selfCollisionSafetyBounds[i] = PX_MAX_F32;
		}

		// OGC Eq. 21-27: each outer redetection stage records a known
		// penetration-free anchor and computes a per-vertex conservative
		// displacement radius.  Every primal update below is kept inside
		// that radius until the next redetection.
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
					bodySelfCollisionSafetyBounds);
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
			AvbdSoftSweepConvergenceObservation sweepObservation;

			// Soft particle primal.  Preserve the reference vertex-block
			// nonlinear Gauss-Seidel ordering until a colored schedule has its
			// own equivalence gate.
			for (PxU32 si = 0; si < numSoftBodies; si++)
			{
				const AvbdSoftBody& sb = softBodies[si];
				for (PxU32 li = 0; li < sb.compiled.particleCount; li++)
				{
					PxU32 pi = sb.compiled.particleStart + li;
					AvbdSoftParticle& sp = particles[pi];
					if (sp.isStatic()) continue;

					// Inertial term
					PxReal m_dtSq = sp.mass * invDtSq;
					PxMat33 H = PxMat33::createDiagonal(PxVec3(m_dtSq));
					PxVec3 f = (sp.predictedPosition - sp.position) * m_dtSq;

					const AvbdParticleElementAdjacency& elementAdjacency =
						sb.compiled.elementAdjacency[li];
					const AvbdParticleObjectiveAdjacency& objectiveAdjacency =
						sb.runtime.objectiveAdjacency[li];
					static const PxU32 eMAX_CACHED_TET_INCIDENCE = 64;
					AvbdTetVertexLinearization tetLinearizations[
						eMAX_CACHED_TET_INCIDENCE];
					const PxU32 tetIncidenceCount =
						elementAdjacency.tetRefs.size();
					const bool cacheTetLinearizations =
						tetIncidenceCount <=
							eMAX_CACHED_TET_INCIDENCE;
					if(stepStats && !cacheTetLinearizations)
					{
						stepStats->
							tetLinearizationCacheFallbackParticleSteps++;
					}

					// Triangle (StVK) contributions
					for (PxU32 ti = 0; ti < elementAdjacency.triRefs.size(); ti++)
					{
						const AvbdParticleElementRef& ref =
							elementAdjacency.triRefs[ti];
						PxVec3 ef; PxMat33 eH;
						avbdEvaluateStVKForceHessian(
							sb.compiled.triElements[ref.index], int(ref.vOrder),
							sb.material.mu, sb.material.lambda, particles, ef, eH);
						f = f + ef;
						H = H + eH;
					}

					// Tetrahedral material-model contributions
					for (PxU32 ti = 0; ti < elementAdjacency.tetRefs.size(); ti++)
					{
						const AvbdParticleElementRef& ref =
							elementAdjacency.tetRefs[ti];
						PxVec3 ef; PxMat33 eH;
						if(sb.material.coRotationalVolumeModel)
							avbdEvaluateCorotationalForceHessianPrepared(
								sb.compiled.tetElements[ref.index],
								int(ref.vOrder),
								sb.material.mu, sb.material.lambda,
								particles, ef, eH,
								cacheTetLinearizations
									? &tetLinearizations[ti]
									: NULL);
						else
							avbdEvaluateNeoHookeanForceHessianPrepared(
								sb.compiled.tetElements[ref.index],
								int(ref.vOrder),
								sb.material.mu, sb.material.lambda,
								sb.material.neoHookeanAlpha,
								particles, ef, eH,
								cacheTetLinearizations
									? &tetLinearizations[ti]
									: NULL);
						f = f + ef;
						H = H + eH;
					}

					// Bending contributions
					for (PxU32 bi = 0; bi < elementAdjacency.bendRefs.size(); bi++)
					{
						const AvbdParticleElementRef& ref =
							elementAdjacency.bendRefs[bi];
						PxVec3 ef; PxMat33 eH;
						avbdEvaluateBendingForceHessian(
							sb.compiled.bendElements[ref.index], int(ref.vOrder),
							sb.material.bendingStiffness, particles, ef, eH);
						f = f + ef;
						H = H + eH;
					}

					// Contact contributions (indexed lookup)
					for (PxU32 k = contactStart[pi];
						k < contactStart[pi + 1]; k++)
					{
						PxVec3 cf; PxMat33 cH;
						const AvbdSoftContactParticleRef& contactRef =
							contactIdxBuf[k];
						const AvbdSoftContact& contact =
							contacts[contactRef.contactIndex];
						avbdEvaluateContactParticleBlock(
							contact.geometry, contact.state,
							particles, contactRef.jacobianScale,
							cf, cH);
						f = f + cf;
						H = H + cH;
					}

					// Scene-external component supports only compiled
					// one-way pin owners. Rigid attachments require the
					// low-level rigid-body block and must never be consumed
					// as a one-way particle-only objective here.
					for (PxU32 oi = 0;
						oi < objectiveAdjacency.objectiveIndices.size(); oi++)
					{
						const PxU32 objectiveIndex =
							objectiveAdjacency.objectiveIndices[oi];
						const AvbdCompiledSoftObjective& objective =
							sb.runtime.compiledObjectives[objectiveIndex];
						if (!avbdIsPinPositionOwner(
							objective.owner))
						{
							PX_ASSERT(
								avbdIsPinPositionOwner(
									objective.owner));
							continue;
						}
						PxVec3 pf; PxMat33 pH;
						avbdEvaluatePinForceHessian(
							objective.point,
							sb.runtime.pins[objective.runtimeStateIndex],
							particles, pi, pf, pH);
						f = f + pf;
						H = H + pH;
					}

					// Stiffness-proportional Rayleigh damping (Newton VBD style):
					// Per-axis damping proportional to elastic stiffness, clamped so no
					// axis gets less damping than mass-proportional (baseline stability).
					if (sp.damping > 0.0f)
					{
						PxReal dampCoeff = sp.damping * sp.mass * invDt;
						PxReal he_xx =
							PxMax(H.column0.x - m_dtSq, 0.0f);
						PxReal he_yy =
							PxMax(H.column1.y - m_dtSq, 0.0f);
						PxReal he_zz =
							PxMax(H.column2.z - m_dtSq, 0.0f);
						PxReal trHe = he_xx + he_yy + he_zz;
						PxReal dx, dy, dz;
						if (trHe > 1e-10f)
						{
							PxReal s = dampCoeff * 3.0f / trHe;
							dx = PxMax(he_xx * s, dampCoeff);
							dy = PxMax(he_yy * s, dampCoeff);
							dz = PxMax(he_zz * s, dampCoeff);
						}
						else
						{
							dx = dy = dz = dampCoeff;
						}
						const PxVec3 dampingDisplacement =
							sp.position - sp.initialPosition;
						f.x -= dx * dampingDisplacement.x;
						f.y -= dy * dampingDisplacement.y;
						f.z -= dz * dampingDisplacement.z;
						H.column0.x += dx;
						H.column1.y += dy;
						H.column2.z += dz;
					}

					// AVBD elastic proximal term: pulls toward the
					// outer-iteration anchor.
					if (sp.elasticK > 0.0f)
					{
						H.column0.x += sp.elasticK;
						H.column1.y += sp.elasticK;
						H.column2.z += sp.elasticK;
						f = f + (sp.outerPosition - sp.position) * sp.elasticK;
					}

					// H^-1 f is the local, length-valued stationarity
					// measure. Keep it separate from trust-region and
					// positive-J feasibility limiting: a rejected step can
					// be zero without the local objective being stationary.
					const PxVec3 localSolveDisplacement =
						avbdSolveSymmetric33(H, f);
					const PxReal localSolveDisplacementSq =
						localSolveDisplacement.magnitudeSquared();
					PxVec3 proposedDisplacement =
						localSolveDisplacement;
					bool trustRegionLimited = false;
					const PxReal maxDx = 1.0f;
					AvbdSoftTetDisplacementLimitResult limitResult;
					if(localSolveDisplacement.isFinite() &&
						PxIsFinite(localSolveDisplacementSq))
					{
						if(localSolveDisplacementSq > maxDx * maxDx)
						{
							proposedDisplacement *=
								maxDx /
								PxSqrt(localSolveDisplacementSq);
							trustRegionLimited = true;
						}
						limitResult = cacheTetLinearizations
							? avbdLimitTetDisplacementFromLinearizations(
								proposedDisplacement,
								tetLinearizations,
								tetIncidenceCount)
							: avbdLimitTetDisplacementObserved(
								sb, pi, particles,
								proposedDisplacement);
					}
					else
					{
						limitResult =
							AvbdSoftTetDisplacementLimitResult(
								PxVec3(0.0f), 0.0f,
								AvbdSoftTetDisplacementLimitReason::
									eNONFINITE_REJECTED);
					}
					const PxVec3 positionBeforeStep = sp.position;
					if(limitResult.appliedDisplacement.isFinite())
					{
						sp.position +=
							limitResult.appliedDisplacement;
						const PxVec3 positionBeforeOgc =
							sp.position;
						avbdTruncateDisplacement(
							sp, sp.outerPosition,
							selfCollisionSafetyBounds[pi]);
						if((sp.position - positionBeforeOgc).
								magnitudeSquared() > 1.0e-20f)
							trustRegionLimited = true;
						limitResult.appliedDisplacement =
							sp.position - positionBeforeStep;
					}
					sweepObservation.observe(
						localSolveDisplacement,
						trustRegionLimited, limitResult);
				}
			}
			maxDxSq =
				sweepObservation.maxAppliedDisplacementSq;
			if(stepStats)
			{
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
		// proportional to displacement from the outer-iteration anchor
		for (PxU32 i = 0; i < numParticles; i++)
		{
			AvbdSoftParticle& sp = particles[i];
			if (sp.isStatic()) continue;
			PxReal disp = (sp.position - sp.outerPosition).magnitude();
			sp.elasticK = PxMin(sp.elasticK + avbdBeta * disp, sp.elasticKMax);
		}
		if(stepStats)
			stepStats->dualMs +=
				stageTimer.getElapsedSeconds() * 1000.0;

		// Re-detect contacts between outer iterations so surface anchors
		// track the deforming geometry instead of going stale.
		if (redetectFn && contactsArray && outerIt + 1 < outerIterations)
		{
			redetectFn(particles, numParticles, softBodies, numSoftBodies,
					   *contactsArray, redetectUserData);
			contacts = contactsArray->begin();
			numContacts = contactsArray->size();
			compileVelocityObjectives(
				contacts, numContacts);
			// Matching rows retain the original frame anchor through state
			// transfer; only contacts born at this redetection are initialized.
			avbdInitializeSoftContactDepenetrationLimits(
				contacts, numContacts, particles,
				softBodies, numSoftBodies, dt);
			// Rebuild per-particle contact index
			buildContactIndex();
		}
		if(stepStats)
			stepStats->redetectMs +=
				stageTimer.getElapsedSeconds() * 1000.0;
	}

	// Stage 3: velocity update
	for (PxU32 i = 0; i < numParticles; i++)
		particles[i].updateVelocityFromPosition(invDt);
	avbdApplyBendingDamping(
		particles, softBodies, numSoftBodies, dt);
	avbdFinalizeSoftComponentVelocities(
		particles, numParticles,
		softBodies, numSoftBodies,
		componentMomentumTargets.begin(),
		componentFinalizeModes.begin(),
		compiledVelocityObjectives.begin(),
		compiledVelocityObjectives.size(), invDt);
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
		stepStats->contactOutputGrowthEvents =
			workspace.contact.outputGrowthEvents;
		stepStats->contactOutputGrowthBytes =
			workspace.contact.outputGrowthBytes;
	}

}

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_COMPONENT_H
