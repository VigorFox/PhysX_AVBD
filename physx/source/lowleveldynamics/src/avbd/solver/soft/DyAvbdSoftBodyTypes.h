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
#ifndef DY_AVBD_SOFT_BODY_TYPES_H
#define DY_AVBD_SOFT_BODY_TYPES_H

#include "foundation/PxMathUtils.h"
#include "foundation/PxMat33.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxVec3.h"

namespace physx
{
namespace Dy
{

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

// Collision vertices are embedded in one simulation tetrahedron, while a
// point on a collision triangle can span three different tetrahedra.  Keep
// the fully expanded point in the prepared contact IR so every solver path
// consumes the same exact Jacobian instead of projecting the point back to a
// single simulation element.
static const PxU32 AVBD_EMBEDDED_VERTEX_SUPPORT = 4;
static const PxU32 AVBD_CONTACT_MAX_PARTICLES = 24;
} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_TYPES_H
