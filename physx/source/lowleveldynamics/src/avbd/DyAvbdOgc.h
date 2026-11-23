// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_OGC_H
#define DY_AVBD_OGC_H

#include "foundation/PxMath.h"
#include "DyAvbdOgcActivation.h"
#include "foundation/PxMathUtils.h"
#include "foundation/PxInlineArray.h"
#include "avbd/DyAvbdOgcProjection.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxQuat.h"
#include "foundation/PxVec3.h"

namespace physx
{
namespace Dy
{

// Offset Geometric Contact (SIGGRAPH 2025) has two inseparable pieces:
// an offset contact energy and a DCD-derived local conservative region.  The
// latter replaces CCD step clipping.  A block is always measured from the pose
// at which its last DCD bounds were compiled; it is never measured from the
// preceding GS iterate and never promoted to a connected-component alpha.

struct AvbdOGCParams
{
	PxReal contactRadius;
	PxReal contactStiffness;
	PxReal safetyRelax;
	PxReal refreshFraction;
	PxReal tau;

	AvbdOGCParams()
		: contactRadius(0.04f)
		, contactStiffness(1.0e5f)
		, safetyRelax(0.45f)
		, refreshFraction(0.01f)
		, tau(-1.0f)
	{
	}

	PxReal getTau() const
	{
		return (tau < 0.0f) ? contactRadius * 0.5f : tau;
	}
};

// OGC Eq. 21.  The detector may cap every distance by its finite query radius;
// the resulting smaller bound remains conservative.
PX_FORCE_INLINE PxReal avbdOgcConservativeBound(PxReal vertexFacetDistance,
	PxReal incidentEdgeDistance, PxReal incidentFacetDistance, PxReal gamma)
{
	const PxReal minimumDistance = PxMin(vertexFacetDistance,
		PxMin(incidentEdgeDistance, incidentFacetDistance));
	if(!PxIsFinite(minimumDistance) || minimumDistance <= 0.0f)
		return 0.0f;
	return PxClamp(gamma, 0.0f, 0.499f) * minimumDistance;
}

PX_FORCE_INLINE PxReal avbdOgcQueryRadius(PxReal contactRadius,
	PxReal inertialDisplacement)
{
	return PxMax(0.0f, contactRadius) + PxMax(0.0f, inertialDisplacement);
}

// The directional regions below are geometry-specific certificates, not the
// general mesh guarantee of Eq.21/27. That guarantee bounds full vertex
// displacement using VF, EE and incident-face distances. A half-space is
// sufficient only when its compiler proves complete feature separation.
// partnerBody is the live rigid endpoint;
// partnerCache/partnerVertices/partnerWeights realize the same DCD witness
// in the triangle already committed by particle-first GS.  Do not bake pHat:
// that lets a sitting box keep falling with gravity while the mesh holds.
static const PxU32 AVBD_OGC_INLINE_PLANES = 8;

struct AvbdOgcHalfSpace
{
	PxVec3 n;
	PxReal bound;
	PxU32 partnerBody;
	PxVec3 partnerAnchor;
	PxVec3 partnerDelta;
	PxU32 partnerCache;
	PxU32 partnerVertices[3];
	PxVec3 partnerWeights;
	// Rigid Eq.21 is a bound on the realizing feature, not COM.  Particle
	// planes leave this at 0 so the candidate is the point itself.
	PxVec3 selfLocalPoint;
};

typedef PxInlineArray<AvbdOgcHalfSpace, AVBD_OGC_INLINE_PLANES> AvbdOgcPlanes;

struct AvbdOgcPointRegion
{
	PxVec3 anchor;
	// Isotropic envelope used by Surface vertex/edge DCD.  Volume leaves this
	// at PX_MAX_F32 and uses half-spaces.  Surface triangle/core adds the
	// same static half-space certificate; it does not become Volume Eq.21.
	PxReal bound;
	AvbdOgcPlanes planes;
	PxU32 planeCount;
};

PX_FORCE_INLINE void avbdOgcResetPointRegion(AvbdOgcPointRegion& region,
	const PxVec3& anchor)
{
	region.anchor = anchor;
	region.bound = PX_MAX_F32;
	region.planeCount = 0;
	region.planes.clear();
}

PX_INLINE AvbdOgcPointRegion* avbdOgcCreatePointRegions(PxU32 count)
{
	AvbdOgcPointRegion* regions = static_cast<AvbdOgcPointRegion*>(
		PX_ALLOC(sizeof(AvbdOgcPointRegion) * count, "AvbdOgcPointRegions"));
	if(regions)
		for(PxU32 i = 0; i < count; ++i)
			PX_PLACEMENT_NEW(regions + i, AvbdOgcPointRegion)();
	return regions;
}

PX_INLINE void avbdOgcDestroyPointRegions(AvbdOgcPointRegion* regions, PxU32 count)
{
	if(regions)
		for(PxU32 i = 0; i < count; ++i)
			regions[i].~AvbdOgcPointRegion();
	PX_FREE(regions);
}

PX_FORCE_INLINE void avbdOgcWriteHalfSpace(AvbdOgcHalfSpace& plane,
	const PxVec3& nn, PxReal bound, PxU32 partnerBody,
	const PxVec3& partnerAnchor, const PxVec3& partnerDelta,
	PxU32 partnerCache, PxU32 v0, PxU32 v1, PxU32 v2,
	const PxVec3& selfLocalPoint = PxVec3(0.0f),
	const PxVec3& partnerWeights = PxVec3(1.0f / 3.0f))
{
	plane.n = nn;
	plane.bound = bound;
	plane.partnerBody = partnerBody;
	plane.partnerAnchor = partnerAnchor;
	plane.partnerDelta = partnerDelta;
	plane.partnerCache = partnerCache;
	plane.partnerVertices[0] = v0;
	plane.partnerVertices[1] = v1;
	plane.partnerVertices[2] = v2;
	plane.partnerWeights = partnerWeights;
	plane.selfLocalPoint = selfLocalPoint;
}

// An embedded collision point moves by sum_i(w_i * delta_i).  Splitting the
// feature certificate n.dot(sum_i(w_i * delta_i)) >= -featureBound across
// independently solved particle blocks therefore has to retain the sign of
// every embedding weight.  If each support satisfies
//
//   sign(w_i) * n.dot(delta_i) >= -featureBound / sum_j(abs(w_j)),
//
// multiplication by abs(w_i) and summation recovers the feature certificate.
// Using n for a negative support reverses this implication and can admit an
// embedded collision vertex through the obstacle while all support domains
// report feasible.
PX_FORCE_INLINE bool avbdOgcEmbeddedSupportHalfSpace(const PxVec3& normal,
	PxReal featureBound, PxReal absoluteWeightSum, PxReal weight,
	PxVec3& supportNormal, PxReal& supportBound)
{
	if(!normal.isFinite() || normal.magnitudeSquared() <= 1.0e-20f ||
		!PxIsFinite(featureBound) || featureBound < 0.0f ||
		!PxIsFinite(absoluteWeightSum) || absoluteWeightSum <= 1.0e-10f ||
		!PxIsFinite(weight) || PxAbs(weight) <= 1.0e-10f)
		return false;
	supportNormal = normal.getNormalized() * (weight > 0.0f ? 1.0f : -1.0f);
	supportBound = featureBound / absoluteWeightSum;
	return PxIsFinite(supportBound);
}

PX_FORCE_INLINE void avbdOgcAddHalfSpace(AvbdOgcPlanes& planes,
	PxU32& planeCount, const PxVec3& n, PxReal bound, PxU32 partnerBody,
	const PxVec3& partnerAnchor, const PxVec3& partnerDelta,
	PxU32 partnerCache = PX_MAX_U32, PxU32 v0 = PX_MAX_U32,
	PxU32 v1 = PX_MAX_U32, PxU32 v2 = PX_MAX_U32,
	const PxVec3& selfLocalPoint = PxVec3(0.0f),
	const PxVec3& partnerWeights = PxVec3(1.0f / 3.0f))
{
	if(!n.isFinite() || n.magnitudeSquared() <= 1.0e-20f ||
		!PxIsFinite(bound) || bound < 0.0f)
		return;
	const PxVec3 nn = n.getNormalized();
	if(planeCount == 0)
		planes.clear();
	for(PxU32 i = 0; i < planeCount; ++i)
	{
		if(planes[i].n == nn && planes[i].selfLocalPoint == selfLocalPoint &&
			planes[i].partnerBody == partnerBody &&
			planes[i].partnerCache == partnerCache &&
			planes[i].partnerVertices[0] == v0 &&
			planes[i].partnerVertices[1] == v1 &&
			planes[i].partnerVertices[2] == v2 &&
			planes[i].partnerWeights == partnerWeights &&
			planes[i].partnerAnchor == partnerAnchor &&
			planes[i].partnerDelta == partnerDelta)
		{
			if(bound < planes[i].bound)
				avbdOgcWriteHalfSpace(planes[i], nn, bound, partnerBody,
					partnerAnchor, partnerDelta, partnerCache, v0, v1, v2,
					selfLocalPoint, partnerWeights);
			return;
		}
	}
	AvbdOgcHalfSpace plane;
	avbdOgcWriteHalfSpace(plane, nn, bound, partnerBody,
		partnerAnchor, partnerDelta, partnerCache, v0, v1, v2, selfLocalPoint, partnerWeights);
	planes.pushBack(plane);
	planeCount = planes.size();
}

// Interpolate displacement of the DCD witness, not absolute positions minus
// a differently sampled anchor.  This is exactly zero at the DCD epoch.
PX_FORCE_INLINE PxVec3 avbdOgcTriangleWitnessDelta(const PxVec3& weights,
	const PxVec3& d0, const PxVec3& d1, const PxVec3& d2)
{
	return weights.x * d0 + weights.y * d1 + weights.z * d2;
}

// truncated: the candidate was mapped onto the certificate.
// spent: the isotropic envelope was exhausted and DCD must recompile.
// A half-space clip is the contact constraint being active, not a spent
// ball.  Treating it as spent rebuilds the manifold on the crushed pose
// every sweep and publishes one or two frames of flattened garbage.
struct AvbdOgcClip
{
	bool truncated;
	bool spent;
	bool feasible;
};

PX_INLINE void avbdOgcPointConstraints(const AvbdOgcPointRegion& region,
    const PxVec3& candidate, AvbdOgcProjectionPlanes& constraints)
{
    for(PxU32 i = 0; i < region.planeCount; ++i)
    {
        const AvbdOgcHalfSpace& p = region.planes[i];
        AvbdOgcProjectionPlane c;
        c.n = PxVec3d(p.n.x, p.n.y, p.n.z);
        c.b = c.n.dot(PxVec3d(p.partnerDelta.x, p.partnerDelta.y, p.partnerDelta.z)) - p.bound;
        // Input and published poses are float. Account for their subtraction
        // and dot-product rounding; this is not a penetration tolerance.
        c.roundoff = 4.0 * FLT_EPSILON * double(p.n.abs().dot(
            region.anchor.abs() + candidate.abs() + p.partnerDelta.abs()) + PxAbs(p.bound));
        constraints.pushBack(c);
    }
}

PX_INLINE AvbdOgcClip avbdOgcTruncatePoint(const AvbdOgcPointRegion& region,
    PxVec3& candidate, const AvbdOgcProjectionPlanes* additional = NULL,
    const PxMat33* metric = NULL,const PxVec3* feasiblePoint = NULL)
{
    AvbdOgcClip clip = { false, false, true };
    AvbdOgcProjectionPlanes constraints;
    avbdOgcPointConstraints(region, candidate, constraints);
    if(additional)
        for(PxU32 i=0;i<additional->size();++i) constraints.pushBack((*additional)[i]);
    const PxVec3d delta = avbdOgcDouble(candidate) - avbdOgcDouble(region.anchor);
    if(avbdOgcRegionContains(constraints, delta, region.bound)) return clip;
    PxVec3d projected;
    PxMat33d H(PxIdentity);
    if(metric) H=PxMat33d(avbdOgcDouble(metric->column0),avbdOgcDouble(metric->column1),avbdOgcDouble(metric->column2));
    AvbdOgcProjectionPlanes solveDomain=constraints;
    avbdOgcCompactParallelPlanes(solveDomain);
    clip.feasible = avbdOgcProjectRegion(solveDomain, delta, region.bound, projected, &H);
    if(!clip.feasible)
    {
        // The caller may provide the already committed point as a constructive
        // feasibility witness. A rank-deficient active set can fail to find the
        // metric projection even though that point proves the domain nonempty.
        // Reject the proposed move and retain the witness; do not turn a local
        // numerical solve failure into a false geometry failure.
        if(feasiblePoint)
        {
            const PxVec3d feasible = avbdOgcDouble(*feasiblePoint) -
                avbdOgcDouble(region.anchor);
            if(avbdOgcRegionContains(constraints, feasible, region.bound))
            {
                clip.feasible = true;
                clip.truncated = candidate != *feasiblePoint;
                clip.spent = region.bound < PX_MAX_F32 * 0.5f &&
                    delta.magnitude() > region.bound;
                candidate = *feasiblePoint;
            }
        }
        return clip;
    }
    PxVec3 result;
    clip.feasible=avbdOgcPublishRegionPoint(region.anchor,projected,constraints,region.bound,result,feasiblePoint);
    if(!clip.feasible) return clip;
    clip.truncated = result != candidate;
    clip.spent = region.bound < PX_MAX_F32 * 0.5f && delta.magnitude() > region.bound;
    candidate = result;
    return clip;
}

PX_INLINE bool avbdOgcPointContains(const AvbdOgcPointRegion& region,
    const PxVec3& candidate)
{
    AvbdOgcProjectionPlanes constraints;
    avbdOgcPointConstraints(region,candidate,constraints);
    return avbdOgcRegionContains(constraints,
        avbdOgcDouble(candidate)-avbdOgcDouble(region.anchor),region.bound);
}

// Largest alpha in [0,1] for which current + alpha*step remains inside the
// point region.  This is an analytic trust-region clip, not a geometric sweep.
PX_FORCE_INLINE PxReal avbdOgcPointStepFraction(
	const AvbdOgcPointRegion& region, const PxVec3& current,
	const PxVec3& step)
{
	if(!current.isFinite() || !step.isFinite() || !PxIsFinite(region.bound) ||
		region.bound < 0.0f)
		return 0.0f;
	const PxVec3 offset = current - region.anchor;
	const PxReal a = step.magnitudeSquared();
	if(a <= 1.0e-20f)
		return 1.0f;
	const PxReal c = offset.magnitudeSquared() - region.bound * region.bound;
	if(c > 1.0e-6f * PxMax(1.0f, region.bound * region.bound))
		return 0.0f;
	const PxReal b = 2.0f * offset.dot(step);
	const PxReal discriminant = b * b - 4.0f * a * c;
	if(discriminant <= 0.0f)
		return c <= 0.0f ? 1.0f : 0.0f;
	const PxReal exit = (-b + PxSqrt(discriminant)) / (2.0f * a);
	return PxClamp(exit, 0.0f, 1.0f);
}

struct AvbdOgcRigidRegion
{
	PxVec3 anchorPosition;
	PxQuat anchorRotation;
	// Maximum distance of any collision feature from the body origin.
	PxReal supportRadius;
	PxReal bound;
	AvbdOgcPlanes planes;
	PxU32 planeCount;
};

PX_FORCE_INLINE PxReal avbdOgcRotationAngle(const PxQuat& a, const PxQuat& b)
{
	// The scalar dot loses small represented rotations when cos(theta/2)
	// rounds to one. The relative imaginary part retains their first order.
	const PxQuat relative=b*a.getConjugate();
	return 2.0f*PxAtan2(relative.getImaginaryPart().magnitude(),PxAbs(relative.w));
}

// ||dp|| + R||dtheta|| majorizes the displacement of every point on a rigid
// shape.  Applying the same local OGC region to this SE(3) block therefore
// preserves the per-feature bound without linearizing rotation as translation.
PX_FORCE_INLINE PxReal avbdOgcRigidSurfaceMotion(
	const AvbdOgcRigidRegion& region, const PxVec3& position,
	const PxQuat& rotation)
{
	return (position - region.anchorPosition).magnitude() +
		PxMax(0.0f, region.supportRadius) *
		avbdOgcRotationAngle(region.anchorRotation, rotation);
}

PX_INLINE AvbdOgcClip avbdOgcTruncateRigid(const AvbdOgcRigidRegion& region,
    PxVec3& resultPosition, PxQuat& resultRotation)
{
    AvbdOgcClip clip = { false, false, true };
    PxVec3 candidatePosition=resultPosition;
    PxQuat candidateRotation=resultRotation;
    const PxReal motion = avbdOgcRigidSurfaceMotion(region, candidatePosition, candidateRotation);
    if(!PxIsFinite(motion) || !PxIsFinite(region.bound) || region.bound < 0.0f)
    { clip.feasible = false; return clip; }
    // Preserve the established isotropic SE(3) trust-region map before solving
    // the affine translation constraints at this candidate rotation.
    if(region.bound < PX_MAX_F32 * 0.5f && motion > region.bound && motion > 1.0e-20f)
    {
        const PxReal alpha = region.bound / motion;
        candidatePosition = region.anchorPosition + (candidatePosition - region.anchorPosition) * alpha;
        candidateRotation = PxSlerp(alpha, region.anchorRotation, candidateRotation).getNormalized();
        clip.truncated = clip.spent = true;
    }
    AvbdOgcProjectionPlanes constraints;
    for(PxU32 i = 0; i < region.planeCount; ++i)
    {
        const AvbdOgcHalfSpace& p = region.planes[i];
        const PxVec3 rotationDelta = candidateRotation.rotate(p.selfLocalPoint) -
            region.anchorRotation.rotate(p.selfLocalPoint);
        const PxVec3 partner = p.partnerDelta - rotationDelta;
        AvbdOgcProjectionPlane c;
        c.n = PxVec3d(p.n.x, p.n.y, p.n.z);
        c.b = c.n.dot(PxVec3d(partner.x, partner.y, partner.z)) - p.bound;
        c.roundoff = 4.0 * FLT_EPSILON * double(p.n.abs().dot(
            region.anchorPosition.abs() + candidatePosition.abs() +
            p.partnerDelta.abs() + rotationDelta.abs()) + PxAbs(p.bound));
        constraints.pushBack(c);
    }
    const double radius = region.bound >= PX_MAX_F32 * 0.5f ? double(PX_MAX_F32) :
        PxMax(0.0, double(region.bound) - double(PxMax(0.0f, region.supportRadius)) *
            avbdOgcRotationAngle(region.anchorRotation, candidateRotation));
    const PxVec3d delta = avbdOgcDouble(candidatePosition) - avbdOgcDouble(region.anchorPosition);
    PxVec3d projected;
    clip.feasible = avbdOgcProjectRegion(constraints, delta, radius, projected);
    if(!clip.feasible) return clip;
    PxVec3 result;
    clip.feasible=avbdOgcPublishRegionPoint(region.anchorPosition,projected,constraints,radius,result);
    if(!clip.feasible) return clip;
    clip.feasible = avbdOgcRegionContains(constraints,
        avbdOgcDouble(result)-avbdOgcDouble(region.anchorPosition),radius);
    if(!clip.feasible) return clip;
    clip.truncated |= result != candidatePosition;
    resultPosition = result;
    resultRotation = candidateRotation;
    return clip;
}

PX_INLINE bool avbdOgcRigidContains(const AvbdOgcRigidRegion& region,
    const PxVec3& position,const PxQuat& rotation)
{
    const double angularMotion=double(PxMax(0.0f,region.supportRadius))*
        avbdOgcRotationAngle(region.anchorRotation,rotation);
    if(region.bound<double(PX_MAX_F32)*0.5 && angularMotion>region.bound) return false;
    AvbdOgcPointRegion point;
    avbdOgcResetPointRegion(point,region.anchorPosition);
    point.bound=region.bound>=PX_MAX_F32*0.5f ? PX_MAX_F32 : PxReal(region.bound-angularMotion);
    point.planes=region.planes;
    point.planeCount=region.planeCount;
    for(PxU32 i=0;i<point.planeCount;++i)
        point.planes[i].partnerDelta-=
            rotation.rotate(point.planes[i].selfLocalPoint)-
            region.anchorRotation.rotate(point.planes[i].selfLocalPoint);
    return avbdOgcPointContains(point,position);
}

} // namespace Dy
} // namespace physx

#endif
