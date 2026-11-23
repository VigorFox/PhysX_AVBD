// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_RIGID_METRIC_H
#define DY_AVBD_RIGID_METRIC_H

#include "foundation/PxMat33.h"
#include "foundation/PxQuat.h"
#include <cmath>

namespace physx
{
namespace Dy
{
PX_FORCE_INLINE PxQuat avbdStoredRotationIncrement(const PxQuat& from,const PxVec3d& delta)
{
    if(delta.isZero())return from;
    const PxQuatd q(from.x,from.y,from.z,from.w),dq(delta.x*.5,delta.y*.5,delta.z*.5,0);
    const PxQuatd result=(q+dq*q).getNormalized();
    return PxQuat(PxReal(result.x),PxReal(result.y),PxReal(result.z),PxReal(result.w));
}
PX_FORCE_INLINE PxVec3d avbdAcceptedRotationDisplacementDouble(const PxQuat& from,const PxQuat& to)
{
    if(from==to || from==-to)return PxVec3d(0);
    const PxQuatd a=PxQuatd(from.x,from.y,from.z,from.w).getNormalized();
    const PxQuatd b=PxQuatd(to.x,to.y,to.z,to.w).getNormalized();PxQuatd d=b*a.getConjugate();
    if(d.w<0)d=-d;const PxVec3d v(d.x,d.y,d.z);const double s=v.magnitude();
    return s>0?v*(2*std::atan2(s,d.w)/s):PxVec3d(0);
}
PX_FORCE_INLINE PxVec3 avbdAcceptedRotationDisplacement(const PxQuat& q0, const PxQuat& q1)
{
	// World logarithm of the actual accepted rotation; this measures its
	// geodesic displacement, not the inverse of the normalized Euler retract.
	PxQuat qerr = q1 * q0.getConjugate();
	if(qerr.w < 0.0f)
		qerr = -qerr;
	const PxReal s = qerr.getImaginaryPart().magnitude();
	if(s == 0.0f)
		return PxVec3(0.0f);
	return qerr.getImaginaryPart() * (2.0f * PxAtan2(s, qerr.w) / s);
}

// A finite positional spring damps material strain, not accumulated world
// displacement projected onto the current tangent. Common finite motion
// leaves C unchanged. The return value uses the existing (k + d/dt) metric.
template<class T> PX_FORCE_INLINE T avbdFiniteSpringResidual(T stiffness,
	T dampingOverDt, T current, T initial, T targetStep)
{
	return (stiffness * current + dampingOverDt * (current - initial - targetStep)) /
		(stiffness + dampingOverDt);
}

PX_FORCE_INLINE PxReal avbdFiniteSpringResidual(PxReal stiffness,
	PxReal dampingOverDt, PxReal current, PxReal initial, PxReal targetStep)
{
	return avbdFiniteSpringResidual<PxReal>(stiffness, dampingOverDt, current, initial, targetStep);
}

template<class T> struct AvbdRotationPotentialT
{
	T energy;
	PxVec3T<T> gradient;
	PxVec3T<T> coordinate,conjugate;
	PxMat33T<T> gaussNewton;
};
typedef AvbdRotationPotentialT<PxReal> AvbdRotationPotential;

// Pull a row covector through r(q)=2 im(q*qReference^-1).
PX_FORCE_INLINE PxVec3 avbdRotationChartGradient(const PxQuat& reference,
	const PxQuat& candidate, const PxVec3& covector)
{
	// At the relinearization pose the chart differential is exactly I.
	if(reference==candidate || reference==-candidate) return covector;
	PxQuat error=candidate*reference.getConjugate();
	if(error.w<0.0f) error=-error;
	return covector*error.w+error.getImaginaryPart().cross(covector);
}

// The existing inertial residual is r = 2 im(q*qHat^-1).  For a world
// left increment its derivative is J = w I - [im(q*qHat^-1)]_x, not I.
// Use J^T I r and the PSD metric J^T I J; line search checks the true energy.
template<class T> PX_FORCE_INLINE AvbdRotationPotentialT<T> avbdRotationInertialPotential(
	const PxQuatT<T>& target, const PxQuatT<T>& candidate, const PxMat33T<T>& inertia)
{
	PxQuatT<T> error = candidate * target.getConjugate();
	if(error.w < 0.0f) error = -error;
	const PxVec3T<T> v(error.x, error.y, error.z);
	const PxVec3T<T> residual = v * T(2);
	const PxMat33T<T> jacobian(PxVec3T<T>(error.w, -v.z, v.y),
		PxVec3T<T>(v.z, error.w, -v.x), PxVec3T<T>(-v.y, v.x, error.w));
	AvbdRotationPotentialT<T> result;
	// A quadratic depends on the symmetric part of I. World-frame products
	// may round its two triangles differently; the gradient and metric must
	// still differentiate the same energy. Evaluate each GN entry once so the
	// assembled full operator and triangular local solve see the same metric.
	PxMat33T<T> symmetricInertia=inertia;
	for(PxU32 r=0;r<3;++r)for(PxU32 c=0;c<r;++c)
		symmetricInertia[r][c]=symmetricInertia[c][r]=T(.5)*(inertia[r][c]+inertia[c][r]);
	const PxVec3T<T> weightedResidual=symmetricInertia*residual;
	result.coordinate=residual;result.conjugate=weightedResidual;
	result.energy = T(.5)*residual.dot(weightedResidual);
	result.gradient = jacobian.getTranspose()*weightedResidual;
	const PxMat33T<T> weightedJacobian=symmetricInertia*jacobian;
	for(PxU32 r=0;r<3;++r)for(PxU32 c=0;c<=r;++c)
		result.gaussNewton[r][c]=result.gaussNewton[c][r]=jacobian[r].dot(weightedJacobian[c]);
	return result;
}

PX_FORCE_INLINE AvbdRotationPotential avbdRotationInertialPotential(
	const PxQuat& target, const PxQuat& candidate, const PxMat33& inertia)
{
	return avbdRotationInertialPotential<PxReal>(target, candidate, inertia);
}
// Relative angular coordinates (0=twist, 1=swing Y, 2=swing Z) and their
// world derivative for the child frame. The parent derivative is its negative.
// The swing/twist chart is singular at a 180-degree swing; report that domain.
PX_FORCE_INLINE bool avbdJointAngularCoordinate(const PxQuat& frameA,
	const PxQuat& frameB, PxU32 component, PxReal& C, PxVec3& gradient)
{
	// Differentiate the relative swing/twist coordinates in the same
	// world chart as the block update; the parent and child see opposite
	// derivatives, including the nominally free twist contribution.
	PxQuat rel = frameA.getConjugate() * frameB;
	if(rel.w < 0.0f) rel = -rel;
	PxReal u, v;
	PxVec3 du, dv;
	PxReal duw, dvw;
	if(component == 0)
	{
		u = rel.x; v = rel.w;
		du = PxVec3(1.0f, 0.0f, 0.0f); duw = 0.0f;
		dv = PxVec3(0.0f); dvw = 1.0f;
	}
	else
	{
		v = rel.w * rel.w + rel.x * rel.x;
		dv = PxVec3(2.0f * rel.x, 0.0f, 0.0f); dvw = 2.0f * rel.w;
		if(component == 1)
		{
			u = rel.w * rel.y - rel.x * rel.z;
			du = PxVec3(-rel.z, rel.w, -rel.x); duw = rel.y;
		}
		else
		{
			u = rel.w * rel.z + rel.x * rel.y;
			du = PxVec3(rel.y, rel.x, rel.w); duw = rel.z;
		}
	}
	const PxReal denom = u * u + v * v;
	if(denom <= 1.0e-16f) return false;
	C = 2.0f * PxAtan2(u, v);
	const PxVec3 g = (du * v - dv * u) * (2.0f / denom);
	const PxReal gw = (duw * v - dvw * u) * (2.0f / denom);
	const PxVec3 imag(rel.x, rel.y, rel.z);
	gradient = frameA.rotate(
		(g * rel.w + imag.cross(g) - imag * gw) * 0.5f);
	return PxIsFinite(C) && gradient.isFinite();
}

} // namespace Dy
} // namespace physx
#endif
