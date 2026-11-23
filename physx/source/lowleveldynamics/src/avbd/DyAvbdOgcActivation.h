// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_OGC_ACTIVATION_H
#define DY_AVBD_OGC_ACTIVATION_H

#include "foundation/PxMath.h"
#include <cmath>
#include <limits>

namespace physx { namespace Dy {

template<typename T> struct AvbdOGCActivationT
{
	T energy, force, hessian;
};
typedef AvbdOGCActivationT<PxReal> AvbdOGCActivation;

template<typename T> PX_FORCE_INLINE AvbdOGCActivationT<T>
avbdOGCActivationFullT(T d, T r, T kc, T tau)
{
	AvbdOGCActivationT<T> result = {T(0), T(0), T(0)};
	if(d >= r) return result;
	const T width = r - tau;
	if(d >= tau)
	{
		const T penetration = r - d;
		result.energy = T(0.5) * kc * penetration * penetration;
		result.force = kc * penetration;
		result.hessian = kc;
	}
	else if(d > T(0))
	{
		// Match -E'(tau)=kc*(r-tau), hence A=kc*tau*(r-tau).
		// An extra factor (r-tau) breaks both units and C1 continuity.
		// With tau=r/2 this also matches E''(tau)=kc.
		const T A = kc * tau * width;
		result.energy = T(0.5) * kc * width * width - A * (std::log(d)-std::log(tau));
		result.force = A / d;
		result.hessian = result.force / d;
	}
	else
	{
		// Closed-boundary hard-contact inputs need an explicit domain transfer;
		// no arbitrary distance clamp or finite, force-inconsistent energy.
		result.energy = result.force = result.hessian = std::numeric_limits<T>::infinity();
	}
	return result;
}

PX_FORCE_INLINE AvbdOGCActivation avbdOGCActivationFull(PxReal d, PxReal r, PxReal kc, PxReal tau)
{
	return avbdOGCActivationFullT(d, r, kc, tau);
}

PX_FORCE_INLINE AvbdOGCActivation avbdOGCActivationQuadratic(PxReal d, PxReal r, PxReal kc)
{
	AvbdOGCActivation result = {0.0f, 0.0f, 0.0f};
	if(d < r)
	{
		const PxReal penetration = r - d;
		result.energy = 0.5f * kc * penetration * penetration;
		result.force = kc * penetration;
		result.hessian = kc;
	}
	return result;
}

// Exact normal dual resolvent: f = trial + rho * inverseForce(f).
// Used by the independent colored primal/dual experiment. r=0 is the
// original Signorini map. Finite iterates still need full candidate checks.
template<typename T> PX_FORCE_INLINE T avbdOGCNormalResolvent(T trial, T rho, T r, T kc)
{
	if(!(r > T(0))) return PxMax(T(0), trial);
	const T soft = (trial + rho*r) / (T(1) + rho/kc);
	if(soft <= T(0)) return T(0);
	const T tau = T(0.5)*r;
	if(soft <= kc*(r-tau)) return soft;
	const T product = rho*kc*tau*(r-tau);
	const T root = std::hypot(trial, T(2)*std::sqrt(product));
	return trial >= T(0) ? T(0.5)*(trial+root) : T(2)*product/(root-trial);
}

// Gap carrying the given compressive load on the same quadratic/log law.
template<typename T> PX_FORCE_INLINE T avbdOGCLoadedGap(T force,T radius,T stiffness)
{
    if(!(radius>T(0)))return T(0);
    const T half=T(.5)*radius,transition=half*stiffness;
    return force>transition ? half*(transition/force) : radius-force/stiffness;
}

// Positive-radius normal: inertial and contact compliance are in series.
// In the logarithmic branch phi'' = kc * (force / transition)^2.
template<typename T> PX_FORCE_INLINE T avbdOGCLoadedNormalMetric(T compliance,T force,T r,T kc)
{
    const T ratio=PxMax(T(1),force/(T(.5)*kc*r));
    return T(1)/(compliance+(T(1)/kc)/(ratio*ratio));
}

// Eliminate z from phi(z) + lambda*(d-z) + rho/2*(d-z)^2.
// The additive lambda^2/(2*rho) is constant during primal. Its derivative
// is -force and its curvature is rho*phi''/(rho+phi''). The same force is
// the next (negative) row multiplier. d=0 is finite: z, not d, enters log.
template<typename T> PX_FORCE_INLINE AvbdOGCActivationT<T>
avbdOGCNormalEnvelope(T d,T lambda,T rho,T r,T kc)
{
	AvbdOGCActivationT<T> result={T(0),T(0),T(0)};
	if(!(rho>T(0))) return result;
	const T f=avbdOGCNormalResolvent(-lambda-rho*d,rho,r,kc);
	if(!(f>T(0))) return result;
	result.force=f;
	if(!(r>T(0)))
	{
		result.energy=T(0.5)*f*(f/rho);result.hessian=rho;return result;
	}
	const T tau=T(0.5)*r,transition=kc*tau;
	if(f<=transition)
	{
		result.energy=T(0.5)*f*(f/kc+f/rho);
		result.hessian=rho/(T(1)+rho/kc);
	}
	else
	{
		const T A=transition*tau;
		result.energy=T(0.5)*A+A*(std::log(f)-std::log(transition))+T(0.5)*f*(f/rho);
		result.hessian=rho/(T(1)+(rho/f)*(A/f));
	}
	return result;
}

} }
#endif
