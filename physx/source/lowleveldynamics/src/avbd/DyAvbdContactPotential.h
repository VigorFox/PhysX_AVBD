// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_CONTACT_POTENTIAL_H
#define DY_AVBD_CONTACT_POTENTIAL_H

#include "foundation/PxMath.h"

namespace physx
{
namespace Dy
{

struct AvbdContactPotential
{
	PxReal energy;
	PxReal gradient;
	PxReal curvature;
};

// Exact integral over the change in gap, including activation and force-limit
// crossings. Integrating the piecewise-linear force avoids subtracting two
// large potential values for a small accepted displacement.
PX_FORCE_INLINE double avbdUnilateralContactEnergyChange(double gap,double change,
	double penalty,double lambda,double minForce)
{
	if(!(penalty>0.0) || change==0.0) return 0.0;
	const double start=gap+lambda/penalty,end=start+change;
	const double lo=PxMin(start,end),hi=PxMax(start,end);
	const double force=PxMin(0.0,minForce),saturation=force/penalty;
	if(start>=0.0 && end>=0.0) return 0.0;
	if(start<=saturation && end<=saturation) return force*change;
	if(lo>=saturation && hi<=0.0) return penalty*change*(start+0.5*change);
	double integral=0.0;
	if(lo<saturation) integral+=force*(PxMin(hi,saturation)-lo);
	const double a=PxMax(lo,saturation),b=PxMin(hi,0.0);
	if(a<b) integral+=0.5*penalty*(b-a)*(a+b);
	return change>0.0 ? integral : -integral;
}

// Primitive for a bounded bilateral force.  Unlike squaring the clamped
// force, this remains linear in C after saturation and still differentiates
// to the authored force limit.
PX_FORCE_INLINE PxReal avbdBoundedRowEnergy(PxReal C, PxReal penalty,
	PxReal lambda, PxReal lower, PxReal upper)
{
	if(!(penalty > 0.0f))
		return PxClamp(lambda, lower, upper) * C;
	const PxReal shifted = C + lambda / penalty;
	if(shifted < lower / penalty)
		return lower * (shifted - 0.5f * lower / penalty);
	if(shifted > upper / penalty)
		return upper * (shifted - 0.5f * upper / penalty);
	return 0.5f * penalty * shifted * shifted;
}

// Integral of clamp(penalty*C + lambda, minForce, 0).  The additive
// constant makes the energy zero at release; lambda is frozen during primal.
// lambda=0 gives the quadratic offset well.  A compressive AL multiplier
// keeps the potential active beyond C=0 until C=-lambda/penalty.
PX_FORCE_INLINE AvbdContactPotential avbdUnilateralContactPotential(
	PxReal C, PxReal penalty, PxReal lambda, PxReal minForce)
{
	AvbdContactPotential result = {0.0f, 0.0f, 0.0f};
	if(!(penalty > 0.0f))
		return result;
	const PxReal shifted = C + lambda / penalty;
	if(shifted >= 0.0f)
		return result;
	const PxReal lower = PxMin(0.0f, minForce);
	const PxReal saturation = lower / penalty;
	if(shifted < saturation)
	{
		result.energy = lower * (shifted - 0.5f * saturation);
		result.gradient = lower;
		return result;
	}
	result.gradient = penalty * shifted;
	result.energy = 0.5f * result.gradient * shifted;
	result.curvature = penalty;
	return result;
}

} // namespace Dy
} // namespace physx

#endif
