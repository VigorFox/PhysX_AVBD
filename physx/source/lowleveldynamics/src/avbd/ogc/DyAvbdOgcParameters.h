// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_OGC_PARAMETERS_H
#define DY_AVBD_OGC_PARAMETERS_H

#include "foundation/PxMath.h"
#include "foundation/PxSimpleTypes.h"

namespace physx
{
namespace Dy
{

// CPU AVBD OGC parameter and activation policy.
//
// This file contains only stateless shared policy. Geometry families consume
// it but do not own the parameter type or activation equations.

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

// Two-stage C2 activation function (OGC Eq. 18-20).
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

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_OGC_PARAMETERS_H
