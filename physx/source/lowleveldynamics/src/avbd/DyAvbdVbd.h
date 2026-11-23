// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_VBD_H
#define DY_AVBD_VBD_H

// Particle state, cooked surface topology and shared prediction/writeback.

#include "foundation/PxBasicTemplates.h"
#include "foundation/PxMat33.h"
#include "foundation/PxMath.h"
#include "foundation/PxQuat.h"
#include "foundation/PxSimpleTypes.h"
#include "foundation/PxVec3.h"
#include "avbd/DyAvbdKernelsScalar.h"

namespace physx
{
namespace Dy
{

struct AvbdSoftParticle
{
	PxVec3 p;
	PxVec3 pHat;
	PxVec3 pStart;
	PxVec3 vel;
	PxVec3 supportedGravityTravel;
	PxReal mass;
	PxReal invMass;
};

struct AvbdSoftTri
{
	PxU32 p0, p1, p2;
	PxReal dmInv00, dmInv01;
	PxReal dmInv10, dmInv11;
	PxReal restArea;
};

struct AvbdSoftBending
{
	PxU32 opp0, opp1;
	PxU32 edgeStart, edgeEnd;
	PxReal restShapeAngle;
	PxReal restLength;
};

PX_FORCE_INLINE PxReal avbdSoftDihedralAngle(const PxVec3& x0,
	const PxVec3& x1, const PxVec3& x2, const PxVec3& x3)
{
	const PxVec3 edge = x3 - x2;
	PxVec3 n0 = (x2 - x0).cross(x3 - x0);
	PxVec3 n1 = (x3 - x1).cross(x2 - x1);
	const PxReal edgeLength = edge.magnitude();
	const PxReal n0Length = n0.magnitude();
	const PxReal n1Length = n1.magnitude();
	if(edgeLength < 1.0e-8f || n0Length < 1.0e-8f || n1Length < 1.0e-8f)
		return 0.0f;
	n0 *= 1.0f / n0Length;
	n1 *= 1.0f / n1Length;
	const PxVec3 edgeDirection = edge * (1.0f / edgeLength);
	return PxAtan2(n0.cross(n1).dot(edgeDirection),
		PxClamp(n0.dot(n1), -1.0f, 1.0f));
}

PX_FORCE_INLINE bool avbdSoftBuildTri(AvbdSoftTri& tri, PxU32 i0, PxU32 i1,
	PxU32 i2, const AvbdSoftParticle* particles)
{
	const PxVec3 x0 = particles[i0].p;
	const PxVec3 e01 = particles[i1].p - x0;
	const PxVec3 e02 = particles[i2].p - x0;
	const PxReal e01Len = e01.magnitude();
	const PxVec3 n = e01.cross(e02);
	const PxReal nLen = n.magnitude();
	if(e01Len < 1.0e-12f || nLen < 2.0e-12f)
		return false;
	const PxVec3 t1 = e01 * (1.0f / e01Len);
	const PxVec3 t2 = n.cross(t1) * (1.0f / nLen);
	const PxReal d00 = e01.dot(t1);
	const PxReal d10 = e01.dot(t2);
	const PxReal d01 = e02.dot(t1);
	const PxReal d11 = e02.dot(t2);
	const PxReal det = d00 * d11 - d01 * d10;
	if(PxAbs(det) < 1.0e-12f)
		return false;
	const PxReal invDet = 1.0f / det;
	tri.p0 = i0;
	tri.p1 = i1;
	tri.p2 = i2;
	tri.dmInv00 = d11 * invDet;
	tri.dmInv01 = -d01 * invDet;
	tri.dmInv10 = -d10 * invDet;
	tri.dmInv11 = d00 * invDet;
	tri.restArea = 0.5f * nLen;
	return true;
}

PX_FORCE_INLINE void avbdSoftPredict(AvbdSoftParticle* particles, PxU32 nbParticles,
	const PxVec3& gravity, PxReal dt)
{
	for(PxU32 i = 0; i < nbParticles; ++i)
	{
		AvbdSoftParticle& sp = particles[i];
		sp.pStart = sp.p;
		if(sp.invMass <= 0.0f)
		{
			sp.pHat = sp.p;
			sp.vel = PxVec3(0.0f);
			continue;
		}
		sp.vel += gravity * dt;
		sp.pHat = sp.p + sp.vel * dt;
	}
}

PX_FORCE_INLINE void avbdSoftWriteback(AvbdSoftParticle* particles, PxU32 nbParticles, PxReal dt)
{
	const PxReal invH = (dt > 0.0f) ? (1.0f / dt) : 0.0f;
	for(PxU32 i = 0; i < nbParticles; ++i)
	{
		AvbdSoftParticle& sp = particles[i];
		if(sp.invMass <= 0.0f)
			continue;
		sp.vel = (sp.p - sp.pStart) * invH;
	}
}

} // namespace Dy
} // namespace physx

#endif
