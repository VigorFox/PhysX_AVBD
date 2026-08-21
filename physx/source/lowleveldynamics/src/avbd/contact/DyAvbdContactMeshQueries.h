// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_CONTACT_MESH_QUERIES_H
#define DY_AVBD_CONTACT_MESH_QUERIES_H

#include "avbd/contact/DyAvbdContactStats.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTypes.h"
#include "foundation/PxArray.h"

namespace physx
{
namespace Dy
{

// CPU AVBD collision-mesh query helpers.
//
// The parity query is a stateless geometric predicate shared by rigid-soft
// and soft-pair contact paths. It intentionally carries no contact policy.

inline bool avbdIsPointInsideTetMesh(
	const PxVec3& point,
	const PxArray<PxU32>& surfaceTriangles,
	const AvbdSoftParticle* particles,
	AvbdSoftCollisionStats* stats = NULL)
{
	int crossings = 0;
	PxVec3 rayDir(0.0f, 1.0f, 0.0f);
	for(PxU32 ti = 0; ti + 2 < surfaceTriangles.size(); ti += 3)
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

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_CONTACT_MESH_QUERIES_H
