// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Mathematical counterexamples against the production certificate functions.
// This is a diagnostic gate: a known defect still returns a failing exit.
#include "foundation/PxMathUtils.h"
#include "avbd/DyAvbdOgc.h"
#include "avbd_probe_foundation.h"
#include <cstdio>

using namespace physx;
using namespace physx::Dy;

int main()
{
	AvbdProbeFoundation foundation;
	AvbdOgcRigidRegion rigid = {};
	rigid.anchorPosition = PxVec3(0);
	rigid.anchorRotation = PxQuat(PxIdentity);
	rigid.bound = PX_MAX_F32;
	rigid.supportRadius = 1;
	const PxVec3 normal(0,1,0);
	for(PxU32 i = 0; i < 2; ++i)
		avbdOgcAddHalfSpace(rigid.planes, rigid.planeCount, normal, 0.05f,
			PX_MAX_U32, PxVec3(0), PxVec3(0), PX_MAX_U32,
			PX_MAX_U32, PX_MAX_U32, PX_MAX_U32, PxVec3(i ? -1.0f : 1.0f,0,0));
	PxVec3 position(0); PxQuat rotation(0.2f, PxVec3(0,0,1));
	const bool rigidFeasible=avbdOgcTruncateRigid(rigid, position, rotation).feasible;
	PxReal maxViolation = 0;
	for(PxU32 i = 0; i < 2; ++i)
	{
		const PxVec3 feature(i ? -1.0f : 1.0f,0,0);
		maxViolation = PxMax(maxViolation,
			-normal.dot(position + rotation.rotate(feature) - feature) - 0.05f);
	}
	const bool rigidPass = rigidFeasible && maxViolation <= 1.0e-6f;
	std::printf("[OGC_CERTIFICATE_TEST] case=rotated_features status=%s maxViolation=%.9g\n",
		rigidPass ? "PASS" : "FAIL", double(maxViolation));

	AvbdOgcPointRegion point = {};
	avbdOgcResetPointRegion(point, PxVec3(0));
	const PxVec3 normals[2] = {PxVec3(1,0,0), PxVec3(-0.5f,0.8660254f,0)};
	for(PxU32 i = 0; i < 2; ++i)
		avbdOgcAddHalfSpace(point.planes, point.planeCount, normals[i], 0,
			PX_MAX_U32, PxVec3(0), PxVec3(0));
	position = PxVec3(-1,-1,0);
	const bool pointFeasible=avbdOgcTruncatePoint(point, position).feasible;
	maxViolation = PxMax(0.0f, PxMax(-normals[0].dot(position), -normals[1].dot(position)));
	const bool pointPass = pointFeasible && maxViolation <= 1.0e-6f;
	std::printf("[OGC_CERTIFICATE_TEST] case=plane_intersection status=%s maxViolation=%.9g\n",
		pointPass ? "PASS" : "FAIL", double(maxViolation));
	// acos(abs(q0.dot(q1))) rounds to zero for represented small rotations.
	// A rigid vertex still moves, so admitting the zero-angle budget is unsafe.
	AvbdOgcRigidRegion small = {};
	small.anchorPosition=PxVec3(0);small.anchorRotation=PxQuat(PxIdentity);
	small.supportRadius=1;small.bound=2.5e-5f;
	const PxQuat tiny(1.e-4f,PxVec3(0,0,1));
	const PxReal angle=avbdOgcRotationAngle(small.anchorRotation,tiny);
	const PxReal displacement=(tiny.rotate(PxVec3(1,0,0))-PxVec3(1,0,0)).magnitude();
	const bool smallPass=angle>9.99e-5f && displacement>small.bound &&
		!avbdOgcRigidContains(small,PxVec3(0),tiny) &&
		PxAbs(avbdOgcRotationAngle(small.anchorRotation,-tiny)-angle)<1.e-8f;
	std::printf("[OGC_CERTIFICATE_TEST] case=represented_small_rotation status=%s angle=%.9g actualVertexMotion=%.9g bound=%.9g\n",
		smallPass?"PASS":"FAIL",double(angle),double(displacement),double(small.bound));
    // An unbounded SQP domain must not inherit a finite ball's inward bias.
    // This target is representable one ULP away; truncation erases its motion.
    const PxVec3 anchor(0,26,0);const PxVec3d request(0,-1.82925432e-6,0);
    AvbdOgcProjectionPlanes planes;PxVec3 nearest,bounded;
    const bool quantized=avbdOgcPublishRegionPoint(anchor,request,planes,PX_MAX_F32,nearest);
    const double before=request.magnitudeSquared(),after=(avbdOgcDouble(nearest)-avbdOgcDouble(anchor)-request).magnitudeSquared();
    const bool ball=avbdOgcPublishRegionPoint(anchor,request,planes,1.e-6,bounded);
    const bool quantizationPass=quantized && after<.01*before && ball && bounded==anchor;
    std::printf("[OGC_CERTIFICATE_TEST] case=unbounded_quantized_descent status=%s energyBefore=%.9g energyAfter=%.9g finiteBallPreserved=%u\n",
        quantizationPass?"PASS":"FAIL",before,after,unsigned(ball && bounded==anchor));
	return rigidPass && pointPass && smallPass && quantizationPass ? 0 : 1;
}
