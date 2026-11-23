// Differentiate the production angular coordinate in the actual world chart.
#include "avbd/DyAvbdRigidMetric.h"
#include <cstdio>

using namespace physx;
using namespace physx::Dy;

static PxU32 state = 0x173b29u;
static PxReal sample()
{
	state = 1664525u * state + 1013904223u;
	return PxReal(state >> 8) / PxReal(0x1000000u) * 2.0f - 1.0f;
}

int main()
{
	const PxReal h = 0.002f;
	PxReal maxError = 0.0f, maxObjectivityError = 0.0f;
	PxU32 checked = 0;
	bool pass = true;
	for(PxU32 trial = 0; trial < 256; ++trial)
	{
		const PxVec3 axis = PxVec3(sample(), sample(), sample()).getNormalized();
		const PxQuat a(sample() * 2.5f, axis);
		const PxQuat b = a * PxQuat(sample() * 0.8f, PxVec3(0, 1, 0)) *
			PxQuat(sample() * 0.6f, PxVec3(0, 0, 1)) *
			PxQuat(sample() * 2.5f, PxVec3(1, 0, 0));
		const PxQuat common(sample() * 2.5f, axis);
		for(PxU32 component = 0; component < 3; ++component)
		{
			PxReal C, rotatedC;
			PxVec3 gradient, rotatedGradient;
			if(!avbdJointAngularCoordinate(a, b, component, C, gradient) ||
				!avbdJointAngularCoordinate(common * a, common * b,
					component, rotatedC, rotatedGradient)) return 2;
			maxObjectivityError = PxMax(maxObjectivityError, PxAbs(C - rotatedC));
			pass &= (rotatedGradient - common.rotate(gradient)).magnitude() < 2.0e-5f;
			for(PxU32 endpoint = 0; endpoint < 2; ++endpoint)
				for(PxU32 k = 0; k < 3; ++k)
				{
					PxVec3 direction(0.0f); direction[k] = 1.0f;
					const PxQuat plus(h, direction), minus(-h, direction);
					PxReal cp, cm;
					PxVec3 scratch;
					if(!avbdJointAngularCoordinate(endpoint ? a : plus * a,
						endpoint ? plus * b : b, component, cp, scratch) ||
						!avbdJointAngularCoordinate(endpoint ? a : minus * a,
						endpoint ? minus * b : b, component, cm, scratch)) return 2;
					const PxReal analytic = endpoint ? gradient[k] : -gradient[k];
					const PxReal numeric = (cp - cm) / (2.0f * h);
					const PxReal error = PxAbs(analytic - numeric) / PxMax(1.0f, PxAbs(analytic));
					maxError = PxMax(maxError, error);
					pass &= PxIsFinite(error) && error < 5.0e-4f;
					++checked;
				}
		}
	}
	// A free twist must not introduce a locked swing residual.
	for(PxU32 i = 0; i < 24; ++i)
		for(PxU32 component = 1; component < 3; ++component)
		{
			PxReal C;
			PxVec3 gradient;
			pass &= avbdJointAngularCoordinate(PxQuat(PxIdentity),
				PxQuat(-2.8f + PxReal(i) * 0.24f, PxVec3(1, 0, 0)), component, C, gradient);
			pass &= PxAbs(C) < 1.0e-6f && PxAbs(gradient.x) < 1.0e-6f;
		}
	pass &= maxObjectivityError < 2.0e-5f;
	std::printf("[JOINT_GEOMETRY_TEST] status=%s derivatives=%u maxRelativeError=%.9g maxObjectivityError=%.9g\n",
		pass ? "PASS" : "FAIL", checked, double(maxError), double(maxObjectivityError));
	return pass ? 0 : 1;
}
