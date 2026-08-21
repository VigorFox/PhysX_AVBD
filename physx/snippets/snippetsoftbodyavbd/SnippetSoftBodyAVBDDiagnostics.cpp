// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

// Shared rotation diagnostics for unit and visual execution.

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include "SnippetSoftBodyAVBDDiagnostics.h"

using namespace physx;
using namespace physx::Dy;

namespace SnippetSoftBodyAVBDDiagnostics
{

// ============================================================================
// Body-level rotation diagnostics
// ============================================================================

bool isRotationTraceEnabled()
{
	const char* value = std::getenv("PHYSX_AVBD_SOFTBODY_ROT_TRACE");
	return value && value[0] && value[0] != '0';
}

PxU32 getRotationTraceInterval()
{
	const char* value = std::getenv("PHYSX_AVBD_SOFTBODY_ROT_TRACE_INTERVAL");
	if (!value || !value[0])
		return 30;
	const int interval = std::atoi(value);
	return interval > 0 ? PxU32(interval) : 30;
}

static PxVec3 computeBodyMassCentroid(const PxArray<AvbdSoftParticle>& particles,
																			const AvbdSoftBody& body)
{
	PxVec3 centroid(0.0f);
	PxReal totalMass = 0.0f;
	for (PxU32 i = 0; i < body.compiled.particleCount; i++)
	{
		const PxU32 pi = body.compiled.particleStart + i;
		const PxReal mass = particles[pi].mass;
		centroid += particles[pi].position * mass;
		totalMass += mass;
	}
	return totalMass > 0.0f ? centroid * (1.0f / totalMass) : PxVec3(0.0f);
}

void captureBodyReferenceLocals(const PxArray<AvbdSoftParticle>& particles,
																			 const PxArray<AvbdSoftBody>& bodies,
																			 PxArray<PxArray<PxVec3> >& refs)
{
	refs.clear();
	refs.resize(bodies.size());
	for (PxU32 bi = 0; bi < bodies.size(); bi++)
	{
		const AvbdSoftBody& body = bodies[bi];
		const PxVec3 centroid = computeBodyMassCentroid(particles, body);
		refs[bi].resize(body.compiled.particleCount);
		for (PxU32 i = 0; i < body.compiled.particleCount; i++)
		{
			const PxU32 pi = body.compiled.particleStart + i;
			refs[bi][i] = particles[pi].position - centroid;
		}
	}
}

static PxQuat estimateBodyRotation(const PxArray<AvbdSoftParticle>& particles,
																	 const AvbdSoftBody& body,
																	 const PxArray<PxVec3>& refLocals)
{
	if (refLocals.size() != body.compiled.particleCount)
		return PxQuat(PxIdentity);

	const PxVec3 centroid = computeBodyMassCentroid(particles, body);

	PxReal sxx = 0.0f, sxy = 0.0f, sxz = 0.0f;
	PxReal syx = 0.0f, syy = 0.0f, syz = 0.0f;
	PxReal szx = 0.0f, szy = 0.0f, szz = 0.0f;

	for (PxU32 i = 0; i < body.compiled.particleCount; i++)
	{
		const PxU32 pi = body.compiled.particleStart + i;
		const PxReal mass = particles[pi].mass;
		const PxVec3 p = particles[pi].position - centroid;
		const PxVec3 q = refLocals[i];
		sxx += mass * p.x * q.x; sxy += mass * p.x * q.y; sxz += mass * p.x * q.z;
		syx += mass * p.y * q.x; syy += mass * p.y * q.y; syz += mass * p.y * q.z;
		szx += mass * p.z * q.x; szy += mass * p.z * q.y; szz += mass * p.z * q.z;
	}

	const PxReal N[4][4] = {
		{ sxx + syy + szz, syz - szy,         szx - sxz,         sxy - syx },
		{ syz - szy,       sxx - syy - szz,   sxy + syx,         szx + sxz },
		{ szx - sxz,       sxy + syx,        -sxx + syy - szz,   syz + szy },
		{ sxy - syx,       szx + sxz,         syz + szy,        -sxx - syy + szz }
	};

	PxReal qv[4] = { 1.0f, 0.0f, 0.0f, 0.0f };
	for (PxU32 iter = 0; iter < 16; iter++)
	{
		PxReal next[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
		for (PxU32 r = 0; r < 4; r++)
			for (PxU32 c = 0; c < 4; c++)
				next[r] += N[r][c] * qv[c];

		const PxReal len = PxSqrt(next[0]*next[0] + next[1]*next[1] + next[2]*next[2] + next[3]*next[3]);
		if (len < 1e-12f)
			return PxQuat(PxIdentity);

		qv[0] = next[0] / len;
		qv[1] = next[1] / len;
		qv[2] = next[2] / len;
		qv[3] = next[3] / len;
	}

	if (qv[0] < 0.0f)
	{
		qv[0] = -qv[0];
		qv[1] = -qv[1];
		qv[2] = -qv[2];
		qv[3] = -qv[3];
	}

	return PxQuat(qv[1], qv[2], qv[3], qv[0]).getNormalized();
}

static PxVec3 estimateBodyOmega(const PxArray<AvbdSoftParticle>& particles,
																const AvbdSoftBody& body)
{
	const PxVec3 centroid = computeBodyMassCentroid(particles, body);
	PxVec3 angularMomentum(0.0f);
	PxMat33 inertia = PxMat33::createDiagonal(PxVec3(0.0f));

	for (PxU32 i = 0; i < body.compiled.particleCount; i++)
	{
		const PxU32 pi = body.compiled.particleStart + i;
		const PxReal mass = particles[pi].mass;
		const PxVec3 r = particles[pi].position - centroid;
		const PxReal r2 = r.dot(r);
		inertia = inertia + (PxMat33::createDiagonal(PxVec3(r2)) - avbdOuter(r, r)) * mass;
		angularMomentum += r.cross(particles[pi].velocity) * mass;
	}

	PxVec3 omega = inertia.getInverse() * angularMomentum;
	if (omega.x != omega.x || omega.y != omega.y || omega.z != omega.z)
		return PxVec3(0.0f);
	return omega;
}

void printBodyRotationTrace(const char* label,
																	 PxU32 frame,
																	 const PxArray<AvbdSoftParticle>& particles,
																	 const PxArray<AvbdSoftBody>& bodies,
																	 const PxArray<PxArray<PxVec3> >& refs)
{
	for (PxU32 bi = 0; bi < bodies.size(); bi++)
	{
		const PxQuat q = estimateBodyRotation(particles, bodies[bi], refs[bi]);
		PxReal angleDeg = PxAcos(PxClamp(q.w, -1.0f, 1.0f)) * (360.0f / PxPi);
		if (angleDeg > 180.0f)
			angleDeg = 360.0f - angleDeg;

		PxVec3 axis(0.0f, 1.0f, 0.0f);
		const PxReal sinHalf = PxSqrt(PxMax(0.0f, 1.0f - q.w * q.w));
		if (sinHalf > 1e-5f)
			axis = PxVec3(q.x, q.y, q.z) * (1.0f / sinHalf);

		const PxVec3 com = computeBodyMassCentroid(particles, bodies[bi]);
		const PxVec3 omega = estimateBodyOmega(particles, bodies[bi]);
		printf("  ROT[%s] frame=%u body=%u angleDeg=%.3f axis=(%.3f,%.3f,%.3f) omega=(%.3f,%.3f,%.3f) com=(%.3f,%.3f,%.3f)\n",
					 label, frame, bi,
					 angleDeg, axis.x, axis.y, axis.z,
					 omega.x, omega.y, omega.z,
					 com.x, com.y, com.z);
	}
}


} // namespace SnippetSoftBodyAVBDDiagnostics
