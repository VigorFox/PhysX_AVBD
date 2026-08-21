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
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.

#include "avbd/backend/cpu/DyAvbdCpuIsa.h"

#include <cfloat>
#include <immintrin.h>

namespace physx {
namespace Dy {

// This file is compiled independently with AVX2+FMA enabled.  It must never
// be inlined into a baseline TU through a header.
PxF32 avbdCpuIsaAvx2FmaProbeDot8(const PxF32* lhs, const PxF32* rhs)
{
	const __m256 values = _mm256_fmadd_ps(
		_mm256_loadu_ps(lhs), _mm256_loadu_ps(rhs), _mm256_setzero_ps());
	PX_ALIGN_PREFIX(32) PxF32 lanes[8] PX_ALIGN_SUFFIX(32);
	_mm256_store_ps(lanes, values);
	_mm256_zeroupper();
	return lanes[0] + lanes[1] + lanes[2] + lanes[3] +
		lanes[4] + lanes[5] + lanes[6] + lanes[7];
}

#if !defined(PX_AVBD_EXCLUDE_EXPERIMENTAL_RIGID_SIMD)

void avbdCpuIsaAvx2FmaRigidNormalContactDynamicEndpoint8(
	const AvbdRigidNormalContactDynamicTarget8& input,
	PxF32* worldPositionX, PxF32* worldPositionY, PxF32* worldPositionZ)
{
	const __m256 active = _mm256_castsi256_ps(_mm256_set_epi32(
		(input.dynamicMask & 0x80u) ? -1 : 0,
		(input.dynamicMask & 0x40u) ? -1 : 0,
		(input.dynamicMask & 0x20u) ? -1 : 0,
		(input.dynamicMask & 0x10u) ? -1 : 0,
		(input.dynamicMask & 0x08u) ? -1 : 0,
		(input.dynamicMask & 0x04u) ? -1 : 0,
		(input.dynamicMask & 0x02u) ? -1 : 0,
		(input.dynamicMask & 0x01u) ? -1 : 0));
	const __m256 two = _mm256_set1_ps(2.0f);
	const __m256 qx = _mm256_loadu_ps(input.rotation[0]);
	const __m256 qy = _mm256_loadu_ps(input.rotation[1]);
	const __m256 qz = _mm256_loadu_ps(input.rotation[2]);
	const __m256 qw = _mm256_loadu_ps(input.rotation[3]);
	const __m256 vx = _mm256_loadu_ps(input.contactPoint[0]);
	const __m256 vy = _mm256_loadu_ps(input.contactPoint[1]);
	const __m256 vz = _mm256_loadu_ps(input.contactPoint[2]);
	const __m256 tx = _mm256_mul_ps(two, _mm256_sub_ps(
		_mm256_mul_ps(qy, vz), _mm256_mul_ps(qz, vy)));
	const __m256 ty = _mm256_mul_ps(two, _mm256_sub_ps(
		_mm256_mul_ps(qz, vx), _mm256_mul_ps(qx, vz)));
	const __m256 tz = _mm256_mul_ps(two, _mm256_sub_ps(
		_mm256_mul_ps(qx, vy), _mm256_mul_ps(qy, vx)));
	const __m256 rx = _mm256_add_ps(
		_mm256_add_ps(vx, _mm256_mul_ps(qw, tx)),
		_mm256_sub_ps(_mm256_mul_ps(qy, tz), _mm256_mul_ps(qz, ty)));
	const __m256 ry = _mm256_add_ps(
		_mm256_add_ps(vy, _mm256_mul_ps(qw, ty)),
		_mm256_sub_ps(_mm256_mul_ps(qz, tx), _mm256_mul_ps(qx, tz)));
	const __m256 rz = _mm256_add_ps(
		_mm256_add_ps(vz, _mm256_mul_ps(qw, tz)),
		_mm256_sub_ps(_mm256_mul_ps(qx, ty), _mm256_mul_ps(qy, tx)));
	_mm256_storeu_ps(worldPositionX, _mm256_and_ps(
		_mm256_add_ps(_mm256_loadu_ps(input.position[0]), rx), active));
	_mm256_storeu_ps(worldPositionY, _mm256_and_ps(
		_mm256_add_ps(_mm256_loadu_ps(input.position[1]), ry), active));
	_mm256_storeu_ps(worldPositionZ, _mm256_and_ps(
		_mm256_add_ps(_mm256_loadu_ps(input.position[2]), rz), active));
	_mm256_zeroupper();
}

#endif // !PX_AVBD_EXCLUDE_EXPERIMENTAL_RIGID_SIMD

namespace {

PX_FORCE_INLINE __m256 avbdAbs8(__m256 value)
{
	return _mm256_andnot_ps(
		_mm256_set1_ps(-0.0f), value);
}

PX_FORCE_INLINE __m256 avbdFinite8(__m256 value)
{
	return _mm256_cmp_ps(
		avbdAbs8(value), _mm256_set1_ps(FLT_MAX), _CMP_LE_OQ);
}

PX_FORCE_INLINE __m256 avbdPositiveFinite8(
	__m256 value, PxF32 minimum)
{
	return _mm256_and_ps(
		avbdFinite8(value),
		_mm256_cmp_ps(
			value, _mm256_set1_ps(minimum), _CMP_GT_OQ));
}

PX_FORCE_INLINE __m256 avbdDeterminantValid8(__m256 determinant)
{
	return _mm256_and_ps(
		avbdFinite8(determinant),
		_mm256_cmp_ps(
			avbdAbs8(determinant), _mm256_set1_ps(1.0e-9f),
			_CMP_GT_OQ));
}

PX_FORCE_INLINE __m256 avbdLinearCombination3(
	__m256 a, __m256 b, __m256 c,
	__m256 x, __m256 y, __m256 z)
{
	return _mm256_fmadd_ps(
		c, z, _mm256_fmadd_ps(b, y, _mm256_mul_ps(a, x)));
}

PX_FORCE_INLINE __m256 avbdDot3(
	__m256 ax, __m256 ay, __m256 az,
	__m256 bx, __m256 by, __m256 bz)
{
	return _mm256_fmadd_ps(
		az, bz, _mm256_fmadd_ps(ay, by, _mm256_mul_ps(ax, bx)));
}

PX_FORCE_INLINE __m256 avbdCrossX(
	__m256 ay, __m256 az, __m256 by, __m256 bz)
{
	return _mm256_fmsub_ps(ay, bz, _mm256_mul_ps(az, by));
}

PX_FORCE_INLINE __m256 avbdCrossY(
	__m256 ax, __m256 az, __m256 bx, __m256 bz)
{
	return _mm256_fmsub_ps(az, bx, _mm256_mul_ps(ax, bz));
}

PX_FORCE_INLINE __m256 avbdCrossZ(
	__m256 ax, __m256 ay, __m256 bx, __m256 by)
{
	return _mm256_fmsub_ps(ax, by, _mm256_mul_ps(ay, bx));
}

PX_FORCE_INLINE __m256 avbdAllFinite9(
	__m256 a0, __m256 a1, __m256 a2,
	__m256 b0, __m256 b1, __m256 b2,
	__m256 c0, __m256 c1, __m256 c2)
{
	__m256 finite = _mm256_and_ps(avbdFinite8(a0), avbdFinite8(a1));
	finite = _mm256_and_ps(finite, avbdFinite8(a2));
	finite = _mm256_and_ps(finite, avbdFinite8(b0));
	finite = _mm256_and_ps(finite, avbdFinite8(b1));
	finite = _mm256_and_ps(finite, avbdFinite8(b2));
	finite = _mm256_and_ps(finite, avbdFinite8(c0));
	finite = _mm256_and_ps(finite, avbdFinite8(c1));
	return _mm256_and_ps(finite, avbdFinite8(c2));
}

} // namespace

void avbdCpuIsaAvx2FmaCorotationalTetPacket8(
	const AvbdTetMaterialPacket8Input& input,
	PxF32 mu, PxF32 lambda,
	AvbdTetMaterialPacket8Output& output)
{
	const __m256 e1x = _mm256_loadu_ps(input.e1X);
	const __m256 e1y = _mm256_loadu_ps(input.e1Y);
	const __m256 e1z = _mm256_loadu_ps(input.e1Z);
	const __m256 e2x = _mm256_loadu_ps(input.e2X);
	const __m256 e2y = _mm256_loadu_ps(input.e2Y);
	const __m256 e2z = _mm256_loadu_ps(input.e2Z);
	const __m256 e3x = _mm256_loadu_ps(input.e3X);
	const __m256 e3y = _mm256_loadu_ps(input.e3Y);
	const __m256 e3z = _mm256_loadu_ps(input.e3Z);

	const __m256 dm0x = _mm256_loadu_ps(input.dm0X);
	const __m256 dm0y = _mm256_loadu_ps(input.dm0Y);
	const __m256 dm0z = _mm256_loadu_ps(input.dm0Z);
	const __m256 dm1x = _mm256_loadu_ps(input.dm1X);
	const __m256 dm1y = _mm256_loadu_ps(input.dm1Y);
	const __m256 dm1z = _mm256_loadu_ps(input.dm1Z);
	const __m256 dm2x = _mm256_loadu_ps(input.dm2X);
	const __m256 dm2y = _mm256_loadu_ps(input.dm2Y);
	const __m256 dm2z = _mm256_loadu_ps(input.dm2Z);

	const __m256 f0x = avbdLinearCombination3(
		e1x, e2x, e3x, dm0x, dm0y, dm0z);
	const __m256 f0y = avbdLinearCombination3(
		e1y, e2y, e3y, dm0x, dm0y, dm0z);
	const __m256 f0z = avbdLinearCombination3(
		e1z, e2z, e3z, dm0x, dm0y, dm0z);
	const __m256 f1x = avbdLinearCombination3(
		e1x, e2x, e3x, dm1x, dm1y, dm1z);
	const __m256 f1y = avbdLinearCombination3(
		e1y, e2y, e3y, dm1x, dm1y, dm1z);
	const __m256 f1z = avbdLinearCombination3(
		e1z, e2z, e3z, dm1x, dm1y, dm1z);
	const __m256 f2x = avbdLinearCombination3(
		e1x, e2x, e3x, dm2x, dm2y, dm2z);
	const __m256 f2y = avbdLinearCombination3(
		e1y, e2y, e3y, dm2x, dm2y, dm2z);
	const __m256 f2z = avbdLinearCombination3(
		e1z, e2z, e3z, dm2x, dm2y, dm2z);

	__m256 r0x = f0x;
	__m256 r0y = f0y;
	__m256 r0z = f0z;
	__m256 r1x = f1x;
	__m256 r1y = f1y;
	__m256 r1z = f1z;
	__m256 r2x = f2x;
	__m256 r2y = f2y;
	__m256 r2z = f2z;

	const __m256 deformationCof0x =
		avbdCrossX(f1y, f1z, f2y, f2z);
	const __m256 deformationCof0y =
		avbdCrossY(f1x, f1z, f2x, f2z);
	const __m256 deformationCof0z =
		avbdCrossZ(f1x, f1y, f2x, f2y);
	const __m256 deformationCof1x =
		avbdCrossX(f2y, f2z, f0y, f0z);
	const __m256 deformationCof1y =
		avbdCrossY(f2x, f2z, f0x, f0z);
	const __m256 deformationCof1z =
		avbdCrossZ(f2x, f2y, f0x, f0y);
	const __m256 deformationCof2x =
		avbdCrossX(f0y, f0z, f1y, f1z);
	const __m256 deformationCof2y =
		avbdCrossY(f0x, f0z, f1x, f1z);
	const __m256 deformationCof2z =
		avbdCrossZ(f0x, f0y, f1x, f1y);
	const __m256 deformationDeterminant = avbdDot3(
		f0x, f0y, f0z,
		deformationCof0x, deformationCof0y, deformationCof0z);
	__m256 cof0x = deformationCof0x;
	__m256 cof0y = deformationCof0y;
	__m256 cof0z = deformationCof0z;
	__m256 determinant = deformationDeterminant;
	__m256 valid = avbdDeterminantValid8(determinant);
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 half = _mm256_set1_ps(0.5f);

	for(PxU32 iteration = 0; iteration < 3; iteration++)
	{
		const __m256 safeDeterminant =
			_mm256_blendv_ps(one, determinant, valid);
		const __m256 invDeterminant =
			_mm256_div_ps(one, safeDeterminant);

		cof0x = avbdCrossX(r1y, r1z, r2y, r2z);
		cof0y = avbdCrossY(r1x, r1z, r2x, r2z);
		cof0z = avbdCrossZ(r1x, r1y, r2x, r2y);
		const __m256 cof1x = avbdCrossX(r2y, r2z, r0y, r0z);
		const __m256 cof1y = avbdCrossY(r2x, r2z, r0x, r0z);
		const __m256 cof1z = avbdCrossZ(r2x, r2y, r0x, r0y);
		const __m256 cof2x = avbdCrossX(r0y, r0z, r1y, r1z);
		const __m256 cof2y = avbdCrossY(r0x, r0z, r1x, r1z);
		const __m256 cof2z = avbdCrossZ(r0x, r0y, r1x, r1y);

		const __m256 i0x = _mm256_mul_ps(cof0x, invDeterminant);
		const __m256 i0y = _mm256_mul_ps(cof0y, invDeterminant);
		const __m256 i0z = _mm256_mul_ps(cof0z, invDeterminant);
		const __m256 i1x = _mm256_mul_ps(cof1x, invDeterminant);
		const __m256 i1y = _mm256_mul_ps(cof1y, invDeterminant);
		const __m256 i1z = _mm256_mul_ps(cof1z, invDeterminant);
		const __m256 i2x = _mm256_mul_ps(cof2x, invDeterminant);
		const __m256 i2y = _mm256_mul_ps(cof2y, invDeterminant);
		const __m256 i2z = _mm256_mul_ps(cof2z, invDeterminant);

		valid = _mm256_and_ps(valid, avbdAllFinite9(
			i0x, i0y, i0z, i1x, i1y, i1z, i2x, i2y, i2z));
		r0x = _mm256_blendv_ps(
			r0x, _mm256_mul_ps(_mm256_add_ps(r0x, i0x), half), valid);
		r0y = _mm256_blendv_ps(
			r0y, _mm256_mul_ps(_mm256_add_ps(r0y, i0y), half), valid);
		r0z = _mm256_blendv_ps(
			r0z, _mm256_mul_ps(_mm256_add_ps(r0z, i0z), half), valid);
		r1x = _mm256_blendv_ps(
			r1x, _mm256_mul_ps(_mm256_add_ps(r1x, i1x), half), valid);
		r1y = _mm256_blendv_ps(
			r1y, _mm256_mul_ps(_mm256_add_ps(r1y, i1y), half), valid);
		r1z = _mm256_blendv_ps(
			r1z, _mm256_mul_ps(_mm256_add_ps(r1z, i1z), half), valid);
		r2x = _mm256_blendv_ps(
			r2x, _mm256_mul_ps(_mm256_add_ps(r2x, i2x), half), valid);
		r2y = _mm256_blendv_ps(
			r2y, _mm256_mul_ps(_mm256_add_ps(r2y, i2y), half), valid);
		r2z = _mm256_blendv_ps(
			r2z, _mm256_mul_ps(_mm256_add_ps(r2z, i2z), half), valid);

		if(iteration + 1 < 3)
		{
			cof0x = avbdCrossX(r1y, r1z, r2y, r2z);
			cof0y = avbdCrossY(r1x, r1z, r2x, r2z);
			cof0z = avbdCrossZ(r1x, r1y, r2x, r2y);
			determinant = avbdDot3(
				r0x, r0y, r0z, cof0x, cof0y, cof0z);
			valid = _mm256_and_ps(
				valid, avbdDeterminantValid8(determinant));
		}
	}

	const __m256 polar2x = r2x;
	const __m256 polar2y = r2y;
	const __m256 polar2z = r2z;
	const __m256 norm0 = avbdDot3(r0x, r0y, r0z, r0x, r0y, r0z);
	valid = _mm256_and_ps(valid, avbdPositiveFinite8(norm0, 1.0e-12f));
	const __m256 invNorm0 = _mm256_div_ps(
		one, _mm256_sqrt_ps(_mm256_blendv_ps(one, norm0, valid)));
	r0x = _mm256_mul_ps(r0x, invNorm0);
	r0y = _mm256_mul_ps(r0y, invNorm0);
	r0z = _mm256_mul_ps(r0z, invNorm0);

	const __m256 projection = avbdDot3(
		r1x, r1y, r1z, r0x, r0y, r0z);
	r1x = _mm256_fnmadd_ps(r0x, projection, r1x);
	r1y = _mm256_fnmadd_ps(r0y, projection, r1y);
	r1z = _mm256_fnmadd_ps(r0z, projection, r1z);
	const __m256 norm1 = avbdDot3(r1x, r1y, r1z, r1x, r1y, r1z);
	valid = _mm256_and_ps(valid, avbdPositiveFinite8(norm1, 1.0e-12f));
	const __m256 invNorm1 = _mm256_div_ps(
		one, _mm256_sqrt_ps(_mm256_blendv_ps(one, norm1, valid)));
	r1x = _mm256_mul_ps(r1x, invNorm1);
	r1y = _mm256_mul_ps(r1y, invNorm1);
	r1z = _mm256_mul_ps(r1z, invNorm1);

	r2x = avbdCrossX(r0y, r0z, r1y, r1z);
	r2y = avbdCrossY(r0x, r0z, r1x, r1z);
	r2z = avbdCrossZ(r0x, r0y, r1x, r1y);
	const __m256 handedness = avbdDot3(
		r2x, r2y, r2z, polar2x, polar2y, polar2z);
	valid = _mm256_and_ps(valid, avbdFinite8(handedness));
	const __m256 flipMask = _mm256_cmp_ps(
		handedness, _mm256_setzero_ps(), _CMP_LT_OQ);
	const __m256 sign = _mm256_blendv_ps(
		one, _mm256_set1_ps(-1.0f), flipMask);
	r1x = _mm256_mul_ps(r1x, sign);
	r1y = _mm256_mul_ps(r1y, sign);
	r1z = _mm256_mul_ps(r1z, sign);
	r2x = _mm256_mul_ps(r2x, sign);
	r2y = _mm256_mul_ps(r2y, sign);
	r2z = _mm256_mul_ps(r2z, sign);

	const __m256 shapeX = _mm256_loadu_ps(input.shapeX);
	const __m256 shapeY = _mm256_loadu_ps(input.shapeY);
	const __m256 shapeZ = _mm256_loadu_ps(input.shapeZ);
	const __m256 restVolume = _mm256_loadu_ps(input.restVolume);
	const __m256 determinantGradientX = avbdLinearCombination3(
		deformationCof0x, deformationCof1x, deformationCof2x,
		shapeX, shapeY, shapeZ);
	const __m256 determinantGradientY = avbdLinearCombination3(
		deformationCof0y, deformationCof1y, deformationCof2y,
		shapeX, shapeY, shapeZ);
	const __m256 determinantGradientZ = avbdLinearCombination3(
		deformationCof0z, deformationCof1z, deformationCof2z,
		shapeX, shapeY, shapeZ);
	const __m256 twoMu = _mm256_set1_ps(2.0f * mu);
	const __m256 strainTrace = _mm256_sub_ps(
		_mm256_add_ps(
			_mm256_add_ps(
				avbdDot3(r0x, r0y, r0z, f0x, f0y, f0z),
				avbdDot3(r1x, r1y, r1z, f1x, f1y, f1z)),
			avbdDot3(r2x, r2y, r2z, f2x, f2y, f2z)),
		_mm256_set1_ps(3.0f));
	const __m256 lambdaTrace = _mm256_mul_ps(
		_mm256_set1_ps(lambda), strainTrace);

	const __m256 p0x = _mm256_fmadd_ps(
		r0x, lambdaTrace, _mm256_mul_ps(_mm256_sub_ps(f0x, r0x), twoMu));
	const __m256 p0y = _mm256_fmadd_ps(
		r0y, lambdaTrace, _mm256_mul_ps(_mm256_sub_ps(f0y, r0y), twoMu));
	const __m256 p0z = _mm256_fmadd_ps(
		r0z, lambdaTrace, _mm256_mul_ps(_mm256_sub_ps(f0z, r0z), twoMu));
	const __m256 p1x = _mm256_fmadd_ps(
		r1x, lambdaTrace, _mm256_mul_ps(_mm256_sub_ps(f1x, r1x), twoMu));
	const __m256 p1y = _mm256_fmadd_ps(
		r1y, lambdaTrace, _mm256_mul_ps(_mm256_sub_ps(f1y, r1y), twoMu));
	const __m256 p1z = _mm256_fmadd_ps(
		r1z, lambdaTrace, _mm256_mul_ps(_mm256_sub_ps(f1z, r1z), twoMu));
	const __m256 p2x = _mm256_fmadd_ps(
		r2x, lambdaTrace, _mm256_mul_ps(_mm256_sub_ps(f2x, r2x), twoMu));
	const __m256 p2y = _mm256_fmadd_ps(
		r2y, lambdaTrace, _mm256_mul_ps(_mm256_sub_ps(f2y, r2y), twoMu));
	const __m256 p2z = _mm256_fmadd_ps(
		r2z, lambdaTrace, _mm256_mul_ps(_mm256_sub_ps(f2z, r2z), twoMu));
	const __m256 negativeVolume = _mm256_sub_ps(
		_mm256_setzero_ps(), restVolume);
	const __m256 forceX = _mm256_mul_ps(
		avbdLinearCombination3(
			p0x, p1x, p2x, shapeX, shapeY, shapeZ),
		negativeVolume);
	const __m256 forceY = _mm256_mul_ps(
		avbdLinearCombination3(
			p0y, p1y, p2y, shapeX, shapeY, shapeZ),
		negativeVolume);
	const __m256 forceZ = _mm256_mul_ps(
		avbdLinearCombination3(
			p0z, p1z, p2z, shapeX, shapeY, shapeZ),
		negativeVolume);

	const __m256 rotatedX = avbdLinearCombination3(
		r0x, r1x, r2x, shapeX, shapeY, shapeZ);
	const __m256 rotatedY = avbdLinearCombination3(
		r0y, r1y, r2y, shapeX, shapeY, shapeZ);
	const __m256 rotatedZ = avbdLinearCombination3(
		r0z, r1z, r2z, shapeX, shapeY, shapeZ);
	const __m256 diagonal = _mm256_mul_ps(
		_mm256_mul_ps(
			twoMu, _mm256_loadu_ps(input.shapeNormSq)),
		restVolume);
	const __m256 outerScale = _mm256_mul_ps(
		_mm256_set1_ps(lambda), restVolume);
	const __m256 hxx = _mm256_fmadd_ps(
		_mm256_mul_ps(rotatedX, rotatedX), outerScale, diagonal);
	const __m256 hyy = _mm256_fmadd_ps(
		_mm256_mul_ps(rotatedY, rotatedY), outerScale, diagonal);
	const __m256 hzz = _mm256_fmadd_ps(
		_mm256_mul_ps(rotatedZ, rotatedZ), outerScale, diagonal);
	const __m256 hxy = _mm256_mul_ps(
		_mm256_mul_ps(rotatedX, rotatedY), outerScale);
	const __m256 hxz = _mm256_mul_ps(
		_mm256_mul_ps(rotatedX, rotatedZ), outerScale);
	const __m256 hyz = _mm256_mul_ps(
		_mm256_mul_ps(rotatedY, rotatedZ), outerScale);
	valid = _mm256_and_ps(valid, avbdAllFinite9(
		forceX, forceY, forceZ, hxx, hyy, hzz, hxy, hxz, hyz));
	valid = _mm256_and_ps(valid, avbdFinite8(deformationDeterminant));
	valid = _mm256_and_ps(valid, avbdFinite8(determinantGradientX));
	valid = _mm256_and_ps(valid, avbdFinite8(determinantGradientY));
	valid = _mm256_and_ps(valid, avbdFinite8(determinantGradientZ));

	_mm256_storeu_ps(output.forceX, forceX);
	_mm256_storeu_ps(output.forceY, forceY);
	_mm256_storeu_ps(output.forceZ, forceZ);
	_mm256_storeu_ps(output.hessianXX, hxx);
	_mm256_storeu_ps(output.hessianYY, hyy);
	_mm256_storeu_ps(output.hessianZZ, hzz);
	_mm256_storeu_ps(output.hessianXY, hxy);
	_mm256_storeu_ps(output.hessianXZ, hxz);
	_mm256_storeu_ps(output.hessianYZ, hyz);
	_mm256_storeu_ps(output.determinant, deformationDeterminant);
	_mm256_storeu_ps(output.determinantGradientX, determinantGradientX);
	_mm256_storeu_ps(output.determinantGradientY, determinantGradientY);
	_mm256_storeu_ps(output.determinantGradientZ, determinantGradientZ);
	output.validMask = PxU8(_mm256_movemask_ps(valid));
	output.padding[0] = output.padding[1] = output.padding[2] = 0;
	_mm256_zeroupper();
}

void avbdCpuIsaAvx2FmaNeoHookeanTetPacket8(
	const AvbdTetMaterialPacket8Input& input,
	PxF32 mu, PxF32 lambda, PxF32 alpha,
	AvbdTetMaterialPacket8Output& output)
{
	const __m256 e1x = _mm256_loadu_ps(input.e1X);
	const __m256 e1y = _mm256_loadu_ps(input.e1Y);
	const __m256 e1z = _mm256_loadu_ps(input.e1Z);
	const __m256 e2x = _mm256_loadu_ps(input.e2X);
	const __m256 e2y = _mm256_loadu_ps(input.e2Y);
	const __m256 e2z = _mm256_loadu_ps(input.e2Z);
	const __m256 e3x = _mm256_loadu_ps(input.e3X);
	const __m256 e3y = _mm256_loadu_ps(input.e3Y);
	const __m256 e3z = _mm256_loadu_ps(input.e3Z);

	const __m256 dm0x = _mm256_loadu_ps(input.dm0X);
	const __m256 dm0y = _mm256_loadu_ps(input.dm0Y);
	const __m256 dm0z = _mm256_loadu_ps(input.dm0Z);
	const __m256 dm1x = _mm256_loadu_ps(input.dm1X);
	const __m256 dm1y = _mm256_loadu_ps(input.dm1Y);
	const __m256 dm1z = _mm256_loadu_ps(input.dm1Z);
	const __m256 dm2x = _mm256_loadu_ps(input.dm2X);
	const __m256 dm2y = _mm256_loadu_ps(input.dm2Y);
	const __m256 dm2z = _mm256_loadu_ps(input.dm2Z);

	const __m256 f0x = avbdLinearCombination3(
		e1x, e2x, e3x, dm0x, dm0y, dm0z);
	const __m256 f0y = avbdLinearCombination3(
		e1y, e2y, e3y, dm0x, dm0y, dm0z);
	const __m256 f0z = avbdLinearCombination3(
		e1z, e2z, e3z, dm0x, dm0y, dm0z);
	const __m256 f1x = avbdLinearCombination3(
		e1x, e2x, e3x, dm1x, dm1y, dm1z);
	const __m256 f1y = avbdLinearCombination3(
		e1y, e2y, e3y, dm1x, dm1y, dm1z);
	const __m256 f1z = avbdLinearCombination3(
		e1z, e2z, e3z, dm1x, dm1y, dm1z);
	const __m256 f2x = avbdLinearCombination3(
		e1x, e2x, e3x, dm2x, dm2y, dm2z);
	const __m256 f2y = avbdLinearCombination3(
		e1y, e2y, e3y, dm2x, dm2y, dm2z);
	const __m256 f2z = avbdLinearCombination3(
		e1z, e2z, e3z, dm2x, dm2y, dm2z);

	const __m256 cof0x = avbdCrossX(f1y, f1z, f2y, f2z);
	const __m256 cof0y = avbdCrossY(f1x, f1z, f2x, f2z);
	const __m256 cof0z = avbdCrossZ(f1x, f1y, f2x, f2y);
	const __m256 cof1x = avbdCrossX(f2y, f2z, f0y, f0z);
	const __m256 cof1y = avbdCrossY(f2x, f2z, f0x, f0z);
	const __m256 cof1z = avbdCrossZ(f2x, f2y, f0x, f0y);
	const __m256 cof2x = avbdCrossX(f0y, f0z, f1y, f1z);
	const __m256 cof2y = avbdCrossY(f0x, f0z, f1x, f1z);
	const __m256 cof2z = avbdCrossZ(f0x, f0y, f1x, f1y);
	const __m256 determinant = avbdDot3(
		f0x, f0y, f0z, cof0x, cof0y, cof0z);

	const __m256 shapeX = _mm256_loadu_ps(input.shapeX);
	const __m256 shapeY = _mm256_loadu_ps(input.shapeY);
	const __m256 shapeZ = _mm256_loadu_ps(input.shapeZ);
	const __m256 shapeNormSq = _mm256_loadu_ps(input.shapeNormSq);
	const __m256 restVolume = _mm256_loadu_ps(input.restVolume);
	const __m256 fmX = avbdLinearCombination3(
		f0x, f1x, f2x, shapeX, shapeY, shapeZ);
	const __m256 fmY = avbdLinearCombination3(
		f0y, f1y, f2y, shapeX, shapeY, shapeZ);
	const __m256 fmZ = avbdLinearCombination3(
		f0z, f1z, f2z, shapeX, shapeY, shapeZ);
	const __m256 cofmX = avbdLinearCombination3(
		cof0x, cof1x, cof2x, shapeX, shapeY, shapeZ);
	const __m256 cofmY = avbdLinearCombination3(
		cof0y, cof1y, cof2y, shapeX, shapeY, shapeZ);
	const __m256 cofmZ = avbdLinearCombination3(
		cof0z, cof1z, cof2z, shapeX, shapeY, shapeZ);

	const __m256 mu8 = _mm256_set1_ps(mu);
	const __m256 lambda8 = _mm256_set1_ps(lambda);
	const __m256 volumeScale = _mm256_sub_ps(
		_mm256_setzero_ps(), restVolume);
	const __m256 safeDeterminant = _mm256_max_ps(
		determinant, _mm256_set1_ps(0.05f));
	const __m256 volumetricForceScale = _mm256_mul_ps(
		lambda8, _mm256_sub_ps(safeDeterminant, _mm256_set1_ps(alpha)));
	const __m256 forceX = _mm256_mul_ps(
		_mm256_fmadd_ps(cofmX, volumetricForceScale,
			_mm256_mul_ps(fmX, mu8)), volumeScale);
	const __m256 forceY = _mm256_mul_ps(
		_mm256_fmadd_ps(cofmY, volumetricForceScale,
			_mm256_mul_ps(fmY, mu8)), volumeScale);
	const __m256 forceZ = _mm256_mul_ps(
		_mm256_fmadd_ps(cofmZ, volumetricForceScale,
			_mm256_mul_ps(fmZ, mu8)), volumeScale);

	const __m256 outerScale = _mm256_mul_ps(lambda8, restVolume);
	__m256 diagonal = _mm256_mul_ps(
		_mm256_mul_ps(mu8, shapeNormSq), restVolume);
	const __m256 compression = _mm256_max_ps(
		_mm256_sub_ps(_mm256_set1_ps(0.5f), determinant),
		_mm256_setzero_ps());
	diagonal = _mm256_fmadd_ps(
		_mm256_mul_ps(_mm256_mul_ps(compression, lambda8), restVolume),
		shapeNormSq, diagonal);
	const __m256 hxx = _mm256_fmadd_ps(
		_mm256_mul_ps(cofmX, cofmX), outerScale, diagonal);
	const __m256 hyy = _mm256_fmadd_ps(
		_mm256_mul_ps(cofmY, cofmY), outerScale, diagonal);
	const __m256 hzz = _mm256_fmadd_ps(
		_mm256_mul_ps(cofmZ, cofmZ), outerScale, diagonal);
	const __m256 hxy = _mm256_mul_ps(
		_mm256_mul_ps(cofmX, cofmY), outerScale);
	const __m256 hxz = _mm256_mul_ps(
		_mm256_mul_ps(cofmX, cofmZ), outerScale);
	const __m256 hyz = _mm256_mul_ps(
		_mm256_mul_ps(cofmY, cofmZ), outerScale);

	__m256 valid = avbdAllFinite9(
		f0x, f0y, f0z, f1x, f1y, f1z, f2x, f2y, f2z);
	valid = _mm256_and_ps(valid, avbdAllFinite9(
		forceX, forceY, forceZ, hxx, hyy, hzz, hxy, hxz, hyz));
	valid = _mm256_and_ps(valid, avbdFinite8(determinant));
	valid = _mm256_and_ps(valid, avbdFinite8(cofmX));
	valid = _mm256_and_ps(valid, avbdFinite8(cofmY));
	valid = _mm256_and_ps(valid, avbdFinite8(cofmZ));
	valid = _mm256_and_ps(valid, avbdFinite8(shapeX));
	valid = _mm256_and_ps(valid, avbdFinite8(shapeY));
	valid = _mm256_and_ps(valid, avbdFinite8(shapeZ));
	valid = _mm256_and_ps(valid, avbdFinite8(shapeNormSq));
	valid = _mm256_and_ps(valid, avbdFinite8(restVolume));

	_mm256_storeu_ps(output.forceX, forceX);
	_mm256_storeu_ps(output.forceY, forceY);
	_mm256_storeu_ps(output.forceZ, forceZ);
	_mm256_storeu_ps(output.hessianXX, hxx);
	_mm256_storeu_ps(output.hessianYY, hyy);
	_mm256_storeu_ps(output.hessianZZ, hzz);
	_mm256_storeu_ps(output.hessianXY, hxy);
	_mm256_storeu_ps(output.hessianXZ, hxz);
	_mm256_storeu_ps(output.hessianYZ, hyz);
	_mm256_storeu_ps(output.determinant, determinant);
	_mm256_storeu_ps(output.determinantGradientX, cofmX);
	_mm256_storeu_ps(output.determinantGradientY, cofmY);
	_mm256_storeu_ps(output.determinantGradientZ, cofmZ);
	output.validMask = PxU8(_mm256_movemask_ps(valid));
	output.padding[0] = output.padding[1] = output.padding[2] = 0;
	_mm256_zeroupper();
}

#if !defined(PX_AVBD_EXCLUDE_EXPERIMENTAL_RIGID_SIMD)
#pragma float_control(precise, on, push)
void avbdCpuIsaAvx2FmaRigidLdltPacket8(
	const AvbdRigidLdltPacket8Input& input,
	AvbdRigidLdltPacket8Output& output)
{
	__m256 yl[3];
	__m256 ya[3];
	__m256 xl[3];
	__m256 xa[3];

	const auto load8 = [](const PxF32* value) {
		return _mm256_loadu_ps(value);
	};
	const auto fnmadd8 = [](__m256 a, __m256 b, __m256 c) {
		return _mm256_fnmadd_ps(a, b, c);
	};

	yl[0] = load8(input.rhsLinear[0]);
	yl[1] = _mm256_sub_ps(
		load8(input.rhsLinear[1]),
		_mm256_mul_ps(load8(input.linearLinear[1][0]), yl[0]));
	yl[2] = _mm256_sub_ps(
		load8(input.rhsLinear[2]),
		_mm256_mul_ps(load8(input.linearLinear[2][0]), yl[0]));
	yl[2] = _mm256_sub_ps(
		yl[2], _mm256_mul_ps(load8(input.linearLinear[2][1]), yl[1]));

	ya[0] = fnmadd8(load8(input.angularLinear[0][0]), yl[0],
		load8(input.rhsAngular[0]));
	ya[0] = fnmadd8(load8(input.angularLinear[0][1]), yl[1], ya[0]);
	ya[0] = fnmadd8(load8(input.angularLinear[0][2]), yl[2], ya[0]);
	ya[1] = fnmadd8(load8(input.angularLinear[1][0]), yl[0],
		load8(input.rhsAngular[1]));
	ya[1] = fnmadd8(load8(input.angularLinear[1][1]), yl[1], ya[1]);
	ya[1] = fnmadd8(load8(input.angularLinear[1][2]), yl[2], ya[1]);
	ya[1] = _mm256_sub_ps(
		ya[1], _mm256_mul_ps(load8(input.angularAngular[1][0]), ya[0]));
	ya[2] = fnmadd8(load8(input.angularLinear[2][0]), yl[0],
		load8(input.rhsAngular[2]));
	ya[2] = fnmadd8(load8(input.angularLinear[2][1]), yl[1], ya[2]);
	ya[2] = fnmadd8(load8(input.angularLinear[2][2]), yl[2], ya[2]);
	ya[2] = _mm256_sub_ps(
		ya[2], _mm256_mul_ps(load8(input.angularAngular[2][0]), ya[0]));
	ya[2] = _mm256_sub_ps(
		ya[2], _mm256_mul_ps(load8(input.angularAngular[2][1]), ya[1]));

	for(PxU32 i = 0; i < 3; ++i)
	{
		yl[i] = _mm256_div_ps(yl[i], load8(input.diagonalLinear[i]));
		ya[i] = _mm256_div_ps(ya[i], load8(input.diagonalAngular[i]));
	}

	xa[2] = ya[2];
	xa[1] = _mm256_sub_ps(
		ya[1], _mm256_mul_ps(load8(input.angularAngular[2][1]), xa[2]));
	xa[0] = _mm256_sub_ps(
		ya[0], _mm256_mul_ps(load8(input.angularAngular[1][0]), xa[1]));
	xa[0] = _mm256_sub_ps(
		xa[0], _mm256_mul_ps(load8(input.angularAngular[2][0]), xa[2]));

	xl[2] = yl[2];
	xl[2] = fnmadd8(load8(input.angularLinear[0][2]), xa[0], xl[2]);
	xl[2] = fnmadd8(load8(input.angularLinear[1][2]), xa[1], xl[2]);
	xl[2] = fnmadd8(load8(input.angularLinear[2][2]), xa[2], xl[2]);
	xl[1] = _mm256_sub_ps(
		yl[1], _mm256_mul_ps(load8(input.linearLinear[2][1]), xl[2]));
	xl[1] = fnmadd8(load8(input.angularLinear[0][1]), xa[0], xl[1]);
	xl[1] = fnmadd8(load8(input.angularLinear[1][1]), xa[1], xl[1]);
	xl[1] = fnmadd8(load8(input.angularLinear[2][1]), xa[2], xl[1]);
	xl[0] = _mm256_sub_ps(
		yl[0], _mm256_mul_ps(load8(input.linearLinear[1][0]), xl[1]));
	xl[0] = _mm256_sub_ps(
		xl[0], _mm256_mul_ps(load8(input.linearLinear[2][0]), xl[2]));
	xl[0] = fnmadd8(load8(input.angularLinear[0][0]), xa[0], xl[0]);
	xl[0] = fnmadd8(load8(input.angularLinear[1][0]), xa[1], xl[0]);
	xl[0] = fnmadd8(load8(input.angularLinear[2][0]), xa[2], xl[0]);

	for(PxU32 i = 0; i < 3; ++i)
	{
		_mm256_storeu_ps(output.linear[i], xl[i]);
		_mm256_storeu_ps(output.angular[i], xa[i]);
	}
	_mm256_zeroupper();
}
#pragma float_control(pop)
void avbdCpuIsaSse2RigidLdltFactorSolvePacket8(
	const AvbdRigidLdltFactorSolvePacket8Input& input,
	AvbdRigidLdltFactorSolvePacket8Output& output);

#pragma float_control(precise, on, push)
namespace
{
	// P60.1: keep the raw matrix inputs lane-major, but load each block only
	// when its factor phase consumes it. This removes the three temporary input
	// matrices that caused the fused P59 kernel's register/stack explosion.
	// The public entry remains isolated until a producer-native storage contract
	// exists; no solver hot loop calls this function today.
	struct AvbdRigidLdltFactorPacket8
	{
		__m256 lLL[3][3];
		__m256 lAL[3][3];
		__m256 lAA[3][3];
		__m256 dL[3];
		__m256 dA[3];
	};

	static PX_NOINLINE PxU32 avbdAvx2FmaFactorPacket8(
		const AvbdRigidLdltFactorSolvePacket8Input& input,
		AvbdRigidLdltFactorPacket8& factor)
	{
		const auto divide8 = [](const __m256 numerator, const __m256 denominator) {
			return _mm256_div_ps(numerator, denominator);
		};
		const auto load8 = [](const PxF32* value) {
			return _mm256_loadu_ps(value);
		};
		const auto subtractProduct3 = [](const __m256 lhs, const __m256 diagonal,
			const __m256 rhs, const __m256 sum) {
			return _mm256_sub_ps(sum,
				_mm256_mul_ps(_mm256_mul_ps(lhs, diagonal), rhs));
		};
		const __m256 zero = _mm256_setzero_ps();
		const __m256 one = _mm256_set1_ps(1.0f);
		__m256 schur[3][3];
		__m256 valid = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
		const __m256 singular = _mm256_set1_ps(input.singularThreshold);

		for(PxU32 i = 0; i < 3; ++i)
		{
			for(PxU32 j = 0; j <= i; ++j)
			{
				__m256 sum = load8(input.linearLinear[i][j]);
				for(PxU32 k = 0; k < j; ++k)
					sum = subtractProduct3(factor.lLL[i][k], factor.dL[k], factor.lLL[j][k], sum);
				if(i == j)
				{
					factor.dL[i] = sum;
					valid = _mm256_and_ps(valid, _mm256_cmp_ps(
						sum, singular, _CMP_GT_OQ));
					valid = _mm256_and_ps(valid, avbdFinite8(sum));
					factor.lLL[i][j] = one;
				}
				else
					factor.lLL[i][j] = divide8(sum, factor.dL[j]);
			}
		}
		for(PxU32 i = 0; i < 3; ++i)
		{
			for(PxU32 j = 0; j < 3; ++j)
			{
				__m256 sum = load8(input.angularLinear[i][j]);
				for(PxU32 k = 0; k < j; ++k)
					sum = subtractProduct3(factor.lAL[i][k], factor.dL[k], factor.lLL[j][k], sum);
				factor.lAL[i][j] = divide8(sum, factor.dL[j]);
			}
		}
		for(PxU32 i = 0; i < 3; ++i)
		{
			for(PxU32 j = 0; j < 3; ++j)
			{
				__m256 sum = load8(input.angularAngular[i][j]);
				for(PxU32 k = 0; k < 3; ++k)
					sum = subtractProduct3(factor.lAL[i][k], factor.dL[k], factor.lAL[j][k], sum);
				schur[i][j] = sum;
			}
		}
		for(PxU32 i = 0; i < 3; ++i)
		{
			for(PxU32 j = 0; j <= i; ++j)
			{
				__m256 sum = schur[i][j];
				for(PxU32 k = 0; k < j; ++k)
					sum = subtractProduct3(factor.lAA[i][k], factor.dA[k], factor.lAA[j][k], sum);
				if(i == j)
				{
					factor.dA[i] = sum;
					valid = _mm256_and_ps(valid, _mm256_cmp_ps(
						sum, singular, _CMP_GT_OQ));
					valid = _mm256_and_ps(valid, avbdFinite8(sum));
					factor.lAA[i][j] = one;
				}
				else
					factor.lAA[i][j] = divide8(sum, factor.dA[j]);
			}
		}

		__m256 minD = _mm256_set1_ps(FLT_MAX);
		__m256 maxD = zero;
		for(PxU32 i = 0; i < 3; ++i)
		{
			const __m256 validL = _mm256_cmp_ps(factor.dL[i], zero, _CMP_GT_OQ);
			const __m256 validA = _mm256_cmp_ps(factor.dA[i], zero, _CMP_GT_OQ);
			valid = _mm256_and_ps(valid, validL);
			valid = _mm256_and_ps(valid, validA);
			minD = _mm256_min_ps(minD, _mm256_blendv_ps(
				_mm256_set1_ps(FLT_MAX), factor.dL[i], validL));
			minD = _mm256_min_ps(minD, _mm256_blendv_ps(
				_mm256_set1_ps(FLT_MAX), factor.dA[i], validA));
			maxD = _mm256_max_ps(maxD, _mm256_blendv_ps(zero, factor.dL[i], validL));
			maxD = _mm256_max_ps(maxD, _mm256_blendv_ps(zero, factor.dA[i], validA));
		}
		const __m256 condition = divide8(maxD, minD);
		valid = _mm256_and_ps(valid, avbdFinite8(condition));
		valid = _mm256_and_ps(valid, _mm256_cmp_ps(
			condition, _mm256_set1_ps(input.conditionNumberThreshold), _CMP_LT_OQ));
		return PxU32(_mm256_movemask_ps(valid));
	}

	static PX_NOINLINE void avbdAvx2FmaSolvePacket8(
		const AvbdRigidLdltFactorSolvePacket8Input& input,
		const AvbdRigidLdltFactorPacket8& factor,
		AvbdRigidLdltFactorSolvePacket8Output& output)
	{
		const auto divide8 = [](const __m256 numerator, const __m256 denominator) {
			return _mm256_div_ps(numerator, denominator);
		};
		const auto load8 = [](const PxF32* value) {
			return _mm256_loadu_ps(value);
		};
		__m256 yL[3], yA[3], xL[3], xA[3];
		for(PxU32 i = 0; i < 3; ++i)
		{
			yL[i] = load8(input.rhsLinear[i]);
			for(PxU32 j = 0; j < i; ++j)
				yL[i] = _mm256_sub_ps(yL[i], _mm256_mul_ps(factor.lLL[i][j], yL[j]));
		}
		for(PxU32 i = 0; i < 3; ++i)
		{
			yA[i] = load8(input.rhsAngular[i]);
			for(PxU32 j = 0; j < 3; ++j)
				yA[i] = _mm256_sub_ps(yA[i], _mm256_mul_ps(factor.lAL[i][j], yL[j]));
			for(PxU32 j = 0; j < i; ++j)
				yA[i] = _mm256_sub_ps(yA[i], _mm256_mul_ps(factor.lAA[i][j], yA[j]));
		}
		for(PxU32 i = 0; i < 3; ++i)
		{
			yL[i] = divide8(yL[i], factor.dL[i]);
			yA[i] = divide8(yA[i], factor.dA[i]);
		}
		for(PxI32 i = 2; i >= 0; --i)
		{
			xA[i] = yA[i];
			for(PxU32 j = PxU32(i + 1); j < 3; ++j)
				xA[i] = _mm256_sub_ps(xA[i], _mm256_mul_ps(factor.lAA[j][i], xA[j]));
		}
		for(PxI32 i = 2; i >= 0; --i)
		{
			xL[i] = yL[i];
			for(PxU32 j = PxU32(i + 1); j < 3; ++j)
				xL[i] = _mm256_sub_ps(xL[i], _mm256_mul_ps(factor.lLL[j][i], xL[j]));
			for(PxU32 j = 0; j < 3; ++j)
				xL[i] = _mm256_sub_ps(xL[i], _mm256_mul_ps(factor.lAL[j][i], xA[j]));
		}
		for(PxU32 i = 0; i < 3; ++i)
		{
			_mm256_storeu_ps(output.linear[i], xL[i]);
			_mm256_storeu_ps(output.angular[i], xA[i]);
		}
		output.validMask = 0xFF;
		output.padding[0] = output.padding[1] = output.padding[2] = 0;
		_mm256_zeroupper();
	}
}

void avbdCpuIsaAvx2FmaRigidLdltFactorSolvePacket8(
	const AvbdRigidLdltFactorSolvePacket8Input& input,
	AvbdRigidLdltFactorSolvePacket8Output& output)
{
	AvbdRigidLdltFactorPacket8 factor;
	if(avbdAvx2FmaFactorPacket8(input, factor) != 0xFFu)
	{
		avbdCpuIsaSse2RigidLdltFactorSolvePacket8(input, output);
		return;
	}
	avbdAvx2FmaSolvePacket8(input, factor, output);
}

namespace
{
PX_FORCE_INLINE __m256 avbdLocalResponseMask8(PxU8 mask)
{
	return _mm256_castsi256_ps(_mm256_set_epi32(
		(mask & 0x80u) ? -1 : 0,
		(mask & 0x40u) ? -1 : 0,
		(mask & 0x20u) ? -1 : 0,
		(mask & 0x10u) ? -1 : 0,
		(mask & 0x08u) ? -1 : 0,
		(mask & 0x04u) ? -1 : 0,
		(mask & 0x02u) ? -1 : 0,
		(mask & 0x01u) ? -1 : 0));
}

PX_FORCE_INLINE __m256 avbdLocalResponseFmaProduct8(__m256 lhs, __m256 rhs)
{
	return _mm256_fmadd_ps(lhs, rhs, _mm256_setzero_ps());
}
}

void avbdCpuIsaAvx2FmaRigidLocalResponsePacket8(
	const AvbdRigidLocalResponsePacket8Input& input,
	AvbdRigidLdltFactorSolvePacket8Input& target, PxU8& touchingMask)
{
	const __m256 active = avbdLocalResponseMask8(input.activeMask);
	const __m256 zero = _mm256_setzero_ps();
	const __m256 nonnegativeLinear = _mm256_max_ps(
		_mm256_loadu_ps(input.linearScale), zero);
	const __m256 nonnegativeAngular = _mm256_max_ps(
		_mm256_loadu_ps(input.angularScale), zero);
	const __m256 invCompliance = _mm256_loadu_ps(input.invCompliance);
	const __m256 crossScale = _mm256_sqrt_ps(_mm256_mul_ps(
		nonnegativeLinear, nonnegativeAngular));
	// Keep the reviewed scalar multiplication order for matrix terms. FMA is
	// still used for the independent RHS products below, while these explicit
	// multiplies preserve the scalar rounding sequence for a live fingerprint.
	const __m256 scaleLinear = _mm256_mul_ps(invCompliance, nonnegativeLinear);
	const __m256 scaleCross = _mm256_mul_ps(invCompliance, crossScale);
	const __m256 scaleAngular = _mm256_mul_ps(invCompliance, nonnegativeAngular);
	const __m256 force = _mm256_loadu_ps(input.force);
	const __m256 rhsLinearScale = avbdLocalResponseFmaProduct8(
		force, nonnegativeLinear);
	const __m256 rhsAngularScale = avbdLocalResponseFmaProduct8(
		force, nonnegativeAngular);
	for(PxU32 i = 0; i < 3; ++i)
	{
		const __m256 gradPosI = _mm256_loadu_ps(input.gradPos[i]);
		const __m256 gradRotI = _mm256_loadu_ps(input.gradRot[i]);
		const __m256 rhsL = _mm256_and_ps(active,
			avbdLocalResponseFmaProduct8(gradPosI, rhsLinearScale));
		const __m256 rhsA = _mm256_and_ps(active,
			avbdLocalResponseFmaProduct8(gradRotI, rhsAngularScale));
		_mm256_storeu_ps(target.rhsLinear[i], _mm256_add_ps(
			_mm256_loadu_ps(target.rhsLinear[i]), rhsL));
		_mm256_storeu_ps(target.rhsAngular[i], _mm256_add_ps(
			_mm256_loadu_ps(target.rhsAngular[i]), rhsA));
		for(PxU32 j = 0; j < 3; ++j)
		{
			const __m256 gradPosJ = _mm256_loadu_ps(input.gradPos[j]);
			const __m256 gradRotJ = _mm256_loadu_ps(input.gradRot[j]);
			const __m256 deltaLL = _mm256_and_ps(active,
				_mm256_mul_ps(_mm256_mul_ps(scaleLinear, gradPosI), gradPosJ));
			const __m256 deltaAL = _mm256_and_ps(active,
				_mm256_mul_ps(_mm256_mul_ps(scaleCross, gradRotI), gradPosJ));
			const __m256 deltaAA = _mm256_and_ps(active,
				_mm256_mul_ps(_mm256_mul_ps(scaleAngular, gradRotI), gradRotJ));
			_mm256_storeu_ps(target.linearLinear[i][j], _mm256_add_ps(
				_mm256_loadu_ps(target.linearLinear[i][j]), deltaLL));
			_mm256_storeu_ps(target.angularLinear[i][j], _mm256_add_ps(
				_mm256_loadu_ps(target.angularLinear[i][j]), deltaAL));
			_mm256_storeu_ps(target.angularAngular[i][j], _mm256_add_ps(
				_mm256_loadu_ps(target.angularAngular[i][j]), deltaAA));
		}
	}
	touchingMask |= PxU8(input.touchingMask & input.activeMask);
	_mm256_zeroupper();
}

void avbdCpuIsaAvx2FmaRigidD6ResponsePacket8(
	const AvbdRigidD6ResponsePacket8Input& input,
	AvbdRigidLdltFactorSolvePacket8Input& target, PxU8& touchingMask)
{
	const __m256 active = avbdLocalResponseMask8(input.activeMask);
	const __m256 zero = _mm256_setzero_ps();
	for(PxU32 row = 0; row < eAVBD_RIGID_D6_RESPONSE_ROW_PACKET_COUNT; ++row)
	{
		const __m256 nonnegativeLinear = _mm256_max_ps(
			_mm256_loadu_ps(input.linearScale[row]), zero);
		const __m256 nonnegativeAngular = _mm256_max_ps(
			_mm256_loadu_ps(input.angularScale[row]), zero);
		const __m256 invCompliance =
			_mm256_loadu_ps(input.invCompliance[row]);
		const __m256 crossScale = _mm256_sqrt_ps(_mm256_mul_ps(
			nonnegativeLinear, nonnegativeAngular));
		const __m256 scaleLinear = _mm256_mul_ps(
			invCompliance, nonnegativeLinear);
		const __m256 scaleCross = _mm256_mul_ps(invCompliance, crossScale);
		const __m256 scaleAngular = _mm256_mul_ps(
			invCompliance, nonnegativeAngular);
		const __m256 force = _mm256_loadu_ps(input.force[row]);
		const __m256 rhsLinearScale = avbdLocalResponseFmaProduct8(
			force, nonnegativeLinear);
		const __m256 rhsAngularScale = avbdLocalResponseFmaProduct8(
			force, nonnegativeAngular);
		for(PxU32 i = 0; i < 3; ++i)
		{
			const __m256 gradPosI = _mm256_loadu_ps(input.gradPos[row][i]);
			const __m256 gradRotI = _mm256_loadu_ps(input.gradRot[row][i]);
			const __m256 rhsL = _mm256_and_ps(active,
				avbdLocalResponseFmaProduct8(gradPosI, rhsLinearScale));
			const __m256 rhsA = _mm256_and_ps(active,
				avbdLocalResponseFmaProduct8(gradRotI, rhsAngularScale));
			_mm256_storeu_ps(target.rhsLinear[i], _mm256_add_ps(
				_mm256_loadu_ps(target.rhsLinear[i]), rhsL));
			_mm256_storeu_ps(target.rhsAngular[i], _mm256_add_ps(
				_mm256_loadu_ps(target.rhsAngular[i]), rhsA));
			for(PxU32 j = 0; j < 3; ++j)
			{
				const __m256 gradPosJ = _mm256_loadu_ps(input.gradPos[row][j]);
				const __m256 gradRotJ = _mm256_loadu_ps(input.gradRot[row][j]);
				const __m256 deltaLL = _mm256_and_ps(active,
					_mm256_mul_ps(_mm256_mul_ps(scaleLinear, gradPosI), gradPosJ));
				const __m256 deltaAL = _mm256_and_ps(active,
					_mm256_mul_ps(_mm256_mul_ps(scaleCross, gradRotI), gradPosJ));
				const __m256 deltaAA = _mm256_and_ps(active,
					_mm256_mul_ps(_mm256_mul_ps(scaleAngular, gradRotI), gradRotJ));
				_mm256_storeu_ps(target.linearLinear[i][j], _mm256_add_ps(
					_mm256_loadu_ps(target.linearLinear[i][j]), deltaLL));
				_mm256_storeu_ps(target.angularLinear[i][j], _mm256_add_ps(
					_mm256_loadu_ps(target.angularLinear[i][j]), deltaAL));
				_mm256_storeu_ps(target.angularAngular[i][j], _mm256_add_ps(
					_mm256_loadu_ps(target.angularAngular[i][j]), deltaAA));
			}
		}
	}
	touchingMask |= PxU8(input.touchingMask & input.activeMask);
	_mm256_zeroupper();
}

void avbdCpuIsaAvx2FmaRigidD6ResponsePacket8View(
	const AvbdRigidD6ResponsePacket8View& input,
	AvbdRigidLdltFactorSolvePacket8Input& target, PxU8& touchingMask)
{
	const __m256 active = avbdLocalResponseMask8(input.activeMask);
	const __m256 zero = _mm256_setzero_ps();
	for(PxU32 row = 0; row < eAVBD_RIGID_D6_RESPONSE_ROW_PACKET_COUNT; ++row)
	{
		const __m256 nonnegativeLinear = _mm256_max_ps(
			_mm256_loadu_ps(input.linearScale[row]), zero);
		const __m256 nonnegativeAngular = _mm256_max_ps(
			_mm256_loadu_ps(input.angularScale[row]), zero);
		const __m256 invCompliance =
			_mm256_loadu_ps(input.invCompliance[row]);
		const __m256 crossScale = _mm256_sqrt_ps(_mm256_mul_ps(
			nonnegativeLinear, nonnegativeAngular));
		const __m256 scaleLinear = _mm256_mul_ps(
			invCompliance, nonnegativeLinear);
		const __m256 scaleCross = _mm256_mul_ps(invCompliance, crossScale);
		const __m256 scaleAngular = _mm256_mul_ps(
			invCompliance, nonnegativeAngular);
		const __m256 force = _mm256_loadu_ps(input.force[row]);
		const __m256 rhsLinearScale = avbdLocalResponseFmaProduct8(
			force, nonnegativeLinear);
		const __m256 rhsAngularScale = avbdLocalResponseFmaProduct8(
			force, nonnegativeAngular);
		for(PxU32 i = 0; i < 3; ++i)
		{
			const __m256 gradPosI = _mm256_loadu_ps(input.gradPos[row][i]);
			const __m256 gradRotI = _mm256_loadu_ps(input.gradRot[row][i]);
			const __m256 rhsL = _mm256_and_ps(active,
				avbdLocalResponseFmaProduct8(gradPosI, rhsLinearScale));
			const __m256 rhsA = _mm256_and_ps(active,
				avbdLocalResponseFmaProduct8(gradRotI, rhsAngularScale));
			_mm256_storeu_ps(target.rhsLinear[i], _mm256_add_ps(
				_mm256_loadu_ps(target.rhsLinear[i]), rhsL));
			_mm256_storeu_ps(target.rhsAngular[i], _mm256_add_ps(
				_mm256_loadu_ps(target.rhsAngular[i]), rhsA));
			for(PxU32 j = 0; j < 3; ++j)
			{
				const __m256 gradPosJ = _mm256_loadu_ps(input.gradPos[row][j]);
				const __m256 gradRotJ = _mm256_loadu_ps(input.gradRot[row][j]);
				const __m256 deltaLL = _mm256_and_ps(active,
					_mm256_mul_ps(_mm256_mul_ps(scaleLinear, gradPosI), gradPosJ));
				const __m256 deltaAL = _mm256_and_ps(active,
					_mm256_mul_ps(_mm256_mul_ps(scaleCross, gradRotI), gradPosJ));
				const __m256 deltaAA = _mm256_and_ps(active,
					_mm256_mul_ps(_mm256_mul_ps(scaleAngular, gradRotI), gradRotJ));
				_mm256_storeu_ps(target.linearLinear[i][j], _mm256_add_ps(
					_mm256_loadu_ps(target.linearLinear[i][j]), deltaLL));
				_mm256_storeu_ps(target.angularLinear[i][j], _mm256_add_ps(
					_mm256_loadu_ps(target.angularLinear[i][j]), deltaAL));
				_mm256_storeu_ps(target.angularAngular[i][j], _mm256_add_ps(
					_mm256_loadu_ps(target.angularAngular[i][j]), deltaAA));
			}
		}
	}
	touchingMask |= PxU8(input.touchingMask & input.activeMask);
	_mm256_zeroupper();
}

namespace
{
	PX_FORCE_INLINE __m256 avbdNormalUpperProduct8(
		__m256 lhs, __m256 rhs, __m256 responseScale, __m256 penalty)
	{
		// The standalone product uses the required FMA ISA without changing the
		// reviewed scalar contract (adding zero is equivalent to one rounded
		// multiply). Response-scale and penalty multiplications remain explicit.
		return _mm256_mul_ps(penalty,
			_mm256_mul_ps(responseScale,
				_mm256_fmadd_ps(lhs, rhs, _mm256_setzero_ps())));
	}

	PX_FORCE_INLINE __m256 avbdNormalActiveMask8(PxU8 activeMask)
	{
		return _mm256_castsi256_ps(_mm256_set_epi32(
			(activeMask & 0x80u) ? -1 : 0,
			(activeMask & 0x40u) ? -1 : 0,
			(activeMask & 0x20u) ? -1 : 0,
			(activeMask & 0x10u) ? -1 : 0,
			(activeMask & 0x08u) ? -1 : 0,
			(activeMask & 0x04u) ? -1 : 0,
			(activeMask & 0x02u) ? -1 : 0,
			(activeMask & 0x01u) ? -1 : 0));
	}

}

void avbdCpuIsaAvx2FmaRigidContactBlockPacket8(
	const AvbdRigidContactBlockPacket8Input& input,
	AvbdRigidLdltFactorSolvePacket8Input& target, PxU8& touchingMask)
{
	const __m256 active = avbdLocalResponseMask8(input.activeMask);
	const __m256 zero = _mm256_setzero_ps();
	const __m256 nonnegativeLinear = _mm256_max_ps(
		_mm256_loadu_ps(input.linearScale), zero);
	const __m256 nonnegativeAngular = _mm256_max_ps(
		_mm256_loadu_ps(input.angularScale), zero);
	const __m256 crossScale = _mm256_sqrt_ps(_mm256_mul_ps(
		nonnegativeLinear, nonnegativeAngular));
	const __m256 scaleLinearBase = nonnegativeLinear;
	const __m256 scaleAngularBase = nonnegativeAngular;
	for(PxU32 row = 0;
		row < eAVBD_RIGID_CONTACT_BLOCK_ROW_COUNT; ++row)
	{
		const __m256 hessian = _mm256_and_ps(active,
			avbdLocalResponseMask8(input.hessianMask[row]));
		const __m256 invCompliance = _mm256_loadu_ps(
			input.invCompliance[row]);
		const __m256 scaleLinear = _mm256_mul_ps(
			invCompliance, scaleLinearBase);
		const __m256 scaleCross = _mm256_mul_ps(
			invCompliance, crossScale);
		const __m256 scaleAngular = _mm256_mul_ps(
			invCompliance, scaleAngularBase);
		const __m256 force = _mm256_loadu_ps(input.force[row]);
		const __m256 rhsLinearScale = avbdLocalResponseFmaProduct8(
			force, scaleLinearBase);
		const __m256 rhsAngularScale = avbdLocalResponseFmaProduct8(
			force, scaleAngularBase);
		for(PxU32 i = 0; i < 3; ++i)
		{
			const __m256 gradPosI = _mm256_loadu_ps(
				input.gradPos[row][i]);
			const __m256 gradRotI = _mm256_loadu_ps(
				input.gradRot[row][i]);
			const __m256 rhsL = _mm256_and_ps(active,
				avbdLocalResponseFmaProduct8(gradPosI, rhsLinearScale));
			const __m256 rhsA = _mm256_and_ps(active,
				avbdLocalResponseFmaProduct8(gradRotI, rhsAngularScale));
			_mm256_storeu_ps(target.rhsLinear[i], _mm256_add_ps(
				_mm256_loadu_ps(target.rhsLinear[i]), rhsL));
			_mm256_storeu_ps(target.rhsAngular[i], _mm256_add_ps(
				_mm256_loadu_ps(target.rhsAngular[i]), rhsA));
			for(PxU32 j = 0; j < 3; ++j)
			{
				const __m256 gradPosJ = _mm256_loadu_ps(
					input.gradPos[row][j]);
				const __m256 gradRotJ = _mm256_loadu_ps(
					input.gradRot[row][j]);
				const __m256 deltaLL = _mm256_and_ps(hessian,
					_mm256_mul_ps(_mm256_mul_ps(scaleLinear, gradPosI),
						gradPosJ));
				const __m256 deltaAL = _mm256_and_ps(hessian,
					_mm256_mul_ps(_mm256_mul_ps(scaleCross, gradRotI),
						gradPosJ));
				const __m256 deltaAA = _mm256_and_ps(hessian,
					_mm256_mul_ps(_mm256_mul_ps(scaleAngular, gradRotI),
						gradRotJ));
				_mm256_storeu_ps(target.linearLinear[i][j], _mm256_add_ps(
					_mm256_loadu_ps(target.linearLinear[i][j]), deltaLL));
				_mm256_storeu_ps(target.angularLinear[i][j], _mm256_add_ps(
					_mm256_loadu_ps(target.angularLinear[i][j]), deltaAL));
				_mm256_storeu_ps(target.angularAngular[i][j], _mm256_add_ps(
					_mm256_loadu_ps(target.angularAngular[i][j]), deltaAA));
			}
		}
	}
	touchingMask |= PxU8(input.touchingMask & input.activeMask);
	_mm256_zeroupper();
}

void avbdCpuIsaAvx2FmaRigidNormalContactPacket8(
	const AvbdRigidNormalContactPacket8Input& input,
	AvbdRigidNormalContactPacket8Output& output)
{
	const __m256 zero = _mm256_setzero_ps();
	const __m256 one = _mm256_set1_ps(1.0f);
	const __m256 two = _mm256_set1_ps(2.0f);
	const __m256 maxFloat = _mm256_set1_ps(PX_MAX_F32);
	const __m256 active = avbdNormalActiveMask8(input.activeMask);
	const __m256 linearScale = _mm256_loadu_ps(input.linearResponseScale);
	const __m256 angularScale = _mm256_loadu_ps(input.angularResponseScale);
	const __m256 nonnegativeLinear = _mm256_max_ps(linearScale, zero);
	const __m256 nonnegativeAngular = _mm256_max_ps(angularScale, zero);
	const __m256 crossScale = _mm256_sqrt_ps(
		_mm256_mul_ps(nonnegativeLinear, nonnegativeAngular));
	const __m256 response = _mm256_or_ps(
		_mm256_cmp_ps(linearScale, zero, _CMP_GT_OQ),
		_mm256_cmp_ps(angularScale, zero, _CMP_GT_OQ));
	const __m256 touching = _mm256_and_ps(active, response);

	const __m256 qx = _mm256_loadu_ps(input.bodyRotation[0]);
	const __m256 qy = _mm256_loadu_ps(input.bodyRotation[1]);
	const __m256 qz = _mm256_loadu_ps(input.bodyRotation[2]);
	const __m256 qw = _mm256_loadu_ps(input.bodyRotation[3]);
	const __m256 vx = _mm256_loadu_ps(input.bodyContactPoint[0]);
	const __m256 vy = _mm256_loadu_ps(input.bodyContactPoint[1]);
	const __m256 vz = _mm256_loadu_ps(input.bodyContactPoint[2]);
	const __m256 tx = _mm256_mul_ps(two, _mm256_sub_ps(
		_mm256_mul_ps(qy, vz), _mm256_mul_ps(qz, vy)));
	const __m256 ty = _mm256_mul_ps(two, _mm256_sub_ps(
		_mm256_mul_ps(qz, vx), _mm256_mul_ps(qx, vz)));
	const __m256 tz = _mm256_mul_ps(two, _mm256_sub_ps(
		_mm256_mul_ps(qx, vy), _mm256_mul_ps(qy, vx)));
	const __m256 rx = _mm256_add_ps(
		_mm256_add_ps(vx, _mm256_mul_ps(qw, tx)),
		_mm256_sub_ps(_mm256_mul_ps(qy, tz), _mm256_mul_ps(qz, ty)));
	const __m256 ry = _mm256_add_ps(
		_mm256_add_ps(vy, _mm256_mul_ps(qw, ty)),
		_mm256_sub_ps(_mm256_mul_ps(qz, tx), _mm256_mul_ps(qx, tz)));
	const __m256 rz = _mm256_add_ps(
		_mm256_add_ps(vz, _mm256_mul_ps(qw, tz)),
		_mm256_sub_ps(_mm256_mul_ps(qx, ty), _mm256_mul_ps(qy, tx)));

	const __m256 worldX = _mm256_add_ps(
		_mm256_loadu_ps(input.bodyPosition[0]), rx);
	const __m256 worldY = _mm256_add_ps(
		_mm256_loadu_ps(input.bodyPosition[1]), ry);
	const __m256 worldZ = _mm256_add_ps(
		_mm256_loadu_ps(input.bodyPosition[2]), rz);
	const __m256 nx = _mm256_loadu_ps(input.normal[0]);
	const __m256 ny = _mm256_loadu_ps(input.normal[1]);
	const __m256 nz = _mm256_loadu_ps(input.normal[2]);
	const __m256 sx = _mm256_loadu_ps(input.staticContactPoint[0]);
	const __m256 sy = _mm256_loadu_ps(input.staticContactPoint[1]);
	const __m256 sz = _mm256_loadu_ps(input.staticContactPoint[2]);
	const __m256 sign = _mm256_loadu_ps(input.sign);
	__m256 geometricGap = _mm256_mul_ps(_mm256_sub_ps(worldX, sx), nx);
	geometricGap = _mm256_add_ps(geometricGap,
		_mm256_mul_ps(_mm256_sub_ps(worldY, sy), ny));
	geometricGap = _mm256_add_ps(geometricGap,
		_mm256_mul_ps(_mm256_sub_ps(worldZ, sz), nz));
	// Match the scalar (A-B).n contract when the packet stores the dynamic
	// body: sign flips only the geometric gap, never the authored penetration.
	const __m256 violation = _mm256_add_ps(
		_mm256_mul_ps(geometricGap, sign),
		_mm256_loadu_ps(input.penetration));
	const __m256 penalty = _mm256_loadu_ps(input.penalty);
	const __m256 lambda = _mm256_loadu_ps(input.lambda);
	const __m256 rawForce = _mm256_min_ps(
		zero, _mm256_add_ps(_mm256_mul_ps(penalty, violation), lambda));
	const __m256 dt = _mm256_loadu_ps(input.dt);
	const __m256 dtPositive = _mm256_cmp_ps(dt, zero, _CMP_GT_OQ);
	const __m256 maxImpulse = _mm256_loadu_ps(input.maxImpulse);
	const __m256 finiteImpulse = _mm256_cmp_ps(maxImpulse, maxFloat, _CMP_LT_OQ);
	const __m256 capped = _mm256_and_ps(
		_mm256_and_ps(dtPositive, finiteImpulse), touching);
	const __m256 safeDt = _mm256_blendv_ps(one, dt, dtPositive);
	const __m256 maxNormalForce = _mm256_div_ps(
		_mm256_max_ps(maxImpulse, zero), safeDt);
	const __m256 saturated = _mm256_and_ps(
		capped, _mm256_cmp_ps(rawForce, _mm256_sub_ps(zero, maxNormalForce),
			_CMP_LT_OQ));
	const __m256 cappedForce = _mm256_max_ps(
		rawForce, _mm256_sub_ps(zero, maxNormalForce));
	const __m256 force = _mm256_blendv_ps(rawForce, cappedForce, capped);
	const __m256 gpx = _mm256_mul_ps(nx, sign);
	const __m256 gpy = _mm256_mul_ps(ny, sign);
	const __m256 gpz = _mm256_mul_ps(nz, sign);
	const __m256 grx = _mm256_mul_ps(_mm256_sub_ps(
		_mm256_mul_ps(ry, nz), _mm256_mul_ps(rz, ny)), sign);
	const __m256 gry = _mm256_mul_ps(_mm256_sub_ps(
		_mm256_mul_ps(rz, nx), _mm256_mul_ps(rx, nz)), sign);
	const __m256 grz = _mm256_mul_ps(_mm256_sub_ps(
		_mm256_mul_ps(rx, ny), _mm256_mul_ps(ry, nx)), sign);
	const __m256 j[6] = {
		_mm256_mul_ps(gpx, linearScale), _mm256_mul_ps(gpy, linearScale),
		_mm256_mul_ps(gpz, linearScale), _mm256_mul_ps(grx, angularScale),
		_mm256_mul_ps(gry, angularScale), _mm256_mul_ps(grz, angularScale)};
	const __m256 hessianGradient[6] = {gpx, gpy, gpz, grx, gry, grz};
	const __m256 rhsMask = _mm256_and_ps(
		touching, _mm256_cmp_ps(force, zero, _CMP_LT_OQ));
	const __m256 hessianMask = _mm256_andnot_ps(saturated, touching);
	for(PxU32 row = 0; row < 3; ++row)
	{
		_mm256_storeu_ps(output.rhsLinear[row],
			_mm256_and_ps(_mm256_mul_ps(j[row], force), rhsMask));
		_mm256_storeu_ps(output.rhsAngular[row],
			_mm256_and_ps(_mm256_mul_ps(j[row + 3], force), rhsMask));
	}
	PxU32 upper = 0;
	for(PxU32 row = 0; row < 6; ++row)
		for(PxU32 column = row; column < 6; ++column)
		{
		{
			const __m256 responseScale =
				(row < 3u && column < 3u)
					? nonnegativeLinear
					: ((row >= 3u && column >= 3u)
						   ? nonnegativeAngular : crossScale);
			_mm256_storeu_ps(output.hessianUpper[upper++],
				_mm256_and_ps(
					avbdNormalUpperProduct8(hessianGradient[row],
						hessianGradient[column], responseScale,
						penalty),
					hessianMask));
		}
		}
	output.touchingMask = PxU8(_mm256_movemask_ps(touching));
	output.hessianMask = PxU8(_mm256_movemask_ps(hessianMask));
	output.forceSaturatedMask = PxU8(_mm256_movemask_ps(saturated));
	output.padding = 0;
	_mm256_zeroupper();
}
#pragma float_control(pop)

void avbdCpuIsaAvx2FmaRigidNormalContactPacket8Accumulate(
	const AvbdRigidNormalContactPacket8Input& input, PxU8 activeMask,
	AvbdRigidNormalContactPacket8AccumulateTarget& target)
{
	AvbdRigidNormalContactPacket8Output output = {};
	avbdCpuIsaAvx2FmaRigidNormalContactPacket8(input, output);
	avbdAccumulateNormalPacketOutput(output, activeMask, target);
}

#endif // !PX_AVBD_EXCLUDE_EXPERIMENTAL_RIGID_SIMD

} // namespace Dy
} // namespace physx
