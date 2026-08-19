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

#include "DyAvbdCpuIsa.h"
#include "foundation/PxMath.h"

#include <emmintrin.h>

namespace physx {
namespace Dy {

// P6.1's reference kernel deliberately has no solver dependency. It proves
// baseline TU isolation and the function-table ABI.
PxF32 avbdCpuIsaSse2ProbeDot8(const PxF32* lhs, const PxF32* rhs)
{
	const __m128 low = _mm_mul_ps(_mm_loadu_ps(lhs), _mm_loadu_ps(rhs));
	const __m128 high = _mm_mul_ps(_mm_loadu_ps(lhs + 4), _mm_loadu_ps(rhs + 4));
	const __m128 sum = _mm_add_ps(low, high);
	PX_ALIGN_PREFIX(16) PxF32 values[4] PX_ALIGN_SUFFIX(16);
	_mm_store_ps(values, sum);
	return values[0] + values[1] + values[2] + values[3];
}

#if !defined(PX_AVBD_EXCLUDE_EXPERIMENTAL_RIGID_SIMD)

void avbdCpuIsaSse2RigidNormalContactDynamicEndpoint8(
	const AvbdRigidNormalContactDynamicTarget8& input,
	PxF32* worldPositionX, PxF32* worldPositionY, PxF32* worldPositionZ)
{
	for(PxU32 lane = 0; lane < eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH; ++lane)
	{
		if((input.dynamicMask & PxU8(1u << lane)) == 0)
		{
			worldPositionX[lane] = 0.0f;
			worldPositionY[lane] = 0.0f;
			worldPositionZ[lane] = 0.0f;
			continue;
		}
		const PxF32 qx = input.rotation[0][lane];
		const PxF32 qy = input.rotation[1][lane];
		const PxF32 qz = input.rotation[2][lane];
		const PxF32 qw = input.rotation[3][lane];
		const PxF32 vx = input.contactPoint[0][lane];
		const PxF32 vy = input.contactPoint[1][lane];
		const PxF32 vz = input.contactPoint[2][lane];
		const PxF32 tx = 2.0f * (qy * vz - qz * vy);
		const PxF32 ty = 2.0f * (qz * vx - qx * vz);
		const PxF32 tz = 2.0f * (qx * vy - qy * vx);
		const PxF32 rx = vx + qw * tx + qy * tz - qz * ty;
		const PxF32 ry = vy + qw * ty + qz * tx - qx * tz;
		const PxF32 rz = vz + qw * tz + qx * ty - qy * tx;
		worldPositionX[lane] = input.position[0][lane] + rx;
		worldPositionY[lane] = input.position[1][lane] + ry;
		worldPositionZ[lane] = input.position[2][lane] + rz;
	}
}

void avbdCpuIsaSse2RigidLdltPacket8(
	const AvbdRigidLdltPacket8Input& input,
	AvbdRigidLdltPacket8Output& output)
{
	for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
	{
		PxF32 yl[3], ya[3], xl[3], xa[3];
		for(PxU32 i = 0; i < 3; ++i)
		{
			PxF32 sum = input.rhsLinear[i][lane];
			for(PxU32 j = 0; j < i; ++j)
				sum -= input.linearLinear[i][j][lane] * yl[j];
			yl[i] = sum;
		}
		for(PxU32 i = 0; i < 3; ++i)
		{
			PxF32 sum = input.rhsAngular[i][lane];
			for(PxU32 j = 0; j < 3; ++j)
				sum -= input.angularLinear[i][j][lane] * yl[j];
			for(PxU32 j = 0; j < i; ++j)
				sum -= input.angularAngular[i][j][lane] * ya[j];
			ya[i] = sum;
		}
		for(PxU32 i = 0; i < 3; ++i)
			yl[i] /= input.diagonalLinear[i][lane];
		for(PxU32 i = 0; i < 3; ++i)
			ya[i] /= input.diagonalAngular[i][lane];
		for(PxI32 i = 2; i >= 0; --i)
		{
			PxF32 sum = ya[i];
			for(PxU32 j = PxU32(i + 1); j < 3; ++j)
				sum -= input.angularAngular[j][i][lane] * xa[j];
			xa[i] = sum;
		}
		for(PxI32 i = 2; i >= 0; --i)
		{
			PxF32 sum = yl[i];
			for(PxU32 j = PxU32(i + 1); j < 3; ++j)
				sum -= input.linearLinear[j][i][lane] * xl[j];
			for(PxU32 j = 0; j < 3; ++j)
				sum -= input.angularLinear[j][i][lane] * xa[j];
			xl[i] = sum;
		}

		for(PxU32 i = 0; i < 3; ++i)
		{
			output.linear[i][lane] = xl[i];
			output.angular[i][lane] = xa[i];
		}
	}
}

namespace {

#pragma float_control(precise, on, push)
static bool avbdSse2Factor3x3(
	const PxF32 aLL[3][3], const PxF32 aAL[3][3],
	const PxF32 aAA[3][3], PxF32 lLL[3][3], PxF32 lAL[3][3],
	PxF32 lAA[3][3], PxF32 dL[3], PxF32 dA[3], PxF32 singularThreshold,
	PxF32& conditionNumber)
{
	for(PxU32 i = 0; i < 3; ++i)
	{
		for(PxU32 j = 0; j <= i; ++j)
		{
			PxF32 sum = aLL[i][j];
			for(PxU32 k = 0; k < j; ++k)
				sum -= lLL[i][k] * dL[k] * lLL[j][k];
			if(i == j)
			{
				dL[i] = sum;
				if(dL[i] <= singularThreshold)
					return false;
				lLL[i][j] = 1.0f;
			}
			else
				lLL[i][j] = sum / dL[j];
		}
	}
	for(PxU32 i = 0; i < 3; ++i)
	{
		for(PxU32 j = 0; j < 3; ++j)
		{
			PxF32 sum = aAL[i][j];
			for(PxU32 k = 0; k < j; ++k)
				sum -= lAL[i][k] * dL[k] * lLL[j][k];
			lAL[i][j] = sum / dL[j];
		}
	}
	PxF32 schur[3][3];
	for(PxU32 i = 0; i < 3; ++i)
	{
		for(PxU32 j = 0; j < 3; ++j)
		{
			PxF32 sum = aAA[i][j];
			for(PxU32 k = 0; k < 3; ++k)
				sum -= lAL[i][k] * dL[k] * lAL[j][k];
			schur[i][j] = sum;
		}
	}
	for(PxU32 i = 0; i < 3; ++i)
	{
		for(PxU32 j = 0; j <= i; ++j)
		{
			PxF32 sum = schur[i][j];
			for(PxU32 k = 0; k < j; ++k)
				sum -= lAA[i][k] * dA[k] * lAA[j][k];
			if(i == j)
			{
				dA[i] = sum;
				if(dA[i] <= singularThreshold)
					return false;
				lAA[i][j] = 1.0f;
			}
			else
				lAA[i][j] = sum / dA[j];
		}
	}
	PxF32 minD = PX_MAX_F32;
	PxF32 maxD = 0.0f;
	for(PxU32 i = 0; i < 3; ++i)
	{
		if(dL[i] > 0.0f)
		{
			minD = PxMin(minD, dL[i]);
			maxD = PxMax(maxD, dL[i]);
		}
		if(dA[i] > 0.0f)
		{
			minD = PxMin(minD, dA[i]);
			maxD = PxMax(maxD, dA[i]);
		}
	}
	conditionNumber = (minD > 0.0f) ? (maxD / minD) : PX_MAX_F32;
	return true;
}

static void avbdSse2SolveFactor3x3(
	const PxF32 lLL[3][3], const PxF32 lAL[3][3],
	const PxF32 lAA[3][3], const PxF32 dL[3], const PxF32 dA[3],
	const PxF32 rhsL[3], const PxF32 rhsA[3], PxF32 outL[3], PxF32 outA[3])
{
	PxF32 yL[3], yA[3], xL[3], xA[3];
	for(PxU32 i = 0; i < 3; ++i)
	{
		PxF32 sum = rhsL[i];
		for(PxU32 j = 0; j < i; ++j)
			sum -= lLL[i][j] * yL[j];
		yL[i] = sum;
	}
	for(PxU32 i = 0; i < 3; ++i)
	{
		PxF32 sum = rhsA[i];
		for(PxU32 j = 0; j < 3; ++j)
			sum -= lAL[i][j] * yL[j];
		for(PxU32 j = 0; j < i; ++j)
			sum -= lAA[i][j] * yA[j];
		yA[i] = sum;
	}
	for(PxU32 i = 0; i < 3; ++i)
	{
		yL[i] /= dL[i];
		yA[i] /= dA[i];
	}
	for(PxI32 i = 2; i >= 0; --i)
	{
		PxF32 sum = yA[i];
		for(PxU32 j = PxU32(i + 1); j < 3; ++j)
			sum -= lAA[j][i] * xA[j];
		xA[i] = sum;
	}
	for(PxI32 i = 2; i >= 0; --i)
	{
		PxF32 sum = yL[i];
		for(PxU32 j = PxU32(i + 1); j < 3; ++j)
			sum -= lLL[j][i] * xL[j];
		for(PxU32 j = 0; j < 3; ++j)
			sum -= lAL[j][i] * xA[j];
		xL[i] = sum;
	}
	for(PxU32 i = 0; i < 3; ++i)
	{
		outL[i] = xL[i];
		outA[i] = xA[i];
	}
}

#pragma float_control(pop)

} // namespace

void avbdCpuIsaSse2RigidLdltFactorSolvePacket8(
	const AvbdRigidLdltFactorSolvePacket8Input& input,
	AvbdRigidLdltFactorSolvePacket8Output& output)
{
	output.validMask = 0;
	for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
	{
		PxF32 baseLL[3][3], baseAL[3][3], baseAA[3][3];
		PxF32 rhsL[3], rhsA[3];
		for(PxU32 i = 0; i < 3; ++i)
		{
			rhsL[i] = input.rhsLinear[i][lane];
			rhsA[i] = input.rhsAngular[i][lane];
			for(PxU32 j = 0; j < 3; ++j)
			{
				baseLL[i][j] = input.linearLinear[i][j][lane];
				baseAL[i][j] = input.angularLinear[i][j][lane];
				baseAA[i][j] = input.angularAngular[i][j][lane];
			}
		}

		PxF32 regularization = input.regularizationCoefficient;
		bool solved = false;
		PxF32 outL[3] = {0.0f, 0.0f, 0.0f};
		PxF32 outA[3] = {0.0f, 0.0f, 0.0f};
		for(PxU32 attempt = 0;
			attempt <= input.maxRegularizationAttempts && !solved; ++attempt)
		{
			PxF32 aLL[3][3], aAL[3][3], aAA[3][3];
			for(PxU32 i = 0; i < 3; ++i)
			{
				for(PxU32 j = 0; j < 3; ++j)
				{
					aLL[i][j] = baseLL[i][j];
					aAL[i][j] = baseAL[i][j];
					aAA[i][j] = baseAA[i][j];
				}
				aLL[i][i] += (attempt == 0 ? 0.0f : regularization);
				aAA[i][i] += (attempt == 0 ? 0.0f : regularization);
			}
			PxF32 lLL[3][3] = {}, lAL[3][3] = {}, lAA[3][3] = {};
			PxF32 dL[3] = {}, dA[3] = {}, conditionNumber = PX_MAX_F32;
			if(avbdSse2Factor3x3(
				aLL, aAL, aAA, lLL, lAL, lAA, dL, dA,
				input.singularThreshold, conditionNumber) &&
				(conditionNumber < input.conditionNumberThreshold ||
				 attempt == input.maxRegularizationAttempts))
			{
				avbdSse2SolveFactor3x3(
					lLL, lAL, lAA, dL, dA, rhsL, rhsA, outL, outA);
				solved = true;
			}
			if(attempt < input.maxRegularizationAttempts)
				regularization *= 10.0f;
		}
		if(solved)
		{
			output.validMask |= PxU8(1u << lane);
			for(PxU32 i = 0; i < 3; ++i)
			{
				output.linear[i][lane] = outL[i];
				output.angular[i][lane] = outA[i];
			}
		}
		else
		{
			for(PxU32 i = 0; i < 3; ++i)
			{
				output.linear[i][lane] = 0.0f;
				output.angular[i][lane] = 0.0f;
			}
		}
	}
}

void avbdCpuIsaSse2RigidLocalResponsePacket8(
	const AvbdRigidLocalResponsePacket8Input& input,
	AvbdRigidLdltFactorSolvePacket8Input& target, PxU8& touchingMask)
{
	for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
	{
		const PxU8 bit = PxU8(1u << lane);
		if((input.activeMask & bit) == 0)
			continue;
		const PxF32 nonnegativeLinear = PxMax(0.0f, input.linearScale[lane]);
		const PxF32 nonnegativeAngular = PxMax(0.0f, input.angularScale[lane]);
		const PxF32 crossScale = PxSqrt(
			nonnegativeLinear * nonnegativeAngular);
		for(PxU32 i = 0; i < 3; ++i)
		{
			for(PxU32 j = 0; j < 3; ++j)
			{
				target.linearLinear[i][j][lane] +=
					input.invCompliance[lane] * nonnegativeLinear *
					input.gradPos[i][lane] * input.gradPos[j][lane];
				target.angularLinear[i][j][lane] +=
					input.invCompliance[lane] * crossScale *
					input.gradRot[i][lane] * input.gradPos[j][lane];
				target.angularAngular[i][j][lane] +=
					input.invCompliance[lane] * nonnegativeAngular *
					input.gradRot[i][lane] * input.gradRot[j][lane];
			}
			target.rhsLinear[i][lane] +=
				input.gradPos[i][lane] *
				(input.force[lane] * nonnegativeLinear);
			target.rhsAngular[i][lane] +=
				input.gradRot[i][lane] *
				(input.force[lane] * nonnegativeAngular);
		}
		if((input.touchingMask & bit) != 0)
			touchingMask |= bit;
	}
}

void avbdCpuIsaSse2RigidD6ResponsePacket8(
	const AvbdRigidD6ResponsePacket8Input& input,
	AvbdRigidLdltFactorSolvePacket8Input& target, PxU8& touchingMask)
{
	for(PxU32 row = 0; row < eAVBD_RIGID_D6_RESPONSE_ROW_PACKET_COUNT; ++row)
	{
		for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
		{
			const PxU8 bit = PxU8(1u << lane);
			if((input.activeMask & bit) == 0)
				continue;
			const PxF32 nonnegativeLinear =
				PxMax(0.0f, input.linearScale[row][lane]);
			const PxF32 nonnegativeAngular =
				PxMax(0.0f, input.angularScale[row][lane]);
			const PxF32 crossScale = PxSqrt(
				nonnegativeLinear * nonnegativeAngular);
			for(PxU32 i = 0; i < 3; ++i)
			{
				for(PxU32 j = 0; j < 3; ++j)
				{
					target.linearLinear[i][j][lane] +=
						input.invCompliance[row][lane] * nonnegativeLinear *
						input.gradPos[row][i][lane] * input.gradPos[row][j][lane];
					target.angularLinear[i][j][lane] +=
						input.invCompliance[row][lane] * crossScale *
						input.gradRot[row][i][lane] * input.gradPos[row][j][lane];
					target.angularAngular[i][j][lane] +=
						input.invCompliance[row][lane] * nonnegativeAngular *
						input.gradRot[row][i][lane] * input.gradRot[row][j][lane];
				}
				target.rhsLinear[i][lane] += input.gradPos[row][i][lane] *
					(input.force[row][lane] * nonnegativeLinear);
				target.rhsAngular[i][lane] += input.gradRot[row][i][lane] *
					(input.force[row][lane] * nonnegativeAngular);
			}
			if((input.touchingMask & bit) != 0)
				touchingMask |= bit;
		}
	}
}

void avbdCpuIsaSse2RigidD6ResponsePacket8View(
	const AvbdRigidD6ResponsePacket8View& input,
	AvbdRigidLdltFactorSolvePacket8Input& target, PxU8& touchingMask)
{
	for(PxU32 row = 0; row < eAVBD_RIGID_D6_RESPONSE_ROW_PACKET_COUNT; ++row)
	{
		for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
		{
			const PxU8 bit = PxU8(1u << lane);
			if((input.activeMask & bit) == 0)
				continue;
			const PxF32 nonnegativeLinear = PxMax(
				0.0f, input.linearScale[row][lane]);
			const PxF32 nonnegativeAngular = PxMax(
				0.0f, input.angularScale[row][lane]);
			const PxF32 crossScale = PxSqrt(
				nonnegativeLinear * nonnegativeAngular);
			for(PxU32 i = 0; i < 3; ++i)
			{
				for(PxU32 j = 0; j < 3; ++j)
				{
					target.linearLinear[i][j][lane] +=
						input.invCompliance[row][lane] * nonnegativeLinear *
						input.gradPos[row][i][lane] * input.gradPos[row][j][lane];
					target.angularLinear[i][j][lane] +=
						input.invCompliance[row][lane] * crossScale *
						input.gradRot[row][i][lane] * input.gradPos[row][j][lane];
					target.angularAngular[i][j][lane] +=
						input.invCompliance[row][lane] * nonnegativeAngular *
						input.gradRot[row][i][lane] * input.gradRot[row][j][lane];
				}
				target.rhsLinear[i][lane] += input.gradPos[row][i][lane] *
					(input.force[row][lane] * nonnegativeLinear);
				target.rhsAngular[i][lane] += input.gradRot[row][i][lane] *
					(input.force[row][lane] * nonnegativeAngular);
			}
			if((input.touchingMask & bit) != 0)
				touchingMask |= bit;
		}
	}
}

namespace
{
	PX_FORCE_INLINE PxU32 avbdRigidNormalUpperIndex(PxU32 row, PxU32 column)
	{
		if(row > column)
		{
			const PxU32 tmp = row;
			row = column;
			column = tmp;
		}
		return row * 6u - row * (row - 1u) / 2u + column - row;
	}

	PX_FORCE_INLINE void avbdSse2RotatePoint(
		const PxF32 q[4], const PxF32 v[3], PxF32 out[3])
	{
		const PxF32 tx = 2.0f * (q[1] * v[2] - q[2] * v[1]);
		const PxF32 ty = 2.0f * (q[2] * v[0] - q[0] * v[2]);
		const PxF32 tz = 2.0f * (q[0] * v[1] - q[1] * v[0]);
		out[0] = v[0] + q[3] * tx + q[1] * tz - q[2] * ty;
		out[1] = v[1] + q[3] * ty + q[2] * tx - q[0] * tz;
		out[2] = v[2] + q[3] * tz + q[0] * ty - q[1] * tx;
	}
}

void avbdCpuIsaSse2RigidContactBlockPacket8(
	const AvbdRigidContactBlockPacket8Input& input,
	AvbdRigidLdltFactorSolvePacket8Input& target, PxU8& touchingMask)
{
	for(PxU32 lane = 0; lane < eAVBD_RIGID_LDLT_PACKET_WIDTH; ++lane)
	{
		const PxU8 bit = PxU8(1u << lane);
		if((input.activeMask & bit) == 0)
			continue;
		const PxF32 nonnegativeLinear =
			PxMax(0.0f, input.linearScale[lane]);
		const PxF32 nonnegativeAngular =
			PxMax(0.0f, input.angularScale[lane]);
		const PxF32 crossScale = PxSqrt(
			nonnegativeLinear * nonnegativeAngular);
		for(PxU32 row = 0;
			row < eAVBD_RIGID_CONTACT_BLOCK_ROW_COUNT; ++row)
		{
			const PxF32 invCompliance = input.invCompliance[row][lane];
			const PxF32 linearCompliance =
				invCompliance * nonnegativeLinear;
			const PxF32 crossCompliance = invCompliance * crossScale;
			const PxF32 angularCompliance =
				invCompliance * nonnegativeAngular;
			const bool addHessian =
				(input.hessianMask[row] & bit) != 0;
			if(addHessian)
			{
				for(PxU32 i = 0; i < 3; ++i)
				{
					for(PxU32 j = 0; j < 3; ++j)
					{
						target.linearLinear[i][j][lane] +=
							linearCompliance * input.gradPos[row][i][lane] *
							input.gradPos[row][j][lane];
						target.angularLinear[i][j][lane] +=
							crossCompliance * input.gradRot[row][i][lane] *
							input.gradPos[row][j][lane];
						target.angularAngular[i][j][lane] +=
							angularCompliance * input.gradRot[row][i][lane] *
							input.gradRot[row][j][lane];
					}
				}
			}
			for(PxU32 i = 0; i < 3; ++i)
			{
				target.rhsLinear[i][lane] +=
					input.gradPos[row][i][lane] *
					(input.force[row][lane] * nonnegativeLinear);
				target.rhsAngular[i][lane] +=
					input.gradRot[row][i][lane] *
					(input.force[row][lane] * nonnegativeAngular);
			}
		}
		if((input.touchingMask & bit) != 0)
			touchingMask |= bit;
	}
}

void avbdCpuIsaSse2RigidNormalContactPacket8(
	const AvbdRigidNormalContactPacket8Input& input,
	AvbdRigidNormalContactPacket8Output& output)
{
	for(PxU32 row = 0; row < 3; ++row)
	{
		for(PxU32 lane = 0; lane < eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH; ++lane)
		{
			output.rhsLinear[row][lane] = 0.0f;
			output.rhsAngular[row][lane] = 0.0f;
		}
	}
	for(PxU32 row = 0; row < eAVBD_RIGID_NORMAL_CONTACT_UPPER_COUNT; ++row)
		for(PxU32 lane = 0; lane < eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH; ++lane)
			output.hessianUpper[row][lane] = 0.0f;
	output.touchingMask = 0;
	output.hessianMask = 0;
	output.forceSaturatedMask = 0;
	output.padding = 0;

	for(PxU32 lane = 0; lane < eAVBD_RIGID_NORMAL_CONTACT_PACKET_WIDTH; ++lane)
	{
		if((input.activeMask & PxU8(1u << lane)) == 0)
			continue;
		const PxF32 linearScale = input.linearResponseScale[lane];
		const PxF32 angularScale = input.angularResponseScale[lane];
		if(linearScale <= 0.0f && angularScale <= 0.0f)
			continue;

		PxF32 q[4] = {
			input.bodyRotation[0][lane], input.bodyRotation[1][lane],
			input.bodyRotation[2][lane], input.bodyRotation[3][lane]};
		const PxF32 point[3] = {
			input.bodyContactPoint[0][lane], input.bodyContactPoint[1][lane],
			input.bodyContactPoint[2][lane]};
		PxF32 r[3];
		avbdSse2RotatePoint(q, point, r);
		const PxF32 worldX = input.bodyPosition[0][lane] + r[0];
		const PxF32 worldY = input.bodyPosition[1][lane] + r[1];
		const PxF32 worldZ = input.bodyPosition[2][lane] + r[2];
		const PxF32 nx = input.normal[0][lane];
		const PxF32 ny = input.normal[1][lane];
		const PxF32 nz = input.normal[2][lane];
		const PxF32 sign = input.sign[lane];
		const PxF32 geometricGap =
			(worldX - input.staticContactPoint[0][lane]) * nx +
			(worldY - input.staticContactPoint[1][lane]) * ny +
			(worldZ - input.staticContactPoint[2][lane]) * nz;
		// The scalar authority always evaluates (A-B).n.  A packet lane
		// stores the dynamic body, so sign flips the dynamic-static gap when
		// that body is B; penetrationDepth is an authored offset and is not
		// sign-flipped.
		const PxF32 violation = geometricGap * sign + input.penetration[lane];
		const PxF32 penalty = input.penalty[lane];
		const PxF32 rawForce = PxMin(0.0f, penalty * violation + input.lambda[lane]);
		PxF32 force = rawForce;
		bool saturated = false;
		if(input.maxImpulse[lane] < PX_MAX_F32 && input.dt[lane] > 0.0f)
		{
			const PxF32 maxNormalForce =
				PxMax(input.maxImpulse[lane], 0.0f) / input.dt[lane];
			force = PxMax(force, -maxNormalForce);
			saturated = rawForce < -maxNormalForce;
		}
		const PxF32 gp[3] = {nx * sign, ny * sign, nz * sign};
		const PxF32 gr[3] = {
			(r[1] * nz - r[2] * ny) * sign,
			(r[2] * nx - r[0] * nz) * sign,
			(r[0] * ny - r[1] * nx) * sign};
		const PxF32 nonnegativeLinear = PxMax(0.0f, linearScale);
		const PxF32 nonnegativeAngular = PxMax(0.0f, angularScale);
		const PxF32 crossScale = PxSqrt(nonnegativeLinear * nonnegativeAngular);
		const PxF32 j[6] = {
			gp[0] * linearScale, gp[1] * linearScale, gp[2] * linearScale,
			gr[0] * angularScale, gr[1] * angularScale, gr[2] * angularScale};
		const PxF32 hessianGradient[6] = {gp[0], gp[1], gp[2],
			gr[0], gr[1], gr[2]};
		if(force < 0.0f)
		{
			for(PxU32 row = 0; row < 3; ++row)
			{
				output.rhsLinear[row][lane] = j[row] * force;
				output.rhsAngular[row][lane] = j[row + 3] * force;
			}
		}
		if(!saturated)
		{
			output.hessianMask |= PxU8(1u << lane);
			for(PxU32 row = 0; row < 6; ++row)
				for(PxU32 column = row; column < 6; ++column)
				{
					const PxF32 responseScale =
						(row < 3u && column < 3u)
							? nonnegativeLinear
							: ((row >= 3u && column >= 3u)
								   ? nonnegativeAngular : crossScale);
					output.hessianUpper[
						avbdRigidNormalUpperIndex(row, column)][lane] =
						penalty * responseScale * hessianGradient[row] *
							hessianGradient[column];
				}
		}
		if(saturated)
			output.forceSaturatedMask |= PxU8(1u << lane);
		output.touchingMask |= PxU8(1u << lane);
	}
}

void avbdCpuIsaSse2RigidNormalContactPacket8Accumulate(
	const AvbdRigidNormalContactPacket8Input& input, PxU8 activeMask,
	AvbdRigidNormalContactPacket8AccumulateTarget& target)
{
	AvbdRigidNormalContactPacket8Output output = {};
	avbdCpuIsaSse2RigidNormalContactPacket8(input, output);
	avbdAccumulateNormalPacketOutput(output, activeMask, target);
}

#endif // !PX_AVBD_EXCLUDE_EXPERIMENTAL_RIGID_SIMD

} // namespace Dy
} // namespace physx
