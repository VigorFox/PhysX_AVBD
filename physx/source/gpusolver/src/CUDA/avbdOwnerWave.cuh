// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#ifndef PXG_AVBD_OWNER_WAVE_CUH
#define PXG_AVBD_OWNER_WAVE_CUH

#include "DyAvbdOwnerWaveContract.h"

#if defined(__CUDACC__)
#define PXG_AVBD_OWNER_WAVE_HD __host__ __device__
#else
#define PXG_AVBD_OWNER_WAVE_HD
#endif

namespace physx
{
namespace avbdOwnerWave
{

static const int kDofs = 6;

// The scalar AVBD authority is compiled without contraction in its baseline
// translation unit. CUDA is otherwise free to fuse the multiply/subtract
// chains below, which changes the low bits of a solved owner and accumulates
// into a visible scene drift over many dependency waves. Keep the device
// arithmetic at the same per-operation round-to-nearest boundary; the host
// side remains ordinary scalar arithmetic for the contract/replay fixtures.
PXG_AVBD_OWNER_WAVE_HD inline PxF32 roundedAdd(PxF32 a, PxF32 b)
{
#if defined(__CUDA_ARCH__)
	return __fadd_rn(a, b);
#else
	return a + b;
#endif
}

PXG_AVBD_OWNER_WAVE_HD inline PxF32 roundedSub(PxF32 a, PxF32 b)
{
#if defined(__CUDA_ARCH__)
	return __fsub_rn(a, b);
#else
	return a - b;
#endif
}

PXG_AVBD_OWNER_WAVE_HD inline PxF32 roundedMul(PxF32 a, PxF32 b)
{
#if defined(__CUDA_ARCH__)
	return __fmul_rn(a, b);
#else
	return a * b;
#endif
}

PXG_AVBD_OWNER_WAVE_HD inline PxF32 roundedDiv(PxF32 a, PxF32 b)
{
#if defined(__CUDA_ARCH__)
	return __fdiv_rn(a, b);
#else
	return a / b;
#endif
}

template <typename Packet>
PXG_AVBD_OWNER_WAVE_HD inline void loadMatrix(const Packet& packet, const int lane,
	PxF32 matrix[kDofs][kDofs])
{
	for (int i = 0; i < kDofs; ++i)
		for (int j = 0; j < kDofs; ++j)
			matrix[i][j] = 0.0f;

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			matrix[i][j] = packet.linearLinear[i][j][lane];
			matrix[i + 3][j] = packet.angularLinear[i][j][lane];
			matrix[j][i + 3] = packet.angularLinear[i][j][lane];
			matrix[i + 3][j + 3] = packet.angularAngular[i][j][lane];
		}
	}
}

template <typename Packet>
PXG_AVBD_OWNER_WAVE_HD inline bool solveLane(const Packet& packet, const int lane,
	PxF32 out[kDofs])
{
	const PxF32 initialRegularization = packet.desc.regularizationCoefficient;
	const int maxAttempts = static_cast<int>(packet.desc.maxRegularizationAttempts);
	for (int attempt = 0; attempt <= maxAttempts; ++attempt)
	{
		PxF32 regularization = 0.0f;
		if (attempt > 0)
		{
			regularization = initialRegularization;
			for (int i = 1; i < attempt; ++i)
				regularization = roundedMul(regularization, 10.0f);
		}
		// Match AvbdLDLT::decompose exactly: decompose the 3x3 linear block,
		// solve the angular-linear block, then factor the 3x3 Schur complement.
		// The former generic 6x6 elimination was algebraically equivalent but
		// accumulated products in a different order and drifted over frames.
		PxF32 linear[3][3] = {};
		PxF32 cross[3][3] = {};
		PxF32 angular[3][3] = {};
		PxF32 lLinear[3][3] = {};
		PxF32 lCross[3][3] = {};
		PxF32 lAngular[3][3] = {};
		PxF32 dLinear[3] = {};
		PxF32 dAngular[3] = {};
		for (int i = 0; i < 3; ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				linear[i][j] = packet.linearLinear[i][j][lane];
				cross[i][j] = packet.angularLinear[i][j][lane];
				angular[i][j] = packet.angularAngular[i][j][lane];
			}
			linear[i][i] = roundedAdd(linear[i][i], regularization);
			angular[i][i] = roundedAdd(angular[i][i], regularization);
		}

		bool valid = true;
		for (int i = 0; i < 3 && valid; ++i)
		{
			for (int j = 0; j <= i; ++j)
			{
				PxF32 sum = linear[i][j];
				for (int k = 0; k < j; ++k)
					sum = roundedSub(sum, roundedMul(
						roundedMul(lLinear[i][k], dLinear[k]), lLinear[j][k]));
				if (i == j)
				{
					dLinear[i] = sum;
					if (dLinear[i] <= packet.desc.singularThreshold)
					{
						valid = false;
						break;
					}
					lLinear[i][j] = 1.0f;
				}
				else
					lLinear[i][j] = roundedDiv(sum, dLinear[j]);
			}
		}
		if (!valid)
			continue;

		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
			{
				PxF32 sum = cross[i][j];
				for (int k = 0; k < j; ++k)
					sum = roundedSub(sum, roundedMul(
						roundedMul(lCross[i][k], dLinear[k]), lLinear[j][k]));
				lCross[i][j] = roundedDiv(sum, dLinear[j]);
			}

		PxF32 schur[3][3] = {};
		for (int i = 0; i < 3; ++i)
			for (int j = 0; j < 3; ++j)
			{
				PxF32 sum = angular[i][j];
				for (int k = 0; k < 3; ++k)
					sum = roundedSub(sum, roundedMul(
						roundedMul(lCross[i][k], dLinear[k]), lCross[j][k]));
				schur[i][j] = sum;
			}

		for (int i = 0; i < 3 && valid; ++i)
		{
			for (int j = 0; j <= i; ++j)
			{
				PxF32 sum = schur[i][j];
				for (int k = 0; k < j; ++k)
					sum = roundedSub(sum, roundedMul(
						roundedMul(lAngular[i][k], dAngular[k]), lAngular[j][k]));
				if (i == j)
				{
					dAngular[i] = sum;
					if (dAngular[i] <= packet.desc.singularThreshold)
					{
						valid = false;
						break;
					}
					lAngular[i][j] = 1.0f;
				}
				else
					lAngular[i][j] = sum / dAngular[j];
			}
		}
		if (!valid)
			continue;

		PxF32 minDiagonal = 3.402823466e+38F;
		PxF32 maxDiagonal = 0.0f;
		for (int i = 0; i < 3; ++i)
		{
			if (dLinear[i] > 0.0f)
			{
				minDiagonal = minDiagonal < dLinear[i] ? minDiagonal : dLinear[i];
				maxDiagonal = maxDiagonal > dLinear[i] ? maxDiagonal : dLinear[i];
			}
			if (dAngular[i] > 0.0f)
			{
				minDiagonal = minDiagonal < dAngular[i] ? minDiagonal : dAngular[i];
				maxDiagonal = maxDiagonal > dAngular[i] ? maxDiagonal : dAngular[i];
			}
		}
		const PxF32 condition = minDiagonal > 0.0f
			? maxDiagonal / minDiagonal : 3.402823466e+38F;
		if (condition >= packet.desc.conditionNumberThreshold && attempt < maxAttempts)
			continue;

		PxF32 yLinear[3] = {};
		PxF32 yAngular[3] = {};
		for (int i = 0; i < 3; ++i)
		{
			PxF32 sum = packet.rhsLinear[i][lane];
			for (int j = 0; j < i; ++j)
				sum = roundedSub(sum, roundedMul(lLinear[i][j], yLinear[j]));
			yLinear[i] = sum;
		}
		for (int i = 0; i < 3; ++i)
		{
			PxF32 sum = packet.rhsAngular[i][lane];
			for (int j = 0; j < 3; ++j)
				sum = roundedSub(sum, roundedMul(lCross[i][j], yLinear[j]));
			for (int j = 0; j < i; ++j)
				sum = roundedSub(sum, roundedMul(lAngular[i][j], yAngular[j]));
			yAngular[i] = sum;
		}
		for (int i = 0; i < 3; ++i)
		{
			yLinear[i] = roundedDiv(yLinear[i], dLinear[i]);
			yAngular[i] = roundedDiv(yAngular[i], dAngular[i]);
		}
		for (int i = 2; i >= 0; --i)
		{
			PxF32 sum = yAngular[i];
			for (int j = i + 1; j < 3; ++j)
				sum = roundedSub(sum, roundedMul(lAngular[j][i], out[j + 3]));
			out[i + 3] = sum;
		}
		for (int i = 2; i >= 0; --i)
		{
			PxF32 sum = yLinear[i];
			for (int j = i + 1; j < 3; ++j)
				sum = roundedSub(sum, roundedMul(lLinear[j][i], out[j]));
			for (int j = 0; j < 3; ++j)
				sum = roundedSub(sum, roundedMul(lCross[j][i], out[j + 3]));
			out[i] = sum;
		}
		return true;
	}
	return false;
}

} // namespace avbdOwnerWave
} // namespace physx

#undef PXG_AVBD_OWNER_WAVE_HD

#endif // PXG_AVBD_OWNER_WAVE_CUH
