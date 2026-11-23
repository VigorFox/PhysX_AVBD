// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_KERNELS_SCALAR_H
#define DY_AVBD_KERNELS_SCALAR_H

// Scalar reference for AVBD KernelId AVBD_LDLT6 / AVBD_LDLT3 (outline 4.2).
// SIMD / CUDA bind the same KernelId; do not fork this math.

#include "foundation/PxMat33.h"
#include "foundation/PxMath.h"
#include "foundation/PxVec3.h"

namespace physx
{
namespace Dy
{

// Restrict the quadratic to the public world-axis motion subspace BEFORE
// solving. Projecting an unconstrained solution changes the free equations.
PX_FORCE_INLINE void avbdKernelRestrict6(PxReal H[6][6],PxReal g[6],PxU32 locked)
{
    for(PxU32 d=0;d<6;++d) if(locked&(1u<<d))
    {
        for(PxU32 j=0;j<6;++j) H[d][j]=H[j][d]=0;
        H[d][d]=1;g[d]=0;
    }
}

static const PxReal AVBD_KERNEL_LDLT_REG = 1.0e-6f;

PX_FORCE_INLINE void avbdKernelFactor6(const PxReal A[6][6], PxReal L[6][6])
{
	for(PxU32 i = 0; i < 6; ++i)
		for(PxU32 j = 0; j < 6; ++j)
			L[i][j] = 0.0f;
	for(PxU32 i = 0; i < 6; ++i)
	{
		for(PxU32 j = i; j < 6; ++j)
		{
			PxReal sum = A[i][j];
			for(PxU32 k = 0; k < i; ++k)
				sum -= L[i][k] * L[j][k];
			if(i == j)
			{
				if(sum <= AVBD_KERNEL_LDLT_REG)
					sum = AVBD_KERNEL_LDLT_REG;
				L[i][i] = PxSqrt(sum);
			}
			else
				L[j][i] = sum / L[i][i];
		}
	}
}

PX_FORCE_INLINE bool avbdKernelBacksolve6(const PxReal L[6][6], const PxReal b[6], PxReal x[6])
{
	PxReal y[6];
	for(PxU32 i = 0; i < 6; ++i)
	{
		PxReal sum = b[i];
		for(PxU32 k = 0; k < i; ++k)
			sum -= L[i][k] * y[k];
		y[i] = sum / L[i][i];
	}
	for(PxI32 i = 5; i >= 0; --i)
	{
		PxReal sum = y[PxU32(i)];
		for(PxU32 k = PxU32(i) + 1; k < 6; ++k)
			sum -= L[k][PxU32(i)] * x[k];
		x[PxU32(i)] = sum / L[PxU32(i)][PxU32(i)];
	}
	for(PxU32 i = 0; i < 6; ++i)
	{
		if(!PxIsFinite(x[i]))
			return false;
	}
	return true;
}

PX_FORCE_INLINE bool avbdKernelLdlt6(PxReal A[6][6], const PxReal b[6], PxReal x[6])
{
	PxReal L[6][6];
	avbdKernelFactor6(A,L);
	return avbdKernelBacksolve6(L,b,x);
}

PX_FORCE_INLINE bool avbdKernelLdlt3(const PxMat33& Ain, const PxVec3& bin, PxVec3& x,
	PxMat33* effectiveMetric = NULL)
{
	const PxReal reg = AVBD_KERNEL_LDLT_REG;
	PxReal A[3][3] = {
		{ Ain.column0.x + reg, Ain.column1.x, Ain.column2.x },
		{ Ain.column0.y, Ain.column1.y + reg, Ain.column2.y },
		{ Ain.column0.z, Ain.column1.z, Ain.column2.z + reg }
	};
	PxReal L[3][3] = {};
	for(PxU32 i = 0; i < 3; ++i)
	{
		for(PxU32 j = i; j < 3; ++j)
		{
			PxReal sum = A[i][j];
			for(PxU32 k = 0; k < i; ++k)
				sum -= L[i][k] * L[j][k];
			if(i == j)
			{
				if(sum <= reg)
					sum = reg;
				L[i][i] = PxSqrt(sum);
			}
			else
				L[j][i] = sum / L[i][i];
		}
	}
	if(effectiveMetric)
		for(PxU32 i=0;i<3;++i)
			for(PxU32 j=0;j<3;++j)
			{
				PxReal entry=0.0f;
				for(PxU32 k=0;k<3;++k) entry+=L[i][k]*L[j][k];
				(*effectiveMetric)(i,j)=entry;
			}
	PxReal y[3];
	const PxReal b[3] = { bin.x, bin.y, bin.z };
	for(PxU32 i = 0; i < 3; ++i)
	{
		PxReal sum = b[i];
		for(PxU32 k = 0; k < i; ++k)
			sum -= L[i][k] * y[k];
		y[i] = sum / L[i][i];
	}
	PxReal xv[3];
	for(PxI32 i = 2; i >= 0; --i)
	{
		PxReal sum = y[PxU32(i)];
		for(PxU32 k = PxU32(i) + 1; k < 3; ++k)
			sum -= L[k][PxU32(i)] * xv[k];
		xv[PxU32(i)] = sum / L[PxU32(i)][PxU32(i)];
	}
	x = PxVec3(xv[0], xv[1], xv[2]);
	return PxIsFinite(x.x) && PxIsFinite(x.y) && PxIsFinite(x.z);
}

} // namespace Dy
} // namespace physx

#endif
