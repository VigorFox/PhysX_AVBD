// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/DyAvbdContactMaterial.h"
#include "avbd/DyKernelTable.h"
#include <immintrin.h>

namespace physx { namespace Dy {
namespace {
struct AvxSpatial
{
    __m256d value;
    static AvxSpatial load(const PxReal* f) {return {_mm256_cvtps_pd(_mm_loadu_ps(f))};}
    static AvxSpatial join(const double* a,const double* b)
    {
        return {_mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(a)),_mm_loadu_pd(b),1)};
    }
    static AvxSpatial splat(double v) {return {_mm256_set1_pd(v)};}
    void split(double* a,double* b) const
    {
        _mm_storeu_pd(a,_mm256_castpd256_pd128(value));
        _mm_storeu_pd(b,_mm256_extractf128_pd(value,1));
    }
    double sum() const
    {
        const __m128d pair=_mm_add_pd(_mm256_castpd256_pd128(value),_mm256_extractf128_pd(value,1));
        return _mm_cvtsd_f64(_mm_add_sd(pair,_mm_unpackhi_pd(pair,pair)));
    }
    AvxSpatial operator+(const AvxSpatial& r) const {return {_mm256_add_pd(value,r.value)};}
    AvxSpatial operator*(const AvxSpatial& r) const {return {_mm256_mul_pd(value,r.value)};}
};
}

void avbdContactMaterialAvx256(void* data)
{
    avbdContactMaterialBatchImpl<AvxSpatial>(data);
}

void avbdRankOneAvx256(void* data)
{
    const AvbdRankOneArgs& a=*static_cast<AvbdRankOneArgs*>(data);
    const PxReal j[6]={a.linear.x,a.linear.y,a.linear.z,a.angular.x,a.angular.y,a.angular.z};
    const __m256 head=_mm256_setr_ps(j[0],j[1],j[2],j[3],j[4],j[5],j[0],j[1]);
    const __m128 tail=_mm_loadu_ps(j+2);
    for(PxU32 r=0;r<6;r+=2)
    {
        const __m256 even=_mm256_set1_ps(a.scale*j[r]), odd=_mm256_set1_ps(a.scale*j[r+1]);
        PxReal* x=a.matrix[r]; PxReal* y=a.matrix[r+1]+2;
        const __m256 weights=_mm256_blend_ps(even,odd,0xc0);
        _mm256_storeu_ps(x,_mm256_add_ps(_mm256_loadu_ps(x),_mm256_mul_ps(weights,head)));
        _mm_storeu_ps(y,_mm_add_ps(_mm_loadu_ps(y),_mm_mul_ps(_mm256_castps256_ps128(odd),tail)));
    }
}
} }
