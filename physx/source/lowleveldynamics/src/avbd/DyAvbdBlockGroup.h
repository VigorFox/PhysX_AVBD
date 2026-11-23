// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_BLOCK_GROUP_H
#define DY_AVBD_BLOCK_GROUP_H
#include "avbd/DyAvbdKernelsScalar.h"
#include "avbd/DyAvbdLinearResponse.h"
#include "foundation/PxMemory.h"

namespace physx { namespace Dy {
template<PxU32 Capacity> struct AvbdGroupFactorT
{
    PxReal jacobian[Capacity][6];
    PxReal curvature;
};
typedef AvbdGroupFactorT<16> AvbdGroupFactor;
struct AvbdGroupSolveArgs
{
    PxU32 count;
    PxReal (*diagonal)[6][6];
    const PxReal (*gradient)[6];
    const AvbdGroupFactor* factors;
    PxU32 factorCount;
    PxReal (*step)[6];
    bool* solved;
    AvbdLinearResponseResult* diagnostics=NULL;
};

struct AvbdGroupOperator
{
    const AvbdGroupSolveArgs& a;
    double inverse[4][6][6];
    void apply(const double* x,double* y) const
    {
        for(PxU32 b=0;b<a.count;++b) for(PxU32 i=0;i<6;++i)
        { y[6*b+i]=0;for(PxU32 j=0;j<6;++j)y[6*b+i]+=double(a.diagonal[b][i][j])*x[6*b+j]; }
        for(PxU32 f=0;f<a.factorCount;++f)
        {
            const AvbdGroupFactor& factor=a.factors[f];double projection[4]={},total=0;
            for(PxU32 b=0;b<a.count;++b)
            { for(PxU32 j=0;j<6;++j)projection[b]+=double(factor.jacobian[b][j])*x[6*b+j];total+=projection[b]; }
            for(PxU32 b=0;b<a.count;++b)for(PxU32 j=0;j<6;++j)
                y[6*b+j]+=double(factor.curvature)*double(factor.jacobian[b][j])*(total-projection[b]);
        }
    }
    void precondition(const double* x,double* y) const
    {
        for(PxU32 b=0;b<a.count;++b)for(PxU32 i=0;i<6;++i)
        {y[6*b+i]=0;for(PxU32 j=0;j<6;++j)y[6*b+i]+=inverse[b][i][j]*x[6*b+j];}
    }
};

// Actual stored pose increments can differ from the linear proposal after
// locks and finite rotations. Evaluate their FULL assembled quadratic model.
inline double avbdGroupModelChange(const AvbdGroupSolveArgs& a,const double* delta)
{
    AvbdGroupOperator op={a,{}};double product[24];op.apply(delta,product);
    double change=0;
    for(PxU32 i=0;i<6*a.count;++i)
        change+=delta[i]*(double(a.gradient[i/6][i%6])+.5*product[i]);
    return change;
}

// Four independent 6x6 factorizations precondition a matrix-free row operator.
// No enlarged rigid unknown or second physical multiplier is stored.
inline bool avbdSolveBlockGroup(const AvbdGroupSolveArgs& a)
{
    if(a.count<2 || a.count>4) return false;
    AvbdGroupOperator op={a,{}};
    double x[24]={},rhs[24],work[96];
    const PxU32 n=6*a.count;
    if(a.diagnostics)*a.diagnostics={false,0,0,std::numeric_limits<double>::infinity()};
    for(PxU32 b=0;b<a.count;++b)
    {
        PxReal factor[6][6];avbdKernelFactor6(a.diagonal[b],factor);
        for(PxU32 j=0;j<6;++j)
        {
            PxReal unit[6]={},solution[6];unit[j]=1;
            if(!avbdKernelBacksolve6(factor,unit,solution)) return false;
            for(PxU32 i=0;i<6;++i) op.inverse[b][i][j]=solution[i];
        }
    }
    for(PxU32 i=0;i<n;++i)rhs[i]=-double(a.gradient[i/6][i%6]);
    // Same n-step algebraic budget. The shared kernel verifies H*x-rhs,
    // instead of accepting only a recursively updated CG residual.
    const auto result=avbdLinearResponse(op,n,rhs,x,work,n,double(PX_EPS_F32),0.);
    if(a.diagnostics)*a.diagnostics=result;
    if(!result.converged) return false;
    for(PxU32 i=0;i<n;++i)
    { a.step[i/6][i%6]=PxReal(x[i]);if(!PxIsFinite(a.step[i/6][i%6])) return false; }
    return true;
}
} }
#endif
