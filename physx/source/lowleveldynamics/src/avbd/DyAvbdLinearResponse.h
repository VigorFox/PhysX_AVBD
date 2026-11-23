// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_LINEAR_RESPONSE_H
#define DY_AVBD_LINEAR_RESPONSE_H

#include "foundation/PxMath.h"
#include <cmath>
#include <limits>

namespace physx { namespace Dy {

// Frozen SPD response H x = rhs. The caller applies the FULL material/inertia
// operator and an SPD preconditioner separately. In particular, block Jacobi
// is never substituted for H^-1. No constraint representation or allocation.
// Four n-vectors are caller-owned scratch. Only a converged result is usable;
// budget exhaustion / invalid input is unresolved, not infeasibility.
struct AvbdLinearResponseResult
{
    bool converged;
    PxU32 iterations, operatorCalls;
    double residual;
};

template<class Operator> PX_INLINE AvbdLinearResponseResult avbdLinearResponse(
    const Operator& op, PxU32 n, const double* rhs, double* x, double* work,
    PxU32 budget, double relativeTolerance, double absoluteTolerance)
{
    AvbdLinearResponseResult result={false,0,0,std::numeric_limits<double>::infinity()};
    if(!(relativeTolerance>0) || !std::isfinite(relativeTolerance) ||
       !(absoluteTolerance>=0) || !std::isfinite(absoluteTolerance))return result;
    if(!n){result.converged=true;result.residual=0;return result;}
    double* r=work;double* z=work+n;double* p=work+2*n;double* product=work+3*n;
    double scale=0;
    for(PxU32 i=0;i<n;++i)
    {
        x[i]=0;r[i]=rhs[i];
        if(!std::isfinite(rhs[i]))return result;
        scale=PxMax(scale,std::abs(rhs[i]));
    }
    const double tolerance=PxMax(absoluteTolerance,relativeTolerance*scale);
    result.residual=scale;
    if(scale<=tolerance){result.converged=true;return result;}
    bool restart=true;double rz=0;
    for(PxU32 k=0;k<budget;++k)
    {
        op.precondition(r,z);double nextRz=0;
        for(PxU32 i=0;i<n;++i)
        {if(!std::isfinite(z[i]))return result;nextRz+=r[i]*z[i];}
        if(!(nextRz>0) || !std::isfinite(nextRz))return result;
        const double beta=restart?0:nextRz/rz;rz=nextRz;
        for(PxU32 i=0;i<n;++i)p[i]=restart?z[i]:z[i]+beta*p[i];
        restart=false;
        op.apply(p,product);++result.operatorCalls;double curvature=0;
        for(PxU32 i=0;i<n;++i)
        {if(!std::isfinite(product[i]))return result;curvature+=p[i]*product[i];}
        if(!(curvature>0) || !std::isfinite(curvature))return result;
        const double alpha=rz/curvature;double recurrence=0;
        for(PxU32 i=0;i<n;++i)
        {
            x[i]+=alpha*p[i];r[i]-=alpha*product[i];
            if(!std::isfinite(x[i]) || !std::isfinite(r[i]))return result;
            recurrence=PxMax(recurrence,std::abs(r[i]));
        }
        result.iterations=k+1;
        if(recurrence<=tolerance || k+1==budget)
        {
            // Never accept the recursively updated residual alone. If roundoff
            // broke conjugacy, restart from the true residual within this budget.
            op.apply(x,product);++result.operatorCalls;result.residual=0;
            for(PxU32 i=0;i<n;++i)
            {
                if(!std::isfinite(product[i]))return result;
                r[i]=rhs[i]-product[i];result.residual=PxMax(result.residual,std::abs(r[i]));
            }
            if(result.residual<=tolerance){result.converged=true;return result;}
            restart=true;
        }
    }
    return result;
}

} }
#endif
