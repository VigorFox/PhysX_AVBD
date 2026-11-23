// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_NORMAL_CONE_H
#define DY_AVBD_NORMAL_CONE_H

#include "foundation/PxMath.h"
#include <cmath>
#include <limits>

namespace physx { namespace Dy {

// Frozen normal/bilateral closure, not a nonlinear step or a pose publisher.
// W = J H^-1 J^T is applied by the caller; no dense Schur matrix is built.
// W f - demand is the gap. Bilateral f is free; normal f >= 0.
// H must be SPD. Row coordinates/tolerance must have caller-defined scales.
// This kernel does not implement Coulomb tangents or finite force bounds.
// The caller supplies six n-vectors and n active bytes. An unresolved result
// must NOT be published or interpreted as an infeasibility certificate.
struct AvbdNormalConeResult
{
    bool converged;
    PxU32 activeIterations, operatorCalls;
    double residual;
};

template<class Operator> PX_INLINE AvbdNormalConeResult avbdSolveNormalCone(
    const Operator& op, PxU32 n, const PxU8* bilateral, const double* demand,
    const double* diagonal, double* force, double* work, PxU8* active,
    PxU32 activeBudget, PxU32 krylovBudget, double tolerance)
{
    AvbdNormalConeResult result={false,0,0,std::numeric_limits<double>::infinity()};
    if(!n){result.converged=true;result.residual=0;return result;}
    if(!(tolerance>0) || !std::isfinite(tolerance))return result;
    double* candidate=work;double* residual=work+n;double* direction=work+2*n;
    double* product=work+3*n;double* z=work+4*n;double* gap=work+5*n;
    for(PxU32 i=0;i<n;++i)
    {
        force[i]=0;active[i]=bilateral[i]?1:0;
        if(!std::isfinite(demand[i]) || !std::isfinite(diagonal[i]) || diagonal[i]<0)
            return result;
    }
    for(PxU32 outer=0;outer<activeBudget;++outer)
    {
        result.activeIterations=outer+1;
        // Solve the current face from zero. PSD/rank-dependent constraints
        // are allowed; no artificial diagonal makes an inconsistent row pass.
        double rz=0,maxResidual=0;
        for(PxU32 i=0;i<n;++i)
        {
            candidate[i]=0;residual[i]=active[i]?demand[i]:0;
            z[i]=diagonal[i]>0?residual[i]/diagonal[i]:0;direction[i]=z[i];
            rz+=residual[i]*z[i];maxResidual=PxMax(maxResidual,std::abs(residual[i]));
        }
        for(PxU32 k=0;k<krylovBudget && maxResidual>tolerance;++k)
        {
            op.apply(direction,product);++result.operatorCalls;
            double curvature=0;
            for(PxU32 i=0;i<n;++i)if(active[i])curvature+=direction[i]*product[i];
            if(!(curvature>0) || !std::isfinite(curvature) || !(rz>0)) return result;
            const double alpha=rz/curvature;double nextRz=0;
            for(PxU32 i=0;i<n;++i)
            {
                candidate[i]+=alpha*direction[i];
                residual[i]=active[i]?residual[i]-alpha*product[i]:0;
                z[i]=diagonal[i]>0?residual[i]/diagonal[i]:0;
                nextRz+=residual[i]*z[i];
            }
            maxResidual=0;
            for(PxU32 i=0;i<n;++i)maxResidual=PxMax(maxResidual,std::abs(residual[i]));
            const double beta=nextRz/rz;rz=nextRz;
            for(PxU32 i=0;i<n;++i)direction[i]=z[i]+beta*direction[i];
        }
        // Always verify the true operator residual, not the CG recurrence.
        op.apply(candidate,product);++result.operatorCalls;
        maxResidual=0;
        for(PxU32 i=0;i<n;++i)
        {
            if(!std::isfinite(product[i]) || !std::isfinite(candidate[i]))return result;
            if(active[i])maxResidual=PxMax(maxResidual,std::abs(product[i]-demand[i]));
        }
        if(!std::isfinite(maxResidual) || maxResidual>tolerance) return result;
        double fraction=1;PxU32 leaving=PX_MAX_U32;
        for(PxU32 i=0;i<n;++i)if(active[i] && !bilateral[i] && candidate[i]<0)
        {
            const double step=force[i]/(force[i]-candidate[i]);
            if(step<fraction || leaving==PX_MAX_U32){fraction=step;leaving=i;}
        }
        if(leaving!=PX_MAX_U32)
        {
            for(PxU32 i=0;i<n;++i)force[i]+=fraction*(candidate[i]-force[i]);
            force[leaving]=0;active[leaving]=0;continue;
        }
        for(PxU32 i=0;i<n;++i)force[i]=candidate[i];
        op.apply(force,gap);++result.operatorCalls;
        PxU32 entering=PX_MAX_U32;double mostNegative=-tolerance;
        result.residual=0;
        for(PxU32 i=0;i<n;++i)
        {
            gap[i]-=demand[i];
            const double violation=active[i]?std::abs(gap[i]):PxMax(0.,-gap[i]);
            result.residual=PxMax(result.residual,violation);
            if(!active[i] && gap[i]<mostNegative){mostNegative=gap[i];entering=i;}
            if(!std::isfinite(gap[i]) || !std::isfinite(force[i]))return result;
        }
        if(entering==PX_MAX_U32){result.converged=true;return result;}
        active[entering]=1;
    }
    return result;
}

} }
#endif
