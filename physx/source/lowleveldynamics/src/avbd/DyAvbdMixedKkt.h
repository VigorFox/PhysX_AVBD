// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_MIXED_KKT_H
#define DY_AVBD_MIXED_KKT_H

#include "DyAvbdLinearResponse.h"

namespace physx { namespace Dy {

// Frozen H x - J^T f = base; gap = offset + J x + compliance*f.
// f = clip(f-rho*gap, lower, upper), for any rho>0. Full H must be SPD.
// Bounds are authored force limits, not a post-solve clamp. A saturated row
// may have nonzero gap. No Coulomb law, geometry, pose or velocity publication.
// This is an operator view of the same row/block program, not another IR.
struct AvbdMixedKktResult
{
    bool converged;
    PxU32 activeIterations,krylovIterations,materialProducts;
    double residual;
};
struct AvbdMixedKktArgs
{
    void* context;
    void (*hessian)(void*,const double*,double*);
    void (*jacobian)(void*,const double*,double*);
    void (*transpose)(void*,const double*,double*);
    void (*precondition)(void*,const double*,double*);
    PxU32 primalSize,rowCount,activeBudget,krylovBudget;
    const double *base,*offset,*compliance,*lower,*upper,*dualDiagonal;
    // force is an optional zero or same-row/same-epoch algebraic seed. The
    // caller owns identity and units; failed trial forces are never published.
    double *solution,*force,*work; // 12*(primalSize+rowCount) scratch doubles
    PxU8* active; // 0=free, 1=lower, 2=upper; rowCount scratch bytes
    double tolerance;
    AvbdMixedKktResult* result;
};

// Preconditioned Lanczos + MINRES rotations (Paige/Saunders). The mixed
// matrix can be indefinite or rank deficient. Its preconditioner must be SPD.
// Always test original residual components, not a least-squares stop alone.
template<class Operator> PX_INLINE AvbdLinearResponseResult avbdMixedMinres(
    const Operator& op,PxU32 n,const double* rhs,double* x,double* work,
    PxU32 budget,double tolerance)
{
    AvbdLinearResponseResult out={false,0,0,std::numeric_limits<double>::infinity()};
    double *r1=work,*r2=work+n,*y=work+2*n,*v=work+3*n;
    double *w1=work+4*n,*w2=work+5*n,*w=work+6*n,*product=work+7*n;
    for(PxU32 i=0;i<n;++i){x[i]=r1[i]=w1[i]=w2[i]=0;r2[i]=rhs[i];}
    op.precondition(r2,y);double betaSquared=0;
    for(PxU32 i=0;i<n;++i)betaSquared+=r2[i]*y[i];
    out.residual=op.error(r2,y);
    if(out.residual<=tolerance){out.converged=true;return out;}
    if(!(betaSquared>0) || !std::isfinite(betaSquared))return out;
    double beta=std::sqrt(betaSquared),oldBeta=0,cs=-1,sn=0,dbar=0,eps=0,phibar=beta;
    for(PxU32 k=0;k<budget;++k)
    {
        for(PxU32 i=0;i<n;++i)v[i]=y[i]/beta;
        op.apply(v,product);++out.operatorCalls;
        if(k)for(PxU32 i=0;i<n;++i)product[i]-=(beta/oldBeta)*r1[i];
        double alpha=0;for(PxU32 i=0;i<n;++i)alpha+=v[i]*product[i];
        for(PxU32 i=0;i<n;++i)product[i]-=(alpha/beta)*r2[i];
        double* temp=r1;r1=r2;r2=temp;
        for(PxU32 i=0;i<n;++i)r2[i]=product[i];
        op.precondition(r2,y);oldBeta=beta;betaSquared=0;
        for(PxU32 i=0;i<n;++i)betaSquared+=r2[i]*y[i];
        if(!(betaSquared>=0) || !std::isfinite(betaSquared))return out;
        beta=std::sqrt(betaSquared);
        const double oldEps=eps,delta=cs*dbar+sn*alpha,gbar=sn*dbar-cs*alpha;
        eps=sn*beta;dbar=-cs*beta;
        const double gamma=std::hypot(gbar,beta);
        if(gamma>0 && std::isfinite(gamma))
        {
            cs=gbar/gamma;sn=beta/gamma;
            const double phi=cs*phibar;phibar*=sn;
            for(PxU32 i=0;i<n;++i)
            {w[i]=(v[i]-oldEps*w1[i]-delta*w2[i])/gamma;x[i]+=phi*w[i];}
            temp=w1;w1=w2;w2=w;w=temp;
        }
        else if(!std::isfinite(gamma))return out;
        out.iterations=k+1;
        if(std::abs(phibar)<=tolerance || (k+1)%16==0 || k+1==budget || beta==0 || gamma==0)
        {
            op.apply(x,product);++out.operatorCalls;
            for(PxU32 i=0;i<n;++i)v[i]=rhs[i]-product[i];
            op.precondition(v,w);out.residual=op.error(v,w);
            if(out.residual<=tolerance){out.converged=true;return out;}
        }
        if(beta==0 || gamma==0)break;
    }
    return out;
}

struct AvbdMixedFace
{
    const AvbdMixedKktArgs& a;
    double *primalScratch,*dualScratch;
    void apply(const double* x,double* y) const
    {
        const PxU32 n=a.primalSize,m=a.rowCount;
        a.hessian(a.context,x,y);
        for(PxU32 i=0;i<m;++i)dualScratch[i]=a.active[i]?0:x[n+i];
        a.transpose(a.context,dualScratch,primalScratch);
        for(PxU32 i=0;i<n;++i)y[i]-=primalScratch[i];
        a.jacobian(a.context,x,y+n);
        for(PxU32 i=0;i<m;++i)y[n+i]=a.active[i]?a.dualDiagonal[i]*x[n+i]:-y[n+i]-a.compliance[i]*x[n+i];
    }
    void precondition(const double* x,double* y) const
    {
        a.precondition(a.context,x,y);
        for(PxU32 i=0;i<a.rowCount;++i)y[a.primalSize+i]=x[a.primalSize+i]/a.dualDiagonal[i];
    }
    double error(const double* raw,const double* preconditioned) const
    {
        double e=0;
        for(PxU32 i=0;i<a.primalSize+a.rowCount;++i)
        {
            if(!std::isfinite(raw[i]) || !std::isfinite(preconditioned[i]))return std::numeric_limits<double>::infinity();
            e=PxMax(e,std::abs(i<a.primalSize?preconditioned[i]:raw[i]));
        }
        return e;
    }
};

PX_INLINE void avbdSolveMixedKkt(const AvbdMixedKktArgs& a)
{
    AvbdMixedKktResult& out=*a.result;
    out={false,0,0,0,std::numeric_limits<double>::infinity()};
    const PxU32 n=a.primalSize,m=a.rowCount,N=n+m;
    if(!(a.tolerance>0) || !std::isfinite(a.tolerance))return;
    if(!N){out.converged=true;out.residual=0;return;}
    for(PxU32 i=0;i<n;++i){a.solution[i]=0;if(!std::isfinite(a.base[i]))return;}
    for(PxU32 i=0;i<m;++i)
    {
        if(!std::isfinite(a.force[i]) || !std::isfinite(a.offset[i]) || !std::isfinite(a.compliance[i]) || a.compliance[i]<0 ||
           !(a.dualDiagonal[i]>0) || !std::isfinite(a.dualDiagonal[i]) ||
           std::isnan(a.lower[i]) || std::isnan(a.upper[i]) || a.lower[i]>a.upper[i] ||
           a.lower[i]==std::numeric_limits<double>::infinity() || a.upper[i]==-std::numeric_limits<double>::infinity())return;
        a.force[i]=PxMax(a.lower[i],PxMin(a.upper[i],a.force[i]));
        a.active[i]=a.force[i]==a.lower[i]?1:a.force[i]==a.upper[i]?2:0;
    }
    double *rhs=a.work,*candidate=rhs+N,*linearWork=candidate+N,*temporary=a.work+10*N,*gap=a.work+11*N;
    const AvbdMixedFace op={a,temporary,temporary+n};
    for(PxU32 outer=0;outer<a.activeBudget;++outer)
    {
        out.activeIterations=outer+1;
        for(PxU32 i=0;i<m;++i)op.dualScratch[i]=a.active[i]?a.force[i]:0;
        a.transpose(a.context,op.dualScratch,rhs);
        for(PxU32 i=0;i<n;++i)rhs[i]+=a.base[i];
        for(PxU32 i=0;i<m;++i)rhs[n+i]=a.active[i]?0:a.offset[i];
        const auto linear=avbdMixedMinres(op,N,rhs,candidate,linearWork,a.krylovBudget,a.tolerance*.1);
        out.krylovIterations+=linear.iterations;out.materialProducts+=linear.operatorCalls;out.residual=linear.residual;
        if(!linear.converged)return;
        for(PxU32 i=0;i<m;++i)if(a.active[i])candidate[n+i]=a.force[i];
        double fraction=1,bound=0;PxU32 leaving=PX_MAX_U32;PxU8 boundState=0;
        for(PxU32 i=0;i<m;++i)if(!a.active[i])
        {
            const double f=candidate[n+i];
            if(!std::isfinite(f))return;
            if(f<a.lower[i] || f>a.upper[i])
            {
                const double target=f<a.lower[i]?a.lower[i]:a.upper[i];
                const double step=(target-a.force[i])/(f-a.force[i]);
                if(step<fraction || leaving==PX_MAX_U32)
                {fraction=step;leaving=i;bound=target;boundState=f<a.lower[i]?1:2;}
            }
        }
        if(leaving!=PX_MAX_U32)
        {
            for(PxU32 i=0;i<m;++i)if(!a.active[i])a.force[i]+=fraction*(candidate[n+i]-a.force[i]);
            a.force[leaving]=bound;a.active[leaving]=boundState;continue;
        }
        for(PxU32 i=0;i<m;++i)a.force[i]=candidate[n+i];
        a.jacobian(a.context,candidate,gap);
        PxU32 entering=PX_MAX_U32;double most=a.tolerance;out.residual=linear.residual;
        for(PxU32 i=0;i<m;++i)
        {
            gap[i]+=a.offset[i]+a.compliance[i]*a.force[i];
            if(!std::isfinite(gap[i]) || !std::isfinite(a.force[i]))return;
            const double violation=a.lower[i]==a.upper[i]?0:a.active[i]==1?PxMax(0.,-gap[i]):
                a.active[i]==2?PxMax(0.,gap[i]):std::abs(gap[i]);
            out.residual=PxMax(out.residual,violation);
            if(a.active[i] && violation>most){most=violation;entering=i;}
        }
        if(entering==PX_MAX_U32)
        {
            if(out.residual>a.tolerance)return;
            for(PxU32 i=0;i<n;++i)a.solution[i]=candidate[i];
            out.converged=true;return;
        }
        a.active[entering]=0;
    }
}

} }
#endif
