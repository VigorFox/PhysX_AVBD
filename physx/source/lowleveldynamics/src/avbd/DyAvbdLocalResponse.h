// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_LOCAL_RESPONSE_H
#define DY_AVBD_LOCAL_RESPONSE_H
#include "DyAvbdCommonProgram.h"
#include "DyKernelTable.h"
#include "DyAvbdLocalAcceleration.h"
#include "DyAvbdLocalCorrection.h"
#include "foundation/PxUserAllocated.h"

namespace physx { namespace Dy {
// Numerical block AL on the same frozen H/J. No physical state or second
// material model. Trial iterates may be infeasible; the caller owns publication.
struct AvbdLocalResponse : public PxUserAllocated
{
    const AvbdCommonProgram& program;const KernelTable& kernels;
    struct Block {double inverse[36],augmentedInverse[36];};
    PxArray<Block> blocks;
    PxArray<double> penalty,work,motionTolerance;
    PxArray<PxU32> colors,colorStarts,colorBlocks;
    PxArray<PxU8> failed,enabled;
    PxU32 materialSolves=0,materialProducts=0,primalProducts=0,activeGroups=0,sweeps=0;
    PxU32 accelerated=0;
    PxU32 corrections=0,correctionProducts=0;
    double stationError=0,forceError=0;
    bool valid;

    static bool solveBlock(const double* matrix,const double* rhs,double* x,PxU32 n)
    {return avbdSolveSmallBlock(matrix,rhs,x,n);}
    AvbdLocalResponse(const AvbdCommonProgram& p,const KernelTable& table):program(p),kernels(table),valid(p.valid)
    {
        if(!valid)return;const PxU32 n=p.dimensions.size();blocks.resize(n);failed.resize(n,0);colors.resize(n,PX_MAX_U32);
        work.resize(4*p.base.size());penalty.resize(p.offset.size(),0);enabled.resize(p.mu.size(),0);PxU32 colorCount=0;
        for(PxU32 b=0;b<n;++b)
        {
            const double* diagonal=NULL;
            for(PxU32 e=p.hOffsets[b];e<p.hOffsets[b+1];++e)if(p.H[e].column==b)diagonal=p.H[e].values;
            if(!diagonal){valid=false;return;}
            for(PxU32 k=0;k<p.dimensions[b];++k)
            {
                double rhs[6]={},x[6]={};rhs[k]=1;
                if(!solveBlock(diagonal,rhs,x,p.dimensions[b])){valid=false;return;}
                for(PxU32 d=0;d<p.dimensions[b];++d)blocks[b].inverse[6*d+k]=x[d];
            }
            PxArray<PxU8> used(colorCount,0);
            auto mark=[&](PxU32 other){if(other<b)used[colors[other]]=1;};
            for(PxU32 e=p.hOffsets[b];e<p.hOffsets[b+1];++e)mark(p.H[e].column);
            for(PxU32 s=p.transposeOffsets[b];s<p.transposeOffsets[b+1];++s)
            {const PxU32 r=p.J[p.transposeEntries[s]].row;for(PxU32 e=p.jOffsets[r];e<p.jOffsets[r+1];++e)mark(p.J[e].column);}
            PxU32 color=0;while(color<colorCount && used[color])++color;
            colors[b]=color;if(color==colorCount)++colorCount;
        }
        colorStarts.resize(colorCount+1,0);for(PxU32 color:colors)++colorStarts[color+1];
        for(PxU32 c=1;c<=colorCount;++c)colorStarts[c]+=colorStarts[c-1];
        PxArray<PxU32> cursor=colorStarts;colorBlocks.resize(n);
        for(PxU32 b=0;b<n;++b)colorBlocks[cursor[colors[b]]++]=b;
        // Local block response sets numerical AL units; no physical compliance
        // is added. All cross-block material terms remain in the primal residual.
        for(const auto& j:p.J)for(PxU32 q=0;q<3;++q)
            for(PxU32 d=0;d<p.dimensions[j.column];++d)for(PxU32 k=0;k<p.dimensions[j.column];++k)
                penalty[3*j.row+q]+=j.values[6*q+d]*blocks[j.column].inverse[6*d+k]*j.values[6*q+k];
        for(PxU32 r=0;r<p.mu.size();++r)
        {
            if(p.coulomb[r])penalty[3*r+1]=penalty[3*r+2]=std::max(penalty[3*r+1],penalty[3*r+2]);
            for(PxU32 q=0;q<3;++q)
            {double& w=penalty[3*r+q];w=w>0?2/w:1.;}
        }
    }
    static void hessian(void* context,const double* x,double* y)
    {static_cast<AvbdLocalResponse*>(context)->program.hessian(x,y);}
    static void precondition(void* context,const double* r,double* z)
    {
        const auto& self=*static_cast<AvbdLocalResponse*>(context);const auto& p=self.program;
        for(PxU32 b=0;b<p.dimensions.size();++b)for(PxU32 d=0;d<p.dimensions[b];++d)
        {double v=0;for(PxU32 k=0;k<p.dimensions[b];++k)v+=self.blocks[b].inverse[6*d+k]*r[p.starts[b]+k];z[p.starts[b]+d]=v;}
    }
    // Initial force-balanced motion and final engineering residual checks use
    // full H products; contacts never request an H-inverse response column.
    bool material(const double* rhs,double* x)
    {
        if(!valid)return false;AvbdLinearResponseResult result={};const PxU32 n=program.base.size();
        AvbdLinearResponseArgs args={this,hessian,precondition,n,4*n,rhs,x,work.begin(),1e-12,1e-14,&result};
        kernels.launch(AVBD_LINEAR_RESPONSE,&args);++materialSolves;materialProducts+=result.operatorCalls;
        valid=result.converged;return valid;
    }
    void projected(PxU32 r,const double* step,const double* force,double* f,double* stiffness,bool updateCapacity=false,double* coordinates=NULL,double* residual=NULL) const
    {program.projected(r,penalty.begin(),step,force,f,stiffness,updateCapacity,coordinates,residual);}
    bool refreshFactors()
    {
        const auto& p=program;
        for(PxU32 b=0;b<blocks.size();++b)
        {
            double A[36]={};const PxU32 n=p.dimensions[b];
            for(PxU32 e=p.hOffsets[b];e<p.hOffsets[b+1];++e)if(p.H[e].column==b)
                for(PxU32 d=0;d<36;++d)A[d]=p.H[e].values[d];
            for(PxU32 s=p.transposeOffsets[b];s<p.transposeOffsets[b+1];++s)
            {
                const auto& j=p.J[p.transposeEntries[s]];if(!enabled[j.row])continue;
                for(PxU32 q=0;q<3;++q)
                {
                    const PxU32 i=3*j.row+q;const double k=p.lower[i]<p.upper[i] && !(q && p.coulomb[j.row] && p.mu[j.row]==0)?penalty[i]/(1+penalty[i]*p.compliance[i]):0;
                    for(PxU32 d=0;d<n;++d)for(PxU32 c=0;c<n;++c)A[6*d+c]+=j.values[6*q+d]*k*j.values[6*q+c];
                }
            }
            for(PxU32 c=0;c<n;++c)
            {double rhs[6]={},x[6]={};rhs[c]=1;if(!solveBlock(A,rhs,x,n)){valid=false;return false;}
                for(PxU32 d=0;d<n;++d)blocks[b].augmentedInverse[6*d+c]=x[d];}
        }
        return true;
    }
    void primal(const double* force,double* step,PxU32 begin,PxU32 end)
    {
        const auto& p=program;
        for(PxU32 slot=begin;slot<end;++slot)
        {
            const PxU32 b=colorBlocks[slot],n=p.dimensions[b],start=p.starts[b];double rhs[6]={},dx[6]={};
            for(PxU32 d=0;d<n;++d)rhs[d]=p.base[start+d];
            for(PxU32 e=p.hOffsets[b];e<p.hOffsets[b+1];++e)
            {
                const auto& h=p.H[e];
                for(PxU32 d=0;d<n;++d)for(PxU32 k=0;k<p.dimensions[h.column];++k)
                    rhs[d]-=h.values[6*d+k]*step[p.starts[h.column]+k];
            }
            for(PxU32 s=p.transposeOffsets[b];s<p.transposeOffsets[b+1];++s)
            {
                const auto& j=p.J[p.transposeEntries[s]];if(!enabled[j.row])continue;
                double f[3],k[3];projected(j.row,step,force,f,k);
                for(PxU32 q=0;q<3;++q)for(PxU32 d=0;d<n;++d)
                    rhs[d]+=j.values[6*q+d]*f[q];
            }
            failed[b]=0;for(PxU32 d=0;d<n;++d)
            {for(PxU32 c=0;c<n;++c)dx[d]+=blocks[b].augmentedInverse[6*d+c]*rhs[c];failed[b]=failed[b] || !std::isfinite(dx[d]);}
            if(!failed[b])for(PxU32 d=0;d<n;++d)step[start+d]+=dx[d];
        }
    }
    void dual(const double* step,double* force,PxU32 begin,PxU32 end)
    {
        for(PxU32 r=begin;r<end;++r)if(enabled[r])
        {double f[3],k[3];projected(r,step,force,f,k,true);for(PxU32 q=0;q<3;++q)force[3*r+q]=f[q];}
    }
    static void augmentedProduct(void* context,const double* x,double* y)
    {
        const auto& self=*static_cast<AvbdLocalResponse*>(context);const auto& p=self.program;
        p.hessian(x,y);
        for(PxU32 r=0;r<p.mu.size();++r)if(self.enabled[r])
        {
            double u[3]={};
            for(PxU32 e=p.jOffsets[r];e<p.jOffsets[r+1];++e)
            {const auto& j=p.J[e];for(PxU32 q=0;q<3;++q)for(PxU32 d=0;d<p.dimensions[j.column];++d)u[q]+=j.values[6*q+d]*x[p.starts[j.column]+d];}
            for(PxU32 q=0;q<3;++q)
            {const PxU32 i=3*r+q;u[q]*=p.lower[i]<p.upper[i] && !(q && p.coulomb[r] && p.mu[r]==0)?self.penalty[i]/(1+self.penalty[i]*p.compliance[i]):0.;}
            for(PxU32 e=p.jOffsets[r];e<p.jOffsets[r+1];++e)
            {const auto& j=p.J[e];for(PxU32 q=0;q<3;++q)for(PxU32 d=0;d<p.dimensions[j.column];++d)y[p.starts[j.column]+d]+=j.values[6*q+d]*u[q];}
        }
    }
    static void augmentedPrecondition(void* context,const double* r,double* z)
    {
        const auto& self=*static_cast<AvbdLocalResponse*>(context);const auto& p=self.program;
        for(PxU32 b=0;b<p.dimensions.size();++b)for(PxU32 d=0;d<p.dimensions[b];++d)
        {double v=0;for(PxU32 k=0;k<p.dimensions[b];++k)v+=self.blocks[b].augmentedInverse[6*d+k]*r[p.starts[b]+k];z[p.starts[b]+d]=v;}
    }
    bool groupPrimal(const double* force,double* step)
    {
        const auto& p=program;const PxU32 n=p.base.size();PxArray<double> rhs(n),delta(n);
        p.hessian(step,rhs.begin());for(PxU32 d=0;d<n;++d)rhs[d]=p.base[d]-rhs[d];
        for(PxU32 r=0;r<p.mu.size();++r)if(enabled[r])
        {
            double f[3],k[3];projected(r,step,force,f,k);
            for(PxU32 e=p.jOffsets[r];e<p.jOffsets[r+1];++e)
            {const auto& j=p.J[e];for(PxU32 q=0;q<3;++q)for(PxU32 d=0;d<p.dimensions[j.column];++d)rhs[p.starts[j.column]+d]+=j.values[6*q+d]*f[q];}
        }
        AvbdLinearResponseResult result={};
        AvbdLinearResponseArgs args={this,augmentedProduct,augmentedPrecondition,n,4*n,rhs.begin(),delta.begin(),work.begin(),1.e-10,1.e-13,&result};
        kernels.launch(AVBD_LINEAR_RESPONSE,&args);
        primalProducts+=result.operatorCalls;
        if(!result.converged){valid=false;return false;}
        for(PxU32 d=0;d<n;++d)step[d]+=delta[d];return true;
    }
    bool includeViolated(const double* step,const double* force,const PxArray<double>& tolerance)
    {
        const auto& p=program;PxArray<PxU32> pending(p.dimensions.size()+1,PX_MAX_U32);PxArray<double> scores(pending.size(),0);
        for(PxU32 r=0;r<p.mu.size();++r)if(!enabled[r])
        {
            double f[3],k[3],u[3],error=0;projected(r,step,force,f,k,true,u);
            for(PxU32 q=0;q<3;++q)
            {const double bound=tolerance.empty()?1.e-10:tolerance[3*r+q];
                const double defect=std::abs(f[q])/penalty[3*r+q];if(defect>bound)error=std::max(error,defect);}
            // Same one-sided numerical feasibility as the strict NCP. Actual
            // stored geometry still has its separate, unchanged certificate.
            if(p.lower[3*r]==0 && std::isinf(p.upper[3*r]) && p.compliance[3*r]==0 && u[0]<-1.e-9)error=std::max(error,-u[0]);
            if(error>0)
            {
                const PxU32 b=p.jOffsets[r]<p.jOffsets[r+1]?p.J[p.jOffsets[r]].column:p.dimensions.size();
                if(error>scores[b]){scores[b]=error;pending[b]=r;}
            }
        }
        bool added=false;for(PxU32 r:pending)if(r!=PX_MAX_U32){enabled[r]=1;added=true;}return added;
    }
    double merit(const double* step,const double* force) const
    {
        const auto& p=program;PxArray<double> residual(p.base.size()),reaction(p.base.size()),scaled(p.base.size());
        p.hessian(step,residual.begin());p.transpose(force,reaction.begin());
        for(PxU32 d=0;d<residual.size();++d)residual[d]-=p.base[d]+reaction[d];
        precondition(const_cast<AvbdLocalResponse*>(this),residual.begin(),scaled.begin());double value=0;
        for(double v:scaled)value+=v*v;
        for(PxU32 r=0;r<p.mu.size();++r)if(enabled[r])
        {
            double f[3],k[3],root[3];projected(r,step,force,f,k,false,NULL,root);
            for(PxU32 q=0;q<3;++q){const double v=root[q]/penalty[3*r+q];value+=v*v;}
        }
        return std::isfinite(value)?value:std::numeric_limits<double>::infinity();
    }
    bool solve(PxArray<double>& step,PxArray<double>& force,AvbdCoulombNcpResult& result,bool grouped=false)
    {
        const auto& p=program;result={false,0,0,0,0};sweeps=0;
        step.resize(p.base.size(),0);force.resize(p.offset.size(),0);
        if(grouped)
        {
            // A force warm start and a free-motion primal are inconsistent.
            // Balance the inherited force before starting this same AL solve.
            PxArray<double> initial=p.base;
            PxArray<double> reaction(initial.size());p.transpose(force.begin(),reaction.begin());
            for(PxU32 d=0;d<initial.size();++d)initial[d]+=reaction[d];
            if(!material(initial.begin(),step.begin()))return false;
        }
        else if(!material(p.base.begin(),step.begin()))return false;
        PxArray<double> residual(step.size()),reaction(step.size()),scaled(step.size());
        PxArray<double> rowTolerance(p.offset.size(),0);
        const bool engineering=motionTolerance.size()==step.size();
        if(engineering)for(const auto& j:p.J)for(PxU32 q=0;q<3;++q)for(PxU32 d=0;d<p.dimensions[j.column];++d)
            rowTolerance[3*j.row+q]+=.1*std::abs(j.values[6*q+d])*motionTolerance[p.starts[j.column]+d];
        for(PxU32 r=0;r<p.mu.size();++r)
        {
            bool contact=p.lower[3*r]==0;
            if(!p.coulomb[r])for(PxU32 q=1;q<3;++q)contact=contact && p.lower[3*r+q]==0 && p.upper[3*r+q]==0;
            enabled[r]=!contact;
            for(PxU32 q=0;q<3;++q)enabled[r]=enabled[r] || force[3*r+q]!=0;
        }
        const PxArray<double> strictTolerance;includeViolated(step.begin(),force.begin(),engineering?rowTolerance:strictTolerance);
        if(!refreshFactors())return false;
        AvbdLocalAcceleration acceleration;PxArray<double> input(step.size()+force.size()),output(input.size()),candidate;
        double scale=1;for(double v:p.base)scale=std::max(scale,std::abs(v));
        for(PxU32 it=0;it<1024;++it)
        {
            for(PxU32 d=0;d<step.size();++d)input[d]=step[d];
            for(PxU32 i=0;i<force.size();++i)input[step.size()+i]=force[i]/penalty[i];
            if(grouped){if(!groupPrimal(force.begin(),step.begin()))return false;}
            else for(PxU32 c=0;c+1<colorStarts.size();++c)
            {AvbdLocalSweepArgs args={this,step.begin(),force.begin(),colorStarts[c],colorStarts[c+1]};kernels.launch(AVBD_LOCAL_PRIMAL,&args);}
            for(PxU8 f:failed)if(f){valid=false;return false;}
            AvbdLocalSweepArgs args={this,step.begin(),force.begin(),0,p.mu.size()};kernels.launch(AVBD_LOCAL_DUAL,&args);
            if(grouped)
            {AvbdLocalCorrection correction(p,penalty,enabled,this,precondition);
                AvbdLocalCorrectionArgs correctionArgs={&correction,&step,&force,&correctionProducts,false};
                kernels.launch(AVBD_LOCAL_CORRECTION,&correctionArgs);
                if(correctionArgs.accepted)++corrections;}
            ++sweeps;result.iterations=sweeps;
            for(PxU32 d=0;d<step.size();++d)output[d]=step[d];
            for(PxU32 i=0;i<force.size();++i)output[step.size()+i]=force[i]/penalty[i];
            if(acceleration.propose(input,output,candidate))
            {
                PxArray<double> proposedForce(force.size());for(PxU32 i=0;i<force.size();++i)proposedForce[i]=candidate[step.size()+i]*penalty[i];
                // Extrapolation may leave the force cone. Project the numerical
                // trial before its merit check, including finite force limits.
                for(PxU32 r=0;r<p.mu.size();++r)
                {
                    for(PxU32 q=0;q<3;++q)
                    {const PxU32 i=3*r+q;proposedForce[i]=std::max(p.lower[i],std::min(p.upper[i],proposedForce[i]));}
                    if(p.coulomb[r])
                    {const double bound=p.mu[r]*std::max(0.,proposedForce[3*r]),mag=std::hypot(proposedForce[3*r+1],proposedForce[3*r+2]);
                        if(mag>bound)for(PxU32 q=1;q<3;++q)proposedForce[3*r+q]*=bound/mag;}
                }
                if(merit(candidate.begin(),proposedForce.begin())<merit(step.begin(),force.begin()))
                {for(PxU32 d=0;d<step.size();++d)step[d]=candidate[d];force=proposedForce;++accelerated;}
            }
            p.hessian(step.begin(),residual.begin());p.transpose(force.begin(),reaction.begin());
            double station=0,stationScale=scale;
            for(PxU32 d=0;d<step.size();++d)
            {residual[d]-=p.base[d]+reaction[d];station=std::max(station,std::abs(residual[d]));stationScale=std::max(stationScale,std::abs(reaction[d]));}
            precondition(this,residual.begin(),scaled.begin());double error=0;
            bool smallStation=true;
            for(PxU32 d=0;d<scaled.size();++d)
            {error=std::max(error,std::abs(scaled[d]));if(engineering)smallStation=smallStation && std::abs(scaled[d])<=.1*motionTolerance[d];}
            double rootError=0;activeGroups=0;
            bool smallForce=true;
            for(PxU32 r=0;r<p.mu.size();++r)if(enabled[r])
            {
                double f[3],k[3],u[3],root[3];projected(r,step.begin(),force.begin(),f,k,true,u,root);activeGroups+=force[3*r]!=0;
                for(PxU32 q=0;q<3;++q)
                {const double defect=std::abs(root[q])/penalty[3*r+q];rootError=std::max(rootError,defect);
                    if(engineering)smallForce=smallForce && defect<=rowTolerance[3*r+q];}
            }
            result.residual=std::max(error,rootError);
            stationError=error;forceError=rootError;
            if(grouped && engineering && smallStation && !smallForce && ((it+1)&it)==0 && it+1<1024)
            {
                // Continue the numerical AL metric only after primal balance
                // is small. Unscaled force and the physical fixed point stay
                // unchanged. Rebuild every dependent numerical factor/history.
                for(PxU32 r=0;r<p.mu.size();++r)if(enabled[r])
                {
                    double f[3],k[3],root[3];projected(r,step.begin(),force.begin(),f,k,true,NULL,root);bool unresolved=false;
                    for(PxU32 q=0;q<3;++q)unresolved=unresolved || std::abs(root[q])/penalty[3*r+q]>rowTolerance[3*r+q];
                    if(unresolved)for(PxU32 q=0;q<3;++q)penalty[3*r+q]*=2;
                }
                if(!refreshFactors())return false;acceleration.reset();continue;
            }
            if(engineering?(smallStation && smallForce):(result.residual<1.e-10 && station/stationScale<1.e-10))
            {
                // Omitted rows have exactly zero force. Recheck every original
                // constraint before qualifying a reduced numerical work set.
                if(includeViolated(step.begin(),force.begin(),engineering?rowTolerance:strictTolerance))
                {if(!refreshFactors())return false;acceleration.reset();continue;}
                if(engineering)
                {
                    // A small diagonal-preconditioned residual can hide a
                    // collective weak mode. Check the full material correction
                    // once near closure, using only block-preconditioned products.
                    if(!material(residual.begin(),scaled.begin()))return false;
                    bool small=true;for(PxU32 d=0;d<scaled.size();++d)small=small && std::abs(scaled[d])<=.1*motionTolerance[d];
                    if(!small)
                    {
                        // In the coupled primal executor this is a certificate,
                        // not an unconstrained H-only correction. Injecting that
                        // correction breaks the contact-balanced AL iteration.
                        if(!grouped){for(PxU32 d=0;d<step.size();++d)step[d]-=scaled[d];acceleration.reset();}
                        continue;
                    }
                }
                result.converged=true;return true;
            }
        }
        return false;
    }
};
} }
#endif
