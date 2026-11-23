// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_COMMON_RESPONSE_H
#define DY_AVBD_COMMON_RESPONSE_H
#include "DyAvbdCommonProgram.h"
#include "DyKernelTable.h"
#include "DyAvbdSparseSymmetric.h"
#include "DyAvbdLocalResponse.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace physx { namespace Dy {
inline bool avbdCommonVerbose()
{const char* value=std::getenv("PHYSX_AVBD_COMMON_STATS");return value && std::strcmp(value,"summary")!=0;}
// Numerical composition of existing Table kernels. Transient coefficients
// only: this object cannot publish a physical pose, multiplier or history.
struct AvbdCommonResponse : public PxUserAllocated
{
    const AvbdCommonProgram& program;const KernelTable& kernels;
    AvbdSparseSymmetric factor;
    PxArray<double> rhs,solution,work;
    PxU32 materialSolves=0,materialProducts=0,activeGroups=0;
    bool valid;
    AvbdLocalResponse* local;
    AvbdCommonResponse(const AvbdCommonProgram& p,const KernelTable& table,bool localBlocks=false):program(p),kernels(table),factor(localBlocks?0:int(p.base.size())),valid(p.valid),local(NULL)
    {
        if(!valid)return;
        if(localBlocks){local=PX_NEW(AvbdLocalResponse)(p,table);valid=local->valid;return;}
        const PxU32 n=p.base.size();rhs.resize(n);solution.resize(n);work.resize(4*n);
        for(const auto& e:p.H)for(PxU32 r=0;r<p.dimensions[e.row];++r)
            for(PxU32 c=0;c<p.dimensions[e.column];++c)
            {
                const PxU32 a=p.starts[e.row]+r,b=p.starts[e.column]+c;
                if(a>=b)factor.add(int(a),int(b),e.values[6*r+c]);
            }
        valid=factor.compile();if(!valid)return;
        AvbdSparseFactorArgs args={&factor,&valid};kernels.launch(AVBD_SPARSE_FACTOR,&args);
    }
    ~AvbdCommonResponse(){PX_DELETE(local);}
    AvbdCommonResponse(const AvbdCommonResponse&)=delete;
    AvbdCommonResponse& operator=(const AvbdCommonResponse&)=delete;
    static void hessian(void* context,const double* x,double* y)
    {static_cast<AvbdCommonResponse*>(context)->program.hessian(x,y);}
    static void precondition(void* context,const double* r,double* z)
    {
        auto& self=*static_cast<AvbdCommonResponse*>(context);bool solved=false;
        AvbdSparseSolveArgs args={&self.factor,r,z,&solved};self.kernels.launch(AVBD_SPARSE_SOLVE,&args);
        if(!solved)std::fill(z,z+self.program.base.size(),std::numeric_limits<double>::quiet_NaN());
    }
    bool material(const double* input,double* output)
    {
        if(!valid)return false;
        if(local){const bool solved=local->material(input,output);materialSolves=local->materialSolves;materialProducts=local->materialProducts;valid=local->valid;return solved;}
        // The sparse factor is a preconditioner, never the acceptance oracle.
        // Existing PCG checks the true full-H residual at the original tolerance.
        AvbdLinearResponseResult result={};const PxU32 n=program.base.size();
        AvbdLinearResponseArgs args={this,hessian,precondition,n,4*n,input,output,work.begin(),1e-12,1e-14,&result};
        kernels.launch(AVBD_LINEAR_RESPONSE,&args);++materialSolves;materialProducts+=result.operatorCalls;
        valid=result.converged;return valid;
    }
    static void response(void* context,const double* f,double* out)
    {
        auto& self=*static_cast<AvbdCommonResponse*>(context);const auto& p=self.program;
        p.transpose(f,self.rhs.begin());
        if(self.material(self.rhs.begin(),self.solution.begin()))p.jacobian(self.solution.begin(),out);
        else std::fill(out,out+p.offset.size(),std::numeric_limits<double>::quiet_NaN());
    }
    // Numerical working set only. All omitted groups have EXACTLY zero force
    // and are checked against the original root after every reduced solve.
    // No geometric contact or cross-block material support is discarded.
    bool contacts(const PxArray<double>& demand,PxArray<double>& force,AvbdCoulombNcpResult& result)
    {
        const auto& p=program;PxArray<PxU32> selected;PxArray<PxU8> included(p.mu.size(),0);
        PxArray<double> motion=demand;
        // A single-DOF variational response bounds lambda_max(J H^-1 J^T)
        // from below. Thus rhoUpper bounds the ORIGINAL root's scaling from
        // above, without evaluating every unused response column.
        PxArray<double> diagonal(p.base.size(),0),rhoUpper(p.mu.size(),0);
        for(const auto& h:p.H)if(h.row==h.column)for(PxU32 d=0;d<p.dimensions[h.row];++d)
            diagonal[p.starts[h.row]+d]=h.values[6*d+d];
        for(const auto& j:p.J)for(PxU32 q=0;q<3;++q)for(PxU32 d=0;d<p.dimensions[j.column];++d)
            rhoUpper[j.row]=std::max(rhoUpper[j.row],p.compliance[3*j.row+q]+
                j.values[6*q+d]*j.values[6*q+d]/diagonal[p.starts[j.column]+d]);
        for(PxU32 r=0;r<p.mu.size();++r)
        {for(PxU32 q=0;q<3;++q)rhoUpper[r]=std::max(rhoUpper[r],p.compliance[3*r+q]);rhoUpper[r]=rhoUpper[r]>0?1/rhoUpper[r]:1.;}
        struct Subset
        {
            AvbdCommonResponse& full;const PxArray<PxU32>& rows;PxArray<double> force,motion;
            Subset(AvbdCommonResponse& f,const PxArray<PxU32>& r):full(f),rows(r),
                force(f.program.offset.size(),0),motion(f.program.offset.size(),0){}
            static void apply(void* context,const double* input,double* output)
            {
                auto& s=*static_cast<Subset*>(context);std::fill(s.force.begin(),s.force.end(),0.);
                for(PxU32 r=0;r<s.rows.size();++r)for(PxU32 q=0;q<3;++q)s.force[3*s.rows[r]+q]=input[3*r+q];
                AvbdCommonResponse::response(&s.full,s.force.begin(),s.motion.begin());
                for(PxU32 r=0;r<s.rows.size();++r)for(PxU32 q=0;q<3;++q)output[3*r+q]=s.motion[3*s.rows[r]+q];
            }
        } subset(*this,selected);
        const PxU32 budget=8*(p.offset.size()+1);result={true,0,0,0,0};
        for(;;)
        {
            bool added=false;double scale=1,omittedError=0;for(double f:force)scale=std::max(scale,std::abs(f));
            PxArray<PxU32> pending(p.dimensions.size()+1,PX_MAX_U32);PxArray<double> violation(pending.size(),0);
            for(PxU32 r=0;r<p.mu.size();++r)if(!included[r])
            {
                const double zero[3]={0,0,0};double root[3];
                avbdCoulombRoot(zero,motion.begin()+3*r,rhoUpper[r],p.mu[r],p.lower.begin()+3*r,p.upper.begin()+3*r,p.coulomb[r]!=0,root);
                const double error=std::max(std::abs(root[0]),std::max(std::abs(root[1]),std::abs(root[2])))/scale;
                bool physical=true;
                for(PxU32 q=0;q<3;++q)
                {
                    if(!std::isfinite(motion[3*r+q]))return false;
                    physical=physical && p.lower[3*r+q]<=1.e-10 && p.upper[3*r+q]>=-1.e-10;
                }
                if(p.coulomb[r])physical=physical && (std::isfinite(p.upper[3*r])?
                    std::abs(p.upper[3*r]*std::min(0.,motion[3*r]))<=1.e-7:motion[3*r]>=-1.e-9);
                if(std::max(std::abs(force[3*r]),std::max(std::abs(force[3*r+1]),std::abs(force[3*r+2])))>1.e-10*scale)
                {included[r]=1;selected.pushBack(r);added=true;}
                else
                {
                    for(PxU32 q=0;q<3;++q)force[3*r+q]=0;
                    if(error<1.e-10 && physical){omittedError=std::max(omittedError,error);continue;}
                    // Add one worst violated group per first support, then
                    // recheck ALL others. This avoids seeding a singular bundle
                    // with every coincident face/time facet at once.
                    const PxU32 key=p.jOffsets[r]<p.jOffsets[r+1]?p.J[p.jOffsets[r]].column:p.dimensions.size();
                    const double score=physical?error:std::max(error,-motion[3*r]);
                    if(pending[key]==PX_MAX_U32 || score>violation[key]){pending[key]=r;violation[key]=score;}
                }
            }
            for(PxU32 r:pending)if(r!=PX_MAX_U32){included[r]=1;selected.pushBack(r);added=true;}
            activeGroups=selected.size();
            if(!added){result.residual=std::max(result.residual,omittedError);return valid && result.converged;}
            if(result.iterations>=budget){result.converged=false;return false;}
            PxArray<double> d,mu,lo,hi,c,f,scratch(avbdCoulombNcpWorkSize(selected.size()));PxArray<int> friction;
            for(PxU32 r:selected)
            {
                mu.pushBack(p.mu[r]);friction.pushBack(p.coulomb[r]);
                for(PxU32 q=0;q<3;++q)
                {const PxU32 i=3*r+q;d.pushBack(demand[i]);lo.pushBack(p.lower[i]);hi.pushBack(p.upper[i]);c.pushBack(p.compliance[i]);f.pushBack(force[i]);}
            }
            AvbdCoulombNcpResult solved={};
            AvbdCoulombNcpArgs args={&subset,Subset::apply,selected.size(),budget-result.iterations,
                d.begin(),mu.begin(),lo.begin(),hi.begin(),c.begin(),friction.begin(),f.begin(),scratch.begin(),&solved};
            if(avbdCommonVerbose())std::fprintf(stderr,
                "[AVBD_COMMON_WORKSET] blocks=%u groups=%u selected=%u used=%u status=BEGIN\n",p.dimensions.size(),p.mu.size(),selected.size(),result.iterations);
            kernels.launch(AVBD_COULOMB_NCP,&args);
            if(avbdCommonVerbose())std::fprintf(stderr,
                "[AVBD_COMMON_WORKSET] selected=%u iterations=%u residual=%.9g status=%s\n",selected.size(),solved.iterations,solved.residual,solved.converged?"PASS":"FAIL");
            result.iterations+=solved.iterations;result.linearIterations+=solved.linearIterations;
            result.sweeps+=solved.sweeps;result.residual=solved.residual;result.converged=solved.converged && valid;
            if(!result.converged)
            {
                const char* path=std::getenv("PHYSX_AVBD_COMMON_FAILED_NCP");
                FILE* file=path?std::fopen(path,"wb"):NULL;
                if(file)
                {
                    auto array=[&](const char* name,const auto& values)
                    {std::fprintf(file,"\"%s\":[",name);for(PxU32 i=0;i<values.size();++i)
                        {const double v=double(values[i]);if(i)std::fprintf(file,",");
                            if(std::isfinite(v))std::fprintf(file,"%.17g",v);else std::fprintf(file,"\"%s\"",v<0?"-inf":"inf");}
                        std::fprintf(file,"],");};
                    std::fprintf(file,"{");array("demand",d);array("mu",mu);array("lower",lo);array("upper",hi);
                    array("compliance",c);array("force",f);array("rows",selected);array("coulomb",friction);
                    std::fprintf(file,"\"response\":[");PxArray<double> unit(f.size(),0),column(f.size());
                    for(PxU32 j=0;j<f.size();++j)
                    {unit[j]=1;Subset::apply(&subset,unit.begin(),column.begin());unit[j]=0;
                        if(j)std::fprintf(file,",");std::fprintf(file,"[");
                        for(PxU32 i=0;i<f.size();++i)std::fprintf(file,"%s%.17g",i?",":"",column[i]);std::fprintf(file,"]");}
                    std::fprintf(file,"],\"columns\":true,\"iterations\":%u,\"residual\":%.17g}\n",result.iterations,result.residual);std::fclose(file);
                }
                return false;
            }
            for(PxU32 r=0;r<selected.size();++r)for(PxU32 q=0;q<3;++q)force[3*selected[r]+q]=f[3*r+q];
            response(this,force.begin(),motion.begin());if(!valid)return false;
            for(PxU32 i=0;i<motion.size();++i)motion[i]+=demand[i];
        }
    }
    bool solve(PxArray<double>& step,PxArray<double>& force,AvbdCoulombNcpResult& result,bool grouped=false)
    {
        if(local)
        {
            const char* capturePath=std::getenv("PHYSX_AVBD_COMMON_FAILED_NCP");
            PxArray<double> initialForce;if(capturePath)initialForce=force;
            const bool solved=local->solve(step,force,result,grouped);valid=local->valid;activeGroups=local->activeGroups;
            materialSolves=local->materialSolves;materialProducts=local->materialProducts;
            if(!solved)
            {FILE* file=capturePath?std::fopen(capturePath,"wb"):NULL;
                if(file){program.write(file,&initialForce,&local->motionTolerance);std::fclose(file);}}
            if(avbdCommonVerbose() || !solved)std::fprintf(stderr,
                "[AVBD_LOCAL_RESPONSE] blocks=%u groups=%u colors=%u sweeps=%u primalProducts=%u corrections=%u correctionIterations=%u residual=%.9g station=%.9g force=%.9g status=%s\n",
                program.dimensions.size(),program.mu.size(),local->colorStarts.size()-1,result.iterations,local->primalProducts,local->corrections,local->correctionProducts,result.residual,local->stationError,local->forceError,solved?"PASS":"FAIL");
            return solved;
        }
        result={false,0,0,0,std::numeric_limits<double>::infinity()};
        step.resize(program.base.size(),0);force.resize(program.offset.size(),0);
        if(!material(program.base.begin(),step.begin()))return false;
        if(!program.mu.empty())
        {
            PxArray<double> demand(program.offset.size());
            program.jacobian(step.begin(),demand.begin());
            for(PxU32 i=0;i<demand.size();++i)demand[i]+=program.offset[i];
            if(!contacts(demand,force,result)){result.converged=false;return false;}
            program.transpose(force.begin(),rhs.begin());
            for(PxU32 i=0;i<rhs.size();++i)rhs[i]+=program.base[i];
            if(!material(rhs.begin(),step.begin())){result.converged=false;return false;}
        }
        else {result.converged=true;result.residual=0;}
        return true;
    }
};
} }
#endif
