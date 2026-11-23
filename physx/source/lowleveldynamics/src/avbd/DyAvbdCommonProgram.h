// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_COMMON_PROGRAM_H
#define DY_AVBD_COMMON_PROGRAM_H
#include "foundation/PxArray.h"
#include "DyAvbdCoulombNcp.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace physx { namespace Dy {
// Fixed physical block factorization, shared by numerical executors.
inline bool avbdSolveSmallBlock(const double* matrix,const double* rhs,double* x,PxU32 n)
    {
        double L[36]={},D[6]={},v[6]={};
        for(PxU32 i=0;i<n;++i)
        {
            for(PxU32 j=0;j<i;++j)
            {double a=matrix[6*i+j];for(PxU32 k=0;k<j;++k)a-=L[6*i+k]*D[k]*L[6*j+k];L[6*i+j]=a/D[j];}
            double a=matrix[6*i+i];for(PxU32 k=0;k<i;++k)a-=L[6*i+k]*L[6*i+k]*D[k];
            if(!(a>0) || !std::isfinite(a))return false;D[i]=a;
            v[i]=rhs[i];for(PxU32 j=0;j<i;++j)v[i]-=L[6*i+j]*v[j];
        }
        for(PxU32 ii=n;ii>0;--ii)
        {const PxU32 i=ii-1;x[i]=v[i]/D[i];for(PxU32 j=i+1;j<n;++j)x[i]-=L[6*j+i]*x[j];if(!std::isfinite(x[i]))return false;}
        return true;
    }

// Frozen numerical view of the EXISTING row program. No poses, physical
// multipliers, material history, solver policy or ownership are stored here.
// Storage and products are sparse; physical blocks retain dimension 3 or 6.
struct AvbdCommonHBlock {PxU32 row,column;double values[36];};
struct AvbdCommonJBlock {PxU32 row,column;double values[18];};
struct AvbdCommonProgram
{
    PxArray<PxU32> dimensions,starts,sourceBlocks,sourceRows;
    PxArray<AvbdCommonHBlock> H;
    PxArray<AvbdCommonJBlock> J;
    PxArray<double> base,offset,lower,upper,compliance,mu;
    PxArray<int> coulomb;
    // Optional PSD metric, aligned with H after finish(). H remains the
    // actual linearized operator; the metric never changes its stationary root.
    PxArray<AvbdCommonHBlock> metricH;
    const AvbdCommonHBlock& metricBlock(PxU32 e) const {return metricH.empty()?H[e]:metricH[e];}
    PxArray<double> materialEnergies;
    // Quadratic local potentials retain their coordinates until subtraction.
    // Conjugates are the step-frozen symmetric affine derivatives of these
    // coordinates; this scratch is neither a second material nor history.
    struct EnergyPair {double coordinate,conjugate;};
    PxArray<EnergyPair> energyPairs;
    PxArray<PxU32> hOffsets,jOffsets,transposeOffsets,transposeEntries;
    bool valid=false;

    void rowPenalty(PxU32 r,const double* inverse,double* rho) const
    {
        double values[3]={};
        for(PxU32 e=jOffsets[r];e<jOffsets[r+1];++e)
        {const auto& j=J[e];const PxU32 n=dimensions[j.column];
            for(PxU32 q=0;q<3;++q)for(PxU32 d=0;d<n;++d)for(PxU32 k=0;k<n;++k)
                values[q]+=j.values[6*q+d]*inverse[36*j.column+6*d+k]*j.values[6*q+k];}
        if(coulomb[r])values[1]=values[2]=std::max(values[1],values[2]);
        for(PxU32 q=0;q<3;++q)rho[q]=values[q]>0?2/values[q]:1;
    }

    bool sameTangent(PxU32 a,PxU32 b) const
    {
        const auto& p=*this;
        if(!p.coulomb[a] || !p.coulomb[b] || p.mu[a]!=p.mu[b] || p.lower[3*a]<0 || p.lower[3*b]<0)return false;
        for(PxU32 q=1;q<3;++q)
            if(p.compliance[3*a+q]!=0 || p.compliance[3*b+q]!=0 || p.offset[3*a+q]!=p.offset[3*b+q] ||
                p.lower[3*a+q]!=-std::numeric_limits<double>::infinity() || p.lower[3*b+q]!=p.lower[3*a+q] ||
                p.upper[3*a+q]!=std::numeric_limits<double>::infinity() || p.upper[3*b+q]!=p.upper[3*a+q])return false;
        if(p.jOffsets[a+1]-p.jOffsets[a]!=p.jOffsets[b+1]-p.jOffsets[b])return false;
        for(PxU32 i=0;i<p.jOffsets[a+1]-p.jOffsets[a];++i)
        {const auto& x=p.J[p.jOffsets[a]+i];const auto& y=p.J[p.jOffsets[b]+i];if(x.column!=y.column)return false;
            for(PxU32 q=1;q<3;++q)for(PxU32 d=0;d<p.dimensions[x.column];++d)if(x.values[6*q+d]!=y.values[6*q+d])return false;}
        return true;
    }
    void projected(PxU32 r,const double* penalty,const double* step,const double* force,double* f,double* stiffness,bool updateCapacity=false,double* coordinates=NULL,double* residual=NULL) const
    {projectedRow(r,penalty+3*r,step,force+3*r,f,stiffness,updateCapacity,coordinates,residual);}
    void rowCoordinates(PxU32 r,const double* step,double* u) const
    {
        for(PxU32 q=0;q<3;++q)u[q]=offset[3*r+q];
        for(PxU32 e=jOffsets[r];e<jOffsets[r+1];++e)
        {const auto& j=J[e];for(PxU32 q=0;q<3;++q)for(PxU32 d=0;d<dimensions[j.column];++d)
            u[q]+=j.values[6*q+d]*step[starts[j.column]+d];}
    }
    bool zeroReactionAdmissibleCoordinates(PxU32 r,const double* u) const
    {
        if(lower[3*r]!=0)return false;
        if(!coulomb[r])for(PxU32 q=1;q<3;++q)if(lower[3*r+q]!=0 || upper[3*r+q]!=0)return false;
        // Nonnegative normal gap gives zero normal projection and therefore
        // a zero Coulomb disk. No penalty or tangential inverse is needed.
        return std::isfinite(u[0]) && std::isfinite(u[1]) && std::isfinite(u[2]) && u[0]>=0;
    }
    bool zeroReactionAdmissible(PxU32 r,const double* step) const
    {double u[3];rowCoordinates(r,step,u);return zeroReactionAdmissibleCoordinates(r,u);}
    // A row-local force view also serves complete-domain queries of omitted
    // numerical rows. The projection law itself has exactly one implementation.
    void projectedRow(PxU32 r,const double* penalty,const double* step,const double* force,double* f,double* stiffness,bool updateCapacity=false,double* coordinates=NULL,double* residual=NULL) const
    {
        const auto& p=*this;double u[3];rowCoordinates(r,step,u);
        if(coordinates)for(PxU32 q=0;q<3;++q)coordinates[q]=u[q];
        for(PxU32 q=0;q<3;++q)
        {
            const PxU32 i=3*r+q;const double rho=penalty[q],denom=1+rho*p.compliance[i];
            const double raw=(force[q]-rho*u[q])/denom;
            f[q]=std::max(p.lower[i],std::min(p.upper[i],raw));
            stiffness[q]=p.lower[i]<p.upper[i]?rho/denom:0.;
            if(residual)residual[q]=avbdScalarProjectionRoot(force[q],(rho/denom)*(u[q]+p.compliance[i]*force[q]),p.lower[i],p.upper[i]);
        }
        if(p.coulomb[r])
        {
            // A primal sweep freezes the current normal capacity. Each local
            // contact potential is then convex and J^T rho J majorizes its
            // gradient. The shared dual refreshes the actual normal capacity;
            // fixed points still obey the original non-associated Coulomb law.
            const double bound=p.mu[r]*std::max(0.,updateCapacity?f[0]:force[0]),mag=std::hypot(f[1],f[2]);
            if(mag>bound)
            {const double scale=bound/mag;for(PxU32 q=1;q<3;++q)
                {if(residual)residual[q]=((mag-bound)/mag)*force[q]+scale*residual[q];f[q]*=scale;}}
            if(p.mu[r]==0)stiffness[1]=stiffness[2]=0;
        }
    }
    void write(FILE* file,const PxArray<double>* initialForce=NULL,const PxArray<double>* motionTolerance=NULL) const
    {
        std::fprintf(file,"{\"complete\":%s,\"reason\":\"local_response_capture\",",valid?"true":"false");
        auto array=[&](const char* name,const auto& values)
        {std::fprintf(file,"\"%s\":[",name);for(PxU32 i=0;i<values.size();++i)
            {const double v=double(values[i]);if(i)std::fprintf(file,",");if(std::isfinite(v))std::fprintf(file,"%.17g",v);
                else std::fprintf(file,"\"%s\"",v<0?"-inf":"inf");}std::fprintf(file,"],");};
        array("dimensions",dimensions);array("sourceBlocks",sourceBlocks);array("sourceRows",sourceRows);
        array("base",base);array("offset",offset);array("lower",lower);array("upper",upper);array("compliance",compliance);
        array("mu",mu);array("coulomb",coulomb);
        if(initialForce)array("initialForce",*initialForce);
        if(motionTolerance)array("motionTolerance",*motionTolerance);
        auto matrix=[&](const char* name,const auto& entries,PxU32 width)
        {std::fprintf(file,"\"%s\":[",name);for(PxU32 e=0;e<entries.size();++e)
            {std::fprintf(file,"%s[%u,%u",e?",":"",entries[e].row,entries[e].column);
                for(PxU32 k=0;k<width;++k)std::fprintf(file,",%.17g",entries[e].values[k]);std::fprintf(file,"]");}std::fprintf(file,"]");};
        matrix("H",H,36);std::fprintf(file,",");
        if(!metricH.empty()){matrix("metricH",metricH,36);std::fprintf(file,",");}
        matrix("J",J,18);std::fprintf(file,"}\n");
    }

    template<class Entry> static void merge(PxArray<Entry>& entries,PxU32 width)
    {
        if(entries.empty())return;
        const auto order=[](const Entry& a,const Entry& b)
        {return a.row<b.row || (a.row==b.row && a.column<b.column);};
        // The row compiler emits J in row order. Sort its tiny support spans
        // instead of globally sorting millions of already grouped entries.
        if(std::is_sorted(entries.begin(),entries.end(),[](const Entry& a,const Entry& b){return a.row<b.row;}))
        {
            for(PxU32 first=0;first<entries.size();)
            {
                PxU32 end=first+1;while(end<entries.size() && entries[end].row==entries[first].row)++end;
                if(!std::is_sorted(entries.begin()+first,entries.begin()+end,order))
                    std::sort(entries.begin()+first,entries.begin()+end,order);
                first=end;
            }
        }
        else std::sort(entries.begin(),entries.end(),order);
        PxU32 size=0;
        for(PxU32 i=0;i<entries.size();++i)
        {
            if(size && entries[size-1].row==entries[i].row && entries[size-1].column==entries[i].column)
                for(PxU32 k=0;k<width;++k)entries[size-1].values[k]+=entries[i].values[k];
            else {if(size!=i)entries[size]=entries[i];++size;}
        }
        entries.resize(size);
    }
    bool finish()
    {
        valid=false;const PxU32 n=dimensions.size(),m=mu.size();
        starts.resize(n+1);starts[0]=0;
        for(PxU32 b=0;b<n;++b)
        {if(dimensions[b]!=3 && dimensions[b]!=6)return false;starts[b+1]=starts[b]+dimensions[b];}
        if(base.size()!=starts[n] || sourceBlocks.size()!=n || sourceRows.size()!=m || coulomb.size()!=m ||
            offset.size()!=3*m || lower.size()!=3*m || upper.size()!=3*m || compliance.size()!=3*m)return false;
        for(PxU32 i=0;i<base.size();++i)if(!std::isfinite(base[i]))return false;
        for(PxU32 i=0;i<m;++i)if(!std::isfinite(mu[i]) || mu[i]<0 || (coulomb[i]!=0 && coulomb[i]!=1))return false;
        for(PxU32 i=0;i<3*m;++i)
            if(!std::isfinite(offset[i]) || !std::isfinite(compliance[i]) || compliance[i]<0 || !(lower[i]<=upper[i]))return false;
        merge(H,36);merge(J,18);merge(metricH,36);
        if(!metricH.empty())
        {
            if(metricH.size()!=H.size())return false;
            for(PxU32 e=0;e<H.size();++e)
            {
                if(metricH[e].row!=H[e].row || metricH[e].column!=H[e].column)return false;
                for(PxU32 k=0;k<36;++k)if(!std::isfinite(metricH[e].values[k]))return false;
            }
        }
        hOffsets.resize(n+1);jOffsets.resize(m+1);transposeOffsets.resize(n+1);
        for(PxU32 b=0;b<=n;++b)hOffsets[b]=transposeOffsets[b]=0;
        for(PxU32 r=0;r<=m;++r)jOffsets[r]=0;
        for(PxU32 e=0;e<H.size();++e)
        {
            if(H[e].row>=n || H[e].column>=n)return false;
            for(PxU32 k=0;k<36;++k)if(!std::isfinite(H[e].values[k]))return false;
            ++hOffsets[H[e].row+1];
        }
        for(PxU32 e=0;e<J.size();++e)
        {
            if(J[e].row>=m || J[e].column>=n)return false;
            for(PxU32 k=0;k<18;++k)if(!std::isfinite(J[e].values[k]))return false;
            ++jOffsets[J[e].row+1];++transposeOffsets[J[e].column+1];
        }
        for(PxU32 b=1;b<=n;++b){hOffsets[b]+=hOffsets[b-1];transposeOffsets[b]+=transposeOffsets[b-1];}
        for(PxU32 r=1;r<=m;++r)jOffsets[r]+=jOffsets[r-1];
        PxArray<PxU32> cursor=transposeOffsets;transposeEntries.resize(J.size());
        for(PxU32 e=0;e<J.size();++e)transposeEntries[cursor[J[e].column]++]=e;
        valid=true;return true;
    }
    // Disjoint destination ranges may execute in parallel without atomics.
    void hessian(const double* x,double* y,PxU32 begin=0,PxU32 end=PX_MAX_U32) const
    {
        end=std::min(end,dimensions.size());
        for(PxU32 b=begin;b<end;++b)for(PxU32 d=0;d<dimensions[b];++d)
        {
            double value=0;
            for(PxU32 e=hOffsets[b];e<hOffsets[b+1];++e)
                for(PxU32 k=0;k<dimensions[H[e].column];++k)value+=H[e].values[6*d+k]*x[starts[H[e].column]+k];
            y[starts[b]+d]=value;
        }
    }
    void jacobian(const double* x,double* y,PxU32 begin=0,PxU32 end=PX_MAX_U32) const
    {
        end=std::min(end,mu.size());
        for(PxU32 r=begin;r<end;++r)for(PxU32 d=0;d<3;++d)
        {
            double value=0;
            for(PxU32 e=jOffsets[r];e<jOffsets[r+1];++e)
                for(PxU32 k=0;k<dimensions[J[e].column];++k)value+=J[e].values[6*d+k]*x[starts[J[e].column]+k];
            y[3*r+d]=value;
        }
    }
    void transpose(const double* f,double* y,PxU32 begin=0,PxU32 end=PX_MAX_U32) const
    {
        end=std::min(end,dimensions.size());
        for(PxU32 b=begin;b<end;++b)for(PxU32 d=0;d<dimensions[b];++d)y[starts[b]+d]=0;
        if(begin==0 && end==dimensions.size())
        {
            // A complete product can traverse sparse forces without touching
            // the potentially much larger zero-force contact Jacobian.
            for(PxU32 r=0;r<mu.size();++r)
            {
                const double* force=f+3*r;if(force[0]==0 && force[1]==0 && force[2]==0)continue;
                for(PxU32 i=jOffsets[r];i<jOffsets[r+1];++i)
                {const auto& e=J[i];for(PxU32 d=0;d<dimensions[e.column];++d)for(PxU32 k=0;k<3;++k)
                    y[starts[e.column]+d]+=e.values[6*k+d]*force[k];}
            }
            return;
        }
        // Partial products preserve disjoint destination ownership, including
        // parallel Table calls. No atomic scatter across physical blocks.
        for(PxU32 b=begin;b<end;++b)for(PxU32 s=transposeOffsets[b];s<transposeOffsets[b+1];++s)
        {
            const auto& e=J[transposeEntries[s]];const double* force=f+3*e.row;
            if(force[0]==0 && force[1]==0 && force[2]==0)continue;
            for(PxU32 d=0;d<dimensions[b];++d)for(PxU32 k=0;k<3;++k)y[starts[b]+d]+=e.values[6*k+d]*force[k];
        }
    }
    // Upper bound for backtracking an inexact direction from a feasible
    // iterate. This is the original affine hard domain, not a gap tolerance.
    // Nonlinear geometry and float storage remain the publisher's decision.
    double linearFeasibleFraction(const double* step) const
    {
        double fraction=1;
        for(PxU32 r=0;r<mu.size();++r)
        {
            const PxU32 k=3*r;
            if(lower[k]!=0 || !std::isinf(upper[k]) || compliance[k]!=0 || offset[k]<0)continue;
            double delta=0,scale=1+std::abs(offset[k]);
            for(PxU32 e=jOffsets[r];e<jOffsets[r+1];++e)
            {const auto& j=J[e];for(PxU32 d=0;d<dimensions[j.column];++d)
                {const double value=j.values[d]*step[starts[j.column]+d];delta+=value;scale+=std::abs(value);}}
            // Roundoff in a tangent direction must not create a zero search
            // radius. Actual geometry is still certified after float storage.
            if(offset[k]+delta>=-32*std::numeric_limits<double>::epsilon()*scale)continue;
            if(delta<0)
            {if(offset[k]==0)return 0;fraction=std::min(fraction,-offset[k]/delta);}
        }
        return fraction;
    }
};
} }
#endif
