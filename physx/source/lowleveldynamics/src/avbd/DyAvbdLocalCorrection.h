// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_LOCAL_CORRECTION_H
#define DY_AVBD_LOCAL_CORRECTION_H
#include "DyAvbdCommonProgram.h"
#include "DyAvbdCoulombNcp.h"
namespace physx { namespace Dy {
// Mixed numerical Newton operator on the existing local program. Unknowns
// are dx and df/rho. No inverse response columns or physical state are stored.
struct AvbdLocalCorrection
{
    const AvbdCommonProgram& p;const PxArray<double>& rho;
    void* context;void (*inverse)(void*,const double*,double*);
    PxArray<PxU32> rows;PxArray<double> A,B,fullForce,primal,scaled,coordinates;
    PxU32 n,size;
    PxArray<PxU32> tangentGroup;
    AvbdLocalCorrection(const AvbdCommonProgram& program,const PxArray<double>& penalty,const PxArray<PxU8>& enabled,
        void* ctx,void (*inv)(void*,const double*,double*)) : p(program),rho(penalty),context(ctx),inverse(inv),n(p.base.size())
    {
        for(PxU32 r=0;r<enabled.size();++r)if(enabled[r])rows.pushBack(r);
        size=n+3*rows.size();A.resize(9*rows.size());B.resize(A.size());
        fullForce.resize(3*rows.size());primal.resize(n);scaled.resize(n);coordinates.resize(fullForce.size());
        tangentGroup.resize(rows.size());
        for(PxU32 s=0;s<rows.size();++s)
        {tangentGroup[s]=s;for(PxU32 t=0;t<s;++t)if(sameTangent(rows[s],rows[t])){tangentGroup[s]=tangentGroup[t];break;}}
    }
    bool sameTangent(PxU32 a,PxU32 b) const {return p.sameTangent(a,b);}
    void canonicalize(double* force) const
    {
        for(PxU32 s=0;s<rows.size();++s)if(tangentGroup[s]==s)
        {
            double normal=0,t0=0,t1=0;PxU32 count=0;
            for(PxU32 t=s;t<rows.size();++t)if(tangentGroup[t]==s)
            {const PxU32 k=3*rows[t];normal+=force[k];t0+=force[k+1];t1+=force[k+2];++count;}
            if(count<2 || !(normal>0))continue;
            for(PxU32 t=s;t<rows.size();++t)if(tangentGroup[t]==s)
            {const PxU32 k=3*rows[t];const double weight=force[k]/normal;force[k+1]=weight*t0;force[k+2]=weight*t1;}
        }
    }
    void pack(double* force) const
    {
        for(PxU32 s=0;s<rows.size();++s)if(tangentGroup[s]!=s)
        {const PxU32 k=3*rows[s],owner=3*rows[tangentGroup[s]];
            for(PxU32 q=1;q<3;++q){force[owner+q]+=force[k+q];force[k+q]=0;}}
    }
    void unpack(double* force) const
    {
        for(PxU32 s=0;s<rows.size();++s)if(tangentGroup[s]==s)
        {
            const PxU32 owner=3*rows[s];double normal=0;PxU32 count=0;
            for(PxU32 t=s;t<rows.size();++t)if(tangentGroup[t]==s){normal+=force[3*rows[t]];++count;}
            if(count<2)continue;const double t0=force[owner+1],t1=force[owner+2];
            for(PxU32 t=s;t<rows.size();++t)if(tangentGroup[t]==s)
            {const PxU32 k=3*rows[t];const double weight=normal>0?force[k]/normal:0.;force[k+1]=weight*t0;force[k+2]=weight*t1;}
        }
    }
    void project(double* force) const
    {
        for(PxU32 s=0;s<rows.size();++s)for(PxU32 q=0;q<3;++q)
        {const PxU32 k=3*rows[s]+q;force[k]=q && tangentGroup[s]!=s?0.:std::max(p.lower[k],std::min(p.upper[k],force[k]));}
        for(PxU32 s=0;s<rows.size();++s)if(tangentGroup[s]==s && p.coulomb[rows[s]])
        {
            const PxU32 k=3*rows[s];double normal=0;
            for(PxU32 t=s;t<rows.size();++t)if(tangentGroup[t]==s)normal+=force[3*rows[t]];
            const double bound=p.mu[rows[s]]*std::max(0.,normal),mag=std::hypot(force[k+1],force[k+2]);
            if(mag>bound)for(PxU32 q=1;q<3;++q)force[k+q]*=bound/mag;
        }
    }
    void jproduct(const double* x,double* y) const
    {
        for(PxU32 s=0;s<rows.size();++s)
        {
            for(PxU32 q=0;q<3;++q)y[3*s+q]=0;
            for(PxU32 e=p.jOffsets[rows[s]];e<p.jOffsets[rows[s]+1];++e)
            {const auto& j=p.J[e];for(PxU32 q=0;q<3;++q)if(!q || tangentGroup[s]==s)
                for(PxU32 d=0;d<p.dimensions[j.column];++d)y[3*s+q]+=j.values[6*q+d]*x[p.starts[j.column]+d];}
        }
    }
    void jtranspose(const double* x,double* y) const
    {
        std::fill(y,y+n,0.);
        for(PxU32 s=0;s<rows.size();++s)for(PxU32 e=p.jOffsets[rows[s]];e<p.jOffsets[rows[s]+1];++e)
        {const auto& j=p.J[e];for(PxU32 q=0;q<3;++q)if(!q || tangentGroup[s]==s)
            for(PxU32 d=0;d<p.dimensions[j.column];++d)y[p.starts[j.column]+d]+=j.values[6*q+d]*x[3*s+q];}
    }
    void evaluate(const double* x,const double* f,double* out,bool derivatives)
    {
        for(PxU32 s=0;s<rows.size();++s)for(PxU32 q=0;q<3;++q)fullForce[3*s+q]=f[3*rows[s]+q];
        p.hessian(x,primal.begin());jtranspose(fullForce.begin(),scaled.begin());
        for(PxU32 d=0;d<n;++d)primal[d]-=p.base[d]+scaled[d];inverse(context,primal.begin(),out);
        jproduct(x,coordinates.begin());
        for(PxU32 s=0;s<rows.size();++s)
        {
            const PxU32 r=rows[s],k=3*r;double u[3],root[3],aggregate[3]={f[k],f[k+1],f[k+2]};
            if(tangentGroup[s]==s)for(PxU32 t=s+1;t<rows.size();++t)if(tangentGroup[t]==s)aggregate[0]+=f[3*rows[t]];
            for(PxU32 q=0;q<3;++q)u[q]=p.offset[k+q]+coordinates[3*s+q]+p.compliance[k+q]*f[k+q];
            avbdCoulombRoot(aggregate,u,rho[k],p.mu[r],p.lower.begin()+k,p.upper.begin()+k,p.coulomb[r]!=0,root);
            const double v=f[k]-rho[k]*u[0];root[0]=f[k]-std::max(p.lower[k],std::min(p.upper[k],v));
            if(tangentGroup[s]!=s){root[1]=f[k+1];root[2]=f[k+2];}
            for(PxU32 q=0;q<3;++q)out[n+3*s+q]=root[q]/rho[k+q];
            if(derivatives)
            {
                double* a=A.begin()+9*s;double* b=B.begin()+9*s;
                AvbdCoulombNcp::rootDerivative(aggregate,u,rho[k],p.mu[r],p.lower.begin()+k,p.upper.begin()+k,p.coulomb[r]!=0,a,b);
                const bool free=v>p.lower[k] && v<p.upper[k];a[0]=free?rho[k]:0.;b[0]=free?0.:1.;
                if(tangentGroup[s]!=s){std::fill(a+3,a+9,0.);std::fill(b+3,b+9,0.);b[4]=b[8]=1.;}
            }
        }
    }
    void apply(const double* v,double* out)
    {
        std::fill(fullForce.begin(),fullForce.end(),0.);
        for(PxU32 s=0;s<rows.size();++s)for(PxU32 q=0;q<3;++q)fullForce[3*s+q]=rho[3*rows[s]+q]*v[n+3*s+q];
        p.hessian(v,primal.begin());jtranspose(fullForce.begin(),scaled.begin());
        for(PxU32 d=0;d<n;++d)primal[d]-=scaled[d];inverse(context,primal.begin(),out);
        jproduct(v,coordinates.begin());
        for(PxU32 s=0;s<rows.size();++s)for(PxU32 q=0;q<3;++q)
        {
            const PxU32 k=3*rows[s];double value=0;
            for(PxU32 j=0;j<3;++j)value+=A[9*s+3*q+j]*(coordinates[3*s+j]+p.compliance[k+j]*fullForce[3*s+j])+B[9*s+3*q+j]*fullForce[3*s+j];
            if(q && tangentGroup[s]==s)for(PxU32 t=s+1;t<rows.size();++t)if(tangentGroup[t]==s)value+=B[9*s+3*q]*fullForce[3*t];
            out[n+3*s+q]=value/rho[k+q];
        }
    }
    void transpose(const double* v,double* out)
    {
        inverse(context,v,scaled.begin());p.hessian(scaled.begin(),out);jproduct(scaled.begin(),coordinates.begin());
        std::fill(fullForce.begin(),fullForce.end(),0.);
        for(PxU32 s=0;s<rows.size();++s)for(PxU32 q=0;q<3;++q)
        {
            const PxU32 k=3*rows[s];double a=0,b=0;
            for(PxU32 j=0;j<3;++j){const double w=v[n+3*s+j]/rho[k+j];a+=A[9*s+3*j+q]*w;b+=B[9*s+3*j+q]*w;}
            fullForce[3*s+q]=a;out[n+3*s+q]=rho[k+q]*(-coordinates[3*s+q]+b+p.compliance[k+q]*a);
        }
        for(PxU32 s=0;s<rows.size();++s)if(tangentGroup[s]!=s)
        {const PxU32 owner=tangentGroup[s],k=3*rows[owner];double b=0;
            for(PxU32 q=1;q<3;++q)b+=B[9*owner+3*q]*v[n+3*owner+q]/rho[k+q];out[n+3*s]+=rho[3*rows[s]]*b;}
        jtranspose(fullForce.begin(),primal.begin());for(PxU32 d=0;d<n;++d)out[d]+=primal[d];
    }
    bool correct(PxArray<double>& x,PxArray<double>& force,PxU32& products)
    {
        if(rows.empty())return false;
        canonicalize(force.begin());PxArray<double> f=force;pack(f.begin());
        PxArray<double> residual(size),rhs(size),direction(size),linear(size),work(6*size),trialX(n),trialF(f.size()),trial(size);
        evaluate(x.begin(),f.begin(),residual.begin(),true);const double before=AvbdCoulombNcp::dot(residual.begin(),residual.begin(),int(size));
        if(!(before>0) || !std::isfinite(before))return false;
        for(PxU32 i=0;i<size;++i)rhs[i]=-residual[i];
        products+=PxU32(AvbdCoulombNcp::lsqr(*this,int(size),rhs.begin(),direction.begin(),work.begin(),int(4*size),.01));
        apply(direction.begin(),linear.begin());const double slope=2*AvbdCoulombNcp::dot(residual.begin(),linear.begin(),int(size));
        if(!(slope<0))return false;
        // Visit the first finite force boundary exactly. Clipping a full
        // Newton trial past it destroys the coupled primal/dual direction.
        double first=1,boundaryValue=0;PxU32 boundary=PX_MAX_U32;
        auto event=[&](double fraction,PxU32 coordinate,double value)
        {if(fraction<=first){first=fraction;boundary=coordinate;boundaryValue=value;}};
        for(PxU32 s=0;s<rows.size();++s)for(PxU32 q=0;q<3;++q)
        {
            const PxU32 r=rows[s],k=3*r+q;if(q && p.coulomb[r])continue;
            const double delta=rho[k]*direction[n+3*s+q];
            if(delta<0 && f[k]>p.lower[k])event((p.lower[k]-f[k])/delta,k,p.lower[k]);
            if(delta>0 && f[k]<p.upper[k])event((p.upper[k]-f[k])/delta,k,p.upper[k]);
        }
        for(PxU32 search=0;search<16;++search)
        {
            const double fraction=std::ldexp(first,-int(search));trialF=f;
            for(PxU32 d=0;d<n;++d)trialX[d]=x[d]+fraction*direction[d];
            for(PxU32 s=0;s<rows.size();++s)
            {
                const PxU32 r=rows[s],k=3*r;
                for(PxU32 q=0;q<3;++q)trialF[k+q]=f[k+q]+fraction*rho[k+q]*direction[n+3*s+q];
            }
            if(search==0 && boundary!=PX_MAX_U32)trialF[boundary]=boundaryValue;
            project(trialF.begin());
            evaluate(trialX.begin(),trialF.begin(),trial.begin(),false);
            const double after=AvbdCoulombNcp::dot(trial.begin(),trial.begin(),int(size));
            if(after<=before+1.e-4*fraction*slope){x.swap(trialX);unpack(trialF.begin());force.swap(trialF);return true;}
        }
        return false;
    }
};
} }
#endif
