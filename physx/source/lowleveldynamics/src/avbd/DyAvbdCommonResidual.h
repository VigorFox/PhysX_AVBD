// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_COMMON_RESIDUAL_H
#define DY_AVBD_COMMON_RESIDUAL_H
#include "DyAvbdCommonProgram.h"
#include "DyAvbdCoulombNcp.h"
namespace physx { namespace Dy {
// Fixed numerical units for a line search. Diagonal response scales the merit
// only; the trial direction still solves the COMPLETE coupled H/J equations.
struct AvbdCommonResidual
{
    // Hd=-g+J^T f makes d a descent direction of the material energy minus
    // work of this FIXED trial load. This is a line-search model, not a
    // Coulomb potential. The original NCP still qualifies convergence.
    static bool workDescent(const AvbdCommonProgram& from,const AvbdCommonProgram& to,
        const double* force,const double* delta,double& work,double& slope)
    {
        if(from.base.size()!=to.base.size() || from.offset.size()!=to.offset.size() ||
            from.materialEnergies.size()!=to.materialEnergies.size() || from.energyPairs.size()!=to.energyPairs.size())return false;
        PxArray<double> reaction(from.base.size());from.transpose(force,reaction.begin());
        // Difference the identical local potentials before summing: subtracting
        // two island totals erases the work of weak modes near convergence.
        work=0;slope=0;
        for(PxU32 i=0;i<from.materialEnergies.size();++i)work+=to.materialEnergies[i]-from.materialEnergies[i];
        for(PxU32 i=0;i<from.energyPairs.size();++i)
        {
            const auto& a=from.energyPairs[i];const auto& b=to.energyPairs[i];
            // Subtract before multiplication; a large unchanged strain must
            // not erase the work of a weak mode inside the same material.
            // Conjugates use the same symmetric, step-frozen coefficient
            // operator at both endpoints (mass/inertia/stiffness/damping).
            volatile double dx=b.coordinate-a.coordinate;
            work+=.5*dx*(b.conjugate+a.conjugate);
        }
        for(PxU32 d=0;d<reaction.size();++d)slope-=(from.base[d]+reaction[d])*delta[d];
        for(PxU32 r=0;r<from.offset.size();++r)work-=force[r]*(to.offset[r]-from.offset[r]);
        return std::isfinite(work) && slope<0 && work<=1.e-4*slope;
    }
    PxArray<double> inverse;
    mutable PxArray<double> compliance;
    const AvbdCommonProgram* frozen; // Borrowed; the publication owns its lifetime.
    explicit AvbdCommonResidual(const AvbdCommonProgram& p):frozen(&p)
    {
        inverse.resize(p.base.size(),0);compliance.resize(p.mu.size(),-1);
        for(const auto& h:p.metricH.empty()?p.H:p.metricH)if(h.row==h.column)
            for(PxU32 d=0;d<p.dimensions[h.row];++d)inverse[p.starts[h.row]+d]=1/h.values[6*d+d];
    }
    // J, the metric and physical compliance are frozen for this line search.
    // Offset/SOC/candidate changes never alter these units. Each row owns its
    // cache slot; concurrent row ranges must be disjoint as with Table output.
    double rowCompliance(PxU32 r) const
    {
        if(compliance[r]<0)
        {
            const auto& p=*frozen;double value=0;
            for(PxU32 e=p.jOffsets[r];e<p.jOffsets[r+1];++e)
            {const auto& j=p.J[e];for(PxU32 q=0;q<3;++q)for(PxU32 d=0;d<p.dimensions[j.column];++d)
                value+=j.values[6*q+d]*j.values[6*q+d]*inverse[p.starts[j.column]+d];}
            for(PxU32 q=0;q<3;++q)value+=p.compliance[3*r+q];
            compliance[r]=value;
        }
        return compliance[r]>0?compliance[r]:1.;
    }
    double evaluate(const AvbdCommonProgram& p,const double* f) const
    {
        PxArray<double> station(p.base.size());p.transpose(f,station.begin());double result=0;
        for(PxU32 d=0;d<station.size();++d)
        {const double v=station[d]+p.base[d];result+=v*v*inverse[d];}
        for(PxU32 r=0;r<p.mu.size();++r)
        {
            if(f[3*r]==0 && f[3*r+1]==0 && f[3*r+2]==0 && p.zeroReactionAdmissibleCoordinates(r,p.offset.begin()+3*r))continue;
            double u[3],root[3];const double w=rowCompliance(r);
            for(PxU32 q=0;q<3;++q)u[q]=p.offset[3*r+q]+p.compliance[3*r+q]*f[3*r+q];
            avbdCoulombRoot(f+3*r,u,1/w,p.mu[r],p.lower.begin()+3*r,p.upper.begin()+3*r,p.coulomb[r]!=0,root);
            for(PxU32 q=0;q<3;++q)result+=root[q]*root[q]*w;
        }
        return std::isfinite(result)?result:std::numeric_limits<double>::infinity();
    }
    bool smallRoot(const AvbdCommonProgram& p,const double* f,const double* motionTolerance,const double* displacement=NULL) const
    {
        for(PxU32 r=0;r<p.mu.size();++r)
        {
            double u[3],root[3],tolerance[3]={};
            if(displacement)p.rowCoordinates(r,displacement,u);
            else for(PxU32 q=0;q<3;++q)u[q]=p.offset[3*r+q];
            if(f[3*r]==0 && f[3*r+1]==0 && f[3*r+2]==0 && p.zeroReactionAdmissibleCoordinates(r,u))continue;
            const double w=rowCompliance(r);
            for(PxU32 q=0;q<3;++q)u[q]+=p.compliance[3*r+q]*f[3*r+q];
            for(PxU32 e=p.jOffsets[r];e<p.jOffsets[r+1];++e)
            {const auto& j=p.J[e];for(PxU32 q=0;q<3;++q)for(PxU32 d=0;d<p.dimensions[j.column];++d)
                tolerance[q]+=std::abs(j.values[6*q+d])*motionTolerance[p.starts[j.column]+d];}
            avbdCoulombRoot(f+3*r,u,1/w,p.mu[r],p.lower.begin()+3*r,p.upper.begin()+3*r,p.coulomb[r]!=0,root);
            for(PxU32 q=0;q<3;++q)if(!(std::abs(root[q])*w<=tolerance[q]))return false;
        }
        return true;
    }
};
} }
#endif
