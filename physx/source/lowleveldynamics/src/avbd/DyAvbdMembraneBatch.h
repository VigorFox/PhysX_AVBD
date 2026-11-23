// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_MEMBRANE_BATCH_H
#define DY_AVBD_MEMBRANE_BATCH_H
#include "avbd/DyAvbdNodalShell.h"
#include "avbd/DyAvbdFourLane.h"
namespace physx { namespace Dy {
struct AvbdMembraneContribution
{
    // One destination block. The full polynomial curvature and the positive
    // solve metric have distinct diagonal geometric contributions.
    AvbdNodalVec3 gradient,cubic;
    AvbdNodalMat33 metric,polynomialMetric;
    AvbdNodalReal energy,quartic;
    bool valid;
    void clear()
    {
        gradient=cubic=AvbdNodalVec3(0);metric=polynomialMetric=AvbdNodalMat33(PxZero);
        energy=quartic=0;valid=true;
    }
};
struct AvbdMembraneBatchArgs
{
    enum { Width=4 };
    const AvbdNodalShellRest* rest[Width];
    AvbdNodalVec3 positions[Width][3];
    AvbdNodalReal mu[Width],lambda[Width];
    PxU32 corner[Width],count;
    AvbdMembraneContribution output;
};
template<class T> struct AvbdMembraneBatchKernel
{
    static void sample(AvbdMembraneBatchArgs& a)
    {
        auto& o=a.output;o.clear();
        for(PxU32 i=0;i<a.count;++i)
        {
            if(a.corner[i]>=3){o.valid=false;return;}
            AvbdNodalWorkT<AvbdNodalReal> work;
            const auto row=avbdSampleNodalShell(*a.rest[i],a.positions[i],a.mu[i],a.lambda[i],
                AvbdNodalReal(0),a.corner[i],static_cast<const AvbdNodalReal*>(NULL),&work);
            o.valid=o.valid && row.valid;o.energy+=row.energy;o.gradient+=row.gradient;
            o.metric+=row.metric+AvbdNodalMat33(PxIdentity)*PxMax(AvbdNodalReal(0),work.geometric);
            o.polynomialMetric+=row.metric+AvbdNodalMat33(PxIdentity)*work.geometric;
            o.cubic+=work.cubic;o.quartic+=work.quartic;
        }
    }
};
template<> struct AvbdMembraneBatchKernel<float> : AvbdFourLane
{
    static void sample(AvbdMembraneBatchArgs& a)
    {
        using namespace aos;
        if(!a.count){a.output.clear();return;}
        a.output.valid=true;
        for(PxU32 i=0;i<a.count;++i)
        {
            if(a.corner[i]>=3 || !(a.rest[i]->area>0 && a.mu[i]>0 && a.mu[i]+a.lambda[i]>0))
            {a.output.valid=false;return;}
            for(PxU32 v=0;v<3;++v)if(!a.positions[i][v].isFinite()){a.output.valid=false;return;}
        }
        const PxU32 id[4]={0,PxMin(1u,a.count-1),PxMin(2u,a.count-1),PxMin(3u,a.count-1)};
        Vector x[3];
        for(PxU32 v=0;v<3;++v)
            x[v]={V4LoadXYZW(float(a.positions[id[0]][v].x),float(a.positions[id[1]][v].x),float(a.positions[id[2]][v].x),float(a.positions[id[3]][v].x)),
                  V4LoadXYZW(float(a.positions[id[0]][v].y),float(a.positions[id[1]][v].y),float(a.positions[id[2]][v].y),float(a.positions[id[3]][v].y)),
                  V4LoadXYZW(float(a.positions[id[0]][v].z),float(a.positions[id[1]][v].z),float(a.positions[id[2]][v].z),float(a.positions[id[3]][v].z))};
        float coefficients[8][4];
        for(PxU32 lane=0;lane<4;++lane)
        {
            const PxU32 i=id[lane];const auto& r=*a.rest[i];
            coefficients[0][lane]=float(r.shape[1].x);coefficients[1][lane]=float(r.shape[2].x);
            coefficients[2][lane]=float(r.shape[1].y);coefficients[3][lane]=float(r.shape[2].y);
            coefficients[4][lane]=float(a.mu[i]*r.area);coefficients[5][lane]=float(a.lambda[i]*r.area);
            const auto shape=a.corner[i]<3?r.shape[a.corner[i]]:AvbdNodalVec3(0);
            coefficients[6][lane]=float(shape.x);coefficients[7][lane]=float(shape.y);
        }
        Lane c[8];for(PxU32 i=0;i<8;++i)c[i]=V4LoadU(coefficients[i]);
        const Vector e=sub(x[1],x[0]),f=sub(x[2],x[0]);
        const Vector F0=add(mul(e,c[0]),mul(f,c[1])),F1=add(mul(e,c[2]),mul(f,c[3]));
        const Lane one=V4One(),half=V4Splat(FLoad(.5f)),quarter=V4Splat(FLoad(.25f)),eighth=V4Splat(FLoad(.125f));
        const Lane u=c[4],l=c[5],s=c[6],t=c[7];
        const Lane aa=V4Sub(dot(F0,F0),one),bb=V4Sub(dot(F1,F1),one),ab=dot(F0,F1),trace=V4Add(aa,bb);
        const Vector area=cross(F0,F1);
        const Lane ua=V4Add(V4Mul(u,aa),V4Mul(V4Mul(half,l),trace));
        const Lane ub=V4Add(V4Mul(u,bb),V4Mul(V4Mul(half,l),trace)),uc=V4Mul(u,ab);
        const Vector gradient=add(mul(F0,V4Add(V4Mul(s,ua),V4Mul(t,uc))),
                                  mul(F1,V4Add(V4Mul(s,uc),V4Mul(t,ub))));
        const Vector w=add(mul(F0,s),mul(F1,t));
        const Lane ss=V4Add(V4Mul(s,s),V4Mul(t,t)),d=V4Mul(u,ss),q=V4Add(u,l);
        const auto metric=[&](Lane ax,Lane ay,Lane bx,Lane by,Lane wx,Lane wy)
        {return V4Add(V4Mul(V4Add(V4Mul(ax,ay),V4Mul(bx,by)),d),V4Mul(V4Mul(wx,wy),q));};
        const Lane energy=V4Add(V4Mul(V4Mul(quarter,u),V4Add(V4Add(V4Mul(aa,aa),V4Mul(bb,bb)),V4Add(V4Mul(ab,ab),V4Mul(ab,ab)))),
            V4Mul(V4Mul(eighth,l),V4Mul(trace,trace)));
        const Vector cubic=mul(w,V4Mul(V4Add(u,V4Mul(half,l)),ss));
        auto& o=a.output;
        const Lane mask=V4LoadXYZW(1.f,a.count>1?1.f:0.f,a.count>2?1.f:0.f,a.count>3?1.f:0.f);
        const Lane maximum=V4Splat(FLoad(PX_MAX_F32));
        const auto sum=[&](Lane value)
        {
            o.valid=o.valid && V4AllGrtrOrEq(maximum,V4Abs(value));
            float valueSum;FStore(V4Dot(value,mask),&valueSum);o.valid=o.valid && PxIsFinite(valueSum);return valueSum;
        };
        o.valid=V4AllGrtr(dot(area,area),V4Zero())!=0;
        o.energy=sum(energy);o.gradient=AvbdNodalVec3(sum(gradient.x),sum(gradient.y),sum(gradient.z));
        const float xx=sum(metric(F0.x,F0.x,F1.x,F1.x,w.x,w.x)),yy=sum(metric(F0.y,F0.y,F1.y,F1.y,w.y,w.y)),
            zz=sum(metric(F0.z,F0.z,F1.z,F1.z,w.z,w.z)),xy=sum(metric(F0.x,F0.y,F1.x,F1.y,w.x,w.y)),
            xz=sum(metric(F0.x,F0.z,F1.x,F1.z,w.x,w.z)),yz=sum(metric(F0.y,F0.z,F1.y,F1.z,w.y,w.z));
        const Lane geometric=V4Add(V4Add(V4Mul(ua,V4Mul(s,s)),V4Mul(ub,V4Mul(t,t))),V4Mul(V4Add(uc,uc),V4Mul(s,t)));
        const float full=sum(geometric),positive=sum(V4Max(geometric,V4Zero()));
        // Clamp each row before reduction, never the summed geometric term.
        o.metric=AvbdNodalMat33(AvbdNodalVec3(xx+positive,xy,xz),AvbdNodalVec3(xy,yy+positive,yz),AvbdNodalVec3(xz,yz,zz+positive));
        o.polynomialMetric=AvbdNodalMat33(AvbdNodalVec3(xx+full,xy,xz),AvbdNodalVec3(xy,yy+full,yz),AvbdNodalVec3(xz,yz,zz+full));
        o.cubic=AvbdNodalVec3(sum(cubic.x),sum(cubic.y),sum(cubic.z));
        o.quartic=sum(V4Mul(V4Add(V4Mul(quarter,u),V4Mul(eighth,l)),V4Mul(ss,ss)));
    }
};
} }
#endif
