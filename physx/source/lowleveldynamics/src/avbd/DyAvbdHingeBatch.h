// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_HINGE_BATCH_H
#define DY_AVBD_HINGE_BATCH_H
#include "avbd/DyAvbdNodalShell.h"
#include "avbd/DyAvbdFourLane.h"

namespace physx { namespace Dy {

// Independent row contributions reduced to one current local equation.
// No state writes, and no assembly result is cached across candidates/colors.
struct AvbdHingeBatchArgs
{
    enum { Width=4 };
    const AvbdNodalShellRest* rest[Width];
    AvbdNodalVec3 positions[Width][4];
    AvbdNodalReal bending[Width];
    PxU32 corner[Width],count;
    AvbdNodalShellSampleT<AvbdNodalReal> output;
};

template<class T> struct AvbdHingeBatchKernel
{
    static void sample(AvbdHingeBatchArgs& a)
    {
        auto& o=a.output;o=AvbdNodalShellSampleT<AvbdNodalReal>();o.valid=true;
        o.areaRatio=std::numeric_limits<AvbdNodalReal>::max();
        for(PxU32 i=0;i<a.count;++i)
        {
            const auto row=avbdSampleNodalShell(*a.rest[i],a.positions[i],
                AvbdNodalReal(0),AvbdNodalReal(0),a.bending[i],a.corner[i]);
            o.valid=o.valid && row.valid;o.energy+=row.energy;o.gradient+=row.gradient;o.metric+=row.metric;
            o.areaRatio=PxMin(o.areaRatio,row.areaRatio);
        }
    }
};

// Four rows occupy the four lanes. Coordinates occupy separate registers.
template<> struct AvbdHingeBatchKernel<float> : AvbdFourLane
{
    static void sample(AvbdHingeBatchArgs& a)
    {
        using namespace aos;
        if(!a.count){a.output=AvbdNodalShellSampleT<AvbdNodalReal>();a.output.valid=true;return;}
        // Padding duplicates a live row and is never written to the output.
        const PxU32 id[4]={0,PxMin(1u,a.count-1),PxMin(2u,a.count-1),PxMin(3u,a.count-1)};
        Vector x[4];
        for(PxU32 v=0;v<4;++v)
            x[v]={V4LoadXYZW(float(a.positions[id[0]][v].x),float(a.positions[id[1]][v].x),float(a.positions[id[2]][v].x),float(a.positions[id[3]][v].x)),
                  V4LoadXYZW(float(a.positions[id[0]][v].y),float(a.positions[id[1]][v].y),float(a.positions[id[2]][v].y),float(a.positions[id[3]][v].y)),
                  V4LoadXYZW(float(a.positions[id[0]][v].z),float(a.positions[id[1]][v].z),float(a.positions[id[2]][v].z),float(a.positions[id[3]][v].z))};
        float stiffness[4],cosines[4],sines[4],w[4][4]={};
        for(PxU32 i=0;i<4;++i)
        {
            const auto& r=*a.rest[id[i]];
            stiffness[i]=float(a.bending[id[i]]*r.length);
            cosines[i]=float(r.restCos);sines[i]=float(r.restSin);
            if(a.corner[id[i]]<4)w[a.corner[id[i]]][i]=1;
        }
        const auto e=sub(x[0],x[2]),f=sub(x[1],x[2]),h=sub(x[3],x[2]);
        const auto raw0=cross(h,e),raw1=cross(f,h);
        const auto r0=V4Sqrt(dot(raw0,raw0)),r1=V4Sqrt(dot(raw1,raw1)),length=V4Sqrt(dot(h,h));
        const auto n0=mul(raw0,V4Recip(r0)),n1=mul(raw1,V4Recip(r1)),axis=mul(h,V4Recip(length));
        const auto cosine=dot(n0,n1),sine=dot(axis,cross(n0,n1)),k=V4LoadU(stiffness);
        const auto rc=V4LoadU(cosines),rs=V4LoadU(sines),dc=V4Sub(cosine,rc),ds=V4Sub(sine,rs);
        const auto energy=V4Mul(V4Mul(V4Load(0.5f),k),V4Add(V4Mul(dc,dc),V4Mul(ds,ds)));
        auto& out=a.output;out=AvbdNodalShellSampleT<AvbdNodalReal>();out.valid=true;
        const Lane mask=V4LoadXYZW(1.f,a.count>1?1.f:0.f,a.count>2?1.f:0.f,a.count>3?1.f:0.f);
        // The tail duplicates a live input but contributes zero to every sum.
        const Lane maximum=V4Splat(FLoad(PX_MAX_F32));
        const auto sum=[&](Lane value)
        {
            out.valid=out.valid && V4AllGrtrOrEq(maximum,V4Abs(value));
            float result;FStore(V4Dot(value,mask),&result);out.valid=out.valid && PxIsFinite(result);return result;
        };
        out.energy=sum(energy);
        float areaRatio;FStore(V4ExtractMin(V4Min(r0,r1)),&areaRatio);out.areaRatio=areaRatio;
        out.valid=out.valid && areaRatio>0;
        bool energyOnly=true;
        for(PxU32 i=0;i<a.count;++i)
        {
            for(const auto& p:a.positions[i])out.valid=out.valid && p.isFinite();
            out.valid=out.valid && a.bending[i]>=0 && a.rest[i]->length>0;
            energyOnly=energyOnly && a.corner[i]>=4;
        }
        if(energyOnly)return;
        const auto edge=V4Sub(V4LoadU(w[3]),V4LoadU(w[2]));
        const auto d0=V4Div(V4Sub(V4Mul(edge,dot(axis,e)),V4Mul(V4Sub(V4LoadU(w[0]),V4LoadU(w[2])),length)),r0);
        const auto d1=V4Div(V4Sub(V4Mul(edge,dot(axis,f)),V4Mul(V4Sub(V4LoadU(w[1]),V4LoadU(w[2])),length)),r1);
        const auto theta=add(mul(n0,d0),mul(n1,d1));
        const auto gradient=mul(theta,V4Mul(k,V4Sub(V4Mul(sine,rc),V4Mul(cosine,rs))));
        out.valid=out.valid && V4AllGrtrOrEq(maximum,V4Abs(length));
        out.gradient=AvbdNodalVec3(sum(gradient.x),sum(gradient.y),sum(gradient.z));
        const float xx=sum(V4Mul(V4Mul(theta.x,theta.x),k)),yy=sum(V4Mul(V4Mul(theta.y,theta.y),k)),
            zz=sum(V4Mul(V4Mul(theta.z,theta.z),k)),xy=sum(V4Mul(V4Mul(theta.x,theta.y),k)),
            xz=sum(V4Mul(V4Mul(theta.x,theta.z),k)),yz=sum(V4Mul(V4Mul(theta.y,theta.z),k));
        out.metric=AvbdNodalMat33(AvbdNodalVec3(xx,xy,xz),AvbdNodalVec3(xy,yy,yz),AvbdNodalVec3(xz,yz,zz));
    }
};

} }
#endif
