// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_FORCE_EXCHANGE_H
#define DY_AVBD_FORCE_EXCHANGE_H
#include "DyAvbdCommonProgram.h"
namespace physx { namespace Dy {
struct AvbdForceExchange {double force[6],motion[6],decrease;};

// Exit from a Coulomb cone along a ray starting strictly inside. This is a
// numerical force boundary, not a change to the physical contact domain.
inline double avbdConeRay(const double* f,const double* d,double mu)
{
    const double a=d[1]*d[1]+d[2]*d[2]-mu*mu*d[0]*d[0];
    const double b=2*(f[1]*d[1]+f[2]*d[2]-mu*mu*f[0]*d[0]);
    const double length=std::hypot(f[1],f[2]),radius=mu*f[0],c=(length-radius)*(length+radius);
    const double inf=std::numeric_limits<double>::infinity();
    if(a==0)return b>0?-c/b:inf;
    const double disc=b*b-4*a*c;if(disc<0 || (a<0 && b<=0))return inf;
    const double root=std::sqrt(disc);
    return b>=0?-2*c/(b+root):(-b+root)/(2*a);
}

// A bounded numerical coordinate for two sticking contacts on one physical
// block. QR reveals a weak exchange direction without squaring conditioning.
// Full original J remains in the motion/reaction update; no row is dropped.
inline AvbdForceExchange avbdStickingExchange(const AvbdCommonProgram& p,const double* inverse,
    const double* step,const double* force,PxU32 a,PxU32 b)
{
    AvbdForceExchange best={};const PxU32 rows[2]={a,b};const auto& ja=p.J[p.jOffsets[a]];
    const auto& jb=p.J[p.jOffsets[b]];const PxU32 n=p.dimensions[ja.column];
    for(PxU32 r:rows)
        if(!p.coulomb[r] || p.lower[3*r]<0 || !(std::hypot(force[3*r+1],force[3*r+2])<p.mu[r]*force[3*r]))return best;
    double columns[6][6]={},R[6][6]={},q[6][6]={},u[6]={};PxU32 order[6]={0,1,2,3,4,5};
    for(PxU32 i=0;i<6;++i)
    {
        const PxU32 row=rows[i/3],coordinate=i%3;const auto& j=i<3?ja:jb;
        u[i]=p.offset[3*row+coordinate]+p.compliance[3*row+coordinate]*force[3*row+coordinate];
        for(PxU32 d=0;d<n;++d){columns[i][d]=j.values[6*coordinate+d];u[i]+=columns[i][d]*step[d];}
    }
    PxU32 rank=0;double original=0;
    for(PxU32 k=0;k<5;++k)
    {
        PxU32 selected=k;double norm=0;
        for(PxU32 j=k;j<6;++j)
        {double value=0;for(PxU32 d=0;d<n;++d)value+=columns[j][d]*columns[j][d];if(value>norm){norm=value;selected=j;}}
        const double length=std::sqrt(norm);if(k==0)original=length;
        if(!(length>64*std::numeric_limits<double>::epsilon()*original))break;
        std::swap(order[k],order[selected]);for(PxU32 d=0;d<n;++d)std::swap(columns[k][d],columns[selected][d]);
        for(PxU32 i=0;i<k;++i)std::swap(R[i][k],R[i][selected]);
        R[k][k]=length;for(PxU32 d=0;d<n;++d)q[k][d]=columns[k][d]/length;
        for(PxU32 j=k+1;j<6;++j)for(PxU32 pass=0;pass<2;++pass)
        {double value=0;for(PxU32 d=0;d<n;++d)value+=q[k][d]*columns[j][d];R[k][j]+=value;
            for(PxU32 d=0;d<n;++d)columns[j][d]-=value*q[k][d];}
        ++rank;
    }
    for(PxU32 free=rank;free<6;++free)
    {
        double v[6]={},sorted[6]={},wrench[6]={},motion[6]={};sorted[free]=1;
        for(PxU32 kk=rank;kk>0;--kk)
        {const PxU32 k=kk-1;double value=-R[k][free];for(PxU32 j=k+1;j<rank;++j)value-=R[k][j]*sorted[j];sorted[k]=value/R[k][k];}
        double length=0;for(double value:sorted)length=std::hypot(length,value);if(!(length>0) || !std::isfinite(length))continue;
        double gap=0;for(PxU32 i=0;i<6;++i){v[order[i]]=sorted[i]/length;gap+=v[order[i]]*u[order[i]];}
        if(gap>0){gap=-gap;for(double& value:v)value=-value;}
        if(!(gap<0))continue;
        double curvature=0,limit=std::numeric_limits<double>::infinity();
        for(PxU32 i=0;i<6;++i)
        {
            const PxU32 row=rows[i/3],coordinate=i%3,index=3*row+coordinate;const auto& j=i<3?ja:jb;
            curvature+=p.compliance[index]*v[i]*v[i];
            for(PxU32 d=0;d<n;++d)wrench[d]+=j.values[6*coordinate+d]*v[i];
            if(v[i]>0)limit=std::min(limit,(p.upper[index]-force[index])/v[i]);
            if(v[i]<0)limit=std::min(limit,(p.lower[index]-force[index])/v[i]);
        }
        for(PxU32 d=0;d<n;++d){for(PxU32 k=0;k<n;++k)motion[d]+=inverse[6*d+k]*wrench[k];curvature+=wrench[d]*motion[d];}
        if(curvature<0 || !std::isfinite(curvature))continue;
        for(PxU32 i=0;i<2;++i)limit=std::min(limit,avbdConeRay(force+3*rows[i],v+3*i,p.mu[rows[i]]));
        const double distance=std::min(limit,curvature>0?-gap/curvature:limit);
        if(!(distance>0) || !std::isfinite(distance))continue;
        const double t=std::nextafter(distance,0.),decrease=-t*(gap+.5*curvature*t);
        if(!(decrease>best.decrease))continue;best.decrease=decrease;
        for(PxU32 i=0;i<6;++i)best.force[i]=v[i]*t;
        for(PxU32 d=0;d<n;++d)best.motion[d]=motion[d]*t;
    }
    return best;
}
} }
#endif
