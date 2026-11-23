// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_NODAL_VOLUME_H
#define DY_AVBD_NODAL_VOLUME_H
#include "avbd/DyAvbdNodalShell.h"

namespace physx { namespace Dy {

template<class T> struct AvbdNodalVolumeRestT
{
    PxVec3T<T> shape[4];
    T volume;
};
typedef AvbdNodalVolumeRestT<AvbdNodalReal> AvbdNodalVolumeRest;

// Same published determinant domain as the rigid tet path. Already compressed
// authored states can recover from their starting determinant.
template<class T> PX_FORCE_INLINE T avbdNodalVolumeLowerBound(T startJ)
{
    return PxMin(T(.05),startJ);
}

template<class T> inline bool avbdNodalVolumePath(const AvbdNodalVolumeRestT<T>& rest,
    const PxVec3T<T>* start,const PxVec3T<T>* end)
{
    const T inverseDet=rest.shape[1].dot(rest.shape[2].cross(rest.shape[3]));
    PxVec3T<T> edge[2][3];T product=1,coeff[4]={};
    for(PxU32 j=0;j<3;++j)
    {
        edge[0][j]=start[j+1]-start[0];edge[1][j]=end[j+1]-end[0];
        const auto l1=[](const PxVec3T<T>& v){return PxAbs(v.x)+PxAbs(v.y)+PxAbs(v.z);};
        product*=PxMax(l1(edge[0][j]),l1(edge[1][j]));
    }
    for(PxU32 a=0;a<2;++a)for(PxU32 b=0;b<2;++b)for(PxU32 c=0;c<2;++c)
        coeff[a+b+c]+=edge[a][0].dot(edge[b][1].cross(edge[c][2]))*inverseDet;
    coeff[1]/=T(3);coeff[2]/=T(3);
    const T roundoff=T(64)*std::numeric_limits<T>::epsilon()*product*PxAbs(inverseDet);
    const T minimum=avbdNodalVolumeLowerBound(coeff[0]);
    for(PxU32 i=0;i<4;++i)
        if(!PxIsFinite(coeff[i]) || !(coeff[i]>roundoff) || coeff[i]<minimum-roundoff)return false;
    return true;
}

// Stable Neo-Hookean density, shifted to zero at rest:
// mu/2 (||F||^2 - 3) - mu (J - 1) + (lambda+mu)/2 (J - 1)^2.
// The lambda+mu coefficient recovers the authored Lame lambda at small strain.
// With one moving node J is affine, so the 3x3 Hessian below is exact and SPD
// for physical Lame coefficients. Inverted elements are outside the domain.
// This kernel consumes only cooked shape gradients and current four supports.
template<class T> inline AvbdNodalShellSampleT<T> avbdSampleNodalVolume(
    const AvbdNodalVolumeRestT<T>& rest,const PxVec3T<T>* x,T mu,T lambda,PxU32 corner,const T* weights=NULL,
    const PxMat33T<T>* startMetric=NULL,T viscosity=T(0),AvbdNodalWorkT<T>* work=NULL)
{
    typedef PxVec3T<T> V;typedef PxMat33T<T> M;
    AvbdNodalShellSampleT<T> out;
    if(!(rest.volume>0 && mu>0 && lambda+T(2)*mu/T(3)>0))return out;
    for(PxU32 i=0;i<4;++i)if(!x[i].isFinite())return out;
    const V a=x[1]-x[0],b=x[2]-x[0],c=x[3]-x[0];
    const M F(a*rest.shape[1].x+b*rest.shape[2].x+c*rest.shape[3].x,
              a*rest.shape[1].y+b*rest.shape[2].y+c*rest.shape[3].y,
              a*rest.shape[1].z+b*rest.shape[2].z+c*rest.shape[3].z);
    const M cof(F.column1.cross(F.column2),F.column2.cross(F.column0),F.column0.cross(F.column1));
    const T J=F.column0.dot(cof.column0),d=J-T(1);
    out.areaRatio=J;
    if(!(J>0))return out;
    const T norm=F.column0.magnitudeSquared()+F.column1.magnitudeSquared()+F.column2.magnitudeSquared();
    const T bulk=lambda+mu;
    out.energy=rest.volume*(T(.5)*mu*(norm-T(3))-mu*d+T(.5)*bulk*d*d);
    if(corner<4 || weights)
    {
        V s(0);if(weights){for(PxU32 i=0;i<4;++i)s+=rest.shape[i]*weights[i];}
        else s=rest.shape[corner];
        const V g=cof*s;
        out.gradient=(F*s*mu+g*(bulk*d-mu))*rest.volume;
        out.metric=M(PxIdentity)*(mu*s.magnitudeSquared()*rest.volume)+
            avbdNodalOuter(g,g)*(bulk*rest.volume);
    }
    if(startMetric)
    {
        // Kelvin-Voigt work in Green strain relative to the common DCD pose.
        // Translation and any finite rigid rotation preserve the metric.
        const M diff=(F.getTranspose()*F-*startMetric)*T(.5);
        const T tr=diff[0].x+diff[1].y+diff[2].z,scale=viscosity*rest.volume;
        const M stress=diff*(2*mu)+M(PxIdentity)*(lambda*tr);
        out.energy+=scale*(mu*(diff[0].magnitudeSquared()+diff[1].magnitudeSquared()+diff[2].magnitudeSquared())+
            T(.5)*lambda*tr*tr);
        if(corner<4 || weights)
        {
            V s(0);if(weights)for(PxU32 i=0;i<4;++i)s+=rest.shape[i]*weights[i];else s=rest.shape[corner];
            const V fs=F*s;const T ss=s.magnitudeSquared();
            out.gradient+=F*(stress*s)*scale;
            out.metric+=(F*F.getTranspose()*(mu*ss)+avbdNodalOuter(fs,fs)*(mu+lambda))*scale;
            if(work)
            {
                work->geometric=scale*s.dot(stress*s);
                work->cubic=fs*(scale*(mu+T(.5)*lambda)*ss);
                work->quartic=scale*(mu*T(.25)+lambda*T(.125))*ss*ss;
            }
        }
    }
    out.valid=PxIsFinite(out.energy) && out.gradient.isFinite() && out.metric.column0.isFinite() &&
        out.metric.column1.isFinite() && out.metric.column2.isFinite();
    return out;
}

struct AvbdNodalVolumeArgs
{
    const AvbdNodalVolumeRest* rest;
    const AvbdNodalVec3* positions;
    AvbdNodalReal mu,lambda;
    PxU32 corner;
    AvbdNodalShellSampleT<AvbdNodalReal>* output;
    const AvbdNodalReal* weights=NULL;
    const AvbdNodalMat33* startMetric=NULL;
    AvbdNodalReal viscosity=0;
    AvbdNodalWorkT<AvbdNodalReal>* work=NULL;
};
struct AvbdNodalVolumeElement : AvbdNodalVolumeRest
{
    PxU32 nodes[4];
    AvbdNodalReal inverseDet;
};
struct AvbdNodalVolumeMaterial
{
    const AvbdNodalVolumeElement* rest;
    AvbdNodalReal mu,lambda;
    AvbdNodalReal minimumJ=0;
    PxU32 metricIndex=PX_MAX_U32;
    AvbdNodalReal viscosity=0;
};
} }
#endif
