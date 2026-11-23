// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef GU_PCM_BOX_PROXIMITY_H
#define GU_PCM_BOX_PROXIMITY_H
#include "foundation/PxTransform.h"
#include "foundation/PxMath.h"
#include <cfloat>

namespace physx { namespace Gu {

// Complete closest-feature query for SEPARATED boxes. Used only when the
// iterative GJK query is unresolved. Face clipping alone can miss a separated
// edge-edge proximity. All computations use B's frame, independent of world
// translation. Overlap is left to the existing full-manifold generator.
inline bool pcmSeparatedBoxProximity(const PxVec3& halfA,const PxVec3& halfB,
    const PxTransform& aToB,PxReal contactDistance,PxVec3& localA,PxVec3& localB,
    PxVec3& normal,PxReal& separation)
{
    const PxVec3d ha(halfA.x,halfA.y,halfA.z),hb(halfB.x,halfB.y,halfB.z);
    const PxVec3d p(aToB.p.x,aToB.p.y,aToB.p.z);
    const PxQuatd q=PxQuatd(aToB.q.x,aToB.q.y,aToB.q.z,aToB.q.w).getNormalized();
    PxVec3d a[8],b[8],axesA[3],axesB[3];
    for(PxU32 k=0;k<3;++k){axesB[k]=PxVec3d(0);axesB[k][k]=1;axesA[k]=q.rotate(axesB[k]);}
    for(PxU32 i=0;i<8;++i)
    {
        a[i]=p+q.rotate(PxVec3d(i&1?ha.x:-ha.x,i&2?ha.y:-ha.y,i&4?ha.z:-ha.z));
        b[i]=PxVec3d(i&1?hb.x:-hb.x,i&2?hb.y:-hb.y,i&4?hb.z:-hb.z);
    }
    auto separates=[&](const PxVec3d& axis)
    {
        double amin=DBL_MAX,amax=-DBL_MAX,bmin=DBL_MAX,bmax=-DBL_MAX;
        for(PxU32 i=0;i<8;++i)
        {
            const double x=axis.dot(a[i]),y=axis.dot(b[i]);
            amin=PxMin(amin,x);amax=PxMax(amax,x);bmin=PxMin(bmin,y);bmax=PxMax(bmax,y);
        }
        return amin>bmax || bmin>amax;
    };
    bool separated=false;
    for(PxU32 k=0;k<3;++k)separated|=separates(axesA[k]) || separates(axesB[k]);
    for(PxU32 k=0;k<3 && !separated;++k)for(PxU32 j=0;j<3;++j)separated|=separates(axesA[k].cross(axesB[j]));
    if(!separated)return false;
    auto clampPoint=[](const PxVec3d& v,const PxVec3d& h)
    {return PxVec3d(PxClamp(v.x,-h.x,h.x),PxClamp(v.y,-h.y,h.y),PxClamp(v.z,-h.z,h.z));};
    double best=DBL_MAX;PxVec3d bestA(0),bestB(0);
    auto consider=[&](const PxVec3d& x,const PxVec3d& y)
    {const double d=(x-y).magnitudeSquared();if(d<best){best=d;bestA=x;bestB=y;}};
    for(PxU32 i=0;i<8;++i)
    {
        consider(a[i],clampPoint(a[i],hb));
        consider(p+q.rotate(clampPoint(q.rotateInv(b[i]-p),ha)),b[i]);
    }
    for(PxU32 i=0;i<8;++i)for(PxU32 axis=0;axis<3;++axis)if(!(i&(1u<<axis)))
    for(PxU32 j=0;j<8;++j)for(PxU32 other=0;other<3;++other)if(!(j&(1u<<other)))
    {
        const PxVec3d u=a[i|(1u<<axis)]-a[i],v=b[j|(1u<<other)]-b[j],r=a[i]-b[j];
        const double aa=u.dot(u),bb=u.dot(v),ee=v.dot(v),c=u.dot(r),f=v.dot(r);
        // Cross product avoids cancellation in aa*ee-bb*bb near parallelism.
        const double denominator=u.cross(v).magnitudeSquared();
        double s=denominator>0?PxClamp((bb*f-c*ee)/denominator,0.,1.):0.;
        double t=(bb*s+f)/ee;
        if(t<0){t=0;s=PxClamp(-c/aa,0.,1.);}
        else if(t>1){t=1;s=PxClamp((bb-c)/aa,0.,1.);}
        consider(a[i]+u*s,b[j]+v*t);
    }
    if(!(best>0) || !(best<=double(contactDistance)*contactDistance))return false;
    const double distance=PxSqrt(best);
    const PxVec3d n=(bestA-bestB)/distance,x=q.rotateInv(bestA-p);
    localA=PxVec3(PxReal(x.x),PxReal(x.y),PxReal(x.z));
    localB=PxVec3(PxReal(bestB.x),PxReal(bestB.y),PxReal(bestB.z));
    normal=PxVec3(PxReal(n.x),PxReal(n.y),PxReal(n.z));separation=PxReal(distance);
    return localA.isFinite() && localB.isFinite() && normal.isFinite() && PxIsFinite(separation);
}
}}
#endif
