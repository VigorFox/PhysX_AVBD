// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef PXS_DEFORMABLE_TRIANGLE_FILTER_H
#define PXS_DEFORMABLE_TRIANGLE_FILTER_H

#include "foundation/PxVecMath.h"
#include <limits>

namespace physx
{
// Bound material distance for *every* pair of barycentric witnesses. Removing
// common translation and rotation makes this effective even on a folded sheet.
PX_INLINE bool pxsTriangleRestFiltered(const PxVec3 (&live)[2][3],const PxVec3 (&rest)[2][3],
    PxReal reach,PxReal filterDistance)
{
    using namespace aos;
    struct Columns {Vec4V x,y,z;};
    const auto pack=[](const PxVec3 (&v)[3])->Columns
    {
        return {V4LoadXYZW(v[0].x,v[1].x,v[2].x,v[0].x),
                V4LoadXYZW(v[0].y,v[1].y,v[2].y,v[0].y),
                V4LoadXYZW(v[0].z,v[1].z,v[2].z,v[0].z)};
    };
    const auto roll=[](const Columns& c)->Columns
    {return {V4PermYZXW(c.x),V4PermYZXW(c.y),V4PermYZXW(c.z)};};
    const auto squared=[](const Columns& a,const Columns& b)
    {
        const Vec4V x=V4Sub(a.x,b.x),y=V4Sub(a.y,b.y),z=V4Sub(a.z,b.z);
        return V4MulAdd(x,x,V4MulAdd(y,y,V4Mul(z,z)));
    };
    const Columns la=pack(live[0]),lb=pack(live[1]),ra=pack(rest[0]),rb=pack(rest[1]);
    Columns otherLive=lb,otherRest=rb;
    Vec4V cross=V4Zero(),largest=V4Zero();
    for(PxU32 i=0;i<3;++i)
    {
        const Vec4V r=squared(ra,otherRest),l=squared(la,otherLive);
        cross=V4Max(cross,V4Sub(r,l));largest=V4Max(largest,V4Max(r,l));
        otherLive=roll(otherLive);otherRest=roll(otherRest);
    }
    // |a-b|^2 equals the barycentric mean of cross distances squared minus
    // the two within-triangle variances. Sum_{i<j} w_i*w_j <= 1/3 per side.
    const auto growth=[&](const Columns& l,const Columns& r)
    {return V4ExtractMax(V4Max(V4Zero(),V4Sub(squared(l,roll(l)),squared(r,roll(r)))));};
    const FloatV variance=FAdd(growth(la,ra),growth(lb,rb));
    const PxReal r2=reach*reach,d2=filterDistance*filterDistance;
    const FloatV bound=FAdd(FLoad(r2),FAdd(V4ExtractMax(cross),FMul(variance,FLoad(1.f/3))));
    // Within-triangle edge lengths are at most twice the largest cross length.
    const FloatV error=FMul(FLoad(64*std::numeric_limits<PxReal>::epsilon()),
        FAdd(V4ExtractMax(largest),FLoad(r2+d2)));
    return FAllGrtr(FLoad(d2),FAdd(bound,error))!=0;
}
}
#endif
