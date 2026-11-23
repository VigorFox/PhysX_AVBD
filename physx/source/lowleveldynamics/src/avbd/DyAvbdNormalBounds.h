// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_NORMAL_BOUNDS_H
#define DY_AVBD_NORMAL_BOUNDS_H
#include "DyAvbdCommonProgram.h"
#include <cfloat>

namespace physx { namespace Dy {
// Derived interval enclosure of <=32 affine rows with identical supports.
// The original rows remain authoritative whenever this enclosure is uncertain.
struct AvbdNormalBound
{
    enum { Width=32 };
    PxU32 first=0,count=0,arity=0,columns[4]={};
    double offset=DBL_MAX,low[24]={},high[24]={};
    double coordinateOffset=0,coordinateCoefficient=0;
    bool eligible=true;
    bool nonnegative(const double* step,const PxArray<PxU32>& starts) const
    {
        if(!eligible)return false;
        double lower=offset,scale=std::abs(offset),largestStep=0;
        for(PxU32 s=0;s<arity;++s)for(PxU32 d=0;d<starts[columns[s]+1]-starts[columns[s]];++d)
        {
            const PxU32 i=6*s+d;const double x=step[starts[columns[s]]+d];
            largestStep=std::max(largestStep,std::abs(x));
            const double a=low[i]*x,b=high[i]*x;
            lower+=std::min(a,b);scale+=std::max(std::abs(a),std::abs(b));
        }
        // Covers products, summation and the direct row evaluation (<=24
        // coordinates). No geometric tolerance or gap floor is introduced.
        // The zero-radius Coulomb proof also requires finite tangential
        // coordinates. Reject the enclosure if any full row could overflow.
        const bool finite=coordinateCoefficient==0 || (coordinateOffset<=DBL_MAX/2 && largestStep<=DBL_MAX/(64*coordinateCoefficient));
        return finite && std::isfinite(lower) && std::isfinite(scale) && lower>=128*DBL_EPSILON*std::max(scale,DBL_MIN);
    }
};
inline void avbdCompileNormalBounds(const AvbdCommonProgram& p,PxArray<AvbdNormalBound>& out)
{
    out.clear();if(!p.valid)return;
    for(PxU32 first=0;first<p.mu.size();)
    {
        AvbdNormalBound b;b.first=first;b.arity=p.jOffsets[first+1]-p.jOffsets[first];
        if(b.arity>4){out.clear();return;}
        for(PxU32 s=0;s<b.arity;++s)
        {
            const auto& j=p.J[p.jOffsets[first]+s];b.columns[s]=j.column;
            for(PxU32 d=0;d<p.dimensions[j.column];++d)b.low[6*s+d]=b.high[6*s+d]=j.values[d];
        }
        for(PxU32 r=first;r<p.mu.size() && b.count<AvbdNormalBound::Width;++r)
        {
            bool same=p.jOffsets[r+1]-p.jOffsets[r]==b.arity;
            for(PxU32 s=0;s<b.arity && same;++s)same=p.J[p.jOffsets[r]+s].column==b.columns[s];
            if(!same)break;
            b.eligible=b.eligible && p.lower[3*r]==0;
            if(!p.coulomb[r])for(PxU32 q=1;q<3;++q)b.eligible=b.eligible && p.lower[3*r+q]==0 && p.upper[3*r+q]==0;
            b.offset=std::min(b.offset,p.offset[3*r]);
            for(PxU32 q=0;q<3;++q)b.coordinateOffset=std::max(b.coordinateOffset,std::abs(p.offset[3*r+q]));
            for(PxU32 s=0;s<b.arity;++s)
            {
                const auto& j=p.J[p.jOffsets[r]+s];
                for(PxU32 d=0;d<p.dimensions[j.column];++d)
                {const PxU32 i=6*s+d;b.low[i]=std::min(b.low[i],j.values[d]);b.high[i]=std::max(b.high[i],j.values[d]);}
                for(PxU32 q=0;q<3;++q)for(PxU32 d=0;d<p.dimensions[j.column];++d)
                    b.coordinateCoefficient=std::max(b.coordinateCoefficient,std::abs(j.values[6*q+d]));
            }
            ++b.count;
        }
        first+=b.count;out.pushBack(b);
    }
}
} }
#endif
