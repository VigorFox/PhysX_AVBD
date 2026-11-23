// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_TET_SEPARATION_H
#define DY_AVBD_TET_SEPARATION_H

#include "foundation/PxVec3.h"

namespace physx { namespace Dy {

// Signed extremum of four embedded corners on the fixed DCD axis.
template<class T,class Position>
PX_CUDA_CALLABLE PX_FORCE_INLINE bool avbdTetSideMinimum(const PxVec3T<T>& normal,
    const PxVec3T<T>& origin,PxU32 side,const Position& position,T& minimum)
{
    const T sign=side?T(-1):T(1);
    for(PxU32 i=0;i<4;++i)
    {
        const auto p=position(i);
        const T h=sign*normal.dot(PxVec3T<T>(p.x,p.y,p.z)-origin);
        if(!PxIsFinite(h))return false;
        if(i==0 || h<minimum)minimum=h;
    }
    return true;
}

} }
#endif
