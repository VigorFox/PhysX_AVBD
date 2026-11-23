// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_ROW_WORKING_SET_H
#define DY_AVBD_ROW_WORKING_SET_H
#include "DyAvbdCommonProgram.h"
#include "DyAvbdNormalBounds.h"

namespace physx { namespace Dy {
// Numerical row subset of ONE frozen program. No physical state or solver.
// A publication owns this view; topology/linearization changes invalidate it.
struct AvbdRowWorkingSet
{
    enum { ExpansionLimit=32 };
    AvbdCommonProgram active;
    PxArray<PxU8> keep;
    PxArray<PxU32> map;
    PxArray<double> seed;
    PxArray<AvbdNormalBound> bounds;
    PxU32 sweeps=0,expansions=0;
    void initialize(const AvbdCommonProgram& full,const double* force,const double* previous)
    {
        sweeps=expansions=0;keep.resize(full.mu.size());map.resize(full.mu.size());
        avbdCompileNormalBounds(full,bounds);
        for(PxU32 r=0;r<keep.size();++r)
        {
            bool contact=full.lower[3*r]==0;
            if(!full.coulomb[r])for(PxU32 q=1;q<3;++q)
                contact=contact && full.lower[3*r+q]==0 && full.upper[3*r+q]==0;
            keep[r]=!contact || full.offset[3*r]<0;
            for(PxU32 q=0;q<3;++q)
                keep[r]=keep[r] || (force && force[3*r+q]!=0) || (previous && previous[3*r+q]!=0);
        }
    }
    bool compile(const AvbdCommonProgram& full,const double* force)
    {
        if(!full.valid || keep.size()!=full.mu.size())return false;
        // Material/inertial operators never depend on the selected rows.
        active=AvbdCommonProgram();active.dimensions=full.dimensions;active.sourceBlocks=full.sourceBlocks;
        active.base=full.base;active.H=full.H;active.metricH=full.metricH;seed.clear();
        for(PxU32 r=0;r<keep.size();++r)
        {
            map[r]=PX_MAX_U32;if(!keep[r])continue;const PxU32 local=active.mu.size();map[r]=local;
            active.sourceRows.pushBack(full.sourceRows[r]);active.mu.pushBack(full.mu[r]);active.coulomb.pushBack(full.coulomb[r]);
            for(PxU32 q=0;q<3;++q)
            {const PxU32 i=3*r+q;active.offset.pushBack(full.offset[i]);active.lower.pushBack(full.lower[i]);
                active.upper.pushBack(full.upper[i]);active.compliance.pushBack(full.compliance[i]);seed.pushBack(force?force[i]:0);}
            for(PxU32 e=full.jOffsets[r];e<full.jOffsets[r+1];++e)
            {auto j=full.J[e];j.row=local;active.J.pushBack(j);}
        }
        return active.finish();
    }
    void scatter(const PxArray<double>& compact,PxArray<double>& full) const
    {
        full.resize(3*map.size());
        for(PxU32 r=0;r<map.size();++r)for(PxU32 q=0;q<3;++q)
            full[3*r+q]=map[r]==PX_MAX_U32?0:compact[3*map[r]+q];
    }
    bool grow(const PxArray<PxU8>& requests)
    {
        if(requests.size()!=keep.size() || expansions==ExpansionLimit)return false;
        bool changed=false;
        for(PxU32 r=0;r<keep.size();++r)if(requests[r] && !keep[r]){keep[r]=1;changed=true;}
        if(changed)++expansions;return changed;
    }
};
} }
#endif
