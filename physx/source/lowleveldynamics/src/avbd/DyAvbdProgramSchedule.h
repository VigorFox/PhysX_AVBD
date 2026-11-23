// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_PROGRAM_SCHEDULE_H
#define DY_AVBD_PROGRAM_SCHEDULE_H
#include "DyAvbdRowProgram.h"
#include <algorithm>

namespace physx { namespace Dy {
struct AvbdProgramSubset
{
    PxArray<PxU32> blocks,rows;
    void collectRows(const AvbdRowProgram& program)
    {
        rows.clear();
        for(PxU32 b:blocks)for(PxU32 s=program.blockOffsets[b];s<program.blockOffsets[b+1];++s)
            rows.pushBack(program.blockRows[s]);
        if(rows.empty())return;
        std::sort(rows.begin(),rows.end());PxU32 count=1;
        for(PxU32 s=1;s<rows.size();++s)if(rows[s]!=rows[count-1])rows[count++]=rows[s];
        rows.resize(count);
    }
};

// Derived execution topology only. Forces and material history remain in the
// original row program. A shared dual is never split across execution groups.
struct AvbdProgramSchedule
{
    PxArray<AvbdProgramSubset> groups;
    PxArray<PxU32> colorOffsets,colorGroups;
    PxU32 maximumBlocks=0;
    static PxU32 root(PxArray<PxU32>& parent,PxU32 b)
    {
        PxU32 r=b;while(parent[r]!=r)r=parent[r];
        while(parent[b]!=b){const PxU32 next=parent[b];parent[b]=r;b=next;}return r;
    }
    template<class Responds,class Active,class Energy>
    void compile(const AvbdRowProgram& program,const Responds& responds,const Active& active,const Energy& energy)
    {
        groups.clear();colorOffsets.clear();colorGroups.clear();maximumBlocks=0;
        PxArray<PxU32> parent;
        program.components(parent,responds,[&](PxU32 r){return active(r) && !energy(r);});
        PxArray<PxU32> size(parent.size(),0);
        for(PxU32 b=0;b<parent.size();++b)if(responds(b))++size[parent[b]];
        // Merge only finite-energy neighbours. Never split a hard connected
        // component merely to satisfy a numerical packet-size target.
        for(PxU32 r=0;r+1<program.rowOffsets.size();++r)if(active(r) && energy(r))
        {
            PxU32 first=PX_MAX_U32;
            program.forEachBlock(r,[&](PxU32 b)
            {
                if(!responds(b))return;const PxU32 p=root(parent,b);
                if(first==PX_MAX_U32){first=p;return;}
                first=root(parent,first);
                if(p!=first && size[p]+size[first]<=4)
                {parent[p]=first;size[first]+=size[p];}
            });
        }
        PxArray<PxU32> group(parent.size(),PX_MAX_U32),index(parent.size(),PX_MAX_U32);
        for(PxU32 b=0;b<parent.size();++b)if(responds(b))
        {
            const PxU32 p=root(parent,b);
            if(index[p]==PX_MAX_U32){index[p]=groups.size();groups.pushBack(AvbdProgramSubset());}
            group[b]=index[p];groups[group[b]].blocks.pushBack(b);
        }
        PxArray<PxU32> visited(groups.size(),PX_MAX_U32);
        for(PxU32 r=0;r+1<program.rowOffsets.size();++r)if(active(r))
            program.forEachBlock(r,[&](PxU32 b)
            {const PxU32 g=group[b];if(g!=PX_MAX_U32 && visited[g]!=r){groups[g].rows.pushBack(r);visited[g]=r;}});
        PxArray<PxU32> colors(groups.size());PxU32 colorCount=0;
        for(PxU32 g=0;g<groups.size();++g)
        {
            maximumBlocks=std::max(maximumBlocks,groups[g].blocks.size());
            PxArray<PxU8> used(colorCount,0);
            for(PxU32 r:groups[g].rows)program.forEachBlock(r,[&](PxU32 b)
            {if(group[b]<g)used[colors[group[b]]]=1;});
            PxU32 color=0;while(color<colorCount && used[color])++color;
            colors[g]=color;if(color==colorCount)++colorCount;
        }
        colorOffsets.resize(colorCount+1,0);colorGroups.resize(groups.size());
        for(PxU32 c:colors)++colorOffsets[c+1];
        for(PxU32 c=1;c<=colorCount;++c)colorOffsets[c]+=colorOffsets[c-1];
        PxArray<PxU32> cursor=colorOffsets;
        for(PxU32 g=0;g<groups.size();++g)colorGroups[cursor[colors[g]]++]=g;
    }
};
} }
#endif
