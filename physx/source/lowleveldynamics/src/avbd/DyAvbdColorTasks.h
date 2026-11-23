// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_COLOR_TASKS_H
#define DY_AVBD_COLOR_TASKS_H
#include "DyAvbdLocalWork.h"
#include "DyAvbdNodalShell.h"

namespace physx { namespace Dy {
struct AvbdColorTasks : PxUserAllocated
{
    enum { LeafRows=16 };
    struct Tile
    {
        PxU32 begin,end,split,partial;
        PxU32 blocks,rows;
    };
    struct Split
    {
        PxU32 block,first,count;
        volatile PxI32 remaining;
    };
    struct Wave
    {
        PxU32 first,count,blocks,maxBlocks,maxRows,splitBlocks,leaves;
    };
    PxArray<Tile> tiles;
    PxArray<Wave> waves;
    PxArray<PxU32> offsets,wideOffsets,wideBlocks;
    PxArray<Split> splits;
    PxArray<AvbdNodalAssembly> nodal;
    PxArray<AvbdLocalPartial> rigid;
    // Incidence tiles address spans, not row IDs. Equal shape and quotas
    // suffice even when the actual material or contact row IDs have changed.
    PxArray<PxU32> incidenceOffsets,incidenceShape;
    PxU32 workers=0,firstTile=0,lanes=0,tileCount=0;
    volatile PxI32 nextTile=0;
    void clear()
    {
        offsets.clear();wideOffsets.clear();wideBlocks.clear();tiles.clear();waves.clear();splits.clear();nodal.clear();rigid.clear();
        incidenceOffsets.clear();incidenceShape.clear();
    }
};
} }
#endif
