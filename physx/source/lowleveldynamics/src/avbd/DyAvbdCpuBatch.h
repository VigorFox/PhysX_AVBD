// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_CPU_BATCH_H
#define DY_AVBD_CPU_BATCH_H
#include "foundation/PxSimpleTypes.h"
#include <algorithm>

namespace physx { namespace Dy {
// CPU dispatch geometry only. A pack never crosses a dependency wave; the
// Table still receives one bounded numerical tile per launch. No queue drain.
struct AvbdCpuBatch
{
    enum { TileWidth=32, TilesPerTask=32, TaskLimit=32 };
    PxU32 width=0,tasks=0,items=0,tiles=0;
    AvbdCpuBatch(PxU32 remaining=0,PxU32 workers=0)
    {
        if(!remaining)return;
        const PxU32 lanes=std::min(PxU32(TaskLimit),std::max(PxU32(1),workers));
        const PxU32 total=remaining/TileWidth+(remaining%TileWidth!=0);
        tasks=std::min(lanes,total);tiles=std::min(total,tasks*TilesPerTask);
        width=(tiles/tasks+(tiles%tasks!=0))*TileWidth;
        items=std::min(remaining,tiles*TileWidth);
    }
    PxU32 offset(PxU32 task) const
    {return !tasks || task>=tasks?items:(tiles/tasks*task+std::min(task,tiles%tasks))*TileWidth;}
};
} }
#endif
