// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_BODY_INDEX_H
#define DY_AVBD_BODY_INDEX_H
#include "foundation/PxHashMap.h"
namespace physx { struct PxsBodyCore; namespace Dy {
// Packing-call scratch only. Rebuild from the current Host body list; do not
// retain identities across sleep/removal/reordering or publish physical state.
// Kernels consume the resulting dense block indices, not this CPU lookup.
class AvbdBodyIndex
{
    PxHashMap<const PxsBodyCore*,PxU32> mIndices;
public:
    template<class Bodies> void build(const Bodies& bodies)
    {
        mIndices.clear();
        for(PxU32 i=0;i<bodies.size();++i)if(bodies[i].core)insert(bodies[i].core,i);
    }
    void insert(const PxsBodyCore* core,PxU32 index)
    {if(core)mIndices.insert(core,index);} // Preserve the first matching entry.
    PxU32 find(const PxsBodyCore* core) const
    {const auto* entry=core?mIndices.find(core):NULL;return entry?entry->second:PX_MAX_U32;}
};
} }
#endif
