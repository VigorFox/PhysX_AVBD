// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_CONTACT_WARM_INDEX_H
#define DY_AVBD_CONTACT_WARM_INDEX_H
#include "foundation/PxArray.h"
#include "foundation/PxHashMap.h"

namespace physx { namespace Dy {
// Transient lookup into the one existing history. No copied force or new
// identity: matching already requires the same Host contact manager pointer.
struct AvbdContactWarmIndex
{
    PxHashMap<const void*,PxU32> heads;
    PxArray<PxU32> next;
    template<class History> explicit AvbdContactWarmIndex(const History& previous)
    {
        next.resize(previous.size(),PX_MAX_U32);
        for(PxU32 end=previous.size();end>0;--end)
        {
            const PxU32 i=end-1;const void* key=previous[i].cm;
            const auto* found=heads.find(key);
            if(found)next[i]=found->second;
            heads[key]=i;
        }
    }
    PxU32 first(const void* key) const
    {const auto* found=heads.find(key);return found?found->second:PX_MAX_U32;}
};
} }
#endif
