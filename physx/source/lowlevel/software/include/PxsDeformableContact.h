// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef PXS_DEFORMABLE_CONTACT_H
#define PXS_DEFORMABLE_CONTACT_H
#include "PxsDeformableMesh.h"
#include "PxsContactGeometry.h"
#include "PxsContactManagerState.h"

namespace physx
{
// NP feature identities are collision-mesh indices, never solver blocks.
// A plane uses a vertex feature; finite shapes use complete triangles/tetrahedra.
struct PxsDeformableContact
{
    PxU32 feature[2];
    PxVec3 point[2],normal;
    PxVec4 weights[2];
    PxReal separation;
    PxU8 vertexMask;
};
struct PxsDeformableCandidate
{
    PxU32 feature[2];
    PxU8 vertexMask;
};
struct PxsDeformableQueryWork
{
    PxsDeformableQuery cursor;
    PxBounds3 bounds;
    PxU32 feature,root,count,offset,written;
    enum { InlineCapacity=8 };
    PxU32 hits[InlineCapacity];
};
struct PxsDeformableContactOutput : public PxUserAllocated
{
    PxArray<PxsDeformableCandidate> candidates;
    PxArray<PxsDeformableContact> contacts,rawContacts;
    PxArray<PxsDeformableQueryWork> queries;
    PxArray<PxU32> offsets,totals;
    PxArray<PxU8> active;
    PxU32 count=0;
    bool valid=false;
    PxReal friction=0,restitution=0,damping=0;
    PxU64 birthEpoch=0;
};
// Lifetime belongs to the registered Host shape. The NP phase fills a single
// snapshot shared by all CMs. Self contact uses an NP output, not a fake CM.
struct PxsDeformableNpState
{
    PxsDeformableMesh mesh;
    PxsContactGeometry selfGeometry;
    PxsContactManagerOutput selfOutput;
    PxsDeformableContactOutput selfContacts;
    PxU64 epoch=0,selfEpoch=0;
    PxsDeformableNpState()
    {
        PxMemZero(&selfOutput,sizeof(selfOutput));
        selfGeometry.deformableOutput=&selfContacts;selfOutput.geometry=&selfGeometry;
    }
};
}
#endif
