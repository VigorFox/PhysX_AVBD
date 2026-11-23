// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef PXS_DEFORMABLE_GEOMETRY_H
#define PXS_DEFORMABLE_GEOMETRY_H

#include "foundation/PxVec4.h"
#include "foundation/PxBounds3.h"
#include "PxDeformableBodyFlag.h"
#include <cmath>

namespace physx
{
struct PxsDeformableNpState;
// CPU NP precision policy. Host buffers and the public SDK retain PxReal.
#ifndef PX_NP_DEFORMABLE_REAL
#define PX_NP_DEFORMABLE_REAL float
#endif
typedef PX_NP_DEFORMABLE_REAL PxsDeformableReal;
typedef PxVec3T<PxsDeformableReal> PxsDeformableVec3;

// Borrowed authored indices, never solver block IDs. Topology changes require
// a new bake; replacing the position buffer does not.
struct PxsDeformableTopology
{
    const void* indices=NULL;
    PxU32 count=0,vertices=0;
    PxU8 arity=0,indexBytes=0;
    bool valid() const
    {return indices && count && vertices && (arity==3 || arity==4) && (indexBytes==2 || indexBytes==4);}
    bool same(const PxsDeformableTopology& b) const
    {return indices==b.indices && count==b.count && vertices==b.vertices && arity==b.arity && indexBytes==b.indexBytes;}
    bool element(PxU32 id,PxU32* out) const
    {
        if(!valid() || !out || id>=count)return false;
        for(PxU32 k=0;k<arity;++k)
        {
            const size_t offset=size_t(id)*arity+k;
            out[k]=indexBytes==2?static_cast<const PxU16*>(indices)[offset]:static_cast<const PxU32*>(indices)[offset];
            if(out[k]>=vertices)return false;
        }
        return true;
    }
};

struct PxsDeformablePointSupport
{
    PxU32 nodes[4];
    PxsDeformableReal weights[4];
    PxU32 count=0;
};

struct PxsDeformableGeometry
{
    // Borrow the pointer slot: legal Host buffer replacement is visible at the
    // next DCD epoch. Embedding is collision vertex -> simulation tetrahedron.
    PxVec4* const* positions=NULL;
    PxVec4* const* restPositions=NULL; // Collision-vertex rest buffer, in collision topology order.
    PxVec4* const* velocities=NULL;
    const PxDeformableBodyFlags* flags=NULL;
    const PxReal* wakeCounter=NULL;
    const PxReal* friction=NULL;
    const PxReal* selfFilterDistance=NULL;
    PxsDeformableNpState* state=NULL;
    PxU32 transformCache=PX_MAX_U32;
    PxReal contactOffset=0;
    bool selfCollision() const {return flags && !(*flags&PxDeformableBodyFlag::eDISABLE_SELF_COLLISION);}
    PxsDeformableTopology collision,simulation;
    const PxU32* embeddingElements=NULL;
    const PxReal* embeddingWeights=NULL;

    bool support(PxU32 vertex,PxsDeformablePointSupport& out) const
    {
        out.count=0;
        if(!positions || !*positions || !collision.valid() || !simulation.valid() || vertex>=collision.vertices)return false;
        if(!embeddingElements && !embeddingWeights)
        {
            if(!collision.same(simulation))return false;
            out.nodes[0]=vertex;out.weights[0]=1;out.count=1;return true;
        }
        if(!embeddingElements || !embeddingWeights || simulation.arity!=4)return false;
        PxU32 nodes[4];if(!simulation.element(embeddingElements[vertex],nodes))return false;
        PxsDeformableReal sum=0,magnitude=0;
        for(PxU32 j=0;j<4;++j)
        {
            const PxsDeformableReal w=embeddingWeights[size_t(vertex)*4+j];
            if(!std::isfinite(w)){out.count=0;return false;}
            sum+=w;magnitude+=std::abs(w);
            PxU32 k=0;while(k<out.count && out.nodes[k]!=nodes[j])++k;
            if(k==out.count){out.nodes[k]=nodes[j];out.weights[k]=0;++out.count;}
            out.weights[k]+=w;
        }
        // Correct only cooking's f32 partition roundoff. Equal vertex counts
        // are not proof of identity; malformed/non-affine input is rejected.
        if(!std::isfinite(sum) || !std::isfinite(magnitude) || !(sum>0) ||
            std::abs(sum-1)>4*PxsDeformableReal(PX_EPS_F32)*magnitude)
        {out.count=0;return false;}
        PxU32 compact=0;
        for(PxU32 k=0;k<out.count;++k)if(out.weights[k]!=0)
        {out.nodes[compact]=out.nodes[k];out.weights[compact++]=out.weights[k]/sum;}
        out.count=compact;return compact!=0;
    }

    bool point(PxU32 vertex,PxsDeformableVec3& p) const
    {
        PxsDeformablePointSupport s;if(!support(vertex,s))return false;
        p=PxsDeformableVec3(0);
        for(PxU32 k=0;k<s.count;++k)
        {
            const PxVec3 v=(*positions)[s.nodes[k]].getXYZ();if(!v.isFinite())return false;
            p+=PxsDeformableVec3(v.x,v.y,v.z)*s.weights[k];
        }
        return p.isFinite();
    }

    bool primitive(PxU32 id,PxsDeformableVec3* points) const
    {
        PxU32 ids[4];if(!collision.element(id,ids))return false;
        for(PxU32 k=0;k<collision.arity;++k)if(!point(ids[k],points[k]))return false;
        return true;
    }
};
}
#endif
