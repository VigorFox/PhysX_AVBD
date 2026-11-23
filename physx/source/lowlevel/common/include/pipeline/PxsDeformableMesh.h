// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef PXS_DEFORMABLE_MESH_H
#define PXS_DEFORMABLE_MESH_H

#include "PxsDeformableGeometry.h"
#include "foundation/PxArray.h"
#include <algorithm>
#include <limits>

namespace physx
{
struct PxsDeformableMeshView
{
    const PxBounds3* bounds;
    const PxU32* order;
    PxU32 leafBase,count;
    bool ready;
};
// NP-owned baked topology and a single DCD snapshot. Kernels below write
// disjoint slots, with a join between vertices, leaves, and each tree level.
// No solver state, duals, live candidate positions or per-pair copies here.
struct PxsDeformableMesh
{
    PxArray<PxsDeformablePointSupport> supports;
    PxArray<PxU32> vertices,order,uniqueVertices,firstIncidentElement;
    PxArray<PxU32> subtreeMaxFeature; // Baked upper endpoint for one-sided self queries.
    PxArray<PxsDeformableVec3> points;
    PxArray<PxBounds3> bounds, pointEnclosures, displacementBounds;
    PxArray<PxVec3> displacement;
    PxsDeformableTopology collision,simulation;
    const PxU32* embeddingElements=NULL;
    const PxReal* embeddingWeights=NULL;
    PxU32 leafBase=0,arity=0;
    bool ready=false;

    bool matches(const PxsDeformableGeometry& geometry) const
    {return leafBase && geometry.collision.same(collision) && geometry.simulation.same(simulation) &&
        geometry.embeddingElements==embeddingElements && geometry.embeddingWeights==embeddingWeights;}

    static PxU32 spread(PxU32 x)
    {
        x&=1023;x=(x|(x<<16))&0x030000ff;x=(x|(x<<8))&0x0300f00f;
        x=(x|(x<<4))&0x030c30c3;return (x|(x<<2))&0x09249249;
    }
    bool bake(const PxsDeformableGeometry& geometry)
    {
        ready=false;leafBase=0;arity=geometry.collision.arity;
        displacement.clear();displacementBounds.clear();
        if(!geometry.collision.valid() || geometry.collision.count>(1u<<28))return false;
        collision=geometry.collision;simulation=geometry.simulation;
        embeddingElements=geometry.embeddingElements;embeddingWeights=geometry.embeddingWeights;
        supports.resize(geometry.collision.vertices);points.resize(supports.size());pointEnclosures.resize(points.size());
        for(PxU32 v=0;v<supports.size();++v)if(!geometry.support(v,supports[v]))return false;
        vertices.resize(geometry.collision.count*arity);
        for(PxU32 p=0;p<geometry.collision.count;++p)
            if(!geometry.collision.element(p,vertices.begin()+p*arity))return false;
        if(!sampleVertices(geometry,0,supports.size()))return false;
        firstIncidentElement.clear();
        firstIncidentElement.resize(points.size(),PX_MAX_U32);
        for(PxU32 i=0;i<vertices.size();++i)
            firstIncidentElement[vertices[i]]=PxMin(firstIncidentElement[vertices[i]],i/arity);
        uniqueVertices=vertices;
        std::sort(uniqueVertices.begin(),uniqueVertices.end());
        uniqueVertices.forceSize_Unsafe(PxU32(std::unique(uniqueVertices.begin(),uniqueVertices.end())-uniqueVertices.begin()));
        PxBounds3 extent=PxBounds3::empty();
        for(const auto& p:points)extent.include(PxVec3(PxReal(p.x),PxReal(p.y),PxReal(p.z)));
        if(!extent.isValid())return false;
        struct Key {PxU32 code,id;};PxArray<Key> keys;keys.resize(vertices.size()/arity);
        const PxVec3 size=extent.maximum-extent.minimum;
        for(PxU32 p=0;p<keys.size();++p)
        {
            PxsDeformableVec3 center(0);
            for(PxU32 k=0;k<arity;++k)center+=points[vertices[p*arity+k]]/PxsDeformableReal(arity);
            PxU32 q[3];for(PxU32 d=0;d<3;++d)
                q[d]=size[d]>0?PxU32(PxClamp((PxReal(center[d])-extent.minimum[d])/size[d],0.f,1.f)*1023):0;
            keys[p]={spread(q[0])|(spread(q[1])<<1)|(spread(q[2])<<2),p};
        }
        std::sort(keys.begin(),keys.end(),[](const Key& a,const Key& b){return a.code<b.code || (a.code==b.code && a.id<b.id);});
        order.resize(keys.size());for(PxU32 p=0;p<keys.size();++p)order[p]=keys[p].id;
        leafBase=1;while(leafBase<order.size())leafBase*=2;
        subtreeMaxFeature.clear();
        if(arity==3)
        {
            subtreeMaxFeature.resize(leafBase);
            const auto upper=[&](PxU32 node)
            {
                if(node<leafBase)return subtreeMaxFeature[node];
                const PxU32 leaf=node-leafBase;
                return leaf<order.size()?order[leaf]:0u;
            };
            for(PxU32 node=leafBase-1;node;--node)
                subtreeMaxFeature[node]=PxMax(upper(2*node),upper(2*node+1));
        }
        bounds.resize(2*leafBase,PxBounds3::empty());
        return true;
    }
    bool sampleVertices(const PxsDeformableGeometry& geometry,PxU32 begin,PxU32 end)
    {
        if(!geometry.positions || !*geometry.positions || end>supports.size() || begin>end)return false;
        if(!geometry.collision.same(collision) || !geometry.simulation.same(simulation) ||
            geometry.embeddingElements!=embeddingElements || geometry.embeddingWeights!=embeddingWeights)return false;
        for(PxU32 v=begin;v<end;++v)
        {
            PxsDeformableVec3 p(0);const auto& support=supports[v];
            for(PxU32 k=0;k<support.count;++k)
            {
                const PxVec3 x=(*geometry.positions)[support.nodes[k]].getXYZ();
                if(!x.isFinite())return false;
                p+=PxsDeformableVec3(x.x,x.y,x.z)*support.weights[k];
            }
            if(!p.isFinite())return false;points[v]=p;pointEnclosures[v]=pointBounds(p);
            if(displacement.size())
            {
                const PxVec3 rest=(*geometry.restPositions)[v].getXYZ();
                if(!rest.isFinite())return false;
                displacement[v]=PxVec3(PxReal(p.x),PxReal(p.y),PxReal(p.z))-rest;
            }
        }
        return true;
    }
    void prepareDisplacements(const PxsDeformableGeometry& geometry)
    {
        const bool active=geometry.selfCollision() && geometry.selfFilterDistance &&
            *geometry.selfFilterDistance>2*geometry.contactOffset && geometry.restPositions && *geometry.restPositions;
        displacement.resize(active?points.size():0);
        displacementBounds.resize(active?bounds.size():0);
    }
    // For any convex feature witnesses, rest distance <= current distance +
    // max |displacement A - displacement B|. The component intervals enclose
    // all vertices, hence all barycentric witnesses, including tet interiors.
    static PxReal restFilterMarginSquared(const PxBounds3& root,PxReal margin)
    {
        const PxReal scale=PxMax(root.minimum.abs().maxElement(),root.maximum.abs().maxElement());
        const PxReal error=32*std::numeric_limits<PxReal>::epsilon()*(4*scale+PxAbs(margin));
        return margin>error?(margin-error)*(margin-error):0;
    }
    static bool entirelyRestFiltered(const PxBounds3& a,const PxBounds3& b,PxReal marginSquared)
    {
        const PxVec3 span=(a.maximum-b.minimum).maximum(b.maximum-a.minimum);
        return span.magnitudeSquared()<marginSquared;
    }
    // Round bounds outward on storage conversion, including the float build.
    // Distance/contact tolerances do not enter the broad-phase enclosure.
    static PxBounds3 pointBounds(const PxsDeformableVec3& p)
    {
        PxVec3 lo,hi;for(PxU32 d=0;d<3;++d)
        {
            const PxReal value=PxReal(p[d]);
            lo[d]=std::nextafter(value,-std::numeric_limits<PxReal>::infinity());
            hi[d]=std::nextafter(value, std::numeric_limits<PxReal>::infinity());
        }
        return PxBounds3(lo,hi);
    }
    void refitLeaves(PxU32 begin,PxU32 end)
    {
        PX_ASSERT(begin<=end && end<=leafBase);
        for(PxU32 i=begin;i<end;++i)
        {
            PxBounds3 b=PxBounds3::empty();
            if(i<order.size())for(PxU32 k=0;k<arity;++k)b.include(pointEnclosures[vertices[order[i]*arity+k]]);
            bounds[leafBase+i]=b;
            if(displacement.size())
            {
                PxBounds3 motion=PxBounds3::empty();
                if(i<order.size())for(PxU32 k=0;k<arity;++k)motion.include(displacement[vertices[order[i]*arity+k]]);
                displacementBounds[leafBase+i]=motion;
            }
        }
    }
    void refitLevel(PxU32 begin,PxU32 end)
    {
        PX_ASSERT(begin>=1 && begin<=end && end<=leafBase);
        for(PxU32 i=begin;i<end;++i)
        {
            PxBounds3 b=bounds[2*i];b.include(bounds[2*i+1]);bounds[i]=b;
            if(displacement.size())
            {PxBounds3 motion=displacementBounds[2*i];motion.include(displacementBounds[2*i+1]);displacementBounds[i]=motion;}
        }
    }
    // At most 64 source nodes per tile (63 internal unions). Different roots
    // own disjoint subtrees. A GPU block may reduce the same tile with local
    // barriers; no device-wide barrier is needed for every binary tree level.
    void refitTile(PxU32 root,PxU32 levels)
    {
        PX_ASSERT(root>=1 && levels<=6 && (root<<levels)<2*leafBase);
        while(levels)
        {
            --levels;refitLevel(root<<levels,(root+1)<<levels);
        }
    }
    size_t bytes() const
    {return sizeof(*this)+size_t(supports.capacity())*sizeof(PxsDeformablePointSupport)+size_t(vertices.capacity()+order.capacity()+subtreeMaxFeature.capacity()+uniqueVertices.capacity()+firstIncidentElement.capacity())*sizeof(PxU32)+
        size_t(points.capacity())*sizeof(PxsDeformableVec3)+size_t(displacement.capacity())*sizeof(PxVec3)+
        size_t(bounds.capacity()+pointEnclosures.capacity()+displacementBounds.capacity())*sizeof(PxBounds3);}
    PxsDeformableMeshView view() const
    {return {bounds.begin(),order.begin(),leafBase,order.size(),ready};}
};

// One query may resume across bounded NP work packets. Overflow returns a
// continuation, never a truncated candidate domain. No heap, recursion, lock,
// atomic append or all-pairs table in the device-compatible query state.
struct PxsDeformableQuery
{
    enum { StackCapacity=32, VisitLimit=64 };
    PxU32 stack[StackCapacity],size=0,visits=0;
    bool valid=true;
    void start(PxU32 root=1){size=1;stack[0]=root;visits=0;valid=true;}
    bool done() const {return size==0;}
    PxU32 next(const PxsDeformableMeshView& mesh,const PxBounds3& query,PxU32* output,PxU32 capacity,
        const PxBounds3* displacementTree=NULL,const PxBounds3* queryMotion=NULL,PxReal marginSquared=0,
        const PxU32* subtreeMax=NULL,PxU32 ownFeature=0)
    {
        if(!mesh.ready || !query.isValid() || !output || !capacity){valid=false;return 0;}
        PxU32 count=0;
        for(PxU32 work=0;size && work<VisitLimit && count<capacity;++work)
        {
            const PxU32 node=stack[--size];++visits;
            if(subtreeMax && node<mesh.leafBase && subtreeMax[node]<=ownFeature)continue;
            if(!mesh.bounds[node].intersects(query))continue;
            if(queryMotion && PxsDeformableMesh::entirelyRestFiltered(*queryMotion,displacementTree[node],marginSquared))continue;
            if(node>=mesh.leafBase)
            {
                const PxU32 leaf=node-mesh.leafBase;
                if(leaf<mesh.count)output[count++]=mesh.order[leaf];
                continue;
            }
            if(size+2>StackCapacity){valid=false;return count;}
            stack[size++]=2*node+1;stack[size++]=2*node;
        }
        return count;
    }
};
PX_COMPILE_TIME_ASSERT(sizeof(PxsDeformableQuery)<=144);
}
#endif
