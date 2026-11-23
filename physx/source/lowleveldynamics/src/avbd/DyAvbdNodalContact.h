// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_NODAL_CONTACT_H
#define DY_AVBD_NODAL_CONTACT_H
#include "PxsDeformableContact.h"
#include "avbd/DyAvbdNodalShell.h"
#include "avbd/DyAvbdTetSeparation.h"
#include "foundation/PxUnionCast.h"
#include <algorithm>
#include <limits>
namespace physx { namespace Dy {
// Borrowed, immutable NP facts. State and multipliers remain in the common
// block/row arrays. No collision detection or second integration loop here.
struct AvbdNodalContact
{
    const PxsContactGeometry* geometry;
    const PxsDeformableContact* feature;
    PxU32 first[2]; // simulation block base, or rigid block (MAX: static)
    PxReal restDistance;
    PxU32 loadNodes[2]; // Dominant opposite signed entries of an inter-actor nodal Jacobian.
    PxReal loadWeights[2];
    PxU32 frictionSlot;
    bool separatedTets;
    // Immutable for this feature in this DCD epoch. Trial vertices are always read live.
    enum Projection { General, TrianglePair, StaticSide0, StaticSide1 };
    Projection projection=General;
    AvbdNodalVec3 axis=AvbdNodalVec3(0);
    AvbdNodalReal projectedRadius=0,staticMinimum=0,staticScale=0,startFloor=0;
    // Mixed solid/box force coordinates use the rigid DCD frame. A fixed
    // boundary or two deformable endpoints retain the world coordinate.
    PxU32 rigidFrame=PX_MAX_U32;
    // From assembly to the last trial of one colored single-node update.
    // That update is the sole writer of this row. Pair updates do not use it.
    struct LocalTet { AvbdNodalReal oppositeMinimum; PxReal weight,initialGap; } localTet={};
};

// Work-conjugate mixed-contact coordinate. Pull relative witness travel
// into the DCD frame, push a row axis into the current frame, and apply its
// rigid wrench at the current opposing material point.
PX_FORCE_INLINE PxVec3 avbdNodalMaterialVector(const PxQuat& start,const PxQuat& end,const PxVec3& travel)
{ return start.rotate(end.rotateInv(travel)); }
PX_FORCE_INLINE PxVec3 avbdNodalWorldAxis(const PxQuat& start,const PxQuat& end,const PxVec3& axis)
{ return end.rotate(start.rotateInv(axis)); }
PX_FORCE_INLINE PxVec3 avbdNodalWitnessArm(const PxQuat& end,const PxVec3& local,const PxVec3& travel,PxReal rigidSign)
{ return end.rotate(local)-travel*rigidSign; }

// Embeddings and GJK witnesses may carry signed barycentrics. The support
// span identifies the endpoint even when a coefficient is negative.
PX_FORCE_INLINE PxU32 avbdNodalPairSide(const AvbdNodalContact& contact,PxU32 block)
{
    return block-contact.first[0]<contact.geometry->deformable[0]->simulation.vertices ? 0u : 1u;
}

// Ordered max of (positive normal displacement, contact side). Each finite
// feature contributes at most four vertices with four embedding weights each.
// The winning row supplies the normal; the complete path certificate still
// admits or rejects the joint initial iterate before colored primal starts.
PX_FORCE_INLINE PxI64 avbdImpactVote(PxReal delta,PxU32 contactSide)
{
    const PxU32 tieBreak=PX_MAX_U32-contactSide;
    return PxI64((PxU64(PxUnionCast<PxU32>(PxF32(delta)))<<32)|PxU64(tieBreak));
}

template<class Travel,class Nominate> PX_INLINE void avbdNominateFixedImpact(
    const AvbdNodalContact& domain,PxU32 side,PxReal loadedGap,
    const Travel& travel,const Nominate& nominate)
{
    const auto& facts=*domain.geometry;
    const auto& feature=*domain.feature;
    const auto& mesh=facts.deformable[side]->state->mesh;
    const PxReal sign=side ? -1.f : 1.f;
    const bool vertex=(feature.vertexMask&(1u<<side))!=0;
    const PxReal clearance=domain.restDistance+facts.shapes[0].radius+facts.shapes[1].radius;
    const auto& obstacle=facts.shapes[1-side];
    const PxVec3 origin=obstacle.shapeToWorld.p;
    const PxReal extent=obstacle.kind==PxsContactShapeGeometry::eBOX ?
        obstacle.halfExtents.dot(obstacle.shapeToWorld.q.rotateInv(feature.normal).abs()) : 0.f;
    const PxReal offset=-clearance-extent;
    for(PxU32 v=0;v<(vertex?1u:mesh.arity);++v)
    {
        const PxU32 id=vertex ? feature.feature[side] : mesh.vertices[mesh.arity*feature.feature[side]+v];
        const auto& q=mesh.points[id];
        const PxVec3 start(PxReal(q.x),PxReal(q.y),PxReal(q.z));
        const PxReal gap=sign*feature.normal.dot(start-origin)+offset;
        const auto& embedded=mesh.supports[id];
        for(PxU32 k=0;k<embedded.count;++k)
        {
            if(embedded.weights[k]<=0)continue;
            const PxU32 block=domain.first[side]+embedded.nodes[k];
            const PxReal relative=sign*feature.normal.dot(travel(block));
            // Only an inward trajectory crossing the hard domain competes.
            // Interior/tangential inertia stays available to store material strain.
            if(relative < -PxMax(0.f,gap))
            {
                const PxReal target=-PxMax(0.f,gap-loadedGap);
                nominate(block,target-relative);
            }
        }
    }
}

// Project two approaching embedded vertices along the DCD separating normal.
// Each pair has scalar mobility J M^-1 J^T. Reduce overlapping proposals
// before publishing an initial guess. Each tet has at most four vertices,
// and each embedded vertex has at most four simulation supports.
template<class Particle,class Nominate> PX_INLINE void avbdNominatePairImpact(
    const AvbdNodalContact& contact,PxReal loadedGap,const Particle& particle,const Nominate& nominate)
{
    struct Vertex { PxU32 id;PxReal height,mobility,multiplier; };
    Vertex vertices[2][4]={};PxU32 count[2];
    const auto& facts=*contact.geometry;
    const auto& feature=*contact.feature;
    for(PxU32 side=0;side<2;++side)
    {
        const auto& mesh=facts.deformable[side]->state->mesh;
        const bool point=(feature.vertexMask&(1u<<side))!=0;
        count[side]=point?1:mesh.arity;
        for(PxU32 i=0;i<count[side];++i)
        {
            auto& v=vertices[side][i];
            v.id=point?feature.feature[side]:mesh.vertices[mesh.arity*feature.feature[side]+i];
            const auto start=mesh.points[v.id];
            PxVec3 offset=PxVec3(PxReal(start.x),PxReal(start.y),PxReal(start.z))-feature.point[0];
            const auto& embedding=mesh.supports[v.id];
            for(PxU32 k=0;k<embedding.count;++k)
            {
                const auto& p=particle(contact.first[side]+embedding.nodes[k]);
                const PxReal w=PxReal(embedding.weights[k]);
                offset+=(p.pHat-p.pStart)*w;
                v.mobility+=p.invMass*w*w;
            }
            v.height=feature.normal.dot(offset);
        }
    }
    const PxReal clearance=contact.restDistance+facts.shapes[0].radius+facts.shapes[1].radius;
    for(PxU32 a=0;a<count[0];++a)for(PxU32 b=0;b<count[1];++b)
    {
        auto& u=vertices[0][a];auto& v=vertices[1][b];
        const PxReal gap=u.height-v.height-clearance;
        if(gap>=0 || u.mobility+v.mobility<=0)continue;
        const PxReal multiplier=(loadedGap-gap)/(u.mobility+v.mobility);
        u.multiplier=PxMax(u.multiplier,multiplier);
        v.multiplier=PxMax(v.multiplier,multiplier);
    }
    for(PxU32 side=0;side<2;++side)for(PxU32 i=0;i<count[side];++i)
    {
        const auto& v=vertices[side][i];
        if(v.multiplier==0)continue;
        const auto& embedding=facts.deformable[side]->state->mesh.supports[v.id];
        for(PxU32 k=0;k<embedding.count;++k)
        {
            const PxU32 b=contact.first[side]+embedding.nodes[k];
            const PxReal delta=particle(b).invMass*PxReal(embedding.weights[k])*v.multiplier;
            if(delta!=0)nominate(b,side?-delta:delta);
        }
    }
}

struct AvbdNodalFriction
{
    const PxsContactGeometry* pair=NULL;
    PxU64 epoch=0,birthEpoch=0;
    PxU32 feature[2];
    PxU8 vertexMask;
    PxVec3 normal,traction;
    PxVec3 stickSlip=PxVec3(0);
};

// Read the last accepted physical step. Sampling an iterate neither ages nor
// updates the source. A new CM can reuse the same geometry pool address.
template<class Law> PX_INLINE bool avbdReadNodalFriction(const AvbdNodalFriction& old,
    const AvbdNodalFriction& current,bool carrySlip,Law& law)
{
    if(old.pair!=current.pair || old.birthEpoch!=current.birthEpoch || current.epoch<=old.epoch ||
       current.epoch-old.epoch!=1 || old.vertexMask!=current.vertexMask ||
       old.feature[0]!=current.feature[0] || old.feature[1]!=current.feature[1] ||
       old.normal.dot(current.normal)<=0) return false;
    if(carrySlip && old.normal==current.normal)
    {
        law.tangentC0=old.stickSlip.dot(law.t0);
        law.tangentC1=old.stickSlip.dot(law.t1);
    }
    law.lambdaT0=old.traction.dot(law.t0);
    law.lambdaT1=old.traction.dot(law.t1);
    return true;
}

// The primary table has eight probes. Overflow is dense, so a hash collision
// never resets a loaded contact. Only the previous, immutable tail is indexed.
struct AvbdFrictionKey
{
    PxU64 pair, features;
    PxU8 mask;

    explicit AvbdFrictionKey(const AvbdNodalFriction& f)
        : pair(reinterpret_cast<size_t>(f.pair)), features((PxU64(f.feature[0])<<32)|f.feature[1]), mask(f.vertexMask) {}
    bool operator<(const AvbdFrictionKey& rhs) const
    {
        if(pair!=rhs.pair)return pair<rhs.pair;
        if(features!=rhs.features)return features<rhs.features;
        return mask<rhs.mask;
    }
    PxU32 digit(PxU32 pass) const
    {
        if(!pass)return mask;
        const PxU64 word=pass<=8 ? features : pair;
        return PxU32(word>>(8*((pass-1)%8))) & 255;
    }
};

struct AvbdFrictionOverflow
{
    PxArray<PxU32> slots,scratch;
    void build(const PxArray<AvbdNodalFriction>& previous,PxU32 capacity)
    {
        slots.resize(previous.size()-capacity);
        for(PxU32 i=0;i<slots.size();++i)slots[i]=capacity+i;
        if(slots.size()<2)return;
        if(slots.size()<=32)
        {
            std::sort(slots.begin(),slots.end(),[&](PxU32 a,PxU32 b)
                {return AvbdFrictionKey(previous[a])<AvbdFrictionKey(previous[b]);});
            return;
        }
        // Larger tails use fixed-width count/scan/scatter, independent of hash clustering.
        scratch.resize(slots.size());
        for(PxU32 pass=0;pass<1+8+sizeof(size_t);++pass)
        {
            PxU32 offsets[256]={};
            for(PxU32 slot:slots)++offsets[AvbdFrictionKey(previous[slot]).digit(pass)];
            PxU32 prefix=0;
            for(PxU32& offset:offsets){const PxU32 count=offset;offset=prefix;prefix+=count;}
            for(PxU32 slot:slots)scratch[offsets[AvbdFrictionKey(previous[slot]).digit(pass)]++]=slot;
            slots.swap(scratch);
        }
    }
    PxU32 find(const PxArray<AvbdNodalFriction>& previous,const AvbdNodalFriction& feature) const
    {
        if(slots.empty())return PX_MAX_U32;
        const AvbdFrictionKey key(feature);
        const PxU32* end=slots.end();
        const PxU32* pos=std::lower_bound(slots.begin(),end,key,[&](PxU32 slot,const AvbdFrictionKey& k)
            {return AvbdFrictionKey(previous[slot])<k;});
        return pos!=end && !(key<AvbdFrictionKey(previous[*pos])) ? *pos : PX_MAX_U32;
    }
};

// Fixed-load derivative of the saturated Coulomb disk. Its radial (sliding)
// eigenvalue vanishes. Only the direction perpendicular to traction remains.
template<class T> PX_FORCE_INLINE PxMat33T<T> avbdNodalSlidingMetric(const PxVec3T<T>& j0,
    const PxVec3T<T>& j1,T f0,T f1,T limit,T scale,T rho)
{
    const PxVec3T<T> transverse=j1*(f0/limit)-j0*(f1/limit);
    return avbdNodalOuter(transverse,transverse)*(scale*rho);
}

PX_INLINE PxVec3 avbdNodalRigidVertex(const PxsContactShapeGeometry& shape,PxU32 i)
{
    switch(shape.kind)
    {
    case PxsContactShapeGeometry::eVERTICES:return shape.vertexToShape*shape.vertices[i];
    case PxsContactShapeGeometry::eBOX:return PxVec3((i&1)?shape.halfExtents.x:-shape.halfExtents.x,
        (i&2)?shape.halfExtents.y:-shape.halfExtents.y,(i&4)?shape.halfExtents.z:-shape.halfExtents.z);
    case PxsContactShapeGeometry::eCAPSULE:return PxVec3(i?shape.halfExtents.x:-shape.halfExtents.x,0,0);
    default:return PxVec3(0);
    }
}
template<class Node> PX_INLINE PxVec3 avbdNodalCollisionPoint(const AvbdNodalContact& c,PxU32 side,PxU32 vertex,const Node& node)
{
    const auto& mesh=c.geometry->deformable[side]->state->mesh;
    if(mesh.leafBase && !mesh.embeddingElements)return node(c.first[side]+vertex);
    const auto& support=mesh.supports[vertex];PxVec3 p(0);
    for(PxU32 k=0;k<support.count;++k)p+=node(c.first[side]+support.nodes[k])*PxReal(support.weights[k]);
    return p;
}

// Adjacent NP features can share collision vertices. Scratch is private to one
// CSR range and one candidate pose. Every feature still projects all corners
// on its own normal, including corners without a force weight on this node.
class AvbdNodalTrialVertices
{
    struct Entry { const void* view; PxU32 vertex, base; PxVec3 position; };
    Entry entries[32];
    PxU32 occupied=0;
public:
    template<class Node> PX_FORCE_INLINE PxVec3 read(const AvbdNodalContact& contact,
        PxU32 side,PxU32 vertex,const Node& node)
    {
        const auto* view=contact.geometry->deformable[side];
        const PxU32 base=contact.first[side],slot=(vertex+16*side)&31,bit=1u<<slot;
        Entry& e=entries[slot];
        if((occupied&bit) && e.vertex==vertex && e.view==view && e.base==base)return e.position;
        const PxVec3 p=avbdNodalCollisionPoint(contact,side,vertex,node);
        e.view=view;e.vertex=vertex;e.base=base;e.position=p;
        occupied|=bit;
        return p;
    }
};

// On two affine tetrahedra the minimum separating gap is concave in time.
// A nonnegative DCD gap is reusable for every trial in this epoch. A start
// within the roundoff interval still goes through the general certificate.
template<class Position> PX_INLINE bool avbdTetsSeparated(const AvbdNodalContact& c,const Position& position,
    PxU32 frozenSide=2,AvbdNodalReal* frozenMinimum=NULL,bool trial=false)
{
    const auto& f=*c.feature;
    const AvbdNodalVec3 origin(f.point[0].x,f.point[0].y,f.point[0].z),n(f.normal.x,f.normal.y,f.normal.z);
    AvbdNodalReal minima[2];
    for(PxU32 s=0;s<2;++s)
    {
        if(s==frozenSide && trial){minima[s]=*frozenMinimum;continue;}
        const auto& mesh=c.geometry->deformable[s]->state->mesh;
        const PxU32* ids=&mesh.vertices[4*f.feature[s]];
        if(!avbdTetSideMinimum(n,origin,s,[&](PxU32 i){return position(s,ids[i]);},minima[s]))return false;
        if(s==frozenSide)*frozenMinimum=minima[s];
    }
    return minima[0]+minima[1]>=c.restDistance+c.geometry->shapes[0].radius+c.geometry->shapes[1].radius;
}

// A rest-filtered self-contact witness can lie on two non-nearest features.
// Its force direction need not separate the complete triangles. A fixed axis
// separating both endpoints of their affine paths separates the entire step.
using AvbdTrianglePoints=AvbdNodalVec3[2][3];

template<class Position> PX_INLINE bool avbdReadTriangles(const AvbdNodalContact& c,
    const Position& position,AvbdTrianglePoints& points)
{
    const auto& origin=c.feature->point[0];
    const AvbdNodalVec3 base(origin.x,origin.y,origin.z);
    for(PxU32 side=0;side<2;++side)
    {
        const auto& mesh=c.geometry->deformable[side]->state->mesh;
        if(mesh.arity!=3 || c.feature->vertexMask)return false;
        const auto* ids=&mesh.vertices[3*c.feature->feature[side]];
        for(PxU32 i=0;i<3;++i)
        {
            const auto p=position(side,ids[i]);
            points[side][i]=AvbdNodalVec3(p.x,p.y,p.z)-base;
            if(!points[side][i].isFinite())return false;
        }
    }
    return true;
}

// Strict intervals exclude coplanar intersections and keep contact thickness.
// Sign identifies which side must remain ahead for the entire affine segment.
PX_INLINE int avbdTriangleOrder(const AvbdTrianglePoints& points,const AvbdNodalVec3& axis,AvbdNodalReal radius)
{
    using R=AvbdNodalReal;
    R low[2],high[2],scale=0;
    const auto absolute=axis.abs();
    for(PxU32 s=0;s<2;++s)
    {
        low[s]=std::numeric_limits<R>::max();high[s]=-low[s];
        for(PxU32 i=0;i<3;++i)
        {
            const R p=axis.dot(points[s][i]);
            low[s]=PxMin(low[s],p);high[s]=PxMax(high[s],p);
            scale=PxMax(scale,absolute.dot(points[s][i].abs()));
        }
    }
    const R bound=radius+R(16)*std::numeric_limits<R>::epsilon()*(scale+radius);
    if(!PxIsFinite(bound))return 0;
    if(low[0]-high[1]>bound)return 1;
    return low[1]-high[0]>bound?-1:0;
}

template<class Test> PX_INLINE bool avbdTriangleAxes(const AvbdTrianglePoints& start,const Test& test)
{
    AvbdNodalVec3 edge[2][3];
    for(PxU32 s=0;s<2;++s)
    {
        for(PxU32 i=0;i<3;++i)edge[s][i]=start[s][(i+1)%3]-start[s][i];
        const auto normal=edge[s][0].cross(edge[s][1]);
        if(test(normal))return true;
        for(PxU32 i=0;i<3;++i)if(test(normal.cross(edge[s][i])))return true;
    }
    for(PxU32 i=0;i<3;++i)for(PxU32 j=0;j<3;++j)
        if(test(edge[0][i].cross(edge[1][j])))return true;
    return false;
}

template<class Node> PX_INLINE bool avbdTrianglePairPath(const AvbdNodalContact& c,const Node& node,PxU32* visits)
{
    AvbdTrianglePoints start,end;
    if(!avbdReadTriangles(c,[&](PxU32 s,PxU32 id){return c.geometry->deformable[s]->state->mesh.points[id];},start) ||
       !avbdReadTriangles(c,[&](PxU32 s,PxU32 id){return avbdNodalCollisionPoint(c,s,id,node);},end))return false;
    if(visits)*visits+=6;
    const AvbdNodalReal radius=c.restDistance+c.geometry->shapes[0].radius+c.geometry->shapes[1].radius;
    return avbdTriangleAxes(start,[&](const AvbdNodalVec3& axis)
    {
        const auto clearance=radius*axis.magnitude();
        const int order=avbdTriangleOrder(start,axis,clearance);
        return order && order==avbdTriangleOrder(end,axis,clearance);
    });
}

// The box/plane contribution and the tolerated start depth belong to the DCD
// snapshot. Only the minimum over the moving feature changes between trials.
template<class Position> PX_INLINE bool avbdFixedSupportGap(const AvbdNodalContact& c,PxU32 side,
    const Position& position,AvbdNodalReal& gap,PxU32* visits)
{
    using R=AvbdNodalReal;
    const auto& mesh=c.geometry->deformable[side]->state->mesh;
    const bool vertex=(c.feature->vertexMask&(1u<<side))!=0;
    const auto& obstacle=c.geometry->shapes[1-side];
    const auto& p=obstacle.kind==PxsContactShapeGeometry::ePLANE?obstacle.shapeToWorld.p:c.feature->point[0];
    const AvbdNodalVec3 origin(p.x,p.y,p.z);
    R minimum=std::numeric_limits<R>::max(),scale=0;
    const PxU32 count=vertex?1:mesh.arity;
    const PxU32 first=c.feature->feature[side];
    for(PxU32 i=0;i<count;++i)
    {
        const auto x=position(side,vertex?first:mesh.vertices[count*first+i]);
        const auto h=c.axis.dot(AvbdNodalVec3(x.x,x.y,x.z)-origin);
        if(!PxIsFinite(h))return false;
        minimum=PxMin(minimum,h);scale=PxMax(scale,PxAbs(h));
    }
    if(visits)*visits+=count;
    gap=(minimum+c.staticMinimum)-c.projectedRadius;
    return PxIsFinite(gap) && gap>=-R(64)*std::numeric_limits<R>::epsilon()*(scale+c.staticScale+PxAbs(c.projectedRadius));
}

PX_INLINE void avbdPrepareNodalProjection(AvbdNodalContact& c)
{
    using V=AvbdNodalVec3;using R=AvbdNodalReal;
    const auto& facts=*c.geometry;
    c.projection=AvbdNodalContact::General;
    const auto old=[&](PxU32 s,PxU32 id){return facts.deformable[s]->state->mesh.points[id];};
    const R radius=c.restDistance+facts.shapes[0].radius+facts.shapes[1].radius;
    const auto& n=c.feature->normal;
    if(facts.deformable[0] && facts.deformable[1])
    {
        if(facts.deformable[0]->state->mesh.arity!=3 || facts.deformable[1]->state->mesh.arity!=3)return;
        AvbdTrianglePoints start;
        if(!avbdReadTriangles(c,old,start))return;
        const auto save=[&](const V& axis)
        {
            const R clearance=radius*axis.magnitude();
            const int order=avbdTriangleOrder(start,axis,clearance);
            if(!order)return false;
            c.axis=order>0?axis:-axis;c.projectedRadius=clearance;c.projection=AvbdNodalContact::TrianglePair;
            return true;
        };
        if(!save(V(n.x,n.y,n.z)))avbdTriangleAxes(start,save);
        return;
    }
    const PxU32 soft=facts.deformable[0]?0u:1u;
    const auto& shape=facts.shapes[1-soft];
    if(!facts.deformable[soft] || c.first[1-soft]!=PX_MAX_U32 ||
        (shape.kind!=PxsContactShapeGeometry::ePLANE && shape.kind!=PxsContactShapeGeometry::eBOX))return;
    c.axis=soft?-V(n.x,n.y,n.z):V(n.x,n.y,n.z);
    c.projectedRadius=radius;c.staticMinimum=0;c.staticScale=0;
    if(shape.kind==PxsContactShapeGeometry::eBOX)
    {
        const auto& q=shape.shapeToWorld.q;
        const auto local=PxQuatT<R>(q.x,q.y,q.z,q.w).rotateInv(c.axis);
        const auto h=local.multiply(V(shape.halfExtents.x,shape.halfExtents.y,shape.halfExtents.z)).abs();
        const auto& p=shape.shapeToWorld.p;const auto& o=c.feature->point[0];
        const R center=c.axis.dot(V(p.x,p.y,p.z)-V(o.x,o.y,o.z)),extent=h.x+h.y+h.z;
        c.staticMinimum=-center-extent;c.staticScale=PxAbs(center)+extent;
    }
    R gap;
    if(avbdFixedSupportGap(c,soft,old,gap,NULL))
    {
        c.startFloor=PxMin(R(0),gap);
        c.projection=soft?AvbdNodalContact::StaticSide1:AvbdNodalContact::StaticSide0;
    }
}

// The minimum of affine corner gaps is concave, so the chord between the
// two endpoint minima gives a conservative time to the fixed support plane.
template<class Node> PX_INLINE PxReal avbdSheetBoxPredictorFraction(const AvbdNodalContact& c,const Node& node)
{
    using R=AvbdNodalReal;
    if(c.projection!=AvbdNodalContact::StaticSide0 && c.projection!=AvbdNodalContact::StaticSide1)return 0;
    const PxU32 side=c.projection==AvbdNodalContact::StaticSide1?1:0;
    const auto& shape=c.geometry->shapes[1-side];
    const auto& mesh=c.geometry->deformable[side]->state->mesh;
    if(shape.kind!=PxsContactShapeGeometry::eBOX || mesh.arity!=3)return 0;
    R start=0,end=0;
    if(!avbdFixedSupportGap(c,side,[&](PxU32,PxU32 id){return mesh.points[id];},start,NULL))return 0;
    avbdFixedSupportGap(c,side,[&](PxU32 s,PxU32 id){return avbdNodalCollisionPoint(c,s,id,node);},end,NULL);
    if(!(start>0 && end<0))return 0;
    const R guard=R(32)*std::numeric_limits<PxReal>::epsilon()*(start+PxAbs(end)+shape.halfExtents.maxElement());
    return PxReal(PxMax(R(0),start-guard)/(start-end));
}

template<class Trial> PX_INLINE bool avbdTriangleFixedBoxPath(const AvbdNodalContact& c,PxU32 side,
    const Trial& trial,AvbdNodalReal radius,PxU32* visits)
{
    using V=AvbdNodalVec3;using R=AvbdNodalReal;using Q=PxQuatT<R>;
    const auto world=[](const PxVec3& p){return V(p.x,p.y,p.z);};
    const auto& mesh=c.geometry->deformable[side]->state->mesh;
    const auto& shape=c.geometry->shapes[1-side];
    // Every point of an affine moving triangle lies in the convex hull
    // of its six endpoint corners. A separating slab for all six is
    // a certificate for the entire path, even while sliding around a box edge.
    const auto& q=shape.shapeToWorld.q;
    const Q boxRotation(q.x,q.y,q.z,q.w);
    const V center=world(shape.shapeToWorld.p),half=world(shape.halfExtents);
    V x[2][3];
    for(PxU32 v=0;v<3;++v)
    {
        const auto id=mesh.vertices[3*c.feature->feature[side]+v];
        const auto& p=mesh.points[id];
        x[0][v]=boxRotation.rotateInv(V(p.x,p.y,p.z)-center);
        x[1][v]=boxRotation.rotateInv(world(trial(side,id))-center);
        if(!x[0][v].isFinite() || !x[1][v].isFinite())return false;
    }
    if(visits)*visits+=6;
    V low=x[0][0],high=low;
    for(const auto& pose:x)for(const auto& point:pose)
    {
        low=low.minimum(point);high=high.maximum(point);
    }
    const V scale=low.abs().maximum(high.abs());
    const R eps=R(32)*std::numeric_limits<R>::epsilon();
    for(PxU32 i=0;i<3;++i)
    {
        const R extent=half[i]+radius,bound=extent+eps*(scale[i]+extent);
        if(low[i]>bound || high[i]<-bound)return true;
    }
    const auto clear=[&](const V& direction)
    {
        const V absolute=direction.abs();
        const R extent=absolute.dot(half)+radius*direction.magnitude();
        const R bound=extent+eps*(absolute.dot(scale)+extent);
        const R first=direction.dot(x[0][0]);
        if(!PxIsFinite(bound) || !(PxAbs(first)>bound))return false;
        const V oriented=first>0?direction:-direction;
        for(const auto& pose:x)for(const auto& point:pose)
            if(!(oriented.dot(point)>bound))return false;
        return true;
    };
    const V basis[]={V(1,0,0),V(0,1,0),V(0,0,1)};
    for(const auto& pose:x)
    {
        if(clear((pose[1]-pose[0]).cross(pose[2]-pose[0])))return true;
        for(PxU32 v=0;v<3;++v)for(const auto& axis:basis)
            if(clear((pose[(v+1)%3]-pose[v]).cross(axis)))return true;
    }
    // A triangle can roll around a box corner while its changing face stays
    // separated. The six-point hull discards the shared time coordinate.
    // The face normal is quadratic in time and its plane offset is cubic.
    const V e0=x[0][1]-x[0][0],f0=x[0][2]-x[0][0];
    const V e1=x[1][1]-x[1][0],f1=x[1][2]-x[1][0];
    const V n0=e0.cross(f0),n1=(e0.cross(f1)+e1.cross(f0))*R(.5),n2=e1.cross(f1);
    const V normals[]={n0,(n0+n1*2)/3,(n1*2+n2)/3,n2};
    const R sign=n0.dot(x[0][0])>=0?R(1):R(-1);
    const V normalScale=n0.abs().maximum(n1.abs()).maximum(n2.abs());
    // The box support and the round skin are convex functions of the normal.
    // Bounding each Bernstein control bounds their weighted sum at every time.
    const R roundoff=eps*(normalScale.dot(scale+half)+radius*normalScale.magnitude());
    R bounds[4];
    for(PxU32 k=0;k<4;++k)
        bounds[k]=normals[k].abs().dot(half)+radius*normals[k].magnitude()+roundoff;
    for(PxU32 v=0;v<3;++v)
    {
        const V p=x[0][v],end=x[1][v];
        const R offsets[]={n0.dot(p),(n0.dot(end)+2*n1.dot(p))/3,(2*n1.dot(end)+n2.dot(p))/3,n2.dot(end)};
        for(PxU32 k=0;k<4;++k)
            if(!(sign*offsets[k]>bounds[k]))return false;
    }
    return true;
}

// A complete finite-feature certificate in the DCD separating axis. There
// is at most one rotating endpoint (the other is affine Particle3). Clearing
// its positive quaternion norm gives a cubic. Each Bernstein coefficient
// is minimized over ALL vertices, independently on each side: O(V0+V1),
// never a Cartesian feature product. No sampling, recursion or f64 mandate.
template<class Node,class Bodies> PX_INLINE bool avbdNodalContactPath(const AvbdNodalContact& c,
    const Node& node,const Bodies& bodies,PxU32* visits=NULL)
{
    const auto trial=[&](PxU32 side,PxU32 id){return avbdNodalCollisionPoint(c,side,id,node);};
    if(c.projection==AvbdNodalContact::TrianglePair)
    {
        AvbdTrianglePoints end;
        if(visits)*visits+=6;
        if(!avbdReadTriangles(c,trial,end))return false;
        if(avbdTriangleOrder(end,c.axis,c.projectedRadius)==1)return true;
    }
    else if(c.projection>=AvbdNodalContact::StaticSide0)
    {
        AvbdNodalReal gap;
        const PxU32 side=c.projection==AvbdNodalContact::StaticSide1?1u:0u;
        if(avbdFixedSupportGap(c,side,trial,gap,visits) && gap>=c.startFloor)return true;
    }
    if(c.separatedTets)
    {
        if(visits)*visits+=8;
        return avbdTetsSeparated(c,[&](PxU32 s,PxU32 v){return avbdNodalCollisionPoint(c,s,v,node);});
    }
    typedef AvbdNodalReal R;typedef AvbdNodalVec3 V;typedef PxQuatT<R> Q;
    const auto convert=[](const PxVec3& p){return V(p.x,p.y,p.z);};
    const auto rotation=[](const Q& q,const V& p)
    {const V u(q.x,q.y,q.z);return p*(q.w*q.w-u.dot(u))+u*(2*u.dot(p))+u.cross(p)*(2*q.w);};
    // Anchor a plane certificate on the physical plane. Subtracting two
    // distances from last frame's elevated witness can erase a small crossing.
    const auto& shapes=c.geometry->shapes;
    const V n=convert(c.feature->normal),origin=convert(shapes[0].kind==PxsContactShapeGeometry::ePLANE ?
        shapes[0].shapeToWorld.p : (shapes[1].kind==PxsContactShapeGeometry::ePLANE ?
        shapes[1].shapeToWorld.p : c.feature->point[0]));
    R D[3]={1,0,0};Q a(PxIdentity),b(PxIdentity),dq(0,0,0,0);
    PxU32 rigid=2;
    for(PxU32 s=0;s<2;++s)if(!c.geometry->deformable[s] && c.first[s]!=PX_MAX_U32)
    {
        rigid=s;const auto& shape=c.geometry->shapes[s];
        const PxQuat start=(shape.shapeToWorld*shape.shapeToBody.getInverse()).q;
        const auto end=bodies[c.first[s]].q;
        a=Q(start.x,start.y,start.z,start.w).getNormalized();b=Q(end.x,end.y,end.z,end.w).getNormalized();
        if(a.dot(b)<0)b=-b;dq=b-a;D[0]=a.dot(a);D[1]=2*a.dot(dq);D[2]=dq.dot(dq);
    }
    R lower[4]={},scale[4]={};
    for(PxU32 s=0;s<2;++s)
    {
        const auto* view=c.geometry->deformable[s];const auto& shape=c.geometry->shapes[s];
        const bool vertex=(c.feature->vertexMask&(1u<<s))!=0;
        const PxU32 count=view?(vertex?1:view->state->mesh.arity):PxMax(1u,shape.count);
        if(!count || count>256)return false;
        R minimum[4]={std::numeric_limits<R>::max(),std::numeric_limits<R>::max(),std::numeric_limits<R>::max(),std::numeric_limits<R>::max()};
        R maximum[4]={};const R sign=s?R(-1):R(1);
        if(!view && shape.kind==PxsContactShapeGeometry::eBOX)
        {
            // Each Bernstein coefficient is affine in local box coordinates.
            // Its extrema over all eight vertices are center +/- the sum of
            // three projected half extents. This also covers a rotating box.
            V axes[3]={V(0),V(0),V(0)};R center[4]={};
            if(s==rigid)
            {
                const PxTransform comStart=shape.shapeToWorld*shape.shapeToBody.getInverse();
                const R initial=n.dot(convert(comStart.p)-origin);
                const R travel=n.dot(convert(bodies[c.first[s]].p)-convert(comStart.p));
                const V constant=rotation(a.getConjugate(),n),quadratic=rotation(dq.getConjugate(),n);
                const V linear=rotation(b.getConjugate(),n)-constant-quadratic;
                const V vectors[3]={constant,linear,quadratic};
                const auto& q=shape.shapeToBody.q;const Q local(q.x,q.y,q.z,q.w);
                for(PxU32 k=0;k<3;++k)
                {
                    axes[k]=local.rotateInv(vectors[k]).multiply(convert(shape.halfExtents));
                    center[k]+=initial*D[k]+vectors[k].dot(convert(shape.shapeToBody.p));
                    center[k+1]+=travel*D[k];
                }
            }
            else
            {
                const auto& q=shape.shapeToWorld.q;
                axes[0]=Q(q.x,q.y,q.z,q.w).rotateInv(n).multiply(convert(shape.halfExtents));
                center[0]=n.dot(convert(shape.shapeToWorld.p)-origin);
            }
            const V radii[4]={axes[0],axes[0]+axes[1]/3,axes[0]+(axes[1]*R(2)+axes[2])/3,axes[0]+axes[1]+axes[2]};
            const R centers[4]={center[0],center[0]+center[1]/3,center[0]+(2*center[1]+center[2])/3,
                center[0]+center[1]+center[2]+center[3]};
            for(PxU32 k=0;k<4;++k)
            {
                const auto h=radii[k].abs();const R extent=h.x+h.y+h.z;
                lower[k]+=sign*centers[k]-extent;scale[k]+=PxAbs(centers[k])+extent;
            }
            if(visits)*visits+=3;
            continue;
        }
        for(PxU32 i=0;i<count;++i)
        {
            if(visits)++*visits;
            V p0,p1;R rot[3]={};
            if(view)
            {
                const auto& mesh=view->state->mesh;
                const PxU32 id=vertex?c.feature->feature[s]:mesh.vertices[mesh.arity*c.feature->feature[s]+i];
                const auto& old=mesh.points[id];p0=V(old.x,old.y,old.z)-origin;
                p1=convert(avbdNodalCollisionPoint(c,s,id,node))-origin;
            }
            else if(s==rigid)
            {
                const V local=convert(shape.shapeToBody.transform(avbdNodalRigidVertex(shape,i)));
                p0=convert((shape.shapeToWorld*shape.shapeToBody.getInverse()).p)-origin;
                p1=convert(bodies[c.first[s]].p)-origin;
                rot[0]=n.dot(rotation(a,local));rot[2]=n.dot(rotation(dq,local));
                rot[1]=n.dot(rotation(b,local))-rot[0]-rot[2];
            }
            else p0=p1=convert(shape.shapeToWorld.transform(avbdNodalRigidVertex(shape,i)))-origin;
            if(rigid==2)
            {
                // For two affine features the separating gap is concave on
                // the segment. Both endpoints bound every intermediate pose.
                const R initial=n.dot(p0),end=n.dot(p1);
                minimum[0]=PxMin(minimum[0],sign*initial);minimum[3]=PxMin(minimum[3],sign*end);
                maximum[0]=PxMax(maximum[0],PxAbs(initial));maximum[3]=PxMax(maximum[3],PxAbs(end));
                continue;
            }
            R power[4]={};const R initial=n.dot(p0),travel=n.dot(p1-p0);
            for(PxU32 k=0;k<3;++k){power[k]+=initial*D[k]+rot[k];power[k+1]+=travel*D[k];}
            const R bernstein[4]={power[0],power[0]+power[1]/3,
                power[0]+2*power[1]/3+power[2]/3,power[0]+power[1]+power[2]+power[3]};
            for(PxU32 k=0;k<4;++k)
            {minimum[k]=PxMin(minimum[k],sign*bernstein[k]);maximum[k]=PxMax(maximum[k],PxAbs(bernstein[k]));}
        }
        for(PxU32 k=0;k<4;k+=rigid==2?3u:1u){lower[k]+=minimum[k];scale[k]+=maximum[k];}
    }
    const R radius=c.restDistance+c.geometry->shapes[0].radius+c.geometry->shapes[1].radius;
    const R denominator[4]={D[0],D[0]+D[1]/3,D[0]+2*D[1]/3+D[2]/3,D[0]+D[1]+D[2]};
    // The DCD roundoff interval admits a touching input; it is not a
    // reusable inward displacement for every new pose or DCD epoch.
    const R admittedDepth=PxMin(R(0),lower[0]/denominator[0]-radius);
    for(PxU32 k=0;k<4;k+=rigid==2?3u:1u)
    {
        const R gap=lower[k]-radius*denominator[k];
        const R tolerance=64*std::numeric_limits<R>::epsilon()*(scale[k]+PxAbs(radius*denominator[k]));
        if(!PxIsFinite(gap))return false;
        if(gap < -tolerance || (k==3 && gap<admittedDepth*denominator[k]))
        {
            const PxU32 obstacle=c.geometry->deformable[0]?1u:0u;
            if(c.geometry->deformable[obstacle])return avbdTrianglePairPath(c,node,visits);
            const auto& shape=shapes[obstacle];
            if(shape.kind!=PxsContactShapeGeometry::eBOX && shape.kind!=PxsContactShapeGeometry::eVERTICES)return false;
            const bool moving=rigid!=2;
            const PxTransform start=shape.shapeToWorld*shape.shapeToBody.getInverse();
            if(!moving)a=b=Q(start.q.x,start.q.y,start.q.z,start.q.w).getNormalized();
            const V rigidEnd=moving?convert(bodies[c.first[obstacle]].p):convert(start.p);
            // The NP separating plane can travel with the rigid feature.
            // In that frame its support value is constant and every affine
            // particle path dotted with the rotating normal is cubic / |q|².
            const auto& orientation=shape.shapeToBody.q;
            const Q toBody(orientation.x,orientation.y,orientation.z,orientation.w);
            const V initial=a.rotateInv(obstacle==1?n:-n).getNormalized();
            const auto separates=[&](const V& local)
            {
                const V normalShape=toBody.rotateInv(local);
                R extent=0;
                if(shape.kind==PxsContactShapeGeometry::eBOX)
                    extent=normalShape.abs().dot(convert(shape.halfExtents));
                else
                {
                    if(!shape.count || shape.count>256)return false;
                    extent=-std::numeric_limits<R>::max();
                    for(PxU32 i=0;i<shape.count;++i)
                        extent=PxMax(extent,normalShape.dot(convert(avbdNodalRigidVertex(shape,i))));
                }
                if(visits)*visits+=shape.kind==PxsContactShapeGeometry::eBOX?3:shape.count;
                extent+=local.dot(convert(shape.shapeToBody.p))+radius;
                const V n0=rotation(a,local),n2=rotation(dq,local),n1=rotation(b,local)-n0-n2;
                const auto side=1-obstacle;
                const auto& mesh=c.geometry->deformable[side]->state->mesh;
                const bool vertex=(c.feature->vertexMask&(1u<<side))!=0;
                for(PxU32 i=0;i<(vertex?1u:mesh.arity);++i)
                {
                    const PxU32 id=vertex?c.feature->feature[side]:mesh.vertices[mesh.arity*c.feature->feature[side]+i];
                    const auto old=mesh.points[id];
                    const V x=V(old.x,old.y,old.z)-convert(start.p);
                    const V y=convert(avbdNodalCollisionPoint(c,side,id,node))-rigidEnd;
                    const V v=y-x;
                    const R p0=x.dot(n0)-extent*D[0],p1=v.dot(n0)+x.dot(n1)-extent*D[1];
                    const R p2=v.dot(n1)+x.dot(n2)-extent*D[2];
                    const R bounds[]={p0,p0+p1/3,p0+(2*p1+p2)/3,y.dot(rotation(b,local))-extent*b.dot(b)};
                    // Equality is sufficient. A failed alternative simply leaves
                    // the original candidate unproven; never spend roundoff as clearance.
                    for(R bound:bounds)if(!(bound>=0) || !PxIsFinite(bound))return false;
                    if(visits)++*visits;
                }
                return true;
            };
            if(moving && dq.magnitudeSquared()>0 && separates(initial))return true;
            const PxU32 soft=1-obstacle;
            if(!moving && shape.kind==PxsContactShapeGeometry::eBOX &&
                c.geometry->deformable[soft]->state->mesh.arity==3 && !c.feature->vertexMask)
                return avbdTriangleFixedBoxPath(c,soft,trial,radius,visits);
            if(c.geometry->deformable[1-obstacle]->simulation.arity!=4)return false;
            // GJK can report an edge axis even though a box face separates
            // the complete path. Test the three shape axes in its own frame.
            const V inShape=toBody.rotateInv(initial);
            for(PxU32 j=0;j<3;++j)
            {
                V face(0);face[j]=inShape[j]>=0?R(1):R(-1);
                if(separates(toBody.rotate(face)))return true;
            }
            // When a tet rolls around a fixed box edge its own face can
            // remain separating even though every fixed box face fails.
            // Affine vertex paths give a quadratic area normal. Its dot
            // product with a vertex is cubic, so four control values
            // certify the entire interval, including the opposite vertex.
            const PxU32 side=1-obstacle;
            const auto& mesh=c.geometry->deformable[side]->state->mesh;
            if(moving || shape.kind!=PxsContactShapeGeometry::eBOX || mesh.arity!=4 ||
                (c.feature->vertexMask&(1u<<side)))return false;
            V x[2][4];
            const V center=convert(shape.shapeToWorld.p),half=convert(shape.halfExtents);
            const auto& shapeRotation=shape.shapeToWorld.q;
            const Q orient(shapeRotation.x,shapeRotation.y,shapeRotation.z,shapeRotation.w);
            for(PxU32 v=0;v<4;++v)
            {
                const auto id=mesh.vertices[4*c.feature->feature[side]+v];
                const auto& old=mesh.points[id];
                x[0][v]=orient.rotateInv(V(old.x,old.y,old.z)-center);
                x[1][v]=orient.rotateInv(convert(avbdNodalCollisionPoint(c,side,id,node))-center);
            }
            for(PxU32 v=0;v<4;++v)
            {
                const PxU32 i=(v+1)%4,j=(v+2)%4,last=(v+3)%4;
                const V e0=x[0][j]-x[0][i],f0=x[0][last]-x[0][i],e1=x[1][j]-x[1][i],f1=x[1][last]-x[1][i];
                V n0=e0.cross(f0),n1=(e0.cross(f1)+e1.cross(f0))*R(.5),n2=e1.cross(f1);
                if(n0.dot(x[0][v]-x[0][i])<0){n0=-n0;n1=-n1;n2=-n2;}
                const V axes[]={n0,(n0+n1*R(2))/3,(n1*R(2)+n2)/3,n2};
                const auto controls=[&](const V& p0,const V& p1,R (&out)[4])
                {
                    out[0]=n0.dot(p0);out[1]=(n0.dot(p1)+2*n1.dot(p0))/3;
                    out[2]=(2*n1.dot(p1)+n2.dot(p0))/3;out[3]=n2.dot(p1);
                };
                R plane[4],interior[4];controls(x[0][i],x[1][i],plane);
                controls(x[0][v]-x[0][i],x[1][v]-x[1][i],interior);
                bool clear=true;
                for(PxU32 t=0;t<4 && clear;++t)
                {
                    const R support=axes[t].abs().dot(half)+(radius>0?radius*axes[t].magnitude():0);
                    clear=interior[t]>0 && plane[t]>=support && PxIsFinite(plane[t]) && PxIsFinite(interior[t]);
                }
                if(visits)*visits+=4;
                if(clear)return true;
            }
            return false;
        }
    }
    return true;
}
} }
#endif
