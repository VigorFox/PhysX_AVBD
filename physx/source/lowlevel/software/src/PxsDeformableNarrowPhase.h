// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Included by the CPU NP implementation, after its ordinary CM task. These
// packets prepare the same NP epoch/continuation; there is no solver here.
#ifndef PXS_DEFORMABLE_NARROW_PHASE_H
#define PXS_DEFORMABLE_NARROW_PHASE_H
#if !PX_SUPPORT_GPU_PHYSX
#include "PxsDeformableContact.h"
#include "PxsDeformableTriangleFilter.h"
#include "GuDistancePointTriangle.h"
#include "GuDistanceSegmentSegment.h"
#include "PxsMaterialManager.h"
#include "PxsMaterialCombiner.h"
#include "foundation/PxAtomic.h"
#include "foundation/PxTime.h"
#include <cstdio>
#include "task/PxCpuDispatcher.h"

namespace
{
struct PxsNpFiniteSupport : PxGjkQuery::Support
{
    const PxsContactShapeGeometry& shape;
    PxVec3 triangle[4];PxU32 count=0;bool nodal;
    PxsNpFiniteSupport(const PxsContactShapeGeometry& s):shape(s),nodal(false){}
    PxReal getMargin() const PX_OVERRIDE {return nodal?0:shape.radius;}
    PxVec3 supportLocal(const PxVec3& direction) const PX_OVERRIDE
    {
        if(!nodal)return ContactGeometrySupport(shape).supportLocal(direction);
        PxU32 best=0;for(PxU32 k=1;k<count;++k)if(triangle[k].dot(direction)>triangle[best].dot(direction))best=k;
        return triangle[best];
    }
};

class PxsDeformableNpBatch : public PxUserAllocated
{
    enum Phase { Sample,Leaves,Tree,QueryInit,QueryCount,QueryScan,QueryFill,Contact,ContactScan,Compact,Done };
    enum { Packet=64,ScanWidth=256,PacketsPerTask=16 };
    struct Pair
    {
        PxsContactGeometry* facts;PxsDeformableContactOutput* output;
        PxU32 soft,treeSide,roots,queries;
        PxReal reach;bool self,plane,enabled;
        PxReal marginSquared;
    };
    static bool simplexBox(const Pair& pair)
    {
        return !pair.facts->deformable[1-pair.soft] &&
            pair.facts->shapes[1-pair.soft].kind==PxsContactShapeGeometry::eBOX;
    }
    struct Range {PxU32 object,first,jobs,items,width;};
    class Task : public Cm::Task
    {
        PxsDeformableNpBatch* batch;
        PxU32 first,end;
        bool unfinished=false;
    public:
        Task(PxsDeformableNpBatch& b,PxU32 from,PxU32 to)
            :Cm::Task(b.context.getContextId()),batch(&b),first(from),end(to){}
        const char* getName() const PX_OVERRIDE{return "PxsContext.narrowPhasePacket";}
        void runInternal() PX_OVERRIDE
        {
            unfinished=batch->executeRange(first,end);
        }
        void release() PX_OVERRIDE
        {
            PxsDeformableNpBatch* b=batch;PxBaseTask* nextTask=mCont;
            const PxU32 from=first,to=end;const bool resume=unfinished;this->~Task();
            if(resume)b->resumeRange(from,to);
            else if(!b->submitNextRange() && PxAtomicDecrement(&b->pending)==0)b->workerFinished();
            if(nextTask)nextTask->removeReference();
        }
    };
    typedef void (*Finish)(PxsContext&,PxsContactManagers&,PxArray<PxsCMDiscreteUpdateTask*>*,PxReal,
        PxsContactManagerOutput*,PxBaseTask*,PxContactModifyCallback*);
    PxsContext& context;PxsContactManagers& managers;PxsContactManagerOutput* cmOutputs;
    PxBaseTask* continuation;PxContactModifyCallback* modify;Finish finish;
    PxReal dt,tolerance;
    PxArray<PxsDeformableGeometry*> shapes;
    PxArray<PxU32> treeSource,treeRoots,treeLevels;
    PxArray<Pair> pairs;PxArray<Range> ranges;
    Phase phase=Sample;
    PxU32 cursor=0;
    volatile PxI32 pending=0,failed=0,continuedTasks=0;
    volatile PxI64 nextPacket=0;
    PxU32 packetWidth=0;
    PxU32 jobs=0,tasksSubmitted=0,waves=0;
    bool profile=false,prepared=false;PxU64 started=0;volatile PxI32 active=0,peak=0,peakPackets=0;

    bool executeRange(PxU32 first,PxU32 end)
    {
        PX_ASSERT(end-first<=PacketsPerTask);
        PxU32 spare=PacketsPerTask-(end-first);bool unfinished=false;
        if(profile)PxAtomicMax(&peak,PxAtomicIncrement(&active));
        for(PxU32 job=first;job<end;++job)execute(job,spare,unfinished);
        if(profile){PxAtomicMax(&peakPackets,PxI32(PacketsPerTask-spare));PxAtomicDecrement(&active);}
        return unfinished && !failed;
    }

    void range(PxU32 object,PxU32 items,PxU32 width)
    {
        if(!items)return;const PxU32 count=(items+width-1)/width;
        ranges.pushBack({object,jobs,count,items,width});jobs+=count;
    }
    void addShape(const PxsDeformableGeometry* view)
    {
        if(!view)return;
        for(auto* old:shapes)if(old==view)return;
        shapes.pushBack(const_cast<PxsDeformableGeometry*>(view));
        if(!view->state || !view->state->mesh.matches(*view)){failed=1;return;}
    }
    static PxVec3 point(const PxsDeformableMesh& mesh,PxU32 vertex)
    {const auto& p=mesh.points[vertex];return PxVec3(PxReal(p.x),PxReal(p.y),PxReal(p.z));}
    static PxBounds3 expanded(PxBounds3 b,PxReal distance)
    {
        b.fattenFast(distance);
        for(PxU32 d=0;d<3;++d)
        {b.minimum[d]=std::nextafter(b.minimum[d],-PX_MAX_F32);b.maximum[d]=std::nextafter(b.maximum[d],PX_MAX_F32);}
        return b;
    }
    static PxBounds3 rigidBounds(const PxsContactShapeGeometry& shape)
    {
        PxBounds3 result=PxBounds3::empty();
        if(shape.kind==PxsContactShapeGeometry::eNONE || shape.kind==PxsContactShapeGeometry::ePLANE)return result;
        const ContactGeometrySupport support(shape);
        for(PxU32 d=0;d<3;++d)
        {
            PxVec3 axis(0);axis[d]=1;
            const PxVec3 local=shape.shapeToWorld.q.rotateInv(axis);
            result.minimum[d]=shape.shapeToWorld.transform(support.supportLocal(-local))[d]-shape.radius;
            result.maximum[d]=shape.shapeToWorld.transform(support.supportLocal(local))[d]+shape.radius;
        }
        return result;
    }
    bool retain(const Pair& pair,PxU32 a,PxU32 b) const
    {
        if(!pair.self)return true;
        if(b<=a)return false;
        const auto& view=*pair.facts->deformable[0];
        const auto& mesh=view.state->mesh;
        const auto* first=&mesh.vertices[mesh.arity*a];
        const auto* second=&mesh.vertices[mesh.arity*b];
        for(PxU32 i=0;i<mesh.arity;++i)for(PxU32 j=0;j<mesh.arity;++j)
            if(first[i]==second[j])return false;
        if(mesh.arity==3 && !mesh.displacement.empty())
        {
            PxVec3 live[2][3],rest[2][3];
            for(PxU32 i=0;i<3;++i)
            {
                live[0][i]=point(mesh,first[i]);live[1][i]=point(mesh,second[i]);
                rest[0][i]=(*view.restPositions)[first[i]].getXYZ();
                rest[1][i]=(*view.restPositions)[second[i]].getXYZ();
            }
            if(pxsTriangleRestFiltered(live,rest,pair.reach,*view.selfFilterDistance))return false;
        }
        return true;
    }
    void initializePair(Pair& pair)
    {
        auto& out=*pair.output;out.valid=false;out.count=0;out.contacts.clear();
        if(!pair.enabled){pair.queries=0;out.candidates.clear();return;}
        const auto& view=*pair.facts->deformable[pair.soft];
        const auto& mesh=view.state->mesh;
        if(pair.self && view.selfFilterDistance && (!PxIsFinite(*view.selfFilterDistance) ||
            (*view.selfFilterDistance!=0 && (!view.restPositions || !*view.restPositions))))
        {failed=1;pair.enabled=false;pair.queries=0;out.candidates.clear();return;}
        if(pair.plane)
        {pair.queries=0;out.candidates.resize(mesh.uniqueVertices.size());return;}
        const bool two=pair.facts->deformable[1-pair.soft]!=NULL;
        pair.treeSide=two?1-pair.soft:pair.soft;
        const auto& tree=pair.facts->deformable[pair.treeSide]->state->mesh;
        const PxBounds3 query=expanded(two?mesh.bounds[1]:rigidBounds(pair.facts->shapes[1-pair.soft]),pair.reach);
        if(pair.self && !mesh.displacementBounds.empty())
            pair.marginSquared=mesh.restFilterMarginSquared(mesh.displacementBounds[1],*view.selfFilterDistance-pair.reach);
        if(!tree.bounds[1].intersects(query) || (pair.self && mesh.displacementBounds.size() &&
            mesh.entirelyRestFiltered(mesh.displacementBounds[1],mesh.displacementBounds[1],pair.marginSquared)))
        {
            pair.enabled=false;pair.queries=0;out.candidates.clear();out.queries.clear();return;
        }
        pair.roots=1;
        if(!two)while(pair.roots<64 && pair.roots<tree.leafBase)pair.roots*=2;
        pair.queries=two?mesh.order.size():pair.roots;
        out.queries.resize(pair.queries);out.totals.resize((pair.queries+ScanWidth-1)/ScanWidth);
    }
    void scanPairs(bool contacts)
    {
        for(auto& pair:pairs)
        {
            auto& out=*pair.output;if(!pair.enabled)continue;
            if(!contacts && pair.plane)continue;
            PxU64 sum=0;
            for(PxU32 t=0;t<out.totals.size();++t)
            {const PxU32 count=out.totals[t];out.totals[t]=PxU32(sum);sum+=count;}
            if(sum>PX_MAX_U32){failed=1;continue;}
            if(contacts){out.count=PxU32(sum);out.contacts.resize(out.count);}
            else out.candidates.resize(PxU32(sum));
        }
    }
    void prepare()
    {
        ranges.clear();jobs=0;cursor=0;
        if(phase<=Tree)
        {
            for(PxU32 s=0;s<shapes.size();++s)
            {
                auto& state=*shapes[s]->state;
                if(state.epoch==context.mCpuDeformableEpoch)continue;
                auto& mesh=state.mesh;
                if(phase==Sample){mesh.ready=false;mesh.prepareDisplacements(*shapes[s]);range(s,mesh.points.size(),Packet);}
                else if(phase==Leaves)range(s,mesh.leafBase,Packet);
                else if(treeSource[s]>1)
                {
                    PxU32 roots=treeSource[s],levels=0;
                    while(roots>1 && levels<6){roots/=2;++levels;}
                    treeRoots[s]=roots;treeLevels[s]=levels;range(s,roots,1);
                }
            }
            return;
        }
        if(phase==QueryInit)for(auto& pair:pairs)initializePair(pair);
        for(PxU32 p=0;p<pairs.size();++p)
        {
            auto& pair=pairs[p];auto& out=*pair.output;if(!pair.enabled)continue;
            switch(phase)
            {
            case QueryInit:range(p,pair.queries,Packet);break;
            case QueryCount:case QueryFill:
                range(p,pair.plane?(phase==QueryFill?out.candidates.size():0):pair.queries,pair.plane?Packet:1);break;
            case QueryScan:if(!pair.plane)range(p,pair.queries,ScanWidth);break;
            case Contact:
            {
                const PxU32 lanes=simplexBox(pair)?1u+pair.facts->deformable[pair.soft]->state->mesh.arity:1u;
                if(out.candidates.size()>PX_MAX_U32/lanes){failed=1;return;}
                const PxU32 count=out.candidates.size()*lanes;
                out.rawContacts.resize(count);out.active.resize(count);out.offsets.resize(count);
                out.totals.resize((count+ScanWidth-1)/ScanWidth);
                range(p,count,4);break;
            }
            case ContactScan:range(p,out.active.size(),ScanWidth);break;
            case Compact:range(p,out.active.size(),Packet);break;
            default:break;
            }
        }
    }
    // Find the closest *eligible* material features. Filtering just the first
    // GJK witness can hide a second, distant-in-rest edge on the same pair.
    static bool filteredWitness(const PxVec3 (&live)[2][4],const PxVec3 (&rest)[2][4],
        PxU32 count,PxReal distance,PxReal reach,PxVec3 (&points)[2])
    {
        PxReal best=reach*reach;const PxReal threshold=distance*distance;
        PxU32 near[4]={};
        if(count==3)
        {
            const PxReal inner=threshold*(1.f-32*PX_EPS_F32);
            for(PxU32 i=0;i<3;++i)for(PxU32 j=0;j<3;++j)
                if((rest[0][i]-rest[1][j]).magnitudeSquared()<inner)near[i]|=1u<<j;
        }
        bool found=false;
        const auto accept=[&](PxU32 side,const PxVec3& a,const PxVec3& b,PxReal restSquared,PxReal current)
        {
            if(current<=best && restSquared>threshold)
            {best=current;points[side]=a;points[1-side]=b;found=true;}
        };
        for(PxU32 side=0;side<2;++side)for(PxU32 i=0;i<count;++i)
        {
            const auto& other=live[1-side];const auto& otherRest=rest[1-side];
            const auto& p=live[side][i];const auto& q=rest[side][i];
            if(count==3)
            {
                const PxU32 close=side?((near[0]|near[1]|near[2])&(1u<<i)):near[i];
                if(close)continue;
                PxReal u,v;
                const auto y=Gu::closestPtPointTriangle(p,other[0],other[1],other[2],u,v);
                const PxReal sq=(p-y).magnitudeSquared();
                if(sq<=best)
                {
                    const auto z=Gu::closestPtPointTriangle(q,otherRest[0],otherRest[1],otherRest[2],u,v);
                    accept(side,p,y,(q-z).magnitudeSquared(),sq);
                }
                continue;
            }
            PxReal nearest=PX_MAX_F32,restNearest=PX_MAX_F32;PxVec3 closest(0);
            bool inside=count==4,liveInside=count==4;
            for(PxU32 face=0;face<(count==4?4u:1u);++face)
            {
                const PxU32 a=face,b=(face+1)%count,c=(face+2)%count;
                PxReal u,v;
                const auto y=Gu::closestPtPointTriangle(p,other[a],other[b],other[c],u,v);
                const auto z=Gu::closestPtPointTriangle(q,otherRest[a],otherRest[b],otherRest[c],u,v);
                const PxReal sq=(p-y).magnitudeSquared();
                if(sq<nearest){nearest=sq;closest=y;}
                restNearest=PxMin(restNearest,(q-z).magnitudeSquared());
                if(inside)
                {
                    const auto n=(otherRest[b]-otherRest[a]).cross(otherRest[c]-otherRest[a]);
                    inside=n.dot(q-otherRest[a])*n.dot(otherRest[(face+3)%4]-otherRest[a])>=0;
                }
                if(liveInside)
                {
                    const auto n=(other[b]-other[a]).cross(other[c]-other[a]);
                    liveInside=n.dot(p-other[a])*n.dot(other[(face+3)%4]-other[a])>=0;
                }
            }
            if(!inside)accept(side,p,closest,restNearest,liveInside?0:nearest);
        }
        for(PxU32 i=0;i<count;++i)for(PxU32 j=i+1;j<count;++j)
        for(PxU32 k=0;k<count;++k)for(PxU32 l=k+1;l<count;++l)
        {
            if((near[i]|near[j])&((1u<<k)|(1u<<l)))continue;
            const PxVec3 a=live[0][j]-live[0][i],b=live[1][l]-live[1][k];
            PxReal s,t;
            const PxReal sq=Gu::distanceSegmentSegmentSquared(live[0][i],a,live[1][k],b,&s,&t);
            if(sq>best || s<=0 || s>=1 || t<=0 || t>=1)continue; // Endpoint features use the vertex tests.
            const PxReal restSq=Gu::distanceSegmentSegmentSquared(rest[0][i],rest[0][j]-rest[0][i],
                rest[1][k],rest[1][l]-rest[1][k]);
            accept(0,live[0][i]+a*s,live[1][k]+b*t,restSq,sq);
        }
        return found;
    }

    void feature(Pair& pair,PxU32 index)
    {
        auto& out=*pair.output;out.active[index]=0;
        const bool patch=simplexBox(pair);
        const PxU32 lanes=patch?1u+pair.facts->deformable[pair.soft]->state->mesh.arity:1u;
        const auto& candidate=out.candidates[index/lanes];auto& contact=out.rawContacts[index];
        if(patch)
        {
            const PxU32 side=pair.soft,other=1-side,element=candidate.feature[side];
            const auto& mesh=pair.facts->deformable[side]->state->mesh;
            const auto& shape=pair.facts->shapes[other];
            const auto faceOf=[&](const PxVec3& x)
            {
                for(PxU32 d=0;d<3;++d)
                {
                    const PxU32 u=(d+1)%3,v=(d+2)%3;const PxReal gap=PxAbs(x[d])-shape.halfExtents[d];
                    // An embedded solid vertex on a box edge has a finite-feature
                    // witness. Use face rows only where the entire contact skin
                    // projects inside the face; edge/corner rows remain with GJK.
                    const PxReal inset=mesh.arity==4?pair.facts->contactDistance:0.f;
                    if(gap>=0 && gap<=pair.reach && PxAbs(x[u])+inset<=shape.halfExtents[u] &&
                        PxAbs(x[v])+inset<=shape.halfExtents[v])
                        return 2*d+PxU32(x[d]>=0);
                }
                return PX_MAX_U32;
            };
            const PxU32 lane=index%lanes;
            if(lane)
            {
                const PxU32 id=mesh.vertices[mesh.arity*element+lane-1];
                if(mesh.firstIncidentElement[id]!=element)return;
                const PxVec3 world=point(mesh,id),local=shape.shapeToWorld.transformInv(world);
                const PxU32 face=faceOf(local);
                if(face==PX_MAX_U32)return;
                const PxU32 axis=face/2;PxVec3 n(0);n[axis]=(face&1)?1.f:-1.f;
                const PxReal gap=PxAbs(local[axis])-shape.halfExtents[axis];
                const PxVec3 outward=shape.shapeToWorld.q.rotate(n);
                contact.normal=side==0?outward:-outward;contact.separation=gap;
                contact.point[side]=world;contact.point[other]=world-outward*gap;
                contact.weights[side]=PxVec4(1,0,0,0);contact.weights[other]=PxVec4(0.f);
                contact.feature[side]=id;contact.feature[other]=face;contact.vertexMask=PxU8(1u<<side);
                out.active[index]=1;return;
            }
            // One face halfspace at every vertex certifies the full affine
            // simplex. Keep its finite feature when face coverage is incomplete.
            const PxU32* ids=mesh.vertices.begin()+mesh.arity*element;
            const PxU32 face=faceOf(shape.shapeToWorld.transformInv(point(mesh,ids[0])));
            if(face!=PX_MAX_U32 && faceOf(shape.shapeToWorld.transformInv(point(mesh,ids[1])))==face &&
                faceOf(shape.shapeToWorld.transformInv(point(mesh,ids[2])))==face &&
                (mesh.arity==3 || faceOf(shape.shapeToWorld.transformInv(point(mesh,ids[3])))==face))return;
        }
        contact.feature[0]=candidate.feature[0];contact.feature[1]=candidate.feature[1];contact.vertexMask=candidate.vertexMask;
        if(pair.plane)
        {
            const PxU32 soft=pair.soft,rigid=1-soft;
            const auto& mesh=pair.facts->deformable[soft]->state->mesh;
            const auto& plane=pair.facts->shapes[rigid];
            const PxVec3 normal=plane.shapeToWorld.q.getBasisVector0();
            const PxVec3 p=point(mesh,candidate.feature[soft]);
            const PxReal gap=normal.dot(p-plane.shapeToWorld.p);
            if(gap>pair.reach)return;
            contact.normal=soft==0?normal:-normal;contact.separation=gap;
            contact.point[soft]=p;contact.point[rigid]=p-normal*gap;
            contact.weights[soft]=PxVec4(1,0,0,0);contact.weights[rigid]=PxVec4(0.0f);out.active[index]=1;return;
        }
        PxsNpFiniteSupport a(pair.facts->shapes[0]),b(pair.facts->shapes[1]);
        PxsNpFiniteSupport* support[2]={&a,&b};PxTransform pose[2];
        for(PxU32 s=0;s<2;++s)
        {
            const auto* view=pair.facts->deformable[s];
            support[s]->nodal=view!=NULL;pose[s]=view?PxTransform(PxIdentity):pair.facts->shapes[s].shapeToWorld;
            if(view)
            {
                support[s]->count=view->state->mesh.arity;
                for(PxU32 k=0;k<support[s]->count;++k)
                    support[s]->triangle[k]=point(view->state->mesh,view->state->mesh.vertices[support[s]->count*candidate.feature[s]+k]);
            }
        }
        PxVec3 normal;PxReal separation;
        if(!PxGjkQuery::proximityInfo(a,b,pose[0],pose[1],pair.reach,tolerance,
            contact.point[0],contact.point[1],normal,separation))return;
        if(!normal.isFinite() || !PxIsFinite(separation)){PxAtomicExchange(&failed,1);return;}
        const auto& geometry=*pair.facts->deformable[pair.soft];
        if(pair.self && geometry.selfFilterDistance && *geometry.selfFilterDistance!=0)
        {
            const auto& mesh=geometry.state->mesh;
            PxVec3 live[2][4],rest[2][4];
            for(PxU32 s=0;s<2;++s)for(PxU32 k=0;k<mesh.arity;++k)
            {
                live[s][k]=support[s]->triangle[k];
                rest[s][k]=(*geometry.restPositions)[mesh.vertices[mesh.arity*candidate.feature[s]+k]].getXYZ();
                if(!rest[s][k].isFinite()){PxAtomicExchange(&failed,1);return;}
            }
            if(!filteredWitness(live,rest,mesh.arity,*geometry.selfFilterDistance,pair.reach,contact.point))return;
            const auto delta=contact.point[0]-contact.point[1];
            if(separation>0 && delta.magnitudeSquared()>0)normal=delta.getNormalized();
        }
        for(PxU32 s=0;s<2;++s)
        {
            contact.weights[s]=PxVec4(0.0f);
            if(support[s]->nodal)
            {
                const PxVec3* x=support[s]->triangle;
                if(support[s]->count==3)
                {
                    PxReal u,v;contact.point[s]=Gu::closestPtPointTriangle(contact.point[s],x[0],x[1],x[2],u,v);
                    contact.weights[s]=PxVec4(1-u-v,u,v,0);
                }
                else
                {
                    const PxMat33 matrix(x[1]-x[0],x[2]-x[0],x[3]-x[0]);
                    if(matrix.getDeterminant()==0){PxAtomicExchange(&failed,1);return;}
                    const PxVec3 w=matrix.getInverse()*(contact.point[s]-x[0]);
                    contact.weights[s]=PxVec4(1-w.x-w.y-w.z,w.x,w.y,w.z);
                    if(!contact.weights[s].isFinite()){PxAtomicExchange(&failed,1);return;}
                }
            }
        }
        const PxVec3 delta=contact.point[0]-contact.point[1];
        if(normal.dot(delta)*separation<0)normal=-normal;
        if(normal.normalize()==0){PxAtomicExchange(&failed,1);return;}
        // GJK's tolerance is for distance iteration. Its witness normal at a
        // near-tie need not separate every vertex used by the downstream
        // certificate. Box versus simplex has a complete, bounded SAT basis.
        const PxU32 soft=pair.soft,rigid=1-soft;
        const auto& box=pair.facts->shapes[rigid];
        if(!pair.facts->deformable[rigid] && box.kind==PxsContactShapeGeometry::eBOX)
        {
            const auto n=support[soft]->count;const auto half=box.halfExtents;
            PxVec3 local[4];for(PxU32 i=0;i<n;++i)local[i]=box.shapeToWorld.transformInv(support[soft]->triangle[i]);
            PxVec3 axis=box.shapeToWorld.q.rotateInv(soft==0?normal:-normal);
            const auto gap=[&](const PxVec3& v)
            {
                PxReal low=v.dot(local[0]);for(PxU32 i=1;i<n;++i)low=PxMin(low,v.dot(local[i]));
                return low-v.abs().dot(half);
            };
            PxReal best=gap(axis);
            if(best<0)
            {
                bool improved=false;
                const auto test=[&](PxVec3 v)
                {
                    if(!v.isFinite() || v.normalize()==0)return;
                    for(PxU32 sign=0;sign<2;++sign,v=-v)
                    {const PxReal g=gap(v);if(g>best){best=g;axis=v;improved=true;}}
                };
                const PxVec3 basis[]={PxVec3(1,0,0),PxVec3(0,1,0),PxVec3(0,0,1)};
                for(const auto& v:basis)test(v);
                for(PxU32 i=0;i<n;++i)for(PxU32 j=i+1;j<n;++j)
                {
                    const auto edge=local[j]-local[i];
                    for(const auto& v:basis)test(edge.cross(v));
                    for(PxU32 k=j+1;k<n;++k)test(edge.cross(local[k]-local[i]));
                }
                if(improved)
                {
                    PxReal low=axis.dot(local[0]);
                    for(PxU32 i=1;i<n;++i)low=PxMin(low,axis.dot(local[i]));
                    const PxReal eps=16*PX_EPS_F32*(half.magnitude()+PxAbs(low));
                    PxU32 ids[4],count=0;
                    for(PxU32 i=0;i<n;++i)if(axis.dot(local[i])<=low+eps)ids[count++]=i;
                    const auto probe=box.shapeToWorld.transformInv(contact.point[soft]);
                    PxReal u=0,v=0;PxVec3 p=local[ids[0]];
                    if(count==2)
                    {
                        const auto e=local[ids[1]]-p;
                        if(e.magnitudeSquared()>0)u=PxClamp((probe-p).dot(e)/e.magnitudeSquared(),0.f,1.f);
                        p+=e*u;
                    }
                    else if(count>=3)p=Gu::closestPtPointTriangle(probe,p,local[ids[1]],local[ids[2]],u,v);
                    contact.weights[soft]=PxVec4(0.f);
                    contact.weights[soft][ids[0]]=1-u-v;
                    if(count>=2)contact.weights[soft][ids[1]]=u;
                    if(count>=3)contact.weights[soft][ids[2]]=v;
                    auto q=p.maximum(-half).minimum(half);
                    for(PxU32 d=0;d<3;++d)if(PxAbs(axis[d])>16*PX_EPS_F32)q[d]=axis[d]>0?half[d]:-half[d];
                    contact.point[soft]=box.shapeToWorld.transform(p);contact.point[rigid]=box.shapeToWorld.transform(q);
                    normal=box.shapeToWorld.q.rotate(soft==0?axis:-axis);
                }
            }
        }
        contact.normal=normal;contact.separation=normal.dot(contact.point[0]-contact.point[1]);
        out.active[index]=1;
    }
    void execute(PxU32 ticket,PxU32& spare,bool& unfinished)
    {
        // At most 32 decisions for a PxU32-indexed range table.
        PxU32 r=0,high=ranges.size();
        for(PxU32 level=0;level<32 && r+1<high;++level)
        {const PxU32 middle=r+(high-r)/2;if(ticket<ranges[middle].first)high=middle;else r=middle;}
        const auto& range=ranges[r];const PxU32 begin=(ticket-range.first)*range.width,end=PxMin(range.items,begin+range.width);
        if(phase<=Tree)
        {
            auto& geometry=*shapes[range.object];auto& mesh=geometry.state->mesh;
            if(phase==Sample){if(!mesh.sampleVertices(geometry,begin,end))PxAtomicExchange(&failed,1);}
            else if(phase==Leaves)mesh.refitLeaves(begin,end);
            else for(PxU32 i=begin;i<end;++i)mesh.refitTile(treeRoots[range.object]+i,treeLevels[range.object]);
            return;
        }
        auto& pair=pairs[range.object];auto& out=*pair.output;
        if(phase==QueryInit)
        {
            const bool two=pair.facts->deformable[1-pair.soft]!=NULL;
            const auto& mesh=pair.facts->deformable[pair.soft]->state->mesh;
            for(PxU32 i=begin;i<end;++i)
            {
                auto& query=out.queries[i];query.root=two?1:pair.roots+i;
                query.feature=two?mesh.order[i]:PX_MAX_U32;
                query.bounds=expanded(two?mesh.bounds[mesh.leafBase+i]:rigidBounds(pair.facts->shapes[1-pair.soft]),pair.reach);
                query.count=query.offset=query.written=0;query.cursor.start(query.root);
            }
        }
        else if(phase==QueryCount || phase==QueryFill)
        {
            if(pair.plane)
            {
                const auto& mesh=pair.facts->deformable[pair.soft]->state->mesh;
                for(PxU32 i=begin;i<end;++i)
                {auto& c=out.candidates[i];c.feature[pair.soft]=mesh.uniqueVertices[i];c.feature[1-pair.soft]=PX_MAX_U32;c.vertexMask=PxU8(1u<<pair.soft);}
                return;
            }
            const auto& mesh=pair.facts->deformable[pair.treeSide]->state->mesh;
            for(PxU32 i=begin;i<end;++i)
            {
                auto& query=out.queries[i];
                if(phase==QueryFill && query.count<=PxsDeformableQueryWork::InlineCapacity)
                {
                    if(query.written==query.count)++spare;
                    while(query.written<query.count)
                    {
                        const PxU32 slot=query.written++;
                        auto& c=out.candidates[query.offset+out.totals[i/ScanWidth]+slot];
                        c.feature[pair.treeSide]=query.hits[slot];c.feature[1-pair.treeSide]=query.feature;c.vertexMask=0;
                    }
                    continue;
                }
                if(query.cursor.done()){++spare;continue;}
                const auto* motion=pair.self && mesh.displacementBounds.size()?mesh.displacementBounds.begin():NULL;
                const auto* subtreeMax=pair.self && !mesh.subtreeMaxFeature.empty()?mesh.subtreeMaxFeature.begin():NULL;
                // Every ticket reserves one 64-visit packet. Only unused
                // reservations can advance this cursor further in this task.
                for(PxU32 packet=0;!query.cursor.done() && (!packet || spare);++packet)
                {
                    if(packet)--spare;
                    PxU32 hits[Packet];const PxU32 count=query.cursor.next(mesh.view(),query.bounds,hits,Packet,
                        motion,motion?motion+mesh.leafBase+i:NULL,pair.marginSquared,subtreeMax,query.feature);
                    if(!query.cursor.valid){PxAtomicExchange(&failed,1);break;}
                    for(PxU32 k=0;k<count;++k)if(retain(pair,query.feature,hits[k]))
                    {
                        if(phase==QueryCount)
                        {
                            if(query.count<PxsDeformableQueryWork::InlineCapacity)query.hits[query.count]=hits[k];
                            ++query.count;
                        }
                        else
                        {
                            if(query.written>=query.count){PxAtomicExchange(&failed,1);break;}
                            auto& c=out.candidates[query.offset+out.totals[i/ScanWidth]+query.written++];
                            c.feature[pair.treeSide]=hits[k];c.feature[1-pair.treeSide]=query.feature;c.vertexMask=0;
                        }
                    }
                }
                if(!query.cursor.done())unfinished=true;
                else if(phase==QueryFill && query.written!=query.count)PxAtomicExchange(&failed,1);
            }
        }
        else if(phase==QueryScan)
        {
            PxU32 sum=0;for(PxU32 i=begin;i<end;++i)
            {auto& query=out.queries[i];query.offset=sum;sum+=query.count;query.cursor.start(query.root);}
            out.totals[begin/ScanWidth]=sum;
        }
        else if(phase==Contact)for(PxU32 i=begin;i<end;++i)feature(pair,i);
        else if(phase==ContactScan)
        {
            PxU32 sum=0;for(PxU32 i=begin;i<end;++i){out.offsets[i]=sum;sum+=out.active[i];}
            out.totals[begin/ScanWidth]=sum;
        }
        else if(phase==Compact)for(PxU32 i=begin;i<end;++i)if(out.active[i])
            out.contacts[out.totals[i/ScanWidth]+out.offsets[i]]=out.rawContacts[i];
    }
    void completedPhase()
    {
        if(phase==Tree)
        {
            bool remaining=false;
            for(PxU32 s=0;s<shapes.size();++s)
            {if(treeSource[s]>1)treeSource[s]=treeRoots[s];remaining|=treeSource[s]>1;}
            if(remaining)return;
            for(auto* shape:shapes){shape->state->mesh.ready=!failed;shape->state->epoch=context.mCpuDeformableEpoch;}
        }
        if(phase==QueryScan)scanPairs(false);
        if(phase==ContactScan)scanPairs(true);
        phase=Phase(PxU32(phase)+1);
    }
public:
    PxsDeformableNpBatch(PxsContext& ctx,PxsContactManagers& cms,PxReal step,PxsContactManagerOutput* outputs,
        PxBaseTask* next,PxContactModifyCallback* callback,Finish done):context(ctx),managers(cms),cmOutputs(outputs),
        continuation(next),modify(callback),finish(done),dt(step),tolerance(1)
    {
        continuation->addReference();
        profile=std::getenv("PHYSX_NP_DEFORMABLE_STATS")!=NULL;if(profile)started=PxTime::getCurrentCounterValue();
        auto* thread=context.getNpThreadContext();tolerance=thread->mNarrowPhaseParams.mToleranceLength;
        PxsMaterialManager* materials=thread->mMaterialManager;context.putNpThreadContext(thread);
        for(PxsContactManager* cm:cms.mContactManagerMapping)
        {
            auto& unit=cm->getWorkUnit();if(!unit.mDeformable[0] && !unit.mDeformable[1])continue;
            if(!cm->mContactGeometry)cm->mContactGeometry=PX_NEW(PxsContactGeometry);
            auto& facts=*cm->mContactGeometry;
            if(!facts.deformableOutput)
            {
                facts.deformableOutput=PX_NEW(PxsDeformableContactOutput);
                facts.deformableOutput->birthEpoch=context.mCpuDeformableEpoch;
            }
            facts.deformable[0]=unit.mDeformable[0];facts.deformable[1]=unit.mDeformable[1];
            facts.contactDistance=unit.getShapeCore0()->mContactOffset+unit.getShapeCore1()->mContactOffset;
            const PxsShapeCore* source[2]={unit.getShapeCore0(),unit.getShapeCore1()};
            const PxsRigidCore* body[2]={unit.mRigidCore0,unit.mRigidCore1};
            const PxU32 cache[2]={unit.mTransformCache0,unit.mTransformCache1};
            const PxU16 dynamic[2]={PxcNpWorkUnitFlag::eDYNAMIC_BODY0|PxcNpWorkUnitFlag::eARTICULATION_BODY0,
                PxcNpWorkUnitFlag::eDYNAMIC_BODY1|PxcNpWorkUnitFlag::eARTICULATION_BODY1};
            PxsMaterialData material[2];
            for(PxU32 s=0;s<2;++s)
            {
                facts.shapes[s]=PxsContactShapeGeometry();
                if(facts.deformable[s])
                {addShape(facts.deformable[s]);material[s].staticFriction=material[s].dynamicFriction=*facts.deformable[s]->friction;}
                else if(body[s])
                {
                    publishContactShapeGeometry(facts.shapes[s],*source[s],*body[s],(unit.mFlags&dynamic[s])!=0,
                        context.getTransformCache().getTransformCache(cache[s]).transform);
                    if(materials)material[s]=*materials->getMaterial(source[s]->mMaterialIndex);
                }
                else failed=1;
            }
            PxReal staticFriction,dynamicFriction,restitution,damping;PxU32 materialFlags;
            PxsCombineMaterials(material[0],material[1],staticFriction,dynamicFriction,restitution,materialFlags,damping);
            facts.deformableOutput->friction=PxMax(staticFriction,dynamicFriction);
            facts.deformableOutput->restitution=restitution;facts.deformableOutput->damping=damping;
            Pair pair={&facts,facts.deformableOutput,facts.deformable[0]?0u:1u,0,0,0,
                PxMax(facts.contactDistance,context.getContactDistances()[cache[0]]+context.getContactDistances()[cache[1]]),false,
                facts.shapes[0].kind==PxsContactShapeGeometry::ePLANE || facts.shapes[1].kind==PxsContactShapeGeometry::ePLANE,
                (unit.mFlags&PxcNpWorkUnitFlag::eDETECT_DISCRETE_CONTACT)!=0};
            if((!facts.deformable[0] && facts.shapes[0].kind==PxsContactShapeGeometry::eNONE) ||
                (!facts.deformable[1] && facts.shapes[1].kind==PxsContactShapeGeometry::eNONE))failed=1;
            pairs.pushBack(pair);
        }
        for(auto* geometry:context.mCpuDeformables)
        {
            if(!geometry->state || !geometry->selfCollision() || (geometry->wakeCounter && *geometry->wakeCounter<=0) ||
                geometry->state->selfEpoch==context.mCpuDeformableEpoch)continue;
            addShape(geometry);auto& state=*geometry->state;auto& facts=state.selfGeometry;
            facts.deformable[0]=facts.deformable[1]=geometry;facts.contactDistance=2*geometry->contactOffset;
            if(!state.selfContacts.birthEpoch) state.selfContacts.birthEpoch=context.mCpuDeformableEpoch;
            state.selfContacts.friction=*geometry->friction;state.selfEpoch=context.mCpuDeformableEpoch;
            pairs.pushBack({&facts,&state.selfContacts,0,1,1,0,facts.contactDistance,true,false,true});
        }
        treeSource.resize(shapes.size());treeRoots.resize(shapes.size());treeLevels.resize(shapes.size());
        for(PxU32 i=0;i<shapes.size();++i)
            treeSource[i]=shapes[i]->state && shapes[i]->state->epoch!=context.mCpuDeformableEpoch?shapes[i]->state->mesh.leafBase:1;
    }
    void advance()
    {
        for(;;)
        {
            if(failed || phase==Done)
            {
                for(auto& pair:pairs)pair.output->valid=!failed;
                if(profile)
                {
                    PxU64 candidates=0,contacts=0,visits=0;
                    for(const auto& pair:pairs)
                    {candidates+=pair.output->candidates.size();contacts+=pair.output->count;
                        for(const auto& q:pair.output->queries)visits+=q.cursor.visits;}
                    std::fprintf(stderr,"[NP_DEFORMABLE_PACKETS] shapes=%u pairs=%u candidates=%llu contacts=%llu treeVisitsFill=%llu waves=%u tasks=%u peakTasks=%d peakPacketsPerTask=%d packetLimit=16 vertexPacket=64 queryVisits=64 contactPacket=4 scalarBytes=%u spanMs=%.6f valid=%u\n",
                        shapes.size(),pairs.size(),static_cast<unsigned long long>(candidates),static_cast<unsigned long long>(contacts),
                        static_cast<unsigned long long>(visits),waves,tasksSubmitted+PxU32(continuedTasks),int(peak),int(peakPackets),unsigned(sizeof(PxsDeformableReal)),
                        PxTime::getCounterFrequency().toTensOfNanos(PxTime::getCurrentCounterValue()-started)*.00001,unsigned(!failed));
                }
                Finish done=finish;PxsContext& ctx=context;PxsContactManagers& cms=managers;auto* outputs=cmOutputs;
                PxBaseTask* next=continuation;auto* callback=modify;const PxReal step=dt;
                done(ctx,cms,NULL,step,outputs,next,callback);delete this;next->removeReference();return;
            }
            if(!prepared){prepare();prepared=true;}
            if(failed)continue;
            if(PxU32(cursor)>=jobs){prepared=false;completedPhase();continue;}
            const PxU32 workers=continuation->getTaskManager()->getCpuDispatcher()->getWorkerCount();
            // A packet is an independent device work item. On the CPU group
            // up to four before paying for a task hand-off, retaining the
            // sixteen-packet upper bound and two stealable ranges per lane.
            const PxU32 first=cursor,remaining=jobs-first;
            const PxU32 lanes=PxMin(1+(remaining-1)/4,PxMax(1u,PxMin(32u,workers*2)));
            const PxU32 items=PxMin(remaining,lanes*PacketsPerTask);
            const PxU32 width=(items+lanes-1)/lanes,count=(items+width-1)/width;
            cursor+=items;++waves;
            if(count==1){if(executeRange(first,cursor))cursor=first;continue;}
            // Within a phase each bounded task releases its lane directly to
            // the next independent range. Only the phase boundary needs fan-in.
            packetWidth=width;
            nextPacket=cursor;
            const PxU32 firstBatchEnd=cursor;
            if(cursor<jobs)tasksSubmitted+=1+(jobs-cursor-1)/width;
            cursor=jobs;
            Task* tasks[32];auto& pool=context.getTaskPool();pool.lock();
            for(PxU32 i=0;i+1<count;++i)
            {
                const PxU32 from=first+i*width;
                tasks[i]=PX_PLACEMENT_NEW(pool.allocateNotThreadSafe(sizeof(Task)),Task)(*this,from,from+width);
                tasks[i]->setContinuation(continuation);
            }
            pool.unlock();pending=PxI32(count);tasksSubmitted+=count-1;
            for(PxU32 i=0;i+1<count;++i)tasks[i]->removeReference();
            const PxU32 inlineFirst=first+(count-1)*width;
            if(executeRange(inlineFirst,firstBatchEnd)){resumeRange(inlineFirst,firstBatchEnd);return;}
            if(submitNextRange() || PxAtomicDecrement(&pending)!=0)return;
        }
    }
    // The last worker enters through this wrapper so the phase is advanced
    // exactly once. The caller sentinel handles the zero-worker dispatcher.
    void workerFinished(){advance();}
    void resumeRange(PxU32 first,PxU32 end)
    {
        if(profile)PxAtomicIncrement(&continuedTasks);
        submitRange(first,end);
    }
    bool submitNextRange()
    {
        const PxI64 from=PxAtomicAdd(&nextPacket,PxI64(packetWidth))-packetWidth;
        if(from>=jobs)return false;
        submitRange(PxU32(from),PxU32(PxMin(from+packetWidth,PxI64(jobs))));
        return true;
    }
    void submitRange(PxU32 first,PxU32 end)
    {
        auto& pool=context.getTaskPool();pool.lock();
        auto* task=PX_PLACEMENT_NEW(pool.allocateNotThreadSafe(sizeof(Task)),Task)(
            *this,first,end);
        task->setContinuation(continuation);pool.unlock();
        task->removeReference();
    }
};
}
#endif
#endif
