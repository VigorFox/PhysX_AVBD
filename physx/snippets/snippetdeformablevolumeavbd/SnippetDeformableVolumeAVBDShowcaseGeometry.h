// Published-state observer for the five-body headless scene. SAT uses the
// collision mesh, independently of NP candidates, simulation tets and duals.
#pragma once
#include "geometry/PxGjkQuery.h"
struct ShowcaseGeometry
{
    struct Hull
    {
        PxVec3d v[8],edges[6],normals[4];
        PxU32 vertices=4,edgeCount=6,faceCount=4;
        PxBounds3 bounds=PxBounds3::empty();
        void vertex(PxU32 i,const PxVec3& p){bounds.include(p);v[i]=PxVec3d(p.x,p.y,p.z);}
        void tetrahedron()
        {
            PxU32 e=0;
            for(PxU32 i=0;i<4;++i)
            {
                normals[i]=(v[(i+1)%4]-v[i]).cross(v[(i+2)%4]-v[i]);
                for(PxU32 j=i+1;j<4;++j)edges[e++]=v[j]-v[i];
            }
        }
        void box(const PxTransform& pose,const PxVec3& half)
        {
            vertices=8;edgeCount=faceCount=3;
            for(PxU32 i=0;i<8;++i)vertex(i,pose.transform(PxVec3(
                i&1?half.x:-half.x,i&2?half.y:-half.y,i&4?half.z:-half.z)));
            for(PxU32 i=0;i<3;++i)normals[i]=edges[i]=v[size_t(1)<<i]-v[0];
        }
    };
    struct Tet {PxU32 id[4];Hull hull;};
    struct Body {std::vector<Tet> tets;std::vector<PxU32> order;PxBounds3 bounds;};
    Body bodies[5],pedestal;
    PxReal pedestalBand=0;
    double external=0,self=0,ground=0,rigid=0,between=0;
    static double depth(const Hull& a,const Hull& b)
    {
        if(!a.bounds.intersects(b.bounds))return 0;
        double best=PX_MAX_F64;
        const auto axis=[&](const PxVec3d& d)
        {
            const double n=d.magnitudeSquared();if(!n)return true;
            double lo[2],hi[2];const Hull* h[2]={&a,&b};
            for(PxU32 s=0;s<2;++s)
            {
                lo[s]=hi[s]=0;
                for(PxU32 i=1;i<h[s]->vertices;++i)
                {const double q=d.dot(h[s]->v[i]-h[s]->v[0]);lo[s]=PxMin(lo[s],q);hi[s]=PxMax(hi[s],q);}
            }
            const double offset=d.dot(b.v[0]-a.v[0]);
            const double gap=PxMin(hi[0]-lo[1]-offset,hi[1]-lo[0]+offset);
            if(gap<=0)return false;
            best=PxMin(best,gap*gap/n);return true;
        };
        for(PxU32 i=0;i<a.faceCount;++i)if(!axis(a.normals[i]))return 0;
        for(PxU32 i=0;i<b.faceCount;++i)if(!axis(b.normals[i]))return 0;
        for(PxU32 i=0;i<a.edgeCount;++i)for(PxU32 j=0;j<b.edgeCount;++j)
            if(!axis(a.edges[i].cross(b.edges[j])))return 0;
        return PxSqrt(best);
    }

    // Measure the first soft/soft contact on published collision tets; freeze
    // this axis for recoil so tangential roll cannot masquerade as rebound.
    static bool impactNormal(const Body& a,const Body& b,PxReal band,PxVec3& normal)
    {
        struct Support : PxGjkQuery::Support
        {
            const Hull& hull;
            explicit Support(const Hull& h):hull(h){}
            PxReal getMargin() const PX_OVERRIDE {return 0;}
            PxVec3 supportLocal(const PxVec3& d) const PX_OVERRIDE
            {
                const PxVec3d axis(d.x,d.y,d.z);
                PxU32 best=0;
                for(PxU32 i=1;i<hull.vertices;++i)if(hull.v[i].dot(axis)>hull.v[best].dot(axis))best=i;
                const auto& p=hull.v[best];return PxVec3(PxReal(p.x),PxReal(p.y),PxReal(p.z));
            }
        };
        PxReal closest=band;bool found=false;
        for(const auto& first:a.tets)
        {
            auto bounds=first.hull.bounds;bounds.fattenFast(band);
            if(!bounds.intersects(b.bounds))continue;
            for(PxU32 j:b.order)
            {
                const auto& second=b.tets[j].hull;
                if(second.bounds.minimum.x>bounds.maximum.x)break;
                if(!bounds.intersects(second.bounds))continue;
                PxVec3 pa,pb,axis;PxReal gap;
                if(!PxGjkQuery::proximityInfo(Support(first.hull),Support(second),PxTransform(PxIdentity),
                    PxTransform(PxIdentity),band,1.f,pa,pb,axis,gap) || gap>closest || !axis.isFinite())continue;
                if(axis.normalize()==0)continue;
                if(axis.dot(a.bounds.getCenter()-b.bounds.getCenter())<0)axis=-axis;
                normal=axis;closest=gap;found=true;
            }
        }
        return found;
    }

    bool observe(PxScene& scene)
    {
        for(PxU32 s=0;s<5;++s)
        {
            auto* actor=getVolumeAvbdRenderVolume(s);
            const auto* mesh=actor?actor->getCollisionMesh():NULL;
            const auto* positions=actor?actor->getPositionInvMassBufferD():NULL;
            if(!mesh || !positions)return false;
            Body& body=bodies[s];body.bounds=PxBounds3::empty();
            if(body.tets.empty())
            {
                body.tets.resize(mesh->getNbTetrahedrons());
                body.order.resize(body.tets.size());
                for(PxU32 i=0;i<body.tets.size();++i)
                {body.order[i]=i;if(!readTet(*mesh,i,body.tets[i].id))return false;}
            }
            for(auto& t:body.tets)
            {
                t.hull.bounds=PxBounds3::empty();
                for(PxU32 k=0;k<4;++k)
                {
                    if(t.id[k]>=mesh->getNbVertices() || !positions[t.id[k]].isFinite())return false;
                    t.hull.vertex(k,positions[t.id[k]].getXYZ());
                }
                t.hull.tetrahedron();body.bounds.include(t.hull.bounds);
            }
            std::sort(body.order.begin(),body.order.end(),[&](PxU32 a,PxU32 b)
                {return body.tets[a].hull.bounds.minimum.x<body.tets[b].hull.bounds.minimum.x;});
        }
        for(PxU32 s=0;s<5;++s)for(PxU32 r=s;r<5;++r)
        {
            if(!bodies[s].bounds.intersects(bodies[r].bounds))continue;
            const auto& list=bodies[r].tets;
            for(size_t i=0;i<bodies[s].tets.size();++i)
            {
                const auto& a=bodies[s].tets[bodies[s].order[i]];
                for(size_t j=s==r?i+1:0;j<list.size();++j)
                {
                    const auto& b=list[bodies[r].order[j]];if(b.hull.bounds.minimum.x>a.hull.bounds.maximum.x)break;
                    if(!a.hull.bounds.intersects(b.hull.bounds))continue;
                    if(s==r)
                    {
                        bool incident=false;for(auto u:a.id)for(auto v:b.id)incident|=u==v;
                        if(!incident)self=PxMax(self,depth(a.hull,b.hull));
                    }
                    else between=PxMax(between,depth(a.hull,b.hull));
                }
            }
        }
        const auto types=PxActorTypeFlag::eRIGID_DYNAMIC|PxActorTypeFlag::eRIGID_STATIC;
        PxActor* actors[3];if(scene.getNbActors(types)!=3 || scene.getActors(types,actors,3)!=3)return false;
        for(auto* actor:actors)
        {
            auto& body=*static_cast<PxRigidActor*>(actor);PxShape* shape=NULL;
            if(body.getShapes(&shape,1)!=1)return false;
            const auto pose=PxShapeExt::getGlobalPose(*shape,body);const auto& geometry=shape->getGeometry();
            Hull box;
            if(geometry.getType()==PxGeometryType::eBOX)
            {
                box.box(pose,static_cast<const PxBoxGeometry&>(geometry).halfExtents);
                if(actor->getType()==PxActorType::eRIGID_STATIC && pedestal.tets.empty())
                {
                    pedestal.tets.resize(1);pedestal.tets.front().hull=box;
                    pedestal.order.push_back(0);pedestal.bounds=box.bounds;
                    pedestalBand=shape->getContactOffset()+getVolumeAvbdRenderVolume(4)->getShape()->getContactOffset();
                }
            }
            else if(geometry.getType()!=PxGeometryType::ePLANE)return false;
            for(const auto& soft:bodies)for(const auto& t:soft.tets)
            {
                if(geometry.getType()==PxGeometryType::eBOX)rigid=PxMax(rigid,depth(box,t.hull));
                else
                {
                    const auto n=pose.q.getBasisVector0();const PxVec3d axis(n.x,n.y,n.z),origin(pose.p.x,pose.p.y,pose.p.z);
                    for(PxU32 i=0;i<4;++i)ground=PxMax(ground,axis.dot(origin-t.hull.v[i]));
                }
            }
        }
        external=PxMax(between,PxMax(ground,rigid));return true;
    }
};
