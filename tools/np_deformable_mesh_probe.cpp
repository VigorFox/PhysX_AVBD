// NP geometry/bounded sparse query oracle. No AVBD solver or trajectory gate.
#include "PxPhysicsAPI.h"
#include "avbd_probe_foundation.h"
#include "PxsDeformableMesh.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <vector>
#include <algorithm>
using namespace physx;

struct Batch
{
    struct Job:PxBaseTask
    {
        Batch& batch;PxU32 begin,end;
        Job(Batch& b,PxU32 first,PxU32 last):batch(b),begin(first),end(last){}
        void run() PX_OVERRIDE{batch.function(begin,end);}
        const char* getName() const PX_OVERRIDE{return "NP.meshProbe.packet";}
        void addReference() PX_OVERRIDE{}
        void removeReference() PX_OVERRIDE{}
        int32_t getReference() const PX_OVERRIDE{return 1;}
        void release() PX_OVERRIDE
        {
            // Only the probe caller waits; workers never wait on this pool.
            std::lock_guard<std::mutex> lock(batch.mutex);
            if(--batch.pending==0)batch.condition.notify_one();
        }
    };
    PxDefaultCpuDispatcher* dispatcher;
    std::function<void(PxU32,PxU32)> function;
    std::mutex mutex;std::condition_variable condition;PxU32 pending=0;
    Batch(PxU32 workers):dispatcher(PxDefaultCpuDispatcherCreate(workers)){}
    ~Batch(){dispatcher->release();}
    void run(PxU32 count,const std::function<void(PxU32,PxU32)>& f)
    {
        function=f;std::vector<Job*> jobs;
        for(PxU32 begin=0;begin<count;begin+=64)jobs.push_back(new Job(*this,begin,PxMin(count,begin+64)));
        pending=PxU32(jobs.size());for(auto* job:jobs)dispatcher->submitTask(*job);
        {std::unique_lock<std::mutex> lock(mutex);condition.wait(lock,[&]{return pending==0;});}
        for(auto* job:jobs)delete job;
    }
};
static bool check(bool condition,const char* what)
{if(!condition)std::printf("[NP_MESH_FAIL] %s\n",what);return condition;}
static bool embedding()
{
    PxVec4 a[4]={PxVec4(0,0,0,1),PxVec4(1,0,0,1),PxVec4(0,1,0,1),PxVec4(0,0,1,1)};
    PxVec4* slot=a;PxU32 sim[4]={0,1,2,3},collision[3]={0,1,2},element[3]={0,0,0};
    PxReal weights[12]={.1f,.2f,.3f,.4f, 1,0,0,0, 0,1,0,0};
    PxsDeformableGeometry g;g.positions=&slot;g.simulation={sim,1,4,4,4};g.collision={collision,1,3,3,4};
    g.embeddingElements=element;g.embeddingWeights=weights;
    PxsDeformableVec3 p;bool ok=check(g.point(0,p) && (p-PxsDeformableVec3(.2f,.3f,.4f)).magnitude()<1.e-6f,"affine embedding");
    PxVec4 b[4];for(PxU32 i=0;i<4;++i)b[i]=a[i]+PxVec4(2,-3,4,0);slot=b;
    ok&=check(g.point(0,p) && (p-PxsDeformableVec3(2.2f,-2.7f,4.4f)).magnitude()<1.e-6f,"buffer replacement");
    weights[0]=.2f;ok&=check(!g.point(0,p),"reject nonpartition weights");weights[0]=.1f;
    weights[0]=std::numeric_limits<PxReal>::quiet_NaN();ok&=check(!g.point(0,p),"reject NaN embedding");weights[0]=.1f;
    b[0].x=std::numeric_limits<PxReal>::infinity();ok&=check(!g.point(0,p),"reject nonfinite position");b[0].x=2;
    element[0]=4;ok&=check(!g.point(0,p),"reject embedding tet range");element[0]=0;
    g.embeddingElements=NULL;g.embeddingWeights=NULL;ok&=check(!g.point(0,p),"no count-based identity inference");
    PxU16 small[3]={0,1,2};g.collision=g.simulation={small,1,3,3,2};
    ok&=check(g.point(2,p) && (p-PxsDeformableVec3(2,-2,4)).magnitude()<1.e-6f,"u16 direct mapping");
    small[2]=8;ok&=check(!g.primitive(0,&p),"reject topology range");return ok;
}
int main(int argc,char** argv)
{
    AvbdProbeFoundation foundation;bool ok=embedding();
    const PxU32 workers=argc>1?PxU32(std::atoi(argv[1])):4;
    const PxU32 grid=argc>2?PxU32(std::atoi(argv[2])):24;
    if(grid<2 || grid>256)return 2;Batch batch(workers);
    PxArray<PxVec4> positions;PxArray<PxU32> indices;
    for(PxU32 y=0;y<=grid;++y)for(PxU32 x=0;x<=grid;++x)positions.pushBack(PxVec4(.2f*x,0,.2f*y,1));
    for(PxU32 y=0;y<grid;++y)for(PxU32 x=0;x<grid;++x)
    {PxU32 a=y*(grid+1)+x;const PxU32 ids[6]={a,a+1,a+grid+2,a,a+grid+2,a+grid+1};for(PxU32 id:ids)indices.pushBack(id);}
    PxVec4* slot=positions.begin();PxsDeformableGeometry g;g.positions=&slot;
    g.collision=g.simulation={indices.begin(),indices.size()/3,positions.size(),3,4};
    PxsDeformableMesh mesh;const auto start=std::chrono::steady_clock::now();
    ok&=check(mesh.bake(g),"bake");
    const double bakeMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    PxU64 visits=0,candidates=0,packets=0;double refitMs=0,queryMs=0;
    for(PxU32 frame=0;frame<12;++frame)
    {
        for(PxU32 v=0;v<positions.size();++v)
        {positions[v].y=.3f*std::sin(PxReal(v% (grid+1))*.2f+frame*.1f);}
        mesh.ready=false;std::atomic<bool> valid(true);const auto begin=std::chrono::steady_clock::now();
        batch.run(mesh.supports.size(),[&](PxU32 a,PxU32 b){if(!mesh.sampleVertices(g,a,b))valid=false;});
        batch.run(mesh.leafBase,[&](PxU32 a,PxU32 b){mesh.refitLeaves(a,b);});
        PxU32 source=mesh.leafBase;
        while(source>1)
        {
            PxU32 roots=source,levels=0;while(roots>1 && levels<6){roots/=2;++levels;}
            batch.run(roots,[&](PxU32 a,PxU32 b){for(PxU32 r=a;r<b;++r)mesh.refitTile(roots+r,levels);});
            source=roots;
        }
        mesh.ready=valid;ok&=check(mesh.ready,"finite refit");
        for(PxU32 leaf=0;leaf<mesh.order.size();++leaf)
        {
            // Independent enclosure oracle from the authored positions; tree
            // traversal cannot validate an incorrectly refitted leaf itself.
            const PxU32 primitive=mesh.order[leaf];
            for(PxU32 k=0;k<3;++k)
                if(!mesh.bounds[mesh.leafBase+leaf].contains(positions[indices[3*primitive+k]].getXYZ()))valid=false;
        }
        ok&=check(valid,"leaf contains authored positions");
        refitMs+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        std::atomic<PxU64> frameVisits(0),frameCandidates(0),framePackets(0);const auto queryStart=std::chrono::steady_clock::now();
        batch.run(mesh.order.size(),[&](PxU32 first,PxU32 last)
        {
            for(PxU32 a=first;a<last;++a)
            {
                PxBounds3 query=mesh.bounds[mesh.leafBase+a];query.fattenFast(.005f);
                PxsDeformableQuery cursor;cursor.start();std::vector<PxU32> found;PxU32 pieces=0;
                do
                {
                    PxU32 ids[7];const PxU32 before=cursor.visits,n=cursor.next(mesh.view(),query,ids,1+frame%7);
                    if(!cursor.valid || cursor.visits-before>PxsDeformableQuery::VisitLimit){valid=false;break;}
                    found.insert(found.end(),ids,ids+n);++pieces;
                }while(!cursor.done());
                std::sort(found.begin(),found.end());
                // Exhaustive independent bound oracle only for small grids.
                if(grid<=24)
                {
                    std::vector<PxU32> expected;
                    for(PxU32 b=0;b<mesh.order.size();++b)if(mesh.bounds[mesh.leafBase+b].intersects(query))expected.push_back(mesh.order[b]);
                    std::sort(expected.begin(),expected.end());if(found!=expected)valid=false;
                    // The half-tree query must cover every larger authored feature,
                    // including when the Morton order differs from authored order.
                    if(frame==0)
                    {
                        PxsDeformableQuery half;half.start();
                        std::vector<PxU32> forward;
                        do
                        {
                            PxU32 ids[7];const auto before=half.visits;
                            const auto n=half.next(mesh.view(),query,ids,7,NULL,NULL,0,mesh.subtreeMaxFeature.begin(),mesh.order[a]);
                            if(!half.valid || half.visits-before>PxsDeformableQuery::VisitLimit){valid=false;break;}
                            for(PxU32 k=0;k<n;++k)if(ids[k]>mesh.order[a])forward.push_back(ids[k]);
                        }while(!half.done());
                        std::sort(forward.begin(),forward.end());
                        expected.erase(expected.begin(),std::upper_bound(expected.begin(),expected.end(),mesh.order[a]));
                        if(forward!=expected)valid=false;
                    }
                }
                else if(std::find(found.begin(),found.end(),mesh.order[a])==found.end())valid=false;
                frameVisits+=cursor.visits;frameCandidates+=found.size();framePackets+=pieces;
            }
        });
        queryMs+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-queryStart).count();
        ok&=check(valid,"complete sparse candidate coverage");visits+=frameVisits;candidates+=frameCandidates;packets+=framePackets;
    }
    // All overlapping leaves must still resume correctly through tiny output
    // packets. Disjoint and incomplete snapshots must not produce fake hits.
    PxsDeformableQuery dense;dense.start();PxU32 found=0;
    do{PxU32 ids[3];found+=dense.next(mesh.view(),mesh.bounds[1],ids,3);}while(dense.valid && !dense.done());
    ok&=check(dense.valid && found==mesh.order.size(),"dense continuation completeness");
    PxsDeformableQuery far;far.start();PxU32 ids[3];
    ok&=check(far.next(mesh.view(),PxBounds3(PxVec3(-20),PxVec3(-19)),ids,3)==0 && far.done() && far.visits==1,"far tree rejection");
    mesh.ready=false;far.start();ok&=check(far.next(mesh.view(),PxBounds3(PxVec3(-20),PxVec3(-19)),ids,3)==0 && !far.valid,"incomplete snapshot rejection");
    auto changed=g;changed.collision.indexBytes=2;
    ok&=check(!mesh.sampleVertices(changed,0,mesh.supports.size()),"topology replacement invalidates bake");
    const double allPairs=12.*mesh.order.size()*mesh.order.size();
    std::printf("[NP_MESH] status=%s scalarBytes=%zu workers=%u grid=%u primitives=%u bytes=%zu bakeMs=%.6f refitMeanMs=%.6f queryWithOracleMeanMs=%.6f nodeVisits=%llu candidates=%llu packets=%llu allPairs=%.0f visitRatio=%.9f\n",
        ok?"PASS":"FAIL",sizeof(PxsDeformableReal),workers,grid,mesh.order.size(),mesh.bytes(),bakeMs,refitMs/12,queryMs/12,
        static_cast<unsigned long long>(visits),static_cast<unsigned long long>(candidates),static_cast<unsigned long long>(packets),allPairs,visits/allPairs);
    return ok?0:1;
}
