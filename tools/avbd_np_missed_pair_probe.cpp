// Captured full-five frame 236: separated at DCD, no NP samples, 20 mm at end.
#include "PxPhysicsAPI.h"
#include "PxImmediateMode.h"
#include "geomutils/PxContactPoint.h"
#include "geometry/PxGjkQuery.h"
#include <cstdio>
#include <vector>
#include <malloc.h>
using namespace physx;
struct Scratch:PxCacheAllocator
{
    std::vector<void*> data;
    PxU8* allocateCacheData(PxU32 n) PX_OVERRIDE
    { void* p=_aligned_malloc(n,16);data.push_back(p);return static_cast<PxU8*>(p); }
    ~Scratch() { for(void* p:data) _aligned_free(p); }
};
struct Points:immediate::PxContactRecorder
{
    PxU32 count=0;PxReal gap=PX_MAX_F32;
    bool recordContacts(const PxContactPoint* points,PxU32 n,PxU32) PX_OVERRIDE
    { count+=n;for(PxU32 i=0;i<n;++i) gap=PxMin(gap,points[i].separation);return true; }
};
struct Box:PxGjkQuery::Support
{
    PxVec3 half;
    Box(PxVec3 h):half(h) {}
    PxReal getMargin() const PX_OVERRIDE {return 0;}
    PxVec3 supportLocal(const PxVec3& n) const PX_OVERRIDE
    { return PxVec3(n.x>=0?half.x:-half.x,n.y>=0?half.y:-half.y,n.z>=0?half.z:-half.z); }
};
int main()
{
    PxDefaultAllocator allocator;PxDefaultErrorCallback errors;
    PxFoundation* foundation=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors);
    PxBoxGeometry a(.34f,.14f,.18f),b(.7f,.55f,2.2f);
    const PxGeometry* ga=&a;const PxGeometry* gb=&b;
    const PxTransform pa(PxVec3(6.56926918f,1.07405734f,-.294446796f),
        PxQuat(-.0389411226f,.265108734f,.256224513f,.928735733f));
    const PxTransform pb(PxVec3(7.5999999f,.550000012f,0));
    const PxReal query=.0055999998f+.02f+.000692079542f+.00339212874f;
    const Box sa(a.halfExtents),sb(b.halfExtents);
    PxVec3 pointA,pointB,normal;PxReal distance=PX_MAX_F32;
    const bool near=PxGjkQuery::proximityInfo(sa,sb,pa,pb,query,1.f,pointA,pointB,normal,distance);
    PxCache cache;Scratch scratch;Points output;
    const bool generated=immediate::PxGenerateContacts(&ga,&gb,&pa,&pb,&cache,1,output,query,.01f,1.f,scratch);
    unsigned misses=0;PxU32 seed=20261026;
    auto random=[&]() { seed=1664525u*seed+1013904223u;return float(seed>>8)*(2.f/16777216.f)-1.f; };
    for(PxU32 test=0;test<5000;++test)
    {
        const PxVec3 delta(random()*.02f,random()*.02f,random()*.02f);
        const PxVec3 axis=PxVec3(random(),random(),random()).getNormalized();
        const float angle=random()*.04f;
        const PxTransform previous(pa.p+delta,PxQuat(angle,axis)*pa.q);
        PxCache warmCache;Scratch warmScratch;Points old,warm;
        immediate::PxGenerateContacts(&ga,&gb,&previous,&pb,&warmCache,1,old,query,.01f,1.f,warmScratch);
        immediate::PxGenerateContacts(&ga,&gb,&pa,&pb,&warmCache,1,warm,query,.01f,1.f,warmScratch);
        if(!warm.count)
        {
            if(misses<5) std::printf("[NP_CACHE_MISS] test=%u oldCount=%u p=%.9g,%.9g,%.9g q=%.9g,%.9g,%.9g,%.9g\n",
                test,old.count,previous.p.x,previous.p.y,previous.p.z,previous.q.x,previous.q.y,previous.q.z,previous.q.w);
            ++misses;
        }
    }
    std::printf("[NP_CACHE_MISS_COUNT] misses=%u\n",misses);
    const bool pass=near && generated && output.count>0 && misses==0;
    std::printf("[NP_MISSED_PAIR] query=%.9g near=%u distance=%.9g contacts=%u minGap=%.9g status=%s\n",
        double(query),unsigned(near),double(distance),output.count,double(output.gap),pass?"PASS":"FAIL");
    foundation->release();return pass?0:1;
}
