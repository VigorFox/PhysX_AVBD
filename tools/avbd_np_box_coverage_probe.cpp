// Public NP replay of the first macroscopic crossing in reordered full-five.
#include "PxPhysicsAPI.h"
#include "PxImmediateMode.h"
#include "geomutils/PxContactPoint.h"
#include "geometry/PxGjkQuery.h"
#include "GuPCMBoxProximity.h"
#include <cstdio>
#include <vector>
#include <malloc.h>
using namespace physx;
struct Scratch:PxCacheAllocator
{
    std::vector<void*> data;
    PxU8* allocateCacheData(PxU32 n) PX_OVERRIDE
    {void* p=_aligned_malloc(n,16);data.push_back(p);return static_cast<PxU8*>(p);}
    ~Scratch(){for(void* p:data)_aligned_free(p);}
};
struct Points:immediate::PxContactRecorder
{
    PxU32 count=0;PxReal gap=PX_MAX_F32;
    bool recordContacts(const PxContactPoint* p,PxU32 n,PxU32) PX_OVERRIDE
    {count+=n;for(PxU32 i=0;i<n;++i)gap=PxMin(gap,p[i].separation);return true;}
};
struct Box:PxGjkQuery::Support
{
    PxVec3 half;Box(PxVec3 h):half(h){}
    PxReal getMargin()const PX_OVERRIDE{return 0;}
    PxVec3 supportLocal(const PxVec3& n)const PX_OVERRIDE
    {return PxVec3(n.x>=0?half.x:-half.x,n.y>=0?half.y:-half.y,n.z>=0?half.z:-half.z);}
};
static unsigned checkClosestFeatures()
{
    PxU32 seed=27831;unsigned failures=0;double maximum=0;
    auto random=[&](){seed=1664525u*seed+1013904223u;return float(seed>>8)/16777216.f;};
    for(unsigned i=0;i<1200;++i)
    {
        const PxReal scale=i%3==0?.01f:i%3==1?1.f:100.f;
        const PxVec3 a=PxVec3(.01f+random(),.01f+random(),.01f+random())*scale;
        const PxVec3 b=PxVec3(.01f+random(),.01f+random(),.01f+random())*scale;
        const PxVec3 axis=PxVec3(random(),random(),random()).getNormalized();
        const PxReal angle=i%8==0?0:i%8==1?1.e-7f:random()*PxPi;
        const PxReal reach=a.magnitude()+b.magnitude()+.01f*scale;
        const PxTransform pose(PxVec3(random()-.5f,random()-.5f,random()-.5f).getNormalized()*reach,PxQuat(angle,axis));
        PxVec3 x,y,n;PxReal distance;
        const bool found=Gu::pcmSeparatedBoxProximity(a,b,pose,4*reach,x,y,n,distance);
        if(!found){++failures;continue;}
        // Independent convex optimality certificate: feasible witnesses plus
        // equality of their distance and the supporting-plane lower bound.
        const PxVec3 na=pose.q.rotateInv(n),delta=pose.transform(x)-y;
        const double lower=double(n.dot(pose.p))-double(a.dot(na.abs()))-double(b.dot(n.abs()));
        double violation=PxAbs(double(distance)-lower)/reach;
        violation=PxMax(violation,double((delta-n*distance).magnitude())/reach);
        violation=PxMax(violation,double(PxAbs(n.magnitudeSquared()-1)));
        for(unsigned k=0;k<3;++k)
        {violation=PxMax(violation,double(PxAbs(x[k])-a[k])/reach);violation=PxMax(violation,double(PxAbs(y[k])-b[k])/reach);}
        maximum=PxMax(maximum,violation);failures+=violation>2.e-5;
        PxVec3 xx,yy,nn;PxReal dd;
        const bool reverse=Gu::pcmSeparatedBoxProximity(b,a,pose.getInverse(),4*reach,xx,yy,nn,dd);
        failures+=!reverse || PxAbs(dd-distance)>2.e-5f*reach;
        failures+=Gu::pcmSeparatedBoxProximity(a,b,pose,.5f*distance,xx,yy,nn,dd);
        failures+=Gu::pcmSeparatedBoxProximity(a,b,PxTransform(PxVec3(0),pose.q),4*reach,xx,yy,nn,dd);
    }
    std::printf("[BOX_FEATURE_CERTIFICATE] cases=1200 maximumRelativeViolation=%.9g failures=%u\n",maximum,failures);
    return failures;
}
int main()
{
    PxDefaultAllocator allocator;PxDefaultErrorCallback errors;
    PxFoundation* foundation=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors);
    PxBoxGeometry a(.340000004f,.140000001f,.179999992f),b(.5f,.349999994f,.699999988f);

    const PxTransform capturedA(PxVec3(3.73132944f,1.75415969f,2.09114599f),
        PxQuat(.558359146f,-.504374087f,.528374076f,.393271863f));
    const PxTransform capturedB(PxVec3(4.69251728f,1.87756062f,1.7830267f),
        PxQuat(-.205754742f,.684907138f,.323658347f,.619526088f));
    unsigned failures=checkClosestFeatures();
    const PxReal queries[]={.0256f,.0604604953f,.1f};
    for(unsigned variant=0;variant<4;++variant)
    for(unsigned order=0;order<2;++order)for(unsigned q=0;q<3;++q)
    {
        const PxReal scale=variant==2?.1f:variant==3?10.f:1.f;
        const PxTransform common=variant==1?PxTransform(PxVec3(2,-3,1),PxQuat(.7f,PxVec3(1,2,3).getNormalized())):PxTransform(PxIdentity);
        const PxBoxGeometry boxA(a.halfExtents*scale),boxB(b.halfExtents*scale);
        const Box sa(boxA.halfExtents),sb(boxB.halfExtents);
        const PxGeometry* ga=order?static_cast<const PxGeometry*>(&boxB):&boxA;
        const PxGeometry* gb=order?static_cast<const PxGeometry*>(&boxA):&boxB;
        const PxTransform first=common*PxTransform(capturedA.p*scale,capturedA.q),second=common*PxTransform(capturedB.p*scale,capturedB.q);
        const PxTransform pa=order?second:first,pb=order?first:second;
        const PxReal query=queries[q]*scale;
        PxVec3 x,y,normal;PxReal distance=PX_MAX_F32;
        const bool near=PxGjkQuery::proximityInfo(order?sb:sa,order?sa:sb,pa,pb,query,scale,x,y,normal,distance);
        PxCache cache;Scratch scratch;Points cold;
        const bool generated=immediate::PxGenerateContacts(&ga,&gb,&pa,&pb,&cache,1,cold,query,.01f*scale,scale,scratch);
        unsigned misses=0;PxU32 seed=20261028;
        auto random=[&](){seed=1664525u*seed+1013904223u;return float(seed>>8)*(2.f/16777216.f)-1.f;};
        for(unsigned test=0;test<500;++test)
        {
            const PxVec3 delta=PxVec3(random()*.02f,random()*.02f,random()*.02f)*scale;
            const PxVec3 axis=PxVec3(random(),random(),random()).getNormalized();
            const PxTransform previous(pa.p+delta,PxQuat(random()*.04f,axis)*pa.q);
            PxCache warmCache;Scratch warmScratch;Points old,warm;
            immediate::PxGenerateContacts(&ga,&gb,&previous,&pb,&warmCache,1,old,query,.01f*scale,scale,warmScratch);
            immediate::PxGenerateContacts(&ga,&gb,&pa,&pb,&warmCache,1,warm,query,.01f*scale,scale,warmScratch);
            if(!warm.count && near){if(misses<2)std::printf("[BOX_COVERAGE_MISS] query=%u test=%u old=%u\n",q,test,old.count);++misses;}
        }
        PxCache farCache;Scratch farScratch;Points far;
        const PxTransform farPose(pa.p+PxVec3(10*scale,0,0),pa.q);
        immediate::PxGenerateContacts(&ga,&gb,&farPose,&pb,&farCache,1,far,query,.01f*scale,scale,farScratch);
        const bool pass=generated && near && cold.count && !misses && !far.count && cold.gap<=query && cold.gap>0;failures+=!pass;
        std::printf("[BOX_COVERAGE] variant=%u order=%u query=%.9g distance=%.9g near=%u cold=%u minGap=%.9g warmMisses=%u far=%u status=%s\n",
            variant,order,double(query),double(distance),unsigned(near),cold.count,double(cold.gap),misses,far.count,pass?"PASS":"FAIL");
    }
    foundation->release();return failures?1:0;
}
