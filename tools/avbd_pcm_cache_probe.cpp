// Public NP regression: captured five-body cache misses an 11 mm overlap.
#include "PxPhysicsAPI.h"
#include "PxImmediateMode.h"
#include "geomutils/PxContactPoint.h"
#include <cstdio>
#include <vector>
#include <malloc.h>
using namespace physx;
struct Errors : PxErrorCallback
{
 void reportError(PxErrorCode::Enum code,const char* message,const char*,int) PX_OVERRIDE
 { std::fprintf(stderr,"[PHYSX_ERROR] code=%u message=%s\n",unsigned(code),message); }
};
struct CacheAllocator : PxCacheAllocator
{
 std::vector<void*> allocations;
 PxU8* allocateCacheData(PxU32 n) PX_OVERRIDE { void* p=_aligned_malloc(n,16);allocations.push_back(p);return static_cast<PxU8*>(p); }
 ~CacheAllocator() { for(void* p:allocations) _aligned_free(p); }
};
struct Recorder : immediate::PxContactRecorder
{
 PxReal low=PX_MAX_F32,high=-PX_MAX_F32,separation=PX_MAX_F32;PxU32 count=0;
 bool recordContacts(const PxContactPoint* p,PxU32 n,PxU32) PX_OVERRIDE
 { for(PxU32 i=0;i<n;++i) {low=PxMin(low,p[i].point.y);high=PxMax(high,p[i].point.y);separation=PxMin(separation,p[i].separation);}count+=n;return true; }
};
#include "../physx/snippets/snippetjointsoftbody/SnippetJointSoftBodyGeometry.h"
#include "GuPCMShapeConvex.h"

static bool radiusBounds()
{
 // Exercise the production radius on eccentric hull bounds and rotated,
 // nonuniform mesh scales. The oracle transforms all eight bound corners.
 Gu::ConvexHullData hull;PxMemZero(&hull,sizeof(hull));
 hull.mAABB.mCenter=PxVec3(1.7f,-.8f,.4f);hull.mAABB.mExtents=PxVec3(.3f,.7f,.2f);
 bool pass=true;PxReal minimumSlack=PX_MAX_F32;
 for(PxU32 i=0;i<50;++i)
 {
  const PxMeshScale scale(PxVec3(.1f+.07f*i,1.3f,.9f),PxQuat(.11f*i,PxVec3(1,2,3).getNormalized()));
  PxReal radius;aos::FStore(Gu::pcmConvexRotationRadius(hull,aos::V3LoadU(scale.scale)),&radius);
  for(PxU32 v=0;v<8;++v)
  {
   const PxVec3 e=hull.mAABB.mExtents;
   const PxVec3 corner=hull.mAABB.mCenter+PxVec3(v&1?e.x:-e.x,v&2?e.y:-e.y,v&4?e.z:-e.z);
   const PxReal slack=radius-(scale.toMat33()*corner).magnitude();minimumSlack=PxMin(minimumSlack,slack);
   pass &= slack>=-8*PX_EPS_F32*radius;
  }
 }
 std::printf("[PCM_RADIUS_TEST] cases=400 minimumSlack=%.9g status=%s\n",minimumSlack,pass?"PASS":"FAIL");
 return pass;
}
static PxConvexMesh* cook(PxPhysics& physics,const PxVec3* points,PxU32 count)
{
 PxConvexMeshDesc desc;desc.points.count=count;desc.points.stride=sizeof(PxVec3);desc.points.data=points;desc.flags=PxConvexFlag::eCOMPUTE_CONVEX;
 return PxCreateConvexMesh(PxCookingParams(physics.getTolerancesScale()),desc,physics.getPhysicsInsertionCallback());
}
int main()
{
 PxDefaultAllocator allocator;Errors errors;
 PxFoundation* foundation=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors);
 PxPhysics* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation,PxTolerancesScale());
 const PxVec3 points[]={
  PxVec3(0.300000072f,-0.300000012f,0.300000012f),
  PxVec3(0.300000072f,-0.300000012f,-0.300000012f),
  PxVec3(0.300000072f,0.300000012f,-0.300000012f),
  PxVec3(0.300000072f,0.300000012f,0.300000012f),
  PxVec3(-0.299999952f,0.300000012f,-0.300000012f),
  PxVec3(-0.188268423f,-0.300000012f,-0.300000012f),
  PxVec3(-0.299999952f,0.261711568f,-0.300000012f),
  PxVec3(-0.189452767f,-0.300000012f,0.00109872222f),
  PxVec3(-0.174368501f,-0.300000012f,0.104831517f),
  PxVec3(-0.215941906f,-0.0968393683f,0.0968390405f),
  PxVec3(-0.248908162f,-0.00109821558f,0.00109773874f),
  PxVec3(-0.226487875f,-0.00109833479f,0.113812119f),
  PxVec3(-0.299999952f,0.300000012f,-0.261711597f),
  PxVec3(-0.188268542f,0.300000012f,0.300000012f),
  PxVec3(-0.0941810608f,-0.273658812f,0.300000012f),
  PxVec3(-0.0842108727f,-0.300000012f,0.300000012f),
  PxVec3(-0.188545108f,0.229683638f,0.300000012f),
  PxVec3(-0.299999952f,0.260939181f,-0.260939181f),
  PxVec3(-0.104194403f,-0.300000012f,0.265720069f)};
 const PxVec3 half=PxVec3(0.359999985f,0.129999995f,0.179999992f);
 const PxTransform originalA[]={PxTransform(PxVec3(-1.16424417f,1.62620711f,0.264017403f),PxQuat(0.0981344059f,0.0554191396f,0.71466279f,0.690330029f)),PxTransform(PxVec3(-1.13416469f,1.64978743f,0.271553874f),PxQuat(0.0986124352f,0.0556390546f,0.722688556f,0.681836665f))};
 const PxTransform originalB[]={PxTransform(PxVec3(-0.778422773f,1.57140315f,0.242906213f),PxQuat(0.0174131449f,0.00272925943f,-0.0871675164f,0.996037722f)),PxTransform(PxVec3(-0.748772144f,1.59471405f,0.242478445f),PxQuat(0.0153022595f,0.00537186489f,-0.0955819786f,0.995289505f))};
 PxConvexMesh* mesh=cook(*physics,points,sizeof(points)/sizeof(points[0]));
 PxVec3 boxPoints[8];for(PxU32 v=0;v<8;++v) boxPoints[v]=PxVec3(v&1?half.x:-half.x,v&2?half.y:-half.y,v&4?half.z:-half.z);
 PxConvexMesh* boxMesh=cook(*physics,boxPoints,8);
 if(!mesh || !boxMesh) return 2;
 PxU32 failures=radiusBounds()?0:1;
 for(PxU32 convexBox=0;convexBox<2;++convexBox) for(PxU32 variant=0;variant<4;++variant)
 {
  const PxReal scale=variant==2?.25f:variant==3?3.f:1.f;
  const PxTransform common=variant==1?PxTransform(PxVec3(2,-3,1),PxQuat(.7f,PxVec3(1,2,3).getNormalized())):PxTransform(PxIdentity);
  PxBoxGeometry box(half*scale);PxConvexMeshGeometry convex(mesh,PxMeshScale(scale)),other(boxMesh,PxMeshScale(scale));
  const PxGeometry* ga=convexBox?static_cast<const PxGeometry*>(&other):static_cast<const PxGeometry*>(&box);
  const PxGeometry* gb=&convex;
  PxTransform a[2],b[2];for(PxU32 frame=0;frame<2;++frame)
  {a[frame]=common*PxTransform(originalA[frame].p*scale,originalA[frame].q);b[frame]=common*PxTransform(originalB[frame].p*scale,originalB[frame].q);}
  PxCache warmCache,coldCache;CacheAllocator scratch;Recorder prime,warm,cold;
  immediate::PxGenerateContacts(&ga,&gb,&a[0],&b[0],&warmCache,1,prime,.024f*scale,.01f*scale,scale,scratch);
  const bool generated=immediate::PxGenerateContacts(&ga,&gb,&a[1],&b[1],&warmCache,1,warm,.024f*scale,.01f*scale,scale,scratch);
  const bool fresh=immediate::PxGenerateContacts(&ga,&gb,&a[1],&b[1],&coldCache,1,cold,.024f*scale,.01f*scale,scale,scratch);
  const PxReal depth=JointSoftBodySnippet::observedPenetration(*ga,a[1],*gb,b[1]);
  const bool pass=generated && fresh && warm.count && cold.count && depth>.01f*scale &&
   warm.separation<=-depth+2e-5f*scale && cold.separation<=-depth+2e-5f*scale;
  failures+=!pass;
  std::printf("[PCM_CACHE_TEST] convexBox=%u variant=%u scale=%.9g depth=%.9g cold=%.9g warm=%.9g contacts=%u status=%s\n",convexBox,variant,scale,depth,cold.separation,warm.separation,warm.count,pass?"PASS":"FAIL");
 }
 boxMesh->release();mesh->release();physics->release();foundation->release();return failures?1:0;
}
