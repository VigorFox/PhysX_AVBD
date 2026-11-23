// Public NP regression: an edge-facing convex needs vertices from both faces.
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
int main()
{
 PxDefaultAllocator allocator;Errors errors;
 PxFoundation* foundation=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors);
 PxPhysics* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation,PxTolerancesScale());
 PxVec3 points[8];for(PxU32 i=0;i<8;++i) points[i]=PxVec3(i&1?1.f:-1.f,i&2?1.f:-1.f,i&4?1.f:-1.f);
 PxConvexMeshDesc desc;desc.points.count=8;desc.points.stride=sizeof(PxVec3);desc.points.data=points;desc.flags=PxConvexFlag::eCOMPUTE_CONVEX;
 PxConvexMesh* mesh=PxCreateConvexMesh(PxCookingParams(physics->getTolerancesScale()),desc,physics->getPhysicsInsertionCallback());
 if(!mesh) return 2;
 PxU32 failures=0;
 for(PxU32 i=0;i<3;++i)
 {
  const PxReal angle=PxPi*(.20f+.05f*i);const PxTransform plane(PxIdentity),pose(PxVec3(-.5f,0,0),PxQuat(angle,PxVec3(0,0,1)));
  PxPlaneGeometry a;PxConvexMeshGeometry b(mesh);const PxGeometry* ga=&a;const PxGeometry* gb=&b;
  PxCache cache;CacheAllocator scratch;Recorder contacts;
  const bool ok=immediate::PxGenerateContacts(&ga,&gb,&plane,&pose,&cache,1,contacts,2.f,.01f,1.f,scratch);
  // The two incident faces contain both extremes of the projected cube.
  // This analytic oracle does not inspect polygon indices or PCM internals.
  const PxReal extent=PxSin(angle)+PxCos(angle);
  const bool pass=ok && contacts.count>=4 && PxAbs(contacts.low+extent)<1.e-5f && PxAbs(contacts.high-extent)<1.e-5f;
  failures+=!pass;
  std::printf("[PLANE_CONVEX] angle=%.9g contacts=%u low=%.9g high=%.9g expectedExtent=%.9g status=%s\n",angle,contacts.count,contacts.low,contacts.high,extent,pass?"PASS":"FAIL");
 }
 // Captured clipped hull: selecting the best-aligned face can miss the
 // actual support vertex even at the detection pose, before any solve.
 const PxVec3 clipped[]={
  PxVec3(.300000012f,.300000072f,-.300000012f),PxVec3(-.300000012f,.300000072f,-.300000012f),
  PxVec3(-.300000012f,.300000072f,.300000012f),PxVec3(.300000012f,.300000072f,.300000012f),
  PxVec3(-.256340742f,-.298689365f,-.300000012f),PxVec3(-.259799838f,-.30000031f,-.259798348f),
  PxVec3(-.300000012f,-.299999952f,-.300000012f),PxVec3(.300000012f,-.181667805f,-.300000012f),
  PxVec3(.300000012f,-.182668686f,-.231018096f),PxVec3(.00109833479f,-.248908043f,.00109803677f),
  PxVec3(-.300000012f,-.298689485f,-.256340981f),PxVec3(-.300000012f,-.181667805f,.300000012f),
  PxVec3(-.231018186f,-.182668686f,.300000012f),PxVec3(.284292221f,-.0933675766f,.300000012f),
  PxVec3(.300000012f,-.0842107534f,.300000012f),PxVec3(.300000012f,-.0941814184f,.273657382f),
  PxVec3(.300000012f,-.107788324f,.201077521f),PxVec3(.285103679f,-.0985326767f,.285104334f)};
 desc.points.count=sizeof(clipped)/sizeof(clipped[0]);desc.points.data=clipped;
 PxConvexMesh* captured=PxCreateConvexMesh(PxCookingParams(physics->getTolerancesScale()),desc,physics->getPhysicsInsertionCallback());
 if(!captured) return 2;
 for(PxU32 sample=0;sample<4;++sample)
 {
  const PxTransform common=sample&1?PxTransform(PxVec3(2,-3,1),PxQuat(.7f,PxVec3(1,2,3).getNormalized())):PxTransform(PxIdentity);
  const PxTransform plane=common*PxTransform(PxVec3(0),PxQuat(0,0,.707106829f,.707106829f));
  PxTransform pose=common*PxTransform(PxVec3(.467059016f,.24012652f,.203615651f),PxQuat(.0417353883f,-.015854178f,-.218149275f,.97489363f));
  const PxMeshScale scale=sample&2?PxMeshScale(PxVec3(.7f,1.3f,.9f),PxQuat(.3f,PxVec3(1,1,0).getNormalized())):PxMeshScale();
  auto support=[&]() { PxReal minimum=PX_MAX_F32;for(PxU32 v=0;v<captured->getNbVertices();++v) minimum=PxMin(minimum,plane.transformInv(pose.transform(scale.toMat33()*captured->getVertices()[v])).x);return minimum; };
  if(sample&2) pose.p+=plane.q.getBasisVector0()*(-.0353189f-support());
  const PxReal minimum=support();
  PxPlaneGeometry a;PxConvexMeshGeometry b(captured,scale);const PxGeometry* ga=&a;const PxGeometry* gb=&b;
  PxCache cache;CacheAllocator scratch;Recorder contacts;
  const bool generated=immediate::PxGenerateContacts(&ga,&gb,&plane,&pose,&cache,1,contacts,.024f,.01f,1.f,scratch);
  const bool supportPass=generated && contacts.count && PxAbs(contacts.separation-minimum)<2.e-6f;
  failures+=!supportPass;
  std::printf("[PLANE_CONVEX_SUPPORT] sample=%u vertices=%u contacts=%u minimum=%.9g reported=%.9g status=%s\n",sample,captured->getNbVertices(),contacts.count,minimum,contacts.separation,supportPass?"PASS":"FAIL");
 }
 captured->release();mesh->release();physics->release();foundation->release();return failures?1:0;
}
