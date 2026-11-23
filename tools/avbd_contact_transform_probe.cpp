// Captured admission/publication counterexample, evaluated through production geometry.
#include "avbd/DyAvbdContactGeometry.h"
#include <cstdio>
using namespace physx;
using namespace physx::Dy;

PX_NOINLINE static PxVec3 particleView(const PxVec3& p, const PxQuat& q, const PxVec3& local)
{
 return avbdContactWorldPoint(p,q,local);
}

PX_NOINLINE static double rigidView(const PxVec3& p, const PxQuat& q,
 const PxVec3& local, const PxVec3& n)
{
 const PxVec3 point=avbdContactWorldPoint(p,q,local);
 return double(n.x)*point.x+double(n.y)*point.y+double(n.z)*point.z;
}

int main()
{
 const PxVec3 p(6.3549957275390625f,2.4314084053039551f,.31319159269332886f);
 const PxQuat q(-.021783111616969109f,.19508080184459686f,-.0068535073660314083f,.98052132129669189f);
 const PxVec3 local(.49999973177909851f,.34999990463256836f,.69999992847442627f);
 const PxVec3 normalLocal(.25103813409805298f,.96775263547897339f,.020852740854024887f);
 const PxVec3 n=avbdContactWorldPoint(PxVec3(0),q,normalLocal);
 const PxVec3 point=particleView(p,q,local);
 const double before=double(n.x)*point.x+double(n.y)*point.y+double(n.z)*point.z;
 const double after=rigidView(p,q,local,n);
 // Independent high-precision rotation, with the same float-authored quaternion.
 const PxVec3d v(q.x,q.y,q.z),r(local.x,local.y,local.z);
 const PxVec3d reference=PxVec3d(p.x,p.y,p.z)+r*(2*double(q.w)*q.w-1)+
  v.cross(r)*(2*double(q.w))+v*(2*v.dot(r));
 const double error=(PxVec3d(point.x,point.y,point.z)-reference).magnitude();
 const bool pass=before==after && error<8*PX_EPS_F32*(1+reference.magnitude());
 std::printf("[CONTACT_TRANSFORM_TEST] case=captured_publication status=%s pointError=%.17g callerGapDifference=%.17g\n",
  pass?"PASS":"FAIL",error,after-before);
 return pass?0:1;
}
