// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Replay a captured solver state. REPRODUCED is evidence of the old defect,
// not a qualification of the current contact compiler or a physical PASS.
#include "avbd/DyAvbdOgc.h"
#include "avbd/DyAvbdContactGeometry.h"
#include "avbd_probe_foundation.h"
#include <cstdio>
#include <fstream>

using namespace physx;
using namespace physx::Dy;

static void readPoint(std::istream& input,PxVec3& point)
{
 input>>point.x>>point.y>>point.z;
}

int main(int argc,char** argv)
{
 if(argc!=2) return 2;
 std::ifstream input(argv[1]);
 if(!input) return 2;
 AvbdProbeFoundation foundation;
 PxVec3 current,anchor,delta;
 PxMat33 metric;
 readPoint(input,current);readPoint(input,anchor);readPoint(input,delta);
 readPoint(input,metric.column0);readPoint(input,metric.column1);readPoint(input,metric.column2);
 AvbdOgcPointRegion region;avbdOgcResetPointRegion(region,anchor);
 PxU32 count=0;input>>count;
 for(PxU32 i=0;i<count;++i)
 {
  PxVec3 normal;PxReal bound;readPoint(input,normal);input>>bound;
  AvbdOgcHalfSpace plane;
  avbdOgcWriteHalfSpace(plane,normal,bound,PX_MAX_U32,PxVec3(0),PxVec3(0),
   PX_MAX_U32,PX_MAX_U32,PX_MAX_U32,PX_MAX_U32);
  region.planes.pushBack(plane);
 }
 region.planeCount=region.planes.size();
 AvbdOgcProjectionPlanes domain;input>>count;
 for(PxU32 i=0;i<count;++i)
 {
  AvbdOgcProjectionPlane plane;
  input>>plane.n.x>>plane.n.y>>plane.n.z>>plane.b>>plane.roundoff;
  domain.pushBack(plane);
 }
 PxVec3 projected=current+delta,unguarded=projected;
 const AvbdOgcClip full=avbdOgcTruncatePoint(region,projected,&domain,&metric,&current);
 AvbdOgcPointRegion freeRegion;avbdOgcResetPointRegion(freeRegion,anchor);
 const AvbdOgcClip free=avbdOgcTruncatePoint(freeRegion,unguarded,&domain,&metric,&current);
 const PxReal fullStep=(projected-current).magnitude(),freeStep=(unguarded-current).magnitude();
 const bool blocked=full.feasible && free.feasible && fullStep<0.0002f && freeStep>0.09f;
 std::printf("[RECOVERY_REPLAY] case=captured_projection status=%s planes=%u projectedStep=%.9g noOgcStep=%.9g\n",
  blocked ? "REPRODUCED" : "MISMATCH",region.planeCount,fullStep,freeStep);

 PxU32 pairCount=0,unproven=0;double minimum=DBL_MAX;
 input>>pairCount;
 for(PxU32 pair=0;pair<pairCount;++pair)
 {
  PxVec3 a0[3],b0[3],a1[3],b1[3];
  for(PxU32 i=0;i<3;++i) readPoint(input,a0[i]);
  for(PxU32 i=0;i<3;++i) readPoint(input,b0[i]);
  for(PxU32 i=0;i<3;++i) readPoint(input,a1[i]);
  for(PxU32 i=0;i<3;++i) readPoint(input,b1[i]);
  const AvbdTrianglePairPlane planes[2]={avbdTrianglePairPlane(a0,b0,a0[0]-b0[0]),
   avbdTrianglePairPlane(a1,b1,a1[0]-b1[0])};
  double certificate=-DBL_MAX;
  for(PxU32 axis=0;axis<2;++axis)
  {
   const PxVec3d n=avbdOgcDouble(planes[axis].normal);
   double gap=DBL_MAX;
   for(PxU32 a=0;a<3;++a) for(PxU32 b=0;b<3;++b)
   {
    gap=PxMin(gap,n.dot(avbdOgcDouble(a0[a])-avbdOgcDouble(b0[b])));
    gap=PxMin(gap,n.dot(avbdOgcDouble(a1[a])-avbdOgcDouble(b1[b])));
   }
   certificate=PxMax(certificate,gap);
  }
  minimum=PxMin(minimum,certificate);
  if(certificate<=1e-6) ++unproven;
 }
 const bool separated=pairCount==5760 && !unproven;
 std::printf("[RECOVERY_REPLAY] case=captured_dcd_to_candidate status=%s pairs=%u unproven=%u minGap=%.9g\n",
  separated ? "CERTIFIED" : "UNPROVEN",pairCount,unproven,minimum);
 if(!input) return 2;
 return blocked && separated ? 0 : 1;
}
