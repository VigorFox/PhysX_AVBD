// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include "avbd/DyKernelTable.h"
#include "avbd/DyAvbdContactPotential.h"
#include "avbd_probe_foundation.h"
#include <cstdio>

using namespace physx;
using namespace physx::Dy;
struct Support { PxU32 block; PxReal weight; };

static bool report(const char* name,bool pass)
{
 std::printf("[CONTACT_FRAME_TEST] case=%s status=%s\n",name,pass ? "PASS" : "FAIL");
 return pass;
}

int main()
{
 AvbdProbeFoundation foundation;
 const KernelTable kernels=KernelTable::makeAvbdCpuScalar();
 AvbdContactFrameGeometry geometry={};geometry.normalLocal=PxVec3(0,0,1);
 PxVec3d points[5]={PxVec3d(.2,.15,.01),PxVec3d(.2,.15,0),
  PxVec3d(0),PxVec3d(1,0,0),PxVec3d(0,1,0)};
 const auto eval=[&](const AvbdContactFrameGeometry& g,const PxVec3d* p,double offset)
 {
  AvbdContactFrameEvaluation value;
  AvbdContactFrameArgs args={&g,p,offset,&value};kernels.launch(AVBD_CONTACT_FRAME,&args);
  return value;
 };
 bool all=report("scalar_table_binding",kernels.isBound(AVBD_CONTACT_FRAME));
 const AvbdContactFrameEvaluation initial=eval(geometry,points,-.04);
 const PxReal before=avbdUnilateralContactPotential(PxReal(initial.gap),1e5f,0,-PX_MAX_F32).energy;
 const double angle=.63,cs=PxCos(angle),sn=PxSin(angle);
 const PxMat33d rotation(PxVec3d(cs,0,-sn),PxVec3d(0,1,0),PxVec3d(sn,0,cs));
 PxVec3d rotated[5];for(PxU32 i=0;i<5;++i) rotated[i]=rotation*points[i];
 const AvbdContactFrameEvaluation turned=eval(geometry,rotated,-.04);
 const AvbdContactPotential potential=avbdUnilateralContactPotential(PxReal(turned.gap),1e5f,0,-PX_MAX_F32);
 PxVec3d force(0),moment(0);
 for(PxU32 i=0;i<5;++i)
 { const PxVec3d f=turned.gradients[i]*(-double(potential.gradient));force+=f;moment+=rotated[i].cross(f); }
 all &= report("finite_common_rotation_energy_and_wrench",initial.valid && turned.valid &&
  PxAbs(before-potential.energy)<1e-5 && force.magnitude()<1e-10 && moment.magnitude()<1e-10);
 std::printf("[CONTACT_FRAME_VALUES] energyBefore=%.9g energyAfter=%.9g netForce=%.9g netMoment=%.9g sceneQualification=0\n",
  double(before),double(potential.energy),force.magnitude(),moment.magnitude());
 for(PxU32 i=0;i<5;++i) rotated[i]+=PxVec3d(3,-2,1);
 const AvbdContactFrameEvaluation translated=eval(geometry,rotated,-.04);
 all &= report("common_translation",translated.valid && PxAbs(translated.gap-turned.gap)<1e-14);

 geometry.normalLocal=PxVec3(.3f,.4f,PxSqrt(.75f));
 points[0]=PxVec3d(.32,.05,-.07);points[3]=PxVec3d(1.1,.1,.03);points[4]=PxVec3d(-.05,.8,.07);
 const AvbdContactFrameEvaluation deformed=eval(geometry,points,-.4);
 double maximumError=0;
 for(PxU32 point=0;point<5;++point) for(PxU32 axis=0;axis<3;++axis)
 {
  const double original=points[point][axis],h=1e-6;
  points[point][axis]=original+h;const auto plus=eval(geometry,points,-.4);
  points[point][axis]=original-h;const auto minus=eval(geometry,points,-.4);
  points[point][axis]=original;
  maximumError=PxMax(maximumError,PxAbs((plus.gap-minus.gap)/(2*h)-deformed.gradients[point][axis]));
 }
 all &= report("all_fifteen_geometric_derivatives",deformed.valid && maximumError<1e-9 && deformed.gradients[2].magnitude()>1e-3);

 // The same carrier's two tangential coordinates have their own complete
 // derivatives. A common rigid velocity must not look like frictional slip.
 const PxVec3 axes[3]={PxVec3(1,0,0),PxVec3(0,1,0),PxVec3(0,0,1)};
 bool virtualWork=true;
 for(PxU32 axis=0;axis<3;++axis)
 {
  AvbdContactFrameGeometry tangent=geometry;tangent.normalLocal=axes[axis];
  const auto value=eval(tangent,points,0);
  double velocity=0;PxVec3d net(0),torque(0);
  for(PxU32 i=0;i<5;++i)
  {
   velocity+=value.gradients[i].dot(PxVec3d(.2,.3,-.1)+PxVec3d(.7,-.5,.9).cross(points[i]));
   net+=value.gradients[i];torque+=points[i].cross(value.gradients[i]);
  }
  virtualWork &= value.valid && PxAbs(velocity)<1e-12 && net.magnitude()<1e-12 && torque.magnitude()<1e-12;
 }
 all &= report("normal_and_tangent_rigid_virtual_work",virtualWork);

 // The target witness and frame share blocks. Query embedding has a negative
 // weight. A block derivative must include every occurrence in all segments.
 const Support supports[8]={{3,1.25f},{4,-.25f},{0,.5f},{1,.25f},{2,.25f},{0,1},{1,1},{2,1}};
 const PxU16 ends[5]={2,5,6,7,8};for(PxU32 i=0;i<5;++i) geometry.supportEnds[i]=ends[i];
 PxVec3 particles[5]={PxVec3(0),PxVec3(1.1f,.1f,.03f),PxVec3(-.05f,.8f,.07f),PxVec3(.32f,.05f,-.07f),PxVec3(.28f,.09f,-.05f)};
 const auto sample=[&]()
 {
  PxVec3d embedded[5];avbdContactFramePoints(geometry,supports,
   [&](const Support& s){return particles[s.block];},embedded);
  return eval(geometry,embedded,-.4);
 };
 const auto embedded=sample();double maxEmbeddedError=0;
 for(PxU32 block=0;block<5;++block)
 {
  const PxVec3d gradient=avbdContactFrameBlockGradient(geometry,supports,embedded,block);
  for(PxU32 axis=0;axis<3;++axis)
  {
   const PxReal original=particles[block][axis];
   particles[block][axis]=original+1e-4f;const double high=particles[block][axis];const auto plus=sample();
   particles[block][axis]=original-1e-4f;const double low=particles[block][axis];const auto minus=sample();
   particles[block][axis]=original;
   maxEmbeddedError=PxMax(maxEmbeddedError,PxAbs((plus.gap-minus.gap)/(high-low)-gradient[axis]));
  }
 }
 all &= report("signed_embedding_and_shared_block_derivatives",embedded.valid && maxEmbeddedError<1e-7);
 points[3]=points[2];all &= report("zero_edge_rejected",!eval(geometry,points,0).valid);
 points[3]=points[2]+PxVec3d(1,0,0);points[4]=points[2]+PxVec3d(2,0,0);
 all &= report("collinear_frame_rejected",!eval(geometry,points,0).valid);
 std::printf("[CONTACT_FRAME_ERRORS] geometric=%.9g embedded=%.9g\n",maximumError,maxEmbeddedError);
 return all ? 0 : 1;
}
