// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include "avbd/DyAvbdContactGeometry.h"
#include "avbd/DyKernelTable.h"
#include "avbd_probe_foundation.h"
#include <cstdio>

using namespace physx;
using namespace physx::Dy;

struct Support { PxU32 block; PxReal weight; };

struct Fixture
{
 AvbdTriangleDomainGeometry geometry;
 Support supports[12];
 PxVec3 positions[12];
 KernelTable kernels;
 Fixture():kernels(KernelTable::makeAvbdCpuScalar())
 {
  geometry.normal=PxVec3(0,0,1);geometry.lowerGap=0;
  const PxVec3 vertices[3]={PxVec3(0,0,0),PxVec3(1,0,0),PxVec3(0,1,0)};
  for(PxU32 i=0;i<6;++i)
  {
   geometry.supportEnds[i]=PxU16(2*i+2);
   supports[2*i]={2*i,1.25f};supports[2*i+1]={2*i+1,-0.25f};
   positions[2*i]=positions[2*i+1]=vertices[i%3]+PxVec3(0,0,i<3 ? 1.0f : 0.0f);
  }
 }
 void compile(PxU32 block,AvbdOgcProjectionPlane (&planes)[9]) const
 {
  PxVec3d points[6],scales[6];double weights[6];
  avbdTriangleDomainPoints(geometry,supports,
   [&](const Support& support){return positions[support.block];},block,points,scales,weights);
  AvbdTriangleDomainArgs args={&geometry,points,scales,weights,planes};
  kernels.launch(AVBD_TRIANGLE_DOMAIN,&args);
 }
};

static bool report(const char* name,bool pass)
{
 std::printf("[TRIANGLE_DOMAIN_TEST] case=%s status=%s\n",name,pass ? "PASS" : "FAIL");
 return pass;
}

static bool checkFactoredScreen()
{
 KernelTable kernels=KernelTable::makeAvbdCpuScalar();
 if(!kernels.isBound(AVBD_TRIANGLE_DOMAIN_SCREEN)) return false;
 PxU32 seed=713u,screened=0,boundaries=0;
 auto random=[&]()->double
 { seed=1664525u*seed+1013904223u;return double(seed&0xffffffu)/8388608.0-1.0; };
 for(PxU32 sample=0;sample<20000;++sample)
 {
  AvbdTriangleDomainGeometry geometry={};
  geometry.normal=PxVec3(PxReal(random()),PxReal(random()),PxReal(random())).getNormalized();
  const PxVec3d normal(geometry.normal.x,geometry.normal.y,geometry.normal.z);
  const double extent=std::pow(10.0,double(sample%13)-6.0);
  geometry.lowerGap=sample%5==0 ? PxReal(-extent*0.25) : 0.0f;
  PxVec3d points[6],scales[6];double weights[6];
  AvbdTrianglePointProjection projections[6];
  for(PxU32 i=0;i<6;++i)
  {
   const PxVec3d a(random()*extent,random()*extent,random()*extent);
   const PxVec3d b(random()*extent,random()*extent,random()*extent);
   points[i]=a*1.25-b*0.25+normal*(i<3 ? 4.0*extent : -4.0*extent);
   scales[i]=a.abs()*1.25+b.abs()*0.25+normal.abs()*(4.0*extent);
   weights[i]=sample%3==0 ? random()*2.0 : (i==sample%6 ? -0.25 : 0.0);
   geometry.projectionIndices[i]=i;
   projections[i]={geometry.normal,i,normal.dot(points[i]),normal.abs().dot(scales[i])};
  }
  AvbdOgcProjectionPlane planes[9];
  avbdTriangleDomainPlanes(geometry,points,scales,weights,planes);
  PxVec3d delta(random()*extent,random()*extent,random()*extent);
  // Force candidates onto, immediately inside and immediately outside a
  // raw boundary, including negative weights and large translated inputs.
  if(sample%2==0 && planes[0].n.magnitudeSquared()>0.0)
  {
   const double target=sample%6==0 ? std::nextafter(planes[0].b,DBL_MAX) :
    sample%6==2 ? planes[0].b : std::nextafter(planes[0].b,-DBL_MAX);
   delta=planes[0].n*(target/planes[0].n.magnitudeSquared());
   ++boundaries;
  }
  bool separated=false;
  AvbdTriangleDomainScreenArgs args={&geometry,projections,weights,&delta,&separated};
  kernels.launch(AVBD_TRIANGLE_DOMAIN_SCREEN,&args);
  AvbdTriangleProjectionRef refs[18];
  // Repeated and permuted factors represent the same Cartesian product;
  // this also exercises the variable-size batch binding used in production.
  for(PxU32 i=0;i<18;++i)
  { const PxU32 point=i<9 ? (8-i)%3 : 3+(17-i)%3;refs[i]={point,weights[point]}; }
  bool batchSeparated=false;
  AvbdTriangleDomainScreenArgs batch={&geometry,projections,NULL,&delta,&batchSeparated};
  batch.refs=refs;batch.countA=9;batch.count=18;
  kernels.launch(AVBD_TRIANGLE_DOMAIN_SCREEN,&batch);
  if(batchSeparated!=separated) return false;
  if(!separated) continue;
  ++screened;
  for(PxU32 i=0;i<9;++i)
   if(planes[i].n.dot(delta)<planes[i].b) return false;
 }
 return screened>1000 && boundaries>1000;
}

static bool checkIntervalScreen()
{
 PxU32 seed=71371,accepted=0,boundaryAccepted=0;
 auto random=[&](){ seed=1664525u*seed+1013904223u;return float(seed&0xffffffu)/8388608.0f-1.0f; };
 KernelTable kernels=KernelTable::makeAvbdCpuScalar();
 for(PxU32 sample=0;sample<24000;++sample)
 {
  AvbdTriangleDomainGeometry g={};Support supports[18];PxVec3 positions[13];
  const float extent=std::pow(10.0f,float(sample%13)-6.0f);
  g.normal=PxVec3(random(),random(),random()).getNormalized();
  const PxVec3d n(g.normal.x,g.normal.y,g.normal.z);
  for(PxU32 i=0;i<13;++i) positions[i]=PxVec3(random(),random(),random())*extent;
  for(PxU32 i=0;i<6;++i)
  {
   // Duplicate moving supports include exact signed cancellation; the
   // absolute input scale still changes when their summed weight is zero.
   supports[3*i]={12,1.25f};supports[3*i+1]={12,sample%3==0 ? -1.25f : -0.25f};
   supports[3*i+2]={i,1.0f};g.supportEnds[i]=PxU16(3*i+3);g.projectionIndices[i]=i;
  }
  PxVec3d oldPoints[6],oldScales[6];double weights[6],absWeights[6];
  avbdTriangleDomainPoints(g,supports,[&](const Support& s){return positions[s.block];},12,oldPoints,oldScales,weights);
  AvbdTrianglePointProjection projections[6];
  for(PxU32 i=0;i<6;++i)
  { projections[i]={g.normal,i,n.dot(oldPoints[i]),n.abs().dot(oldScales[i])};absWeights[i]=1.25+PxAbs(double(supports[3*i+1].weight)); }
  const PxVec3 candidate=positions[12]+PxVec3(random(),random(),random())*(extent*0.01f);
  const PxVec3d delta=avbdOgcDouble(candidate)-avbdOgcDouble(positions[12]);
  PxVec3d points[6],scales[6];double zero[6];
  avbdTriangleDomainPoints(g,supports,[&](const Support& s){return s.block==12 ? candidate : positions[s.block];},PX_MAX_U32,points,scales,zero);
  double minimum=DBL_MAX;
  for(PxU32 a=0;a<3;++a) for(PxU32 b=3;b<6;++b) minimum=PxMin(minimum,n.dot(points[a]-points[b]));
  g.lowerGap=PxReal(minimum+double(extent)*double(PxI32(sample%7)-3)*FLT_EPSILON);
  bool admitted=false;
  AvbdTriangleDomainScreenArgs args={&g,projections,weights,&delta,&admitted};
  args.absoluteWeights=absWeights;args.inputIntervals=true;kernels.launch(AVBD_TRIANGLE_DOMAIN_SCREEN,&args);
  if(!admitted) continue;
  ++accepted;if(double(g.lowerGap)>minimum) ++boundaryAccepted;
  AvbdOgcProjectionPlane planes[9];avbdTriangleDomainPlanes(g,points,scales,zero,planes);
  if(!avbdTriangleDomainAdmits(planes,9,PxVec3d(0.0))) return false;
 }
 return accepted>1000 && boundaryAccepted>1000;
}

static bool checkMotionBound()
{
 PxU32 seed=13301u,accepted=0;
 auto random=[&](){ seed=1664525u*seed+1013904223u;return float(seed&0xffffffu)/8388608.0f-1.0f; };
 for(PxU32 sample=0;sample<20000;++sample)
 {
  Fixture f;f.geometry.normal=PxVec3(random(),random(),random()).getNormalized();
  const PxVec3d n=avbdOgcDouble(f.geometry.normal);const double norm=n.abs().x+n.abs().y+n.abs().z;
  const float extent=std::pow(10.0f,float(sample%13)-6.0f);
  for(PxU32 i=0;i<12;++i) f.positions[i]=PxVec3(random(),random(),random())*extent;
  PxVec3d points[6],scales[6];double weights[6];
  avbdTriangleDomainPoints(f.geometry,f.supports,[&](const Support& s){return f.positions[s.block];},PX_MAX_U32,points,scales,weights);
  double minimum=DBL_MAX,maximum=-DBL_MAX,scaleA=0.0,scaleB=0.0;
  for(PxU32 i=0;i<6;++i)
  {
   const double scale=n.abs().dot(scales[i]);
   const double value=n.dot(points[i])+(i<3 ? 1.0 : -1.0)*4.0*FLT_EPSILON*scale;
   if(i<3) { minimum=PxMin(minimum,value);scaleA=PxMax(scaleA,scale); }
   else { maximum=PxMax(maximum,value);scaleB=PxMax(scaleB,scale); }
  }
  double motion=0.0;
  for(PxU32 i=0;i<12;++i)
  {
   const PxVec3 before=f.positions[i];f.positions[i]+=PxVec3(random(),random(),random())*(extent*0.01f);
   const PxVec3d delta=avbdOgcDouble(f.positions[i])-avbdOgcDouble(before);
   motion=PxMax(motion,PxMax(PxAbs(delta.x),PxMax(PxAbs(delta.y),PxAbs(delta.z))));
  }
  // Independent query displacement of a negative embedding support.
  const PxVec3 before=f.positions[1];f.positions[1]+=PxVec3(random(),random(),random())*(extent*0.01f);
  const PxVec3d delta=avbdOgcDouble(f.positions[1])-avbdOgcDouble(before);
  const double local=PxMax(PxAbs(delta.x),PxMax(PxAbs(delta.y),PxAbs(delta.z)));
  avbdTriangleDomainPoints(f.geometry,f.supports,[&](const Support& s){return f.positions[s.block];},PX_MAX_U32,points,scales,weights);
  double actual=DBL_MAX;
  for(PxU32 a=0;a<3;++a) for(PxU32 b=3;b<6;++b) actual=PxMin(actual,n.dot(points[a]-points[b]));
  f.geometry.lowerGap=PxReal(actual+extent*0.25*double(PxI32(sample%3)-1));
  const double lower=f.geometry.lowerGap;
  AvbdTriangleBatchBound bound={minimum-maximum-lower+4.0*FLT_EPSILON*PxAbs(lower),scaleA+scaleB+PxAbs(lower),norm*3.0,norm*0.25};
  if(!bound.admits(motion,local)) continue;
  ++accepted;AvbdOgcProjectionPlane planes[9];
  avbdTriangleDomainPlanes(f.geometry,points,scales,weights,planes);
  if(!avbdTriangleDomainAdmits(planes,9,PxVec3d(0.0))) return false;
 }
 return accepted>2000;
}

int main()
{
 AvbdProbeFoundation foundation;
 Fixture fixture;
 bool all=report("scalar_table_binding",fixture.kernels.isBound(AVBD_TRIANGLE_DOMAIN));
 all &= report("factored_screen_signed_scale_boundary_oracle",checkFactoredScreen());
 all &= report("interval_screen_actual_candidate_signed_cancellation",checkIntervalScreen());
 all &= report("snapshot_bound_arbitrary_support_and_query_motion",checkMotionBound());
 bool jacobian=true,negative=false;
 for(PxU32 block=0;block<12;++block)
 {
  AvbdOgcProjectionPlane before[9];fixture.compile(block,before);
  for(PxU32 axis=0;axis<3;++axis)
  {
   const PxReal original=fixture.positions[block][axis];
   fixture.positions[block][axis]+=0.01f;
   const double actual=double(fixture.positions[block][axis])-original;
   AvbdOgcProjectionPlane after[9];fixture.compile(block,after);
   for(PxU32 i=0;i<9;++i)
    jacobian &= PxAbs(after[i].b-before[i].b+before[i].n[axis]*actual)<1e-12;
   fixture.positions[block][axis]=original;
  }
  if(block==1) negative=before[0].n.z==-0.25;
 }
 all &= report("all_signed_support_jacobians",jacobian && negative);

 AvbdOgcProjectionPlane original[9];fixture.compile(PX_MAX_U32,original);
 const PxVec3 shift(3,-5,-2);
 for(PxU32 i=0;i<12;++i) fixture.positions[i]+=shift;
 AvbdOgcProjectionPlane translated[9];fixture.compile(PX_MAX_U32,translated);
 bool translation=true;
 for(PxU32 i=0;i<9;++i) translation &= PxAbs(original[i].b-translated[i].b)<1e-12;
 all &= report("common_translation_no_world_anchor",translation);
 const PxQuat rotation(0.63f,PxVec3(1,2,3).getNormalized());
 for(PxU32 i=0;i<12;++i) fixture.positions[i]=rotation.rotate(fixture.positions[i]);
 fixture.geometry.normal=rotation.rotate(fixture.geometry.normal);
 AvbdOgcProjectionPlane rotated[9];fixture.compile(PX_MAX_U32,rotated);
 bool covariance=true;
 for(PxU32 i=0;i<9;++i) covariance &= PxAbs(original[i].b-rotated[i].b)<2e-6;
 all &= report("frame_rotation_covariance",covariance);

 Fixture feature;
 feature.positions[0].z=feature.positions[1].z=-0.1f;
 AvbdOgcProjectionPlane full[9];feature.compile(PX_MAX_U32,full);
 bool rejected=false;for(PxU32 i=0;i<9;++i) rejected |= full[i].b>full[i].roundoff;
 all &= report("complete_triangle_rejects_centroid_false_green",rejected);
 feature.geometry.lowerGap=-0.1f;
 feature.compile(PX_MAX_U32,full);
 bool overlapAccepted=true;for(PxU32 i=0;i<9;++i) overlapAccepted &= full[i].b<=full[i].roundoff;
 feature.positions[0].z=feature.positions[1].z=-0.2f;
 feature.compile(PX_MAX_U32,full);
 rejected=false;for(PxU32 i=0;i<9;++i) rejected |= full[i].b>full[i].roundoff;
 all &= report("immutable_overlap_lower_bound",overlapAccepted && rejected);

 Fixture coupled;
 AvbdOgcProjectionPlane planes[9];coupled.compile(0,planes);
 const double firstLimit=planes[0].b/planes[0].n.z;
 const PxVec3d recovery(0,0,-1.6);
 bool oldRejected=false;
 for(PxU32 i=0;i<9;++i) oldRejected |= planes[i].n.dot(recovery)<planes[i].b-planes[i].roundoff;
 // Move every opposite support outward. Its live displacement must become
 // available to the query block, including both signed embedding weights.
 for(PxU32 i=6;i<12;++i) coupled.positions[i].z-=2;
 coupled.compile(0,planes);
 const double nextLimit=planes[0].b/planes[0].n.z;
 bool newAccepted=true;
 for(PxU32 i=0;i<9;++i) newAccepted &= planes[i].n.dot(recovery)>=planes[i].b-planes[i].roundoff;
 all &= report("live_partner_releases_relative_domain",oldRejected && newAccepted && nextLimit<firstLimit-1.5);

 // Cached raw planes must retain the tighter final acceptance boundary;
 // projection-anchor normalization/roundoff is deliberately not reused here.
 bool cachedAdmission=true;
 for(PxU32 block=0;block<12;++block)
 {
  AvbdOgcProjectionPlane cached[9];coupled.compile(block,cached);
  for(PxU32 sample=0;sample<129;++sample)
  {
   const double z=(double(sample)-64.0)*0.0625;
   const PxVec3d trial(0.125,-0.25,z);
   AvbdOgcProjectionPlane fresh[9];coupled.compile(block,fresh);
   bool expected=true;
   for(PxU32 i=0;i<9;++i)
    if(fresh[i].n.dot(trial)<fresh[i].b-fresh[i].roundoff) expected=false;
   cachedAdmission &= avbdTriangleDomainAdmits(cached,9,trial)==expected;
  }
  for(PxU32 i=0;i<9;++i) if(cached[i].n.z!=0.0)
  {
   const double boundary=(cached[i].b-cached[i].roundoff)/cached[i].n.z;
   for(PxU32 side=0;side<3;++side)
   {
    const double z=side==0 ? std::nextafter(boundary,-DBL_MAX) :
     side==1 ? boundary : std::nextafter(boundary,DBL_MAX);
    const PxVec3d trial(0,0,z);
    AvbdOgcProjectionPlane fresh[9];coupled.compile(block,fresh);
    bool expected=true;
    for(PxU32 j=0;j<9;++j)
     if(fresh[j].n.dot(trial)<fresh[j].b-fresh[j].roundoff) expected=false;
    cachedAdmission &= avbdTriangleDomainAdmits(cached,9,trial)==expected;
   }
  }
 }
 all &= report("cached_final_admission_matches_fresh_at_boundaries",cachedAdmission);

 // Committed-point reuse must match fresh evaluation after every write,
 // including signed/shared supports and a whole-pose replacement at a barrier.
 Fixture reuse;
 reuse.geometry.normal=PxVec3(0.3f,-0.7f,1.2f).getNormalized();
 AvbdAffinePointEvaluation committed[6];
 double localWeights[12][6]={};
 for(PxU32 point=0;point<6;++point)
 {
  reuse.geometry.pointIndices[point]=point;
  committed[point].supportStart=2*point;committed[point].supportCount=2;
  // Adjacent affine points share writers; the negative support is retained.
  reuse.supports[2*point+1].block=(2*point+2)%12;
  for(PxU32 k=2*point;k<2*point+2;++k)
   localWeights[reuse.supports[k].block][point]+=double(reuse.supports[k].weight);
 }
 auto refreshPoint=[&](PxU32 point)
 {
  avbdEvaluateAffinePoint(reuse.supports+committed[point].supportStart,
   committed[point].supportCount,
   [&](const Support& support){return reuse.positions[support.block];},committed[point]);
 };
 for(PxU32 point=0;point<6;++point) refreshPoint(point);
 bool committedExact=true;
 PxU32 unexpectedReads=0,seed=7;
 for(PxU32 commit=0;commit<384;++commit)
 {
  seed=1664525u*seed+1013904223u;
  const PxU32 writer=seed%12;
  reuse.positions[writer][(seed>>8)%3]=PxReal(PxI32(seed>>4)%100000)*0.003f;
  const bool barrier=commit%31==0;
  if(barrier) for(PxU32 i=0;i<12;++i) reuse.positions[i]+=PxVec3(-0.125f,0.25f,0.0625f);
  for(PxU32 point=0;point<6;++point)
   if(barrier || reuse.supports[2*point].block==writer || reuse.supports[2*point+1].block==writer)
    refreshPoint(point);
  for(PxU32 block=0;block<12;++block)
  {
   AvbdOgcProjectionPlane fresh[9],cached[9];reuse.compile(block,fresh);
   PxVec3d points[6],scales[6];double weights[6];
   avbdTriangleDomainPoints(reuse.geometry,reuse.supports,
    [&](const Support& support){++unexpectedReads;return reuse.positions[support.block];},
    block,points,scales,weights,committed,localWeights[block]);
   AvbdTriangleDomainArgs args={&reuse.geometry,points,scales,weights,cached};
   reuse.kernels.launch(AVBD_TRIANGLE_DOMAIN,&args);
   for(PxU32 i=0;i<9;++i)
    committedExact &= avbdTriangleDomainCoefficient(avbdOgcDouble(reuse.geometry.normal),
     localWeights[block],i/3,i%3+3)==fresh[i].n && cached[i].n==fresh[i].n && cached[i].b==fresh[i].b &&
     cached[i].roundoff==fresh[i].roundoff;
  }
 }
 all &= report("committed_affine_points_exact_after_local_and_barrier_writes",committedExact);
 all &= report("committed_point_reads_do_not_resample_supports",unexpectedReads==0);

 // The represented boundary and the input-interval boundary need not come
 // from the same feature. Compare reduction with all unreduced half-spaces.
 AvbdOgcProjectionPlanes unreduced, reduced;
 AvbdOgcParallelPlaneBounds extrema;
 for(PxU32 i=0;i<64;++i)
 {
  AvbdOgcProjectionPlane plane;
  plane.n=PxVec3d(1.0,0.0,0.0);
  plane.b=0.25+double(i)*0.01;
  plane.roundoff=(i%2) ? 0.2 : 0.0;
  unreduced.pushBack(plane);extrema.include(plane);
 }
 extrema.append(reduced);
 bool reducedExact=reduced.size()==2;
 for(PxU32 i=0;i<10000;++i)
 {
  const PxVec3d point(double(i)*0.0001,0.0,0.0);
  reducedExact &= avbdOgcRegionContains(unreduced,point,double(PX_MAX_F32))==
   avbdOgcRegionContains(reduced,point,double(PX_MAX_F32));
 }
 PxVec3d unreducedProjection,reducedProjection;
 const PxMat33d identity(PxIdentity);
 reducedExact &= avbdOgcProjectHalfSpacesFast(unreduced,PxVec3d(0.0),unreducedProjection,identity) &&
  avbdOgcProjectHalfSpacesFast(reduced,PxVec3d(0.0),reducedProjection,identity) &&
  unreducedProjection==reducedProjection;
 all &= report("equal_coefficient_reduction_preserves_both_domains",reducedExact);

 const PxVec3 a[5]={PxVec3(0,0,1),PxVec3(1,0,1),PxVec3(0,1,1),PxVec3(1,1,2),PxVec3(2,0,2)};
 PxVec3 b[4]={PxVec3(0,0,0),PxVec3(1,0,0),PxVec3(0,1,0),PxVec3(1,1,-1)};
 const AvbdTrianglePairPlane shared=avbdPointSetsPlane(a,5,b,4,PxVec3(0,0,1));
 bool covered[20]={},grouped=shared.gap==1.0f;
 for(PxU32 ia=0;ia<5;ia+=3) for(PxU32 ib=0;ib<4;ib+=3)
 {
  Fixture group;group.geometry.normal=shared.normal;
  for(PxU32 k=0;k<3;++k)
  {
   group.positions[2*k]=group.positions[2*k+1]=a[PxMin(ia+k,4u)];
   group.positions[6+2*k]=group.positions[7+2*k]=b[PxMin(ib+k,3u)];
  }
  group.compile(PX_MAX_U32,planes);
  for(PxU32 x=0;x<3;++x) for(PxU32 y=0;y<3;++y)
  { covered[PxMin(ia+x,4u)*4+PxMin(ib+y,3u)]=true;grouped &= planes[x*3+y].b<=planes[x*3+y].roundoff; }
 }
 for(PxU32 i=0;i<20;++i) grouped &= covered[i];
 all &= report("shared_plane_triplets_cover_full_cartesian_product",grouped);
 b[3].z=3.0f;
 all &= report("full_support_plane_detects_new_extremum",avbdPointSetsPlane(a,5,b,4,PxVec3(0,0,1)).gap<0.0f);

 PxVec3 outer[8],inner[8];
 for(PxU32 i=0;i<8;++i)
 {
  outer[i]=PxVec3((i&1)?1.0f:-1.0f,(i&2)?1.0f:-1.0f,
   (i&4)?1.0f:-1.0f);
  inner[i]=PxVec3((i&1)?0.2f:-0.2f,0.7f+((i&2)?0.2f:-0.2f),
   (i&4)?0.2f:-0.2f);
 }
 const AvbdTrianglePairPlane containmentPlane=avbdPointSetsPlane(outer,8,inner,8,
  PxVec3(0.0f,-1.0f,0.0f));
 const PxReal lowerGap=containmentPlane.gap;
 for(PxU32 i=0;i<8;++i) inner[i].y+=1.0f;
 const PxReal openedGap=avbdPointSetsPlane(outer,8,inner,8,containmentPlane.normal).gap;
 for(PxU32 i=0;i<8;++i) inner[i].y-=1.1f;
 const PxReal deepenedGap=avbdPointSetsPlane(outer,8,inner,8,containmentPlane.normal).gap;
 all &= report("containment_signed_interval_opens_through_zero",
  PxAbs(lowerGap+0.5f)<1.0e-6f && openedGap>0.0f &&
  deepenedGap<lowerGap);
 return all ? 0 : 1;
}
