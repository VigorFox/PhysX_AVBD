// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// A reduced domain counterexample, not a full solver/scene qualification.
#include "avbd/DyAvbdContactGeometry.h"
#include "avbd/DyAvbdCoupledDomain.h"
#include "avbd/DyAvbdOgc.h"
#include "avbd/DyKernelTable.h"
#include "avbd_probe_foundation.h"
#include <cstdio>
#include <cstring>

using namespace physx;
using namespace physx::Dy;

#include "fixtures/avbd_coupled_kkt_stall.h"

static bool capturedKktComplementarity()
{
 PxArray<AvbdCoupledDomainBlock> blocks(PxU32(sizeof(capturedFree)/sizeof(capturedFree[0])));
 for(PxU32 i=0;i<blocks.size();++i) blocks[i].step=capturedFree[i];
 PxArray<AvbdCoupledDomainTerm> terms;
 PxArray<AvbdCoupledDomainConstraint> constraints;
 for(const auto& t:capturedTerms) terms.pushBack(t);
 for(const auto& c:capturedConstraints) constraints.pushBack(c);
 double violation=0.0,descent=0.0,complementarity=0.0;
 const bool solved=avbdSolveCoupledDomainKkt(blocks,constraints,terms,violation,true);
 for(PxU32 i=0;i<blocks.size();++i) descent+=capturedForce[i].dot(blocks[i].step);
 for(const auto& c:constraints)
 {
  double slack=c.tolerance-c.rhs;
  for(PxU32 j=c.termStart;j<c.termStart+c.termCount;++j)
   slack+=terms[j].gradient.dot(blocks[terms[j].block].step);
  complementarity=PxMax(complementarity,PxAbs(c.multiplier*slack));
 }
 // Independent optimality checks, not the change between consecutive iterates.
 // Old early termination reported success with descent=-0.009422 and gap=0.314.
 const bool pass=solved && violation<1.0e-12 && descent>0.04 && complementarity<1.0e-8;
 std::printf("[DOMAIN_STATIONARITY] case=captured_kkt_complementarity status=%s descent=%.17g complementarity=%.17g violation=%.17g\n",
  pass ? "PASS" : "FAIL",descent,complementarity,violation);
 return pass;
}

struct Support { PxU32 block; PxReal weight; };

static bool capturedContactInterval()
{
 // Uninstrumented Debug showcase, frame 224. The old pose is legal but has
 // only 95 nm of its original input interval left, not another full 4.9 um.
 const PxVec3 normal(-.4973047376f,.8675759435f,-4.604332389e-6f);
 const PxReal weights[2]={.5000001192f,.4999998808f};
 const PxVec3 points[2]={PxVec3(6.248954773f,.7195199728f,-.1464745402f),
  PxVec3(6.621760368f,.9487384558f,.3741201162f)};
 AvbdContactAdmissionState state(2.47665906f,0.0,.01254689693f);
 for(PxU32 i=0;i<2;++i) state.add(weights[i],normal,points[i]);
 const double remaining=state.gap+state.tolerance(state);
 if(!(state.gap<0.0 && remaining>0.0 && remaining<1.0e-7)) return false;
 PxArray<AvbdCoupledDomainBlock> blocks(2);
 PxArray<AvbdCoupledDomainTerm> terms;
 PxArray<AvbdCoupledDomainConstraint> constraints;
 for(PxU32 i=0;i<2;++i)
 {
  const PxVec3d gradient=avbdOgcDouble(normal*weights[i]);
  blocks[i].step=-gradient*1.0e-3;
  AvbdCoupledDomainTerm term={i,gradient,gradient};terms.pushBack(term);
 }
 AvbdCoupledDomainConstraint constraint={};
 constraint.termCount=2;constraint.rhs=-state.gap;
 constraint.tolerance=state.tolerance(state);constraints.pushBack(constraint);
 double violation=0.0;
 if(!avbdSolveCoupledDomainKkt(blocks,constraints,terms,violation,true)) return false;
 // Independent geometric oracle: evaluate the absolute authored plane,
 // including the continuous correction, rather than a recentered plane.
 double gap=double(2.47665906f);
 for(PxU32 i=0;i<2;++i) for(PxU32 axis=0;axis<3;++axis)
  gap+=double(weights[i])*double(normal[axis])*(double(points[i][axis])+blocks[i].step[axis]);
 const bool pass=gap>=-constraint.tolerance-1.0e-12 &&
  PxAbs(gap+constraint.tolerance)<1.0e-10;
 std::printf("[DOMAIN_STATIONARITY] case=captured_contact_interval status=%s remaining=%.17g correctedGap=%.17g tolerance=%.17g\n",
  pass ? "PASS" : "FAIL",remaining,gap,constraint.tolerance);
 return pass;
}

static bool capturedRoundedContact()
{
 // Uninstrumented Debug: the continuous boundary has only 24 nm left;
 // separately rounding the two endpoint positions can consume 104 nm.
 const PxVec3 p[2]={PxVec3(7.463197708f,1.203762293f,.5711932778f),
  PxVec3(6.86998558f,1.109401822f,.3102070093f)};
 const PxVec3 n(-.1887254119f,.9820299149f,-1.209158199e-5f);
 const PxReal w[2]={.7499999404f,.2500000596f};
 const PxVec3d direction[2]={PxVec3d(-8.89908953902229e-6,-2.831393508481042e-5,-1.0132447755535384e-5),
  PxVec3d(-2.1754560154693848e-5,7.553241734762844e-5,2.4805725655759796e-5)};
 AvbdContactAdmissionState old(.221545741f,0.0,.004519421607f);
 for(PxU32 i=0;i<2;++i) old.add(w[i],n,p[i]);
 PxArray<AvbdCoupledDomainBlock> blocks(2);
 PxArray<AvbdCoupledDomainTerm> terms;
 PxArray<AvbdCoupledDomainConstraint> constraints;
 AvbdCoupledDomainConstraint c={};c.termCount=2;c.rhs=-old.gap;c.tolerance=old.tolerance(old);
 double margin=0.0;
 AvbdContactAdmissionState rounded(.221545741f,0.0,.004519421607f);
 for(PxU32 i=0;i<2;++i)
 {
  const PxVec3d gradient=avbdOgcDouble(n)*double(w[i]);
  blocks[i].step=direction[i]*.125;
  const PxVec3 delta(PxReal(blocks[i].step.x),PxReal(blocks[i].step.y),PxReal(blocks[i].step.z));
  rounded.add(w[i],n,p[i]+delta);
  margin+=avbdCoupledPublicationError(p[i],blocks[i].step,gradient);
  terms.pushBack({i,gradient,gradient});
 }
 const bool originalRejected=rounded.gap < -rounded.tolerance(rounded);
 c.rhs+=margin;constraints.pushBack(c);
 double violation=0.0;
 const bool solved=avbdSolveCoupledDomainKkt(blocks,constraints,terms,violation,true);
 AvbdContactAdmissionState after(.221545741f,0.0,.004519421607f);
 for(PxU32 i=0;i<2;++i)
 {
  const PxVec3 d(PxReal(blocks[i].step.x),PxReal(blocks[i].step.y),PxReal(blocks[i].step.z));
  after.add(w[i],n,p[i]+d);
 }
 const bool pass=originalRejected && solved && after.gap>=-old.tolerance(after) &&
  after.gap>=-after.tolerance(after);
 std::printf("[DOMAIN_STATIONARITY] case=captured_rounded_contact status=%s beforeMargin=%.17g afterMargin=%.17g proposalMargin=%.17g\n",
  pass ? "PASS" : "FAIL",rounded.gap+rounded.tolerance(rounded),after.gap+after.tolerance(after),margin);
 return pass;
}

static bool sparseKktEndpoints()
{
 PxArray<AvbdCoupledDomainBlock> blocks(10006);
 for(PxU32 i=0;i<blocks.size();++i) blocks[i].step=PxVec3d(0.25);
 blocks[17].step=blocks[9001].step=PxVec3d(0.0);
 PxArray<AvbdCoupledDomainTerm> terms;
 PxArray<AvbdCoupledDomainConstraint> constraints;
 for(PxU32 axis=0;axis<2;++axis)
 {
  PxVec3d normal(0.0);normal[axis]=1.0;
  AvbdCoupledDomainConstraint constraint={};
  constraint.termStart=terms.size();constraint.termCount=2;constraint.rhs=axis ? 1.0 : 2.0;
  AvbdCoupledDomainTerm a={17,normal,normal},b={9001,-normal,-normal};
  terms.pushBack(a);terms.pushBack(b);constraints.pushBack(constraint);
 }
 double violation=0.0;
 if(!avbdSolveCoupledDomainKkt(blocks,constraints,terms,violation)) return false;
 if(blocks[17].step!=PxVec3d(1.0,0.5,0.0) || blocks[9001].step!=PxVec3d(-1.0,-0.5,0.0)) return false;
 for(PxU32 i=0;i<blocks.size();++i)
  if(i!=17 && i!=9001 && blocks[i].step!=PxVec3d(0.25)) return false;
 return violation==0.0;
}

static bool repeatedInputPlanes()
{
 // Independent projection with duplicated left sides and different input
 // intervals. x0-x1>=2, y0-y1>=4 has the minimizer (1,2),(-1,-2).
 PxArray<AvbdCoupledDomainBlock> full(2),compact(2);
 full[0].step=full[1].step=compact[0].step=compact[1].step=PxVec3d(0.0);
 PxArray<AvbdCoupledDomainTerm> terms;
 PxArray<AvbdCoupledDomainConstraint> constraints,reduced;
 for(PxU32 axis=0;axis<2;++axis)
 {
  PxVec3d normal(0.0);normal[axis]=1;
  const PxU32 begin=terms.size();
  AvbdCoupledDomainTerm a={0,normal,normal},b={1,-normal,-normal};
  terms.pushBack(a);terms.pushBack(b);
  AvbdCoupledDomainConstraint strongest={};bool first=true;
  for(PxU32 i=0;i<64;++i)
  {
   AvbdCoupledDomainConstraint c={};c.termStart=begin;c.termCount=2;
   c.tolerance=double(i%7)*0.125;
   c.rhs=double((axis+1)*2)-double(63-i)*0.0625+c.tolerance;
   constraints.pushBack(c);
   if(first || c.rhs-c.tolerance>strongest.rhs-strongest.tolerance)
   { strongest=c;first=false; }
  }
  reduced.pushBack(strongest);
 }
 double violation=0;
 if(!avbdSolveCoupledDomainKkt(full,constraints,terms,violation,true) ||
    !avbdSolveCoupledDomainKkt(compact,reduced,terms,violation,true)) return false;
 return (full[0].step-PxVec3d(1,2,0)).magnitude()<1e-6 &&
  (compact[0].step-PxVec3d(1,2,0)).magnitude()<1e-6 &&
  (full[1].step-PxVec3d(-1,-2,0)).magnitude()<1e-6 &&
  (compact[1].step-PxVec3d(-1,-2,0)).magnitude()<1e-6;
}

static bool publicationScopeCounterexample()
{
 // A continuous QP can be feasible while its cropped publication is not.
 // x<=y and z<=y start at zero. Only the first row is initially violated
 // by the free Newton proposal (0,-.2,-1); it activates x and y.
 PxArray<AvbdCoupledDomainBlock> blocks(3);
 blocks[0].step=PxVec3d(0);blocks[1].step=PxVec3d(-.2,0,0);
 blocks[2].step=PxVec3d(-1,0,0);
 PxArray<AvbdCoupledDomainTerm> terms;
 terms.pushBack({0,PxVec3d(-1,0,0),PxVec3d(-1,0,0)});
 terms.pushBack({1,PxVec3d(1,0,0),PxVec3d(1,0,0)});
 terms.pushBack({2,PxVec3d(-1,0,0),PxVec3d(-1,0,0)});
 terms.pushBack({1,PxVec3d(1,0,0),PxVec3d(1,0,0)});
 PxArray<AvbdCoupledDomainConstraint> active;
 AvbdCoupledDomainConstraint c={};c.termCount=2;active.pushBack(c);
 double violation=0;
 const bool solved=avbdSolveCoupledDomainKkt(blocks,active,terms,violation);
 const double firstGap=blocks[1].step.x-blocks[0].step.x;
 const double fullGap=blocks[1].step.x-blocks[2].step.x;
 const double croppedGap=blocks[1].step.x;
 PxArray<AvbdCoupledDomainBlock> free(3);
 free[0].step=PxVec3d(0);free[1].step=PxVec3d(-.2,0,0);free[2].step=PxVec3d(-1,0,0);
 PxArray<PxU8> writable;
 bool scoped=avbdCoupledDomainPublicationScope(free,active,terms,writable,blocks) &&
  avbdSolveCoupledDomainKkt(blocks,active,terms,violation);
 // Query the previously inactive complete domain after restricting variables.
 if(blocks[1].step.x-blocks[2].step.x<0)
 { c.termStart=2;active.pushBack(c); }
 scoped &= avbdCoupledDomainPublicationScope(free,active,terms,writable,blocks) &&
  avbdSolveCoupledDomainKkt(blocks,active,terms,violation);
 const double publishedGap=blocks[1].step.x-(writable[2] ? blocks[2].step.x : 0.0);
 const bool pass=solved && firstGap>=-1e-12 && fullGap>=-1e-12 && croppedGap<0 &&
  scoped && writable[2] && publishedGap>=-1e-12;
 std::printf("[DOMAIN_PUBLICATION] status=%s kktSolved=%u fullGap=%.9g croppedGap=%.9g scopedGap=%.9g physicalQualification=0\n",
  pass?"PASS":"FAIL",PxU32(solved),fullGap,croppedGap,publishedGap);
 return pass;
}

int main(int argc,char** argv)
{
 AvbdProbeFoundation foundation;
 // Exercise the same publication-scope compiler as the production closure.
 if(argc==2 && std::strcmp(argv[1],"--publication-scope")==0)
  return publicationScopeCounterexample()?0:1;
 if(argc==2 && std::strcmp(argv[1],"--contact-interval")==0)
  return capturedContactInterval()?0:1;
 if(argc!=1) { std::fprintf(stderr,"Usage: probe [--publication-scope|--contact-interval]\n");return 2; }
 if(!capturedContactInterval() || !capturedRoundedContact() || !publicationScopeCounterexample()) return 1;
 if(!capturedKktComplementarity()) return 1;
 if(!repeatedInputPlanes())
 { std::printf("[DOMAIN_STATIONARITY] repeated_input_planes=FAIL\n");return 1; }
 if(!sparseKktEndpoints())
 { std::printf("[DOMAIN_STATIONARITY] sparse_endpoints=FAIL\n");return 1; }
 const KernelTable kernels=KernelTable::makeAvbdCpuScalar();
 AvbdTriangleDomainGeometry geometry;
 geometry.normal=PxVec3(0,0,1);geometry.lowerGap=0;
 Support supports[6];
 PxVec3 positions[6];
 const PxVec3 triangle[3]={PxVec3(0,0,0),PxVec3(1,0,0),PxVec3(0,1,0)};
 for(PxU32 i=0;i<6;++i)
 {
  geometry.supportEnds[i]=PxU16(i+1);
  supports[i]={i,1.0f};positions[i]=triangle[i%3];
 }
 const auto compile=[&](PxU32 block,AvbdOgcProjectionPlane* planes)
 {
  PxVec3d points[6],scales[6];double weights[6];
  avbdTriangleDomainPoints(geometry,supports,
   [&](const Support& support){return positions[support.block];},
   block,points,scales,weights);
  AvbdTriangleDomainArgs args={&geometry,points,scales,weights,planes};
  kernels.launch(AVBD_TRIANGLE_DOMAIN,&args);
 };
 const auto energy=[&]()
 {
  double result=0;
  for(PxU32 i=0;i<6;++i)
  {
   const double residual=double(positions[i].z)-(i<3 ? -2.0 : 1.0);
   result+=0.5*residual*residual;
  }
  return result;
 };
 const double before=energy();
 double maxLocalStep=0;
 bool feasible=true;
 // Exact block Newton proposals for this quadratic. Production domain
 // sampling, scalar dispatch and projection are used without relaxed bounds.
 for(PxU32 block=0;block<6;++block)
 {
  AvbdOgcProjectionPlane planes[9];compile(block,planes);
  AvbdOgcProjectionPlanes domain;
  for(PxU32 i=0;i<9;++i) domain.pushBack(planes[i]);
  AvbdOgcPointRegion region;avbdOgcResetPointRegion(region,positions[block]);
  PxVec3 candidate=positions[block];candidate.z=block<3 ? -2.0f : 1.0f;
  const AvbdOgcClip clip=avbdOgcTruncatePoint(region,candidate,&domain,NULL,&positions[block]);
  feasible &= clip.feasible;
  maxLocalStep=PxMax(maxLocalStep,double((candidate-positions[block]).magnitude()));
 }
 // The production coupled-domain KKT helper starts from the same six local
 // Newton proposals.  Each scalar domain condition obtains its gradients by
 // dispatching the production triangle-domain kernel for every Particle3.
 PxArray<AvbdCoupledDomainBlock> blocks;
 PxArray<AvbdCoupledDomainConstraint> constraints;
 PxArray<AvbdCoupledDomainTerm> terms;
 blocks.resize(6);
 for(PxU32 block=0;block<6;++block)
  blocks[block].step=PxVec3d(0,0,block<3 ? -2.0 : 1.0);
 AvbdOgcProjectionPlane full[9];compile(PX_MAX_U32,full);
 for(PxU32 planeIndex=0;planeIndex<9;++planeIndex)
 {
  AvbdCoupledDomainConstraint constraint={};
  constraint.termStart=terms.size();
  constraint.rhs=full[planeIndex].b;
  constraint.tolerance=full[planeIndex].roundoff;
  for(PxU32 block=0;block<6;++block)
  {
   AvbdOgcProjectionPlane local[9];compile(block,local);
   if(local[planeIndex].n.magnitudeSquared()<=1.0e-30) continue;
   AvbdCoupledDomainTerm term;
   term.block=block;term.gradient=local[planeIndex].n;
   term.inverseMetricGradient=term.gradient;
   terms.pushBack(term);
  }
  constraint.termCount=terms.size()-constraint.termStart;
  constraints.pushBack(constraint);
 }
 double kktViolation=0.0;
 const bool kktSolved=avbdSolveCoupledDomainKkt(blocks,constraints,terms,kktViolation,true);
 double directionalDerivative=0;
 for(PxU32 i=0;i<6;++i)
 {
  const double gradient=double(positions[i].z)-(i<3 ? -2.0 : 1.0);
  directionalDerivative+=gradient*blocks[i].step.z;
  positions[i]+=PxVec3(PxReal(blocks[i].step.x),PxReal(blocks[i].step.y),
   PxReal(blocks[i].step.z));
 }
 AvbdOgcProjectionPlane joint[9];compile(PX_MAX_U32,joint);
 bool jointFeasible=true;
 for(PxU32 i=0;i<9;++i) jointFeasible &= joint[i].b<=joint[i].roundoff;
 const double after=energy();
 const bool closed=feasible && maxLocalStep<1e-6 && kktSolved && jointFeasible &&
  after<before && directionalDerivative<0.0;
 std::printf("[DOMAIN_STATIONARITY] status=%s localMaxStep=%.9g kktSolved=%u kktViolation=%.9g jointFeasible=%u energyBefore=%.9g energyAfter=%.9g jointDirectionalDerivative=%.9g physicalQualification=0\n",
  closed ? "PASS" : "FAIL",maxLocalStep,PxU32(kktSolved),kktViolation,
  PxU32(jointFeasible),before,after,directionalDerivative);
 return closed ? 0 : 1;
}
