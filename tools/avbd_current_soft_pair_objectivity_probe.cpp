// Current soft-pair frame row through production C/J/potential and candidate
// evaluation. Constructed row; not a complete scene qualification.
#include "avbd/DyAvbdInternal.h"
#include "avbd_probe_foundation.h"
#include <cstdio>
using namespace physx;
using namespace physx::Dy;
static bool publishedInputInterval()
{
 AvbdRowBuilder row={};row.header().flags=eAVBD_ROW_CONTACT|eAVBD_ROW_HETEROGENEOUS;
 row.header().supportCount=1;row.scalar().contactOffset=-1;row.scalar().C0=0;row.scalar().n=PxVec3(0,1,0);
 PxArray<AvbdRowSupport> supports;AvbdRowSupport support={};support.weight=1;supports.pushBack(support);
 const auto oldPoint=[](const AvbdRowSupport&) { return PxVec3(0,1000,0); };
 const auto badPoint=[](const AvbdRowSupport&) { return PxVec3(0,.9999f,0); };
 const auto goodPoint=[](const AvbdRowSupport&) { return PxVec3(0,1,0); };
 // The old/new scale accepts the bad point, but its self interval rejects it.
 // Publication must reject it immediately, for Particle3, Rigid6 and joint moves.
 if(avbdHeterogeneousCandidateAdmissiblePoints(row,supports,row.scalar().n,row.scalar().n,oldPoint,badPoint)) return false;
 if(!avbdHeterogeneousCandidateAdmissiblePoints(row,supports,row.scalar().n,row.scalar().n,oldPoint,goodPoint)) return false;
 return avbdHeterogeneousCandidateAdmissiblePoints(row,supports,row.scalar().n,row.scalar().n,goodPoint,goodPoint);
}
int main()
{
 AvbdProbeFoundation foundation;
 if(!publishedInputInterval())
 { std::printf("[PUBLISHED_INPUT_INTERVAL] status=FAIL\n");return 1; }
 const KernelTable kernels=KernelTable::makeAvbdCpuScalar();
 AvbdSoftParticle particles[5]={};
 const PxVec3 start[5]={PxVec3(0),PxVec3(1,0,0),PxVec3(0,1,0),PxVec3(.25f,.25f,.01f),PxVec3(.25f,.25f,.01f)};
 PxArray<AvbdBlock> blocks;PxArray<AvbdRowSupport> supports;
 PxArray<AvbdSolverBody> bodies;PxArray<AvbdContactPlaneGeometry> planes;
 PxArray<AvbdContactFrameGeometry> frames;
 for(PxU32 i=0;i<5;++i)
 {
  particles[i].p=start[i];AvbdBlock block={};block.type=eAVBD_BLOCK_PARTICLE3;
  block.particle=&particles[i];blocks.pushBack(block);
 }
 const PxU32 indices[8]={3,4,0,1,2,0,1,2};
 const PxReal weights[8]={1.25f,-.25f,.5f,.25f,.25f,1,1,1};
 for(PxU32 i=0;i<8;++i)
 { AvbdRowSupport support={};support.block=indices[i];support.weight=weights[i];supports.pushBack(support); }
 AvbdContactFrameGeometry frame={};frame.normalLocal=PxVec3(0,0,1);
 const PxU16 ends[5]={2,5,6,7,8};for(PxU32 i=0;i<5;++i) frame.supportEnds[i]=ends[i];frames.pushBack(frame);
 const AvbdContactGeometryView geometry(planes,frames,kernels);
 AvbdRowBuilder row={};row.header().bodyA=row.header().bodyB=PX_MAX_U32;row.scalar().n=PxVec3(0,0,1);
 row.header().kernelId=eAVBD_ROW_KERNEL_CONTACT_FRAME;row.header().geometryIndex=1;row.header().supportCount=8;
 row.scalar().contactOffset=-.04f;row.scalar().contactRadius=.04f;row.scalar().contactStiffness=row.scalar().rho=1e5f;row.scalar().minLambda=-PX_MAX_F32;
 row.header().flags=eAVBD_ROW_UNILATERAL|eAVBD_ROW_CONTACT|eAVBD_ROW_HETEROGENEOUS;
 const auto value=[&]() { return avbdHeterogeneousContactPotential(row,
  avbdHeterogeneousRowC(row,supports,blocks,bodies,geometry)); };
 const PxReal before=value().energy;
 const PxQuat rotation(.63f,PxVec3(0,1,0));
 for(PxU32 i=0;i<5;++i) particles[i].p=rotation.rotate(particles[i].p);
 const AvbdContactPotential turned=value();
 const auto eval=avbdContactFrameRow(row,supports,blocks,bodies,geometry);
 PxVec3 force(0),moment(0);
 avbdForEachContactFrameBlock(row,supports,[&](PxU32 block)
 {
  const PxVec3 f=avbdContactFrameGradient(row,supports,geometry,eval,block)*(-turned.gradient);
  force+=f;moment+=particles[block].p.cross(f);
 });
 bool pass=eval.valid && PxAbs(turned.energy-before)<1e-3f && force.magnitude()<1e-3f && moment.magnitude()<1e-3f;
 std::printf("[CURRENT_SOFT_PAIR_OBJECTIVITY] status=%s energyBefore=%.9g energyAfter=%.9g netForce=%.9g netMoment=%.9g capturedScene=0\n",
  pass ? "PASS" : "FAIL",double(before),double(turned.energy),double(force.magnitude()),double(moment.magnitude()));
 particles[3].p.x+=.003f;
 const auto deformed=avbdContactFrameRow(row,supports,blocks,bodies,geometry);
 double maxError=0;bool sameCandidate=true;
 for(PxU32 block=0;block<5;++block)
 {
  const PxVec3 gradient=avbdContactFrameGradient(row,supports,geometry,deformed,block);
  for(PxU32 axis=0;axis<3;++axis)
  {
   const PxVec3 original=particles[block].p;PxVec3 high=original,low=original;
   high[axis]+=1e-4f;low[axis]-=1e-4f;
   const auto plus=avbdContactFrameRow(row,supports,blocks,bodies,geometry,block,&high);
   const auto minus=avbdContactFrameRow(row,supports,blocks,bodies,geometry,block,&low);
   maxError=PxMax(maxError,PxAbs((plus.gap-minus.gap)/(double(high[axis])-low[axis])-gradient[axis]));
   particles[block].p=high;
   sameCandidate &= PxReal(plus.gap)==avbdHeterogeneousRowC(row,supports,blocks,bodies,geometry);
   particles[block].p=original;
  }
 }
 bool derivatives=deformed.valid && maxError<1e-6 && sameCandidate;
 pass &= derivatives;
 std::printf("[CURRENT_SOFT_PAIR_DERIVATIVES] status=%s maxError=%.9g candidateMatches=%u\n",derivatives ? "PASS" : "FAIL",maxError,PxU32(sameCandidate));
 bool material=true;
 for(PxU32 axis=0;axis<2;++axis)
 {
  const PxVec3 localAxis=axis ? PxVec3(0,1,0) : PxVec3(1,0,0);
  const auto tangent=avbdContactFrameRow(row,supports,blocks,bodies,geometry,PX_MAX_U32,NULL,&localAxis);
  PxVec3 jacobians[5],velocities[5];double velocity=0,response=0,initialEnergy=0,rigidWork=0;
  for(PxU32 block=0;block<5;++block)
  {
   particles[block].invMass=1.0f/(1.0f+.2f*block);
   const PxVec3 rigid=PxVec3(.2f,.3f,-.1f)+PxVec3(.7f,-.5f,.9f).cross(particles[block].p);
   jacobians[block]=avbdContactFrameGradient(row,supports,geometry,tangent,block);
   rigidWork+=double(jacobians[block].dot(rigid));
   velocities[block]=rigid+(block==3 ? PxVec3(1,.5f,.25f) : PxVec3(0));
   velocity+=double(jacobians[block].dot(velocities[block]));
   response+=double(particles[block].invMass)*jacobians[block].magnitudeSquared();
   initialEnergy+=.5*double(velocities[block].magnitudeSquared())/particles[block].invMass;
  }
  const double impulse=-velocity/response;
  double finalEnergy=0,impulseScale=0,momentScale=0;
  PxVec3d net(0),torque(0);
  for(PxU32 block=0;block<5;++block)
  {
   const PxVec3 change=jacobians[block]*PxReal(particles[block].invMass*impulse);
   const PxVec3d momentum=avbdOgcDouble(change)/double(particles[block].invMass);
   net+=momentum;torque+=avbdOgcDouble(particles[block].p).cross(momentum);
   impulseScale+=momentum.magnitude();momentScale+=particles[block].p.magnitude()*momentum.magnitude();
   finalEnergy+=.5*double((velocities[block]+change).magnitudeSquared())/particles[block].invMass;
  }
  const double expectedChange=-.5*velocity*velocity/response;
  material &= tangent.valid && PxAbs(rigidWork)<1e-5 && net.magnitude()<8*FLT_EPSILON*impulseScale &&
   torque.magnitude()<8*FLT_EPSILON*momentScale &&
   PxAbs(finalEnergy-initialEnergy-expectedChange)<32*FLT_EPSILON*(initialEnergy+finalEnergy);
 }
 pass &= material;
 std::printf("[CURRENT_SOFT_PAIR_MATERIAL_WORK] status=%s\n",material ? "PASS" : "FAIL");
 return pass ? 0 : 1;
}
