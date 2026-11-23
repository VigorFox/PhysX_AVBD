#include "../physx/snippets/snippetjointsoftbody/SnippetJointSoftBodyCommon.h"
#include <cstdio>
using namespace JointSoftBodySnippet;
static void prep(const FiniteJointData& data,const PxTransform& a,const PxTransform& b,Px1DConstraint (&rows)[FINITE_JOINT_ROWS])
{ PxVec3p offset,pa,pb;PxConstraintInvMassScale scale;finiteJointPrep(rows,offset,FINITE_JOINT_ROWS,scale,&data,a,b,false,pa,pb); }
static PxTransform move(PxTransform a,PxU32 axis,PxReal h)
{ if(axis<3) a.p[axis]+=h;else { PxVec3 v(0);v[axis-3]=1;a.q=(PxQuat(h,v)*a.q).getNormalized(); }return a; }
static PxU32 collisionFiltering()
{
 PxU32 failures=0;
 const char* names[]={"non_neighbor_self_contact","adjacent_compression","cross_instance","legacy_exclusion"};
 for(PxU32 test=0;test<4;++test)
 {
  World world;Settings settings;settings.workers=2;
  if(!world.initScene(settings,names[test],PxVec3(0),false,0)) return 1;
  PxRigidDynamic* bodies[2];
  for(PxU32 i=0;i<2;++i)
  {
   bodies[i]=PxCreateDynamic(*world.physics,PxTransform(PxVec3(i?.6f:-.6f,0,0)),PxBoxGeometry(.25f,.25f,.25f),*world.material,10);
   if(!bodies[i]) return 1;
   PxShape* shape=NULL;bodies[i]->getShapes(&shape,1);
   const PxFilterData filterData(test==2?i+1:1,test==3?0:(i?(test==0?3:2):1),1,1);
   shape->setSimulationFilterData(filterData);
   bodies[i]->setLinearVelocity(PxVec3(i?-1.f:1.f,0,0));bodies[i]->setLinearDamping(0);
   world.scene->addActor(*bodies[i]);
  }
  PxReal minimumGap=PX_MAX_F32;
  bool fetched=true;
  for(PxU32 frame=0;frame<60;++frame)
  {
   world.scene->simulate(1.f/60);fetched&=world.scene->fetchResults(true);
   minimumGap=PxMin(minimumGap,bodies[1]->getGlobalPose().p.x-bodies[0]->getGlobalPose().p.x-.5f);
  }
  const bool collides=test==0 || test==2;
  const bool pass=fetched && !world.errors.fatal.load() && (collides?
   world.events.contacts>0 && minimumGap>=-1.e-3f:world.events.contacts==0 && minimumGap<-.5f);
  std::printf("[FINITE_JOINT_FILTER] case=%s contacts=%u minimumGap=%.9g status=%s\n",names[test],world.events.contacts,minimumGap,pass?"PASS":"FAIL");
  failures+=!pass;bodies[0]->release();bodies[1]->release();
 }
 return failures;
}
static int inversionDomain()
{
 // Actual native solverPrep coordinates for three unit sections sharing
 // one material corner. Passing through det(F)=0 costs finite energy.
 const PxReal stretches[]={1.f,.2f,0.f,-.2f};
 bool rejectedInversion=true;
 for(PxU32 sample=0;sample<4;++sample)
 {
  const PxReal stretch=stretches[sample];double energy=0;PxReal maxForce=0;
  for(PxU32 axis=0;axis<3;++axis)
  {
   PxVec3 e(0);e[axis]=1;
   const PxQuat frame=axis==0?PxQuat(PxIdentity):axis==1?PxQuat(PxHalfPi,PxVec3(0,0,1)):PxQuat(-PxHalfPi,PxVec3(0,1,0));
   FiniteJointData data;data.c2b[0]=PxTransform(e*.5f,frame);data.c2b[1]=PxTransform(-e*.5f,frame);
   for(PxU32 r=0;r<6;++r) data.drives[r]=PxD6JointDrive(r<3?120000.f:10000.f,0,PX_MAX_F32,false);
   Px1DConstraint rows[FINITE_JOINT_ROWS];prep(data,PxTransform(PxIdentity),PxTransform(e*(axis==0?stretch:1.f)),rows);
   for(PxU32 r=0;r<FINITE_JOINT_ROWS;++r)
   {
    const PxReal c=rows[r].geometricError,k=rows[r].mods.spring.stiffness;
    energy+=.5*double(k)*c*c;maxForce=PxMax(maxForce,PxAbs(k*c));
   }
  }
  if(std::isnan(energy)) return 2;
  if(stretch<=0) rejectedInversion &= std::isinf(energy) && energy>0;
  std::printf("[FINITE_JOINT_DOMAIN] detF=%.9g energy=%.9g maxCoordinateForce=%.9g\n",stretch,energy,maxForce);
 }
 std::printf("[FINITE_JOINT_ENERGY_BARRIER] status=%s\n",rejectedInversion?"PASS":"FAIL");
 return rejectedInversion?0:1;
}
int main(int argc,char** argv)
{
 if(argc==2 && std::strcmp(argv[1],"--inversion-domain")==0) return inversionDomain();
 PxU32 checks=0,failures=0;PxReal maxDerivativeError=0,maxObjectivityError=0,maxTorque=0,maxExchangeError=0;
 for(PxU32 sample=0;sample<160;++sample)
 {
  FiniteJointData data;data.c2b[0]=PxTransform(PxVec3(.13f,.2f,-.3f),PxQuat(.37f,PxVec3(1,2,3).getNormalized()));data.c2b[1]=PxTransform(PxVec3(-.2f,.3f,.1f),PxQuat(-.22f,PxVec3(3,1,2).getNormalized()));
  for(PxU32 i=0;i<6;++i) data.drives[i]=PxD6JointDrive(1000.f*(i+1),20.f*(i+1),PX_MAX_F32,false);
  const PxReal angle=3.05f*PxReal(sample%40)/39.f;
  const PxTransform a(PxVec3(-.3f,.2f,.1f),PxQuat(.01f*sample,PxVec3(2,3,1).getNormalized()));
  const PxTransform b(PxVec3(.4f,.1f,-.2f),a.q*data.c2b[0].q*PxQuat(angle,PxVec3(1+PxReal(sample%3),2,3).getNormalized())*data.c2b[1].q.getConjugate());
  Px1DConstraint rows[FINITE_JOINT_ROWS];prep(data,a,b,rows);
  FiniteJointData exchanged=data;exchanged.c2b[0]=data.c2b[1];exchanged.c2b[1]=data.c2b[0];
  Px1DConstraint swapped[FINITE_JOINT_ROWS];prep(exchanged,b,a,swapped);
  double energy=0,otherEnergy=0;
  for(PxU32 r=0;r<FINITE_JOINT_ROWS;++r) { energy+=.5*rows[r].mods.spring.stiffness*rows[r].geometricError*rows[r].geometricError;otherEnergy+=.5*swapped[r].mods.spring.stiffness*swapped[r].geometricError*swapped[r].geometricError; }
  const PxReal exchangeError=PxReal(std::abs(energy-otherEnergy)/PxMax(1.,energy));maxExchangeError=PxMax(maxExchangeError,exchangeError);++checks;failures+=exchangeError>2.e-5f;
  const PxTransform common(PxVec3(2,-1,3),PxQuat(.83f,PxVec3(2,1,3).getNormalized()));Px1DConstraint transformed[FINITE_JOINT_ROWS];prep(data,common*a,common*b,transformed);
  for(PxU32 r=0;r<FINITE_JOINT_ROWS;++r)
  {
   const PxReal objective=PxAbs(rows[r].geometricError-transformed[r].geometricError);maxObjectivityError=PxMax(maxObjectivityError,objective);++checks;failures+=objective>2.e-5f;
   const PxVec3 force=rows[r].linear0-rows[r].linear1;
   const PxVec3 torque=a.p.cross(rows[r].linear0)-b.p.cross(rows[r].linear1)+rows[r].angular0-rows[r].angular1;
   maxTorque=PxMax(maxTorque,torque.magnitude());++checks;failures+=force.magnitude()>2.e-5f || torque.magnitude()>2.e-5f;
   for(PxU32 endpoint=0;endpoint<2;++endpoint) for(PxU32 axis=0;axis<6;++axis)
   {
    const PxReal h=5.e-4f;Px1DConstraint plus[FINITE_JOINT_ROWS],minus[FINITE_JOINT_ROWS];
    prep(data,endpoint?a:move(a,axis,h),endpoint?move(b,axis,h):b,plus);
    prep(data,endpoint?a:move(a,axis,-h),endpoint?move(b,axis,-h):b,minus);
    const PxReal derivative=(plus[r].geometricError-minus[r].geometricError)/(2*h);
    const PxVec3 j=axis<3?(endpoint?-rows[r].linear1:rows[r].linear0):(endpoint?-rows[r].angular1:rows[r].angular0);
    const PxReal expected=j[axis%3],error=PxAbs(derivative-expected);maxDerivativeError=PxMax(maxDerivativeError,error);++checks;
    if(error>2.e-3f+3.e-3f*PxAbs(expected)) { if(failures<5) std::printf("BAD sample=%u row=%u endpoint=%u axis=%u expected=%.9g derivative=%.9g\n",sample,r,endpoint,axis,expected,derivative);++failures; }
   }
  }
 }
 const PxVec3 halfTurn=finiteJointRotation(PxQuat(PxIdentity),PxQuat(PxPi-1.e-4f,PxVec3(0,1,0)));++checks;failures+=halfTurn.y<3.0f;
 std::printf("[FINITE_JOINT_MATH] checks=%u failures=%u maxDerivativeError=%.9g maxObjectivityError=%.9g maxTorque=%.9g maxExchangeError=%.9g halfTurnStrain=%.9g\n",checks,failures,maxDerivativeError,maxObjectivityError,maxTorque,maxExchangeError,halfTurn.y);
 failures+=collisionFiltering();return failures?1:0;
}
