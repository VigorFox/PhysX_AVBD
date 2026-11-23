// Public SDK counterexamples for objective spring damping and its step epoch.
#include "../physx/snippets/snippetjointsoftbody/SnippetJointSoftBodyCommon.h"
#include "avbd/DyAvbdRigidMetric.h"
#include <cstdio>
#include <algorithm>
using namespace JointSoftBodySnippet;

static PxRigidDynamic* makeBody(World& world, const PxVec3& p)
{
 PxRigidDynamic* b=PxCreateDynamic(*world.physics,PxTransform(p),PxSphereGeometry(.1f),*world.material,1);
 if(!b || !PxRigidBodyExt::setMassAndUpdateInertia(*b,1)) return NULL;
 b->setLinearDamping(0);b->setAngularDamping(0);b->setMaxAngularVelocity(1000);
 world.scene->addActor(*b);return b;
}
static FiniteJoint* connect(World& world,PxRigidDynamic& a,PxRigidDynamic& b,PxReal k,PxReal d)
{
 FiniteJoint* joint=FiniteJoint::create(*world.physics,a,PxTransform(PxVec3(1,0,0)),b,PxTransform(PxVec3(-1,0,0)));
 if(joint) for(PxU32 r=0;r<6;++r) joint->setDrive(PxD6Drive::Enum(r),PxD6JointDrive(k,d,PX_MAX_F32,false));
 return joint;
}
static bool commonMotion(PxU32 workers,PxReal angle,bool transformed=false)
{
 World world;Settings settings;settings.workers=workers;
 if(!world.initScene(settings,"common-motion",PxVec3(0),false,0)) return false;
 PxRigidDynamic* a=makeBody(world,PxVec3(-1,0,0));PxRigidDynamic* b=makeBody(world,PxVec3(1,0,0));
 if(!a || !b) return false;
 const PxTransform initial=transformed?PxTransform(PxVec3(.7f,-.3f,1.1f),
  PxQuat(.83f,PxVec3(-2,1,3).getNormalized())):PxTransform(PxIdentity);
 a->setGlobalPose(initial*a->getGlobalPose());b->setGlobalPose(initial*b->getGlobalPose());
 FiniteJoint* joint=connect(world,*a,*b,1000,1000);if(!joint) return false;
 const PxReal h=1.f/60;const PxVec3 axis=PxVec3(1,2,3).getNormalized();const PxQuat rotation(angle,axis);
 const PxVec3 translation=transformed?PxVec3(.2f,-.1f,.3f):PxVec3(0);
 const PxVec3 targetA=translation+rotation.rotate(a->getGlobalPose().p),targetB=translation+rotation.rotate(b->getGlobalPose().p);
 const PxQuat orientation=rotation*initial.q;
 // The production predictor uses normalize(q + dt/2 * omega * q).
 // Prescribe its exact common finite pose; an unstrained joint must add no force.
 a->setLinearVelocity((targetA-a->getGlobalPose().p)/h);b->setLinearVelocity((targetB-b->getGlobalPose().p)/h);
 const PxVec3 omega=axis*(2*PxTan(angle/2)/h);a->setAngularVelocity(omega);b->setAngularVelocity(omega);
 world.scene->simulate(h);const bool fetched=world.scene->fetchResults(true);
 const PxReal positionError=PxMax((a->getGlobalPose().p-targetA).magnitude(),(b->getGlobalPose().p-targetB).magnitude());
 const PxReal rotationError=PxMax(finiteJointRotation(orientation,a->getGlobalPose().q).magnitude(),finiteJointRotation(orientation,b->getGlobalPose().q).magnitude());
 const bool pass=fetched && !world.errors.fatal.load() && positionError<2.e-5f && rotationError<2.e-5f;
 std::printf("[SPRING_DAMPING] case=common-motion angle=%.9g workers=%u transformed=%u positionError=%.9g rotationError=%.9g status=%s\n",angle,workers,unsigned(transformed),positionError,rotationError,pass?"PASS":"FAIL");
 joint->release();a->release();b->release();return pass;
}
static bool linearEpoch(PxU32 workers)
{
 World world;Settings settings;settings.workers=workers;
 if(!world.initScene(settings,"linear-epoch",PxVec3(0),false,0)) return false;
 PxRigidDynamic* a=makeBody(world,PxVec3(-1,0,0));PxRigidDynamic* b=makeBody(world,PxVec3(1,0,0));
 if(!a || !b) return false;
 const PxReal k=6,d=3,h=1.f/60;
 FiniteJoint* joint=connect(world,*a,*b,k,d);if(!joint) return false;
 bool pass=true;PxReal maxError=0;
 for(PxU32 step=0;step<2;++step)
 {
  const PxReal start=a->getGlobalPose().p.x-b->getGlobalPose().p.x+2;
  const PxReal relativeSpeed=step?0:-1;
  a->setLinearVelocity(PxVec3(relativeSpeed/2,0,0));b->setLinearVelocity(PxVec3(-relativeSpeed/2,0,0));
  // Independent closed-form backward Euler solution of two unit masses.
  const PxReal expected=(start*(1+2*d*h)+h*relativeSpeed)/(1+2*k*h*h+2*d*h);
  world.scene->simulate(h);pass&=world.scene->fetchResults(true);
  const PxReal actual=a->getGlobalPose().p.x-b->getGlobalPose().p.x+2;
  maxError=PxMax(maxError,PxAbs(expected-actual));
 }
 pass&=!world.errors.fatal.load() && maxError<2.e-6f;
 std::printf("[SPRING_DAMPING] case=linear-epoch workers=%u maxError=%.9g status=%s\n",workers,maxError,pass?"PASS":"FAIL");
 joint->release();a->release();b->release();return pass;
}
static bool lockedPredictor()
{
 World world;Settings settings;settings.workers=2;
 if(!world.initScene(settings,"locked-predictor",PxVec3(0),false,0)) return false;
 PxRigidDynamic* body=makeBody(world,PxVec3(1,2,3));if(!body) return false;
 body->setRigidDynamicLockFlag(PxRigidDynamicLockFlag::eLOCK_LINEAR_Y,true);
 body->setRigidDynamicLockFlag(PxRigidDynamicLockFlag::eLOCK_ANGULAR_X,true);
 const PxReal h=1.f/60;body->setLinearVelocity(PxVec3(3,0,-2));body->setAngularVelocity(PxVec3(0,40,-30));
 const PxQuat target=PxQuat(0,20*h,-15*h,1).getNormalized();const PxVec3 position=body->getGlobalPose().p+PxVec3(3,0,-2)*h;
 world.scene->simulate(h);const bool fetched=world.scene->fetchResults(true);
 const PxReal error=PxMax((body->getGlobalPose().p-position).magnitude(),finiteJointRotation(target,body->getGlobalPose().q).magnitude());
 const bool pass=fetched && !world.errors.fatal.load() && error<2.e-5f;
 std::printf("[SPRING_DAMPING] case=locked-predictor maxError=%.9g status=%s\n",error,pass?"PASS":"FAIL");
 body->release();return pass;
}
static bool springLimitActivation()
{
 World world;Settings settings;settings.workers=2;
 if(!world.initScene(settings,"spring-limit-activation",PxVec3(0),false,0)) return false;
 PxRigidDynamic* a=makeBody(world,PxVec3(-.495f,0,0));PxRigidDynamic* b=makeBody(world,PxVec3(.495f,0,0));
 if(!a || !b) return false;
 PxDistanceJoint* joint=PxDistanceJointCreate(*world.physics,a,PxTransform(PxIdentity),b,PxTransform(PxIdentity));
 if(!joint) return false;
 const PxReal k=6,d=3,h=1.f/60;
 joint->setMaxDistance(1);joint->setTolerance(0);joint->setStiffness(k);joint->setDamping(d);
 joint->setDistanceJointFlag(PxDistanceJointFlag::eMIN_DISTANCE_ENABLED,false);
 joint->setDistanceJointFlag(PxDistanceJointFlag::eMAX_DISTANCE_ENABLED,true);
 joint->setDistanceJointFlag(PxDistanceJointFlag::eSPRING_ENABLED,true);
 const PxReal start=b->getGlobalPose().p.x-a->getGlobalPose().p.x,speed=18;
 a->setLinearVelocity(PxVec3(-speed/2,0,0));b->setLinearVelocity(PxVec3(speed/2,0,0));
 const PxReal expected=(start+h*speed+2*k*h*h+2*d*h*start)/(1+2*k*h*h+2*d*h);
 world.scene->simulate(h);const bool fetched=world.scene->fetchResults(true);
 const PxReal actual=b->getGlobalPose().p.x-a->getGlobalPose().p.x,error=PxAbs(actual-expected);
 const bool pass=fetched && !world.errors.fatal.load() && error<2.e-6f;
 std::printf("[SPRING_DAMPING] case=spring-limit-activation expected=%.9g actual=%.9g error=%.9g status=%s\n",expected,actual,error,pass?"PASS":"FAIL");
 joint->release();a->release();b->release();return pass;
}
static bool coordinateObjectivity()
{
 FiniteJointData data;data.c2b[0]=PxTransform(PxVec3(1,0,0));data.c2b[1]=PxTransform(PxVec3(-1,0,0));
 for(PxU32 r=0;r<6;++r) data.drives[r]=PxD6JointDrive(1000,1000,PX_MAX_F32,false);
 const PxTransform a(PxVec3(-1,0,0)),b(PxVec3(1,0,0));
 PxReal maximum=0,oldMaximum=0;
 for(PxU32 sample=0;sample<48;++sample)
 {
  const PxTransform motion(PxVec3(.4f,-.2f,.6f),PxQuat(3.f*sample/47,PxVec3(1,2,3).getNormalized()));
  const PxTransform endA=motion*a,endB=motion*b;
  Px1DConstraint start[FINITE_JOINT_ROWS],end[FINITE_JOINT_ROWS];PxConstraintInvMassScale scale;PxVec3p offset,anchorA,anchorB;
  finiteJointPrep(start,offset,FINITE_JOINT_ROWS,scale,&data,a,b,false,anchorA,anchorB);
  finiteJointPrep(end,offset,FINITE_JOINT_ROWS,scale,&data,endA,endB,false,anchorA,anchorB);
  const PxVec3 angularStep=motion.q.getImaginaryPart()*2;
  for(PxU32 row=0;row<FINITE_JOINT_ROWS;++row)
  {
   maximum=PxMax(maximum,PxAbs(physx::Dy::avbdFiniteSpringResidual(1000,60000,end[row].geometricError,start[row].geometricError,0)));
   const PxReal old=end[row].linear0.dot(endA.p-a.p)-end[row].linear1.dot(endB.p-b.p)+
    (end[row].angular0-end[row].angular1).dot(angularStep);
   oldMaximum=PxMax(oldMaximum,PxAbs(old));
  }
 }
 const bool pass=maximum<2.e-6f && oldMaximum>1;
 std::printf("[SPRING_DAMPING] case=coordinate-objectivity angles=48 maxResidual=%.9g oldProjectedIncrement=%.9g status=%s\n",maximum,oldMaximum,pass?"PASS":"FAIL");
 return pass;
}
int main(int argc,char** argv)
{
 // Finite target admission is a whole-step qualification, separate from
 // the material coordinate/derivative tests below.
 if(argc>1 && std::strcmp(argv[1],"--large-motion")==0)
 {
  bool pass=true;
  for(PxU32 workers=0;workers<=2;workers+=2)
  {
   pass=commonMotion(workers,1.2f) && pass;
   pass=commonMotion(workers,2.4f,true) && pass;
  }
  return pass?0:1;
 }
 PxU32 failed=0;
 for(PxU32 workers=0;workers<=2;workers+=2) failed+=!linearEpoch(workers);
 failed+=!coordinateObjectivity();
 failed+=!lockedPredictor();
 failed+=!springLimitActivation();
 // Differentiate the independently authored incremental elastic+damping energy.
 PxReal maxError=0;
 for(PxU32 i=0;i<100;++i)
 {
  const double c=-.8+.017*i,c0=.12,k=18,d=37,target=.03,eps=1.e-5;
  const auto energy=[&](double x){return .5*k*x*x+.5*d*(x-c0-target)*(x-c0-target);};
  const double expected=(energy(c+eps)-energy(c-eps))/(2*eps);
  const double actual=(k+d)*physx::Dy::avbdFiniteSpringResidual(PxReal(k),PxReal(d),PxReal(c),PxReal(c0),PxReal(target));
  const PxReal error=PxReal(std::abs(expected-actual)/std::max(1.,std::abs(expected)));maxError=PxMax(maxError,error);
 }
 const bool derivative=maxError<2.e-6f;failed+=!derivative;
 std::printf("[SPRING_DAMPING] case=energy-gradient maxRelativeError=%.9g status=%s\n",maxError,derivative?"PASS":"FAIL");
 return failed?1:0;
}
