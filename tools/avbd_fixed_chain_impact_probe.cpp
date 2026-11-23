#include "PxPhysicsAPI.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
using namespace physx;

struct Errors : PxErrorCallback {
  unsigned count = 0;
  void reportError(PxErrorCode::Enum code, const char* message, const char*, int) override {
    if (code != PxErrorCode::eDEBUG_INFO) { ++count; std::fprintf(stderr, "%s\n", message); }
  }
};
struct Contacts : PxSimulationEventCallback {
  unsigned hits = 0;
  unsigned breaks = 0;
  unsigned brokenMask = 0;
  PxConstraint* expected[5]={};
  void onConstraintBreak(PxConstraintInfo* info, PxU32 n) override {
    breaks += n;
    for(PxU32 k=0;k<n;++k) {
      unsigned j=0;
      while(j<5 && expected[j]!=info[k].constraint) ++j;
      brokenMask |= 1u<<j;
    }
  }
  void onWake(PxActor**, PxU32) override {}
  void onSleep(PxActor**, PxU32) override {}
  void onTrigger(PxTriggerPair*, PxU32) override {}
  void onAdvance(const PxRigidBody* const*, const PxTransform*, PxU32) override {}
  void onContact(const PxContactPairHeader&, const PxContactPair* pairs, PxU32 n) override {
    for (PxU32 i = 0; i < n; ++i) if (pairs[i].contactCount) ++hits;
  }
};
static PxFilterFlags filter(PxFilterObjectAttributes, PxFilterData,
    PxFilterObjectAttributes, PxFilterData, PxPairFlags& flags, const void*, PxU32) {
  flags = PxPairFlag::eCONTACT_DEFAULT | PxPairFlag::eNOTIFY_TOUCH_FOUND |
      PxPairFlag::eNOTIFY_TOUCH_PERSISTS | PxPairFlag::eNOTIFY_CONTACT_POINTS;
  return PxFilterFlag::eDEFAULT;
}
static double energy(const PxRigidDynamic& b) {
  const PxVec3 w = (b.getGlobalPose() * b.getCMassLocalPose()).q.rotateInv(b.getAngularVelocity());
  return .5 * b.getMass() * double(b.getLinearVelocity().magnitudeSquared()) +
      .5 * double(w.dot(b.getMassSpaceInertiaTensor().multiply(w)));
}

// A public-API reproduction of SnippetJoint's five-box chain and space-key
// projectile. Optional scalars permit light/strong and centered/oblique controls.
int main(int argc, char** argv) {
  const bool tgs = argc > 1 && std::strcmp(argv[1], "tgs") == 0;
  const bool revolute = argc > 2 && std::strcmp(argv[2], "revolute") == 0;
  const bool fixed = argc > 2 && std::strcmp(argv[2], "fixed") == 0;
  const float speed = argc > 3 ? float(std::atof(argv[3])) : 200.f;
  const float radius = argc > 4 ? float(std::atof(argv[4])) : 3.f;
  const float offset = argc > 5 ? float(std::atof(argv[5])) : .25f;
  const float slope = argc > 6 ? float(std::atof(argv[6])) : 0.f;
  const float gravity = argc > 7 ? float(std::atof(argv[7])) : -9.81f;
  const unsigned frames = argc > 8 ? unsigned(std::atoi(argv[8])) : 1200u;
  const unsigned launchFrame = argc > 9 ? unsigned(std::atoi(argv[9])) : 120u;
  const float breakForce = argc > 10 ? float(std::atof(argv[10])) : 1000.f;
  const float breakTorque = argc > 11 ? float(std::atof(argv[11])) : breakForce * 100.f;
  const float dt = argc > 12 ? float(std::atof(argv[12])) : 1.f / 60.f;
  const unsigned maxShots = argc > 13 ? unsigned(std::atoi(argv[13])) : 1u;
  PxDefaultAllocator allocator; Errors errors; Contacts contacts;
  PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
  PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
  PxInitExtensions(*physics, nullptr);
  PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(2);
  PxSceneDesc desc(physics->getTolerancesScale());
  desc.gravity = PxVec3(0, gravity, 0); desc.cpuDispatcher = dispatcher;
  desc.solverType = tgs ? PxSolverType::eTGS : PxSolverType::eAVBD;
  desc.filterShader = filter; desc.simulationEventCallback = &contacts;
  PxScene* scene = physics->createScene(desc);
  PxMaterial* material = physics->createMaterial(.5f, .5f, .6f);
  std::vector<PxRigidDynamic*> bodies;
  std::vector<PxJoint*> joints;
  for (unsigned i = 0; i < 5; ++i) {
    auto* body = PxCreateDynamic(*physics, PxTransform(PxVec3(2.f + i * 4, 20, 0)),
        PxBoxGeometry(2, .5f, .5f), *material, 1.f);
    auto* previous = i ? bodies.back() : nullptr;
    const PxTransform anchor = i ? PxTransform(PxVec3(2,0,0)) : PxTransform(PxVec3(0,20,0));
    PxJoint* joint;
    if (fixed) {
      auto* j = PxFixedJointCreate(*physics, previous, anchor, body, PxTransform(PxVec3(-2,0,0)));
      j->setBreakForce(breakForce, breakTorque);
      j->setConstraintFlag(PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES, true);
      j->setConstraintFlag(PxConstraintFlag::eDISABLE_PREPROCESSING, true); joint = j;
    } else if (revolute) {
      auto* j = PxRevoluteJointCreate(*physics, previous, anchor, body, PxTransform(PxVec3(-2,0,0)));
      j->setLimit(PxJointAngularLimitPair(-PxPi/4, PxPi/4));
      j->setRevoluteJointFlag(PxRevoluteJointFlag::eLIMIT_ENABLED, true); joint = j;
    } else {
      auto* j = PxPrismaticJointCreate(*physics, previous, anchor, body, PxTransform(PxVec3(-2,0,0)));
      j->setLimit(PxJointLinearLimitPair(-2, 2, PxSpring(0,0)));
      j->setPrismaticJointFlag(PxPrismaticJointFlag::eLIMIT_ENABLED, true); joint = j;
    }
    contacts.expected[i]=joint->getConstraint();
    scene->addActor(*body); bodies.push_back(body); joints.push_back(joint);
  }
  PxRigidDynamic* ball = nullptr;
  unsigned shots = 0, lastLaunch = launchFrame;
  std::printf("frame,contacts,chain_ke,chain_pe,ball_ke,ball_pe,max_v,max_w,anchor,limit,broken,callbacks,ball_vx,ball_vy,ball_vz");
  for (unsigned b=0; b<5; ++b) std::printf(",x%u,y%u,z%u,vx%u,vy%u,vz%u,wx%u,wy%u,wz%u",b,b,b,b,b,b,b,b,b);
  for (unsigned j=0; j<5; ++j) std::printf(",force%u,torque%u",j,j);
  for (unsigned b=0; b<5; ++b) std::printf(",qx%u,qy%u,qz%u,qw%u,sleep%u",b,b,b,b,b);
  std::printf(",shots,callback_mask\n");
  for (unsigned frame = 0; frame < frames; ++frame) {
    bool allSleeping = true;
    for (auto* body : bodies) allSleeping = allSleeping && body->isSleeping();
    if (frame == launchFrame || (shots && shots < maxShots &&
        frame > lastLaunch + 60u && allSleeping)) {
      const PxVec3 direction = PxVec3(slope,0,-1).getNormalized();
      ball = PxCreateDynamic(*physics,
          PxTransform(bodies[2]->getGlobalPose().p + PxVec3(0,offset,0) - direction*12.f),
          PxSphereGeometry(radius), *material, 10.f);
      ball->setAngularDamping(.5f); ball->setLinearVelocity(direction*speed);
      scene->addActor(*ball);
      ++shots; lastLaunch = frame;
    }
    contacts.hits=0; scene->simulate(dt);
    if (!scene->fetchResults(true)) return 2;
    double ke=0, pe=0; float v=0,w=0,anchorError=0,limitError=0;
    for (auto* body : bodies) {
      ke+=energy(*body); pe-=body->getMass()*gravity*(body->getGlobalPose().p.y-20.);
      v=PxMax(v,body->getLinearVelocity().magnitude()); w=PxMax(w,body->getAngularVelocity().magnitude());
    }
    unsigned broken = 0;
    for (unsigned i=0; i<5; ++i) {
      if (joints[i]->getConstraintFlags() & PxConstraintFlag::eBROKEN) {
        broken |= 1u << i;
        continue;
      }
      const auto a = (i ? bodies[i-1]->getGlobalPose() : PxTransform(PxIdentity)) *
          joints[i]->getLocalPose(PxJointActorIndex::eACTOR0);
      const auto b = bodies[i]->getGlobalPose() * joints[i]->getLocalPose(PxJointActorIndex::eACTOR1);
      const auto delta=b.p-a.p, axis=a.q.rotate(PxVec3(1,0,0));
      anchorError=PxMax(anchorError,((revolute || fixed)?delta:delta-axis*axis.dot(delta)).magnitude());
      if (fixed) continue;
      float coordinate=revolute?static_cast<PxRevoluteJoint*>(joints[i])->getAngle():
          static_cast<PxPrismaticJoint*>(joints[i])->getPosition();
      // getAngle preserves the quaternion's cover; the physical twist limit
      // uses the principal angle, as does the AVBD joint geometry evaluator.
      if (revolute) coordinate=std::remainder(coordinate,2*PxPi);
      limitError=PxMax(limitError,PxAbs(coordinate)-(revolute?PxPi/4:2.f));
    }
    std::printf("%u,%u,%.12g,%.12g,%.12g,%.12g,%.9g,%.9g,%.9g,%.9g",frame+1,contacts.hits,ke,pe,
        ball?energy(*ball):0,ball?-ball->getMass()*gravity*(ball->getGlobalPose().p.y-20.):0,v,w,anchorError,limitError);
    const PxVec3 ballVelocity = ball ? ball->getLinearVelocity() : PxVec3(0);
    std::printf(",%u,%u,%.9g,%.9g,%.9g",broken,contacts.breaks,ballVelocity.x,ballVelocity.y,ballVelocity.z);
    for (auto* body:bodies) {
      const auto p=body->getGlobalPose().p, lv=body->getLinearVelocity(), av=body->getAngularVelocity();
      std::printf(",%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g",p.x,p.y,p.z,lv.x,lv.y,lv.z,av.x,av.y,av.z);
    }
    for (auto* joint:joints) {
      PxVec3 force, torque;
      joint->getConstraint()->getForce(force, torque);
      std::printf(",%.9g,%.9g", force.magnitude(), torque.magnitude());
    }
    for (auto* body:bodies) {
      const PxQuat q=body->getGlobalPose().q;
      std::printf(",%.9g,%.9g,%.9g,%.9g,%u",q.x,q.y,q.z,q.w,unsigned(body->isSleeping()));
    }
    std::printf(",%u,%u\n", shots,contacts.brokenMask);
  }
  for(auto* joint:joints) joint->release();
  scene->release(); material->release(); dispatcher->release();
  PxCloseExtensions(); physics->release(); foundation->release();
  return errors.count ? 1 : 0;
}
