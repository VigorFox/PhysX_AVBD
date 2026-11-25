#include "PxPhysicsAPI.h"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
using namespace physx;

struct Contacts : PxSimulationEventCallback
{
    PxActor* supports[2]={};
    unsigned mask=0;
    double load[2]={};
    double supportMoment=0.;
    void onContact(const PxContactPairHeader& header,const PxContactPair* pairs,PxU32 count) PX_OVERRIDE
    {
        for(PxU32 i=0;i<count;++i)
            if(pairs[i].contactCount)
                for(unsigned j=0;j<2;++j)
                    if(supports[j]==header.actors[0] || supports[j]==header.actors[1])
                    {
                        mask|=1u<<j;
                        PxContactPairPoint points[16];
                        const PxU32 n=pairs[i].extractContacts(points,16);
                        for(PxU32 k=0;k<n;++k)
                        {
                            const double normal=std::abs(points[k].impulse.y);
                            load[j]+=normal;
                            supportMoment+=points[k].position.x*normal;
                        }
                    }
    }
    void onConstraintBreak(PxConstraintInfo*,PxU32) PX_OVERRIDE {}
    void onWake(PxActor**,PxU32) PX_OVERRIDE {}
    void onSleep(PxActor**,PxU32) PX_OVERRIDE {}
    void onTrigger(PxTriggerPair*,PxU32) PX_OVERRIDE {}
    void onAdvance(const PxRigidBody*const*,const PxTransform*,const PxU32) PX_OVERRIDE {}
};

static PxFilterFlags report(PxFilterObjectAttributes,PxFilterData,PxFilterObjectAttributes,PxFilterData,
    PxPairFlags& flags,const void*,PxU32)
{
    flags=PxPairFlag::eCONTACT_DEFAULT|PxPairFlag::eNOTIFY_TOUCH_FOUND|
        PxPairFlag::eNOTIFY_TOUCH_PERSISTS|PxPairFlag::eNOTIFY_CONTACT_POINTS;
    return PxFilterFlag::eDEFAULT;
}

static bool impactReport(PxPhysics& physics,PxSceneDesc desc,float cap,float dt)
{
    Contacts contacts;
    desc.simulationEventCallback=&contacts;
    desc.gravity=PxVec3(0);
    auto* scene=physics.createScene(desc);
    auto* material=physics.createMaterial(0,0,.65f);
    auto* support=PxCreatePlane(physics,PxPlane(0,1,0,0),*material);
    contacts.supports[0]=support;
    auto* ball=PxCreateDynamic(physics,PxTransform(PxVec3(0,.49f,0)),PxSphereGeometry(.5f),*material,1.f);
    PxRigidBodyExt::setMassAndUpdateInertia(*ball,2.f);
    ball->setMaxContactImpulse(cap);
    ball->setSolverIterationCounts(4,1);
    ball->setLinearDamping(0);
    ball->setLinearVelocity(PxVec3(0,-4,0));
    scene->addActor(*support);
    scene->addActor(*ball);
    scene->simulate(dt);
    const bool fetched=scene->fetchResults(true);
    const double measured=2.*(ball->getLinearVelocity().y+4.);
    const double reported=contacts.load[0];
    // A fixed plane returns m * (1 + e) * closingSpeed, subject to the public limit.
    const double expected = PxMin(double(cap), 2.0 * (1.0 + 0.65) * 4.0);
    const double impactTolerance = 2.e-5 + 0.005 * expected;
    const bool ok = fetched && contacts.mask == 1 && measured >= -1.e-5 && measured <= cap + 2.e-5 &&
        std::abs(measured - expected) <= impactTolerance && std::abs(measured - reported) < 2.e-5;
    std::printf("[IMPACT_REPORT] cap=%.8g expected=%.8g measured=%.8g reported=%.8g status=%s\n",
        double(cap), expected, measured, reported, ok ? "PASS" : "FAIL");
    ball->release();support->release();scene->release();material->release();
    return ok;
}

static bool sphereRolling(PxPhysics& physics, PxSceneDesc desc, PxU32 hz, bool friction, bool rolling)
{
    desc.simulationEventCallback=nullptr;
    PxScene* scene=physics.createScene(desc);
    auto* material=physics.createMaterial(friction ? .5f : 0.f, friction ? .5f : 0.f, 0);
    auto* ground=PxCreatePlane(physics,PxPlane(0,1,0,0),*material);
    auto* ball=PxCreateDynamic(physics,PxTransform(PxVec3(0,1,0)),PxSphereGeometry(1),*material,100);
    ball->setAngularDamping(0);
    ball->setLinearDamping(0);
    ball->setAngularVelocity(PxVec3(0,0,-5));
    if(rolling)ball->setLinearVelocity(PxVec3(5,0,0));
    scene->addActor(*ground);
    scene->addActor(*ball);
    bool valid=true;
    for(PxU32 i=0;i<hz*10 && valid;++i)
    {
        scene->simulate(1.f/hz);
        valid=scene->fetchResults(true);
    }
    // I=2*m*R^2/5. Coulomb sliding conserves angular momentum about the
    // plane contact until v=-R*w. Pure rolling needs no tangential impulse.
    const PxReal vx=rolling ? 5.f : friction ? 10.f/7.f : 0.f;
    const PxReal wz=friction ? -vx : -5.f;
    const auto v=ball->getLinearVelocity(),w=ball->getAngularVelocity();
    valid &= (v-PxVec3(vx,0,0)).magnitude()<.02f && (w-PxVec3(0,0,wz)).magnitude()<.02f &&
        PxAbs(ball->getGlobalPose().p.y-1)<.02f;
    std::printf("[SPHERE_ROLL] hz=%u friction=%u rolling=%u vx=%g wz=%g status=%s\n",
        hz,unsigned(friction),unsigned(rolling),double(v.x),double(w.z),valid?"PASS":"FAIL");
    ball->release();ground->release();scene->release();material->release();
    return valid;
}

static bool offsetSphereImpact(PxPhysics& physics, PxSceneDesc desc, PxReal dt)
{
    desc.gravity=PxVec3(0);
    desc.simulationEventCallback=nullptr;
    auto* scene=physics.createScene(desc);
    auto* material=physics.createMaterial(0,0,.2f);
    auto* ground=PxCreatePlane(physics,PxPlane(0,1,0,0),*material);
    auto* ball=PxCreateDynamic(physics,PxTransform(PxVec3(0,1.02f,0)),PxSphereGeometry(1),*material,100);
    PxShape* shape=nullptr;
    ball->getShapes(&shape,1);
    shape->setLocalPose(PxTransform(PxVec3(.15f,0,0)));
    ball->setCMassLocalPose(PxTransform(PxVec3(.5f,0,0)));
    ball->setLinearVelocity(PxVec3(0,-4,0));
    ball->setLinearDamping(0);
    ball->setAngularDamping(0);
    scene->addActor(*ground);
    scene->addActor(*ball);
    bool ok=true;
    for(PxU32 i=0;i<8 && ok && ball->getLinearVelocity().y < -3.99f;++i)
    {
        scene->simulate(dt);
        ok=scene->fetchResults(true);
    }
    // A vertical impulse has a real moment about the displaced center of mass.
    const PxReal impulse=ball->getMass()*(ball->getLinearVelocity().y+4);
    const PxReal expected=-.35f*impulse/ball->getMassSpaceInertiaTensor().z;
    const PxReal spin=ball->getAngularVelocity().z;
    ok &= expected < -1 && PxAbs(spin-expected)<.02f*PxAbs(expected);
    std::printf("[SPHERE_OFFSET] wz=%g expected=%g status=%s\n",double(spin),double(expected),ok?"PASS":"FAIL");
    ball->release();ground->release();scene->release();material->release();
    return ok;
}

static bool sphereGlance(PxPhysics& physics, PxSceneDesc desc, PxReal dt)
{
    desc.gravity=PxVec3(0);
    desc.simulationEventCallback=nullptr;
    auto* scene=physics.createScene(desc);
    auto* material=physics.createMaterial(0,0,.5f);
    auto* a=PxCreateDynamic(physics,PxTransform(PxVec3(-2,.8f,0)),PxSphereGeometry(1),*material,100);
    auto* b=PxCreateDynamic(physics,PxTransform(PxVec3(2,0,0)),PxSphereGeometry(1),*material,100);
    PxRigidDynamic* spheres[]={a,b};
    for(auto* sphere : spheres)
    {
        sphere->setLinearDamping(0);
        sphere->setAngularDamping(0);
    }
    a->setLinearVelocity(PxVec3(16,0,0));
    scene->addActor(*a);scene->addActor(*b);
    bool ok=true;
    for(PxU32 i=0;i<PxU32(.6f/dt) && ok;++i)
    {
        scene->simulate(dt);ok=scene->fetchResults(true);
    }
    const PxReal spin=PxMax(a->getAngularVelocity().magnitude(),b->getAngularVelocity().magnitude());
    const PxReal speedB=b->getLinearVelocity().magnitude();
    const PxReal momentumError=(a->getLinearVelocity()+b->getLinearVelocity()-PxVec3(16,0,0)).magnitude();
    ok &= spin<.002f && speedB>1 && momentumError<.02f;
    std::printf("[SPHERE_GLANCE] spin=%g speedB=%g momentumErrorPerMass=%g status=%s\n",
        double(spin),double(speedB),double(momentumError),ok?"PASS":"FAIL");
    a->release();b->release();scene->release();material->release();
    return ok;
}

static bool groundedSphereFriction(PxPhysics& physics, PxSceneDesc desc, PxReal dt)
{
    desc.gravity=PxVec3(0);
    desc.simulationEventCallback=nullptr;
    auto* scene=physics.createScene(desc);
    auto* material=physics.createMaterial(.8f,.8f,0);
    auto* fixed=PxCreateDynamic(physics,PxTransform(PxIdentity),PxSphereGeometry(1),*material,100);
    auto* ball=PxCreateDynamic(physics,PxTransform(PxVec3(-3,0,0)),PxSphereGeometry(1),*material,100);
    auto* joint=PxFixedJointCreate(physics,fixed,PxTransform(PxIdentity),nullptr,PxTransform(PxIdentity));
    ball->setLinearDamping(0);
    ball->setAngularDamping(0);
    ball->setLinearVelocity(PxVec3(8,0,0));
    ball->setAngularVelocity(PxVec3(0,0,6));
    scene->addActor(*fixed);
    scene->addActor(*ball);
    bool ok=true;
    PxReal expected=0, reported=0;
    for(PxU32 i=0;i<PxU32(.5f/dt) && ok;++i)
    {
        const PxVec3 arm=ball->getGlobalPose().p, v=ball->getLinearVelocity(), w=ball->getAngularVelocity();
        scene->simulate(dt);
        ok=scene->fetchResults(true);
        const PxVec3 impulse=(ball->getLinearVelocity()-v)*ball->getMass();
        // Angular impulse about the fixed joint origin, at the same DCD epoch.
        expected+=(arm.cross(impulse)+(ball->getAngularVelocity()-w)*ball->getMassSpaceInertiaTensor().z).z;
        PxVec3 force, torque;
        joint->getConstraint()->getForce(force,torque);
        reported+=dt*torque.z;
    }
    ok &= PxAbs(expected)>10 && PxAbs(reported-expected)<.03f*PxAbs(expected);
    std::printf("[SPHERE_FIXED] expected=%g reported=%g status=%s\n",double(expected),double(reported),ok?"PASS":"FAIL");
    joint->release();ball->release();fixed->release();scene->release();material->release();
    return ok;
}

static bool freeJointSpin(PxPhysics& physics, PxSceneDesc desc, unsigned hz)
{
    desc.gravity = PxVec3(0);
    desc.simulationEventCallback = nullptr;
    PxScene* scene = physics.createScene(desc);
    const PxReal inertias[] = { 0.75f, 1.25f };
    PxRigidDynamic* members[2];
    for (unsigned i = 0; i < 2; ++i)
    {
        members[i] = physics.createRigidDynamic(PxTransform(PxVec3(0, 0, i ? 0.4f : -0.4f)));
        members[i]->setMass(2.f);
        members[i]->setMassSpaceInertiaTensor(PxVec3(inertias[i]));
        members[i]->setLinearDamping(0);
        members[i]->setAngularDamping(0);
        members[i]->setSleepThreshold(0);
        members[i]->setSolverIterationCounts(4, 1);
        members[i]->setAngularVelocity(PxVec3(0, 0, 2.f));
        scene->addActor(*members[i]);
    }
    PxFixedJoint* joint = PxFixedJointCreate(physics, members[0], PxTransform(PxVec3(0, 0, 0.4f)),
                                             members[1], PxTransform(PxVec3(0, 0, -0.4f)));
    bool fetched = true;
    PxReal worstMomentum = 0, worstRelative = 0;
    for (unsigned i = 0; i < hz / 5; ++i)
    {
        scene->simulate(1.f / PxReal(hz));
        if (!scene->fetchResults(true)) { fetched = false; break; }
        PxVec3 angularMomentum(0);
        for (unsigned j = 0; j < 2; ++j)
            angularMomentum += members[j]->getGlobalPose().p.cross(2.f * members[j]->getLinearVelocity()) +
                               inertias[j] * members[j]->getAngularVelocity();
        worstMomentum = PxMax(worstMomentum, (angularMomentum - PxVec3(0, 0, 4.f)).magnitude());
        worstRelative = PxMax(worstRelative,
                              (members[0]->getAngularVelocity() - members[1]->getAngularVelocity()).magnitude());
    }
    const bool ok = fetched && worstMomentum < 0.05f && worstRelative < 0.01f;
    std::printf("[FREE_JOINT_SPIN] angularMomentumError_Nms=%.8g relativeSpeed_rad_s=%.8g status=%s\n",
                double(worstMomentum), double(worstRelative), ok ? "PASS" : "FAIL");
    joint->release();
    for (auto* member : members) member->release();
    scene->release();
    return ok;
}

int main(int argc,char** argv)
{
    PxDefaultAllocator allocator;
    PxDefaultErrorCallback error;
    auto* foundation=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,error);
    auto* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation,PxTolerancesScale(),false,nullptr);
    PxInitExtensions(*physics,nullptr);
    auto* dispatcher=PxDefaultCpuDispatcherCreate(0);
    Contacts contacts;
    PxSceneDesc desc(physics->getTolerancesScale());
    desc.solverType=argc>1 && !std::strcmp(argv[1],"tgs")?PxSolverType::eTGS:PxSolverType::eAVBD;
    desc.gravity=PxVec3(0,-9.81f,0);
    desc.cpuDispatcher=dispatcher;
    desc.filterShader=report;
    desc.simulationEventCallback=&contacts;
    auto* scene=physics->createScene(desc);
    auto* material=physics->createMaterial(.5f,.5f,0);
    PxRigidStatic* supports[2];
    for(unsigned i=0;i<2;++i)
    {
        supports[i]=PxCreateStatic(*physics,PxTransform(PxVec3(i?.8f:-.8f,.15f,0)),
                                   PxBoxGeometry(.18f,.15f,.45f),*material);
        contacts.supports[i]=supports[i];
        scene->addActor(*supports[i]);
    }
    const PxVec3 offset(.25f,.40f,0);
    auto* beam=PxCreateDynamic(*physics,PxTransform(PxVec3(0,.45f,0)),PxBoxGeometry(1.2f,.15f,.3f),*material,1.f);
    PxRigidBodyExt::setMassAndUpdateInertia(*beam,4.f);
    auto* load=physics->createRigidDynamic(PxTransform(beam->getGlobalPose().p+offset));
    load->setMass(2.f);
    load->setMassSpaceInertiaTensor(PxVec3(.06f));
    PxRigidDynamic* bodies[]={beam,load};
    for(auto* body : bodies)
    {
        body->setSolverIterationCounts(4,1);
        body->setSleepThreshold(0.f);
        body->setLinearDamping(0.f);
        body->setAngularDamping(0.f);
        scene->addActor(*body);
    }
    auto* joint=PxFixedJointCreate(*physics,beam,PxTransform(offset),load,PxTransform(PxIdentity));
    const auto momentum=[&](){return beam->getLinearVelocity()*4.f+load->getLinearVelocity()*2.f;};
    const unsigned hz=argc>2?unsigned(std::atoi(argv[2])):60;
    if(hz!=30 && hz!=60 && hz!=120)return 4;
    const float dt=1.f/float(hz);
    PxReal maxAnchor=0.f,minHeight=1.f,maxAngle=0.f,releaseError=0.f;
    unsigned two=0;
    double supportedImpulse[2]={};
    double moment=0., gravityMoment=0.;
    PxVec3 beforeRelease(0);
    for(unsigned i=0;i<2*hz+hz/5;++i)
    {
        if(i==2*hz)
        {
            beforeRelease=momentum();
            for(auto* actor:supports) scene->removeActor(*actor);
            beam->wakeUp();
            load->wakeUp();
        }
        contacts.mask=0;
        contacts.load[0]=contacts.load[1]=0.;
        contacts.supportMoment=0.;
        scene->simulate(dt);
        if(!scene->fetchResults(true))return 2;
        const auto p=beam->getGlobalPose(), q=load->getGlobalPose();
        if(!p.isValid() || !q.isValid() || !momentum().isFinite())return 3;
        maxAnchor=PxMax(maxAnchor,(p.transform(offset)-q.p).magnitude());
        maxAngle=PxMax(maxAngle,p.q.getAngle(q.q));
        if(i>=hz && i<2*hz)
        {
            minHeight=PxMin(minHeight,p.p.y);
            two+=contacts.mask==3;
            for(unsigned j=0;j<2;++j)supportedImpulse[j]+=contacts.load[j];
            moment+=contacts.supportMoment;
            gravityMoment+=dt*9.81*(4.*p.p.x+2.*q.p.x);
        }
        if(i>=2*hz)
        {
            const PxVec3 expected=beforeRelease+desc.gravity*(6.f*dt*float(i-2*hz+1));
            releaseError=PxMax(releaseError,(momentum()-expected).magnitude());
        }
    }
    const double total=supportedImpulse[0]+supportedImpulse[1];
    const bool pass=two*100>=90*hz && minHeight>.43f && maxAnchor<.012f && maxAngle<.035f && releaseError<.15f &&
        supportedImpulse[0]>0. && supportedImpulse[1]>0. && std::abs(total-6.*9.81)<.6 &&
        std::abs(moment-gravityMoment)<.35;
    std::printf("[TWO_SUPPORTS] simultaneous=%u/%u height=%.8g anchor_m=%.8g angle_rad=%.8g release_Ns=%.8g left_Ns=%.8g right_Ns=%.8g support_Nms=%.8g gravity_Nms=%.8g status=%s\n",
        two,hz,double(minHeight),double(maxAnchor),double(maxAngle),double(releaseError),
        supportedImpulse[0],supportedImpulse[1],moment,gravityMoment,pass?"PASS":"FAIL");
    joint->release();
    load->release();beam->release();
    for(auto* actor:supports) actor->release();
    material->release();scene->release();
    bool impacts=true;
    const float limits[]={0.f,.01f,20.f};
    for(float limit:limits) impacts&=impactReport(*physics,desc,limit,dt);
    const bool spin = freeJointSpin(*physics, desc, hz);
    bool spheres=sphereGlance(*physics,desc,dt);
    spheres &= offsetSphereImpact(*physics,desc,dt);
    if(desc.solverType==PxSolverType::eAVBD)
        spheres &= groundedSphereFriction(*physics,desc,dt);
    const bool choices[]={false,true};
    for(bool friction : choices)
        for(bool rolling : choices)
            spheres &= sphereRolling(*physics,desc,hz,friction,rolling);
    dispatcher->release();PxCloseExtensions();physics->release();foundation->release();
    return pass && impacts && spin && spheres ? 0 : 1;
}
