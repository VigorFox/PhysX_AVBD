#include "PxPhysicsAPI.h"
#include <cstring>
#include "../physx/snippets/snippetjointsoftbody/SnippetFiniteJoint.h"
#include "../physx/snippets/snippetjointsoftbody/SnippetVolumeMaterial.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace physx;
struct Frame:PxConstraintConnector
{
    JointSoftBodySnippet::FiniteJointData data;PxConstraint* constraint=NULL;
    void* prepareData() PX_OVERRIDE{return &data;}
    void onConstraintRelease() PX_OVERRIDE{constraint=NULL;}
    void onComShift(PxU32) PX_OVERRIDE{if(constraint)constraint->markDirty();}
    void onOriginShift(const PxVec3&) PX_OVERRIDE{}
    void* getExternalReference(PxU32& id) PX_OVERRIDE{id=PxConcreteType::eFIRST_USER_EXTENSION+1;return this;}
    bool updatePvdProperties(pvdsdk::PvdDataStream&,const PxConstraint*,PxPvdUpdateType::Enum) const PX_OVERRIDE{return false;}
    void updateOmniPvdProperties() const PX_OVERRIDE{}
    PxBase* getSerializable() PX_OVERRIDE{return NULL;}
    PxConstraintSolverPrep getPrep() const PX_OVERRIDE{return JointSoftBodySnippet::finiteJointPrep;}
    const void* getConstantBlock() const PX_OVERRIDE{return &data;}
};
struct Modify:PxContactModifyCallback
{
    float impulse=PX_MAX_F32;
    void onContactModify(PxContactModifyPair* const pairs,PxU32 count) PX_OVERRIDE
    {for(PxU32 p=0;p<count;++p)for(PxU32 i=0;i<pairs[p].contacts.size();++i)pairs[p].contacts.setMaxImpulse(i,impulse);}
};
static bool useModification=true;
static PxFilterFlags filter(PxFilterObjectAttributes,PxFilterData,PxFilterObjectAttributes,PxFilterData,
    PxPairFlags& flags,const void*,PxU32)
{flags=PxPairFlag::eCONTACT_DEFAULT;if(useModification)flags|=PxPairFlag::eMODIFY_CONTACTS;return PxFilterFlag::eDEFAULT;}
// Independent analytic chain, deliberately crossing three four-body execution
// groups. The original isotropic frame energy is exactly k/2 |x_i-x_j|^2.
// Compare against its tridiagonal implicit solution, not solver internals.
static bool chain(PxPhysics& physics,PxScene& scene,unsigned N)
{
    const double stiffness=1000,dt=double(1.f/60);
    PxArray<PxRigidDynamic*> bodies(N,NULL);PxArray<Frame> frames(N-1);
    PxArray<double> diagonal(N),rhs(N),expected(N);double initialMomentum=0;
    scene.setGravity(PxVec3(0));
    for(unsigned i=0;i<N;++i)
    {
        const float mass=1+.1f*i,velocity=.2f*(int(i%5)-2);
        bodies[i]=physics.createRigidDynamic(PxTransform(PxIdentity));bodies[i]->setMass(mass);
        bodies[i]->setMassSpaceInertiaTensor(PxVec3(1));bodies[i]->setSleepThreshold(0);
        bodies[i]->setLinearDamping(0);bodies[i]->setAngularDamping(0);
        bodies[i]->setLinearVelocity(PxVec3(velocity,0,0));scene.addActor(*bodies[i]);
        diagonal[i]=double(mass)/(dt*dt)+stiffness*(unsigned(i>0)+unsigned(i+1<N));
        rhs[i]=double(mass)*double(velocity)/dt;initialMomentum+=double(mass)*double(velocity);
        if(i)
        {
            Frame& f=frames[i-1];for(unsigned end=0;end<2;++end)f.data.c2b[end]=PxTransform(PxIdentity);
            for(unsigned d=0;d<6;++d)f.data.drives[d]=PxD6JointDrive(d<3?float(stiffness):0,0,PX_MAX_F32,false);
            PxConstraintShaderTable shaders={JointSoftBodySnippet::finiteJointPrep,NULL,PxConstraintFlag::eCOLLISION_ENABLED};
            shaders.elasticFramePrep=JointSoftBodySnippet::finiteJointMaterialPrep;
            f.constraint=physics.createConstraint(bodies[i-1],bodies[i],f,shaders,sizeof(f.data));
        }
    }
    for(unsigned i=1;i<N;++i){const double ratio=-stiffness/diagonal[i-1];diagonal[i]+=ratio*stiffness;rhs[i]-=ratio*rhs[i-1];}
    for(unsigned j=N;j>0;--j){const unsigned i=j-1;expected[i]=(rhs[i]+(i+1<N?stiffness*expected[i+1]:0))/diagonal[i];}
    scene.simulate(float(dt));bool ok=scene.fetchResults(true);double error=0,bdf=0,momentum=0,angular=0;
    for(unsigned i=0;i<N;++i)
    {
        const PxTransform pose=bodies[i]->getGlobalPose();const PxVec3 velocity=bodies[i]->getLinearVelocity();
        ok=ok && pose.isFinite() && velocity.isFinite();
        error=PxMax(error,std::abs(double(pose.p.x)-expected[i]));
        error=PxMax(error,std::abs(double(pose.p.y))+std::abs(double(pose.p.z)));
        bdf=PxMax(bdf,double((pose.p/float(dt)-velocity).magnitude()));
        momentum+=double(bodies[i]->getMass())*double(velocity.x);angular=PxMax(angular,double(bodies[i]->getAngularVelocity().magnitude()));
    }
    ok=ok && error<1.e-6 && bdf<2.e-5 && std::abs(momentum-initialMomentum)<2.e-4 && angular<1.e-5;
    std::printf("[COMMON_CHAIN] bodies=%u positionError=%.9g bdfError=%.9g momentumError=%.9g angular=%.9g status=%s\n",
        N,error,bdf,std::abs(momentum-initialMomentum),angular,ok?"PASS":"FAIL");
    for(auto& f:frames)f.constraint->release();for(auto b:bodies)b->release();return ok;
}
static bool contactMechanism(PxPhysics& physics,PxScene& scene,PxMaterial& material,unsigned mode)
{
    const float dt=1.f/60;const unsigned count=mode==12?4:mode==16?64:1,steps=mode==12?30:120;
    std::vector<PxRigidDynamic*> bodies;PxRigidActor* floor=NULL;PxRigidDynamic* boundary=NULL;
    if(mode!=12)
    {
        if(mode==15)
        {
            boundary=physics.createRigidDynamic(PxTransform(PxVec3(0,-.5f,0)));
            boundary->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC,true);
            PxRigidActorExt::createExclusiveShape(*boundary,PxBoxGeometry(3,.5f,3),material);floor=boundary;
        }
        else floor=PxCreatePlane(physics,PxPlane(0,1,0,0),material);
        scene.addActor(*floor);
    }
    double mass=0;PxVec3 momentum0(0),center0(0);
    for(unsigned i=0;i<count;++i)
    {
        const PxVec3 position(mode==16?2.f*(i%8):0,mode==12?2+1.015f*i:.505f,mode==16?2.f*(i/8):0);
        auto* b=physics.createRigidDynamic(PxTransform(position));
        if(!PxRigidActorExt::createExclusiveShape(*b,PxBoxGeometry(PxVec3(.5f)),material))return false;
        PxRigidBodyExt::setMassAndUpdateInertia(*b,1+.2f*i);b->setSleepThreshold(0);
        b->setLinearDamping(0);b->setAngularDamping(0);
        if(mode==12){b->setLinearVelocity(PxVec3(.2f,-1,.1f));b->setRigidDynamicLockFlags(
            PxRigidDynamicLockFlag::eLOCK_ANGULAR_X|PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y|PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);}
        if(mode==14 || mode==16)b->setLinearVelocity(PxVec3(1,0,0));
        scene.addActor(*b);bodies.push_back(b);mass+=b->getMass();
        momentum0+=b->getLinearVelocity()*b->getMass();center0+=b->getGlobalPose().p*b->getMass();
    }
    center0/=float(mass);double momentumError=0,comError=0,penetration=0,tilt=0;bool ok=true;unsigned contactPairs=0;
    for(unsigned frame=0;frame<steps;++frame)
    {
        if(mode==13){bodies[0]->addForce(PxVec3(0,-20,0));bodies[0]->addTorque(PxVec3(0,0,4));}
        if(boundary)boundary->setKinematicTarget(PxTransform(PxVec3(0,-.5f+.05f*dt*(frame+1),0)));
        scene.simulate(dt);ok=scene.fetchResults(true) && ok;
        PxSimulationStatistics statistics;scene.getSimulationStatistics(statistics);contactPairs+=statistics.nbDiscreteContactPairsWithContacts;
        PxVec3 momentum(0),center(0);
        for(auto* b:bodies)
        {
            const auto p=b->getGlobalPose();ok=ok && p.isFinite() && b->getLinearVelocity().isFinite();
            momentum+=b->getLinearVelocity()*b->getMass();center+=p.p*b->getMass();
            tilt=PxMax(tilt,double(1-p.q.rotate(PxVec3(0,1,0)).y));
            if(mode!=12)
            {
                const float floorY=boundary?boundary->getGlobalPose().p.y+.5f:0;
                for(unsigned v=0;v<8;++v)
                {const auto q=p.transform(PxVec3(v&1?.5f:-.5f,v&2?.5f:-.5f,v&4?.5f:-.5f));penetration=PxMax(penetration,double(floorY-q.y));}
            }
        }
        if(mode==12)
        {
            const double t=(frame+1)*double(dt);
            const PxVec3 expectedP=momentum0+scene.getGravity()*float(mass*t);
            const PxVec3 expectedX=center0+momentum0/float(mass)*float(t)+scene.getGravity()*float(.5*dt*dt*(frame+1)*(frame+2));
            momentumError=PxMax(momentumError,double((momentum-expectedP).magnitude()/float(mass)));
            comError=PxMax(comError,double((center/float(mass)-expectedX).magnitude()));
            for(unsigned i=0;i+1<count;++i)
            {PxVec3 direction;float depth=0;if(PxGeometryQuery::computePenetration(direction,depth,PxBoxGeometry(PxVec3(.5f)),bodies[i]->getGlobalPose(),PxBoxGeometry(PxVec3(.5f)),bodies[i+1]->getGlobalPose()))penetration=PxMax(penetration,double(depth));}
        }
    }
    const PxVec3 final=bodies[0]->getGlobalPose().p,velocity=bodies[0]->getLinearVelocity();
    ok=ok && penetration<1.e-5 && contactPairs>0;
    if(mode==12)ok=ok && momentumError<.01 && comError<.002 && final.x>.09;
    if(mode==13)ok=ok && tilt<.01 && final.y<.6 && velocity.magnitude()<.1;
    if(mode==14)ok=ok && final.x>.03 && velocity.magnitude()<.1 && final.y<.6;
    if(mode==15)ok=ok && std::abs(final.y-.605f)<.04 && std::abs(velocity.y-.05f)<.1 && tilt<.01;
    if(mode==16)for(unsigned i=0;i<count;++i)
    {
        const auto pose=bodies[i]->getGlobalPose();
        ok=ok && pose.p.x-2.f*(i%8)>.03f && pose.p.y<.6f && bodies[i]->getLinearVelocity().magnitude()<.1f;
    }
    std::printf("[LOCAL_CONTACT_MECHANISM] mode=%u bodies=%u frames=%u contacts=%u momentumError=%.9g comError=%.9g penetration=%.9g tilt=%.9g finalX=%.9g finalY=%.9g speed=%.9g status=%s\n",
        mode,count,steps,contactPairs,momentumError,comError,penetration,tilt,double(final.x),double(final.y),double(velocity.magnitude()),ok?"PASS":"FAIL");
    for(auto* b:bodies)b->release();if(floor)floor->release();return ok;
}
// An independent contact must not change a distant elastic assembly's free
// fall. Both the assembly COM and a plain rigid body have the same analytic
// semi-implicit gravity trajectory, regardless of stiffness or worker count.
static bool gravityFlight(PxPhysics& physics,PxScene& scene,PxMaterial& material,bool contact)
{
    const unsigned N=12,steps=120;const float dt=1.f/60,height=30;
    std::vector<PxRigidDynamic*> bodies;PxArray<Frame> frames(N-1);
    PxRigidStatic* ground=NULL;PxRigidDynamic* resting=NULL;
    if(contact)
    {
        ground=PxCreatePlane(physics,PxPlane(0,1,0,0),material);scene.addActor(*ground);
        resting=PxCreateDynamic(physics,PxTransform(PxVec3(0,.505f,0)),PxBoxGeometry(PxVec3(.5f)),material,1);
        resting->setSleepThreshold(0);resting->setLinearVelocity(PxVec3(0,-2,0));scene.addActor(*resting);
    }
    double totalMass=0;
    for(unsigned i=0;i<N;++i)
    {
        auto* b=PxCreateDynamic(physics,PxTransform(PxVec3(5+1.2f*i,height,0)),PxBoxGeometry(PxVec3(.5f)),material,1+.1f*i);
        b->setLinearDamping(0);b->setAngularDamping(0);b->setSleepThreshold(0);scene.addActor(*b);bodies.push_back(b);totalMass+=b->getMass();
        if(i)
        {
            Frame& f=frames[i-1];f.data.c2b[0]=PxTransform(PxVec3(.6f,0,0));f.data.c2b[1]=PxTransform(PxVec3(-.6f,0,0));
            for(unsigned d=0;d<6;++d)f.data.drives[d]=PxD6JointDrive(d<3?1.e6f:1.e3f,0,PX_MAX_F32,false);
            PxConstraintShaderTable shaders={JointSoftBodySnippet::finiteJointPrep,NULL,PxConstraintFlag::eCOLLISION_ENABLED};
            shaders.elasticFramePrep=JointSoftBodySnippet::finiteJointMaterialPrep;
            f.constraint=physics.createConstraint(bodies[i-1],b, f,shaders,sizeof(f.data));
        }
    }
    auto* rigid=PxCreateDynamic(physics,PxTransform(PxVec3(-5,height,0)),PxBoxGeometry(PxVec3(.5f)),material,1);
    rigid->setLinearDamping(0);rigid->setAngularDamping(0);rigid->setSleepThreshold(0);scene.addActor(*rigid);
    bool ok=true;double positionError=0,rigidError=0,speedError=0,speedDifference=0,shapeError=0;
    double center=height,velocity=0;unsigned contacts=0;
    for(unsigned frame=1;frame<=steps;++frame)
    {
        scene.simulate(dt);ok=scene.fetchResults(true) && ok;
        PxSimulationStatistics statistics;scene.getSimulationStatistics(statistics);contacts+=statistics.nbDiscreteContactPairsWithContacts;
        center=velocity=0;
        for(auto* b:bodies){center+=double(b->getMass())*b->getGlobalPose().p.y;velocity+=double(b->getMass())*b->getLinearVelocity().y;}
        center/=totalMass;velocity/=totalMass;
        const double expected=height+.5*double(scene.getGravity().y)*dt*dt*frame*(frame+1),v=double(scene.getGravity().y)*dt*frame;
        positionError=PxMax(positionError,std::abs(center-expected));speedError=PxMax(speedError,std::abs(velocity-v));
        rigidError=PxMax(rigidError,std::abs(double(rigid->getGlobalPose().p.y)-expected));
        speedDifference=PxMax(speedDifference,std::abs(velocity-rigid->getLinearVelocity().y));
        for(auto* b:bodies)shapeError=PxMax(shapeError,std::abs(double(b->getGlobalPose().p.y)-center));
        ok=ok && std::isfinite(center) && std::isfinite(velocity);
    }
    ok=ok && positionError<.005 && rigidError<.005 && speedError<.01 && speedDifference<.01 && shapeError<.005 && (!contact || contacts>0);
    std::printf("[GRAVITY_FLIGHT] contact=%u frames=%u contacts=%u softPositionError=%.9g rigidPositionError=%.9g softSpeedError=%.9g speedDifference=%.9g shapeError=%.9g finalY=%.9g finalVy=%.9g rigidY=%.9g rigidVy=%.9g status=%s\n",
        unsigned(contact),steps,contacts,positionError,rigidError,speedError,speedDifference,shapeError,center,velocity,double(rigid->getGlobalPose().p.y),double(rigid->getLinearVelocity().y),ok?"PASS":"FAIL");
    for(auto& f:frames)f.constraint->release();for(auto* b:bodies)b->release();rigid->release();
    if(resting)resting->release();if(ground)ground->release();return ok;
}
static bool colorClique(PxPhysics& physics,PxScene& scene)
{
    scene.setGravity(PxVec3(0));std::vector<PxRigidDynamic*> bodies;std::vector<PxJoint*> joints;
    for(unsigned i=0;i<40;++i)
    {
        auto* b=physics.createRigidDynamic(PxTransform(PxVec3(.2f*i,0,0)));b->setMass(1);b->setMassSpaceInertiaTensor(PxVec3(1));
        b->setLinearDamping(0);b->setAngularDamping(0);b->setSleepThreshold(0);b->setLinearVelocity(PxVec3(.2f,0,0));scene.addActor(*b);bodies.push_back(b);
        for(unsigned k=0;k<i;++k)
        {
            auto* j=PxD6JointCreate(physics,bodies[k],PxTransform(PxIdentity),b,PxTransform(bodies[k]->getGlobalPose().p-b->getGlobalPose().p));
            for(unsigned d=0;d<6;++d)j->setMotion(PxD6Axis::Enum(d),PxD6Motion::eFREE);
            j->setDrive(PxD6Drive::eX,PxD6JointDrive(100,0,PX_MAX_F32,false));joints.push_back(j);
        }
    }
    const float dt=1.f/60;scene.simulate(dt);bool ok=scene.fetchResults(true);double error=0,speed=0;
    for(unsigned i=0;i<bodies.size();++i)
    {
        error=PxMax(error,std::abs(double(bodies[i]->getGlobalPose().p.x)-double(.2f*i)-double(.2f)*dt));
        speed=PxMax(speed,std::abs(double(bodies[i]->getLinearVelocity().x)-.2));
    }
    ok &= error<1.e-5 && speed<1.e-4;
    std::printf("[LOCAL_COLOR_CLIQUE] bodies=40 positionError=%.9g speedError=%.9g status=%s\n",error,speed,ok?"PASS":"FAIL");
    for(auto* j:joints)j->release();for(auto* b:bodies)b->release();return ok;
}
static bool highDegree(PxPhysics& physics,PxScene& scene,unsigned rows)
{
    scene.setGravity(PxVec3(0));std::vector<PxRigidDynamic*> bodies;std::vector<PxJoint*> joints;
    double stiffness=0;for(unsigned r=0;r<rows;++r)stiffness+=1+.125*(r%7);
    for(unsigned b=0;b<8;++b)
    {
        auto* body=physics.createRigidDynamic(PxTransform(PxVec3(float(2*b),0,0)));
        body->setMass(1+.25f*b);body->setMassSpaceInertiaTensor(PxVec3(1));body->setSleepThreshold(0);
        body->setLinearDamping(0);body->setAngularDamping(0);body->setLinearVelocity(PxVec3(.2f+.01f*b,0,0));
        scene.addActor(*body);bodies.push_back(body);
        for(unsigned r=0;r<rows;++r)
        {
            auto* j=PxD6JointCreate(physics,NULL,PxTransform(PxVec3(float(2*b+1),0,0)),body,PxTransform(PxIdentity));
            for(unsigned d=0;d<6;++d)j->setMotion(PxD6Axis::Enum(d),PxD6Motion::eFREE);
            j->setDrive(PxD6Drive::eX,PxD6JointDrive(1+.125f*(r%7),0,PX_MAX_F32,false));joints.push_back(j);
        }
    }
    const float dt=1.f/60;scene.simulate(dt);bool ok=scene.fetchResults(true);double error=0,bdf=0;
    for(unsigned b=0;b<bodies.size();++b)
    {
        const double inertial=double(bodies[b]->getMass())/(double(dt)*dt);
        const double expected=(inertial*double(.2f+.01f*b)*dt+stiffness)/(inertial+stiffness);
        const auto pose=bodies[b]->getGlobalPose();const auto velocity=bodies[b]->getLinearVelocity();
        error=PxMax(error,std::abs(double(pose.p.x)-2*b-expected));
        bdf=PxMax(bdf,std::abs(double(velocity.x)-(double(pose.p.x)-2*b)/dt));
        ok &= pose.isFinite() && velocity.isFinite() && std::abs(pose.p.y)+std::abs(pose.p.z)<1.e-6 && bodies[b]->getAngularVelocity().magnitude()<1.e-6;
    }
    ok &= error<1.e-5 && bdf<1.e-4;
    std::printf("[LOCAL_HIGH_DEGREE] bodies=8 rows=%u positionError=%.9g bdfError=%.9g status=%s\n",rows,error,bdf,ok?"PASS":"FAIL");
    for(auto* j:joints)j->release();for(auto* b:bodies)b->release();return ok;
}
int main(int argc,char** argv)
{
    const unsigned mode=argc>1?unsigned(std::atoi(argv[1])):0;
    useModification=argc<=2 || mode==1;
    PxDefaultAllocator allocator;PxDefaultErrorCallback errors;
    PxFoundation* foundation=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors);
    PxPhysics* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation,PxTolerancesScale(),false,NULL);
    const unsigned threads=argc>3?unsigned(std::atoi(argv[3])):4;
    PxInitExtensions(*physics,NULL);PxDefaultCpuDispatcher* dispatcher=PxDefaultCpuDispatcherCreate(threads);
    PxSceneDesc desc(physics->getTolerancesScale());desc.gravity=PxVec3(0,-9.81f,0);desc.solverType=PxSolverType::eAVBD;
    Modify modify;if(mode==1)modify.impulse=.001f;
    desc.cpuDispatcher=dispatcher;desc.filterShader=filter;desc.contactModifyCallback=&modify;
    PxScene* scene=physics->createScene(desc);PxMaterial* material=physics->createMaterial(mode==5?0:.4f,mode==5?0:.4f,0);
    if(mode==21)
    {
        const bool ok=colorClique(*physics,*scene);
        material->release();scene->release();dispatcher->release();PxCloseExtensions();physics->release();foundation->release();return ok?0:1;
    }
    if(mode==19 || mode==20)
    {
        const bool ok=highDegree(*physics,*scene,mode==19?257:1025);
        material->release();scene->release();dispatcher->release();PxCloseExtensions();physics->release();foundation->release();return ok?0:1;
    }
    if(mode==17 || mode==18)
    {
        const bool ok=gravityFlight(*physics,*scene,*material,mode==18);
        material->release();scene->release();dispatcher->release();PxCloseExtensions();physics->release();foundation->release();return ok?0:1;
    }
    if(mode>=12 && mode<=16)
    {
        const bool ok=contactMechanism(*physics,*scene,*material,mode);
        material->release();scene->release();dispatcher->release();PxCloseExtensions();physics->release();foundation->release();return ok?0:1;
    }
    if(mode==6)
    {
        const unsigned count=argc>4?unsigned(std::atoi(argv[4])):12;
        if(count<2 || count>4096)return 2;
        const bool ok=chain(*physics,*scene,count);
        std::printf("[COMMON_PROGRAM_SCENE] mode=6 fetchedFinite=%s\n",ok?"PASS":"FAIL");
        material->release();scene->release();dispatcher->release();PxCloseExtensions();physics->release();foundation->release();return ok?0:1;
    }
    if(mode==7)
    {
        // Incompatible hard anchors have no equilibrium. Even a numerical
        // convergence report must not bypass actual admission or reach BDF.
        scene->setGravity(PxVec3(0));auto* body=physics->createRigidDynamic(PxTransform(PxIdentity));
        body->setMass(1);body->setMassSpaceInertiaTensor(PxVec3(1));body->setSleepThreshold(0);
        body->setLinearDamping(0);body->setAngularDamping(0);body->setLinearVelocity(PxVec3(.2f,0,0));scene->addActor(*body);
        auto* a=PxD6JointCreate(*physics,NULL,PxTransform(PxIdentity),body,PxTransform(PxIdentity));
        auto* b=PxD6JointCreate(*physics,NULL,PxTransform(PxVec3(.1f,0,0)),body,PxTransform(PxIdentity));
        scene->simulate(1.f/60);scene->fetchResults(true);
        const double positionError=body->getGlobalPose().p.magnitude(),velocityError=(body->getLinearVelocity()-PxVec3(.2f,0,0)).magnitude();
        const bool unchanged=positionError==0 && velocityError==0;
        std::printf("[COMMON_CONFLICT] positionError=%.9g velocityError=%.9g unpublished=%s\n",positionError,velocityError,unchanged?"PASS":"FAIL");
        a->release();b->release();body->release();material->release();scene->release();dispatcher->release();
        PxCloseExtensions();physics->release();foundation->release();return unchanged?0:1;
    }
    if(mode==9 || mode==10)
    {
        scene->setGravity(PxVec3(0));auto* body=physics->createRigidDynamic(PxTransform(PxIdentity));
        body->setMass(1);body->setMassSpaceInertiaTensor(PxVec3(1));body->setSleepThreshold(0);
        body->setLinearDamping(0);body->setAngularDamping(0);scene->addActor(*body);
        std::vector<PxJoint*> joints;
        if(mode==9)
        {
            body->setLinearVelocity(PxVec3(.2f,0,0));
            for(unsigned i=0;i<87;++i)joints.push_back(PxSphericalJointCreate(*physics,NULL,
                PxTransform(PxIdentity),body,PxTransform(PxIdentity)));
        }
        else
        {
            auto* joint=PxD6JointCreate(*physics,NULL,PxTransform(PxVec3(1,0,0)),body,PxTransform(PxIdentity));
            for(unsigned i=0;i<6;++i)joint->setMotion(PxD6Axis::Enum(i),PxD6Motion::eFREE);
            PxD6JointDrive drive(10000,0,2,false);drive.flags|=PxD6JointDriveFlag::eOUTPUT_FORCE;
            joint->setDrive(PxD6Drive::eX,drive);joint->setConstraintFlag(PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES,true);
            joints.push_back(joint);
        }
        const float dt=1.f/60;scene->simulate(dt);bool ok=scene->fetchResults(true);
        if(mode==9)
        {
            const bool unpublished=body->getGlobalPose().p==PxVec3(0) && body->getLinearVelocity()==PxVec3(.2f,0,0);
            const double error=body->getGlobalPose().p.magnitude();
            const double bdf=(body->getLinearVelocity()-body->getGlobalPose().p/dt).magnitude();
            const bool physical=error<1.e-4 && bdf<1.e-5 && body->getLinearVelocity().magnitude()<.02;
            ok=ok && (unpublished || physical);
            std::printf("[LOCAL_CAPACITY] unpublished=%s\n",unpublished?"PASS":"FAIL");
            std::printf("[LOCAL_HIGH_DEGREE] bodies=1 rows=261 positionError=%.9g bdfError=%.9g status=%s\n",error,bdf,physical?"PASS":"FAIL");
        }
        else
        {
            PxVec3 force,torque;joints[0]->getConstraint()->getForce(force,torque);
            const double expected=2*double(dt)*dt,error=std::abs(body->getGlobalPose().p.x-expected);
            ok=ok && error<1.e-7 && std::abs(force.magnitude()-2)<1.e-5;
            std::printf("[LOCAL_FORCE_LIMIT] positionError=%.9g force=%.9g status=%s\n",error,double(force.magnitude()),ok?"PASS":"FAIL");
        }
        for(auto* joint:joints)joint->release();body->release();material->release();scene->release();dispatcher->release();
        PxCloseExtensions();physics->release();foundation->release();return ok?0:1;
    }
    if(mode==8 || mode==11)
    {
        // Two independent dynamic components share only a prescribed support.
        // Check their separate implicit solutions and one BDF from actual poses.
        scene->setGravity(PxVec3(0));auto* fixed=physics->createRigidDynamic(PxTransform(PxIdentity));
        fixed->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC,true);scene->addActor(*fixed);
        const float boundary=mode==11?.01f:0;
        if(mode==11)fixed->setKinematicTarget(PxTransform(PxVec3(boundary,0,0)));
        PxRigidDynamic* moving[2];PxD6Joint* joints[2];double expected[2];const float dt=1.f/60;
        for(unsigned i=0;i<2;++i)
        {
            const float mass=1.f+i,stiffness=i?10000.f:1000.f,velocity=i?-.3f:.2f;
            moving[i]=physics->createRigidDynamic(PxTransform(PxIdentity));moving[i]->setMass(mass);
            moving[i]->setMassSpaceInertiaTensor(PxVec3(1));moving[i]->setSleepThreshold(0);
            moving[i]->setLinearDamping(0);moving[i]->setAngularDamping(0);moving[i]->setLinearVelocity(PxVec3(velocity,0,0));scene->addActor(*moving[i]);
            joints[i]=PxD6JointCreate(*physics,fixed,PxTransform(PxIdentity),moving[i],PxTransform(PxIdentity));
            joints[i]->setMotion(PxD6Axis::eX,PxD6Motion::eFREE);
            joints[i]->setDrive(PxD6Drive::eX,PxD6JointDrive(stiffness,0,PX_MAX_F32,false));
            expected[i]=(double(mass)*velocity*dt+double(stiffness)*dt*dt*boundary)/(mass+double(stiffness)*dt*dt);
        }
        scene->simulate(dt);bool ok=scene->fetchResults(true);double error=0,bdf=0;
        for(unsigned i=0;i<2;++i)
        {const auto pose=moving[i]->getGlobalPose();error=PxMax(error,std::abs(double(pose.p.x)-expected[i]));
         error=PxMax(error,double(std::abs(pose.p.y)+std::abs(pose.p.z)));bdf=PxMax(bdf,double((pose.p/dt-moving[i]->getLinearVelocity()).magnitude()));}
        ok=ok && error<1.e-6 && bdf<2.e-5 && fixed->getGlobalPose().p==PxVec3(boundary,0,0);
        std::printf("[COMMON_INDEPENDENT] positionError=%.9g bdfError=%.9g status=%s\n",error,bdf,ok?"PASS":"FAIL");
        std::printf("[COMMON_PROGRAM_SCENE] mode=%u fetchedFinite=%s\n",mode,ok?"PASS":"FAIL");
        for(auto* joint:joints)joint->release();for(auto* b:moving)b->release();fixed->release();
        material->release();scene->release();dispatcher->release();PxCloseExtensions();physics->release();foundation->release();return ok?0:1;
    }
    const unsigned count=mode==4?4:3;
    const float half=argc>2 && std::strcmp(argv[2],"overlap")!=0?.501f:.49f;
    const PxVec3 positions[4]={PxVec3(-half,0,0),PxVec3(half,0,0),PxVec3(-half,1,0),PxVec3(-half,0,1)};
    PxRigidDynamic* bodies[4]={};
    for(unsigned i=0;i<count;++i)
    {
        bodies[i]=physics->createRigidDynamic(PxTransform(positions[i]));bodies[i]->setMass(float(i+1));
        bodies[i]->setMassSpaceInertiaTensor(PxVec3(.1f,.13f,.17f)*float(i+1));
        bodies[i]->setSleepThreshold(0);bodies[i]->setLinearDamping(0);bodies[i]->setAngularDamping(0);
        if(i<2){PxShape* shape=PxRigidActorExt::createExclusiveShape(*bodies[i],PxSphereGeometry(.5f),*material);shape->setContactOffset(.02f);}
        bodies[i]->setLinearVelocity(PxVec3(i==0?.1f:i==1?-.2f:0,.1f*float(i),0));scene->addActor(*bodies[i]);
    }
    if(mode==2)bodies[0]->setRigidDynamicLockFlags(PxRigidDynamicLockFlag::eLOCK_LINEAR_Z|PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
    if(mode==3){bodies[1]->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC,true);bodies[1]->setKinematicTarget(PxTransform(PxVec3(.48f,0,0)));}
    Frame frame;frame.data.c2b[0]=PxTransform(PxVec3(0,.5f,0));frame.data.c2b[1]=PxTransform(PxVec3(0,-.5f,0));
    for(PxU32 i=0;i<6;++i)frame.data.drives[i]=PxD6JointDrive(300.f+30.f*i,2,PX_MAX_F32,false);
    PxConstraintShaderTable shaders={JointSoftBodySnippet::finiteJointPrep,NULL,PxConstraintFlag::eCOLLISION_ENABLED};
    shaders.elasticFramePrep=JointSoftBodySnippet::finiteJointMaterialPrep;
    frame.constraint=physics->createConstraint(bodies[0],bodies[2],frame,shaders,sizeof(frame.data));
    PxD6Joint* drive=PxD6JointCreate(*physics,NULL,PxTransform(positions[2]+PxVec3(.1f,0,0)),bodies[2],PxTransform(PxIdentity));
    for(PxU32 i=0;i<6;++i)drive->setMotion(PxD6Axis::Enum(i),PxD6Motion::eFREE);
    drive->setConstraintFlag(PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES,true);
    drive->setDrive(PxD6Drive::eX,PxD6JointDrive(200,3,.8f,false));
    JointSoftBodySnippet::VolumeMaterial* volume=NULL;
    if(mode==4){PxRigidActor* actors[4]={bodies[0],bodies[1],bodies[2],bodies[3]};volume=JointSoftBodySnippet::VolumeMaterial::create(*physics,actors,1.2f,500);}
    scene->simulate(1.f/60);bool fetched=scene->fetchResults(true);
    float bdfError=0;
    for(unsigned i=0;i<count;++i)
    {
        fetched=fetched && bodies[i]->getGlobalPose().isFinite();
        if(!bodies[i]->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC))
            bdfError=PxMax(bdfError,((bodies[i]->getGlobalPose().p-positions[i])*60-bodies[i]->getLinearVelocity()).magnitude());
    }
    if(argc>2)
    {
        fetched=fetched && bdfError<2.e-5f;
        std::printf("[COMMON_BDF] linearError=%.9g\n",double(bdfError));
        PxVec3 direction,driveForce,driveTorque;PxReal depth=0;
        const bool overlap=PxGeometryQuery::computePenetration(direction,depth,PxSphereGeometry(.5f),bodies[0]->getGlobalPose(),
            PxSphereGeometry(.5f),bodies[1]->getGlobalPose());
        drive->getConstraint()->getForce(driveForce,driveTorque);
        fetched=fetched && (mode==1 || !overlap || depth<5.e-7f) && driveForce.magnitude()<.80001f;
        std::printf("[COMMON_PHYSICAL] overlap=%u depth=%.9g driveForce=%.9g\n",unsigned(overlap),double(depth),double(driveForce.magnitude()));
    }
    std::printf("[COMMON_PROGRAM_SCENE] mode=%u fetchedFinite=%s\n",mode,fetched?"PASS":"FAIL");
    if(volume)volume->release();drive->release();frame.constraint->release();
    for(unsigned i=0;i<count;++i)bodies[i]->release();material->release();scene->release();
    dispatcher->release();PxCloseExtensions();physics->release();foundation->release();return fetched?0:1;
}
