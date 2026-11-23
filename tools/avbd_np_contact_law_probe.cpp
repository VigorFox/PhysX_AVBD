// Public-API normal force-limit and action/reaction checks for NP contacts.
#include "PxPhysicsAPI.h"
#include <cstdio>
#include <cmath>
using namespace physx;
struct Errors:PxErrorCallback
{
    unsigned count=0;
    void reportError(PxErrorCode::Enum code,const char* message,const char*,int) PX_OVERRIDE
    {
        if(code!=PxErrorCode::eDEBUG_INFO && code!=PxErrorCode::eDEBUG_WARNING)
        { ++count;std::printf("ERROR %s\n",message); }
    }
};
struct Modify:PxContactModifyCallback
{
    float limit=0;unsigned points=0;
    void onContactModify(PxContactModifyPair* const pairs,PxU32 count) PX_OVERRIDE
    {
        for(unsigned p=0;p<count;++p) for(unsigned i=0;i<pairs[p].contacts.size();++i)
        { pairs[p].contacts.setMaxImpulse(i,limit);++points; }
    }
};
static PxFilterFlags filter(PxFilterObjectAttributes,PxFilterData,PxFilterObjectAttributes,
    PxFilterData,PxPairFlags& flags,const void*,PxU32)
{ flags=PxPairFlag::eCONTACT_DEFAULT|PxPairFlag::eMODIFY_CONTACTS;return PxFilterFlag::eDEFAULT; }
int main()
{
    PxDefaultAllocator allocator;Errors errors;
    PxFoundation* foundation=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors);
    PxPhysics* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation,PxTolerancesScale(),false,NULL);
    PxInitExtensions(*physics,NULL);
    PxDefaultCpuDispatcher* dispatcher=PxDefaultCpuDispatcherCreate(1);
    PxMaterial* material=physics->createMaterial(0,0,0);
    unsigned failures=0;
    for(unsigned scenario=0;scenario<8;++scenario)
    {
        const unsigned test=scenario%4;const bool bodyLimit=scenario>=4;
        PxSceneDesc desc(physics->getTolerancesScale());desc.gravity=PxVec3(0);
        const float limit=test==0?0.0f:test==3?PX_MAX_F32:0.001f;
        Modify modify;modify.limit=bodyLimit?PX_MAX_F32:limit;
        desc.cpuDispatcher=dispatcher;desc.filterShader=filter;desc.contactModifyCallback=&modify;
        desc.solverType=PxSolverType::eAVBD;
        PxScene* scene=physics->createScene(desc);
        PxRigidDynamic* bodies[3]={};PxD6Joint* joint=NULL;
        const unsigned count=test>=2?3:2;
        for(unsigned i=0;i<count;++i)
        {
            const PxVec3 p=i==2?PxVec3(-.49f,2,0):PxVec3(i?.49f:-.49f,0,0);
            bodies[i]=physics->createRigidDynamic(PxTransform(p));
            bodies[i]->setMass(test>=2 && i==1?1024.f:1.f);
            bodies[i]->setMassSpaceInertiaTensor(PxVec3(.1f)*bodies[i]->getMass());
            bodies[i]->setLinearDamping(0);bodies[i]->setAngularDamping(0);
            bodies[i]->setSleepThreshold(0);bodies[i]->setMaxContactImpulse(bodyLimit?limit:PX_MAX_F32);
            if(i<2)
            {
                PxShape* shape=physics->createShape(PxSphereGeometry(.5f),*material);
                shape->setContactOffset(.02f);shape->setRestOffset(0);
                bodies[i]->attachShape(*shape);shape->release();
            }
            scene->addActor(*bodies[i]);
        }
        if(test>=2)
        {
            // A force-free public connector still joins the island. Its
            // presence cannot change the shared contact law or impulse cap.
            joint=PxD6JointCreate(*physics,bodies[0],PxTransform(PxIdentity),bodies[2],PxTransform(PxIdentity));
            for(unsigned axis=0;axis<6;++axis) joint->setMotion(PxD6Axis::Enum(axis),PxD6Motion::eFREE);
            if(test==3)
            {
                // A real, weak elastic coordinate selects pose-derived
                // velocities. The initially relaxed spring is internal.
                joint->setLocalPose(PxJointActorIndex::eACTOR0,PxTransform(PxVec3(0,1,0)));
                joint->setLocalPose(PxJointActorIndex::eACTOR1,PxTransform(PxVec3(0,-1,0)));
                joint->setDrive(PxD6Drive::eX,PxD6JointDrive(.001f,0,PX_MAX_F32,false));
            }
        }
        PxVec3 before[3];for(unsigned i=0;i<count;++i) before[i]=bodies[i]->getGlobalPose().p;
        scene->simulate(1.f/60.f);const bool fetched=scene->fetchResults(true);
        double maximumImpulse=0,maximumPoseImpulse=0,maximumStorage=0;PxVec3 momentum(0);bool bounded=true;
        for(unsigned i=0;i<count;++i)
        {
            const PxVec3 p=bodies[i]->getLinearVelocity()*bodies[i]->getMass();
            maximumImpulse=PxMax(maximumImpulse,double(p.magnitude()));momentum+=p;
            const double poseImpulse=double((bodies[i]->getGlobalPose().p-before[i]).magnitude())*bodies[i]->getMass()*60;
            maximumPoseImpulse=PxMax(maximumPoseImpulse,poseImpulse);
            double spacing2=0;
            for(unsigned axis=0;axis<3;++axis)
            {
                const float a=before[i][axis],b=bodies[i]->getGlobalPose().p[axis];
                const double sa=double(std::nextafter(a,PX_MAX_F32))-a;
                const double sb=double(std::nextafter(b,PX_MAX_F32))-b;
                const double spacing=PxMax(std::abs(sa),std::abs(sb));spacing2+=spacing*spacing;
            }
            // Two actual coordinate ULPs cover proposal and publication;
            // this is a storage interval, not a percentage of the impulse cap.
            const double storage=2.*std::sqrt(spacing2)*bodies[i]->getMass()*60;
            maximumStorage=PxMax(maximumStorage,storage);
            bounded &= poseImpulse<=limit+storage;
        }
        const bool pass=fetched && modify.points>0 && bounded && maximumImpulse<=limit+2.e-5 &&
            momentum.magnitude()<(test==3?.01f:2.e-5f) &&
            (limit==0 || maximumPoseImpulse>(test==3?1.e-5:limit*.01));
        std::printf("[NP_CONTACT_LAW] case=%u source=%s maxImpulse=%.12g poseImpulse=%.12g limit=%.12g poseStorageAllowance=%.12g momentum=%.12g status=%s\n",
            test,bodyLimit?"body":"contact",maximumImpulse,maximumPoseImpulse,double(limit),maximumStorage,double(momentum.magnitude()),pass?"PASS":"FAIL");
        failures+=!pass;
        if(joint) joint->release();
        for(unsigned i=0;i<count;++i) bodies[i]->release();
        scene->release();
    }
    material->release();dispatcher->release();PxCloseExtensions();physics->release();foundation->release();
    return failures+errors.count?1:0;
}
