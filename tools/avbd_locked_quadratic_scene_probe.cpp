// Public API oracle: a diagonal spring with one translation locked has an
// exact scalar backward-Euler response. The locked reaction is NOT zero.
#include "PxPhysicsAPI.h"
#include <cstdio>
#include <cmath>
using namespace physx;
struct Errors : PxErrorCallback
{
    unsigned count=0;
    void reportError(PxErrorCode::Enum code,const char* text,const char*,int) PX_OVERRIDE
    {if(code!=PxErrorCode::eDEBUG_INFO){++count;std::printf("ERROR %s\n",text);}}
};
int main()
{
    PxDefaultAllocator allocator;Errors errors;
    PxFoundation* foundation=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors);
    PxPhysics* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation,PxTolerancesScale(),false,NULL);
    PxInitExtensions(*physics,NULL);PxDefaultCpuDispatcher* dispatcher=PxDefaultCpuDispatcherCreate(2);
    PxMaterial* material=physics->createMaterial(0,0,0);unsigned failures=0;
    for(unsigned axis=0;axis<2;++axis)for(unsigned stiffnessIndex=0;stiffnessIndex<2;++stiffnessIndex)
    {
        PxSceneDesc desc(physics->getTolerancesScale());desc.gravity=PxVec3(0);desc.gravity[axis]=9.81f;
        desc.cpuDispatcher=dispatcher;desc.filterShader=PxDefaultSimulationFilterShader;desc.solverType=PxSolverType::eAVBD;
        PxScene* scene=physics->createScene(desc);
        PxRigidDynamic* body=PxCreateDynamic(*physics,PxTransform(PxIdentity),PxSphereGeometry(.1f),*material,1);
        PxRigidBodyExt::setMassAndUpdateInertia(*body,1);body->setLinearDamping(0);body->setAngularDamping(0);
        body->setSleepThreshold(0);body->setStabilizationThreshold(0);
        body->setRigidDynamicLockFlags(PxRigidDynamicLockFlags(PxU8(63u^(1u<<axis))));scene->addActor(*body);
        const PxQuat rotation(.81f,PxVec3(0,0,1));const PxTransform anchor(PxVec3(0),rotation);
        PxD6Joint* joint=PxD6JointCreate(*physics,NULL,anchor,body,anchor);
        for(unsigned a=0;a<6;++a)joint->setMotion(PxD6Axis::Enum(a),PxD6Motion::eFREE);
        const PxReal stiffness=stiffnessIndex?1.e9f:1.e6f;
        joint->setDrive(PxD6Drive::eX,PxD6JointDrive(stiffness,0,PX_MAX_F32,false));
        const PxReal dt=1.f/60;const double projection=rotation.rotate(PxVec3(1,0,0))[axis];
        const double expected=double(desc.gravity[axis])/(1./(double(dt)*dt)+double(stiffness)*projection*projection);
        const unsigned before=errors.count;scene->simulate(dt);const bool fetched=scene->fetchResults(true);
        const PxVec3 position=body->getGlobalPose().p;
        const double relative=std::abs(double(position[axis])-expected)/expected;
        bool locked=true;for(unsigned a=0;a<3;++a)if(a!=axis)locked&=position[a]==0;
        const bool pass=fetched && errors.count==before && locked && std::isfinite(relative) && relative<5.e-4;
        std::printf("[LOCKED_QUADRATIC] axis=%u stiffness=%.9g expected=%.12g actual=%.12g relativeError=%.9g status=%s\n",
            axis,double(stiffness),expected,double(position[axis]),relative,pass?"PASS":"FAIL");
        failures+=!pass;joint->release();body->release();scene->release();
    }
    material->release();dispatcher->release();PxCloseExtensions();physics->release();foundation->release();
    return failures||errors.count?1:0;
}
