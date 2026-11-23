// NP feasibility restoration must preserve public rigid translation locks.
#include "PxPhysicsAPI.h"
#include <cstdio>
using namespace physx;
struct Errors : PxErrorCallback
{
    unsigned count=0;
    void reportError(PxErrorCode::Enum code,const char* text,const char*,int) PX_OVERRIDE
    {
        if(code!=PxErrorCode::eDEBUG_INFO && code!=PxErrorCode::eDEBUG_WARNING)
        { ++count;std::printf("ERROR %s\n",text); }
    }
};
int main()
{
    PxDefaultAllocator allocator;Errors errors;
    PxFoundation* foundation=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors);
    PxPhysics* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation,PxTolerancesScale(),false,NULL);
    PxInitExtensions(*physics,NULL);
    PxDefaultCpuDispatcher* dispatcher=PxDefaultCpuDispatcherCreate(2);
    PxMaterial* material=physics->createMaterial(0,0,0);unsigned failures=0;
    for(unsigned axis=0;axis<3;axis+=2)
    {
        PxSceneDesc desc(physics->getTolerancesScale());desc.gravity=PxVec3(0);
        desc.cpuDispatcher=dispatcher;desc.filterShader=PxDefaultSimulationFilterShader;
        desc.solverType=PxSolverType::eAVBD;PxScene* scene=physics->createScene(desc);
        PxVec3 n(0,1,0);n[axis]=1;n.normalize();
        const PxQuat rotation(axis==0?-PxPi/4:PxPi/4,axis==0?PxVec3(0,0,1):PxVec3(1,0,0));
        PxRigidDynamic* boundary=PxCreateDynamic(*physics,PxTransform(PxVec3(0),rotation),PxBoxGeometry(.25f,.25f,.25f),*material,1);
        PxRigidDynamic* body=PxCreateDynamic(*physics,PxTransform(n*.505f),PxSphereGeometry(.25f),*material,1);
        boundary->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC,true);
        body->setRigidDynamicLockFlag(PxRigidDynamicLockFlag::Enum(1u<<axis),true);
        body->setLinearDamping(0);body->setAngularDamping(0);body->setSleepThreshold(0);
        scene->addActor(*boundary);scene->addActor(*body);
        const PxTransform target(n*.05f,rotation),before=body->getGlobalPose();
        boundary->setKinematicTarget(target);const unsigned oldErrors=errors.count;
        scene->simulate(1.f/60.f);const bool fetched=scene->fetchResults(true);
        PxShape* shapes[2];boundary->getShapes(&shapes[0],1);body->getShapes(&shapes[1],1);
        PxVec3 direction;PxReal depth=0;
        const bool overlap=PxGeometryQuery::computePenetration(direction,depth,shapes[0]->getGeometry(),
            PxShapeExt::getGlobalPose(*shapes[0],*boundary),shapes[1]->getGeometry(),PxShapeExt::getGlobalPose(*shapes[1],*body));
        const PxVec3 delta=body->getGlobalPose().p-before.p;
        const bool pass=fetched && errors.count==oldErrors && PxAbs(delta[axis])<1.e-6f && delta.y>.04f &&
            (!overlap || depth<5.e-6f) && (boundary->getGlobalPose().p-target.p).magnitude()<1.e-6f;
        std::printf("[NP_LOCKED_MOTION] axis=%u lockedDelta=%.9g freeDelta=%.9g depth=%.9g errors=%u status=%s\n",
            axis,double(delta[axis]),double(delta.y),double(depth),errors.count-oldErrors,pass?"PASS":"FAIL");
        failures+=!pass;body->release();boundary->release();scene->release();
    }
    material->release();dispatcher->release();PxCloseExtensions();physics->release();foundation->release();
    return failures||errors.count?1:0;
}
