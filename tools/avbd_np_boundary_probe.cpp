// Public API counterexamples for an NP domain with a prescribed boundary.
#include "PxPhysicsAPI.h"
#include <cstdio>
using namespace physx;
struct Errors:PxErrorCallback
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
    PxMaterial* material=physics->createMaterial(0,0,0);
    unsigned failed=0;
    for(unsigned test=0;test<3;++test)
    {
        PxSceneDesc desc(physics->getTolerancesScale());desc.gravity=PxVec3(0);
        desc.cpuDispatcher=dispatcher;desc.filterShader=PxDefaultSimulationFilterShader;
        desc.solverType=PxSolverType::eAVBD;
        PxScene* scene=physics->createScene(desc);
        PxRigidDynamic* body[2];PxShape* shape[2];
        for(unsigned i=0;i<2;++i)
        {
            body[i]=physics->createRigidDynamic(PxTransform(PxVec3(0,i?.255f:-.25f,0)));
            shape[i]=physics->createShape(PxBoxGeometry(.25f,.25f,.25f),*material);
            shape[i]->setContactOffset(.04f);shape[i]->setRestOffset(0);
            body[i]->attachShape(*shape[i]);PxRigidBodyExt::updateMassAndInertia(*body[i],1.f);
            body[i]->setLinearDamping(0);body[i]->setAngularDamping(0);body[i]->setSleepThreshold(0);
            scene->addActor(*body[i]);
        }
        body[0]->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC,true);
        const PxTransform target(PxVec3(0,test==2?-.30f:-.20f,0),
            test==1?PxQuat(.08f,PxVec3(0,0,1)):PxQuat(PxIdentity));
        body[0]->setKinematicTarget(target);
        const unsigned before=errors.count;
        scene->simulate(1.f/60.f);const bool fetched=scene->fetchResults(true);
        PxVec3 normal;PxReal depth=0;
        const bool overlap=PxGeometryQuery::computePenetration(normal,depth,shape[0]->getGeometry(),
            body[0]->getGlobalPose(),shape[1]->getGeometry(),body[1]->getGlobalPose());
        const float moved=body[1]->getGlobalPose().p.y-.255f;
        const bool targetReached=(body[0]->getGlobalPose().p-target.p).magnitude()<1.e-6f &&
            PxAbs(body[0]->getGlobalPose().q.dot(target.q))>1.f-1.e-6f;
        const bool pass=fetched && errors.count==before && targetReached &&
            (!overlap || depth<=5.e-6f) && (test==2?PxAbs(moved)<1.e-5f:moved>.04f);
        std::printf("[NP_BOUNDARY] case=%u targetReached=%u overlap=%u depth=%.9g moved=%.9g errors=%u status=%s\n",
            test,unsigned(targetReached),unsigned(overlap),double(depth),double(moved),errors.count-before,pass?"PASS":"FAIL");
        failed+=!pass;
        for(unsigned i=0;i<2;++i) { shape[i]->release();body[i]->release(); }
        scene->release();
    }
    material->release();dispatcher->release();PxCloseExtensions();physics->release();foundation->release();
    return failed||errors.count?1:0;
}
