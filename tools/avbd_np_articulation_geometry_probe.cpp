// Public API regression: a link and an ordinary body use the same COM-local
// geometry contract, including a translated and rotated mass frame.
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
    PxMaterial* material=physics->createMaterial(0,0,0);
    unsigned failures=0;PxTransform ordinaryResult(PxIdentity);
    for(unsigned test=0;test<6;++test)
    {
        const bool link=(test&1)!=0;const unsigned frame=test/2;
        PxSceneDesc desc(physics->getTolerancesScale());desc.gravity=PxVec3(0);
        desc.cpuDispatcher=dispatcher;desc.filterShader=PxDefaultSimulationFilterShader;
        desc.solverType=PxSolverType::eAVBD;PxScene* scene=physics->createScene(desc);
        PxRigidStatic* plane=PxCreatePlane(*physics,PxPlane(0,1,0,0),*material);scene->addActor(*plane);
        const PxTransform pose(PxVec3(0,.26f,0));
        PxArticulationReducedCoordinate* articulation=link?physics->createArticulationReducedCoordinate():NULL;
        PxRigidBody* body=link?static_cast<PxRigidBody*>(articulation->createLink(NULL,pose)):
            static_cast<PxRigidBody*>(physics->createRigidDynamic(pose));
        PxShape* shape=physics->createShape(PxBoxGeometry(.25f,.25f,.25f),*material);
        shape->setContactOffset(.02f);shape->setRestOffset(0);body->attachShape(*shape);
        const PxTransform massFrame=frame==0?PxTransform(PxIdentity):
            PxTransform(PxVec3(.1f,-.75f,.03f),frame==2?PxQuat(.4f,PxVec3(0,0,1)):PxQuat(PxIdentity));
        body->setCMassLocalPose(massFrame);body->setMass(1);body->setMassSpaceInertiaTensor(PxVec3(.1f));
        body->setLinearDamping(0);body->setAngularDamping(0);
        if(link)
        {
            articulation->setSleepThreshold(0);scene->addArticulation(*articulation);
            articulation->setRootLinearVelocity(PxVec3(0,-2,0));
        }
        else
        {
            PxRigidDynamic* dynamic=static_cast<PxRigidDynamic*>(body);
            dynamic->setSleepThreshold(0);dynamic->setLinearVelocity(PxVec3(0,-2,0));scene->addActor(*dynamic);
        }
        const unsigned before=errors.count;
        scene->simulate(1.f/60.f);const bool fetched=scene->fetchResults(true);
        const PxTransform result=PxShapeExt::getGlobalPose(*shape,*body);float minimum=PX_MAX_F32;
        for(unsigned v=0;v<8;++v)
            minimum=PxMin(minimum,result.transform(PxVec3((v&1)?.25f:-.25f,(v&2)?.25f:-.25f,(v&4)?.25f:-.25f)).y);
        if(!link) ordinaryResult=result;
        // A free one-link articulation has the same physical inputs and one
        // Rigid6 block as this ordinary body. A mass-frame change must not
        // silently remove its geometry certificate or switch its contact law.
        const float poseDifference=(result.p-ordinaryResult.p).magnitude();
        const float rotationDifference=1.f-PxAbs(result.q.dot(ordinaryResult.q));
        const bool pass=fetched && errors.count==before && result.isFinite() && minimum>=-5.e-6f &&
            poseDifference<1.e-5f && rotationDifference<1.e-6f;
        std::printf("[NP_ARTIC_GEOMETRY] link=%u frame=%u minimumY=%.9g poseDifference=%.9g errors=%u status=%s\n",
            unsigned(link),frame,double(minimum),double(poseDifference),errors.count-before,pass?"PASS":"FAIL");
        failures+=!pass;shape->release();
        if(link) articulation->release();else body->release();
        plane->release();scene->release();
    }
    material->release();dispatcher->release();PxCloseExtensions();physics->release();foundation->release();
    return failures||errors.count?1:0;
}
