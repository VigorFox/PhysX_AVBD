// Constrain one falling Rigid6 block with enough host solverPrep rows for three leaves.
#include "PxPhysicsAPI.h"
#include "avbd_probe_foundation.h"
#include <cstdio>
#include <cstdlib>
using namespace physx;
int main(int argc,char** argv)
{
    AvbdProbeFoundation foundation;
    PxPhysics* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation.foundation,PxTolerancesScale());
    auto* pool=PxDefaultCpuDispatcherCreate(argc>1?PxU32(std::atoi(argv[1])):4);
    PxSceneDesc desc(physics->getTolerancesScale());
    desc.solverType=PxSolverType::eAVBD;
    desc.gravity=PxVec3(0,-9.81f,0);
    desc.cpuDispatcher=pool;desc.filterShader=PxDefaultSimulationFilterShader;
    PxScene* scene=physics->createScene(desc);
    PxMaterial* material=physics->createMaterial(.4f,.4f,0);
    auto* body=PxCreateDynamic(*physics,PxTransform(PxVec3(0,1,0)),PxBoxGeometry(.2f,.2f,.2f),*material,1.f);
    scene->addActor(*body);
    body->setSleepThreshold(0);
    const PxVec3 anchor=body->getGlobalPose().p;
    PxFixedJoint* joints[96];
    for(auto& joint:joints)
        joint=PxFixedJointCreate(*physics,nullptr,PxTransform(anchor),body,PxTransform(PxIdentity));
    body->setLinearVelocity(PxVec3(.5f,0,0));
    PxReal maxDisplacement=0;
    bool valid=true;
    for(PxU32 i=0;i<24;++i)
    {
        if(!scene->simulate(1.f/60) || !scene->fetchResults(true)){valid=false;break;}
        const auto p=body->getGlobalPose();
        valid&=p.isSane() && body->getLinearVelocity().isFinite();
        maxDisplacement=PxMax(maxDisplacement,(p.p-anchor).magnitude());
    }
    valid &= maxDisplacement<.02f;
    std::printf("[WIDE_RIGID] excursion=%g status=%s\n",double(maxDisplacement),valid?"PASS":"FAIL");
    for(auto* joint:joints)joint->release();
    body->release();material->release();scene->release();pool->release();physics->release();
    return valid?0:1;
}

