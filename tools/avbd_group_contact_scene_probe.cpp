// A real NP pair with BOTH endpoints in one material group. This exercises
// factor packing, unequal endpoint conditioning, shared force and publication.
#include "PxPhysicsAPI.h"
#include "../physx/snippets/snippetjointsoftbody/SnippetFiniteJoint.h"
#include <cstdio>
using namespace physx;
struct GroupContactErrors : PxErrorCallback
{
    unsigned failures=0;
    void reportError(PxErrorCode::Enum code,const char* text,const char*,int) PX_OVERRIDE
    {if(code!=PxErrorCode::eDEBUG_INFO){++failures;std::printf("ERROR %s\n",text);}}
};
struct GroupContactJoint : PxConstraintConnector
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
static double mechanicalEnergy(const GroupContactJoint& joint,PxRigidDynamic* const* bodies)
{
    PxTransform poses[2];double energy=0;
    for(unsigned i=0;i<2;++i)
    {
        poses[i]=bodies[i]->getGlobalPose()*bodies[i]->getCMassLocalPose();
        const PxVec3 v=bodies[i]->getLinearVelocity(),w=poses[i].q.rotateInv(bodies[i]->getAngularVelocity());
        energy+=.5*(double(bodies[i]->getMass())*v.magnitudeSquared()+w.dot(bodies[i]->getMassSpaceInertiaTensor().multiply(w)));
    }
    const auto coordinates=JointSoftBodySnippet::finiteJointCoordinates(
        poses[0]*joint.data.c2b[0],poses[1]*joint.data.c2b[1],poses[0].p,poses[1].p);
    for(unsigned i=0;i<18;++i)
        energy+=.5*double(joint.data.drives[JointSoftBodySnippet::finiteJointDriveIndex(i)].stiffness)*double(coordinates.value[i])*double(coordinates.value[i]);
    return energy;
}
int main()
{
    PxDefaultAllocator allocator;GroupContactErrors errors;
    PxFoundation* foundation=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors);
    PxPhysics* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation,PxTolerancesScale(),false,NULL);
    PxInitExtensions(*physics,NULL);PxDefaultCpuDispatcher* dispatcher=PxDefaultCpuDispatcherCreate(4);
    unsigned failures=0;
    for(unsigned friction=0;friction<2;++friction)
    {
        PxSceneDesc desc(physics->getTolerancesScale());desc.gravity=PxVec3(0);
        desc.cpuDispatcher=dispatcher;desc.filterShader=PxDefaultSimulationFilterShader;desc.solverType=PxSolverType::eAVBD;
        PxScene* scene=physics->createScene(desc);
        PxMaterial* material=physics->createMaterial(friction?.6f:0,friction?.6f:0,0);
        PxRigidDynamic* bodies[2];GroupContactJoint joint;
        joint.data.c2b[0]=PxTransform(PxVec3(.55f,0,0));joint.data.c2b[1]=PxTransform(PxVec3(-.55f,0,0));
        for(unsigned i=0;i<6;++i)joint.data.drives[i]=PxD6JointDrive(100,1,PX_MAX_F32,false);
        for(unsigned i=0;i<2;++i)
        {
            bodies[i]=physics->createRigidDynamic(PxTransform(PxVec3(i?.55f:-.55f,0,0)));
            PxShape* shape=PxRigidActorExt::createExclusiveShape(*bodies[i],PxBoxGeometry(.5f,.2f,.2f),*material);
            shape->setContactOffset(.02f);shape->setRestOffset(0);
            PxRigidBodyExt::setMassAndUpdateInertia(*bodies[i],i?2.f:1.f);
            bodies[i]->setLinearDamping(0);bodies[i]->setAngularDamping(0);bodies[i]->setSleepThreshold(0);
            bodies[i]->setMaxAngularVelocity(1000);
            bodies[i]->setLinearVelocity(PxVec3(i?-1.f:2.f,i?-.2f:.4f,0));scene->addActor(*bodies[i]);
        }
        PxConstraintShaderTable table={JointSoftBodySnippet::finiteJointPrep,NULL,PxConstraintFlag::eCOLLISION_ENABLED};
        table.elasticFramePrep=JointSoftBodySnippet::finiteJointMaterialPrep;
        joint.constraint=physics->createConstraint(bodies[0],bodies[1],joint,table,sizeof(joint.data));
        const double initialEnergy=mechanicalEnergy(joint,bodies);
        double maxMomentum=0,maxSpeed=0,maxEnergy=initialEnergy;bool finite=true;
        for(unsigned frame=0;frame<120;++frame)
        {
            scene->simulate(1.f/60);if(!scene->fetchResults(true))++failures;
            const PxVec3 momentum=bodies[0]->getLinearVelocity()+2.f*bodies[1]->getLinearVelocity();
            maxMomentum=PxMax(maxMomentum,double(momentum.magnitude()));
            maxEnergy=PxMax(maxEnergy,mechanicalEnergy(joint,bodies));
            for(unsigned i=0;i<2;++i)
            {
                finite=finite && bodies[i]->getGlobalPose().isFinite() && bodies[i]->getLinearVelocity().isFinite();
                maxSpeed=PxMax(maxSpeed,double(bodies[i]->getLinearVelocity().magnitude()));
            }
        }
        const bool passed=finite && maxMomentum<.01 && maxSpeed<4 && maxEnergy<=1.05*initialEnergy;
        failures+=!passed;
        std::printf("[GROUP_CONTACT_SCENE] friction=%u momentumError=%.9g maxSpeed=%.9g initialEnergy=%.9g maxEnergy=%.9g status=%s\n",friction,maxMomentum,maxSpeed,initialEnergy,maxEnergy,passed?"PASS":"FAIL");
        joint.constraint->release();for(unsigned i=0;i<2;++i)bodies[i]->release();
        scene->release();material->release();
    }
    dispatcher->release();PxCloseExtensions();physics->release();foundation->release();
    return failures+errors.failures?1:0;
}
