// Public constraint lifecycle, finite energy and shared velocity epoch.
#include "PxPhysicsAPI.h"
#include "avbd/DyAvbdElasticFrame.h"
#include "../physx/snippets/snippetjointsoftbody/SnippetFiniteJoint.h"
#include <cstdio>
#include <cmath>
using namespace physx;
struct Errors : PxErrorCallback
{
    unsigned failures=0,rejected=0;bool expected=false;
    void reportError(PxErrorCode::Enum code,const char* text,const char*,int) PX_OVERRIDE
    { if(code!=PxErrorCode::eDEBUG_INFO && code!=PxErrorCode::eDEBUG_WARNING)
      { if(expected) ++rejected;else { ++failures;std::printf("ERROR %s\n",text); } } }
} errors;
struct Connector : PxConstraintConnector
{
    JointSoftBodySnippet::FiniteJointData data;PxConstraint* constraint=NULL;
    unsigned releases=0;static unsigned rows,materials;
    static PxU32 prep(Px1DConstraint* r,PxVec3p& o,PxU32 n,PxConstraintInvMassScale& m,
        const void* d,const PxTransform& a,const PxTransform& b,bool e,PxVec3p& x,PxVec3p& y)
    { ++rows;return JointSoftBodySnippet::finiteJointPrep(r,o,n,m,d,a,b,e,x,y); }
    static bool material(PxConstraintElasticFrameData& out,const void* block)
    { ++materials;return JointSoftBodySnippet::finiteJointMaterialPrep(out,block); }
    void* prepareData() PX_OVERRIDE { return &data; }
    void onConstraintRelease() PX_OVERRIDE { ++releases;constraint=NULL; }
    void onComShift(PxU32) PX_OVERRIDE { if(constraint) constraint->markDirty(); }
    void onOriginShift(const PxVec3&) PX_OVERRIDE {}
    void* getExternalReference(PxU32& id) PX_OVERRIDE { id=PxConcreteType::eFIRST_USER_EXTENSION+1;return this; }
    bool updatePvdProperties(pvdsdk::PvdDataStream&,const PxConstraint*,PxPvdUpdateType::Enum) const PX_OVERRIDE { return false; }
    void updateOmniPvdProperties() const PX_OVERRIDE {}
    PxBase* getSerializable() PX_OVERRIDE { return NULL; }
    PxConstraintSolverPrep getPrep() const PX_OVERRIDE { return prep; }
    const void* getConstantBlock() const PX_OVERRIDE { return &data; }
};
unsigned Connector::rows=0;unsigned Connector::materials=0;
static double energy(const Connector& c,PxRigidDynamic* const* b)
{
    PxConstraintElasticFrameData data;JointSoftBodySnippet::finiteJointMaterialPrep(data,&c.data);
    PxTransform poses[2];double E=0;
    for(unsigned i=0;i<2;++i)
    {
        poses[i]=b[i]->getGlobalPose()*b[i]->getCMassLocalPose();
        const PxVec3 v=b[i]->getLinearVelocity(),w=poses[i].q.rotateInv(b[i]->getAngularVelocity());
        E+=.5*(double(b[i]->getMass())*double(v.magnitudeSquared())+double(w.dot(b[i]->getMassSpaceInertiaTensor().multiply(w))));
    }
    PxTransformT<double> bodies[2],frames[2];Dy::avbdElasticFrameSections(Dy::avbdCookFrameGeometry(data),poses,bodies,frames);
    const auto C=Dy::finiteJointCoordinates(frames[0],frames[1],bodies[0].p,bodies[1].p);
    for(unsigned i=0;i<18;++i) E+=.5*double(data.stiffness[Dy::finiteJointDriveIndex(i)])*C.value[i]*C.value[i];
    return E;
}
int main()
{
    PxDefaultAllocator allocator;PxFoundation* foundation=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors);
    PxPhysics* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation,PxTolerancesScale(),false,NULL);
    PxInitExtensions(*physics,NULL);PxDefaultCpuDispatcher* dispatcher=PxDefaultCpuDispatcherCreate(4);
    unsigned failures=0;
    for(unsigned type=0;type<3;++type)
    {
        const PxSolverType::Enum solver=type==0?PxSolverType::eAVBD:type==1?PxSolverType::ePGS:PxSolverType::eTGS;
        PxSceneDesc desc(physics->getTolerancesScale());desc.gravity=PxVec3(0);desc.cpuDispatcher=dispatcher;
        desc.filterShader=PxDefaultSimulationFilterShader;desc.solverType=solver;
        PxScene* scene=physics->createScene(desc);Connector c;
        c.data.c2b[0]=PxTransform(PxVec3(.5f,0,0));c.data.c2b[1]=PxTransform(PxVec3(-.5f,0,0));
        for(unsigned i=0;i<6;++i) c.data.drives[i]=PxD6JointDrive(1000.f+100.f*i,3.f,PX_MAX_F32,false);
        PxRigidDynamic* bodies[2];
        for(unsigned i=0;i<2;++i)
        {
            bodies[i]=physics->createRigidDynamic(PxTransform(PxVec3(i?.5f:-.5f,0,0)));
            bodies[i]->setMass(i?2.f:1.f);bodies[i]->setMassSpaceInertiaTensor(PxVec3(.2f,.3f,.4f)*(i?2.f:1.f));
            bodies[i]->setLinearDamping(0);bodies[i]->setAngularDamping(0);bodies[i]->setSleepThreshold(0);
            bodies[i]->setMaxAngularVelocity(1000);bodies[i]->setLinearVelocity(PxVec3(.3f,.1f,-.2f));scene->addActor(*bodies[i]);
        }
        bodies[1]->setGlobalPose(PxTransform(bodies[1]->getGlobalPose().p,PxQuat(type?1.2f:3.05f,PxVec3(1,2,1).getNormalized())));
        PxConstraintShaderTable table={Connector::prep,NULL,PxConstraintFlag::Enum(0)};table.elasticFramePrep=Connector::material;
        c.constraint=physics->createConstraint(bodies[0],bodies[1],c,table,sizeof(c.data));
        errors.expected=true;const unsigned rejected=errors.rejected;
        c.constraint->setBreakForce(1,1);PxReal linear,angular;c.constraint->getBreakForce(linear,angular);
        failures+=linear!=PX_MAX_F32 || angular!=PX_MAX_F32;
        c.constraint->setMinResponseThreshold(1);failures+=c.constraint->getMinResponseThreshold()!=0;
        PxVec3 force(1),torque(1);c.constraint->getForce(force,torque);failures+=!force.isZero() || !torque.isZero();
        c.constraint->setConstraintFunctions(c,table);
        PxRigidStatic* fixed=physics->createRigidStatic(PxTransform(PxIdentity));
        PxConstraint* invalid=physics->createConstraint(fixed,bodies[1],c,table,sizeof(c.data));
        failures+=invalid!=NULL;if(invalid) invalid->release();fixed->release();
        failures+=errors.rejected-rejected!=5;errors.expected=false;
        Connector::rows=Connector::materials=0;
        const double initial=energy(c,bodies);double maximum=initial,maxMomentum=0;
        const PxVec3 initialMomentum=3.f*PxVec3(.3f,.1f,-.2f);
        for(unsigned f=0;f<180;++f)
        {
            scene->simulate(1.f/60);if(!scene->fetchResults(true)) ++failures;
            maximum=PxMax(maximum,energy(c,bodies));
            const PxVec3 p=bodies[0]->getLinearVelocity()+2.f*bodies[1]->getLinearVelocity();
            maxMomentum=PxMax(maxMomentum,double((p-initialMomentum).magnitude()));
        }
        const bool passed=type?(Connector::materials==0 && Connector::rows>0):
            (Connector::materials==180 && Connector::rows==0 && maximum<=initial*1.05 && maxMomentum<.01);
        failures+=!passed;
        std::printf("[ELASTIC_FRAME_SCENE] solver=%u materialPreps=%u rowPreps=%u initialEnergy=%.9g maxEnergy=%.9g momentumError=%.9g status=%s\n",
            unsigned(solver),Connector::materials,Connector::rows,initial,maximum,maxMomentum,passed?"PASS":"FAIL");
        if(type==0)
        {
            for(unsigned i=0;i<6;++i) c.data.drives[i]=PxD6JointDrive(0,0,PX_MAX_F32,false);
            c.constraint->markDirty();
            const PxVec3 before[2]={bodies[0]->getLinearVelocity(),bodies[1]->getLinearVelocity()};
            for(unsigned i=0;i<4;++i) { scene->simulate(1.f/60);scene->fetchResults(true); }
            double drift=0;for(unsigned i=0;i<2;++i) drift=PxMax(drift,double((bodies[i]->getLinearVelocity()-before[i]).magnitude()));
            const bool free=Connector::materials==184 && Connector::rows==0 && drift<1.e-5;
            failures+=!free;std::printf("[ELASTIC_FRAME_CLEAR] velocityDrift=%.9g status=%s\n",drift,free?"PASS":"FAIL");
        }
        c.constraint->release();failures+=c.releases!=1;
        for(unsigned i=0;i<2;++i) bodies[i]->release();scene->release();
    }
    dispatcher->release();PxCloseExtensions();physics->release();foundation->release();
    return failures+errors.failures?1:0;
}
