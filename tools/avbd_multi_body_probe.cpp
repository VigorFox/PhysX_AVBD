// Public Host lifecycle and production material-kernel regression.
#include "PxPhysicsAPI.h"
#include "avbd/DyAvbdTetrahedron.h"
#include "avbd/DyAvbdBlockGroup.h"
#include "extensions/PxCollectionExt.h"
#include "../physx/snippets/snippetdeformablevolumeavbd/SnippetDeformableVolumeAVBDCommon.h"
#include "../physx/snippets/snippetjointsoftbody/SnippetVolumeMaterial.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <limits>
using namespace physx;
static PxDefaultAllocator allocator;
struct Errors : PxErrorCallback
{
    bool expected=false; unsigned unexpected=0, rejected=0;
    void reportError(PxErrorCode::Enum c,const char* m,const char*,int) PX_OVERRIDE
    {
        if(c==PxErrorCode::eDEBUG_INFO || c==PxErrorCode::eDEBUG_WARNING) { std::printf("info: %s\n",m);return; }
        if(expected) ++rejected;
        else { ++unexpected; std::printf("ERROR: %s\n",m); }
    }
} errors;
struct Material : PxConstraintConnector
{
    PxConstraintTetrahedronData data;
    PxConstraint* constraint=NULL;
    unsigned releases=0;
    static unsigned preps;
    Material() { for(unsigned i=0;i<4;++i) data.localPoints[i]=PxVec3(0);data.restDeterminant=1;data.energyScale=100; }
    static bool prep(PxConstraintTetrahedronData& out,const void* block)
    { ++preps;out=*static_cast<const PxConstraintTetrahedronData*>(block);return true; }
    void* prepareData() PX_OVERRIDE { return &data; }
    void onConstraintRelease() PX_OVERRIDE { ++releases; constraint=NULL; }
    void onComShift(PxU32) PX_OVERRIDE { if(constraint) constraint->markDirty(); }
    void onOriginShift(const PxVec3&) PX_OVERRIDE {}
    void* getExternalReference(PxU32& id) PX_OVERRIDE { id=PxConcreteType::eFIRST_USER_EXTENSION+3;return this; }
    bool updatePvdProperties(pvdsdk::PvdDataStream&,const PxConstraint*,PxPvdUpdateType::Enum) const PX_OVERRIDE { return false; }
    void updateOmniPvdProperties() const PX_OVERRIDE {}
    PxBase* getSerializable() PX_OVERRIDE { return NULL; }
    PxConstraintSolverPrep getPrep() const PX_OVERRIDE { return NULL; }
    const void* getConstantBlock() const PX_OVERRIDE { return &data; }
    void connectToConstraint(PxConstraint* c) PX_OVERRIDE { constraint=c; }
    PxConstraint* create(PxPhysics& p,PxRigidActor** a)
    {
        PxConstraintShaderTable t={NULL,NULL,PxConstraintFlag::eCOLLISION_ENABLED};t.tetrahedronPrep=prep;
        return p.createConstraint(a,4,*this,t,sizeof(data));
    }
};
unsigned Material::preps=0;
static int failures=0;
#define CHECK(x) do { if(!(x)) { ++failures;std::printf("FAIL line %d: %s\n",__LINE__,#x); } } while(0)
static const PxVec3 rest[4]={PxVec3(0),PxVec3(1,0,0),PxVec3(0,1,0),PxVec3(0,0,1)};
static void step(PxScene& s) { s.simulate(1.0f/60);s.fetchResults(true); }
static double determinant(PxRigidActor** a)
{
    PxVec3d p[4];for(unsigned i=0;i<4;++i) { PxVec3 v=a[i]->getGlobalPose().p;p[i]=PxVec3d(v.x,v.y,v.z); }
    return (p[1]-p[0]).dot((p[2]-p[0]).cross(p[3]-p[0]));
}
static bool kernel()
{
    using namespace physx::Dy;
    const AvbdTetrahedronMaterial m={1.0,120.0,0};
    PxVec3d p[4]={PxVec3d(.1,-.2,.3),PxVec3d(1.1,.1,.2),PxVec3d(.2,.8,.4),PxVec3d(-.1,.2,1.4)};
    const AvbdTetrahedronEvaluation e=avbdEvaluateTetrahedron(m,p);
    CHECK(e.valid);
    for(unsigned s=0;s<4;++s) for(unsigned k=0;k<3;++k)
    {
        const double saved=p[s][k],h=1.e-6;
        p[s][k]=saved+h;const double plus=avbdEvaluateTetrahedron(m,p).energy;
        p[s][k]=saved-h;const double minus=avbdEvaluateTetrahedron(m,p).energy;
        p[s][k]=saved;
        CHECK(std::abs((plus-minus)/(2*h)-e.gradient*e.jacobians[s][k])<1.e-6);
    }
    PxVec3d force(0),torque(0);
    for(unsigned s=0;s<4;++s) { force+=e.jacobians[s];torque+=p[s].cross(e.jacobians[s]); }
    CHECK(force.magnitude()<1.e-12 && torque.magnitude()<1.e-12);
    PxVec3d q[4];for(unsigned s=0;s<4;++s) q[s]=p[s]+PxVec3d(7,-4,2);
    double err[4]={};CHECK(avbdTetrahedronPathAdmissible(m,p,q,err));
    CHECK(PxAbs(avbdEvaluateTetrahedron(m,q).energy-e.energy)<1.e-10);
    const double co=std::cos(.37),si=std::sin(.37);
    for(unsigned s=0;s<4;++s) q[s]=PxVec3d(co*p[s].x-si*p[s].y,si*p[s].x+co*p[s].y,p[s].z);
    CHECK(PxAbs(avbdEvaluateTetrahedron(m,q).energy-e.energy)<1.e-10);
    // Angular derivatives of arbitrary COM-local material points.
    for(unsigned s=0;s<4;++s) for(unsigned axis=0;axis<3;++axis)
    {
        const PxVec3d arm(.2,-.3,.4),saved=p[s];PxVec3d direction(0);direction[axis]=1;
        const double h=1.e-6;
        p[s]=saved+direction.cross(arm)*h;const double plus=avbdEvaluateTetrahedron(m,p).energy;
        p[s]=saved-direction.cross(arm)*h;const double minus=avbdEvaluateTetrahedron(m,p).energy;p[s]=saved;
        CHECK(PxAbs((plus-minus)/(2*h)-e.gradient*arm.cross(e.jacobians[s])[axis])<1.e-6);
    }
    p[0]=PxVec3d(0);p[1]=PxVec3d(1,0,0);p[2]=PxVec3d(0,1,0);p[3]=PxVec3d(0,0,.01);
    CHECK(avbdEvaluateTetrahedron(m,p).valid);
    p[3].z=0;CHECK(!avbdEvaluateTetrahedron(m,p).valid);
    p[3].z=-.1;CHECK(!avbdEvaluateTetrahedron(m,p).valid);
    // Both endpoint determinants are positive; the straight path crosses zero.
    p[3].z=1;for(unsigned s=0;s<4;++s) q[s]=p[s];q[1].x=-1;q[2].y=-1;
    CHECK(avbdEvaluateTetrahedron(m,q).valid);
    CHECK(!avbdTetrahedronPathAdmissible(m,p,q,err));
    q[1]=p[1];q[2]=p[2];q[3]=PxVec3d(.2,.3,.1);
    CHECK(avbdTetrahedronPathAdmissible(m,p,q,err));
    q[3].z=-.1;CHECK(!avbdTetrahedronPathAdmissible(m,p,q,err));
    // Positive arithmetic orientation smaller than its input-operation error
    // cannot be used as a certified admissible material state.
    p[1]=PxVec3d(1,0,0);p[2]=PxVec3d(1,1.e-16,0);p[3]=PxVec3d(0,0,1);
    for(unsigned s=0;s<4;++s) q[s]=p[s];
    CHECK(!avbdTetrahedronPathAdmissible(m,p,q,err));
    // Independent residual oracle for a coupled four-Rigid6 linear system.
    PxReal diagonal[4][6][6]={},gradient[4][6]={},solution[4][6]={};
    AvbdGroupFactor factor={};factor.curvature=120;
    for(PxU32 b=0;b<4;++b)
    {
        for(PxU32 k=0;k<6;++k) diagonal[b][k][k]=1+PxReal(b);
        factor.jacobian[b][0]=b==0 ? -3.0f : 1.0f;
        diagonal[b][0][0]+=factor.curvature*factor.jacobian[b][0]*factor.jacobian[b][0];
        gradient[b][0]=factor.jacobian[b][0]*3;
    }
    bool solved=false;AvbdGroupSolveArgs group={4,diagonal,gradient,&factor,1,solution,&solved};
    CHECK(avbdSolveBlockGroup(group));
    double projection=0,momentum=0;
    for(PxU32 b=0;b<4;++b) { projection+=factor.jacobian[b][0]*solution[b][0];momentum+=(b+1)*solution[b][0]; }
    CHECK(PxAbs(momentum)<1.e-5);
    for(PxU32 b=0;b<4;++b)
        CHECK(PxAbs((b+1)*solution[b][0]+120*factor.jacobian[b][0]*projection+gradient[b][0])<1.e-4);
    return failures==0;
}
static void mixedProgram(PxPhysics& physics,PxCpuDispatcher& dispatcher)
{
    // A one-DOF material has an independently computed implicit equilibrium.
    // Adding a remote FEM actor must not duplicate the prepared material row.
    double lo=.8,hi=1;
    for(unsigned i=0;i<80;++i)
    {
        const double j=.5*(lo+hi),d=j-1;
        const double residual=3600*(j-.8)+10000*(2*d*(d-std::log(j))+d*d*d/j);
        if(residual>0) hi=j;else lo=j;
    }
    const double reference=.5*(lo+hi);
    for(unsigned mixed=0;mixed<2;++mixed)
    {
        PxSceneDesc desc(physics.getTolerancesScale());desc.cpuDispatcher=&dispatcher;
        desc.gravity=PxVec3(0);desc.solverType=PxSolverType::eAVBD;desc.filterShader=PxDefaultSimulationFilterShader;
        PxScene* scene=physics.createScene(desc);PxRigidActor* a[4];
        a[0]=physics.createRigidStatic(PxTransform(rest[0]));a[1]=physics.createRigidStatic(PxTransform(rest[1]));
        a[2]=physics.createRigidDynamic(PxTransform(rest[2]));
        static_cast<PxRigidDynamic*>(a[2])->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC,true);
        auto* dynamic=physics.createRigidDynamic(PxTransform(PxVec3(0,0,.8f)));
        dynamic->setMass(1);dynamic->setMassSpaceInertiaTensor(PxVec3(1));
        dynamic->setLinearDamping(0);dynamic->setAngularDamping(0);a[3]=dynamic;
        for(unsigned i=0;i<4;++i) scene->addActor(*a[i]);
        Material m;m.data.localPoints[1]=rest[1];m.data.energyScale=10000;
        PxConstraint* c=m.create(physics,a);CHECK(c);
        PxDeformableVolumeMesh* mesh=NULL;PxDeformableVolume* volume=NULL;
        if(mixed)
        {
            mesh=VolumeAvbdSnippet::cookCube(physics,PxVec3(0),1);
            PxDeformableVolumeMaterial* soft=physics.createDeformableVolumeMaterial(2.e5f,.3f,.1f);
            volume=VolumeAvbdSnippet::createVisualVolume(physics,*scene,*mesh,*soft,
                PxTransform(PxVec3(10,10,0)),PxVec3(1),100);soft->release();CHECK(volume);
        }
        step(*scene);
        const double value=dynamic->getGlobalPose().p.z;
        std::printf("[MULTI_BODY_MIXED] fem=%u z=%.9g reference=%.9g error=%.9g\n",mixed,value,reference,PxAbs(value-reference));
        CHECK(PxAbs(value-reference)<1.e-4);
        c->release();
        // Reuse the Host constraint pool across material and 1D shaders. The
        // replacement has its own rest data, and a remote FEM must not double
        // its spring either. This also checks retained preparation invalidation.
        const double springReference=(3600*.8+10000)/13600;
        for(unsigned replacement=0;replacement<4;++replacement)
        {
            dynamic->setGlobalPose(PxTransform(PxVec3(0,0,.8f)));
            dynamic->setLinearVelocity(PxVec3(0));dynamic->setAngularVelocity(PxVec3(0));
            PxD6Joint* joint=PxD6JointCreate(physics,NULL,PxTransform(PxVec3(0,0,1)),dynamic,PxTransform(PxIdentity));
            CHECK(joint);
            for(unsigned axis=0;axis<6;++axis) joint->setMotion(PxD6Axis::Enum(axis),PxD6Motion::eFREE);
            joint->setDrive(PxD6Drive::eZ,PxD6JointDrive(10000,0,PX_MAX_F32));
            step(*scene);
            CHECK(PxAbs(double(dynamic->getGlobalPose().p.z)-springReference)<1.e-4);
            joint->release();
            dynamic->setGlobalPose(PxTransform(PxVec3(0,0,.8f)));
            dynamic->setLinearVelocity(PxVec3(0));dynamic->setAngularVelocity(PxVec3(0));
            c=m.create(physics,a);CHECK(c);step(*scene);
            CHECK(PxAbs(double(dynamic->getGlobalPose().p.z)-reference)<1.e-4);
            c->release();
        }
        if(volume) volume->release();if(mesh) mesh->release();
        for(unsigned i=0;i<4;++i) a[i]->release();scene->release();
    }
}
static void sleepCounting(PxPhysics& physics,PxCpuDispatcher& dispatcher)
{
    PxSceneDesc desc(physics.getTolerancesScale());desc.cpuDispatcher=&dispatcher;
    desc.gravity=PxVec3(0);desc.solverType=PxSolverType::eAVBD;desc.filterShader=PxDefaultSimulationFilterShader;
    PxScene* scene=physics.createScene(desc);PxRigidActor* actors[4];
    for(PxU32 i=0;i<4;++i)
    {
        auto* body=physics.createRigidDynamic(PxTransform(rest[i]));
        body->setMass(1);body->setMassSpaceInertiaTensor(PxVec3(1));
        body->setLinearDamping(0);body->setAngularDamping(0);
        body->setLinearVelocity(PxVec3(.014f,0,0));actors[i]=body;scene->addActor(*body);
    }
    Material material;PxConstraint* constraint=material.create(physics,actors);CHECK(constraint);
    // With one physical constraint per actor, the ordinary accumulated-energy
    // sleep test keeps this unforced rigid translation active. Three counted
    // graph edges for that same constraint incorrectly discard its momentum.
    for(PxU32 frame=0;frame<80;++frame) step(*scene);
    PxVec3 momentum(0);PxU32 awake=0;
    for(PxU32 i=0;i<4;++i)
    {
        auto* body=static_cast<PxRigidDynamic*>(actors[i]);
        momentum+=body->getLinearVelocity();awake+=!body->isSleeping();
    }
    std::printf("[MULTI_BODY_SLEEP] awake=%u momentum=%.9g\n",awake,momentum.x);
    CHECK(awake==4);CHECK(PxAbs(momentum.x-.056f)<1.e-4f);
    constraint->release();for(PxU32 i=0;i<4;++i) actors[i]->release();scene->release();
}
static void materialFrames(PxPhysics& physics,PxCpuDispatcher& dispatcher)
{
    PxSceneDesc desc(physics.getTolerancesScale());desc.cpuDispatcher=&dispatcher;
    desc.gravity=PxVec3(0);desc.solverType=PxSolverType::eAVBD;desc.filterShader=PxDefaultSimulationFilterShader;
    PxScene* scene=physics.createScene(desc);PxRigidActor* actors[4];
    for(PxU32 i=0;i<4;++i)
    {
        auto* body=physics.createRigidDynamic(PxTransform(rest[i]));
        body->setMass(1);body->setMassSpaceInertiaTensor(PxVec3(1));
        actors[i]=body;scene->addActor(*body);
    }
    auto* material=JointSoftBodySnippet::VolumeMaterial::create(physics,actors,1,10000);CHECK(material);
    // A mass-frame change must preserve each authored actor-space material
    // anchor, including supports 2 and 3. Unequal shifts prevent a common
    // translation from hiding a missed callback.
    for(PxU32 i=0;i<4;++i)
    {
        static_cast<PxRigidDynamic*>(actors[i])->setCMassLocalPose(PxTransform(
            PxVec3(.13f,.17f,.19f)*PxReal(i+1),PxQuat(.3f*PxReal(i+1),PxVec3(0,0,1))));
        CHECK(material->elasticEnergy()<1.e-8);step(*scene);
        for(PxU32 j=0;j<4;++j) CHECK((actors[j]->getGlobalPose().p-rest[j]).magnitude()<2.e-6f);
    }
    const PxVec3 shift(7,-4,2);scene->shiftOrigin(shift);step(*scene);
    CHECK(material->elasticEnergy()<1.e-8);
    for(PxU32 i=0;i<4;++i) CHECK((actors[i]->getGlobalPose().p+shift-rest[i]).magnitude()<2.e-6f);
    std::printf("[MULTI_BODY_FRAMES] energy=%.9g failures=%d\n",material->elasticEnergy(),failures);
    material->release();for(PxU32 i=0;i<4;++i) actors[i]->release();scene->release();
}
int main(int argc,char** argv)
{
    kernel();
    PxFoundation* f=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors);
    PxPhysics* p=PxCreatePhysics(PX_PHYSICS_VERSION,*f,PxTolerancesScale());
    PxInitExtensions(*p,NULL);
    PxDefaultCpuDispatcher* dispatcher=PxDefaultCpuDispatcherCreate(2);
    sleepCounting(*p,*dispatcher);
    if(argc>1 && std::strcmp(argv[1],"--sleep-count-only")==0)
    {
        dispatcher->release();PxCloseExtensions();p->release();f->release();
        return failures || errors.unexpected ? 1 : 0;
    }
    materialFrames(*p,*dispatcher);
    mixedProgram(*p,*dispatcher);
    if(argc>1 && std::strcmp(argv[1],"--mixed-only")==0)
    {
        dispatcher->release();PxCloseExtensions();p->release();f->release();
        return failures || errors.unexpected ? 1 : 0;
    }
    PxSceneDesc desc(p->getTolerancesScale());desc.cpuDispatcher=dispatcher;desc.filterShader=PxDefaultSimulationFilterShader;desc.gravity=PxVec3(0);desc.solverType=PxSolverType::eAVBD;
    PxScene* scene=p->createScene(desc);PxScene* other=p->createScene(desc);
    PxRigidActor* a[4];
    for(unsigned i=0;i<4;++i)
    {
        PxRigidDynamic* d=p->createRigidDynamic(PxTransform(rest[i]));d->setMass(1);d->setMassSpaceInertiaTensor(PxVec3(1));d->setLinearDamping(0);d->setAngularDamping(0);a[i]=d;
    }
    Material material;
    PxConstraint* c=material.create(*p,a);CHECK(c!=NULL);
    if(!c) return 2;
    CHECK(c->getNbActors()==4 && c->getScene()==NULL);
    for(unsigned i=0;i<4;++i) { scene->addActor(*a[i]);CHECK(c->getScene()==(i==3 ? scene : NULL)); }
    PxRigidActor* got[4]={};CHECK(c->getActors(got,4)==4);for(unsigned i=0;i<4;++i) CHECK(got[i]==a[i]);
    const unsigned before=Material::preps;
    static_cast<PxRigidDynamic*>(a[3])->addForce(PxVec3(0,0,-60),PxForceMode::eIMPULSE);
    double minimum=1;PxVec3 total(0);
    for(unsigned n=0;n<120;++n) { step(*scene);minimum=PxMin(minimum,determinant(a)); }
    for(unsigned i=0;i<4;++i) total+=static_cast<PxRigidDynamic*>(a[i])->getLinearVelocity();
    CHECK(minimum>0);CHECK((total-PxVec3(0,0,-60)).magnitude()<.02f);
    CHECK(Material::preps-before==120); // Six Host edges, exactly one prepared physical term.
    CHECK(static_cast<PxRigidDynamic*>(a[0])->getLinearVelocity().magnitude()>.01f);
    // Reset a rest configuration to exercise ordinary island sleep/wake.
    for(unsigned i=0;i<4;++i) { auto* d=static_cast<PxRigidDynamic*>(a[i]);d->setGlobalPose(PxTransform(rest[i]));d->putToSleep(); }
    step(*scene);for(unsigned i=0;i<4;++i) CHECK(static_cast<PxRigidDynamic*>(a[i])->isSleeping());
    static_cast<PxRigidDynamic*>(a[3])->addForce(PxVec3(0,0,1),PxForceMode::eIMPULSE);step(*scene);
    for(unsigned i=0;i<4;++i) CHECK(!static_cast<PxRigidDynamic*>(a[i])->isSleeping());
    // An inactive kinematic at index 2 must remain a declared read dependency.
    auto* k=static_cast<PxRigidDynamic*>(a[2]);k->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC,true);step(*scene);
    k->setKinematicTarget(PxTransform(rest[2]+PxVec3(0,.05f,0)));step(*scene);
    CHECK(determinant(a)>0);k->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC,false);step(*scene);
    // Unsupported mutation is rejected without replacing a valid binding.
    PxRigidActor* duplicate[4]={a[0],a[1],a[2],a[2]};errors.expected=true;CHECK(!c->setActors(duplicate,4));errors.expected=false;
    CHECK(c->getScene()==scene);
    PxRigidDynamic* replacement=p->createRigidDynamic(PxTransform(rest[2]));other->addActor(*replacement);
    PxRigidActor* rebound[4]={a[0],a[1],replacement,a[3]};errors.expected=true;CHECK(!c->setActors(rebound,4));errors.expected=false;
    other->removeActor(*replacement);scene->addActor(*replacement);CHECK(c->setActors(rebound,4));
    CHECK(c->getScene()==scene);
    scene->simulate(1.0f/60);errors.expected=true;CHECK(!c->setActors(a,4));errors.expected=false;scene->fetchResults(true);
    scene->removeActor(*replacement);CHECK(c->getScene()==NULL);
    // No constraint scene does not mean all support scenes are writable.
    PxRigidActor* detached[4];for(unsigned i=0;i<4;++i) detached[i]=p->createRigidDynamic(PxTransform(rest[i]));
    scene->simulate(1.0f/60);errors.expected=true;
    const unsigned rejectedBefore=errors.rejected;
    CHECK(!c->setActors(detached,4));c->markDirty();c->release();
    errors.expected=false;scene->fetchResults(true);CHECK(errors.rejected==rejectedBefore+3);
    CHECK(material.releases==0 && c->getActors(got,4)==4 && got[2]==replacement);
    for(unsigned i=0;i<4;++i) detached[i]->release();
    scene->addActor(*replacement);CHECK(c->getScene()==scene);
    replacement->release();CHECK(c->getScene()==NULL && !c->isValid());
    scene->removeActor(*a[0]);scene->addActor(*a[0]);CHECK(c->getScene()==NULL);
    CHECK(c->setActors(a,4));CHECK(c->isValid() && c->getScene()==scene);step(*scene);
    errors.expected=true;c->setBreakForce(1,1);c->setFlag(PxConstraintFlag::eCOLLISION_ENABLED,false);
    c->setBreakForce(PX_MAX_F32,std::numeric_limits<PxReal>::infinity());
    c->setBreakForce(std::numeric_limits<PxReal>::quiet_NaN(),PX_MAX_F32);
    PxVec3 force(1),torque(1);c->getForce(force,torque);c->setMinResponseThreshold(1);errors.expected=false;
    CHECK(force==PxVec3(0) && torque==PxVec3(0) && c->getMinResponseThreshold()==0);
    errors.expected=true;c->setFlag(PxConstraintFlag::eVISUALIZATION,true);errors.expected=false;
    CHECK(!(c->getFlags()&PxConstraintFlag::eVISUALIZATION));
    unsigned disabledPreps=Material::preps;
    c->setFlag(PxConstraintFlag::eDISABLE_CONSTRAINT,true);step(*scene);CHECK(Material::preps==disabledPreps);
    c->setFlag(PxConstraintFlag::eDISABLE_CONSTRAINT,false);step(*scene);CHECK(Material::preps==disabledPreps+1);
    PxReal lin,ang;c->getBreakForce(lin,ang);CHECK(lin==PX_MAX_F32 && ang==PX_MAX_F32);CHECK(c->getFlags()&PxConstraintFlag::eCOLLISION_ENABLED);
    PxCollection* collection=PxCreateCollection();collection->add(*c);PxSerializationRegistry* registry=PxSerialization::createSerializationRegistry(*p);
    errors.expected=true;CHECK(!PxSerialization::isSerializable(*collection,*registry));errors.expected=false;collection->release();registry->release();
    c->release();CHECK(material.releases==1);
    for(unsigned i=0;i<4;++i) a[i]->release();
    // A static pair in the first two slots must not hide a dynamic fourth slot.
    for(unsigned i=0;i<4;++i)
    {
        a[i]=i<2 ? static_cast<PxRigidActor*>(p->createRigidStatic(PxTransform(rest[i]))) :
            static_cast<PxRigidActor*>(p->createRigidDynamic(PxTransform(rest[i])));
        scene->addActor(*a[i]);
    }
    static_cast<PxRigidDynamic*>(a[2])->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC,true);
    Material boundary;boundary.data.localPoints[1]=rest[1];
    c=boundary.create(*p,a);CHECK(c && c->getScene()==scene);
    step(*scene);CHECK(determinant(a)>0);
    scene->removeActor(*a[1]);CHECK(c->getScene()==NULL);scene->addActor(*a[1]);CHECK(c->getScene()==scene);
    c->release();for(unsigned i=0;i<4;++i) a[i]->release();
    // World support and all four deletion slots use the same Host lifecycle.
    a[0]=NULL;for(unsigned i=1;i<4;++i) { a[i]=p->createRigidDynamic(PxTransform(rest[i]));scene->addActor(*a[i]); }
    Material world;c=world.create(*p,a);CHECK(c && c->getScene()==scene);step(*scene);c->release();
    for(unsigned i=1;i<4;++i) a[i]->release();
    for(unsigned removed=0;removed<4;++removed)
    {
        Material deletion;
        for(unsigned i=0;i<4;++i) { a[i]=p->createRigidDynamic(PxTransform(rest[i]));scene->addActor(*a[i]); }
        c=deletion.create(*p,a);CHECK(c && c->getScene()==scene);
        a[removed]->setActorFlag(PxActorFlag::eDISABLE_SIMULATION,true);CHECK(c->getScene()==NULL);
        a[removed]->setActorFlag(PxActorFlag::eDISABLE_SIMULATION,false);CHECK(c->getScene()==scene);
        a[removed]->release();a[removed]=NULL;CHECK(!c->isValid() && c->getScene()==NULL);
        c->release();for(unsigned i=0;i<4;++i) if(a[i]) a[i]->release();
    }
    for(unsigned solver=0;solver<2;++solver)
    {
        desc.solverType=solver==0 ? PxSolverType::ePGS : PxSolverType::eTGS;
        PxScene* unsupported=p->createScene(desc);
        for(unsigned i=0;i<4;++i) { a[i]=p->createRigidDynamic(PxTransform(rest[i]));unsupported->addActor(*a[i]); }
        Material rejected;errors.expected=true;CHECK(rejected.create(*p,a)==NULL);errors.expected=false;
        for(unsigned i=0;i<4;++i) a[i]->release();unsupported->release();
    }
    // Existing two-body binary collections still round-trip in this fork's
    // explicitly versioned layout. Four-body serialization remains rejected.
    PxRigidDynamic* sa=p->createRigidDynamic(PxTransform(rest[0]));
    PxRigidDynamic* sb=p->createRigidDynamic(PxTransform(rest[1]));
    PxD6Joint* joint=PxD6JointCreate(*p,sa,PxTransform(PxVec3(.5f,0,0)),sb,PxTransform(PxVec3(-.5f,0,0)));
    collection=PxCreateCollection();collection->add(*joint);
    registry=PxSerialization::createSerializationRegistry(*p);PxSerialization::complete(*collection,*registry);
    PxDefaultMemoryOutputStream stream;CHECK(PxSerialization::serializeCollectionToBinary(stream,*collection,*registry));
    void* memory=allocator.allocate(stream.getSize()+128,"Probe serialized collection",__FILE__,__LINE__);
    void* aligned=reinterpret_cast<void*>((reinterpret_cast<size_t>(memory)+127)&~size_t(127));
    PxMemCopy(aligned,stream.getData(),stream.getSize());
    PxCollection* restored=PxSerialization::createCollectionFromBinary(aligned,*registry);CHECK(restored);
    if(restored)
    {
        unsigned constraints=0;
        for(PxU32 i=0;i<restored->getNbObjects();++i) if(auto* rc=restored->getObject(i).is<PxConstraint>())
        { ++constraints;CHECK(rc->getNbActors()==2); }
        CHECK(constraints==1);scene->addCollection(*restored);step(*scene);
        PxCollectionExt::releaseObjects(*restored);restored->release();
    }
    PxMemCopy(aligned,stream.getData(),stream.getSize());bool foundVersion=false;
    for(PxU32 i=0;i+32<=stream.getSize();++i) if(std::memcmp(static_cast<char*>(aligned)+i,PX_BINARY_SERIAL_VERSION,32)==0)
    {
        std::memcpy(static_cast<char*>(aligned)+i,"04A07573A4E94B1E98F95D5FBDE3AF6F",32);foundVersion=true;break;
    }
    CHECK(foundVersion);errors.expected=true;CHECK(PxSerialization::createCollectionFromBinary(aligned,*registry)==NULL);errors.expected=false;
    allocator.deallocate(memory);registry->release();collection->release();joint->release();sa->release();sb->release();
    other->release();scene->release();dispatcher->release();PxCloseExtensions();p->release();f->release();
    CHECK(errors.unexpected==0);
    std::printf("[MULTI_BODY] minDet=%.9g momentum=(%.7g,%.7g,%.7g) preps=%u rejected=%u failures=%d\n",minimum,total.x,total.y,total.z,Material::preps,errors.rejected,failures);
    return failures ? 1 : 0;
}
