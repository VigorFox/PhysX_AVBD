// Same external collision, dimensions, mass, E/nu and four-sweep budget.
// Rigid6 beam lattice and Particle3 tetrahedra have different constitutive
// discretizations. Compare quality and whole-step cost, never trajectories.
#include "avbd_probe_step_timer.h"
#include "PxPhysicsAPI.h"
#include "avbd_probe_foundation.h"
#include "avbd_probe_volume_energy.h"
#include "../physx/snippets/snippetjointsoftbody/SnippetJointSoftBodyCommon.h"
#include "../physx/snippets/snippetdeformablevolumeavbd/SnippetDeformableVolumeAVBDCommon.h"
using namespace physx;
int main(int argc,char** argv)
{
    const bool secondLattice=argc>1 && !std::strcmp(argv[1],"rigid6-soft");
    const bool rigid=secondLattice || (argc>1 && !std::strcmp(argv[1],"rigid6"));
    const PxU32 workers=argc>2?PxU32(std::atoi(argv[2])):4;
    const PxReal height=(secondLattice?4.2f:4.f)+(argc>3?PxReal(std::atof(argv[3])):0.f);
    const PxReal impactX=3.f+(argc>4?PxReal(std::atof(argv[4])):0.f);
    _putenv_s("PHYSX_AVBD_COMMON_STEP","budget");_putenv_s("PHYSX_AVBD_LOCAL_ITERATIONS","4");
    _putenv_s("PHYSX_NP_GEOMETRY_CERTIFICATES","2");
    AvbdProbeFoundation foundation;
    auto* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation.foundation,PxTolerancesScale());PxInitExtensions(*physics,NULL);
    auto* dispatcher=PxDefaultCpuDispatcherCreate(workers);
    PxSceneDesc desc(physics->getTolerancesScale());desc.solverType=PxSolverType::eAVBD;
    desc.gravity=PxVec3(0);desc.cpuDispatcher=dispatcher;desc.filterShader=JointSoftBodySnippet::filter;
    auto* scene=physics->createScene(desc);auto* material=physics->createMaterial(0,0,0);
    JointSoftBodySnippet::Lattice lattice,receivingLattice;
    PxDeformableVolume* volume=NULL;PxDeformableVolumeMesh* mesh=NULL;PxDeformableVolumeMaterial* softMaterial=NULL;
    if(rigid)
    {
        JointSoftBodySnippet::Settings settings;settings.grid=2;settings.young=2.e4f;settings.poisson=.3f;
        settings.dampingRatio=0;settings.volumeMaterial=true;
        if(!lattice.build(*physics,*scene,*material,settings,PxTransform(PxVec3(impactX,height,2)),1))return 2;
        for(auto* node:lattice.nodes){node->setLinearVelocity(PxVec3(0,-1,0));node->setSleepThreshold(0);node->setSolverIterationCounts(4,1);}
        if(secondLattice)
        {
            if(!receivingLattice.build(*physics,*scene,*material,settings,PxTransform(PxVec3(3,2,2)),2))return 2;
            for(auto* node:receivingLattice.nodes){node->setSleepThreshold(0);node->setSolverIterationCounts(4,1);}
        }
    }
    else
    {
        mesh=VolumeAvbdSnippet::cookCube(*physics,PxVec3(0),2.f);
        softMaterial=physics->createDeformableVolumeMaterial(2.e4f,.3f,0,0);
        softMaterial->setMaterialModel(PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);volume=physics->createDeformableVolume();
        auto* shape=physics->createShape(PxTetrahedronMeshGeometry(mesh->getCollisionMesh()),&softMaterial,1,true,PxShapeFlag::eSIMULATION_SHAPE);
        shape->setSimulationFilterData(PxFilterData(1,0,0,0));shape->setContactOffset(.02f);shape->setRestOffset(0);
        if(!volume->attachShape(*shape) || !volume->attachSimulationMesh(*mesh->getSimulationMesh(),*mesh->getDeformableVolumeAuxData()))return 2;
        shape->release();volume->setLinearDamping(0);volume->setSolverIterationCounts(4,1);
        volume->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION,true);
        PxVec4 *p=NULL,*v=NULL,*c=NULL,*r=NULL;PxDeformableVolumeExt::allocateAndInitializeHostMirror(*volume,NULL,p,v,c,r);
        PxDeformableVolumeExt::updateMass(*volume,100.f,50.f,p);
        for(PxU32 i=0;i<mesh->getSimulationMesh()->getNbVertices();++i)
        {p[i]=PxVec4(p[i].getXYZ()+PxVec3(impactX,height,2),p[i].w);v[i]=PxVec4(0,-1,0,p[i].w);}
        PxDeformableVolumeExt::updateEmbeddedCollisionMesh(*volume,p,c);
        PxDeformableVolumeExt::copyToDevice(*volume,PxDeformableVolumeDataFlag::eALL,p,v,c,r);
        VolumeAvbdSnippet::hostFree(p);VolumeAvbdSnippet::hostFree(v);VolumeAvbdSnippet::hostFree(c);VolumeAvbdSnippet::hostFree(r);
        if(!scene->addActor(*volume))return 2;
    }
    PxRigidDynamic* receiver=NULL;
    if(!secondLattice)
    {
        receiver=PxCreateDynamic(*physics,PxTransform(PxVec3(3,2,2)),PxBoxGeometry(1,.8f,1),*material,100.f);
        receiver->setLinearDamping(0);receiver->setAngularDamping(0);receiver->setSleepThreshold(0);receiver->setSolverIterationCounts(4,1);
        scene->addActor(*receiver);
    }
    const auto d=[](const PxVec3& p){return PxVec3d(p.x,p.y,p.z);};
    PxVec3d momentum,weightedCenter,angular;double mass=0,kinetic=0,elastic=0,minJ=1;
    bool finite=true;double receiverSpeed=0;
    const auto observe=[&]()
    {
        momentum=weightedCenter=angular=PxVec3d(0);mass=kinetic=elastic=0;
        const auto add=[&](double m,const PxVec3& p,const PxVec3& v)
        {mass+=m;momentum+=d(v)*m;weightedCenter+=d(p)*m;angular+=(d(p)-PxVec3d(3,2,2)).cross(d(v))*m;kinetic+=.5*m*d(v).magnitudeSquared();};
        const auto body=[&](PxRigidDynamic* b)
        {
            const auto pose=b->getGlobalPose()*b->getCMassLocalPose();add(b->getMass(),pose.p,b->getLinearVelocity());
            const auto w=d(pose.q.rotateInv(b->getAngularVelocity())),inertia=d(b->getMassSpaceInertiaTensor());
            kinetic+=.5*(w.x*w.x*inertia.x+w.y*w.y*inertia.y+w.z*w.z*inertia.z);
            const PxQuatT<double> q(pose.q.x,pose.q.y,pose.q.z,pose.q.w);
            angular+=q.rotate(inertia.multiply(w));
        };
        const auto softLattice=[&](const JointSoftBodySnippet::Lattice& l)
        {
            for(auto* b:l.nodes)body(b);
            for(auto* joint:l.joints)elastic+=joint->elasticEnergy();
            for(auto* m:l.materials)elastic+=m->elasticEnergy();
        };
        if(rigid)softLattice(lattice);
        else
        {
            const auto* p=volume->getSimPositionInvMassBufferD();const auto* v=volume->getSimVelocityBufferD();
            elastic=avbdProbeVolumeEnergy(*mesh->getSimulationMesh(),p,2.e4,.3,minJ);
            for(PxU32 i=0;i<mesh->getSimulationMesh()->getNbVertices();++i)add(1/double(p[i].w),p[i].getXYZ(),v[i].getXYZ());
        }
        if(receiver){body(receiver);receiverSpeed=receiver->getLinearVelocity().magnitude();}
        else
        {
            const auto incoming=momentum;const double incomingMass=mass;
            softLattice(receivingLattice);
            receiverSpeed=(momentum-incoming).magnitude()/(mass-incomingMass);
        }
        finite &= momentum.isFinite() && angular.isFinite() && weightedCenter.isFinite() && std::isfinite(kinetic+elastic);
    };
    observe();const auto P0=momentum,X0=weightedCenter/mass,L0=angular;const double initialKinetic=kinetic;
    const double initialEnergy=kinetic+elastic;
    double maxEnergy=initialEnergy,maxMomentum=0,maxAngular=0,maxCom=0,maxKinetic=kinetic,maxRigidSpeed=0;AvbdProbeStepTimer timer;
    bool simulated=true;
    for(PxU32 frame=1;frame<=180;++frame)
    {
        timer.begin();
        if(!scene->simulate(1.f/60) || !scene->fetchResults(true)){simulated=false;break;}
        timer.end();observe();
        maxMomentum=PxMax(maxMomentum,(momentum-P0).magnitude()/P0.magnitude());
        maxAngular=PxMax(maxAngular,(angular-L0).magnitude());
        maxCom=PxMax(maxCom,(weightedCenter/mass-X0-P0*(double(frame)*double(1.f/60)/mass)).magnitude());
        maxKinetic=PxMax(maxKinetic,kinetic);maxRigidSpeed=PxMax(maxRigidSpeed,receiverSpeed);
        maxEnergy=PxMax(maxEnergy,kinetic+elastic);
    }
    double mean,p95,cpuMs;timer.summarize(mean,p95,cpuMs);
    const bool pass=simulated && finite && minJ>0 && maxEnergy<1.05*initialEnergy && timer.times.size()==180 && maxMomentum<.005 && maxCom<.001 &&
        maxKinetic<1.05*initialKinetic && maxRigidSpeed>.1;
    std::printf("[MIXED_REACTION] model=%s workers=%u frames=%u iterations=4 height=%.6f impactX=%.6f mass=%.9g maxMomentumError=%.9g maxAngularAbs=%.9g maxComError=%.9g initialKinetic=%.9g maxKinetic=%.9g peakRigidSpeed=%.9g initialEnergy=%.9g peakEnergy=%.9g meanMs=%.6f p95Ms=%.6f cpuMs=%.6f status=%s\n",
        secondLattice?"rigid6-soft":rigid?"rigid6":"particle3",workers,unsigned(timer.times.size()),height,impactX,mass,maxMomentum,maxAngular,maxCom,initialKinetic,maxKinetic,maxRigidSpeed,initialEnergy,maxEnergy,mean,p95,cpuMs,pass?"PASS":"FAIL");
    receivingLattice.release();lattice.release();if(volume)volume->release();if(mesh)mesh->release();if(softMaterial)softMaterial->release();
    if(receiver)receiver->release();material->release();scene->release();dispatcher->release();PxCloseExtensions();physics->release();
    return pass?0:1;
}
