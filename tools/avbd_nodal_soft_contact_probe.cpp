// Two independent native actors share one NP contact and one dual per row.
// Public-state observer only: no second solver or corrective writeback.
#include "avbd_probe_step_timer.h"
#include "PxPhysicsAPI.h"
#include "avbd_probe_foundation.h"
#include "avbd_probe_volume_energy.h"
#include "../physx/snippets/snippetdeformablevolumeavbd/SnippetDeformableVolumeAVBDCommon.h"
#include <cstdio>
#include <cstdlib>
using namespace physx;

// With a horizontal plane, even a changing contact patch must obey the
// integrated Coulomb cone. Its total normal impulse follows from gravity
// and the observed change in vertical momentum of the volume.
static bool slideOnPlane(PxPhysics& physics,PxScene& scene,PxReal friction)
{
    const PxReal step=1.f/60.f,drive=scene.getGravity().x,initialSpeed=drive==0?5.f:0.f;
    auto* groundMaterial=physics.createMaterial(friction,friction,0);
    if(!groundMaterial)return false;
    auto* ground=PxCreatePlane(physics,PxPlane(0,1,0,0),*groundMaterial);
    if(!ground || !scene.addActor(*ground)){PX_RELEASE(ground);groundMaterial->release();return false;}
    PxArray<PxVec3> vertices;PxArray<PxU32> triangles;
    VolumeAvbdSnippet::createSubdividedCubeSurface(vertices,triangles,2.f,4);
    auto* mesh=VolumeAvbdSnippet::cookVisualMesh(physics,vertices,triangles,5);
    auto* material=physics.createDeformableVolumeMaterial(3.e5f,.42f,friction,.009f);
    if(!mesh || !material){PX_RELEASE(mesh);PX_RELEASE(material);ground->release();groundMaterial->release();return false;}
    material->setMaterialModel(PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);
    auto* body=VolumeAvbdSnippet::createVisualVolume(physics,scene,*mesh,*material,
        PxTransform(PxVec3(0,1.2f,0)),PxVec3(1),160.f);
    if(!body){mesh->release();material->release();ground->release();groundMaterial->release();return false;}
    body->setLinearDamping(0);
    body->setSolverIterationCounts(4,1);
    const bool initialized=VolumeAvbdSnippet::setVisualVelocity(*body,PxVec3(initialSpeed,0,0));
    PxReal excess=0;PxVec3 velocity(0);PxU32 frame=0;
    for(;initialized && frame<120;++frame)
    {
        if(!scene.simulate(step) || !scene.fetchResults(true))break;
        const auto* p=body->getSimPositionInvMassBufferD();
        const auto* v=body->getSimVelocityBufferD();
        double mass=0;PxVec3d momentum(0);
        for(PxU32 i=0;i<mesh->getSimulationMesh()->getNbVertices();++i)
        {
            const double m=1./p[i].w;
            mass+=m;momentum+=PxVec3d(v[i].x,v[i].y,v[i].z)*m;
        }
        velocity=PxVec3(PxReal(momentum.x/mass),PxReal(momentum.y/mass),PxReal(momentum.z/mass));
        if(!velocity.isFinite())break;
        const PxReal normalImpulse=velocity.y+9.81f*(frame+1)*step;
        const PxReal freeSpeed=initialSpeed+drive*(frame+1)*step;
        excess=PxMax(excess,std::hypot(velocity.x-freeSpeed,velocity.z)-friction*normalImpulse);
    }
    const bool pass=frame==120 && velocity.isFinite() && excess<=.05f;
    std::printf("[SOFT_PLANE_SLIDE] frames=%u friction=%.6g tangentialGravity=%.6g finalVx=%.6g excessImpulsePerMass=%.6g status=%s\n",
        frame,double(friction),double(drive),double(velocity.x),double(excess),pass?"PASS":"FAIL");
    body->release();mesh->release();material->release();ground->release();groundMaterial->release();
    return pass;
}
int main(int argc,char** argv)
{
    const bool plane=argc>2 && std::strcmp(argv[2],"plane")==0;
    const PxU32 workers=argc>1?PxU32(std::atoi(argv[1])):4;
    const PxReal height=4.2f+(argc>2?PxReal(std::atof(argv[2])):0.f);
    const PxReal impactX=3.f+(argc>3?PxReal(std::atof(argv[3])):0.f);
    const PxReal commonSpeed=argc>4?PxReal(std::atof(argv[4])):0.f;
    const PxReal closingSpeed=argc>5?PxReal(std::atof(argv[5])):1.f;
    const PxReal relativeSlide=argc>6?PxReal(std::atof(argv[6])):0.f;
    if(!(closingSpeed>0) || !PxIsFinite(closingSpeed))return 2;
    _putenv_s("PHYSX_AVBD_COMMON_STEP","budget");_putenv_s("PHYSX_AVBD_LOCAL_ITERATIONS","4");
    _putenv_s("PHYSX_NP_GEOMETRY_CERTIFICATES","2");AvbdProbeFoundation foundation;
    auto* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation.foundation,PxTolerancesScale());PxInitExtensions(*physics,NULL);
    auto* dispatcher=PxDefaultCpuDispatcherCreate(workers);PxSceneDesc desc(physics->getTolerancesScale());
    desc.solverType=PxSolverType::eAVBD;desc.gravity=PxVec3(0);desc.cpuDispatcher=dispatcher;desc.filterShader=PxDefaultSimulationFilterShader;
    if(plane)desc.gravity=PxVec3(argc>4?PxReal(std::atof(argv[4])):0.f,-9.81f,0);
    auto* scene=physics->createScene(desc);
    if(plane)
    {
        const bool passed=slideOnPlane(*physics,*scene,argc>3?PxReal(std::atof(argv[3])):0.f);
        scene->release();dispatcher->release();PxCloseExtensions();physics->release();return passed?0:1;
    }
    auto* mesh=VolumeAvbdSnippet::cookCube(*physics,PxVec3(0),2.f);
    auto* material=physics->createDeformableVolumeMaterial(2.e4f,.3f,0,0);
    material->setMaterialModel(PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);
    PxDeformableVolume* body[2]={};
    for(PxU32 side=0;side<2;++side)
    {
        body[side]=physics->createDeformableVolume();
        auto* shape=physics->createShape(PxTetrahedronMeshGeometry(mesh->getCollisionMesh()),&material,1,true,PxShapeFlag::eSIMULATION_SHAPE);
        shape->setContactOffset(.02f);shape->setRestOffset(0);
        if(!body[side]->attachShape(*shape) || !body[side]->attachSimulationMesh(*mesh->getSimulationMesh(),*mesh->getDeformableVolumeAuxData()))return 2;
        shape->release();body[side]->setLinearDamping(0);body[side]->setSolverIterationCounts(4,1);
        body[side]->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION,true);
        PxVec4 *p=NULL,*v=NULL,*c=NULL,*r=NULL;PxDeformableVolumeExt::allocateAndInitializeHostMirror(*body[side],NULL,p,v,c,r);
        PxDeformableVolumeExt::updateMass(*body[side],100.f,50.f,p);
        for(PxU32 i=0;i<mesh->getSimulationMesh()->getNbVertices();++i)
        {p[i]=PxVec4(p[i].getXYZ()+PxVec3(side?3:impactX,side?2:height,2),p[i].w);v[i]=PxVec4(commonSpeed+(side?-.5f:.5f)*relativeSlide,side?0.f:-closingSpeed,0,p[i].w);}
        PxDeformableVolumeExt::updateEmbeddedCollisionMesh(*body[side],p,c);
        PxDeformableVolumeExt::copyToDevice(*body[side],PxDeformableVolumeDataFlag::eALL,p,v,c,r);
        VolumeAvbdSnippet::hostFree(p);VolumeAvbdSnippet::hostFree(v);VolumeAvbdSnippet::hostFree(c);VolumeAvbdSnippet::hostFree(r);
        if(!scene->addActor(*body[side]))return 2;
    }
    const auto vec=[](const PxVec3& p){return PxVec3d(p.x,p.y,p.z);};
    double mass=0,kinetic=0,elastic=0,minJ=1,receiverSpeed=0;PxVec3d momentum(0),center(0),angular(0),comVelocity[2];
    bool finite=true;
    const auto observe=[&]()
    {
        mass=kinetic=elastic=0;momentum=center=angular=PxVec3d(0);
        for(PxU32 side=0;side<2;++side)
        {
            PxVec3d actorMomentum(0);double actorMass=0;
            const auto* p=body[side]->getSimPositionInvMassBufferD();const auto* v=body[side]->getSimVelocityBufferD();
            elastic+=avbdProbeVolumeEnergy(*mesh->getSimulationMesh(),p,2.e4,.3,minJ);
            for(PxU32 i=0;i<mesh->getSimulationMesh()->getNbVertices();++i)
            {
                const double m=1/double(p[i].w);mass+=m;center+=vec(p[i].getXYZ())*m;momentum+=vec(v[i].getXYZ())*m;
                actorMomentum+=vec(v[i].getXYZ())*m;actorMass+=m;
                kinetic+=.5*m*(vec(v[i].getXYZ())-PxVec3d(commonSpeed,0,0)).magnitudeSquared();
                angular+=(vec(p[i].getXYZ())-PxVec3d(3,2,2)).cross(vec(v[i].getXYZ()))*m;
            }
            comVelocity[side]=actorMomentum/actorMass;
        }
        center/=mass;receiverSpeed=-comVelocity[1].y;
        finite &= momentum.isFinite() && angular.isFinite() && center.isFinite() && std::isfinite(kinetic+elastic);
    };
    observe();const PxVec3d P0=momentum,X0=center,L0=angular;const double E0=kinetic;
    const double initialEnergy=kinetic+elastic;
    double maxEnergy=initialEnergy,maxMomentum=0,maxAngular=0,maxCom=0,maxKinetic=kinetic,maxReceiver=0,maxTangentError=0;PxU32 frames=0;
    AvbdProbeStepTimer timer;
    for(;frames<180;++frames)
    {
        timer.begin();if(!scene->simulate(1.f/60) || !scene->fetchResults(true))break;
        timer.end();observe();
        for(PxU32 s=0;s<2;++s)maxTangentError=PxMax(maxTangentError,std::abs(comVelocity[s].x-commonSpeed-(s?-.5:.5)*relativeSlide));
        maxMomentum=PxMax(maxMomentum,(momentum-P0).magnitude()/std::abs(P0.y));
        maxAngular=PxMax(maxAngular,(angular-L0).magnitude());
        maxCom=PxMax(maxCom,(center-X0-P0*(double(frames+1)*double(1.f/60)/mass)).magnitude());
        maxKinetic=PxMax(maxKinetic,kinetic);maxReceiver=PxMax(maxReceiver,receiverSpeed);
        maxEnergy=PxMax(maxEnergy,kinetic+elastic);
    }
    const double separatingSpeed=comVelocity[0].y-comVelocity[1].y;
    const bool pass=finite && minJ>0 && maxEnergy<1.05*initialEnergy && frames==180 && maxMomentum<.005 && maxCom<.001 && maxKinetic<1.05*E0 && maxReceiver>.1 &&
        separatingSpeed>.1*closingSpeed && (!(commonSpeed || relativeSlide) || maxTangentError<.01);
    std::printf("[SOFT_CONTACT_MOTION] commonSpeed=%.6g closingSpeed=%.6g relativeSlide=%.6g finalSlide=%.6g maxTangentError=%.6g separatingSpeed=%.6g\n",
        double(commonSpeed),double(closingSpeed),double(relativeSlide),comVelocity[0].x-comVelocity[1].x,maxTangentError,separatingSpeed);
    double mean,p95,cpuMs;timer.summarize(mean,p95,cpuMs);
    std::printf("[NODAL_SOFT_CONTACT] workers=%u frames=%u iterations=4 height=%.6f impactX=%.6f maxMomentumError=%.9g maxAngularAbs=%.9g maxComError=%.9g initialKinetic=%.9g maxKinetic=%.9g peakReceiverSpeed=%.9g initialEnergy=%.9g peakEnergy=%.9g minJ=%.9g meanMs=%.6f p95Ms=%.6f cpuMs=%.6f status=%s\n",
        workers,frames,height,impactX,maxMomentum,maxAngular,maxCom,E0,maxKinetic,maxReceiver,initialEnergy,maxEnergy,minJ,mean,p95,cpuMs,pass?"PASS":"FAIL");
    for(auto* b:body)b->release();mesh->release();material->release();scene->release();dispatcher->release();PxCloseExtensions();physics->release();return pass?0:1;
}
