// Whole-step observer through the public Host API, with strain authored BEFORE
// the first simulate. This must not become the material's stress-free shape.
#include "PxPhysicsAPI.h"
#include "avbd_probe_foundation.h"
#include "avbd_probe_volume_energy.h"
#include "../physx/snippets/snippetdeformablevolumeavbd/SnippetDeformableVolumeAVBDCommon.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
using namespace physx;

static bool checkAttachment(PxPhysics& physics,PxScene& scene,PxDeformableVolume& volume,bool element)
{
    auto* p=volume.getSimPositionInvMassBufferD();
    const auto* mesh=volume.getSimulationMesh();
    const auto* tets=mesh->getTetrahedrons();
    const bool small=mesh->getTetrahedronMeshFlags().isSet(PxTetrahedronMeshFlag::e16_BIT_INDICES);
    const PxU32 id=0;
    const PxVec4 w(.1f,.2f,.3f,.4f);
    const auto point=[&](){
        if(!element) return p[id].getXYZ();
        PxVec3 x(0);
        for(PxU32 k=0;k<4;++k) x+=w[k]*p[small?static_cast<const PxU16*>(tets)[k]:static_cast<const PxU32*>(tets)[k]].getXYZ();
        return x;
    };
    const PxVec3 offset(0,.35f,0);
    auto* body=physics.createRigidDynamic(PxTransform(point()-offset));
    body->setMass(80); body->setMassSpaceInertiaTensor(PxVec3(15));
    body->setAngularDamping(0); body->setLinearDamping(0); body->setSleepThreshold(0);
    body->setCMassLocalPose(PxTransform(PxVec3(.07f,0,0)));
    if(!scene.addActor(*body)) return false;
    const PxVec4 local(offset,0);
    PxDeformableAttachmentData d;
    const PxU32 s=element?1:0,r=1-s;
    d.actor[s]=&volume; d.actor[r]=body;
    d.type[s]=element?PxDeformableAttachmentTargetType::eTETRAHEDRON:PxDeformableAttachmentTargetType::eVERTEX;
    d.type[r]=PxDeformableAttachmentTargetType::eRIGID;
    d.indices[s]={&id,sizeof(id),1}; d.coords[r]={&local,sizeof(local),1};
    if(element) d.coords[s]={&w,sizeof(w),1};
    auto* attachment=physics.createDeformableAttachment(d);
    if(!attachment) return false;
    body->setLinearVelocity(PxVec3(1,0,0));
    const PxVec3 incoming=body->getMass()*body->getLinearVelocity();
    PxReal gap=0,drift=0,softImpulse=0,spin=0;
    for(PxU32 i=0;i<240;++i)
    {
        if(!scene.simulate(1.f/60) || !scene.fetchResults(true)) return false;
        const auto* v=volume.getSimVelocityBufferD();
        PxVec3 soft(0);
        for(PxU32 n=0;n<mesh->getNbVertices();++n) soft+=v[n].getXYZ()/p[n].w;
        gap=PxMax(gap,(point()-body->getGlobalPose().transform(offset)).magnitude());
        drift=PxMax(drift,(soft+body->getMass()*body->getLinearVelocity()-incoming).magnitude()/incoming.magnitude());
        softImpulse=PxMax(softImpulse,soft.x);
        spin=PxMax(spin,body->getAngularVelocity().magnitude());
    }
    scene.removeActor(*body);
    const PxVec3 shift(.025f,0,0);
    attachment->updatePose(PxTransform(shift));
    if(!scene.addActor(*body)) return false;
    for(PxU32 i=0;i<40;++i) if(!scene.simulate(1.f/60) || !scene.fetchResults(true)) return false;
    const PxReal reentryGap=(point()-body->getGlobalPose().transform(offset+shift)).magnitude();
    attachment->release();
    const PxVec3 released(0,1,0);
    body->setLinearVelocity(released); body->setAngularVelocity(PxVec3(0));
    for(PxU32 i=0;i<30;++i) if(!scene.simulate(1.f/60) || !scene.fetchResults(true)) return false;
    const PxReal freeError=(body->getLinearVelocity()-released).magnitude();
    bool pass=gap<.02f && drift<.01f && softImpulse>5 && spin>.1f && freeError<.001f && reentryGap<.005f &&
        (point()-body->getGlobalPose().transform(offset)).magnitude()>.1f;
    attachment=physics.createDeformableAttachment(d);
    if(!attachment) return false;
    body->release();
    PxActor* actors[2]={};
    attachment->getActors(actors[0],actors[1]);
    pass &= actors[r]==NULL && actors[s]==&volume;
    attachment->release();
    pass &= scene.simulate(1.f/60) && scene.fetchResults(true);
    std::printf("[VOLUME_ATTACHMENT] type=%s peakGap=%g relativeMomentum=%g softImpulse=%g spin=%g reentryGap=%g releasedVelocityError=%g status=%s\n",
        element?"tet":"vertex",double(gap),double(drift),double(softImpulse),double(spin),double(reentryGap),double(freeError),pass?"PASS":"FAIL");
    return pass;
}

int main(int argc,char** argv)
{
    const PxU32 workers=argc>1?PxU32(std::atoi(argv[1])):4;
    const char* mode=argc>2?argv[2]:"strain";
    const bool plane=std::strcmp(mode,"plane-rejection")==0,sphere=std::strcmp(mode,"sphere-rejection")==0;
    const bool self=std::strcmp(mode,"self-rejection")==0;
    const bool floor=std::strcmp(mode,"plane-contact")==0,reaction=std::strcmp(mode,"rigid-reaction")==0;
    const bool compressed=std::strcmp(mode,"compression-release")==0;
    const bool element=std::strcmp(mode,"attachment-tet")==0,attached=element || std::strcmp(mode,"attachment-vertex")==0;
    const bool strained=compressed || std::strcmp(mode,"strain")==0,stationary=std::strcmp(mode,"rest")==0 || plane || sphere || self || floor || attached;
    if(!strained && !stationary && !reaction && std::strcmp(mode,"flight"))return 2;
    _putenv_s("PHYSX_AVBD_COMMON_STEP","budget");_putenv_s("PHYSX_AVBD_LOCAL_ITERATIONS","4");
    _putenv_s("PHYSX_NP_GEOMETRY_CERTIFICATES","2");
    AvbdProbeFoundation foundation;
    auto* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation.foundation,PxTolerancesScale());
    PxInitExtensions(*physics,NULL);
    auto* dispatcher=PxDefaultCpuDispatcherCreate(workers);
    PxSceneDesc desc(physics->getTolerancesScale());desc.solverType=PxSolverType::eAVBD;
    desc.gravity=floor?PxVec3(0,-9.81f,0):PxVec3(0);desc.cpuDispatcher=dispatcher;desc.filterShader=PxDefaultSimulationFilterShader;
    auto* scene=physics->createScene(desc);
    auto* mesh=VolumeAvbdSnippet::cookCube(*physics,PxVec3(0),2.f);
    auto* material=physics->createDeformableVolumeMaterial(2.e4f,.3f,0.f,0.f);
    material->setMaterialModel(PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);
    auto* volume=physics->createDeformableVolume();
    auto* shape=physics->createShape(PxTetrahedronMeshGeometry(mesh->getCollisionMesh()),&material,1,true,PxShapeFlag::eSIMULATION_SHAPE);
    if(!shape || !volume->attachShape(*shape) || !volume->attachSimulationMesh(*mesh->getSimulationMesh(),*mesh->getDeformableVolumeAuxData()))return 2;
    shape->release();volume->setLinearDamping(0);
    volume->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION,!self);
    volume->setSolverIterationCounts(4,1);
    PxVec4 *p=NULL,*v=NULL,*collision=NULL,*rest=NULL;
    PxDeformableVolumeExt::allocateAndInitializeHostMirror(*volume,NULL,p,v,collision,rest);
    PxDeformableVolumeExt::updateMass(*volume,100.f,50.f,p);
    const auto* simulation=mesh->getSimulationMesh();const PxU32 count=simulation->getNbVertices();
    const PxVec3 velocity=(stationary || compressed)?PxVec3(0):(reaction?PxVec3(0,-1,0):PxVec3(.5f,-.25f,.2f));
    const PxQuat rotation=(floor || reaction)?PxQuat(PxIdentity):PxQuat(.51f,PxVec3(1,2,3).getNormalized());
    PxArray<PxVec3> initial;
    for(PxU32 i=0;i<count;++i)
    {
        PxVec3 x=p[i].getXYZ();
        if(compressed)x.y*=.85f;
        else if(strained){x.x*=1.05f;x.z*=.98f;x.y+=.08f*x.x*x.z;}
        x=rotation.rotate(x)+PxVec3(3,4,2);p[i]=PxVec4(x,p[i].w);v[i]=PxVec4(velocity,p[i].w);initial.pushBack(x);
    }
    PxDeformableVolumeExt::updateEmbeddedCollisionMesh(*volume,p,collision);
    PxDeformableVolumeExt::copyToDevice(*volume,PxDeformableVolumeDataFlag::eALL,p,v,collision,rest);
    VolumeAvbdSnippet::hostFree(p);VolumeAvbdSnippet::hostFree(v);VolumeAvbdSnippet::hostFree(collision);VolumeAvbdSnippet::hostFree(rest);
    // Deliberately intersect unsupported Host geometry. The experimental
    // route must reject it, not silently publish a contact-free falling body.
    if(plane || sphere)
    {
        auto* rigidMaterial=physics->createMaterial(0,0,0);
        PxRigidStatic* obstacle=plane?PxCreatePlane(*physics,PxPlane(0,1,0,-4),*rigidMaterial):
            PxCreateStatic(*physics,PxTransform(PxVec3(3,4,2)),PxSphereGeometry(1),*rigidMaterial);
        if(!obstacle || !scene->addActor(*obstacle))return 2;rigidMaterial->release();
    }
    PxRigidDynamic* receiver=NULL;
    if(floor || reaction)
    {
        auto* rigidMaterial=physics->createMaterial(0,0,0);
        if(floor)
        {auto* obstacle=PxCreatePlane(*physics,PxPlane(0,1,0,-2.95f),*rigidMaterial);scene->addActor(*obstacle);}
        else
        {
            receiver=PxCreateDynamic(*physics,PxTransform(PxVec3(3,2,2)),PxBoxGeometry(1,.8f,1),*rigidMaterial,100.f);
            receiver->setLinearDamping(0);receiver->setAngularDamping(0);receiver->setSleepThreshold(0);
            receiver->setSolverIterationCounts(4,1);scene->addActor(*receiver);
        }
        rigidMaterial->release();
    }
    if(!scene->addActor(*volume))return 2;
    if(attached)
    {
        const bool ok=checkAttachment(*physics,*scene,*volume,element);
        volume->release();mesh->release();material->release();scene->release();dispatcher->release();PxCloseExtensions();physics->release();
        return ok?0:1;
    }
    p=volume->getSimPositionInvMassBufferD();v=volume->getSimVelocityBufferD();
    const auto vec=[](const PxVec3& q){return PxVec3d(q.x,q.y,q.z);};
    const auto axis=rotation.rotate(PxVec3(0,1,0));
    double maxHeightRatio=0,maxKinetic=0;
    const auto energy=[&](double& minJ)
    {
        double sum=avbdProbeVolumeEnergy(*simulation,p,2.e4,.3,minJ);
        for(PxU32 i=0;i<count;++i)sum+=.5*vec(v[i].getXYZ()).magnitudeSquared()/double(p[i].w);
        return sum;
    };
    double mass=0,minJ=1,maximumShapeMotion=0,maximumComError=0,maximumMomentumError=0;
    PxVec3d center0(0);for(PxU32 i=0;i<count;++i){const double m=1/double(p[i].w);mass+=m;center0+=vec(p[i].getXYZ())*m;}center0/=mass;
    const double softMass=mass,rigidMass=receiver?receiver->getMass():0;
    const auto totalCenter0=(center0*mass+(receiver?vec(receiver->getGlobalPose().p)*rigidMass:PxVec3d(0)))/(mass+rigidMass);
    double peakRigidSpeed=0,minFloorGap=1.e10;
    const double initialEnergy=energy(minJ);double peakEnergy=initialEnergy;bool finite=true;
    for(PxU32 frame=1;frame<=180;++frame)
    {
        if(!scene->simulate(1.f/60) || !scene->fetchResults(true)){std::printf("[NODAL_VOLUME_STEP] mode=%s workers=%u frame=%u status=SIMULATION_FAILED\n",mode,workers,frame);return 2;}
        PxVec3d center(0),momentum(0);const auto translation=vec(velocity)*(frame*double(1.f/60));
        for(PxU32 i=0;i<count;++i)
        {
            finite &= p[i].isFinite() && v[i].isFinite();const double m=1/double(p[i].w);
            center+=vec(p[i].getXYZ())*m;momentum+=vec(v[i].getXYZ())*m;
            maximumShapeMotion=PxMax(maximumShapeMotion,(vec(p[i].getXYZ())-vec(initial[i])-translation).magnitude());
        }
        if(compressed)
        {
            double lo=PX_MAX_F64,hi=-PX_MAX_F64,kinetic=0;
            for(PxU32 i=0;i<count;++i)
            {
                const double y=axis.dot(p[i].getXYZ());lo=PxMin(lo,y);hi=PxMax(hi,y);
                kinetic+=.5*vec(v[i].getXYZ()).magnitudeSquared()/double(p[i].w);
            }
            maxHeightRatio=PxMax(maxHeightRatio,(hi-lo)/2.);maxKinetic=PxMax(maxKinetic,kinetic);
        }
        if(receiver)
        {
            center+=vec(receiver->getGlobalPose().p)*rigidMass;
            momentum+=vec(receiver->getLinearVelocity())*rigidMass;
            peakRigidSpeed=PxMax(peakRigidSpeed,double(receiver->getLinearVelocity().magnitude()));
            maximumComError=PxMax(maximumComError,(center/(mass+rigidMass)-totalCenter0-translation*(mass/(mass+rigidMass))).magnitude());
        }
        else maximumComError=PxMax(maximumComError,(center/mass-center0-translation).magnitude());
        if(floor)for(PxU32 i=0;i<count;++i)minFloorGap=PxMin(minFloorGap,double(p[i].y)-2.95);
        maximumMomentumError=PxMax(maximumMomentumError,(momentum-vec(velocity)*softMass).magnitude()/(softMass*PxMax(.1,double(velocity.magnitude()))));
        peakEnergy=PxMax(peakEnergy,energy(minJ)+(receiver?.5*rigidMass*receiver->getLinearVelocity().magnitudeSquared():0));
    }
    if(compressed)std::printf("[COMPRESSION_RELEASE] initialHeightRatio=0.85 peakHeightRatio=%.8g peakKineticFraction=%.8g\n",
        maxHeightRatio,maxKinetic/initialEnergy);
    const bool recoil=!compressed || (maxHeightRatio>1.02 && maxKinetic>.25*initialEnergy);
    const bool pass=floor ? finite && minJ>0 && minFloorGap>-.001 : recoil && finite && minJ>0 && maximumComError<.001 && maximumMomentumError<.005 &&
        (reaction?peakRigidSpeed>.1:(strained?maximumShapeMotion>.01:maximumShapeMotion<.001)) && peakEnergy<=1.05*initialEnergy+.001;
    std::printf("[NODAL_VOLUME_STEP] mode=%s workers=%u nodes=%u frames=180 iterations=4 maxShapeMotion=%.9g maxComError=%.9g maxMomentumError=%.9g minJ=%.9g initialEnergy=%.9g peakEnergy=%.9g status=%s\n",
        mode,workers,count,maximumShapeMotion,maximumComError,maximumMomentumError,minJ,initialEnergy,peakEnergy,pass?"PASS":"FAIL");
    if(floor || reaction)std::printf("[NODAL_CONTACT_OBSERVATION] peakRigidSpeed=%.9g minFloorGap=%.9g\n",peakRigidSpeed,minFloorGap);
    volume->release();mesh->release();material->release();scene->release();dispatcher->release();PxCloseExtensions();physics->release();
    return pass?0:1;
}
