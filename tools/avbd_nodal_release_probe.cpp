// Independent whole-step observer: a stretched free sheet must not acquire
// momentum from its internal material. No exact trajectory comparison.
#include "PxPhysicsAPI.h"
#include "avbd_probe_foundation.h"
#include "avbd_probe_sheet_energy.h"
#include "avbd_probe_step_timer.h"
#include "../physx/snippets/snippetdeformablesurfaceavbd/SnippetDeformableSurfaceAVBDCommon.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
using namespace physx;
struct SheetProbeFoundation : AvbdProbeFoundation
{
    PxU32 completedFrames=0;
    double peakMomentum=0,peakBallistic=0,peakEnergy=0;
    void reportError(PxErrorCode::Enum code,const char* message,const char* file,int line) override
    {
        std::fprintf(stderr,"[NODAL_RELEASE] completedFrames=%u maxMomentumError=%.9g maxBallisticError=%.9g peakEnergy=%.9g status=SIMULATION_FAILED\n",
            completedFrames,peakMomentum,peakBallistic,peakEnergy);
        AvbdProbeFoundation::reportError(code,message,file,line);
    }
};
int main(int argc,char** argv)
{
    const PxU32 workers=argc>1?PxU32(std::atoi(argv[1])):4;
    const bool frictionless=argc>2 && std::strcmp(argv[2],"impact-frictionless")==0;
    const bool impact=frictionless || (argc>2 && std::strcmp(argv[2],"impact")==0);
    const bool reset=argc>2 && std::strcmp(argv[2],"reset")==0;
    const bool trace=std::getenv("PHYSX_AVBD_MOMENTUM_TRACE")!=NULL;
    _putenv_s("PHYSX_AVBD_COMMON_STEP","budget");_putenv_s("PHYSX_AVBD_LOCAL_ITERATIONS","4");
    _putenv_s("PHYSX_NP_GEOMETRY_CERTIFICATES","2");
    SheetProbeFoundation foundation;
    PxPhysics* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation.foundation,PxTolerancesScale());
    auto* dispatcher=PxDefaultCpuDispatcherCreate(workers);
    PxSceneDesc desc(physics->getTolerancesScale());desc.solverType=PxSolverType::eAVBD;
    desc.gravity=PxVec3(0);desc.cpuDispatcher=dispatcher;desc.filterShader=PxDefaultSimulationFilterShader;
    auto* scene=physics->createScene(desc);PxTriangleMesh* mesh=NULL;PxDeformableSurfaceMaterial* material=NULL;
    auto* surface=SurfaceAvbdSnippet::createDrapedSurface(*physics,*scene,mesh,material,0,SurfaceAvbdSnippet::kReferenceDrapeGrid);
    if(!surface)return 2;
    surface->setSolverIterationCounts(4,1);
    surface->setLinearDamping(0);surface->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION,true);
    const PxU32 n=mesh->getNbVertices();auto* p=surface->getPositionInvMassBufferH();auto* v=surface->getVelocityBufferH();
    const auto* rest=surface->getRestPositionBufferH();const PxVec3 velocity(.5f,-.25f,.2f);
    PxRigidDynamic* box=NULL;
    if(impact)
    {
        const PxReal friction=frictionless?0.f:.6f;
        auto* rigidMaterial=physics->createMaterial(friction,friction,0);
        const PxReal x=argc>3?PxReal(std::atof(argv[3])):.1f,z=argc>4?PxReal(std::atof(argv[4])):.1f;
        box=PxCreateDynamic(*physics,PxTransform(PxVec3(x,5,z)),PxBoxGeometry(.3f,.3f,.3f),
            *rigidMaterial,20.f/(.6f*.6f*.6f));rigidMaterial->release();
        if(!box)return 2;
        PxShape* shape=NULL;box->getShapes(&shape,1);shape->setContactOffset(.04f);shape->setRestOffset(0);
        box->setLinearDamping(0);box->setAngularDamping(0);box->setLinearVelocity(PxVec3(0,-2,0));
        box->setSolverIterationCounts(4,1);
        material->setDynamicFriction(friction);
        if(!scene->addActor(*box))return 2;
    }
    PxDeformableSurfaceDataFlags dirty(PxDeformableSurfaceDataFlag::ePOSITION_INVMASS);
    dirty|=PxDeformableSurfaceDataFlag::eVELOCITY;
    if(reset)
    {
        // Warm the material on a different strain before overwriting host state.
        for(PxU32 i=0;i<n;++i)p[i].z*=1.04f;
        surface->markDirty(dirty);
        for(PxU32 i=0;i<12;++i)if(!scene->simulate(1.f/60) || !scene->fetchResults(true))return 2;
    }
    if(!impact)for(PxU32 i=0;i<n;++i)
    {
        p[i]=PxVec4(rest[i].getXYZ(),p[i].w);p[i].x*=1.05f;p[i].z*=.98f;
        p[i].y+=.15f*PxSin(1.7f*(p[i].x+.4f))*PxCos(1.1f*(p[i].z-.3f));v[i]=PxVec4(velocity,0);
    }
    surface->markDirty(dirty);
    const auto energy=[&]()
    {
        double sum=avbdProbeSheetEnergy(*mesh,rest,p,v);
        if(box)
        {
            const auto pose=box->getGlobalPose()*box->getCMassLocalPose();
            const auto w=pose.q.rotateInv(box->getAngularVelocity());
            sum+=.5*box->getMass()*box->getLinearVelocity().magnitudeSquared()+
                .5*w.dot(box->getMassSpaceInertiaTensor().multiply(w));
        }
        return sum;
    };
    const auto vec=[](const PxVec3& x){return PxVec3d(x.x,x.y,x.z);};
    double sheetMass=0;PxVec3d center0(0);
    for(PxU32 i=0;i<n;++i){const double mass=1./p[i].w;sheetMass+=mass;center0+=vec(p[i].getXYZ())*mass;}
    const double totalMass=sheetMass+(box?box->getMass():0);
    if(box)center0+=vec(box->getGlobalPose().p)*double(box->getMass());
    center0/=totalMass;
    const PxVec3d target=box?vec(box->getLinearVelocity())*double(box->getMass()):vec(velocity)*sheetMass;
    const PxVec3d initialAngular=box?(vec(box->getGlobalPose().p)-center0).cross(target):PxVec3d(0);
    double maxAngularDrift=0;
    const double initialEnergy=energy();foundation.peakEnergy=initialEnergy;
    const double initialStrain=impact?0:initialEnergy-.5*sheetMass*vec(velocity).magnitudeSquared();
    double minStrain=initialStrain;
    const double dt=double(1.f/60);bool finite=true;double maxSheetSpeed=0;
    AvbdProbeStepTimer timer;
    for(PxU32 frame=1;frame<=180;++frame)
    {
        timer.begin();
        if(!scene->simulate(PxReal(dt)) || !scene->fetchResults(true))
        {std::printf("[NODAL_RELEASE] mode=%s workers=%u completedFrames=%u status=SIMULATION_FAILED\n",impact?"impact":"release",workers,frame-1);return 2;}
        timer.end();
        PxVec3d center(0),momentum(0),angular(0);double sheetKinetic=0;
        for(PxU32 i=0;i<n;++i)
        {
            const double mass=1./p[i].w;
            finite &= p[i].isFinite() && v[i].isFinite();center+=vec(p[i].getXYZ())*mass;momentum+=vec(v[i].getXYZ())*mass;
            sheetKinetic+=.5*mass*vec(v[i].getXYZ()).magnitudeSquared();
            angular+=(vec(p[i].getXYZ())-center0).cross(vec(v[i].getXYZ()))*mass;
        }
        maxSheetSpeed=PxMax(maxSheetSpeed,momentum.magnitude()/sheetMass);
        if(box)
        {
            const auto pose=box->getGlobalPose()*box->getCMassLocalPose();
            angular+=(vec(pose.p)-center0).cross(vec(box->getLinearVelocity()))*double(box->getMass())+
                vec(pose.q.rotate(box->getMassSpaceInertiaTensor().multiply(pose.q.rotateInv(box->getAngularVelocity()))));
            center+=vec(box->getGlobalPose().p)*double(box->getMass());
            momentum+=vec(box->getLinearVelocity())*double(box->getMass());
        }
        maxAngularDrift=PxMax(maxAngularDrift,(angular-initialAngular).magnitude());
        finite &= angular.isFinite();
        foundation.peakMomentum=PxMax(foundation.peakMomentum,(momentum-target).magnitude()/target.magnitude());
        foundation.peakBallistic=PxMax(foundation.peakBallistic,(center/totalMass-center0-target*(frame*dt/totalMass)).magnitude());
        const double currentEnergy=energy();
        if(!impact)minStrain=PxMin(minStrain,currentEnergy-sheetKinetic);
        if(trace)std::printf("[SHEET_MOMENTUM] frame=%u x=%.9g y=%.9g z=%.9g boxPy=%.9g energy=%.9g\n",
            frame,momentum.x,momentum.y,momentum.z,box?double(box->getLinearVelocity().y)*box->getMass():0,currentEnergy);
        finite &= momentum.isFinite() && center.isFinite() && std::isfinite(currentEnergy);
        foundation.peakEnergy=PxMax(foundation.peakEnergy,currentEnergy);
        foundation.completedFrames=frame;
    }
    const bool pass=finite && foundation.peakMomentum<=.005 && foundation.peakBallistic<=.001 &&
        foundation.peakEnergy<=1.05*initialEnergy && (impact?maxSheetSpeed>.001:(initialStrain>0 && minStrain<.5*initialStrain));
    double mean,p95,cpu;timer.summarize(mean,p95,cpu);
    std::printf("[SHEET_ANGULAR_MOMENTUM] maxDrift=%.9g units=kg*m^2/s\n",maxAngularDrift);
    std::printf("[NODAL_RELEASE] mode=%s workers=%u nodes=%u iterations=4 frames=180 maxMomentumError=%.9g maxBallisticError=%.9g initialEnergy=%.9g peakEnergy=%.9g initialStrainEnergy=%.9g minStrainEnergy=%.9g peakSheetSpeed=%.9g meanMs=%.6f p95Ms=%.6f cpuMs=%.6f status=%s\n",
        frictionless?"impact-frictionless":impact?"impact":reset?"reset":"release",workers,n,foundation.peakMomentum,foundation.peakBallistic,
        initialEnergy,foundation.peakEnergy,initialStrain,minStrain,maxSheetSpeed,mean,p95,cpu,pass?"PASS":"FAIL");
    if(box)box->release();
    surface->release();mesh->release();material->release();scene->release();dispatcher->release();physics->release();return pass?0:1;
}
