// Drive a membrane vertex across the opposite edge of the last cooked face.
// The second epoch resets the same actor to rest, with no rejected rows.
#include "avbd_probe_foundation.h"
#include <cstring>
#include "../physx/snippets/snippetdeformablesurfaceavbd/SnippetDeformableSurfaceAVBDCommon.h"

using namespace physx;

int main(int argc,char** argv)
{
    const PxU32 workers=argc>1?PxU32(std::atoi(argv[1])):8;
    const PxReal dt=1.f/60.f;
    AvbdProbeFoundation foundation;
    auto* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation.foundation,PxTolerancesScale());
    auto* dispatcher=PxDefaultCpuDispatcherCreate(workers);
    PxSceneDesc desc(physics->getTolerancesScale());
    desc.solverType=PxSolverType::eAVBD;
    desc.gravity=PxVec3(0);
    desc.cpuDispatcher=dispatcher;
    desc.filterShader=PxDefaultSimulationFilterShader;
    auto* scene=physics->createScene(desc);
    PxTriangleMesh* mesh=NULL;
    PxDeformableSurfaceMaterial* material=NULL;
    if(!scene)return 2;
    auto* sheet=SurfaceAvbdSnippet::createDrapedSurface(*physics,*scene,mesh,material,0,{34,34,.1f});
    if(!sheet)return 2;
    sheet->setLinearDamping(0);
    sheet->setSleepThreshold(0);
    sheet->setSolverIterationCounts(4,1);
    sheet->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION,true);
    auto* p=sheet->getPositionInvMassBufferH();
    auto* v=sheet->getVelocityBufferH();
    auto* rest=sheet->getRestPositionBufferH();
    const auto indices=mesh->getTriangles();
    const bool small=mesh->getTriangleMeshFlags().isSet(PxTriangleMeshFlag::e16_BIT_INDICES);
    const auto vertex=[&](PxU32 i)->PxU32
    {return small?static_cast<const PxU16*>(indices)[i]:static_cast<const PxU32*>(indices)[i];};
    const PxU32 last=3*(mesh->getNbTriangles()-1), moving=vertex(last);
    const PxVec3 opposite=(rest[vertex(last+1)].getXYZ()+rest[vertex(last+2)].getXYZ())*.5f;
    bool pass=true;
    const bool reviseRest=argc>2 && std::strcmp(argv[2],"rest-revision")==0;
    for(PxU32 epoch=0;epoch<(reviseRest?4u:2u);++epoch)
    {
        for(PxU32 i=0;i<mesh->getNbVertices();++i)
        {
            if(epoch==2)rest[i].x+=.125f;
            p[i]=PxVec4(rest[i].getXYZ(),i==moving?1.f:0.f);
            v[i]=PxVec4(0.f);
        }
        if(epoch==0)v[moving]=PxVec4((opposite-rest[moving].getXYZ())*(1.25f/dt),0.f);
        PxDeformableSurfaceDataFlags dirty(PxDeformableSurfaceDataFlag::ePOSITION_INVMASS);
        dirty |= PxDeformableSurfaceDataFlag::eVELOCITY;
        if(epoch==2)dirty |= PxDeformableSurfaceDataFlag::eREST_POSITION;
        sheet->markDirty(dirty);
        // Every face has at most one moving corner, so signed area varies
        // linearly over the predictor. Its endpoint sign is an independent oracle.
        const auto area=[&](PxU32 face,bool predict)
        {
            PxVec3 x[3];
            for(PxU32 k=0;k<3;++k)
            {
                const PxU32 i=vertex(3*face+k);
                x[k]=p[i].getXYZ();
                if(predict)x[k]+=dt*v[i].getXYZ();
            }
            return (x[1]-x[0]).cross(x[2]-x[0]).y;
        };
        PxU32 crossing=0;
        for(PxU32 f=0;f<mesh->getNbTriangles();++f)
            crossing+=area(f,true)*area(f,false)<=0;
        const PxU32 lastFace=mesh->getNbTriangles()-1;
        pass &= (area(lastFace,true)*area(lastFace,false)<=0)==(epoch==0);
        if(!scene->simulate(dt) || !scene->fetchResults(true))return 1;
        PxReal minSigned=PX_MAX_F32;
        for(PxU32 f=0;f<mesh->getNbTriangles();++f)minSigned=PxMin(minSigned,area(f,false));
        for(PxU32 i=0;i<mesh->getNbVertices();++i)
            pass &= p[i].isFinite() && v[i].isFinite() && (i==moving || p[i].getXYZ()==rest[i].getXYZ());
        pass &= minSigned>0 && (epoch==0?crossing>0:crossing==0);
        std::printf("[WARM_ORACLE] epoch=%u faces=%u crossing=%u minSigned=%.8g status=%s\n",
            epoch,mesh->getNbTriangles(),crossing,double(minSigned),pass?"PASS":"FAIL");
    }
    sheet->release();mesh->release();material->release();scene->release();
    dispatcher->release();physics->release();
    return pass?0:1;
}
