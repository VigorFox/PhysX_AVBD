// Stress the 256-row work-item boundary with one mesh vertex shared by 288 faces.
// Radius and perturbation are in metres. This is a storage/dispatch regression.
#include "PxPhysicsAPI.h"
#include "avbd_probe_foundation.h"
#include <cstdio>
#include <cstring>

using namespace physx;

int main(int argc,char** argv)
{
    const bool bending=argc>1 && std::strcmp(argv[1],"bending")==0;
    const PxU32 workers=argc>2?PxU32(std::atoi(argv[2])):8;
    _putenv_s("PHYSX_AVBD_COMMON_STEP","budget");
    _putenv_s("PHYSX_AVBD_LOCAL_ITERATIONS","4");
    AvbdProbeFoundation foundation;
    PxPhysics* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation.foundation,PxTolerancesScale());
    auto* dispatcher=PxDefaultCpuDispatcherCreate(workers);
    PxSceneDesc desc(physics->getTolerancesScale());
    desc.solverType=PxSolverType::eAVBD;desc.gravity=PxVec3(0);
    desc.cpuDispatcher=dispatcher;desc.filterShader=PxDefaultSimulationFilterShader;
    auto* scene=physics->createScene(desc);
    const PxU32 faces=288;
    PxVec3 vertices[faces+1];PxU32 indices[faces][3];
    vertices[0]=PxVec3(0,2,0);
    for(PxU32 i=0;i<faces;++i)
    {
        const PxReal a=PxTwoPi*PxReal(i)/faces;
        vertices[1+i]=vertices[0]+1.5f*PxVec3(PxCos(a),0,PxSin(a));
        indices[i][0]=0;indices[i][1]=1+(i+1)%faces;indices[i][2]=1+i;
    }
    PxTriangleMeshDesc meshDesc;
    meshDesc.points.count=faces+1;meshDesc.points.data=vertices;meshDesc.points.stride=sizeof(PxVec3);
    meshDesc.triangles.count=faces;meshDesc.triangles.data=indices;meshDesc.triangles.stride=sizeof(indices[0]);
    PxCookingParams cooking(physics->getTolerancesScale());cooking.buildTriangleAdjacencies=true;
    auto* mesh=PxCreateTriangleMesh(cooking,meshDesc,physics->getPhysicsInsertionCallback());
    auto* mat=physics->createDeformableSurfaceMaterial(1.e5f,.3f,0,.02f,bending?.01f:0.f);
    auto* surf=physics->createDeformableSurface(PxDeformableSurfaceBackend::eCPU_AVBD);
    if(!scene || !mesh || !mat || !surf)return 2;
    auto* shape=physics->createShape(PxTriangleMeshGeometry(mesh),&mat,1,true,PxShapeFlag::eSIMULATION_SHAPE);
    if(!shape || !surf->attachShape(*shape))return 2;
    shape->release();
    auto* p=surf->getPositionInvMassBufferH();auto* v=surf->getVelocityBufferH();auto* rest=surf->getRestPositionBufferH();
    if(!p || !v || !rest)return 2;
    PxU32 center=PX_MAX_U32;
    for(PxU32 i=0;i<mesh->getNbVertices();++i)
    {
        const PxVec3 x=mesh->getVertices()[i];const bool free=(x-vertices[0]).magnitudeSquared()<1.e-8f;
        p[i]=PxVec4(x,free?1.f:0.f);rest[i]=PxVec4(x,0);v[i]=PxVec4(0.f);
        if(free){center=i;p[i].x+=.02f;}
    }
    if(center==PX_MAX_U32 || mesh->getNbTriangles()!=faces)return 2;
    surf->setSolverIterationCounts(4,1);surf->setLinearDamping(0);surf->setSleepThreshold(0);
    surf->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION,true);
    surf->markDirty(PxDeformableSurfaceDataFlag::eALL);
    if(!scene->addActor(*surf))return 2;
    PxReal best=.02f,pinError=0,peak=0;bool valid=true;PxU32 frames=0;
    for(;frames<12;++frames)
    {
        if(!scene->simulate(1.f/60) || !scene->fetchResults(true))break;
        for(PxU32 i=0;i<mesh->getNbVertices();++i)
        {
            valid &= p[i].isFinite() && v[i].isFinite();
            const PxReal error=(p[i].getXYZ()-rest[i].getXYZ()).magnitude();
            if(i==center){best=PxMin(best,error);peak=PxMax(peak,error);}
            else pinError=PxMax(pinError,error);
        }
    }
    const bool pass=valid && frames==12 && best<.019f && peak<.021f && pinError<1.e-6f;
    std::printf("[WIDE_STAR] material=%s workers=%u faces=%u frames=%u best=%g peak=%g pinError=%g status=%s\n",
        bending?"bending":"membrane",workers,faces,frames,double(best),double(peak),double(pinError),pass?"PASS":"FAIL");
    surf->release();mesh->release();mat->release();scene->release();dispatcher->release();physics->release();
    return pass?0:1;
}
