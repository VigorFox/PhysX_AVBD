// Public API first-epoch NP input probe. Until response is implemented this
// MUST reject publication, after reporting a complete shared NP view.
#include "PxPhysicsAPI.h"
#include "avbd_probe_foundation.h"
#include "../physx/snippets/snippetdeformablesurfaceavbd/SnippetDeformableSurfaceAVBDCommon.h"
#include <cstdlib>
#include <cstring>
using namespace physx;
// Identical DCD triangles. Only the authored distance between the two
// disconnected patches changes. The runner inspects the published NP count.
static bool selfFilter(PxPhysics& physics,PxScene& scene,bool far)
{
    const PxReal restGap=far?.5f:.03f;
    PxVec3 vertices[6]={PxVec3(0),PxVec3(1,0,0),PxVec3(0,0,1),
        PxVec3(0,restGap,0),PxVec3(1,restGap,0),PxVec3(0,restGap,1)};
    PxU32 indices[6]={0,2,1,3,4,5};PxTriangleMeshDesc desc;
    desc.points.data=vertices;desc.points.count=6;desc.points.stride=sizeof(PxVec3);
    desc.triangles.data=indices;desc.triangles.count=2;desc.triangles.stride=3*sizeof(PxU32);
    PxCookingParams cook(physics.getTolerancesScale());cook.buildTriangleAdjacencies=true;
    auto* mesh=PxCreateTriangleMesh(cook,desc,physics.getPhysicsInsertionCallback());
    auto* material=physics.createDeformableSurfaceMaterial(1000,.3f,0,.01f,0);
    auto* cloth=physics.createDeformableSurface(PxDeformableSurfaceBackend::eCPU_AVBD);
    if(!mesh || !material || !cloth || mesh->getNbVertices()!=6)return false;
    auto* shape=physics.createShape(PxTriangleMeshGeometry(mesh),&material,1,true,PxShapeFlag::eSIMULATION_SHAPE);
    if(!shape)return false;
    shape->setContactOffset(.04f);shape->setRestOffset(.005f);
    if(!cloth->attachShape(*shape))return false;
    shape->release();cloth->setSolverIterationCounts(4,1);
    cloth->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION,false);
    cloth->setSelfCollisionFilterDistance(.1f);
    for(PxU32 i=0;i<6;++i)
    {
        PxVec3 p=mesh->getVertices()[i];cloth->getRestPositionBufferH()[i]=PxVec4(p,0);
        if(p.y>0)p.y=.03f;
        cloth->getPositionInvMassBufferH()[i]=PxVec4(p,1);
        cloth->getVelocityBufferH()[i]=PxVec4(0.f);
    }
    cloth->markDirty(PxDeformableSurfaceDataFlag::eALL);
    const bool ok=scene.addActor(*cloth) && scene.simulate(1.f/60) && scene.fetchResults(true);
    cloth->release();mesh->release();material->release();return ok;
}
int main(int argc,char** argv)
{
    const bool box=argc>1 && !std::strcmp(argv[1],"box");
    _putenv_s("PHYSX_AVBD_COMMON_STEP","budget");_putenv_s("PHYSX_AVBD_LOCAL_ITERATIONS","4");
    _putenv_s("PHYSX_NP_GEOMETRY_CERTIFICATES","2");
    AvbdProbeFoundation foundation;
    auto* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation.foundation,PxTolerancesScale());
    PxInitExtensions(*physics,NULL);auto* dispatcher=PxDefaultCpuDispatcherCreate(4);
    PxSceneDesc desc(physics->getTolerancesScale());desc.solverType=PxSolverType::eAVBD;
    desc.gravity=PxVec3(0);desc.cpuDispatcher=dispatcher;desc.filterShader=PxDefaultSimulationFilterShader;
    auto* scene=physics->createScene(desc);PxTriangleMesh* mesh=NULL;PxDeformableSurfaceMaterial* material=NULL;
    if(argc>1 && (!std::strcmp(argv[1],"self-near") || !std::strcmp(argv[1],"self-far")))
    {
        const bool ok=selfFilter(*physics,*scene,!std::strcmp(argv[1],"self-far"));
        scene->release();dispatcher->release();PxCloseExtensions();physics->release();return ok?0:1;
    }
    auto* surface=SurfaceAvbdSnippet::createHangingMembrane(*physics,*scene,mesh,material,2,2,0.f);
    if(!surface)return 3;
    PxVec3 center(0);for(PxU32 v=0;v<mesh->getNbVertices();++v)center+=surface->getPositionInvMassBufferH()[v].getXYZ()/PxReal(mesh->getNbVertices());
    auto* rigidMaterial=physics->createMaterial(0,0,0);
    auto* obstacle=box?PxCreateStatic(*physics,PxTransform(center),PxBoxGeometry(.25f,.25f,.25f),*rigidMaterial):
        PxCreatePlane(*physics,PxPlane(1,0,0,-center.x),*rigidMaterial);
    if(!obstacle || !scene->addActor(*obstacle))return 3;
    if(!scene->simulate(1.f/60) || !scene->fetchResults(true))return 2;
    // Reaching this line would be an erroneous accepted collision step.
    scene->release();dispatcher->release();PxCloseExtensions();physics->release();return 1;
}
