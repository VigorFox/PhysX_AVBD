// Explicit minimal material contract: no material damping, self collision with
// zero penetration filter. This is NOT the unchanged default snippet asset.
#include "PxPhysicsAPI.h"
#include "avbd_probe_foundation.h"
#include "../physx/snippets/snippetdeformablesurfaceavbd/SnippetDeformableSurfaceAVBDCommon.h"
#include <chrono>
#include <vector>
#include <algorithm>
using namespace physx;
// Independent complete triangle/box SAT observer, including triangles whose
// vertices are all outside the box. The one-millimetre core is a scene-scale
// observation tolerance, not a solver setting or candidate filter.
static bool triangleBoxCore(const PxVec3* triangle)
{
    PxVec3d p[3],edge[3];const PxVec3d extent(1.149,.999,1.149);
    for(PxU32 i=0;i<3;++i)p[i]=PxVec3d(triangle[i].x,triangle[i].y-1.,triangle[i].z);
    for(PxU32 i=0;i<3;++i)edge[i]=p[(i+1)%3]-p[i];
    const auto separated=[&](const PxVec3d& axis)
    {
        double low=p[0].dot(axis),high=low;
        for(PxU32 i=1;i<3;++i){const double q=p[i].dot(axis);low=PxMin(low,q);high=PxMax(high,q);}
        const double radius=extent.x*std::abs(axis.x)+extent.y*std::abs(axis.y)+extent.z*std::abs(axis.z);
        return axis.magnitudeSquared()>0 && (low>=radius || high<=-radius);
    };
    const PxVec3d basis[3]={PxVec3d(1,0,0),PxVec3d(0,1,0),PxVec3d(0,0,1)};
    for(PxU32 d=0;d<3;++d)if(separated(basis[d]))return false;
    if(separated(edge[0].cross(edge[1])))return false;
    for(PxU32 e=0;e<3;++e)for(PxU32 d=0;d<3;++d)if(separated(edge[e].cross(basis[d])))return false;
    return true;
}
struct Errors:AvbdProbeFoundation
{
    PxU32 frame=0;
    void reportError(PxErrorCode::Enum,const char* message,const char*,int) override
    {std::fprintf(stderr,"[NODAL_CLOTH_CONTACT] frame=%u status=SIMULATION_FAILED reason=%s\n",frame,message);std::exit(2);}
};
int main(int argc,char** argv)
{
    const PxVec3 crossing[3]={PxVec3(-3,1,-3),PxVec3(3,1,-3),PxVec3(0,1,3)};
    const PxVec3 outside[3]={PxVec3(-3,2.1f,-3),PxVec3(3,2.1f,-3),PxVec3(0,2.1f,3)};
    if(!triangleBoxCore(crossing) || triangleBoxCore(outside))return 2;
    const PxU32 workers=argc>1?PxU32(std::atoi(argv[1])):4;
    const bool self=argc<3 || std::strcmp(argv[2],"no-self")!=0;
    _putenv_s("PHYSX_AVBD_COMMON_STEP","budget");_putenv_s("PHYSX_AVBD_LOCAL_ITERATIONS","4");
    _putenv_s("PHYSX_NP_GEOMETRY_CERTIFICATES","2");Errors errors;
    auto* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*errors.foundation,PxTolerancesScale());PxInitExtensions(*physics,NULL);
    auto* dispatcher=PxDefaultCpuDispatcherCreate(workers);PxSceneDesc desc(physics->getTolerancesScale());
    desc.solverType=PxSolverType::eAVBD;desc.gravity=PxVec3(0,-9.81f,0);desc.cpuDispatcher=dispatcher;desc.filterShader=PxDefaultSimulationFilterShader;
    auto* scene=physics->createScene(desc);PxTriangleMesh* mesh=NULL;PxDeformableSurfaceMaterial* material=NULL;
    auto* surface=SurfaceAvbdSnippet::createDrapedSurface(*physics,*scene,mesh,material,0.2f,SurfaceAvbdSnippet::kReferenceDrapeGrid);
    if(!surface)return 2;material->setElasticityDamping(0);material->setBendingDamping(0);
    surface->setSelfCollisionFilterDistance(0);surface->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION,!self);
    surface->setSolverIterationCounts(4,1);surface->setLinearDamping(0);
    auto* rigidMaterial=physics->createMaterial(.7f,.6f,0);
    auto* ground=PxCreatePlane(*physics,PxPlane(0,1,0,0),*rigidMaterial);
    auto* box=PxCreateStatic(*physics,PxTransform(PxVec3(0,1,0)),PxBoxGeometry(1.15f,1,1.15f),*rigidMaterial);
    scene->addActor(*ground);scene->addActor(*box);
    double maxStretch=1,minY=1.e10,maxLowerEnergy=0,initialLowerEnergy=0,minAreaRatio=1;
    PxU32 inside=0,triangleHits=0;bool finite=true;
    const auto* rest=surface->getRestPositionBufferH();const auto* p=surface->getPositionInvMassBufferH();const auto* v=surface->getVelocityBufferH();
    for(PxU32 i=0;i<mesh->getNbVertices();++i)initialLowerEnergy+=9.81*p[i].y/p[i].w;
    std::vector<double> times;
    for(errors.frame=1;errors.frame<=180;++errors.frame)
    {
        const auto start=std::chrono::steady_clock::now();if(!scene->simulate(1.f/60) || !scene->fetchResults(true))return 2;
        times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());double energy=0;
        for(PxU32 i=0;i<mesh->getNbVertices();++i)
        {
            finite &= p[i].isFinite() && v[i].isFinite();minY=PxMin(minY,double(p[i].y));
            const auto local=p[i].getXYZ()-PxVec3(0,1,0);
            if(PxAbs(local.x)<1.149f && PxAbs(local.y)<.999f && PxAbs(local.z)<1.149f)++inside;
            energy+=(.5*v[i].getXYZ().magnitudeSquared()+9.81*p[i].y)/p[i].w;
        }
        maxLowerEnergy=PxMax(maxLowerEnergy,energy);
        const bool small=mesh->getTriangleMeshFlags().isSet(PxTriangleMeshFlag::e16_BIT_INDICES);
        for(PxU32 t=0;t<mesh->getNbTriangles();++t)
        {
            PxVec3 x[3],r[3];
            for(PxU32 j=0;j<3;++j)
            {
                const PxU32 a=small?static_cast<const PxU16*>(mesh->getTriangles())[3*t+j]:static_cast<const PxU32*>(mesh->getTriangles())[3*t+j];
                x[j]=p[a].getXYZ();r[j]=rest[a].getXYZ();
            }
            triangleHits+=triangleBoxCore(x)?1:0;
            minAreaRatio=PxMin(minAreaRatio,double((x[1]-x[0]).cross(x[2]-x[0]).magnitude()/(r[1]-r[0]).cross(r[2]-r[0]).magnitude()));
            for(PxU32 j=0;j<3;++j)maxStretch=PxMax(maxStretch,double((x[j]-x[(j+1)%3]).magnitude()/(r[j]-r[(j+1)%3]).magnitude()));
        }
    }
    double mean=0;for(double t:times)mean+=t/times.size();std::sort(times.begin(),times.end());
    const bool pass=finite && !inside && !triangleHits && minAreaRatio>0 && minY>=-.001 && maxStretch<1.5 && maxLowerEnergy<=1.05*initialLowerEnergy;
    std::printf("[NODAL_CLOTH_CONTACT] workers=%u frames=180 iterations=4 self=%u nodes=%u maxStretch=%.9g minY=%.9g insideVertices=%u triangleBoxCoreHits=%u minAreaRatio=%.9g initialLowerEnergy=%.9g maxLowerEnergy=%.9g meanMs=%.6f p95Ms=%.6f status=%s\n",
        workers,unsigned(self),mesh->getNbVertices(),maxStretch,minY,inside,triangleHits,minAreaRatio,initialLowerEnergy,maxLowerEnergy,mean,times[size_t(.95*times.size())],pass?"PASS":"FAIL");
    surface->release();mesh->release();material->release();scene->release();rigidMaterial->release();dispatcher->release();PxCloseExtensions();physics->release();return pass?0:1;
}
