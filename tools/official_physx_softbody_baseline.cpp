// Standalone observer for unmodified PhysX 5.9 public APIs.
// The regular cubes are NOT the original AVBD five-asset showcase.
#include "PxPhysicsAPI.h"
#include "extensions/PxDeformableVolumeExt.h"
#include "extensions/PxCudaHelpersExt.h"
#include "cudamanager/PxCudaContext.h"
#include "official_physx_baseline_geometry.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
using namespace physx;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static void number(double v){if(std::isfinite(v))std::printf("%.12g",v);else std::printf("null");}
static void js(const char* s)
{
    std::putchar('"');
    for(;*s;++s){if(*s=='"'||*s=='\\')std::putchar('\\');
        if(static_cast<unsigned char>(*s)<32)std::printf("\\u%04x",unsigned(static_cast<unsigned char>(*s)));else std::putchar(*s);}
    std::putchar('"');
}
static void modulePath(const char* name)
{
#ifdef _WIN32
    char p[32768]={};HMODULE h=GetModuleHandleA(name);
    if(h&&GetModuleFileNameA(h,p,sizeof(p)))js(p);else std::printf("null");
#else
    (void)name;std::printf("null");
#endif
}
static void modules()
{
    std::printf("\"physxModule\":");modulePath("PhysX_64.dll");
    std::printf(",\"commonModule\":");modulePath("PhysXCommon_64.dll");
    std::printf(",\"foundationModule\":");modulePath("PhysXFoundation_64.dll");
    std::printf(",\"cookingModule\":");modulePath("PhysXCooking_64.dll");
    std::printf(",\"gpuModule\":");modulePath("PhysXGpu_64.dll");
}
struct Errors:PxErrorCallback
{
    std::atomic<unsigned> serious{0},warnings{0};
    void reportError(PxErrorCode::Enum code,const char* message,const char* file,int line)override
    {
        if(code==PxErrorCode::eDEBUG_INFO||code==PxErrorCode::eDEBUG_WARNING||code==PxErrorCode::ePERF_WARNING)++warnings;else ++serious;
        std::fprintf(stderr,"PhysX error %u: %s (%s:%d)\n",unsigned(code),message,file,line);
    }
};
struct Config
{
    std::string backend="fem-gpu",scene="five-cubes",solver="tgs";
    unsigned frames=360,iterations=30,voxels=8,workers=4;bool selfCollision=false;
    float dt=1.f/60.f,side=2.5f,young=2.e5f,poisson=.3f,density=100.f,friction=.1f;
};
static unsigned uintArgument(const char* s)
{
    if(!*s||*s=='-')throw std::runtime_error("invalid unsigned argument");
    size_t end=0;const auto value=std::stoull(s,&end);
    if(s[end]||value>std::numeric_limits<unsigned>::max())throw std::runtime_error("unsigned argument out of range");
    return unsigned(value);
}
static float realArgument(const char* s)
{
    size_t end=0;const float value=std::stof(s,&end);
    if(s[end]||!std::isfinite(value))throw std::runtime_error("invalid finite real argument");
    return value;
}
static Config parse(int argc,char** argv)
{
    Config c;
    for(int i=1;i<argc;++i)
    {
        const std::string a=argv[i];
        if(a=="--help"){std::puts("--backend fem-gpu|rigid-cpu --case five-cubes|drop|reaction --solver pgs|tgs --self-collision on|off --frames 360 --iterations 30 --voxels 8 --workers 4 --dt 0.016666667 --side 2.5 --young 200000 --poisson .3 --density 100 --friction .1");std::exit(0);}
        if(i+1>=argc)throw std::runtime_error("argument missing value");const char* v=argv[++i];
        if(a=="--backend")c.backend=v;else if(a=="--case")c.scene=v;else if(a=="--solver")c.solver=v;
        else if(a=="--self-collision"){if(std::strcmp(v,"on")&&std::strcmp(v,"off"))throw std::runtime_error("invalid self-collision");c.selfCollision=!std::strcmp(v,"on");}
        else if(a=="--frames")c.frames=uintArgument(v);else if(a=="--iterations")c.iterations=uintArgument(v);
        else if(a=="--voxels")c.voxels=uintArgument(v);else if(a=="--workers")c.workers=uintArgument(v);
        else if(a=="--dt")c.dt=realArgument(v);else if(a=="--side")c.side=realArgument(v);
        else if(a=="--young")c.young=realArgument(v);else if(a=="--poisson")c.poisson=realArgument(v);
        else if(a=="--density")c.density=realArgument(v);else if(a=="--friction")c.friction=realArgument(v);
        else throw std::runtime_error("unknown argument: "+a);
    }
    if((c.backend!="fem-gpu"&&c.backend!="rigid-cpu")||(c.solver!="pgs"&&c.solver!="tgs")||
       (c.scene!="five-cubes"&&c.scene!="drop"&&c.scene!="reaction")||!c.frames||c.iterations<1||c.iterations>255||
       !c.voxels||c.voxels>64||c.workers>256||!(c.dt>0&&c.dt<1&&c.side>0&&c.young>0&&c.density>0&&
       c.poisson>=0&&c.poisson<.5f&&c.friction>=0))throw std::runtime_error("invalid configuration");
    return c;
}
struct V
{
    double x=0,y=0,z=0;V()=default;V(double a,double b,double c):x(a),y(b),z(c){}
    explicit V(const PxVec4& a):x(a.x),y(a.y),z(a.z){}explicit V(const PxVec3& a):x(a.x),y(a.y),z(a.z){}
    V operator-(V b)const{return V(x-b.x,y-b.y,z-b.z);}V operator+(V b)const{return V(x+b.x,y+b.y,z+b.z);}
    V operator*(double s)const{return V(x*s,y*s,z*s);}double dot(V b)const{return x*b.x+y*b.y+z*b.z;}
    V cross(V b)const{return V(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);}double norm()const{return std::sqrt(dot(*this));}
};
static void vec(V v){std::putchar('[');number(v.x);std::putchar(',');number(v.y);std::putchar(',');number(v.z);std::putchar(']');}
struct Body
{
    PxDeformableVolume* soft=nullptr;PxRigidDynamic* rigid=nullptr;
    PxVec4 *positions=nullptr,*velocities=nullptr,*collision=nullptr;
    unsigned ns=0,nc=0,nt=0;std::vector<PxU32> indices;std::vector<double> detRest;
    PxVec3 initialPosition{0},initialVelocity{0};double restVolume=0,mass=0;
};
struct Measurement
{
    bool finite=true;double minDet=1,volumeRatio=1,ground=0,speed=0,kinetic=0,gravity=0;
    unsigned inverted=0,sleeping=0;V momentum,com;PxBounds3 bounds=PxBounds3::empty();
};
struct Samples
{
    std::vector<double> v;void add(double x){v.push_back(x);}
    double mean()const{double s=0;for(double x:v)s+=x;return v.empty()?0:s/v.size();}
    double quantile(double p)const{if(v.empty())return 0;auto s=v;std::sort(s.begin(),s.end());return s[size_t(std::ceil(p*(s.size()-1)))];}
};
struct App
{
    Config c;PxDefaultAllocator allocator;Errors errors;
    PxFoundation* foundation=nullptr;PxPhysics* physics=nullptr;PxCudaContextManager* cuda=nullptr;
    PxDefaultCpuDispatcher* dispatcher=nullptr;PxScene* scene=nullptr;
    PxMaterial* rigidMaterial=nullptr;PxDeformableVolumeMaterial* softMaterial=nullptr;
    PxShape* groundShape=nullptr; // Non-owning; the scene owns the ground actor.
    bool extensions=false;std::vector<Body> bodies;
    explicit App(const Config& cfg):c(cfg){}
    ~App(){shutdown();}
    void shutdown()
    {
        for(Body& b:bodies)
        {
            if(b.soft){b.soft->release();b.soft=nullptr;}if(b.rigid){b.rigid->release();b.rigid=nullptr;}
            if(cuda){Ext::PxCudaHelpersExt::freePinnedHostBuffer(*cuda,b.positions);Ext::PxCudaHelpersExt::freePinnedHostBuffer(*cuda,b.velocities);Ext::PxCudaHelpersExt::freePinnedHostBuffer(*cuda,b.collision);}
        }
        if(scene){scene->release();scene=nullptr;}if(dispatcher){dispatcher->release();dispatcher=nullptr;}
        if(softMaterial){softMaterial->release();softMaterial=nullptr;}if(rigidMaterial){rigidMaterial->release();rigidMaterial=nullptr;}
        if(extensions){PxCloseExtensions();extensions=false;}if(physics){physics->release();physics=nullptr;}
        if(cuda){cuda->release();cuda=nullptr;}if(foundation){foundation->release();foundation=nullptr;}
    }
    void read(Body& b)
    {
        if(!b.soft)return;PxScopedCudaLock lock(*cuda);auto* ctx=cuda->getCudaContext();
        if(ctx->memcpyDtoH(b.positions,reinterpret_cast<CUdeviceptr>(b.soft->getSimPositionInvMassBufferD()),sizeof(PxVec4)*b.ns)||
           ctx->memcpyDtoH(b.velocities,reinterpret_cast<CUdeviceptr>(b.soft->getSimVelocityBufferD()),sizeof(PxVec4)*b.ns)||
           ctx->memcpyDtoH(b.collision,reinterpret_cast<CUdeviceptr>(b.soft->getPositionInvMassBufferD()),sizeof(PxVec4)*b.nc))throw std::runtime_error("GPU readback failed");
    }
    void sync(){if(cuda){PxScopedCudaLock lock(*cuda);if(cuda->getCudaContext()->streamSynchronize(CUstream(0)))throw std::runtime_error("GPU synchronization failed");}}
    OfficialPhysxBaselineGeometry::Report geometry()const
    {
        using namespace OfficialPhysxBaselineGeometry;
        std::vector<BodyView> views(bodies.size());
        std::vector<std::array<PxVec4,8>> rigidVertices(bodies.size());
        static const PxU32 boxTets[]={0,1,3,7,0,3,2,7,0,2,6,7,0,6,4,7,0,4,5,7,0,5,1,7};
        for(unsigned i=0;i<bodies.size();++i)
        {
            const Body& b=bodies[i];BodyView& v=views[i];
            if(b.soft)
            {
                auto* mesh=b.soft->getCollisionMesh();v.positions=b.collision;v.vertexCount=b.nc;
                v.tetrahedra=mesh->getTetrahedrons();v.tetrahedronCount=mesh->getNbTetrahedrons();
                v.indices16Bit=bool(mesh->getTetrahedronMeshFlags()&PxTetrahedronMeshFlag::e16_BIT_INDICES);
            }
            else
            {
                for(unsigned k=0;k<8;++k)
                    rigidVertices[i][k]=PxVec4(b.rigid->getGlobalPose().transform(PxVec3(
                        (k&1?1.f:-1.f)*c.side*.5f,(k&2?1.f:-1.f)*c.side*.5f,(k&4?1.f:-1.f)*c.side*.5f)),0);
                v.positions=rigidVertices[i].data();v.vertexCount=8;v.tetrahedra=boxTets;v.tetrahedronCount=6;
            }
        }
        return observe(views.data(),views.size());
    }
    void init()
    {
        foundation=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors);if(!foundation)throw std::runtime_error("foundation unavailable");
        if(c.backend=="fem-gpu"){PxCudaContextManagerDesc d;cuda=PxCreateCudaContextManager(*foundation,d,PxGetProfilerCallback());if(!cuda||!cuda->contextIsValid())throw std::runtime_error("official GPU FEM unavailable: valid CUDA context required");}
        physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation,PxTolerancesScale(),true,nullptr);if(!physics)throw std::runtime_error("physics creation failed");
        extensions=PxInitExtensions(*physics,nullptr);if(!extensions)throw std::runtime_error("extensions creation failed");
        PxSceneDesc d(physics->getTolerancesScale());d.gravity=PxVec3(0,c.scene=="reaction"?0.f:-9.81f,0);
        dispatcher=PxDefaultCpuDispatcherCreate(c.workers);if(!dispatcher)throw std::runtime_error("dispatcher creation failed");
        d.cpuDispatcher=dispatcher;d.filterShader=PxDefaultSimulationFilterShader;
        d.solverType=c.solver=="pgs"?PxSolverType::ePGS:PxSolverType::eTGS;d.flags|=PxSceneFlag::eENABLE_PCM;
        if(cuda){d.cudaContextManager=cuda;d.flags|=PxSceneFlag::eENABLE_GPU_DYNAMICS;d.broadPhaseType=PxBroadPhaseType::eGPU;d.gpuMaxNumPartitions=8;}
        scene=physics->createScene(d);if(!scene)throw std::runtime_error("scene creation failed");
        rigidMaterial=physics->createMaterial(.5f,.5f,0);if(!rigidMaterial)throw std::runtime_error("rigid material creation failed");
        if(c.scene!="reaction")
        {
            auto* ground=PxCreatePlane(*physics,PxPlane(0,1,0,0),*rigidMaterial);
            if(!ground)throw std::runtime_error("ground creation failed");
            scene->addActor(*ground);
            if(ground->getShapes(&groundShape,1)!=1||!groundShape)throw std::runtime_error("missing ground measurement shape");
        }
        if(cuda){softMaterial=physics->createDeformableVolumeMaterial(c.young,c.poisson,c.friction,0);if(!softMaterial)throw std::runtime_error("FEM material creation failed");}
        const unsigned count=c.scene=="five-cubes"?5:c.scene=="reaction"?2:1;bodies.reserve(count);
        for(unsigned i=0;i<count;++i)
        {
            bodies.emplace_back();Body& b=bodies.back();
            b.initialPosition=c.scene=="reaction"?PxVec3((i?1.f:-1.f)*(.6f*c.side),0,0):
                PxVec3(c.scene=="five-cubes"?.032f*c.side*(i%2):0,c.scene=="drop"?1.8f*c.side:(.6f+1.2f*i)*c.side,0);
            b.initialVelocity=c.scene=="reaction"?PxVec3(i?-1.f:1.f,0,0):PxVec3(0);
            if(cuda)
            {
                b.soft=PxDeformableVolumeExt::createDeformableVolumeBox(PxTransform(b.initialPosition),PxVec3(c.side),*softMaterial,*cuda,.4f*c.side,c.density,c.voxels,1.f);
                if(!b.soft)throw std::runtime_error("official box cooking/creation failed");
                b.soft->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION,!c.selfCollision);
                b.soft->setSolverIterationCounts(c.iterations);scene->addActor(*b.soft);
                auto* mesh=b.soft->getSimulationMesh();auto* collisionMesh=b.soft->getCollisionMesh();
                if(!mesh||!collisionMesh||!mesh->getNbVertices()||!mesh->getNbTetrahedrons()||
                    !collisionMesh->getNbVertices()||!collisionMesh->getNbTetrahedrons()||
                    !mesh->getTetrahedrons()||!collisionMesh->getTetrahedrons())
                    throw std::runtime_error("missing or empty simulation/collision tetrahedron mesh");
                b.ns=mesh->getNbVertices();b.nt=mesh->getNbTetrahedrons();b.nc=collisionMesh->getNbVertices();
                b.positions=Ext::PxCudaHelpersExt::allocPinnedHostBuffer<PxVec4>(*cuda,b.ns);
                b.velocities=Ext::PxCudaHelpersExt::allocPinnedHostBuffer<PxVec4>(*cuda,b.ns);
                b.collision=Ext::PxCudaHelpersExt::allocPinnedHostBuffer<PxVec4>(*cuda,b.nc);
                if(!b.positions||!b.velocities||!b.collision)throw std::runtime_error("pinned allocation failed");read(b);
                b.indices.resize(4*b.nt);b.detRest.resize(b.nt);
                const bool shortIndices=bool(mesh->getTetrahedronMeshFlags()&PxTetrahedronMeshFlag::e16_BIT_INDICES);
                for(unsigned k=0;k<4*b.nt;++k)b.indices[k]=shortIndices?static_cast<const PxU16*>(mesh->getTetrahedrons())[k]:static_cast<const PxU32*>(mesh->getTetrahedrons())[k];
                for(unsigned t=0;t<b.nt;++t)
                {
                    const PxU32* ix=&b.indices[t*4];const V a(b.positions[ix[0]]);
                    const double det=(V(b.positions[ix[1]])-a).dot((V(b.positions[ix[2]])-a).cross(V(b.positions[ix[3]])-a));
                    if(!std::isfinite(det)||det==0)throw std::runtime_error("degenerate rest tetrahedron");
                    b.detRest[t]=det;b.restVolume+=std::abs(det)/6;
                }
                for(unsigned k=0;k<b.ns;++k){if(b.positions[k].w>0)b.mass+=1./b.positions[k].w;b.velocities[k]=PxVec4(b.initialVelocity,0);}
                {PxScopedCudaLock lock(*cuda);if(cuda->getCudaContext()->memcpyHtoD(reinterpret_cast<CUdeviceptr>(b.soft->getSimVelocityBufferD()),b.velocities,sizeof(PxVec4)*b.ns))throw std::runtime_error("velocity upload failed");}
                b.soft->markDirty(PxDeformableVolumeDataFlag::eSIM_VELOCITY);
            }
            else
            {
                b.rigid=PxCreateDynamic(*physics,PxTransform(b.initialPosition),PxBoxGeometry(PxVec3(.5f*c.side)),*rigidMaterial,c.density);
                if(!b.rigid)throw std::runtime_error("rigid creation failed");
                b.rigid->setSolverIterationCounts(c.iterations);b.rigid->setLinearVelocity(b.initialVelocity);scene->addActor(*b.rigid);
                b.mass=b.rigid->getMass();b.restVolume=double(c.side)*c.side*c.side;
            }
        }
        sync();printConfig();
    }
    void printConfig()
    {
        std::printf("{\"type\":\"config\",\"schema\":1,\"sdkVersion\":%u,\"assetScope\":\"regular-cubes-not-original-five-assets\",\"backend\":",PX_PHYSICS_VERSION);js(c.backend.c_str());
        std::printf(",\"case\":");js(c.scene.c_str());std::printf(",\"rigidSolver\":");js(c.solver.c_str());
        std::printf(",\"deformableSolver\":%s,\"frames\":%u,\"iterations\":%u,\"voxels\":%u,\"workers\":%u,\"selfCollision\":%s,\"dt\":%.9g,\"side\":%.9g,\"young\":%.9g,\"poisson\":%.9g,\"density\":%.9g,\"softDynamicFriction\":%.9g,\"groundFriction\":0.5,\"maxInvMassRatio\":%s,\"elasticityDamping\":0,\"gravityY\":%.9g,\"gpu\":",
            cuda?"\"XPBD\"":"null",c.frames,c.iterations,c.voxels,c.workers,c.selfCollision?"true":"false",c.dt,c.side,c.young,c.poisson,c.density,c.friction,cuda?"50":"null",scene->getGravity().y);
        if(cuda)js(cuda->getDeviceName());else std::printf("null");
        const auto broadPhase=scene->getBroadPhaseType();
        std::printf(",\"sceneBroadPhase\":%u,\"sceneBroadPhaseName\":",unsigned(broadPhase));
        js(broadPhase==PxBroadPhaseType::eGPU?"gpu":broadPhase==PxBroadPhaseType::eABP?"abp":
            broadPhase==PxBroadPhaseType::eMBP?"mbp":broadPhase==PxBroadPhaseType::eSAP?"sap":"other");
        std::printf(",\"actualSceneSolver\":%u,\"rigidMaterialStaticFriction\":%.9g,\"rigidMaterialDynamicFriction\":%.9g,\"rigidMaterialRestitution\":%.9g,\"rigidMaterialFrictionCombineMode\":%u,\"softDynamicFrictionOptionApplicable\":%s",
            unsigned(scene->getSolverType()),rigidMaterial->getStaticFriction(),rigidMaterial->getDynamicFriction(),
            rigidMaterial->getRestitution(),unsigned(rigidMaterial->getFrictionCombineMode()),cuda?"true":"false");
        std::printf(",\"softMaterialActual\":");
        if(softMaterial)std::printf("{\"young\":%.9g,\"poisson\":%.9g,\"dynamicFriction\":%.9g,\"elasticityDamping\":%.9g}",
            softMaterial->getYoungsModulus(),softMaterial->getPoissons(),softMaterial->getDynamicFriction(),softMaterial->getElasticityDamping());
        else std::printf("null");
        std::printf(",\"sceneFlags\":%u,\"sceneCCD\":%s,\"groundPenetrationApplicable\":%s,\"crossBodyGeometryApplicable\":%s,\"selfCollisionOptionApplicable\":%s,\"deformationMetricScope\":",
            unsigned(scene->getFlags()),scene->getFlags().isSet(PxSceneFlag::eENABLE_CCD)?"true":"false",
            c.scene!="reaction"?"true":"false",bodies.size()>1?"true":"false",cuda?"true":"false");
        js(cuda?"simulation-tetrahedron-gradients":"rigid-identity-no-FEM-measurement");
        std::printf(",\"groundContactOffset\":");if(groundShape)number(groundShape->getContactOffset());else std::printf("null");
        std::printf(",\"groundRestOffset\":");if(groundShape)number(groundShape->getRestOffset());else std::printf("null");
        std::printf(",\"continuousPathPenetrationMeasured\":false");
        std::printf(",\"energyScope\":\"kinetic-plus-gravity-only-elastic-energy-not-measured\",\"pairGeometryScope\":\"cross-body-collision-tetrahedra-SAT-MTD-not-union-depth\",\"selfPenetrationMeasured\":false,\"stepTiming\":\"simulate-fetchResults-then-CUDA-default-stream-sync\",");modules();std::puts("}");
        for(unsigned i=0;i<bodies.size();++i)
        {
            const Body& b=bodies[i];std::printf("{\"type\":\"body\",\"index\":%u,\"position\":",i);vec(V(b.initialPosition));
            PxU32 positionIterations=0,velocityIterations=0;
            if(b.soft)b.soft->getSolverIterationCounts(positionIterations,velocityIterations);
            else b.rigid->getSolverIterationCounts(positionIterations,velocityIterations);
            std::printf(",\"positionIterations\":%u,\"velocityIterations\":%u,\"velocityIterationsUsed\":%s",
                positionIterations,velocityIterations,b.soft?"false":"true");
            std::printf(",\"orientation\":[0,0,0,1],\"initialVelocity\":");vec(V(b.initialVelocity));
            std::printf(",\"mass\":%.12g,\"restVolume\":%.12g,\"simVertices\":%u,\"simTetrahedra\":%u,\"collisionVertices\":%u,\"collisionTetrahedra\":%u",b.mass,b.restVolume,b.ns,b.nt,b.nc,b.soft?b.soft->getCollisionMesh()->getNbTetrahedrons():0);
            if(b.soft)std::printf(",\"linearDamping\":%.9g,\"sleepThreshold\":%.9g,\"settlingThreshold\":%.9g,\"settlingDamping\":%.9g,\"maxVelocity\":%.9g,\"maxDepenetrationVelocity\":%.9g,\"selfCollisionFilterDistance\":%.9g,\"selfCollisionStressTolerance\":%.9g,\"contactOffset\":%.9g,\"restOffset\":%.9g,\"materialModel\":%u",b.soft->getLinearDamping(),b.soft->getSleepThreshold(),b.soft->getSettlingThreshold(),b.soft->getSettlingDamping(),b.soft->getMaxLinearVelocity(),b.soft->getMaxDepenetrationVelocity(),b.soft->getSelfCollisionFilterDistance(),b.soft->getSelfCollisionStressTolerance(),b.soft->getShape()->getContactOffset(),b.soft->getShape()->getRestOffset(),unsigned(softMaterial->getMaterialModel()));
            else std::printf(",\"linearDamping\":%.9g,\"angularDamping\":%.9g,\"sleepThreshold\":%.9g",b.rigid->getLinearDamping(),b.rigid->getAngularDamping(),b.rigid->getSleepThreshold());
            if(b.soft)
            {
                std::printf(",\"deformableBodyFlags\":%u,\"deformableVolumeFlags\":%u,\"selfCollisionEnabled\":%s,\"rigidCCD\":null,\"rigidSpeculativeCCD\":null",
                    unsigned(b.soft->getDeformableBodyFlags()),unsigned(b.soft->getDeformableVolumeFlags()),
                    b.soft->getDeformableBodyFlags().isSet(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION)?"false":"true");
            }
            else
            {
                PxShape* shape=nullptr;
                if(b.rigid->getShapes(&shape,1)!=1||!shape)throw std::runtime_error("missing rigid measurement shape");
                std::printf(",\"contactOffset\":%.9g,\"restOffset\":%.9g,\"rigidBodyFlags\":%u,\"rigidCCD\":%s,\"rigidSpeculativeCCD\":%s,\"deformableBodyFlags\":null,\"deformableVolumeFlags\":null,\"selfCollisionEnabled\":null",
                    shape->getContactOffset(),shape->getRestOffset(),unsigned(b.rigid->getRigidBodyFlags()),
                    b.rigid->getRigidBodyFlags().isSet(PxRigidBodyFlag::eENABLE_CCD)?"true":"false",
                    b.rigid->getRigidBodyFlags().isSet(PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD)?"true":"false");
            }
            std::puts("}");
        }
    }
    Measurement measure(const Body& b)const
    {
        Measurement m;
        if(b.soft)
        {
            m.sleeping=b.soft->isSleeping()?1:0;m.minDet=1.e300;double volume=0,massSum=0;
            for(unsigned k=0;k<b.ns;++k)
            {
                const V p(b.positions[k]),v(b.velocities[k]);m.finite=m.finite&&b.positions[k].isFinite()&&b.positions[k].w>=0&&
                    std::isfinite(v.x)&&std::isfinite(v.y)&&std::isfinite(v.z);
                const double mass=b.positions[k].w>0?1./b.positions[k].w:0;
                m.kinetic+=.5*mass*v.dot(v);m.gravity-=mass*scene->getGravity().y*p.y;m.momentum=m.momentum+v*mass;
                m.com=m.com+p*mass;massSum+=mass;m.speed=std::max(m.speed,v.norm());
            }
            for(unsigned t=0;t<b.nt;++t)
            {
                const PxU32* ix=&b.indices[t*4];const V a(b.positions[ix[0]]);
                const double det=(V(b.positions[ix[1]])-a).dot((V(b.positions[ix[2]])-a).cross(V(b.positions[ix[3]])-a));
                const double ratio=det/b.detRest[t];m.minDet=std::min(m.minDet,ratio);if(ratio<=0)++m.inverted;volume+=ratio*std::abs(b.detRest[t])/6;
            }
            m.volumeRatio=volume/b.restVolume;
            if(massSum>0)m.com=m.com*(1./massSum);else m.finite=false;
            for(unsigned k=0;k<b.nc;++k){m.finite=m.finite&&b.collision[k].isFinite();m.bounds.include(PxVec3(b.collision[k].x,b.collision[k].y,b.collision[k].z));if(c.scene!="reaction")m.ground=std::max(m.ground,-double(b.collision[k].y));}
        }
        else
        {
            const PxTransform pose=b.rigid->getGlobalPose();const V v(b.rigid->getLinearVelocity());
            const PxVec3 w=pose.q.rotateInv(b.rigid->getAngularVelocity()),inertia=b.rigid->getMassSpaceInertiaTensor();
            m.finite=pose.isFinite()&&b.rigid->getLinearVelocity().isFinite()&&w.isFinite();m.sleeping=b.rigid->isSleeping()?1:0;
            m.kinetic=.5*b.mass*v.dot(v)+.5*(double(inertia.x)*w.x*w.x+double(inertia.y)*w.y*w.y+double(inertia.z)*w.z*w.z);
            m.gravity=-b.mass*scene->getGravity().y*pose.p.y;m.momentum=v*b.mass;m.com=V(pose.p);
            for(unsigned k=0;k<8;++k)
            {
                const PxVec3 local((k&1?1.f:-1.f)*c.side*.5f,(k&2?1.f:-1.f)*c.side*.5f,(k&4?1.f:-1.f)*c.side*.5f),p=pose.transform(local);m.bounds.include(p);
                m.speed=std::max(m.speed,V(b.rigid->getLinearVelocity()+b.rigid->getAngularVelocity().cross(p-pose.p)).norm());if(c.scene!="reaction")m.ground=std::max(m.ground,-double(p.y));
            }
        }
        m.finite=m.finite&&std::isfinite(m.minDet)&&std::isfinite(m.volumeRatio)&&std::isfinite(m.kinetic)&&std::isfinite(m.gravity)&&std::isfinite(m.speed);
        return m;
    }
    int run()
    {
        const auto totalStart=Clock::now();init();const double initMs=ms(totalStart,Clock::now());
        Samples steps,reads,observed,penetrations,pairDepths,geometryTimes,activeSteps;double minDet=1,minVolume=1,maxSpeed=0,maxEnergy=0,maxMomentumChange=0;
        unsigned penetrationFrames=0,longestRun=0,currentRun=0,invertedFrames=0,framesDone=0;bool finite=true;
        unsigned pairFrames=0,pairCurrentRun=0,pairLongestRun=0;bool allGeometryValid=true;
        V initialMomentum;double initialKinetic=0,initialGravity=0;
        for(const Body& b:bodies){const auto m=measure(b);initialMomentum=initialMomentum+m.momentum;
            initialKinetic+=m.kinetic;initialGravity+=m.gravity;finite&=m.finite;
            minDet=std::min(minDet,m.minDet);minVolume=std::min(minVolume,m.volumeRatio);maxSpeed=std::max(maxSpeed,m.speed);}
        maxEnergy=initialKinetic+initialGravity;
        std::printf("{\"type\":\"initialState\",\"bodies\":[");
        for(unsigned i=0;i<bodies.size();++i){if(i)std::putchar(',');const auto m=measure(bodies[i]);
            std::printf("{\"com\":");vec(m.com);std::printf(",\"kineticEnergy\":");number(m.kinetic);
            std::printf(",\"gravitationalEnergy\":");number(m.gravity);std::printf(",\"momentum\":");vec(m.momentum);
            std::printf(",\"minDet\":");number(m.minDet);std::printf(",\"volumeRatio\":");number(m.volumeRatio);std::printf("}");}
        std::puts("]}");
        const auto initialGeometry=geometry();
        allGeometryValid=initialGeometry.valid;
        std::printf("{\"type\":\"initialGeometry\",\"valid\":%s,\"overlappingTetPairs\":%llu,\"maxTetMtd\":%.12g,\"degenerateTetrahedra\":%llu,\"invalidTetrahedra\":%llu}\n",
            initialGeometry.valid?"true":"false",static_cast<unsigned long long>(initialGeometry.overlappingTetPairs),initialGeometry.maxTetMtd,
            static_cast<unsigned long long>(initialGeometry.degenerateTetrahedra),static_cast<unsigned long long>(initialGeometry.invalidTetrahedra));
        if(!finite||errors.serious.load())throw std::runtime_error("invalid initialized simulation state");
        for(unsigned frame=0;frame<c.frames;++frame)
        {
            bool awakeBefore=false;for(const Body& b:bodies)awakeBefore|=b.soft?!b.soft->isSleeping():!b.rigid->isSleeping();
            const auto start=Clock::now();scene->simulate(c.dt);PxU32 fetchError=0;
            if(!scene->fetchResults(true,&fetchError)||fetchError)throw std::runtime_error("fetchResults failed");sync();const auto simulated=Clock::now();
            for(Body& b:bodies)read(b);const auto readEnd=Clock::now();std::vector<Measurement> measurements;
            Measurement all;all.minDet=1.e300;all.volumeRatio=1.e300;
            for(const Body& b:bodies)
            {
                measurements.push_back(measure(b));const Measurement& m=measurements.back();all.finite&=m.finite;all.minDet=std::min(all.minDet,m.minDet);
                all.volumeRatio=std::min(all.volumeRatio,m.volumeRatio);all.ground=std::max(all.ground,m.ground);all.speed=std::max(all.speed,m.speed);
                all.kinetic+=m.kinetic;all.gravity+=m.gravity;all.momentum=all.momentum+m.momentum;all.inverted+=m.inverted;all.sleeping+=m.sleeping;
            }
            unsigned aabbPairs=0;for(unsigned a=0;a<measurements.size();++a)for(unsigned b=a+1;b<measurements.size();++b)if(measurements[a].bounds.intersects(measurements[b].bounds))++aabbPairs;
            const auto geometryStart=Clock::now();const auto geo=geometry();geometryTimes.add(ms(geometryStart,Clock::now()));
            allGeometryValid&=geo.valid;pairDepths.add(geo.maxTetMtd);
            if(geo.overlappingTetPairs){++pairFrames;++pairCurrentRun;pairLongestRun=std::max(pairLongestRun,pairCurrentRun);}else pairCurrentRun=0;
            const auto measured=Clock::now();const double stepMs=ms(start,simulated),readMs=ms(simulated,readEnd),observedMs=ms(start,measured);
            steps.add(stepMs);reads.add(readMs);observed.add(observedMs);penetrations.add(all.ground);
            const bool active=awakeBefore||all.sleeping<bodies.size();if(active)activeSteps.add(stepMs);
            minDet=std::min(minDet,all.minDet);minVolume=std::min(minVolume,all.volumeRatio);maxSpeed=std::max(maxSpeed,all.speed);
            maxEnergy=std::max(maxEnergy,all.kinetic+all.gravity);maxMomentumChange=std::max(maxMomentumChange,(all.momentum-initialMomentum).norm());
            if(all.ground>0){++penetrationFrames;++currentRun;longestRun=std::max(longestRun,currentRun);}else currentRun=0;
            if(all.inverted)++invertedFrames;finite&=all.finite;++framesDone;
            std::printf("{\"type\":\"frame\",\"frame\":%u,\"time\":%.12g,\"stepMs\":%.12g,\"readbackMs\":%.12g,\"observedMs\":%.12g,\"finite\":%s,\"minDet\":",frame+1,(frame+1)*double(c.dt),stepMs,readMs,observedMs,all.finite?"true":"false");number(all.minDet);
            std::printf(",\"minBodyVolumeRatio\":");number(all.volumeRatio);std::printf(",\"groundPenetration\":");number(all.ground);
            std::printf(",\"maxVertexSpeed\":");number(all.speed);std::printf(",\"kineticEnergy\":");number(all.kinetic);std::printf(",\"gravitationalEnergy\":");number(all.gravity);
            std::printf(",\"momentum\":");vec(all.momentum);std::printf(",\"invertedTetrahedra\":%u,\"sleepingBodies\":%u,\"active\":%s,\"aabbOverlapPairsNotContacts\":%u,\"pairGeometryValid\":%s,\"maxTetMtd\":",
                all.inverted,all.sleeping,active?"true":"false",aabbPairs,geo.valid?"true":"false");number(geo.maxTetMtd);
            std::printf(",\"overlappingTetPairs\":%llu,\"candidateTetPairs\":%llu,\"degenerateCollisionTetrahedra\":%llu,\"invalidCollisionTetrahedra\":%llu,\"geometryMs\":%.12g,\"bodies\":[",
                static_cast<unsigned long long>(geo.overlappingTetPairs),static_cast<unsigned long long>(geo.candidateTetPairs),
                static_cast<unsigned long long>(geo.degenerateTetrahedra),static_cast<unsigned long long>(geo.invalidTetrahedra),geometryTimes.v.back());
            for(unsigned i=0;i<measurements.size();++i){if(i)std::putchar(',');const auto& m=measurements[i];std::printf("{\"com\":");vec(m.com);
                std::printf(",\"momentum\":");vec(m.momentum);std::printf(",\"minDet\":");number(m.minDet);std::printf(",\"volumeRatio\":");number(m.volumeRatio);
                std::printf(",\"maxSpeed\":");number(m.speed);std::printf(",\"sleeping\":%s}",m.sleeping?"true":"false");}
            std::puts("]}");
            if(!finite||errors.serious.load())break;
        }
        const bool completed=framesDone==c.frames,ok=completed&&finite&&!errors.serious.load();
        std::printf("{\"type\":\"summary\",\"completed\":%s,\"finite\":%s,\"executionOk\":%s,\"physicalQualification\":\"not-assigned-observations-only\",\"framesDone\":%u,\"errors\":%u,\"warnings\":%u,\"initializationMs\":%.12g,\"runtimeMs\":%.12g,\"stepMeanMs\":%.12g,\"stepP95Ms\":%.12g,\"stepMaxMs\":%.12g,\"readbackMeanMs\":%.12g,\"observedMeanMs\":%.12g,\"groundPenetrationMax\":%.12g,\"groundPenetrationFrameP95\":%.12g,\"groundPenetrationFrameP99\":%.12g,\"groundPenetrationFrames\":%u,\"groundPenetrationDuration\":%.12g,\"groundPenetrationLongestDuration\":%.12g,\"minDet\":",completed?"true":"false",finite?"true":"false",ok?"true":"false",framesDone,errors.serious.load(),errors.warnings.load(),initMs,ms(totalStart,Clock::now()),steps.mean(),steps.quantile(.95),steps.quantile(1),reads.mean(),observed.mean(),penetrations.quantile(1),penetrations.quantile(.95),penetrations.quantile(.99),penetrationFrames,penetrationFrames*double(c.dt),longestRun*double(c.dt));
        number(minDet);std::printf(",\"minBodyVolumeRatio\":");number(minVolume);std::printf(",\"maxVertexSpeed\":");number(maxSpeed);
        std::printf(",\"maxKineticGravityEnergy\":");number(maxEnergy);std::printf(",\"invertedFrames\":%u,\"momentumChangeMax\":",invertedFrames);number(maxMomentumChange);
        std::printf(",\"momentumInterpretation\":\"reaction-case-zero-gravity-other-cases-have-ground-and-gravity\",\"pairGeometryScope\":\"cross-body-collision-tetrahedra-SAT-MTD-not-union-depth\",\"pairGeometryAllValid\":%s,\"measurementOk\":%s,\"maxTetMtd\":%.12g,\"tetMtdFrameP95\":%.12g,\"tetMtdFrameP99\":%.12g,\"pairOverlapFrames\":%u,\"pairOverlapDuration\":%.12g,\"pairOverlapLongestDuration\":%.12g,\"geometryMeanMs\":%.12g,\"activeFrames\":%u,\"activeStepMeanMs\":%.12g,\"activeStepP95Ms\":%.12g,",
            allGeometryValid?"true":"false",(ok&&allGeometryValid)?"true":"false",pairDepths.quantile(1),pairDepths.quantile(.95),pairDepths.quantile(.99),pairFrames,pairFrames*double(c.dt),pairLongestRun*double(c.dt),geometryTimes.mean(),unsigned(activeSteps.v.size()),activeSteps.mean(),activeSteps.quantile(.95));
        modules();std::puts("}");return ok&&allGeometryValid?0:1;
    }
};
int main(int argc,char** argv)
{
    try{App app(parse(argc,argv));const int result=app.run();app.shutdown();
        std::printf("{\"type\":\"cleanup\",\"errors\":%u,\"warnings\":%u,\"ok\":%s}\n",app.errors.serious.load(),app.errors.warnings.load(),app.errors.serious.load()?"false":"true");
        return result?result:(app.errors.serious.load()?1:0);}
    catch(const std::exception& e){std::printf("{\"type\":\"summary\",\"completed\":false,\"executionOk\":false,\"reason\":");js(e.what());std::puts("}");return 2;}
}
