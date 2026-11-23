// Public API transport check: NP full geometry -> unified candidate domain.
// Independent full-shape query observes the stored output pose.
#include "PxPhysicsAPI.h"
#include <cstdio>
#include <cstdlib>
using namespace physx;
struct GeometryErrors:PxErrorCallback
{
    unsigned count=0;
    void reportError(PxErrorCode::Enum code,const char* message,const char*,int) PX_OVERRIDE
    {
        if(code!=PxErrorCode::eDEBUG_INFO && code!=PxErrorCode::eDEBUG_WARNING)
        { ++count;std::printf("ERROR %s\n",message); }
    }
};
int main(int argc,char** argv)
{
    PxDefaultAllocator allocator;GeometryErrors errors;
    PxFoundation* foundation=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors);
    PxPhysics* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation,PxTolerancesScale(),false,NULL);
    PxInitExtensions(*physics,NULL);
    PxDefaultCpuDispatcher* dispatcher=PxDefaultCpuDispatcherCreate(2);
    PxMaterial* material=physics->createMaterial(0,0,0);
    PxVec3 vertices[8];for(unsigned i=0;i<8;++i)
        vertices[i]=PxVec3((i&1)?.2f:-.2f,(i&2)?.3f:-.3f,(i&4)?.15f:-.15f);
    PxConvexMeshDesc mesh;mesh.points.count=8;mesh.points.stride=sizeof(PxVec3);mesh.points.data=vertices;
    mesh.flags=PxConvexFlag::eCOMPUTE_CONVEX;
    PxConvexMesh* hull=PxCreateConvexMesh(PxCookingParams(physics->getTolerancesScale()),mesh,physics->getPhysicsInsertionCallback());
    unsigned failures=0;
    const unsigned first=argc>1?unsigned(std::atoi(argv[1])):0,last=argc>1?first+1:16;
    for(unsigned test=first;test<last;++test)
    {
        PxSceneDesc desc(physics->getTolerancesScale());desc.gravity=PxVec3(0);
        desc.cpuDispatcher=dispatcher;desc.filterShader=PxDefaultSimulationFilterShader;desc.solverType=PxSolverType::eAVBD;
        PxScene* scene=physics->createScene(desc);
        const bool convexTarget=test>=8;
        PxRigidStatic* plane=convexTarget?PxCreateStatic(*physics,PxTransform(PxVec3(0,-.25f,0)),PxBoxGeometry(2,.25f,2),*material):
            PxCreatePlane(*physics,PxPlane(0,1,0,0),*material);
        scene->addActor(*plane);
        PxRigidDynamic* body=physics->createRigidDynamic(PxTransform(PxVec3(0,.8f,0),PxQuat(.2f,PxVec3(0,0,1))));
        const unsigned kind=test%4;const bool disabled=(test%8)>=4;
        PxGeometryHolder geometry;
        if(kind==0) geometry.storeAny(PxBoxGeometry(.2f,.3f,.15f));
        if(kind==1) geometry.storeAny(PxSphereGeometry(.2f));
        if(kind==2) geometry.storeAny(PxCapsuleGeometry(.1f,.25f));
        if(kind==3) geometry.storeAny(PxConvexMeshGeometry(hull,PxMeshScale(PxVec3(.7f,1.1f,1.2f),PxQuat(.4f,PxVec3(0,0,1)))));
        PxShape* shape=physics->createShape(geometry.any(),*material);
        shape->setLocalPose(PxTransform(PxVec3(.1f,0,.05f),PxQuat(.3f,PxVec3(0,0,1))));
        shape->setContactOffset(1.f);shape->setRestOffset(0);body->attachShape(*shape);
        body->setCMassLocalPose(PxTransform(PxVec3(.05f,.1f,0),PxQuat(.1f,PxVec3(0,1,0))));
        body->setMass(1);body->setMassSpaceInertiaTensor(PxVec3(.1f));
        body->setLinearDamping(0);body->setAngularDamping(0);body->setSleepThreshold(0);
        body->setMaxContactImpulse(disabled?0.f:PX_MAX_F32);body->setLinearVelocity(PxVec3(0,-50,0));
        scene->addActor(*body);
        PxRigidDynamic* independent=NULL;
        if(argc>2)
        {
            independent=physics->createRigidDynamic(PxTransform(PxVec3(10,3,0)));
            independent->setMass(1);independent->setMassSpaceInertiaTensor(PxVec3(1));
            independent->setLinearVelocity(PxVec3(1,2,3));independent->setLinearDamping(0);
            independent->setAngularDamping(0);independent->setSleepThreshold(0);scene->addActor(*independent);
        }
        scene->simulate(1.f/60.f);const bool fetched=scene->fetchResults(true);
        PxVec3 direction;PxReal depth=0;PxShape* planeShape=NULL;plane->getShapes(&planeShape,1);
        const bool overlap=PxGeometryQuery::computePenetration(direction,depth,shape->getGeometry(),
            PxShapeExt::getGlobalPose(*shape,*body),planeShape->getGeometry(),PxShapeExt::getGlobalPose(*planeShape,*plane));
        const float moved=.8f-body->getGlobalPose().p.y;
        if(kind==3 && !convexTarget && !disabled)
        {
            const auto& meshGeometry=static_cast<const PxConvexMeshGeometry&>(shape->getGeometry());
            const PxTransform pose=PxShapeExt::getGlobalPose(*shape,*body);float low=PX_MAX_F32;
            for(PxU32 v=0;v<meshGeometry.convexMesh->getNbVertices();++v)
                low=PxMin(low,pose.transform(meshGeometry.scale.toMat33()*meshGeometry.convexMesh->getVertices()[v]).y);
            std::printf("[NP_GEOMETRY_VERTEX_ORACLE] minimumY=%.9g queryDepth=%.9g\n",double(low),double(depth));
        }
        const bool pass=fetched && (disabled?(overlap && depth>.05f && moved>.8f):(!overlap || depth<5.e-6f)) && moved>.1f;
        if(!pass)
        {
            const PxTransform pose=body->getGlobalPose();
            std::printf("[NP_GEOMETRY_POSE] kind=%u p=%.9g,%.9g,%.9g q=%.9g,%.9g,%.9g,%.9g\n",
                kind,pose.p.x,pose.p.y,pose.p.z,pose.q.x,pose.q.y,pose.q.z,pose.q.w);
        }
        std::printf("[NP_GEOMETRY_SCENE] kind=%u convexTarget=%u disabled=%u overlap=%u depth=%.9g moved=%.9g status=%s\n",
            kind,unsigned(convexTarget),unsigned(disabled),unsigned(overlap),double(depth),double(moved),pass?"PASS":"FAIL");
        if(independent)
        {
            const PxVec3 expected=PxVec3(10,3,0)+PxVec3(1,2,3)/60.f;
            const float error=(independent->getGlobalPose().p-expected).magnitude();
            const bool independentPass=error<2.e-6f;
            std::printf("[NP_COMPONENT_INDEPENDENCE] positionError=%.9g status=%s\n",double(error),independentPass?"PASS":"FAIL");
            failures+=!independentPass;independent->release();
        }
        failures+=!pass;shape->release();body->release();plane->release();scene->release();
    }
    hull->release();material->release();dispatcher->release();PxCloseExtensions();physics->release();foundation->release();
    return failures+errors.count?1:0;
}
