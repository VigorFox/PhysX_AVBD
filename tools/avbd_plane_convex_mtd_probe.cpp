// The public penetration observer must include the convex instance scale.
#include "PxPhysicsAPI.h"
#include <cstdio>
using namespace physx;
int main()
{
    PxDefaultAllocator allocator;PxDefaultErrorCallback errors;
    PxFoundation* foundation=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors);
    PxPhysics* physics=PxCreatePhysics(PX_PHYSICS_VERSION,*foundation,PxTolerancesScale());
    PxVec3 vertices[8];for(PxU32 v=0;v<8;++v) vertices[v]=PxVec3(v&1?.2f:-.2f,v&2?.3f:-.3f,v&4?.15f:-.15f);
    PxConvexMeshDesc desc;desc.points.count=8;desc.points.stride=sizeof(PxVec3);desc.points.data=vertices;desc.flags=PxConvexFlag::eCOMPUTE_CONVEX;
    PxConvexMesh* mesh=PxCreateConvexMesh(PxCookingParams(physics->getTolerancesScale()),desc,physics->getPhysicsInsertionCallback());
    if(!mesh) return 2;
    unsigned failures=0;
    for(PxU32 test=0;test<16;++test)
    {
        const PxVec3 n=(test&8)?PxVec3(1,2,3).getNormalized():PxVec3(0,1,0);
        const PxPlane plane(n,-.2f);const PxTransform planePose=PxTransformFromPlaneEquation(plane);
        const PxMeshScale scale((test&4)?PxVec3(.7f,1.1f,1.2f):PxVec3(1.7f),PxQuat(.4f,PxVec3(0,0,1)));
        PxConvexMeshGeometry geometry(mesh,scale);
        PxTransform pose(PxVec3(0),PxQuat(.9f,PxVec3(1,3,2).getNormalized()));
        PxReal minimum=PX_MAX_F32;
        for(PxU32 i=0;i<mesh->getNbVertices();++i)
            minimum=PxMin(minimum,plane.distance(pose.transform(scale.toMat33()*mesh->getVertices()[i])));
        const PxReal desired=(test&1)?-.03f:.1f;
        pose.p+=n*(desired-minimum);
        PxVec3 direction;PxReal depth=0;
        const bool overlap=(test&2)?PxGeometryQuery::computePenetration(direction,depth,geometry,pose,PxPlaneGeometry(),planePose):
            PxGeometryQuery::computePenetration(direction,depth,PxPlaneGeometry(),planePose,geometry,pose);
        const bool pass=overlap==(desired<0) && (!overlap || PxAbs(depth+desired)<2.e-6f);
        failures+=!pass;
        std::printf("[PLANE_CONVEX_MTD] case=%u overlap=%u depth=%.9g expected=%.9g status=%s\n",test,unsigned(overlap),double(depth),double(PxMax(0.f,-desired)),pass?"PASS":"FAIL");
    }
    mesh->release();physics->release();foundation->release();return failures?1:0;
}
