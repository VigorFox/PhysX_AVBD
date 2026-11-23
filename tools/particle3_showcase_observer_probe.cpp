// Analytic positive and negative controls for the published-state observer.
#include "PxPhysicsAPI.h"
#include <algorithm>
#include <vector>
#include <cstdio>
using namespace physx;
PxDeformableVolume* getVolumeAvbdRenderVolume(PxU32);
bool readTet(const PxTetrahedronMesh&,PxU32,PxU32 (&)[4]);
#include "../physx/snippets/snippetdeformablevolumeavbd/SnippetDeformableVolumeAVBDShowcaseGeometry.h"

int main()
{
    const auto tet=[](const PxTransform& pose)
    {
        ShowcaseGeometry::Hull h;
        const PxVec3 p[4]={PxVec3(0),PxVec3(1,0,0),PxVec3(0,1,0),PxVec3(0,0,1)};
        for(PxU32 i=0;i<4;++i)h.vertex(i,pose.transform(p[i]));
        h.tetrahedron();return h;
    };
    PxU32 failures=0;
    for(PxU32 i=0;i<2;++i)
    {
        const PxTransform pose=i?PxTransform(PxVec3(3,-2,1),PxQuat(.6f,PxVec3(1,2,3).getNormalized())):PxTransform(PxIdentity);
        const auto a=tet(pose);
        for(PxReal shift:{0.f,.2f,.4f})
        {
            const auto b=tet(pose*PxTransform(PxVec3(shift)));
            const double expected=PxMax(0.,(1.-3.*shift)/PxSqrt(3.));
            failures+=PxAbs(ShowcaseGeometry::depth(a,b)-expected)>2.e-6;
            failures+=PxAbs(ShowcaseGeometry::depth(b,a)-expected)>2.e-6;
        }
        ShowcaseGeometry::Hull box;box.box(pose,PxVec3(2));
        failures+=PxAbs(ShowcaseGeometry::depth(box,a)-2.)>2.e-6;
        ShowcaseGeometry::Body first,second;
        first.tets.resize(1);first.tets[0].hull=a;first.bounds=a.bounds;
        second.tets.resize(1);second.order.push_back(0);
        second.tets[0].hull=tet(pose*PxTransform(PxVec3(1.01f,0,0)));
        second.bounds=second.tets[0].hull.bounds;
        PxVec3 normal(0);
        failures+=!ShowcaseGeometry::impactNormal(first,second,.02f,normal) ||
            normal.dot(pose.rotate(PxVec3(-1,0,0)))<.98f;
        failures+=ShowcaseGeometry::impactNormal(first,second,.005f,normal);
    }
    std::printf("[FIVE_OBSERVER_CHECK] failures=%u status=%s\n",failures,failures?"FAIL":"PASS");
    return failures?1:0;
}
