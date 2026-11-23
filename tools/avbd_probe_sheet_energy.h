// Independent energy of the zero-bending, E*h = 20 kN/m headless sheet.
#pragma once
#include "PxPhysicsAPI.h"
inline double avbdProbeSheetEnergy(const physx::PxTriangleMesh& mesh,const physx::PxVec4* rest,
    const physx::PxVec4* p,const physx::PxVec4* v)
{
    using namespace physx;
    const PxU32 n=mesh.getNbVertices();
    double result=0;
    for(PxU32 i=0;i<n;++i)result+=.5*double(v[i].getXYZ().magnitudeSquared())/p[i].w;
    const bool shortIndices=mesh.getTriangleMeshFlags().isSet(PxTriangleMeshFlag::e16_BIT_INDICES);
    const double nu=.3,E=2.e4,mu=E/(2*(1+nu)),lambda=E*nu/(1-nu*nu);
    for(PxU32 t=0;t<mesh.getNbTriangles();++t)
    {
        PxU32 ids[3];for(PxU32 j=0;j<3;++j)ids[j]=shortIndices?static_cast<const PxU16*>(mesh.getTriangles())[3*t+j]:static_cast<const PxU32*>(mesh.getTriangles())[3*t+j];
        const auto vec=[](const PxVec4& a){return PxVec3d(a.x,a.y,a.z);};
        const PxVec3d a=vec(rest[ids[1]])-vec(rest[ids[0]]),b=vec(rest[ids[2]])-vec(rest[ids[0]]);
        const double length=a.magnitude(),twiceArea=a.cross(b).magnitude();
        const PxVec3d f0=(vec(p[ids[1]])-vec(p[ids[0]]))/length;
        const PxVec3d f1=(vec(p[ids[2]])-vec(p[ids[0]])-f0*(a.dot(b)/length))*(length/twiceArea);
        const double ex=.5*(f0.dot(f0)-1),ey=.5*(f1.dot(f1)-1),xy=.5*f0.dot(f1);
        result+=.5*twiceArea*(mu*(ex*ex+ey*ey+2*xy*xy)+.5*lambda*(ex+ey)*(ex+ey));
    }
    return result;
}
