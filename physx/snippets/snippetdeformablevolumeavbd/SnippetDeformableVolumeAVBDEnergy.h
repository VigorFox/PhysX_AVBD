// Independent public-state observer for the stable Neo-Hookean material.
// Keep this outside the timed simulate/fetch interval.
#pragma once
#include "PxPhysicsAPI.h"
#include <cmath>

inline double avbdProbeVolumeEnergy(const physx::PxTetrahedronMesh& mesh,
    const physx::PxVec4* positions, double young, double poisson, double& minJ)
{
    using namespace physx;
    const auto vec=[](const PxVec3& p){return PxVec3d(p.x,p.y,p.z);};
    const double mu=young/(2*(1+poisson)),lambda=young*poisson/((1+poisson)*(1-2*poisson));
    const bool shortIndices=mesh.getTetrahedronMeshFlags().isSet(PxTetrahedronMeshFlag::e16_BIT_INDICES);
    const auto* rest=mesh.getVertices();
    double energy=0;
    for(PxU32 t=0;t<mesh.getNbTetrahedrons();++t)
    {
        PxU32 ids[4];
        for(PxU32 j=0;j<4;++j)
            ids[j]=shortIndices?static_cast<const PxU16*>(mesh.getTetrahedrons())[4*t+j]:
                static_cast<const PxU32*>(mesh.getTetrahedrons())[4*t+j];
        const auto start=vec(rest[ids[0]]),current=vec(positions[ids[0]].getXYZ());
        const PxMat33d D(vec(rest[ids[1]])-start,vec(rest[ids[2]])-start,vec(rest[ids[3]])-start);
        const PxMat33d F=PxMat33d(vec(positions[ids[1]].getXYZ())-current,
            vec(positions[ids[2]].getXYZ())-current,vec(positions[ids[3]].getXYZ())-current)*D.getInverse();
        const double J=F.getDeterminant(),delta=J-1;
        const double norm=F.column0.magnitudeSquared()+F.column1.magnitudeSquared()+F.column2.magnitudeSquared();
        if(!std::isfinite(J) || !std::isfinite(norm)){minJ=0;return HUGE_VAL;}
        minJ=PxMin(minJ,J);
        energy+=std::abs(D.getDeterminant())/6*(.5*mu*(norm-3)-mu*delta+.5*(lambda+mu)*delta*delta);
    }
    return energy;
}
