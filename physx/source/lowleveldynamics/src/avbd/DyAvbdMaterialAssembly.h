// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_MATERIAL_ASSEMBLY_H
#define DY_AVBD_MATERIAL_ASSEMBLY_H
#include "DyAvbdBlockGroup.h"
#include "DyAvbdElasticFrame.h"
#include "DyAvbdTetrahedron.h"

namespace physx { namespace Dy {
PX_FORCE_INLINE void avbdAddOuter(PxReal H[6][6],const PxVec3& jLin,const PxVec3& jAng,PxReal s)
{
    for(PxU32 i=0;i<3;++i) for(PxU32 j=0;j<3;++j)
    {
        H[i][j]+=s*jLin[i]*jLin[j];H[i][j+3]+=s*jLin[i]*jAng[j];
        H[i+3][j]+=s*jAng[i]*jLin[j];H[i+3][j+3]+=s*jAng[i]*jAng[j];
    }
}
// Transient numerical destination for the existing row program. Diagonal
// blocks and off-diagonal factors are stamped from the SAME material sample.
template<class Scalar,PxU32 FactorCapacity=16> struct AvbdRigidAssemblyT
{
    PxReal hessian[6][6];Scalar gradient[6];
    const PxU32* coupledRows;
    PxU32 count,member;
    AvbdGroupFactorT<FactorCapacity>* factors;
    bool consistent;
    const PxU32* factorOffsets;
    PxU32 lockedMask=0;
    void record(PxU32 row,const PxVec3& linear,const PxVec3& angular,PxReal curvature,PxU32 rank=0)
    {
        PxU32 lo=0,hi=count;
        while(lo<hi){const PxU32 mid=(lo+hi)/2;if(coupledRows[mid]<row)lo=mid+1;else hi=mid;}
        if(lo==count || coupledRows[lo]!=row)return;
        if(factorOffsets[lo]+rank>=factorOffsets[lo+1]){consistent=false;return;}
        AvbdGroupFactorT<FactorCapacity>& factor=factors[factorOffsets[lo]+rank-factorOffsets[0]];
        if(factor.curvature!=0 && factor.curvature!=curvature)consistent=false;
        factor.curvature=curvature;
        for(PxU32 k=0;k<3;++k)
        {
            factor.jacobian[member][k]=(lockedMask&(1u<<k))?0:linear[k];
            factor.jacobian[member][k+3]=(lockedMask&(1u<<(k+3)))?0:angular[k];
        }
    }
    void addFactor(PxU32 row,const PxVec3& linear,const PxVec3& angular,PxReal curvature,PxU32 rank=0)
    {avbdAddOuter(hessian,linear,angular,curvature);record(row,linear,angular,curvature,rank);}
};
typedef AvbdRigidAssemblyT<PxReal> AvbdRigidAssembly;

template<class Scalar> inline void avbdAssembleElasticFrame(PxU32 row,const AvbdElasticFrameEvaluation& e,
    const PxU32 endpoints[2],const PxU32* blocks,PxU32 count,AvbdRigidAssemblyT<Scalar>* output)
{
    for(PxU32 b=0;b<count;++b)
    {
        double gradient[6]={},root[9][6]={};bool incident=false;
        for(PxU32 s=0;s<2;++s)if(endpoints[s]==blocks[b])
        {
            incident=true;
            for(PxU32 d=0;d<6;++d)
            {
                gradient[d]+=e.gradient[6*s+d];
                for(PxU32 r=0;r<9;++r)root[r][d]+=e.root[r][6*s+d];
            }
        }
        if(!incident)continue;
        for(PxU32 d=0;d<6;++d)output[b].gradient[d]+=Scalar(gradient[d]);
        for(PxU32 r=0;r<9;++r)output[b].addFactor(row,
            PxVec3(PxReal(root[r][0]),PxReal(root[r][1]),PxReal(root[r][2])),
            PxVec3(PxReal(root[r][3]),PxReal(root[r][4]),PxReal(root[r][5])),1,r);
    }
}

template<class Scalar> inline void avbdAssembleTetrahedron(PxU32 row,const AvbdTetrahedronEvaluation& e,
    const PxU32 supports[4],const PxVec3 arms[4],const PxU32* blocks,PxU32 count,AvbdRigidAssemblyT<Scalar>* output)
{
    for(PxU32 b=0;b<count;++b)
    {
        PxVec3d linear(0),angular(0);bool incident=false;
        // Several points may belong to one body. Merge their Jacobians
        // BEFORE forming an outer product, including all within-body terms.
        for(PxU32 s=0;s<4;++s)if(supports[s]==blocks[b])
        {
            incident=true;const PxVec3d& j=e.jacobians[s];
            linear+=j;angular+=PxVec3d(arms[s].x,arms[s].y,arms[s].z).cross(j);
        }
        if(!incident)continue;
        output[b].addFactor(row,PxVec3(PxReal(linear.x),PxReal(linear.y),PxReal(linear.z)),
            PxVec3(PxReal(angular.x),PxReal(angular.y),PxReal(angular.z)),PxReal(e.curvature));
        const PxVec3d gl=linear*e.gradient,ga=angular*e.gradient;
        for(PxU32 d=0;d<3;++d){output[b].gradient[d]+=Scalar(gl[d]);output[b].gradient[d+3]+=Scalar(ga[d]);}
    }
}
} }
#endif
