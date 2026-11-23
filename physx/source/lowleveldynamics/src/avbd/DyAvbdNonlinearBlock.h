// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_NONLINEAR_BLOCK_H
#define DY_AVBD_NONLINEAR_BLOCK_H
#include "DyAvbdCurvatureMetric.h"
#include "DyAvbdElasticFrame.h"

namespace physx { namespace Dy {
// Under-relax the multiplier ascent by the row's inertial response. The
// augmented primal force still uses rho. No multiplier decay is introduced,
// so an unsaturated fixed point still requires C=0, independent of rho.
inline double avbdInertialDualStep(double rho,double mobility,double dt)
{return rho/(1+rho*dt*dt*mobility);}

// One actual nonlinear block sample. No pose, dual or integrator is owned here.
// K retains signed curvature; M supplies units only for nonpositive modes.
struct AvbdNonlinearBlock
{
    double K[36]={},M[36]={},gradient[6]={},step[6]={};
    PxU32 dimension=6,locked=0;
    bool solved=false;
    void solve()
    {
        solved=false;
        for(double& x:step)x=0;
        if(dimension!=3 && dimension!=6)return;
        double model[36],metric[36],rhs[6]={};
        for(PxU32 i=0;i<dimension;++i)
        {
            if(!std::isfinite(gradient[i]))return;
            rhs[i]=(locked&(1u<<i))?0:-gradient[i];
            for(PxU32 j=0;j<dimension;++j)
            {
                const PxU32 k=6*i+j;
                if(!std::isfinite(K[k]) || !std::isfinite(M[k]))return;
                const bool fixed=(locked&((1u<<i)|(1u<<j)))!=0;
                model[k]=fixed?(i==j?1:0):K[k];
                metric[k]=fixed?(i==j?1:0):M[k];
            }
        }
        solved=avbdRegularizeLocalCurvature(model,metric,dimension) &&
            avbdSolveSmallBlock(model,rhs,step,dimension);
    }
};

// Merge all occurrences of the target body before forming its local block.
// This also handles repeated endpoints in mathematical mechanism probes.
inline void avbdAccumulateFrameBlock(const AvbdElasticFrameContribution& e,
    const PxU32 ends[2],PxU32 target,AvbdNonlinearBlock& block,bool useGaussNewton=true)
{
    for(PxU32 a=0;a<2;++a)if(ends[a]==target)for(PxU32 i=0;i<6;++i)
    {
        block.gradient[i]+=e.gradient[6*a+i];
        for(PxU32 b=0;b<2;++b)if(ends[b]==target)for(PxU32 j=0;j<6;++j)
        {
            const double sign=(a && i<3?-1:1)*(b && j<3?-1:1);
            const PxU32 k=9*(i<3?i:i+3*a)+(j<3?j:j+3*b);
            block.K[6*i+j]+=sign*e.hessian[k];if(useGaussNewton)block.M[6*i+j]+=sign*e.gaussNewton[k];
        }
    }
}
inline void avbdAccumulateFrameBlock(const AvbdElasticFrameEvaluation& e,
    const PxU32 ends[2],PxU32 target,AvbdNonlinearBlock& block,bool useGaussNewton=false)
{
    avbdAccumulateFrameBlock(static_cast<const AvbdElasticFrameContribution&>(e),ends,target,block,useGaussNewton);
    if(useGaussNewton)return;
    double root[9][6]={};
    for(PxU32 a=0;a<2;++a)if(ends[a]==target)
        for(PxU32 i=0;i<6;++i)
        {
            for(PxU32 r=0;r<9;++r)root[r][i]+=e.root[r][6*a+i];
        }
    for(PxU32 r=0;r<9;++r)for(PxU32 i=0;i<6;++i)for(PxU32 j=0;j<6;++j)
        block.M[6*i+j]+=root[r][i]*root[r][j];
}
} }
#endif
