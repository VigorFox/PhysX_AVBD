// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_ELASTIC_FRAME_H
#define DY_AVBD_ELASTIC_FRAME_H
#include "DyAvbdElasticFrameData.h"
#include <cfloat>

namespace physx { namespace Dy {
struct AvbdElasticFrameEnergy
{
    double energy;
    bool valid;
};
struct AvbdElasticFrameContribution : AvbdElasticFrameEnergy
{
    double gradient[12],hessian[81],gaussNewton[81];
};
// One physical Rigid6 endpoint: symmetric matrices need only 21 entries.
struct AvbdElasticFrameBlockContribution : AvbdElasticFrameEnergy
{
    double gradient[6],hessian[21],gaussNewton[21];
};
// Only the explicit reference path requests a positive root and work pairs.
struct AvbdElasticFrameEvaluation : AvbdElasticFrameContribution
{
    double root[9][12],energyPairs[FINITE_JOINT_ROWS][2];
};
enum class AvbdElasticFrameRequest { Energy, Contribution, Reference, ReferenceGradient };
struct AvbdElasticFrameArgs
{
    const AvbdElasticFrameMaterial* material;
    const PxTransform* poses;
    AvbdElasticFrameRequest request;
    void* output;
    PxU32 derivativeBody=2; // 0/1: one endpoint block; 2: all coupled derivatives.
};
// Symmetric Jacobi eigensolve in derivative space, not a physical solver
// block. Pair-scaled tests preserve weak rotational modes beside stiff
// translations; no global eigenvalue/velocity floor is introduced.
inline bool avbdElasticFramePositiveRoot(double (&H)[81],double (&root)[9][12])
{
    double Q[9][9]={};for(PxU32 i=0;i<9;++i) Q[i][i]=1;
    bool converged=false;
    for(PxU32 sweep=0;sweep<64 && !converged;++sweep)
    {
        converged=true;
        for(PxU32 p=0;p<9;++p) for(PxU32 q=p+1;q<9;++q)
        {
            const double off=H[9*p+q],a=H[9*p+p],b=H[9*q+q];
            if(PxAbs(off)<=16*DBL_EPSILON*std::sqrt(PxAbs(a)*PxAbs(b))) continue;
            converged=false;
            const double tau=(b-a)/(2*off);
            const double t=std::copysign(1.0,tau)/(PxAbs(tau)+std::hypot(1.0,tau));
            const double c=1/std::sqrt(1+t*t),s=t*c;
            H[9*p+p]=a-t*off;H[9*q+q]=b+t*off;H[9*p+q]=H[9*q+p]=0;
            for(PxU32 k=0;k<9;++k)
            {
                if(k!=p && k!=q)
                {
                    const double x=H[9*k+p],y=H[9*k+q];
                    H[9*k+p]=H[9*p+k]=c*x-s*y;H[9*k+q]=H[9*q+k]=s*x+c*y;
                }
                const double x=Q[k][p],y=Q[k][q];Q[k][p]=c*x-s*y;Q[k][q]=s*x+c*y;
            }
        }
    }
    if(!converged) return false;
    for(PxU32 r=0;r<9;++r)
    {
        const double scale=std::sqrt(PxMax(0.0,H[9*r+r]));
        for(PxU32 j=0;j<3;++j)
        {
            root[r][j]=scale*Q[j][r];root[r][6+j]=-root[r][j];
            root[r][3+j]=scale*Q[3+j][r];root[r][9+j]=scale*Q[6+j][r];
        }
    }
    return true;
}
inline AvbdElasticFrameEvaluation avbdEvaluateElasticFrame(
    const AvbdElasticFrameMaterial& m,const PxTransform* poses,bool derivatives,bool curvature=true,bool positiveRoot=true,PxU32 derivativeBody=2)
{
    AvbdElasticFrameEvaluation result={};
    PxTransformT<double> bodies[2],frames[2];avbdElasticFrameSections(m.geometry,poses,bodies,frames);
    result.energy=avbdElasticFramePotential(m.parameters.stiffness,m.parameters.dampingOverDt,m.reference,bodies,frames,
        derivatives?result.gradient:NULL,derivatives && curvature?result.hessian:NULL,
        derivatives && !positiveRoot?result.gaussNewton:NULL,positiveRoot?2:derivativeBody);
    // Retain stable coordinate work differences for the explicit reference
    // program. Budget contributions and energy trials never allocate these.
    const auto c=finiteJointCoordinates<double,false>(frames[0],frames[1],bodies[0].p,bodies[1].p);
    const double q[4]={m.reference.rotation.x,m.reference.rotation.y,m.reference.rotation.z,m.reference.rotation.w};
    for(PxU32 r=0;r<FINITE_JOINT_ROWS;++r)
    {
        const PxU32 axis=finiteJointDriveIndex(r);
        const double initial=r<6?m.reference.translation[r/3][r%3]:2*q[(r-6)/4]*q[(r-6)%4];
        const double C=c.value[r],force=m.parameters.stiffness[axis]*C+m.parameters.dampingOverDt[axis]*(C-initial);
        result.energyPairs[r][0]=C;result.energyPairs[r][1]=force;
    }
    result.valid=PxIsFinite(result.energy);
    if(derivatives)
    {
        if(curvature)
        {
            if(positiveRoot)
            {double H[81];for(PxU32 i=0;i<81;++i)H[i]=result.hessian[i];result.valid &= avbdElasticFramePositiveRoot(H,result.root);}
        }
        for(PxU32 i=0;i<12;++i) result.valid &= PxIsFinite(result.gradient[i]);
        if(!positiveRoot)for(PxU32 i=0;i<81;++i)result.valid &= PxIsFinite(result.gaussNewton[i]) && PxIsFinite(result.hessian[i]);
        for(PxU32 r=0;r<9;++r) for(PxU32 j=0;j<12;++j) result.valid &= PxIsFinite(result.root[r][j]);
    }
    return result;
}
inline AvbdElasticFrameEnergy avbdEvaluateElasticFrameEnergy(const AvbdElasticFrameMaterial& m,const PxTransform* poses)
{
    PxTransformT<double> bodies[2],frames[2];avbdElasticFrameSections(m.geometry,poses,bodies,frames);
    AvbdElasticFrameEnergy result;
    result.energy=avbdElasticFramePotential(m.parameters.stiffness,m.parameters.dampingOverDt,m.reference,bodies,frames);
    result.valid=PxIsFinite(result.energy);return result;
}
inline void avbdEvaluateElasticFrameContribution(
    const AvbdElasticFrameMaterial& m,const PxTransformT<double>* bodies,
    AvbdElasticFrameContribution& result,PxU32 derivativeBody=2)
{
    PxTransformT<double> frames[2];avbdElasticFrameWorldFrames(m.geometry,bodies,frames);
    result.energy=avbdElasticFramePotential(m.parameters.stiffness,m.parameters.dampingOverDt,m.reference,bodies,frames,
        result.gradient,result.hessian,result.gaussNewton,derivativeBody);
    result.valid=PxIsFinite(result.energy);
    for(PxU32 i=0;i<12;++i)result.valid &= PxIsFinite(result.gradient[i]);
    for(PxU32 i=0;i<81;++i)result.valid &= PxIsFinite(result.hessian[i]) && PxIsFinite(result.gaussNewton[i]);
}
inline AvbdElasticFrameContribution avbdEvaluateElasticFrameContribution(
    const AvbdElasticFrameMaterial& m,const PxTransformT<double>* bodies,PxU32 derivativeBody=2)
{
    AvbdElasticFrameContribution result;
    avbdEvaluateElasticFrameContribution(m,bodies,result,derivativeBody);return result;
}
inline AvbdElasticFrameContribution avbdEvaluateElasticFrameContribution(
    const AvbdElasticFrameMaterial& m,const PxTransform* poses,PxU32 derivativeBody=2)
{
    const PxTransformT<double> bodies[2]={avbdElasticFrameDouble(poses[0]),avbdElasticFrameDouble(poses[1])};
    return avbdEvaluateElasticFrameContribution(m,bodies,derivativeBody);
}
inline void avbdEvaluateElasticFrameBlock(
    const AvbdElasticFrameMaterial& m,const PxTransformT<double>* bodies,PxU32 endpoint,
    AvbdElasticFrameBlockContribution& result)
{
    if(endpoint>1){result.energy=0;result.valid=false;return;}
    PxTransformT<double> frames[2];avbdElasticFrameWorldFrames(m.geometry,bodies,frames);
    result.energy=endpoint==0?
        avbdElasticFramePotentialImpl<0,true>(m.parameters.stiffness,m.parameters.dampingOverDt,m.reference,bodies,frames,
            result.gradient,result.hessian,result.gaussNewton):
        avbdElasticFramePotentialImpl<1,true>(m.parameters.stiffness,m.parameters.dampingOverDt,m.reference,bodies,frames,
            result.gradient,result.hessian,result.gaussNewton);
    result.valid=PxIsFinite(result.energy);
    for(PxU32 i=0;i<6;++i)result.valid &= PxIsFinite(result.gradient[i]);
    for(PxU32 i=0;i<21;++i)result.valid &= PxIsFinite(result.hessian[i]) && PxIsFinite(result.gaussNewton[i]);
}
inline AvbdElasticFrameBlockContribution avbdEvaluateElasticFrameBlock(
    const AvbdElasticFrameMaterial& m,const PxTransformT<double>* bodies,PxU32 endpoint)
{
    AvbdElasticFrameBlockContribution result;
    avbdEvaluateElasticFrameBlock(m,bodies,endpoint,result);return result;
}
} }
#endif
