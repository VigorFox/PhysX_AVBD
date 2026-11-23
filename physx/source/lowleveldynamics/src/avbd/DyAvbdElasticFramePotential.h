// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_ELASTIC_FRAME_POTENTIAL_H
#define DY_AVBD_ELASTIC_FRAME_POTENTIAL_H
#include "DyAvbdElasticFrameCurvature.h"
#include "PxConstraintDesc.h"

namespace physx { namespace Dy {
// Frozen at the common step-start epoch, not persistent material history.
struct AvbdElasticFrameReference
{
    // Store both scaled projections directly: reconstructing B from A/q0
    // magnifies roundoff at a frozen high-damping reference.
    PxVec3d translation[2];
    PxQuatT<double> rotation;
};
// Reference and live projections must subtract in the same physical chart.
// Limit reassociation locally: a projection roundoff times an arbitrary
// damping coefficient is a real force, not a cross-run determinism issue.
#if PX_VC
#pragma float_control(precise,on,push)
#endif
inline double avbdFrameTranslation(const PxQuatT<double>& q,PxU32 axis,
    const PxVec3d& difference,PxVec3d& direction)
{
    PxVec3d basis(0);basis[axis]=0.7071067811865475244;
    direction=q.rotate(basis);return direction.dot(difference);
}
#if PX_VC
#pragma float_control(pop)
#endif
inline AvbdElasticFrameReference avbdPrepareFrameReference(const PxTransformT<double>* frames)
{
    AvbdElasticFrameReference result;const PxVec3d d=frames[0].p-frames[1].p;
    for(PxU32 b=0;b<2;++b)for(PxU32 i=0;i<3;++i)
    {PxVec3d direction;result.translation[b][i]=avbdFrameTranslation(frames[b].q,i,d,direction);}
    result.rotation=(frames[0].q.getConjugate()*frames[1].q).getNormalized();return result;
}

inline PxU32 avbdFramePackedIndex(PxU32 a,PxU32 b)
{return a*(13-a)/2+b-a;} // Upper triangle, 0 <= a <= b < 6.

// Select the output chart at compile time, without duplicating the potential.
// Packed endpoints use physical (p, world theta) coordinates, including B's
// translation sign. Coupled/reference requests retain the nine-coordinate map.
template<PxU32 Endpoint,bool Packed> struct AvbdFrameDerivativeLayout
{
    static_assert(!Packed || Endpoint<2,"A packed contribution has one endpoint");
    enum { Jacobians=Packed?6:9, Gradients=Packed?6:12, Matrices=Packed?21:81,
        Count=Endpoint<2?6:9 };
    static PxU32 coordinate(PxU32 a){return Packed?a:finiteJointDerivativeIndex(a,Endpoint);}
    static PxU32 canonical(PxU32 a){return Packed?finiteJointDerivativeIndex(a,Endpoint):a;}
    static double sign(PxU32 a){return Packed && Endpoint==1 && a<3?-1.0:1.0;}
    static PxU32 matrix(PxU32 a,PxU32 b)
    {return Packed?avbdFramePackedIndex(a,b):9*coordinate(a)+coordinate(b);}
};

// The same 18-coordinate potential, contracted before differentiation.
// Derivative coordinates are (pA-pB, world thetaA, world thetaB), not a
// nine-DOF physical block. Null outputs avoid unused derivative work.
template<PxU32 Endpoint,bool Packed,class Stiffness> inline double avbdElasticFramePotentialImpl(const Stiffness* stiffness,
    const double* damping,const AvbdElasticFrameReference& reference,
    const PxTransformT<double>* bodies,const PxTransformT<double>* frames,
    double* gradient=NULL,double* hessian=NULL,double* metric=NULL)
{
    typedef AvbdFrameDerivativeLayout<Endpoint,Packed> Layout;
    const auto& fa=frames[0];const auto& fb=frames[1];
    const PxVec3d difference=fa.p-fb.p,armA=fa.p-bodies[0].p,armB=fb.p-bodies[1].p;
    const PxU32 count=Layout::Count;
    if(gradient)for(PxU32 j=0;j<Layout::Gradients;++j)gradient[j]=0;
    if(hessian)for(PxU32 j=0;j<Layout::Matrices;++j)hessian[j]=0;
    if(metric)for(PxU32 j=0;j<Layout::Matrices;++j)metric[j]=0;
    PxVec3d axes[3],dd[Layout::Jacobians];
    if(gradient)
    {
        for(PxU32 j=0;j<3;++j){axes[j]=PxVec3d(0);axes[j][j]=1;}
        for(PxU32 c=0;c<Layout::Jacobians;++c)
        {
            const PxU32 j=Layout::canonical(c);
            dd[c]=j<3?axes[j]*Layout::sign(c):j<6?axes[j-3].cross(armA):-axes[j-6].cross(armB);
        }
    }
    double energy=0;PxVec3d forceByFrame[2]={PxVec3d(0),PxVec3d(0)};
    for(PxU32 row=0;row<6;++row)
    {
        const PxU32 axis=row%3,owner=row/3;
        PxVec3d u;
        const double C=avbdFrameTranslation(row<3?fa.q:fb.q,axis,difference,u),delta=C-reference.translation[row/3][axis];
        const double k=stiffness[axis],dh=damping[axis],force=k*C+dh*delta;
        energy+=.5*(k*C*C+dh*delta*delta);
        if(!gradient)continue;
        forceByFrame[owner]+=u*force;
        if(!hessian && !metric)continue;
        // J_theta = (arm +/- frame difference) x u. Form each vector once,
        // instead of rebuilding three basis rotations and two dot products
        // for every coordinate of every strain row.
        const PxVec3d angularA=(armA-(owner==0?difference:PxVec3d(0))).cross(u);
        const PxVec3d angularB=(-armB-(owner==1?difference:PxVec3d(0))).cross(u);
        double J[Layout::Jacobians];
        for(PxU32 c=0;c<Layout::Jacobians;++c)
        {
            const PxU32 j=Layout::canonical(c);
            J[c]=j<3?Layout::sign(c)*u[j]:j<6?angularA[j-3]:angularB[j-6];
        }
        for(PxU32 a=0;a<count;++a)for(PxU32 b=a;b<count;++b)
        {
            const PxU32 x=Layout::coordinate(a),y=Layout::coordinate(b);
            const PxU32 index=Layout::matrix(a,b);
            const double gn=(k+dh)*J[x]*J[y];
            if(metric)metric[index]+=gn;
            if(hessian)hessian[index]+=gn;
        }
    }
    if(gradient)
    {
        const PxVec3d force=forceByFrame[0]+forceByFrame[1];
        const PxVec3d angular[2]={armA.cross(force)-difference.cross(forceByFrame[0]),
            -armB.cross(force)-difference.cross(forceByFrame[1])};
        for(PxU32 j=0;j<3;++j)
        {
            if(Packed){gradient[j]=Endpoint?-force[j]:force[j];gradient[3+j]=angular[Endpoint][j];}
            else {gradient[j]=force[j];gradient[6+j]=-force[j];gradient[3+j]=angular[0][j];gradient[9+j]=angular[1][j];}
        }
        // The second variation is linear in force*u. Sum the six rows before
        // differentiation, leaving one independently owned correction per
        // upper entry, with the same world-left connection and arm terms.
        if(hessian)
        {
            PxVec3d du[Layout::Jacobians];
            for(PxU32 c=0;c<Layout::Jacobians;++c)
            {
                const PxU32 j=Layout::canonical(c);
                du[c]=j<3?PxVec3d(0):axes[(j-3)%3].cross(forceByFrame[j<6?0:1]);
            }
            for(PxU32 a=0;a<count;++a)for(PxU32 b=a;b<count;++b)
            {
                const PxU32 x=Layout::coordinate(a),y=Layout::coordinate(b);
                const PxU32 j=Layout::canonical(x),l=Layout::canonical(y);
                double second=du[x].dot(dd[y])+du[y].dot(dd[x]);
                if(j>=3 && l<6)
                    second+=finiteJointRotationSecond(forceByFrame[0],j-3,l-3).dot(difference)+
                        force.dot(finiteJointRotationSecond(armA,j-3,l-3));
                if(j>=6)
                    second+=finiteJointRotationSecond(forceByFrame[1],j-6,l-6).dot(difference)-
                        force.dot(finiteJointRotationSecond(armB,j-6,l-6));
                hessian[Layout::matrix(a,b)]+=second;
            }
        }
    }
    const PxQuatT<double> raw=fa.q.getConjugate()*fb.q;
    const double inverseNorm=1/raw.magnitude();const auto relative=raw*inverseNorm;
    const auto aligned=relative.dot(reference.rotation)<0?-reference.rotation:reference.rotation;
    const double q[4]={relative.x,relative.y,relative.z,relative.w};
    const double r[4]={aligned.x,aligned.y,aligned.z,aligned.w};
    double delta[4],v[4]={},weight[4]={},stableGradient[4]={};
    for(PxU32 i=0;i<4;++i)delta[i]=q[i]-r[i];
    for(PxU32 i=0;i<3;++i)
    {
        const double k=stiffness[3+i],dh=damping[3+i];
        energy+=2*k*q[i]*q[i];v[i]=dh*r[i];weight[i]=k+dh;
        stableGradient[i]+=4*k*q[i];
        // Product differences avoid subtracting large quadratic energies near rest.
        for(PxU32 j=0;j<4;++j)
        {
            const double c=delta[i]*q[j]+r[i]*delta[j];energy+=2*dh*c*c;
            if(gradient){stableGradient[i]+=4*dh*c*q[j];stableGradient[j]+=4*dh*c*q[i];}
        }
    }
    if(!gradient)return energy;
    double S[4][4],g[4]={},dq[3][4],weightedNorm=0;
    for(PxU32 i=0;i<4;++i)
    {
        weightedNorm+=weight[i]*q[i]*q[i];
        for(PxU32 j=0;j<4;++j)
        {S[i][j]=(i==j?weight[i]:0)-r[i]*v[j]-v[i]*r[j];g[i]+=4*S[i][j]*q[j];}
    }
    for(PxU32 j=0;j<3;++j)
    {
        const auto d=(fa.q.getConjugate()*PxQuatT<double>(axes[j].x,axes[j].y,axes[j].z,0)*fb.q)*(.5*inverseNorm);
        const double x[4]={d.x,d.y,d.z,d.w};double force=0;
        // The stable product-residual gradient differs from 4*S*q only by
        // a radial term on the unit sphere. Contract with its tangent here;
        // retain the quadratic g/H pair below for the second variation.
        for(PxU32 i=0;i<4;++i)
        {
            dq[j][i]=x[i];
            force+=stableGradient[i]*x[i];
        }
        if(Packed)gradient[3+j]+=Endpoint?force:-force;
        else {gradient[3+j]-=force;gradient[9+j]+=force;}
    }
    if(!hessian && !metric)return energy;
    // Contract in the three world rotation directions once. A/B derivatives
    // are opposite copies of these directions, not six independent tangents.
    // Each upper entry below owns its output; no row-wise matrix reduction or
    // atomics are required by a SIMT binding of this fixed-size contraction.
    double tangentH[3][3]={},tangentM[3][3]={},gdq[3]={},gq=0;
    if(hessian)
    {
        double Sd[3][4]={};
        for(PxU32 i=0;i<4;++i)
        {
            gq+=g[i]*q[i];
            for(PxU32 a=0;a<3;++a)
            {
                gdq[a]+=g[i]*dq[a][i];
                for(PxU32 k=0;k<4;++k)Sd[a][i]+=S[i][k]*dq[a][k];
            }
        }
        for(PxU32 a=0;a<3;++a)for(PxU32 b=a;b<3;++b)
        {
            for(PxU32 i=0;i<4;++i)tangentH[a][b]+=4*dq[a][i]*Sd[b][i];
            tangentH[b][a]=tangentH[a][b];
        }
    }
    if(metric)for(PxU32 a=0;a<3;++a)for(PxU32 b=a;b<3;++b)
    {
        for(PxU32 i=0;i<4;++i)tangentM[a][b]+=4*(weight[i]+weightedNorm)*dq[a][i]*dq[b][i];
        tangentM[b][a]=tangentM[a][b];
    }
    for(PxU32 a=3;a<count;++a)for(PxU32 b=a;b<count;++b)
    {
        const PxU32 x=Layout::coordinate(a)-3,y=Layout::coordinate(b)-3;
        const PxU32 j=Layout::canonical(x+3)-3,l=Layout::canonical(y+3)-3;
        const PxU32 index=Layout::matrix(a,b);
        const PxU32 axisA=j%3,axisB=l%3;
        const bool same=(j<3)==(l<3);
        const double sign=same?1.0:-1.0;
        if(metric)metric[index]+=sign*tangentM[axisA][axisB];
        if(!hessian)continue;
        // e_u e_v = -delta_uv + epsilon_uvk e_k. This retains the
        // antisymmetric cross-endpoint connection term of the world chart.
        const double second=axisA==axisB?-.25*sign*gq:
            (same?0.0:((axisA+1)%3==axisB?-.5:.5)*gdq[3-axisA-axisB]);
        hessian[index]+=sign*tangentH[axisA][axisB]+second;
    }
    if(!Packed)for(PxU32 j=0;j<9;++j)for(PxU32 k=0;k<j;++k)
    {if(hessian)hessian[9*j+k]=hessian[9*k+j];if(metric)metric[9*j+k]=metric[9*k+j];}
    return energy;
}
template<class Stiffness> inline double avbdElasticFramePotential(const Stiffness* stiffness,
    const double* damping,const AvbdElasticFrameReference& reference,
    const PxTransformT<double>* bodies,const PxTransformT<double>* frames,
    double* gradient=NULL,double* hessian=NULL,double* metric=NULL,PxU32 body=2)
{
    if(body==0)return avbdElasticFramePotentialImpl<0,false>(stiffness,damping,reference,bodies,frames,gradient,hessian,metric);
    if(body==1)return avbdElasticFramePotentialImpl<1,false>(stiffness,damping,reference,bodies,frames,gradient,hessian,metric);
    return avbdElasticFramePotentialImpl<2,false>(stiffness,damping,reference,bodies,frames,gradient,hessian,metric);
}
} }
#endif
