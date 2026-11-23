// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_ELASTIC_FRAME_CURVATURE_H
#define DY_AVBD_ELASTIC_FRAME_CURVATURE_H
#include "DyAvbdElasticFrameMath.h"

namespace physx { namespace Dy
{
// Symmetric second variation of exp(theta) * v at theta=0, world-left.
template<class T> inline PxVec3T<T> finiteJointRotationSecond(const PxVec3T<T>& v,PxU32 j,PxU32 k)
{
    PxVec3T<T> r(T(0));r[j]+=T(.5)*v[k];r[k]+=T(.5)*v[j];
    if(j==k) r-=v;
    return r;
}

// Hessian of sum .5*weight*C^2 - bias*C, in (pA-pB, thetaA, thetaB).
// These are relative derivative coordinates, NOT a nine-DOF solver block.
// Values and first derivatives remain the same shared material source. The
// caller owns PSD projection and assembly into its existing Rigid6 blocks.
inline PxU32 finiteJointDerivativeIndex(PxU32 i,PxU32 body)
{return body==1 && i>=3?i+3:i;}
template<class T> inline void finiteJointMaterialGaussNewton(
    const FiniteJointCoordinates<T>& c,const T* weight,T* hessian,PxU32 body=2)
{
    const PxU32 count=body<2?6:9;
    for(PxU32 j=0;j<81;++j)hessian[j]=T(0);
    for(PxU32 row=0;row<FINITE_JOINT_ROWS;++row)
    {
        T J[9];for(PxU32 j=0;j<3;++j)
        {J[j]=c.linear[row][j];J[3+j]=c.angularA[row][j];J[6+j]=-c.angularB[row][j];}
        for(PxU32 a=0;a<count;++a)for(PxU32 b=a;b<count;++b)
        {const PxU32 j=finiteJointDerivativeIndex(a,body),k=finiteJointDerivativeIndex(b,body);
            hessian[9*j+k]+=weight[row]*J[j]*J[k];}
    }
    for(PxU32 j=0;j<9;++j)for(PxU32 k=0;k<j;++k)hessian[9*j+k]=hessian[9*k+j];
}
template<class T> inline void finiteJointMaterialHessian(
    const PxTransformT<T>& fa,const PxTransformT<T>& fb,
    const PxVec3T<T>& centerA,const PxVec3T<T>& centerB,
    const T* weight,const T* bias,T* hessian,
    const FiniteJointCoordinates<T>* coordinates=NULL,T* metric=NULL,PxU32 body=2)
{
    FiniteJointCoordinates<T> local;
    if(!coordinates)local=finiteJointCoordinates(fa,fb,centerA,centerB);
    const auto& c=coordinates?*coordinates:local;
    const PxU32 count=body<2?6:9;
    const PxVec3T<T> d=fa.p-fb.p,armA=fa.p-centerA,armB=fb.p-centerB;
    PxVec3T<T> axes[3],dd[9];
    for(PxU32 j=0;j<3;++j)
    {
        axes[j]=PxVec3T<T>(T(0));axes[j][j]=T(1);
        dd[j]=axes[j];dd[3+j]=axes[j].cross(armA);dd[6+j]=-axes[j].cross(armB);
    }
    finiteJointMaterialGaussNewton(c,weight,hessian,body);
    if(metric)for(PxU32 j=0;j<81;++j)metric[j]=hessian[j];
    for(PxU32 row=0;row<6;++row)
    {
        const PxVec3T<T> u=c.linear[row];const PxU32 owner=row<3?3:6;
        PxVec3T<T> du[9];for(PxU32 j=0;j<9;++j) du[j]=PxVec3T<T>(T(0));
        for(PxU32 j=0;j<3;++j) du[owner+j]=axes[j].cross(u);
        const T force=weight[row]*c.value[row]-bias[row];
        for(PxU32 a=0;a<count;++a) for(PxU32 b=a;b<count;++b)
        {
            const PxU32 j=finiteJointDerivativeIndex(a,body),k=finiteJointDerivativeIndex(b,body);
            T second=du[j].dot(dd[k])+du[k].dot(dd[j]);
            if(j>=owner && j<owner+3 && k>=owner && k<owner+3)
                second+=finiteJointRotationSecond(u,j-owner,k-owner).dot(d);
            if(j>=3 && k<6) second+=u.dot(finiteJointRotationSecond(armA,j-3,k-3));
            if(j>=6) second-=u.dot(finiteJointRotationSecond(armB,j-6,k-6));
            hessian[9*j+k]+=force*second;
        }
    }
    // q(thetaA,thetaB)=qa^-1 exp(-thetaA/2) exp(thetaB/2) qb.
    // Sign-even strain products avoid any shortest-arc branch at pi.
    const PxQuatT<T> raw=fa.q.getConjugate()*fb.q;
    const T inverseNorm=T(1)/raw.magnitude();const PxQuatT<T> relative=raw*inverseNorm;
    const T q[4]={relative.x,relative.y,relative.z,relative.w};T dq[6][4];
    for(PxU32 j=0;j<3;++j)
    {
        const PxQuatT<T> v=(fa.q.getConjugate()*PxQuatT<T>(axes[j].x,axes[j].y,axes[j].z,T(0))*fb.q)*(T(.5)*inverseNorm);
        const T a[4]={v.x,v.y,v.z,v.w};
        for(PxU32 t=0;t<4;++t) {dq[j][t]=-a[t];dq[3+j][t]=a[t];}
    }
    for(PxU32 aIndex=3;aIndex<count;++aIndex) for(PxU32 bIndex=aIndex;bIndex<count;++bIndex)
    {
        const PxU32 j=finiteJointDerivativeIndex(aIndex,body)-3,k=finiteJointDerivativeIndex(bIndex,body)-3;
        T d2q[4]={T(0),T(0),T(0),T(0)};
        if((j<3)==(k<3))
        {
            if(j==k) for(PxU32 t=0;t<4;++t) d2q[t]=-T(.25)*q[t];
        }
        else
        {
            const PxVec3T<T>& a=axes[j];const PxVec3T<T>& b=axes[k-3];
            const PxQuatT<T> v=(fa.q.getConjugate()*PxQuatT<T>(a.x,a.y,a.z,T(0))*PxQuatT<T>(b.x,b.y,b.z,T(0))*fb.q)*(-T(.25)*inverseNorm);
            d2q[0]=v.x;d2q[1]=v.y;d2q[2]=v.z;d2q[3]=v.w;
        }
        for(PxU32 a=0;a<3;++a) for(PxU32 b=0;b<4;++b)
        {
            const PxU32 row=6+4*a+b;
            const T second=T(2)*(d2q[a]*q[b]+q[a]*d2q[b]+dq[j][a]*dq[k][b]+dq[k][a]*dq[j][b]);
            hessian[9*(3+j)+3+k]+=(weight[row]*c.value[row]-bias[row])*second;
        }
    }
    for(PxU32 j=0;j<9;++j) for(PxU32 k=0;k<j;++k) hessian[9*j+k]=hessian[9*k+j];
}
} }
#endif
