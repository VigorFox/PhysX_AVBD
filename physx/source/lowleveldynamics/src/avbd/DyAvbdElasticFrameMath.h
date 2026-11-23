// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_ELASTIC_FRAME_MATH_H
#define DY_AVBD_ELASTIC_FRAME_MATH_H
#include "foundation/PxTransform.h"
#include <cmath>

namespace physx { namespace Dy
{


// One strain/Jacobian source for the PxReal shader and precision diagnostics.
// T changes arithmetic precision only, not the material or its coordinate chart.
template<class T> inline PxVec3T<T> finiteJointRotation(const PxQuatT<T>& a,const PxQuatT<T>& b)
{
    PxQuatT<T> q=(a.getConjugate()*b).getNormalized();if(q.w<0) q=-q;
    const PxVec3T<T> v=q.getImaginaryPart();const T s=v.magnitude();
    return s>0 ? v*(T(2*std::atan2(double(s),double(q.w)))/s) : PxVec3T<T>(T(0));
}
enum { FINITE_JOINT_ROWS = 18 };
inline PxU32 finiteJointDriveIndex(PxU32 row)
{ return row<6 ? row%3 : 3+(row-6)/4; }

template<class T> struct FiniteJointCoordinates
{
    T value[FINITE_JOINT_ROWS];
    PxVec3T<T> linear[FINITE_JOINT_ROWS],angularA[FINITE_JOINT_ROWS],angularB[FINITE_JOINT_ROWS];
};
template<class T,bool derivatives=true> inline FiniteJointCoordinates<T> finiteJointCoordinates(
    const PxTransformT<T>& fa,const PxTransformT<T>& fb,const PxVec3T<T>& centerA,const PxVec3T<T>& centerB)
{
    FiniteJointCoordinates<T> result={};
    const PxVec3T<T> difference=fa.p-fb.p;
    // Endpoint frames avoid the discontinuous shortest-geodesic midpoint at
    // pi. Equal energy weights retain exchange symmetry and the rest tangent.
    const T scale=T(0.7071067811865475244);
    for(PxU32 i=0;i<6;++i)
    {
        PxVec3T<T> e(T(0));e[i%3]=scale;
        const PxVec3T<T> axis=(i<3?fa.q:fb.q).rotate(e);
        if(derivatives)
        {
            result.linear[i]=axis;
            result.angularA[i]=(fa.p-centerA).cross(axis);
            result.angularB[i]=(fb.p-centerB).cross(axis);
            if(i<3) result.angularA[i]+=axis.cross(difference);
            else result.angularB[i]-=axis.cross(difference);
        }
        result.value[i]=axis.dot(difference);
    }
    // Even quaternion products are smooth on SO(3), independent of q's sign.
    // Sum_j (2*q_i*q_j)^2 = 4*q_i^2: positive arbitrary anisotropic stiffness,
    // unlike a weighted matrix trace which needs triangle inequalities.
    const PxQuatT<T> relative=(fa.q.getConjugate()*fb.q).getNormalized();
    const T q[4]={relative.x,relative.y,relative.z,relative.w};
    PxVec3T<T> derivative[4];
    if(derivatives)for(PxU32 axis=0;axis<3;++axis)
    {
        PxVec3T<T> world(T(0));world[axis]=1;
        const PxVec3T<T> u=fa.q.rotateInv(world),v=relative.getImaginaryPart();
        const PxVec3T<T> dv=(u*relative.w+u.cross(v))*T(.5);
        for(PxU32 j=0;j<3;++j) derivative[j][axis]=dv[j];
        derivative[3][axis]=-T(.5)*u.dot(v);
    }
    for(PxU32 i=0;i<3;++i)
        for(PxU32 j=0;j<4;++j)
        {
            const PxU32 row=6+4*i+j;
            if(derivatives)
            {
                result.linear[row]=PxVec3T<T>(T(0));
                result.angularA[row]=result.angularB[row]=-(derivative[i]*q[j]+derivative[j]*q[i])*T(2);
            }
            result.value[row]=T(2)*q[i]*q[j];
        }
    return result;
}
} }
#endif
