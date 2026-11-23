// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_NODAL_SHELL_H
#define DY_AVBD_NODAL_SHELL_H
#include "foundation/PxMat33.h"
#include "foundation/PxMath.h"
#include <cmath>
#include <limits>

namespace physx { namespace Dy {
// New nodal arithmetic has one compile-time policy. Host buffers remain PxReal.
#ifndef PX_AVBD_NODAL_REAL
#define PX_AVBD_NODAL_REAL float
#endif
typedef PX_AVBD_NODAL_REAL AvbdNodalReal;
typedef PxVec3T<AvbdNodalReal> AvbdNodalVec3;
typedef PxMat33T<AvbdNodalReal> AvbdNodalMat33;

template<class T> struct AvbdNodalShellRestT
{
    enum { Membrane, Hinge };
    PxU32 kind=Membrane;
    PxVec3T<T> shape[3]={PxVec3T<T>(0),PxVec3T<T>(0),PxVec3T<T>(0)};
    T area=0,length=0,restCos=T(1),restSin=T(0);
    PxU32 arity()const{return kind==Membrane?3:4;}
};
typedef AvbdNodalShellRestT<AvbdNodalReal> AvbdNodalShellRest;
template<class T> struct AvbdNodalWorkT
{
    PxVec3T<T> cubic=PxVec3T<T>(0);
    T geometric=0,quartic=0;
    bool polynomial=false;
};
template<class T> struct AvbdNodalShellSampleT
{
    T energy=0,areaRatio=0;
    PxVec3T<T> gradient=PxVec3T<T>(0);
    PxMat33T<T> metric=PxMat33T<T>(PxZero);
    bool valid=false;
};
template<class T> inline PxMat33T<T> avbdNodalOuter(const PxVec3T<T>& a,const PxVec3T<T>& b)
{return PxMat33T<T>(a*b.x,a*b.y,a*b.z);}
template<class T> inline PxMat33T<T> avbdNodalSkew(const PxVec3T<T>& a)
{return PxMat33T<T>(PxVec3T<T>(0,a.z,-a.y),PxVec3T<T>(-a.z,0,a.x),PxVec3T<T>(a.y,-a.x,0));}

// With one moving vertex, the triangle normal is affine along the candidate.
// Test its analytical minimum, not just endpoint areas. This admits rotations
// while preventing a finite local update from passing through a collapsed face.
template<class T> inline bool avbdNodalTrianglePath(const PxVec3T<T>& startNormal,const PxVec3T<T>& endNormal)
{
    const auto delta=endNormal-startNormal;const T d=delta.magnitudeSquared();
    const T t=d>0?PxClamp(-startNormal.dot(delta)/d,T(0),T(1)):T(0);
    const T scale=PxMax(startNormal.magnitudeSquared(),endNormal.magnitudeSquared());
    const T epsilon=std::numeric_limits<T>::epsilon();
    return scale>0 && (startNormal+delta*t).magnitudeSquared()>T(64)*epsilon*epsilon*scale;
}

// With all vertices moving, the area vector is a quadratic Bezier curve.
// A strict separating plane for its three controls certifies the full step.
template<class T> inline bool avbdMembranePath(const PxVec3T<T> (&x)[3],const PxVec3T<T> (&y)[3])
{
    const auto e=x[1]-x[0],f=x[2]-x[0],g=y[1]-y[0],h=y[2]-y[0];
    const auto a=e.cross(f),b=(e.cross(h)+g.cross(f))*T(.5),c=g.cross(h);
    const T scale=PxMax(a.magnitudeSquared(),PxMax(b.magnitudeSquared(),c.magnitudeSquared()));
    const T margin=T(8)*std::numeric_limits<T>::epsilon()*scale;
    const auto admits=[&](const auto& p,const auto& q,const auto& r)
    {
        const auto axis=(p+r)*T(.5);
        return axis.dot(p)>margin && axis.dot(q)>margin && axis.dot(r)>margin;
    };
    if(admits(a,b,c))return true;
    const auto ab=(a+b)*T(.5),bc=(b+c)*T(.5),mid=(ab+bc)*T(.5);
    return admits(a,ab,mid) && admits(mid,bc,c);
}

// One element, one requested 3-DOF destination. Outside supports are read-only
// boundary values. A sentinel destination requests energy only, without a tangent.
// mu/lambda are plane-stress coefficients integrated through the sheet thickness.
template<class T> inline AvbdNodalShellSampleT<T> avbdSampleNodalShell(
    const AvbdNodalShellRestT<T>& rest,const PxVec3T<T>* x,T mu,T lambda,T bend,PxU32 corner,const T* weights=NULL,
    AvbdNodalWorkT<T>* work=NULL)
{
    typedef PxVec3T<T> V;typedef PxMat33T<T> M;
    AvbdNodalShellSampleT<T> out;
    for(PxU32 i=0;i<rest.arity();++i)if(!x[i].isFinite())return out;
    if(rest.kind==rest.Membrane)
    {
        if(!(rest.area>0 && mu>0 && mu+lambda>0))return out;
        // Differences avoid subtracting two large world-coordinate products.
        const V e=x[1]-x[0],f=x[2]-x[0];
        const V F0=e*rest.shape[1].x+f*rest.shape[2].x;
        const V F1=e*rest.shape[1].y+f*rest.shape[2].y;
        const T a=F0.magnitudeSquared()-T(1),b=F1.magnitudeSquared()-T(1),c=F0.dot(F1),trace=a+b;
        const T u=mu*rest.area,l=lambda*rest.area;
        out.areaRatio=F0.cross(F1).magnitude();
        if(!(out.areaRatio>T(0)))return out;
        out.energy=T(.25)*u*(a*a+b*b+T(2)*c*c)+T(.125)*l*trace*trace;
        if(corner<3 || weights)
        {
            V shape(0);if(weights){for(PxU32 i=0;i<3;++i)shape+=rest.shape[i]*weights[i];}
            else shape=rest.shape[corner];
            const T s=shape.x,t=shape.y;
            out.gradient=(F0*(u*a+T(.5)*l*trace)+F1*(u*c))*s+
                (F1*(u*b+T(.5)*l*trace)+F0*(u*c))*t;
            const V w=F0*s+F1*t;
            out.metric=(avbdNodalOuter(F0,F0)+avbdNodalOuter(F1,F1))*(u*(s*s+t*t))+
                avbdNodalOuter(w,w)*(u+l);
            if(work)
            {
                const T ss=s*s+t*t;
                work->polynomial=true;
                work->geometric=(u*a+T(.5)*l*trace)*s*s+(u*b+T(.5)*l*trace)*t*t+T(2)*u*c*s*t;
                work->cubic=w*((u+T(.5)*l)*ss);
                work->quartic=(T(.25)*u+T(.125)*l)*ss*ss;
            }
        }
    }
    else if(rest.kind==rest.Hinge)
    {
        if(!(bend>=0 && rest.length>0))return out;
        const V e=x[0]-x[2],f=x[1]-x[2],h=x[3]-x[2];
        V n0=h.cross(e),n1=f.cross(h);const T r0=n0.magnitude(),r1=n1.magnitude();
        if(!(r0>0 && r1>0))return out;n0/=r0;n1/=r1;
        const T length=h.magnitude();const V axis=h/length;
        const T cosine=n0.dot(n1),sine=axis.dot(n0.cross(n1)),k=bend*rest.length;
        const T dc=cosine-rest.restCos,ds=sine-rest.restSin;
        out.energy=T(.5)*k*(dc*dc+ds*ds);out.areaRatio=PxMin(r0,r1);
        if(corner<4 || weights)
        {
            T one[4]={};if(!weights){one[corner]=1;weights=one;}
            const T edge=weights[3]-weights[2];
            const V dTheta=n0*((edge*axis.dot(e)-(weights[0]-weights[2])*length)/r0)+
                n1*((edge*axis.dot(f)-(weights[1]-weights[2])*length)/r1);
            out.gradient=dTheta*(k*(sine*rest.restCos-cosine*rest.restSin));
            out.metric=avbdNodalOuter(dTheta,dTheta)*k;
        }
    }
    else return out;
    out.valid=PxIsFinite(out.energy) && out.gradient.isFinite() && out.metric.column0.isFinite() &&
        out.metric.column1.isFinite() && out.metric.column2.isFinite();return out;
}

// A true 3x3 SPD solve. Fixed work and nine matrix coefficients; no rotational
// inertia, dummy angular unknowns, global correction, or iterative inner solve.
template<class T> inline bool avbdFactorNodal3(const PxMat33T<T>& A,T (&L)[3][3])
{
    for(PxU32 i=0;i<3;++i)for(PxU32 j=0;j<=i;++j)
    {
        T value=A[j][i];for(PxU32 k=0;k<j;++k)value-=L[i][k]*L[j][k];
        if(i==j){if(!(value>0) || !PxIsFinite(value))return false;L[i][j]=std::sqrt(value);}
        else L[i][j]=value/L[j][j];
    }
    return true;
}
template<class T> inline bool avbdSolveFactoredNodal3(const T (&L)[3][3],const PxVec3T<T>& g,PxVec3T<T>& step)
{
    T y[3];
    for(PxU32 i=0;i<3;++i){T v=-g[i];for(PxU32 j=0;j<i;++j)v-=L[i][j]*y[j];y[i]=v/L[i][i];}
    for(PxU32 i=3;i-->0;){T v=y[i];for(PxU32 j=i+1;j<3;++j)v-=L[j][i]*step[j];step[i]=v/L[i][i];}
    return step.isFinite();
}
// Intersection of affine plane rows in the current one-node chart.
// Negative embedding weights give upper bounds on the same coordinate.
template<class T> struct AvbdNodalPlaneRangeT
{
    PxVec3T<T> axis=PxVec3T<T>(0);
    T low=-std::numeric_limits<T>::max(),high=std::numeric_limits<T>::max();
    bool parallel=true;
    void include(const PxVec3T<T>& n,T weight,T gap)
    {
        if(weight==0 || !parallel)return;
        if(axis.isZero())axis=n;
        if(axis==-n)weight=-weight;
        else if(axis!=n){parallel=false;return;}
        const T bound=-PxMax(T(0),gap)/weight;
        if(weight>0)low=PxMax(low,bound);else high=PxMin(high,bound);
    }
    void merge(const AvbdNodalPlaneRangeT& b)
    {
        if(!b.parallel){parallel=false;return;}
        if(b.axis.isZero() || !parallel)return;
        if(axis.isZero()){*this=b;return;}
        if(axis==b.axis){low=PxMax(low,b.low);high=PxMin(high,b.high);}
        else if(axis==-b.axis){low=PxMax(low,-b.high);high=PxMin(high,-b.low);}
        else parallel=false;
    }
};
typedef AvbdNodalPlaneRangeT<AvbdNodalReal> AvbdNodalPlaneRange;

template<class T> inline bool avbdSolveNodal3(const PxMat33T<T>& A,const PxVec3T<T>& g,PxVec3T<T>& step,
    const AvbdNodalPlaneRangeT<T>* range=NULL)
{
    T L[3][3];if(!avbdFactorNodal3(A,L) || !avbdSolveFactoredNodal3(L,g,step))return false;
    if(!range || !range->parallel || range->axis.isZero())return true;
    const T d=range->axis.dot(step),bound=PxClamp(d,range->low,range->high);
    if(d==bound)return true;
    PxVec3T<T> response;
    if(!avbdSolveFactoredNodal3(L,-range->axis,response))return false;
    const T mobility=range->axis.dot(response);
    if(!(mobility>0))return false;
    step+=response*((bound-d)/mobility);
    return step.isFinite();
}
// Eliminate the first translation in the coupled system [A B; B' C].
// Factor A once for the gradient and three coupling columns.
template<class T> inline bool avbdSolveNodalPair(const PxMat33T<T>& A,const PxMat33T<T>& B,const PxMat33T<T>& C,
    const PxVec3T<T>& ga,const PxVec3T<T>& gb,PxVec3T<T>& da,PxVec3T<T>& db,
    const AvbdNodalPlaneRangeT<T>* planes=NULL)
{
    using V=PxVec3T<T>;
    T L[3][3],schur[3][3];PxMat33T<T> transfer;
    if(!avbdFactorNodal3(A,L) || !avbdSolveFactoredNodal3(L,ga,da))return false;
    for(PxU32 i=0;i<3;++i)if(!avbdSolveFactoredNodal3(L,B[i],transfer[i]))return false;
    const auto Bt=B.getTranspose();
    if(!avbdFactorNodal3(C+Bt*transfer,schur) || !avbdSolveFactoredNodal3(schur,gb+Bt*da,db))return false;
    da+=transfer*db;
    if(!da.isFinite() || !planes)return da.isFinite();
    const bool used[]={planes[0].parallel && !planes[0].axis.isZero(),planes[1].parallel && !planes[1].axis.isZero()};
    const T z[]={planes[0].axis.dot(da),planes[1].axis.dot(db)};
    const auto clamp=[&](PxU32 i,T value){return PxClamp(value,planes[i].low,planes[i].high);};
    if((!used[0] || z[0]==clamp(0,z[0])) && (!used[1] || z[1]==clamp(1,z[1])))return true;
    // Columns of H^-1 J^T. The factors and the contact cross-block are shared
    // with the free step. Only the two normal coordinates are constrained.
    V colA[2]={V(0),V(0)},colB[2]={V(0),V(0)};
    T S[2][2]={};
    for(PxU32 i=0;i<2;++i)if(used[i])
    {
        if(i==0 && !avbdSolveFactoredNodal3(L,-planes[0].axis,colA[i]))return false;
        if(!avbdSolveFactoredNodal3(schur,i==1?-planes[1].axis:Bt*colA[i],colB[i]))return false;
        colA[i]+=transfer*colB[i];
        S[0][i]=planes[0].axis.dot(colA[i]);S[1][i]=planes[1].axis.dot(colB[i]);
        if(!(S[i][i]>0))return false;
    }
    T multiplier[2]={};
    if(!used[0] || !used[1])
    {
        const PxU32 i=used[0]?0u:1u;
        multiplier[i]=(clamp(i,z[i])-z[i])/S[i][i];
    }
    else
    {
        // Minimize on each finite side of the rectangle, including its
        // corners. At most four scalar candidates, independent of valence.
        T best=std::numeric_limits<T>::max();
        for(PxU32 i=0;i<2;++i)
        {
            const PxU32 j=1-i;
            const T slope=S[j][i]/S[i][i],conditional=S[j][j]-slope*S[j][i];
            if(!(conditional>0))return false;
            for(T bound:{planes[i].low,planes[i].high})
            {
                if(PxAbs(bound)==std::numeric_limits<T>::max())continue;
                const T distance=bound-z[i],other=z[j]+slope*distance;
                const T residual=clamp(j,other)-other;
                const T cost=distance*(distance/S[i][i])+residual*(residual/conditional);
                if(cost<best)
                {
                    best=cost;multiplier[j]=residual/conditional;
                    multiplier[i]=distance/S[i][i]-slope*multiplier[j];
                }
            }
        }
        if(best==std::numeric_limits<T>::max())return false;
    }
    da+=colA[0]*multiplier[0]+colA[1]*multiplier[1];
    db+=colB[0]*multiplier[0]+colB[1]*multiplier[1];
    return da.isFinite() && db.isFinite();
}
// Position response of a membrane at an inertial trial pose. The local Newton
// system is in the three independent Green strain coordinates (including
// engineering shear). A returned nodal force has zero net force and torque.
template<class T> inline bool avbdMembraneTrialForce(const AvbdNodalShellRestT<T>& rest,
    const PxVec3T<T> (&x)[3],const T (&mobility)[3],const PxVec3T<T> (&previous)[3],
    T mu,T lambda,PxVec3T<T> (&force)[3])
{
    typedef PxVec3T<T> V;typedef PxMat33T<T> M;
    const V e=x[1]-x[0],f=x[2]-x[0];
    const V a=e*rest.shape[1].x+f*rest.shape[2].x,b=e*rest.shape[1].y+f*rest.shape[2].y;
    const T u=rest.area*mu,l=rest.area*lambda;
    if(!(u>0 && u+l>0))return false;
    const T diag=(2*u+l)/(4*u*(u+l)),off=-l/(4*u*(u+l));
    M response(V(diag,off,0),V(off,diag,0),V(0,0,1/u));
    M gradient[3];V strain((a.dot(a)-1)/2,(b.dot(b)-1)/2,a.dot(b));
    for(PxU32 i=0;i<3;++i)
    {
        const T s=rest.shape[i].x,t=rest.shape[i].y;
        gradient[i]=M(a*s,b*t,a*t+b*s);
        response+=gradient[i].getTranspose()*gradient[i]*mobility[i];
        strain-=gradient[i].getTranspose()*previous[i];
    }
    V stress(0);
    if(!avbdSolveNodal3(response,strain,stress))return false;
    for(PxU32 i=0;i<3;++i)force[i]=gradient[i]*stress;
    return true;
}
struct AvbdMembranePrimalArgs
{
    const AvbdNodalShellRest& rest;
    const AvbdNodalVec3 (&position)[3];
    const AvbdNodalReal (&mobility)[3];
    const AvbdNodalVec3 (&previous)[3];
    AvbdNodalReal mu,lambda;
    AvbdNodalVec3 (&force)[3];
    bool solved=false;
};
struct AvbdNodalShellArgs
{
    const AvbdNodalShellRest* rest;
    const AvbdNodalVec3* positions;
    AvbdNodalReal mu,lambda,bending;
    PxU32 corner;
    AvbdNodalShellSampleT<AvbdNodalReal>* output;
    const AvbdNodalReal* weights=NULL; // rank-one local coordinate, never a second material
    AvbdNodalWorkT<AvbdNodalReal>* work=NULL;
};
struct AvbdNodalShellMaterial
{
    const AvbdNodalShellRest* rest;
    AvbdNodalReal mu,lambda,bending;
    PxVec3* previous=NULL;
    bool warm=false;
};
struct AvbdNodalSolveArgs
{
    const AvbdNodalMat33* metric;const AvbdNodalVec3* gradient;
    AvbdNodalVec3* step;bool solved=false;
    const AvbdNodalMat33* cross=NULL;
    const AvbdNodalMat33* otherMetric=NULL;
    const AvbdNodalVec3* otherGradient=NULL;
    AvbdNodalVec3* otherStep=NULL;
    const AvbdNodalPlaneRange* range=NULL;
};
} }
#endif
