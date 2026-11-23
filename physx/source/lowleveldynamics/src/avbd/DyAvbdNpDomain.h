// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_NP_DOMAIN_H
#define DY_AVBD_NP_DOMAIN_H

#include "foundation/PxArray.h"
#include "foundation/PxTransform.h"
#include "avbd/DyAvbdOgcProjection.h"

namespace physx { namespace Dy {

// Complete convex support sets in COM coordinates, compiled from CPU NP.
// The normal is a separating direction, not a selected contact point. All
// vertices participate, including vertices omitted by manifold reduction.
struct AvbdNpDomain
{
    PxArray<PxVec3d> points[2];
    PxTransform fixedPose[2];
    PxVec3d normal;
    double radius[2];
    double restDistance=0;
    bool fixedAxis=false; // An infinite plane has only its authored normal.
    // Optional complete support bundle in the existing row array. Fixed
    // endpoints need only one extreme; rotating endpoints keep every vertex.
    PxU32 rowStart=PX_MAX_U32,first[2]={0,0},count[2]={0,0};
};

PX_INLINE PxQuatd avbdNpRotation(const PxQuat& q)
{ return PxQuatd(q.x,q.y,q.z,q.w).getNormalized(); }

PX_INLINE PxVec3d avbdNpPointMotion(const PxVec3& p,const PxQuat& q,
    const PxVec3& start,const PxQuat& rotation,const PxVec3& local)
{
    // Evaluate the displacement itself; subtracting rounded world points
    // creates a world-position ULP staircase in the Coulomb slip coordinate.
    const PxVec3d r=avbdOgcDouble(local);
    return avbdOgcDouble(p)-avbdOgcDouble(start)+avbdNpRotation(q).rotate(r)-avbdNpRotation(rotation).rotate(r);
}

struct AvbdNpRange
{
    double lo,hi,scale;
};

PX_INLINE AvbdNpRange avbdNpSupportRange(const AvbdNpDomain& domain,
    PxU32 side,const PxTransform& pose,const PxVec3d* axis=NULL)
{
    const PxVec3d& normal=axis?*axis:domain.normal;
    AvbdNpRange out={DBL_MAX,-DBL_MAX,0};
    const PxQuatd q=avbdNpRotation(pose.q);
    const PxVec3d p=avbdOgcDouble(pose.p);
    const double center=normal.dot(p);
    const PxVec3d localNormal=q.rotateInv(normal);
    for(PxU32 i=0;i<domain.points[side].size();++i)
    {
        const PxVec3d& v=domain.points[side][i];
        const double projection=localNormal.dot(v);
        out.lo=PxMin(out.lo,projection);out.hi=PxMax(out.hi,projection);
        out.scale=PxMax(out.scale,v.magnitudeSquared());
    }
    out.lo+=center-domain.radius[side];out.hi+=center+domain.radius[side];
    out.scale=std::sqrt(out.scale)+PxAbs(normal.x*p.x)+PxAbs(normal.y*p.y)+
        PxAbs(normal.z*p.z)+domain.radius[side];
    return out;
}

PX_INLINE double avbdNpGap(const AvbdNpDomain& domain,const PxTransform* poses)
{
    return avbdNpSupportRange(domain,0,poses[0]).lo-
        avbdNpSupportRange(domain,1,poses[1]).hi;
}

// The same complete support coordinate used by the certificate. Rounded
// support radii are orientation independent; they must not rotate as anchors.
PX_INLINE double avbdNpGapGradient(const AvbdNpDomain& domain,const PxTransform* poses,PxVec3d* arms)
{
    double extremes[2]={DBL_MAX,-DBL_MAX};
    for(PxU32 side=0;side<2;++side)
    {
        const PxQuatd q=avbdNpRotation(poses[side].q);arms[side]=PxVec3d(0);
        for(const auto& point:domain.points[side])
        {
            const PxVec3d arm=q.rotate(point);const double value=domain.normal.dot(arm);
            if(side?value>extremes[side]:value<extremes[side]){extremes[side]=value;arms[side]=arm;}
        }
    }
    return domain.normal.dot(avbdOgcDouble(poses[0].p)-avbdOgcDouble(poses[1].p))+
        extremes[0]-extremes[1]-domain.radius[0]-domain.radius[1]-domain.restDistance;
}

PX_INLINE void avbdNpCompileFeatures(AvbdNpDomain& domain,const PxTransform* poses,const bool* rotates,PxU32 firstRow)
{
    domain.rowStart=firstRow;
    for(PxU32 s=0;s<2;++s)
    {
        domain.first[s]=0;domain.count[s]=domain.points[s].size();
        if(rotates[s])continue;
        const PxVec3d n=avbdNpRotation(poses[s].q).rotateInv(domain.normal);double extreme=s?-DBL_MAX:DBL_MAX;
        for(PxU32 v=0;v<domain.points[s].size();++v)
        {const double p=n.dot(domain.points[s][v]);if(s?p>extreme:p<extreme){extreme=p;domain.first[s]=v;}}
        domain.count[s]=1;
    }
}

PX_INLINE double avbdNpFeatureGradient(const AvbdNpDomain& domain,const PxTransform* poses,PxU32 feature,PxVec3d* arms)
{
    const PxU32 index[2]={domain.first[0]+feature/domain.count[1],domain.first[1]+feature%domain.count[1]};
    for(PxU32 s=0;s<2;++s)arms[s]=avbdNpRotation(poses[s].q).rotate(domain.points[s][index[s]]);
    return domain.normal.dot(avbdOgcDouble(poses[0].p)-avbdOgcDouble(poses[1].p)+arms[0]-arms[1])-
        domain.radius[0]-domain.radius[1]-domain.restDistance;
}

// A cubic in Bernstein form is bounded by its control values. Subdivision
// only tightens that bound; exhaustion rejects, never silently accepts.
PX_INLINE bool avbdNpNonnegativeCubic(const double* b,double roundoff,PxU32 depth=0)
{
    const double low=PxMin(PxMin(b[0],b[1]),PxMin(b[2],b[3]));
    if(low>=-roundoff) return true;
    if(b[0]<-roundoff || b[3]<-roundoff || depth==16) return false;
    const double a=(b[0]+b[1])*.5,c=(b[1]+b[2])*.5,d=(b[2]+b[3])*.5;
    const double e=(a+c)*.5,f=(c+d)*.5,g=(e+f)*.5;
    const double left[4]={b[0],a,e,g},right[4]={g,f,d,b[3]};
    return avbdNpNonnegativeCubic(left,roundoff,depth+1) &&
        avbdNpNonnegativeCubic(right,roundoff,depth+1);
}

// Numerator of rotation by an unnormalised quaternion. Its dependence on q
// is quadratic. This avoids sampling a finite rotational trajectory.
PX_INLINE PxVec3d avbdNpRotationNumerator(const PxQuatd& q,const PxVec3d& v)
{
    const PxVec3d u(q.x,q.y,q.z);
    return v*(q.w*q.w-u.dot(u))+u*(2*u.dot(v))+u.cross(v)*(2*q.w);
}

PX_INLINE PxVec3d avbdNpRotationDerivative(const PxQuatd& q,const PxQuatd& dq,const PxVec3d& v)
{
    const PxVec3d u(q.x,q.y,q.z),du(dq.x,dq.y,dq.z);
    return v*(2*(q.w*dq.w-u.dot(du)))+du*(2*u.dot(v))+u*(2*du.dot(v))+
        u.cross(v)*(2*dq.w)+du.cross(v)*(2*q.w);
}

// Full finite path in one common world axis. Multiplication by the positive
// quaternion norms makes the support gap a degree-five polynomial. Unlike
// a selected minimum vertex, all these facets have smooth endpoint gradients.
struct AvbdNpPathFeature
{
    double gap[5];PxVec3d angular[5][2];
    bool valid;
};
PX_INLINE double avbdNpBernstein5(const double* power,PxU32 k)
{
    double sum=power[0],ratio=1;
    for(PxU32 j=1;j<=k;++j){ratio*=double(k-j+1)/double(6-j);sum+=ratio*power[j];}
    return sum;
}
PX_INLINE PxVec3d avbdNpBernstein5(const PxVec3d* power,PxU32 k)
{
    PxVec3d sum=power[0];double ratio=1;
    for(PxU32 j=1;j<=k;++j){ratio*=double(k-j+1)/double(6-j);sum+=power[j]*ratio;}
    return sum;
}
// A path is affine in each local support vertex. Compile the shared pose
// polynomial once, then evaluate all vertex pairs without rebuilding it.
struct AvbdNpPathPolynomial
{
    struct Terms
    {
        PxVec3d gap[5],angular[5][2][3];
        Terms(){for(PxU32 k=0;k<5;++k){gap[k]=PxVec3d(0);
            for(PxU32 s=0;s<2;++s)for(PxU32 d=0;d<3;++d)angular[k][s][d]=PxVec3d(0);}}
    };
    AvbdNpPathFeature constant;
    Terms point[2];
    bool derivatives;
    double coefficientScale=1;
    AvbdNpPathPolynomial():derivatives(false)
    {constant.valid=false;for(PxU32 k=0;k<5;++k){constant.gap[k]=0;constant.angular[k][0]=constant.angular[k][1]=PxVec3d(0);}}
    AvbdNpPathPolynomial(const AvbdNpDomain& domain,const PxTransform* start,const PxTransform* end,bool gradients=true,const PxVec3d* proofDirection=NULL):AvbdNpPathPolynomial()
    {
    const PxVec3d& normal=proofDirection?*proofDirection:domain.normal;
    derivatives=gradients;constant.valid=true;
    double D[2][3],dD[2][3][3]={};PxVec3d R[2][3],dR[2][3][3];
    for(PxU32 s=0;s<2;++s)
    {
        const PxQuatd a=avbdNpRotation(start[s].q);PxQuatd b=avbdNpRotation(end[s].q);
        if(a.dot(b)<0)b=-b;const PxQuatd delta=b-a;
        D[s][0]=a.dot(a);D[s][1]=2*a.dot(delta);D[s][2]=delta.dot(delta);
        R[s][0]=avbdNpRotationNumerator(a.getConjugate(),normal);
        R[s][2]=avbdNpRotationNumerator(delta.getConjugate(),normal);
        R[s][1]=avbdNpRotationNumerator(b.getConjugate(),normal)-R[s][0]-R[s][2];
        if(derivatives)for(PxU32 d=0;d<3;++d)
        {
            PxVec3d axis(0);axis[d]=.5;const PxQuatd db=PxQuatd(axis.x,axis.y,axis.z,0)*b;
            dD[s][d][1]=2*a.dot(db);dD[s][d][2]=2*delta.dot(db);
            dR[s][d][0]=PxVec3d(0);
            dR[s][d][2]=avbdNpRotationDerivative(delta.getConjugate(),db.getConjugate(),normal);
            dR[s][d][1]=avbdNpRotationDerivative(b.getConjugate(),db.getConjugate(),normal)-dR[s][d][2];
        }
    }
    const double initial=normal.dot(avbdOgcDouble(start[0].p)-avbdOgcDouble(start[1].p))-
        domain.radius[0]-domain.radius[1]-domain.restDistance;
    const double travel=normal.dot(avbdOgcDouble(end[0].p)-avbdOgcDouble(start[0].p)-
        avbdOgcDouble(end[1].p)+avbdOgcDouble(start[1].p));
    struct Affine
    {
        double scalar[6];PxVec3d vertex[2][6];
        Affine(){for(PxU32 k=0;k<6;++k){scalar[k]=0;vertex[0][k]=vertex[1][k]=PxVec3d(0);}}
    } P,dP[2][3];
    double W[6]={},dW[2][3][6]={};
    for(PxU32 i=0;i<3;++i)for(PxU32 j=0;j<3;++j)
    {
        const PxU32 k=i+j;const double product=D[0][i]*D[1][j];
        P.scalar[k]+=initial*product;P.scalar[k+1]+=travel*product;W[k+1]+=product;
        P.vertex[0][k]+=R[0][i]*D[1][j];P.vertex[1][k]-=R[1][j]*D[0][i];
        if(derivatives)for(PxU32 d=0;d<3;++d)
        {
            const double a=dD[0][d][i]*D[1][j],b=D[0][i]*dD[1][d][j];
            dP[0][d].scalar[k]+=initial*a;dP[1][d].scalar[k]+=initial*b;
            dP[0][d].scalar[k+1]+=travel*a;dP[1][d].scalar[k+1]+=travel*b;
            dP[0][d].vertex[0][k]+=dR[0][d][i]*D[1][j];dP[0][d].vertex[1][k]-=R[1][j]*dD[0][d][i];
            dP[1][d].vertex[0][k]+=R[0][i]*dD[1][d][j];dP[1][d].vertex[1][k]-=dR[1][d][j]*D[0][i];
            dW[0][d][k+1]+=a;dW[1][d][k+1]+=b;
        }
    }
    for(PxU32 k=1;k<=5;++k)
    {
        const double weight=avbdNpBernstein5(W,k);
        if(!(weight>0)){constant.valid=false;return;}
        constant.gap[k-1]=avbdNpBernstein5(P.scalar,k)/weight;
        for(PxU32 s=0;s<2;++s)point[s].gap[k-1]=avbdNpBernstein5(P.vertex[s],k)/weight;
        constant.angular[k-1][0]=constant.angular[k-1][1]=PxVec3d(0);
        if(derivatives)for(PxU32 s=0;s<2;++s)for(PxU32 d=0;d<3;++d)
        {
            const double scale=avbdNpBernstein5(dW[s][d],k)/weight;
            constant.angular[k-1][s][d]=avbdNpBernstein5(dP[s][d].scalar,k)/weight-constant.gap[k-1]*scale;
            for(PxU32 v=0;v<2;++v)point[v].angular[k-1][s][d]=avbdNpBernstein5(dP[s][d].vertex[v],k)/weight-point[v].gap[k-1]*scale;
        }
    }
    }
    AvbdNpPathFeature evaluate(const AvbdNpDomain& domain,PxU32 feature) const
    {
        AvbdNpPathFeature out=constant;if(!out.valid)return out;
        const PxU32 ids[2]={domain.first[0]+feature/domain.count[1],domain.first[1]+feature%domain.count[1]};
        for(PxU32 v=0;v<2;++v)for(PxU32 k=0;k<5;++k)
        {
            const auto& x=domain.points[v][ids[v]];out.gap[k]+=point[v].gap[k].dot(x);
            if(derivatives)for(PxU32 s=0;s<2;++s)for(PxU32 d=0;d<3;++d)out.angular[k][s][d]+=point[v].angular[k][s][d].dot(x);
        }
        return out;
    }
};
PX_INLINE AvbdNpPathFeature avbdNpPathFeature(const AvbdNpDomain& domain,
    const PxTransform* start,const PxTransform* end,PxU32 feature,bool derivatives=true)
{
    return AvbdNpPathPolynomial(domain,start,end,derivatives).evaluate(domain,feature);
}

PX_INLINE double avbdNpPathGap(const AvbdNpDomain& domain,const PxTransform* start,const PxTransform* end)
{
    double gap=DBL_MAX;
    const AvbdNpPathPolynomial polynomial(domain,start,end,false);if(!polynomial.constant.valid)return -DBL_MAX;
    // Every control gap is c + a.vertexA + b.vertexB. Its complete Cartesian
    // minimum is the sum of the two independent minima. Enumerate every
    // support point; this changes neither facets nor their physical forces.
    for(PxU32 k=0;k<5;++k)
    {
        double value=polynomial.constant.gap[k];
        for(PxU32 s=0;s<2;++s)
        {
            double minimum=DBL_MAX;
            for(PxU32 i=0;i<domain.count[s];++i)
                minimum=PxMin(minimum,polynomial.point[s].gap[k].dot(domain.points[s][domain.first[s]+i]));
            if(minimum==DBL_MAX)return -DBL_MAX;
            value+=minimum;
        }
        gap=PxMin(gap,value);
    }
    return gap;
}

PX_INLINE bool avbdNpDomainPath(const AvbdNpDomain& domain,
    const PxTransform* start,const PxTransform* end)
{
    AvbdNpRange ranges[2][2];
    for(PxU32 k=0;k<2;++k) for(PxU32 s=0;s<2;++s)
        ranges[k][s]=avbdNpSupportRange(domain,s,k?end[s]:start[s]);
    const double scale=ranges[0][0].scale+ranges[0][1].scale+
        ranges[1][0].scale+ranges[1][1].scale;
    const double roundoff=128*DBL_EPSILON*PxMax(scale,DBL_MIN);
    for(PxU32 k=0;k<2;++k)
        if(!PxIsFinite(ranges[k][0].lo) || !PxIsFinite(ranges[k][1].hi) ||
            ranges[k][0].lo-ranges[k][1].hi < -roundoff) return false;
    if(domain.rowStart!=PX_MAX_U32)return avbdNpPathGap(domain,start,end)>=-roundoff;
    // A fixed-orientation support boundary translates affinely. Use that
    // exact boundary rather than inventing a moving midpoint clearance that
    // can reject a collision-free arc above a static floor.
    double plane[2];
    for(PxU32 k=0;k<2;++k)
        plane[k]=start[1].q==end[1].q?ranges[k][1].hi:
            start[0].q==end[0].q?ranges[k][0].lo:(ranges[k][0].lo+ranges[k][1].hi)*.5;
    for(PxU32 s=0;s<2;++s)
    {
        // Translation alone has an affine gap; endpoint support suffices.
        if(start[s].q==end[s].q) continue;
        const double sign=s?-1.0:1.0;
        const PxQuatd q0=avbdNpRotation(start[s].q);
        PxQuatd q1=avbdNpRotation(end[s].q);
        if(q0.dot(q1)<0) q1=-q1;
        const PxQuatd dq=q1-q0;
        const double d[3]={q0.dot(q0),2*q0.dot(dq),dq.dot(dq)};
        const double x0=sign*(domain.normal.dot(avbdOgcDouble(start[s].p))-plane[0])-domain.radius[s];
        const double dx=sign*(domain.normal.dot(avbdOgcDouble(end[s].p)-
            avbdOgcDouble(start[s].p))-(plane[1]-plane[0]));
        for(PxU32 i=0;i<domain.points[s].size();++i)
        {
            const PxVec3d& v=domain.points[s][i];
            const double r0=sign*domain.normal.dot(avbdNpRotationNumerator(q0,v));
            const double r2=sign*domain.normal.dot(avbdNpRotationNumerator(dq,v));
            const double r1=sign*domain.normal.dot(avbdNpRotationNumerator(q1,v))-r0-r2;
            const double c[4]={x0*d[0]+r0,x0*d[1]+dx*d[0]+r1,
                x0*d[2]+dx*d[1]+r2,dx*d[2]};
            const double b[4]={c[0],c[0]+c[1]/3,c[0]+2*c[1]/3+c[2]/3,
                c[0]+c[1]+c[2]+c[3]};
            if(!avbdNpNonnegativeCubic(b,roundoff)) return false;
        }
    }
    return true;
}

// Complete Cartesian minimum without Cartesian storage. The returned IDs
// are witnesses, not a replacement for the manifold's force/torque support.
struct AvbdNpQuery
{
    static const PxU32 PointLimit=256;
    double gap[5]={},roundoff=0,startGap=0;
    PxU32 point[5][2]={};
    PxU32 axisFrame=PX_MAX_U32,axesTested=1,pointVisits=0;
    PxU32 axisCandidates=0;
    PxVec3d proofAxis=PxVec3d(0);
    bool valid=false,accepted=false;
};
// A continuous separating axis may rotate with either finite convex body.
// Multiply the gap by both quaternion squared norms: translation is degree
// five, the relative support rotations degree four. No temporal sampling.
PX_INLINE AvbdNpPathPolynomial avbdNpTransportedPolynomial(const AvbdNpDomain& domain,
    const PxTransform* start,const PxTransform* end,PxU32 frame)
{
    AvbdNpPathPolynomial out;
    double D[2][3];PxVec3d R[2][3][3];
    for(PxU32 s=0;s<2;++s)
    {
        const PxQuatd a=avbdNpRotation(start[s].q);PxQuatd b=avbdNpRotation(end[s].q);
        if(a.dot(b)<0)b=-b;const PxQuatd delta=b-a;
        D[s][0]=a.dot(a);D[s][1]=2*a.dot(delta);D[s][2]=delta.dot(delta);
        for(PxU32 d=0;d<3;++d)
        {
            PxVec3d axis(0);axis[d]=1;
            R[s][d][0]=avbdNpRotationNumerator(a,axis);R[s][d][2]=avbdNpRotationNumerator(delta,axis);
            R[s][d][1]=avbdNpRotationNumerator(b,axis)-R[s][d][0]-R[s][d][2];
        }
    }
    const PxU32 other=1-frame;
    const PxVec3d local=avbdNpRotation(start[frame].q).rotateInv(domain.normal);
    PxVec3d normal[3];for(PxU32 k=0;k<3;++k)normal[k]=R[frame][0][k]*local.x+R[frame][1][k]*local.y+R[frame][2][k]*local.z;
    double scalar[6]={},weight[6]={};PxVec3d vertex[2][6];
    for(PxU32 s=0;s<2;++s)for(PxU32 k=0;k<6;++k)vertex[s][k]=PxVec3d(0);
    const PxVec3d position=avbdOgcDouble(start[0].p)-avbdOgcDouble(start[1].p);
    const PxVec3d travel=avbdOgcDouble(end[0].p)-avbdOgcDouble(end[1].p)-position;
    const double radius=domain.radius[0]+domain.radius[1]+domain.restDistance,sign=frame?-1.0:1.0;
    for(PxU32 i=0;i<3;++i)for(PxU32 j=0;j<3;++j)
    {
        const PxU32 k=i+j;const double product=D[0][i]*D[1][j];weight[k+1]+=product;
        scalar[k]-=radius*product;vertex[frame][k]+=local*(sign*product);
        scalar[k]+=normal[i].dot(position)*D[other][j];scalar[k+1]+=normal[i].dot(travel)*D[other][j];
        for(PxU32 d=0;d<3;++d)vertex[other][k][d]-=sign*normal[i].dot(R[other][d][j]);
    }
    out.constant.valid=true;
    for(PxU32 k=1;k<=5;++k)
    {
        const double w=avbdNpBernstein5(weight,k);if(!(w>0)){out.constant.valid=false;return out;}
        out.coefficientScale=PxMax(out.coefficientScale,1/w);
        out.constant.gap[k-1]=avbdNpBernstein5(scalar,k)/w;
        for(PxU32 s=0;s<2;++s)out.point[s].gap[k-1]=avbdNpBernstein5(vertex[s],k)/w;
    }
    return out;
}
struct AvbdNpControlMinimum
{
    enum { TileWidth=32 };
    double gap[5];PxU32 point[5];bool valid=true;
    AvbdNpControlMinimum(){for(PxU32 k=0;k<5;++k){gap[k]=DBL_MAX;point[k]=0;}}
    void merge(const AvbdNpControlMinimum& other)
    {
        valid=valid && other.valid;
        for(PxU32 k=0;k<5;++k)if(other.gap[k]<gap[k]){gap[k]=other.gap[k];point[k]=other.point[k];}
    }
};
// One support tile, five independent min reductions. A GPU warp can project
// one vertex per lane and reduce these five scalar/witness pairs; no derivative
// arrays, atomics, or pairwise vertex product are inputs to this leaf.
PX_INLINE AvbdNpControlMinimum avbdNpControlTile(const PxVec3d* coefficients,
    const PxVec3d* points,PxU32 begin,PxU32 count)
{
    AvbdNpControlMinimum result;
    if(count>AvbdNpControlMinimum::TileWidth){result.valid=false;return result;}
    for(PxU32 i=begin;i<begin+count;++i)
    {
        const PxVec3d& point=points[i];
        for(PxU32 k=0;k<5;++k)
        {
            const double value=coefficients[k].dot(point);
            result.valid=result.valid && PxIsFinite(value);
            if(value<result.gap[k]){result.gap[k]=value;result.point[k]=i;}
        }
    }
    return result;
}
PX_INLINE AvbdNpQuery avbdNpAxisQuery(const AvbdNpDomain& domain,
    const PxTransform* start,const PxTransform* end,PxU32 frame,const PxVec3d* axis=NULL)
{
    AvbdNpQuery result;result.axisFrame=frame;result.proofAxis=axis?*axis:domain.normal;
    if(axis && frame!=PX_MAX_U32)return result;
    for(PxU32 s=0;s<2;++s)if(domain.points[s].empty() ||
        domain.points[s].size()>AvbdNpQuery::PointLimit)return result;
    const AvbdNpPathPolynomial polynomial=frame==PX_MAX_U32?AvbdNpPathPolynomial(domain,start,end,false,axis):
        avbdNpTransportedPolynomial(domain,start,end,frame);
    if(!polynomial.constant.valid)return result;
    double scale=0,initial=0;bool endpoints=true;
    for(PxU32 k=0;k<2;++k)
    {
        const auto a=avbdNpSupportRange(domain,0,k?end[0]:start[0],axis);
        const auto b=avbdNpSupportRange(domain,1,k?end[1]:start[1],axis);
        if(k==0)initial=a.lo-b.hi-domain.restDistance;
        scale+=a.scale+b.scale;
        if(frame!=PX_MAX_U32)scale+=avbdOgcDouble(k?end[0].p:start[0].p).magnitude()+avbdOgcDouble(k?end[1].p:start[1].p).magnitude();
        endpoints=endpoints && PxIsFinite(a.lo) && PxIsFinite(b.hi);
    }
    // Transport adds two quadratic rotation products. 4096 covers their
    // fixed convolution/normalization operation count and control weights;
    // require a positive margin for this additional sufficient certificate.
    result.roundoff=(frame==PX_MAX_U32?128:4096*polynomial.coefficientScale)*DBL_EPSILON*PxMax(scale,DBL_MIN);
    // Two endpoint scans plus five control projections per complete support.
    // Reuse the initial range from this exact query, never across candidates.
    result.pointVisits=7*(domain.points[0].size()+domain.points[1].size());
    result.valid=endpoints && PxIsFinite(result.roundoff);
    result.accepted=result.valid;
    AvbdNpControlMinimum minimum[2];
    for(PxU32 s=0;s<2;++s)
    {
        for(PxU32 begin=0;begin<domain.points[s].size();begin+=AvbdNpControlMinimum::TileWidth)
        {
            const auto tile=avbdNpControlTile(polynomial.point[s].gap,domain.points[s].begin(),begin,
                PxMin(PxU32(AvbdNpControlMinimum::TileWidth),domain.points[s].size()-begin));
            minimum[s].merge(tile);
        }
        if(!minimum[s].valid){result.valid=result.accepted=false;return result;}
    }
    for(PxU32 k=0;k<5;++k)
    {
        const double value=polynomial.constant.gap[k]+minimum[0].gap[k]+minimum[1].gap[k];
        for(PxU32 s=0;s<2;++s)result.point[k][s]=minimum[s].point[k];
        result.gap[k]=value;result.valid=result.valid && PxIsFinite(value);
        result.accepted=result.accepted && value>=(frame==PX_MAX_U32?-result.roundoff:result.roundoff);
    }
    result.startGap=initial;
    result.accepted=result.valid && result.accepted && initial>=-result.roundoff;
    return result;
}
PX_INLINE AvbdNpQuery avbdNpCompleteQuery(const AvbdNpDomain& domain,
    const PxTransform* start,const PxTransform* end)
{
    AvbdNpQuery result=avbdNpAxisQuery(domain,start,end,PX_MAX_U32);
    if(result.accepted || !result.valid || domain.fixedAxis || result.startGap < -result.roundoff)return result;
    for(PxU32 s=0;s<2;++s)
    {
        if(start[s].q==end[s].q)continue;
        AvbdNpQuery candidate=avbdNpAxisQuery(domain,start,end,s);
        ++result.axesTested;result.pointVisits+=candidate.pointVisits;
        if(candidate.accepted){candidate.axesTested=result.axesTested;candidate.pointVisits=result.pointVisits;return candidate;}
    }
    // A DCD axis need not remain separating as finite bodies move around an
    // edge. Rank a fixed number of directions from the endpoint frames; the
    // best direction still needs the original full continuous certificate.
    PxVec3d basis[2][3],selected(0);double best=0;
    for(PxU32 s=0;s<2;++s)for(PxU32 d=0;d<3;++d)
    {PxVec3d v(0);v[d]=1;basis[s][d]=avbdNpRotation(end[s].q).rotate(v);}
    auto consider=[&](PxVec3d normal)
    {
        const double length=normal.magnitude();if(!(length>0) || !PxIsFinite(length))return;
        normal/=length;++result.axisCandidates;
        double forward=DBL_MAX,backward=DBL_MAX;
        for(PxU32 k=0;k<2;++k)
        {
            const auto a=avbdNpSupportRange(domain,0,k?end[0]:start[0],&normal);
            const auto b=avbdNpSupportRange(domain,1,k?end[1]:start[1],&normal);
            forward=PxMin(forward,a.lo-b.hi-domain.restDistance);
            backward=PxMin(backward,b.lo-a.hi-domain.restDistance);
        }
        result.pointVisits+=2*(domain.points[0].size()+domain.points[1].size());
        if(forward>best){best=forward;selected=normal;}
        if(backward>best){best=backward;selected=-normal;}
    };
    for(PxU32 s=0;s<2;++s)for(PxU32 d=0;d<3;++d)consider(basis[s][d]);
    for(PxU32 i=0;i<3;++i)for(PxU32 j=0;j<3;++j)consider(basis[0][i].cross(basis[1][j]));
    if(best>0)
    {
        auto candidate=avbdNpAxisQuery(domain,start,end,PX_MAX_U32,&selected);
        ++result.axesTested;result.pointVisits+=candidate.pointVisits;
        if(candidate.accepted)
        {candidate.axesTested=result.axesTested;candidate.pointVisits=result.pointVisits;
            candidate.axisCandidates=result.axisCandidates;return candidate;}
    }
    return result;
}

struct AvbdNpDomainArgs
{
    const AvbdNpDomain* domain;
    const PxTransform* start;
    const PxTransform* end;
    bool* accepted;
    AvbdNpQuery* query=NULL;
};
}}
#endif
