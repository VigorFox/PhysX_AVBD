// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Original non-associated Coulomb root correction. Numerical trials only.
#ifndef DY_AVBD_COULOMB_NCP_H
#define DY_AVBD_COULOMB_NCP_H
#include <algorithm>
#include <cmath>
#include <limits>
#include "foundation/PxArray.h"
#include "foundation/PxSimpleTypes.h"

namespace physx { namespace Dy {

// Numerical operator view, never a second physics representation. Each group
// contains either one normal plus two tangents, or three scalar bounded rows.
// Scalar padding uses zero J and force bounds [0,0]. apply() must use the FULL
// J H^-1 J^T response; a local block inverse is not an equivalent operator.
// Compliance is physical and explicit. Trials do not publish pose/dual/BDF.
struct AvbdCoulombNcpResult
{
    bool converged;
    PxU32 iterations,linearIterations,sweeps;
    double residual;
};
struct AvbdCoulombNcpArgs
{
    void* context;
    void (*apply)(void*,const double*,double*);
    PxU32 groups,iterations;
    const double *demand,*mu,*lower,*upper,*compliance;
    const int* coulomb; // NULL: all groups are Coulomb; otherwise 0 or 1
    double *force,*work;
    AvbdCoulombNcpResult* result;
};
inline PxU32 avbdCoulombNcpWorkSize(PxU32 groups) {return 82*groups;}

// Evaluate f-clamp(f-shift) without erasing a small shift in a large f.
inline double avbdScalarProjectionRoot(double f,double shift,double lo,double hi)
{return std::max(f-hi,std::min(f-lo,shift));}

// Original (actual normal force) root, also used at nonlinear stored poses.
inline void avbdCoulombRoot(const double* f,const double* u,double rho,double mu,
    const double* lo,const double* hi,bool friction,double* residual)
{
    for(int q=0;q<3;++q)residual[q]=avbdScalarProjectionRoot(f[q],rho*u[q],lo[q],hi[q]);
    if(friction)
    {
        const double a=f[1]-rho*u[1],b=f[2]-rho*u[2];
        const double radius=mu*std::max(0.,f[0]),length=std::hypot(a,b);
        const double excess=length>radius?(length-radius)/length:0.,scale=length>radius?radius/length:1.;
        residual[1]=excess*f[1]+scale*(rho*u[1]);residual[2]=excess*f[2]+scale*(rho*u[2]);
    }
}

namespace AvbdCoulombNcp {
// Derivatives of the ORIGINAL actual-normal-force root above: dr=A du+B df.
// Compliance belongs to du, not to a modified force cone.
inline void rootDerivative(const double* f,const double* u,double rho,double mu,
    const double* lo,const double* hi,bool friction,double* A,double* B)
{
    std::fill(A,A+9,0.);std::fill(B,B+9,0.);
    for(int q=0;q<3;++q)
    {const double v=f[q]-rho*u[q];if(v>lo[q] && v<hi[q])A[4*q]=rho;else B[4*q]=1.;}
    if(!friction)return;
    const double v[2]={f[1]-rho*u[1],f[2]-rho*u[2]},length=std::hypot(v[0],v[1]);
    const double radius=mu*std::max(0.,f[0]);double D[4]={1,0,0,1};
    if(length>radius)
    {
        const double a=v[0]/length,b=v[1]/length,scale=radius/length;
        D[0]=scale*(1-a*a);D[1]=D[2]=-scale*a*b;D[3]=scale*(1-b*b);
        if(f[0]>0){B[3]=-mu*a;B[6]=-mu*b;}
    }
    for(int i=0;i<2;++i)for(int j=0;j<2;++j)
    {A[3*(i+1)+j+1]=rho*D[2*i+j];B[3*(i+1)+j+1]=(i==j?1.:0.)-D[2*i+j];}
}
struct WorkSpan
{
    double* values;int count;
    WorkSpan(int size,double*& cursor):values(cursor),count(size)
    {cursor+=size;std::fill(values,values+size,0.);}
    double* data(){return values;}
    double* begin(){return values;}
    double* end(){return values+count;}
    double& operator[](int i){return values[i];}
    const double& operator[](int i) const {return values[i];}
};
inline double norm(const double* x,int n)
{double r=0;for(int i=0;i<n;++i)r=std::hypot(r,x[i]);return r;}
inline double dot(const double* x,const double* y,int n)
{double r=0;for(int i=0;i<n;++i)r+=x[i]*y[i];return r;}
inline double largestEigenvalue(const double* a)
{
    const double q=(a[0]+a[4]+a[8])/3;
    const double p=std::sqrt(((a[0]-q)*(a[0]-q)+(a[4]-q)*(a[4]-q)+(a[8]-q)*(a[8]-q)+
        2*(a[1]*a[1]+a[2]*a[2]+a[5]*a[5]))/6);
    if(p==0)return q;
    double b[9];for(int i=0;i<9;++i)b[i]=(a[i]-(i%4==0?q:0))/p;
    const double determinant=b[0]*(b[4]*b[8]-b[5]*b[7])-b[1]*(b[3]*b[8]-b[5]*b[6])+b[2]*(b[3]*b[7]-b[4]*b[6]);
    return q+2*p*std::cos(std::acos(std::max(-1.,std::min(1.,determinant*.5)))/3);
}

template<class Operator> int lsqr(Operator& op,int n,const double* rhs,double* x,
    double* work,int budget,double forcing)
{
    double *u=work,*v=u+n,*w=v+n,*av=w+n,*atu=av+n,*r=atu+n;
    for(int i=0;i<n;++i){u[i]=rhs[i];x[i]=0;}
    double beta=norm(u,n),bnorm=beta;if(beta==0)return 0;
    for(int i=0;i<n;++i)u[i]/=beta;
    op.transpose(u,v);double alpha=norm(v,n);if(alpha==0)return 0;
    for(int i=0;i<n;++i){v[i]/=alpha;w[i]=v[i];}
    double rhobar=alpha,phibar=beta,anorm2=alpha*alpha;
    for(int k=0;k<budget;++k)
    {
        op.apply(v,av);for(int i=0;i<n;++i)u[i]=av[i]-alpha*u[i];
        beta=norm(u,n);if(beta>0)for(int i=0;i<n;++i)u[i]/=beta;
        op.transpose(u,atu);for(int i=0;i<n;++i)v[i]=atu[i]-beta*v[i];
        alpha=norm(v,n);if(alpha>0)for(int i=0;i<n;++i)v[i]/=alpha;
        anorm2+=alpha*alpha+beta*beta;
        const double rho=std::hypot(rhobar,beta);if(!(rho>0) || !std::isfinite(rho))return k;
        const double c=rhobar/rho,s=beta/rho,theta=s*alpha,phi=c*phibar;
        rhobar=-c*alpha;phibar*=s;
        for(int i=0;i<n;++i){x[i]+=(phi/rho)*w[i];w[i]=v[i]-(theta/rho)*w[i];}
        if((k+1)%16==0 || k+1==budget || alpha==0 || beta==0)
        {
            op.apply(x,av);for(int i=0;i<n;++i)r[i]=rhs[i]-av[i];
            op.transpose(r,atu);const double rn=norm(r,n),arn=norm(atu,n);
            if(rn<=forcing*bnorm || arn<=1.e-13*std::sqrt(anorm2)*rn || alpha==0 || beta==0)return k+1;
        }
    }
    return budget;
}

template<class Response> struct Solver
{
    Response& response;
    int m;
    const double *demand,*mu,*lower,*upper,*compliance;
    const int* coulomb;
    WorkSpan rho,diagonal,A,B,motion,r,direction,rhs,trial,trialR,
        temporary,dualTemporary,linearWork,columns,basis;
    int iterations=0,linearIterations=0,sweeps=0;
    double error=std::numeric_limits<double>::infinity();
    bool valid=true;
    bool friction(int c) const {return !coulomb || coulomb[c]!=0;}
    double lo(int i) const {return lower?lower[i]:(friction(i/3) && i%3==0?0.:-std::numeric_limits<double>::infinity());}
    double hi(int i) const {return upper?upper[i]:std::numeric_limits<double>::infinity();}
    double clip(double value,int i) const {return std::min(hi(i),std::max(lo(i),value));}
    void responseApply(const double* f,double* out)
    {response.apply(f,out);if(compliance)for(int i=0;i<m;++i)out[i]+=compliance[i]*f[i];}
    Solver(Response& op,int count,const double* offset,const double* coefficients,double* workspace,
        const int* kinds=NULL,const double* minForce=NULL,const double* maxForce=NULL,const double* softness=NULL):response(op),m(3*count),
        demand(offset),mu(coefficients),lower(minForce),upper(maxForce),compliance(softness),coulomb(kinds),
        rho(count,workspace),diagonal(9*count,workspace),A(9*count,workspace),B(9*count,workspace),
        motion(m,workspace),r(m,workspace),direction(m,workspace),rhs(m,workspace),trial(m,workspace),trialR(m,workspace),
        temporary(m,workspace),dualTemporary(m,workspace),linearWork(6*m,workspace),columns(3*m,workspace),basis(m,workspace)
    {
        for(int c=0;c<count;++c)
        {
            for(int j=0;j<3;++j)
            {
                const int i=3*c+j;
                if(std::isnan(lo(i)) || std::isnan(hi(i)) || lo(i)>hi(i) ||
                    lo(i)==std::numeric_limits<double>::infinity() || hi(i)==-std::numeric_limits<double>::infinity() ||
                    !std::isfinite(demand[i]) || (compliance && (!(compliance[i]>=0) || !std::isfinite(compliance[i]))))
                {valid=false;return;}
            }
            if(friction(c) && (lo(3*c)!=0 || hi(3*c)<0 ||
                std::isfinite(lo(3*c+1)) || std::isfinite(lo(3*c+2)) ||
                std::isfinite(hi(3*c+1)) || std::isfinite(hi(3*c+2)))){valid=false;return;}
            for(int j=0;j<3;++j)
            {
                basis[3*c+j]=1;responseApply(basis.data(),temporary.data());basis[3*c+j]=0;
                for(int i=0;i<3;++i)diagonal[9*c+3*i+j]=temporary[3*c+i];
            }
            const double eigenvalue=largestEigenvalue(diagonal.data()+9*c);
            // A zero-response normal or singular tangent needs a different
            // local resolvent. Reject this unsupported input, never divide by
            // zero and accidentally qualify a NaN candidate.
            const double* block=diagonal.data()+9*c;
            const bool fixed=!friction(c) && lo(3*c)==hi(3*c) && lo(3*c+1)==hi(3*c+1) && lo(3*c+2)==hi(3*c+2);
            if((!(eigenvalue>0) && !fixed) || !std::isfinite(eigenvalue) || !(mu[c]>=0) || !std::isfinite(mu[c]))
            {valid=false;return;}
            if(friction(c))
            {
                if(!(block[0]>0) || (mu[c]>0 && !(block[4]*block[8]-block[5]*block[7]>0)))
                {valid=false;return;}
            }
            else for(int j=0;j<3;++j)if(!(block[4*j]>0) && lo(3*c+j)!=hi(3*c+j)){valid=false;return;}
            rho[c]=eigenvalue>0?1/eigenvalue:1.;
        }
    }
    void evaluate(const double* f,double* residual,bool derivative)
    {
        responseApply(f,motion.data());for(int i=0;i<m;++i)motion[i]+=demand[i];
        if(derivative){std::fill(A.begin(),A.end(),0.);std::fill(B.begin(),B.end(),0.);}
        for(int c=0;c<m/3;++c)
        {
            if(!friction(c))
            {
                for(int q=0;q<3;++q)
                {
                    const int i=3*c+q;const double v=f[i]-rho[c]*motion[i];
                    residual[i]=f[i]-clip(v,i);
                    if(derivative){if(v>lo(i) && v<hi(i))A[9*c+4*q]=rho[c];else B[9*c+4*q]=1.;}
                }
                continue;
            }
            const int i=3*c;const double v0=f[i]-rho[c]*motion[i];
            const double v1=f[i+1]-rho[c]*motion[i+1],v2=f[i+2]-rho[c]*motion[i+2];
            // Equivalent root map: couple the tangent capacity to the normal
            // projection in this Newton trial. At a normal root it equals f_n.
            // qualified() still verifies the ORIGINAL actual-f_n disk law.
            const double radius=mu[c]*clip(v0,i),length=std::hypot(v1,v2);
            residual[i]=f[i]-clip(v0,i);
            const double factor=length>radius?radius/length:1.;
            residual[i+1]=f[i+1]-factor*v1;residual[i+2]=f[i+2]-factor*v2;
            if(derivative)
            {
                double* a=A.data()+9*c;double* b=B.data()+9*c;
                const bool normalFree=v0>0 && v0<hi(i);
                if(normalFree)a[0]=rho[c];else b[0]=1;
                double D[4]={1,0,0,1};
                if(length>radius)
                {
                    const double u1=v1/length,u2=v2/length;
                    D[0]=factor*(1-u1*u1);D[1]=D[2]=-factor*u1*u2;D[3]=factor*(1-u2*u2);
                    if(normalFree){b[3]=-mu[c]*u1;b[6]=-mu[c]*u2;a[3]=mu[c]*rho[c]*u1;a[6]=mu[c]*rho[c]*u2;}
                }
                for(int p=0;p<2;++p)for(int q=0;q<2;++q)
                {a[3*(p+1)+q+1]=rho[c]*D[2*p+q];b[3*(p+1)+q+1]=(p==q?1.:0.)-D[2*p+q];}
            }
        }
    }
    void apply(const double* x,double* y)
    {
        responseApply(x,temporary.data());
        for(int c=0;c<m/3;++c)for(int i=0;i<3;++i)
        {
            y[3*c+i]=0;for(int j=0;j<3;++j)y[3*c+i]+=A[9*c+3*i+j]*temporary[3*c+j]+B[9*c+3*i+j]*x[3*c+j];
        }
    }
    void transpose(const double* x,double* y)
    {
        for(int c=0;c<m/3;++c)for(int i=0;i<3;++i)
        {dualTemporary[3*c+i]=0;for(int j=0;j<3;++j)dualTemporary[3*c+i]+=A[9*c+3*j+i]*x[3*c+j];}
        responseApply(dualTemporary.data(),y);
        for(int c=0;c<m/3;++c)for(int i=0;i<3;++i)for(int j=0;j<3;++j)y[3*c+i]+=B[9*c+3*j+i]*x[3*c+j];
    }
    bool qualified(const double* f)
    {
        double scale=1;error=0;bool physical=true;
        for(int i=0;i<m;++i)
        {if(!std::isfinite(r[i]) || !std::isfinite(f[i]) || !std::isfinite(motion[i]))
         {error=std::numeric_limits<double>::infinity();return false;}
         scale=std::max(scale,std::max(std::abs(f[i]),std::abs(demand[i])*rho[i/3]));}
        for(int c=0;c<m/3;++c)
        {
            if(!friction(c))
            {
                for(int q=0;q<3;++q)
                {const int i=3*c+q;error=std::max(error,std::abs(f[i]-clip(f[i]-rho[c]*motion[i],i))/scale);
                 physical&=f[i]>=lo(i)-1.e-10 && f[i]<=hi(i)+1.e-10;}
                continue;
            }
            const int i=3*c;
            double root[3];const double lows[3]={lo(i),lo(i+1),lo(i+2)},highs[3]={hi(i),hi(i+1),hi(i+2)};
            avbdCoulombRoot(f+i,motion.data()+i,rho[c],mu[c],lows,highs,true,root);
            for(int q=0;q<3;++q)error=std::max(error,std::abs(root[q])/scale);
            physical&=f[i]>=-1.e-10 && f[i]<=hi(i)+1.e-10 &&
                std::hypot(f[i+1],f[i+2])-mu[c]*f[i]<=1.e-7;
            if(std::isfinite(hi(i)))
                physical&=std::abs((motion[i]>=0?f[i]:hi(i)-f[i])*motion[i])<=1.e-7;
            else physical&=motion[i]>=-1.e-9 && std::abs(f[i]*motion[i])<=1.e-7;
        }
        return error<1.e-10 && physical;
    }
    // Exact two-coordinate box QP with Coulomb tangents held fixed. Nearly
    // redundant rows otherwise exchange load only by their tiny gap difference
    // on every scalar GS sweep. No geometric merging or compliance is used.
    void scalarPair(double* f,int i,const double* column)
    {
        if(lo(i)==hi(i))return;
        const double a=column[i];if(!(a>0))return;
        int partner=-1;double correlation=0;
        for(int j=0;j<m;++j)if(j!=i && (!friction(j/3) || j%3==0) && lo(j)!=hi(j))
        {
            const double b=diagonal[9*(j/3)+4*(j%3)];if(!(b>0))continue;
            const double score=std::abs(column[j])/std::sqrt(a*b);
            if(score>correlation){correlation=score;partner=j;}
        }
        if(partner<0)return;const int j=partner;
        basis[j]=1;responseApply(basis.data(),dualTemporary.data());basis[j]=0;
        const double b=dualTemporary[j],c=.5*(column[j]+dualTemporary[i]);
        const double u=motion[i],v=motion[j];double best=0,bestI=f[i],bestJ=f[j];
        auto consider=[&](double x,double y)
        {
            if(!std::isfinite(x) || !std::isfinite(y) || x<lo(i) || x>hi(i) || y<lo(j) || y>hi(j))return;
            const double dx=x-f[i],dy=y-f[j];
            const double value=dx*(u+.5*a*dx)+dy*(v+.5*b*dy)+c*dx*dy;
            if(value<best){best=value;bestI=x;bestJ=y;}
        };
        const double determinant=std::fma(a,b,-c*c);
        if(determinant>0)consider(f[i]+(c*v-b*u)/determinant,f[j]+(c*u-a*v)/determinant);
        const double boundsI[2]={lo(i),hi(i)},boundsJ[2]={lo(j),hi(j)};
        for(int k=0;k<2;++k)
        {
            if(std::isfinite(boundsI[k]))consider(boundsI[k],clip(f[j]-(v+c*(boundsI[k]-f[i]))/b,j));
            if(std::isfinite(boundsJ[k]))consider(clip(f[i]-(u+c*(boundsJ[k]-f[j]))/a,i),boundsJ[k]);
        }
        const double dx=bestI-f[i],dy=bestJ-f[j];f[i]=bestI;f[j]=bestJ;
        for(int k=0;k<m;++k)motion[k]+=column[k]*dx+dualTemporary[k]*dy;
    }
    void sweep(double* f)
    {
        ++sweeps;responseApply(f,motion.data());for(int i=0;i<m;++i)motion[i]+=demand[i];
        for(int c=0;c<m/3;++c)
        {
            const int i=3*c;const double* block=diagonal.data()+9*c;
            for(int j=0;j<3;++j)
            {basis[i+j]=1;responseApply(basis.data(),columns.data()+j*m);basis[i+j]=0;}
            if(!friction(c))
            {
                for(int q=0;q<3;++q)
                {
                    const double value=lo(i+q)==hi(i+q)?lo(i+q):clip(f[i+q]-motion[i+q]/block[4*q],i+q);
                    const double change=value-f[i+q];f[i+q]=value;
                    for(int j=0;j<m;++j)motion[j]+=columns[q*m+j]*change;
                    scalarPair(f,i+q,columns.data()+q*m);
                }
                continue;
            }
            const double normal=clip(f[i]-motion[i]/block[0],i),change=normal-f[i];f[i]=normal;
            for(int j=0;j<m;++j)motion[j]+=columns[j]*change;
            scalarPair(f,i,columns.data());
            const double b1=motion[i+1]-block[4]*f[i+1]-block[5]*f[i+2];
            const double b2=motion[i+2]-block[7]*f[i+1]-block[8]*f[i+2];
            const double radius=mu[c]*f[i];
            double t1=0,t2=0;
            auto unconstrained=[&](double shift)
            {const double a=block[4]+shift,d=block[8]+shift,b=block[5],det=a*d-b*b;
             t1=(b*b2-d*b1)/det;t2=(b*b1-a*b2)/det;};
            if(radius>0)
            {
                unconstrained(0);
                if(std::hypot(t1,t2)>radius)
                {
                    double lo=0,hi=std::hypot(b1,b2)/radius;
                    for(int k=0;k<60;++k){const double mid=(lo+hi)*.5;unconstrained(mid);if(std::hypot(t1,t2)>radius)lo=mid;else hi=mid;}
                    unconstrained(hi);
                }
            }
            const double d1=t1-f[i+1],d2=t2-f[i+2];f[i+1]=t1;f[i+2]=t2;
            for(int j=0;j<m;++j)motion[j]+=columns[m+j]*d1+columns[2*m+j]*d2;
        }
    }
    // Two redundant sticking contacts can have an incompatible sticking face.
    // Min-norm Newton cannot move along its self-equilibrated force direction.
    // Select another face by bounded dual work; only the original NCP can
    // accept the result. This is not a potential for Coulomb friction.
    bool escapeStickPair(double* force)
    {
        if(m>192)return false;
        responseApply(force,motion.data());for(int i=0;i<m;++i)motion[i]+=demand[i];
        auto originalNorm=[&](const double* f,const double* u)
        {
            double value=0;for(int c=0;c<m/3;++c)
            {double root[3],low[3],high[3];for(int q=0;q<3;++q){low[q]=lo(3*c+q);high[q]=hi(3*c+q);}
                avbdCoulombRoot(f+3*c,u+3*c,rho[c],mu[c],low,high,friction(c),root);
                for(double v:root)value=std::hypot(value,v);}
            return value;
        };
        const double before=originalNorm(force,motion.data());double forceScale=1;
        for(int i=0;i<m;++i)forceScale=std::max(forceScale,std::max(std::abs(force[i]),std::abs(demand[i])*rho[i/3]));
        auto sticking=[&](int c){return friction(c) && force[3*c]>0 &&
            std::hypot(force[3*c+1],force[3*c+2])<mu[c]*force[3*c];};
        double best=0,bestDelta[6]={};int bestIds[6]={};
        for(int a=0;a<m/3;++a)if(sticking(a))for(int b=a+1;b<m/3;++b)if(sticking(b))
        {
            int ids[6]={3*a,3*a+1,3*a+2,3*b,3*b+1,3*b+2};double H[6][6],Q[6][6]={},original[6][6],scale=0;
            for(int j=0;j<6;++j)
            {basis[ids[j]]=1;responseApply(basis.data(),temporary.data());basis[ids[j]]=0;
                for(int i=0;i<6;++i)H[i][j]=temporary[ids[i]];Q[j][j]=1;}
            for(int i=0;i<6;++i)for(int j=0;j<=i;++j)
            {const double v=.5*(H[i][j]+H[j][i]);H[i][j]=H[j][i]=original[i][j]=original[j][i]=v;scale=std::max(scale,std::abs(v));}
            if(!(scale>0))continue;
            const double roundoff=64*std::numeric_limits<double>::epsilon()*scale;
            bool diagonalized=false;
            for(int sweep=0;sweep<32 && !diagonalized;++sweep)
            {
                diagonalized=true;
                for(int i=0;i<6;++i)for(int j=i+1;j<6;++j)
                {
                    const double off=H[i][j];if(std::abs(off)<=roundoff)continue;diagonalized=false;
                    const double tau=(H[j][j]-H[i][i])/(2*off);
                    const double t=std::copysign(1.,tau)/(std::abs(tau)+std::hypot(1.,tau)),c=1/std::sqrt(1+t*t),s=t*c;
                    H[i][i]-=t*off;H[j][j]+=t*off;H[i][j]=H[j][i]=0;
                    for(int k=0;k<6;++k)
                    {
                        if(k!=i && k!=j){const double x=H[k][i],y=H[k][j];H[k][i]=H[i][k]=c*x-s*y;H[k][j]=H[j][k]=s*x+c*y;}
                        const double x=Q[k][i],y=Q[k][j];Q[k][i]=c*x-s*y;Q[k][j]=s*x+c*y;
                    }
                }
            }
            double z[6]={},maximum=0,load=0;bool psd=diagonalized;
            for(int k=0;k<6;++k)
            {
                psd=psd && H[k][k]>=-roundoff;if(H[k][k]>roundoff)continue;
                double projection=0;for(int i=0;i<6;++i)projection+=Q[i][k]*motion[ids[i]];
                for(int i=0;i<6;++i)z[i]-=Q[i][k]*projection;
            }
            for(int i=0;i<6;++i){maximum=std::max(maximum,std::abs(z[i]));load=std::max(load,std::abs(motion[ids[i]]));}
            if(!psd || !(maximum>64*std::numeric_limits<double>::epsilon()*load))continue;
            for(double& v:z)v/=maximum;
            double alpha=std::numeric_limits<double>::infinity(),linear=0,quadratic=0;
            for(int side=0;side<2;++side)
            {
                const int c=side?b:a,j=3*side,i=3*c;const double radius=mu[c]*force[i];
                const double qa=mu[c]*mu[c]*z[j]*z[j]-z[j+1]*z[j+1]-z[j+2]*z[j+2];
                const double qb=2*(mu[c]*radius*z[j]-force[i+1]*z[j+1]-force[i+2]*z[j+2]);
                const double qc=radius*radius-force[i+1]*force[i+1]-force[i+2]*force[i+2];
                if(!(qc>0)){alpha=0;break;}
                if(z[j]<0)alpha=std::min(alpha,-force[i]/z[j]);
                if(z[j]>0 && std::isfinite(hi(i)))alpha=std::min(alpha,(hi(i)-force[i])/z[j]);
                const double discriminant=qb*qb-4*qa*qc;
                if(discriminant>0 && (qa<0 || qb<0))alpha=std::min(alpha,2*qc/(-qb+std::sqrt(discriminant)));
                else if(qa==0 && qb<0)alpha=std::min(alpha,-qc/qb);
            }
            for(int i=0;i<6;++i)
            {linear+=motion[ids[i]]*z[i];for(int j=0;j<6;++j)quadratic+=z[i]*original[i][j]*z[j];}
            if(!(linear<0))continue;
            if(quadratic>0)alpha=std::min(alpha,-linear/quadratic);
            const double change=alpha*(linear+.5*alpha*quadratic);
            if(alpha>0 && std::isfinite(alpha) && change<best)
            {
                std::copy(force,force+m,trial.data());for(int i=0;i<6;++i)trial[ids[i]]+=alpha*z[i];
                responseApply(trial.data(),temporary.data());for(int i=0;i<m;++i)temporary[i]+=demand[i];
                // A null direction may leave the root unchanged until a face
                // changes. Permit roundoff in THIS numerical trial, never an
                // increase of the original NCP at physical publication.
                if(originalNorm(trial.data(),temporary.data())>before+64*std::numeric_limits<double>::epsilon()*forceScale)continue;
                best=change;for(int i=0;i<6;++i){bestIds[i]=ids[i];bestDelta[i]=alpha*z[i];}
            }
        }
        if(!(best<0))return false;
        for(int i=0;i<6;++i)force[bestIds[i]]+=bestDelta[i];return true;
    }
    bool solve(double* force,int budget)
    {
        if(!valid || budget<0)return false;
        for(int i=0;i<m;++i)if(!std::isfinite(force[i]))return false;
        for(int k=0;k<budget;++k)
        {
            iterations=k;evaluate(force,r.data(),true);if(qualified(force))return true;
            for(int i=0;i<m;++i)rhs[i]=-r[i];
            // Inexact Newton forcing shrinks with the ORIGINAL nonlinear
            // defect. Solving every trial Jacobian to machine precision wastes
            // work on non-root faces. Final physics checks remain unchanged.
            const double forcing=std::max(std::sqrt(std::numeric_limits<double>::epsilon()),std::min(.1,std::sqrt(error)));
            linearIterations+=lsqr(*this,m,rhs.data(),direction.data(),linearWork.data(),4*m,forcing);
            apply(direction.data(),rhs.data());
            const double slope=2*dot(r.data(),rhs.data(),m),merit=dot(r.data(),r.data(),m);
            for(int i=0;i<m;++i)rhs[i]+=r[i];
            bool accepted=false;
            const bool consistent=norm(rhs.data(),m)<.9*std::sqrt(merit);
            if(consistent && slope<0)
                for(int line=0;line<24;++line)
                {
                    const double alpha=std::ldexp(1.,-line);
                    for(int i=0;i<m;++i)trial[i]=force[i]+alpha*direction[i];
                    evaluate(trial.data(),trialR.data(),false);
                    if(dot(trialR.data(),trialR.data(),m)<=merit+1.e-4*alpha*slope)
                    {std::copy(trial.begin(),trial.end(),force);accepted=true;break;}
                }
            if(!accepted && (consistent || !escapeStickPair(force)))sweep(force);
        }
        evaluate(force,r.data(),false);iterations=budget;return qualified(force);
    }
};
}
inline void avbdSolveCoulombNcp(const AvbdCoulombNcpArgs& a)
{
    auto& out=*a.result;out={false,0,0,0,std::numeric_limits<double>::infinity()};
    if(!a.groups){out.converged=true;out.residual=0;return;}
    if(!a.apply || !a.work || !a.force || !a.demand || !a.mu ||
        a.groups>PxU32(std::numeric_limits<int>::max()/82) || a.iterations>PxU32(std::numeric_limits<int>::max()))return;
    if(a.coulomb)for(PxU32 i=0;i<a.groups;++i)if(a.coulomb[i]!=0 && a.coulomb[i]!=1)return;
    struct Response
    {
        const AvbdCoulombNcpArgs& args;
        PxArray<double> cache;bool direct=false;
        explicit Response(const AvbdCoulombNcpArgs& a):args(a)
        {
            // Bounded numerical cache of THIS frozen operator only. Large
            // sets remain matrix-free; final qualification uses the callback.
            const int n=3*int(a.groups);if(n>192)return;
            cache.resize(n*n);PxArray<double> basis(n,0),column(n);
            for(int j=0;j<n;++j)
            {basis[j]=1;a.apply(a.context,basis.begin(),column.begin());basis[j]=0;
                for(int i=0;i<n;++i)cache[n*i+j]=column[i];}
        }
        void apply(const double* x,double* y)
        {
            if(direct || cache.empty()){args.apply(args.context,x,y);return;}
            const int n=3*int(args.groups);
            for(int i=0;i<n;++i){double sum=0;for(int j=0;j<n;++j)sum+=cache[n*i+j]*x[j];y[i]=sum;}
        }
    } response(a);
    AvbdCoulombNcp::Solver<Response> solver(response,int(a.groups),a.demand,a.mu,a.work,
        a.coulomb,a.lower,a.upper,a.compliance);
    out.converged=solver.solve(a.force,int(a.iterations));
    out.iterations=PxU32(solver.iterations);out.linearIterations=PxU32(solver.linearIterations);
    out.sweeps=PxU32(solver.sweeps);out.residual=solver.error;
    if(!response.cache.empty())
    {
        response.direct=true;solver.evaluate(a.force,solver.r.data(),false);
        out.converged=solver.qualified(a.force);
        if(!out.converged && out.iterations<a.iterations)
        {out.converged=solver.solve(a.force,int(a.iterations-out.iterations));out.iterations+=PxU32(solver.iterations);}
        out.linearIterations=PxU32(solver.linearIterations);out.sweeps=PxU32(solver.sweeps);out.residual=solver.error;
    }
}
} }
#endif
