// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_NONLINEAR_GROUP_H
#define DY_AVBD_NONLINEAR_GROUP_H
#include "DyAvbdNonlinearBlock.h"
namespace physx { namespace Dy {
// A bounded numerical grouping of physical Rigid6 blocks. No inner iteration,
// poses, duals or material history are owned by this Table leaf.
template<PxU32 Capacity> struct AvbdNonlinearGroupT
{
    double K[Capacity][Capacity][36]={},M[Capacity][Capacity][36]={},gradient[Capacity][6]={},step[Capacity][6]={};
    PxU32 count=0,locked[Capacity]={},factorizations=0,substitutions=0;
    double translationMass[Capacity]={}; // Nonzero only for a closed, admitted translation chart.
    bool solved=false,metric=false;
    // Scatter only the row's actual group support. Repeated endpoints must be
    // merged before this call. Jacobians are packed in member-list order;
    // a two-endpoint row never carries a full group-width temporary.
    template<class Scalar> void addRankOne(const PxU32* members,PxU32 memberCount,
        const Scalar (*jacobian)[6],double curvature,bool offDiagonal=false)
    {
        for(PxU32 a=0;a<memberCount;++a)for(PxU32 b=0;b<memberCount;++b)
        {
            const PxU32 i=members[a],j=members[b];if(offDiagonal && i==j)continue;
            for(PxU32 d=0;d<6;++d)for(PxU32 k=0;k<6;++k)
            {
                const double h=curvature*jacobian[a][d]*jacobian[b][k];
                K[i][j][6*d+k]+=h;M[i][j][6*d+k]+=h;
            }
        }
    }
    // Numerical scratch is row-contiguous; physical unknowns remain Rigid6.
    // The lower triangle is written before use, with inverse diagonal entries.
    bool factor(const double (&A)[Capacity][Capacity][36],double (&L)[6*Capacity][6*Capacity]) const
    {
        for(PxU32 i=0;i<6*count;++i)for(PxU32 j=0;j<=i;++j)
        {
            double dot=0;for(PxU32 k=0;k<j;++k)dot+=L[i][k]*L[j][k];
            double x=A[i/6][j/6][6*(i%6)+j%6]-dot;
            if(i==j)
            {if(!(x>0) || !std::isfinite(x))return false;x=1/std::sqrt(x);}
            else x*=L[j][j];
            if(!std::isfinite(x))return false;L[i][j]=x;
        }
        return true;
    }
    // Closed charts share one traversal for the gradient and three constraint
    // right hand sides. No extra factorization or iterative solve is introduced.
    template<PxU32 Rhs> bool substitute(const double (&L)[6*Capacity][6*Capacity],double (&values)[6*Capacity][4]) const
    {
        for(PxU32 i=0;i<6*count;++i)
        {
            double x[Rhs];for(PxU32 q=0;q<Rhs;++q)x[q]=values[i][q];
            for(PxU32 j=0;j<i;++j)for(PxU32 q=0;q<Rhs;++q)x[q]-=L[i][j]*values[j][q];
            for(PxU32 q=0;q<Rhs;++q)values[i][q]=x[q]*L[i][i];
        }
        for(PxU32 ii=6*count;ii>0;--ii)
        {
            const PxU32 i=ii-1;double x[Rhs];for(PxU32 q=0;q<Rhs;++q)x[q]=values[i][q];
            for(PxU32 j=i+1;j<6*count;++j)for(PxU32 q=0;q<Rhs;++q)x[q]-=L[j][i]*values[j][q];
            for(PxU32 q=0;q<Rhs;++q)
            {values[i][q]=x[q]*L[i][i];if(!std::isfinite(values[i][q]))return false;}
        }
        return true;
    }
    void solve()
    {
        solved=false;factorizations=0;substitutions=0;metric=false;
        if(!count || count>Capacity)return;
        for(PxU32 i=0;i<count;++i)for(PxU32 r=0;r<6;++r)
        {
            step[i][r]=0;if(!std::isfinite(gradient[i][r]))return;
            if(locked[i]&(1u<<r))gradient[i][r]=0;
            for(PxU32 j=0;j<count;++j)for(PxU32 c=0;c<6;++c)
            {
                const PxU32 k=6*r+c;
                if(!std::isfinite(K[i][j][k]) || !std::isfinite(M[i][j][k]))return;
                if((locked[i]&(1u<<r)) || (locked[j]&(1u<<c)))K[i][j][k]=M[i][j][k]=(i==j && r==c)?1:0;
            }
        }
        double L[6*Capacity][6*Capacity];++factorizations;
        if(!factor(K,L))
        {
            metric=true;++factorizations;if(!factor(M,L))return;
        }
        bool closed=true;double scale=0;
        for(PxU32 i=0;i<count;++i){closed &= translationMass[i]>0 && !(locked[i]&7);scale=std::max(scale,translationMass[i]);}
        double values[6*Capacity][4]={},weights[Capacity]={};
        for(PxU32 i=0;i<count;++i)
        {
            if(closed)weights[i]=translationMass[i]/scale;
            for(PxU32 d=0;d<6;++d)
            {values[6*i+d][0]=-gradient[i][d];if(closed && d<3)values[6*i+d][1+d]=weights[i];}
        }
        substitutions=closed?4:1;
        if(!(closed?substitute<4>(L,values):substitute<1>(L,values)))return;
        for(PxU32 i=0;i<count;++i)for(PxU32 d=0;d<6;++d)step[i][d]=values[6*i+d][0];
        if(closed)
        {
            double S[36]={},residual[6]={},lambda[6]={};
            for(PxU32 d=0;d<3;++d)for(PxU32 i=0;i<count;++i)
            {
                residual[d]+=weights[i]*step[i][d];
                for(PxU32 k=0;k<3;++k)S[6*k+d]+=weights[i]*values[6*i+k][1+d];
            }
            ++factorizations;if(!avbdSolveSmallBlock(S,residual,lambda,3))return;
            for(PxU32 i=0;i<count;++i)for(PxU32 d=0;d<6;++d)
                for(PxU32 k=0;k<3;++k)step[i][d]-=values[6*i+d][1+k]*lambda[k];
        }
        solved=true;
    }
};
typedef AvbdNonlinearGroupT<4> AvbdNonlinearGroup;
typedef AvbdNonlinearGroupT<16> AvbdLocalNonlinearGroup;
struct AvbdLocalNonlinearGroupArgs { AvbdLocalNonlinearGroup* group; };
struct AvbdNonlinearGroupArgs { AvbdNonlinearGroup* group; };

inline void avbdAccumulateFrameGroup(const AvbdElasticFrameContribution& e,
    const PxU32 ends[2],const PxU32* ids,AvbdNonlinearGroup& group)
{
    for(PxU32 a=0;a<2;++a)for(PxU32 i=0;i<group.count;++i)if(ends[a]==ids[i])
        for(PxU32 r=0;r<6;++r)
        {
            group.gradient[i][r]+=e.gradient[6*a+r];
            for(PxU32 b=0;b<2;++b)for(PxU32 j=0;j<group.count;++j)if(ends[b]==ids[j])
                for(PxU32 c=0;c<6;++c)
                {
                    const double sign=(a && r<3?-1:1)*(b && c<3?-1:1);
                    const PxU32 k=9*(r<3?r:r+3*a)+(c<3?c:c+3*b);
                    group.K[i][j][6*r+c]+=sign*e.hessian[k];group.M[i][j][6*r+c]+=sign*e.gaussNewton[k];
                }
        }
}
} }
#endif
