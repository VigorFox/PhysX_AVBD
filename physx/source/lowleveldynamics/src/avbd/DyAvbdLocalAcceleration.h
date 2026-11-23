// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_LOCAL_ACCELERATION_H
#define DY_AVBD_LOCAL_ACCELERATION_H
#include "foundation/PxArray.h"
#include <cmath>
#include <algorithm>
namespace physx { namespace Dy {
// Bounded numerical secant history. The caller checks the original equations
// before using a proposal. No physical block, force law or constraint is added.
struct AvbdLocalAcceleration
{
    PxArray<double> previousResidual,previousOutput,dr[4],dy[4];PxU32 count=0;
    void reset(){count=0;previousResidual.clear();previousOutput.clear();}
    static double dot(const PxArray<double>& a,const PxArray<double>& b)
    {double result=0;for(PxU32 i=0;i<a.size();++i)result+=a[i]*b[i];return result;}
    bool propose(const PxArray<double>& input,const PxArray<double>& output,PxArray<double>& candidate)
    {
        const PxU32 n=input.size();PxArray<double> residual(n);
        for(PxU32 i=0;i<n;++i)residual[i]=output[i]-input[i];
        if(previousResidual.size()==n)
        {
            if(count==4){for(PxU32 j=1;j<4;++j){dr[j-1].swap(dr[j]);dy[j-1].swap(dy[j]);}--count;}
            dr[count].resize(n);dy[count].resize(n);
            for(PxU32 i=0;i<n;++i){dr[count][i]=residual[i]-previousResidual[i];dy[count][i]=output[i]-previousOutput[i];}
            ++count;
        }
        previousResidual=residual;previousOutput=output;if(!count)return false;
        PxArray<double> q[4];double R[4][4]={},rhs[4]={},gamma[4]={};PxU32 selected[4]={},rank=0;
        // Most recent independent differences first; twice-reorthogonalized
        // QR avoids squaring a nearly singular Gram matrix's condition number.
        for(PxU32 kk=count;kk>0;--kk)
        {
            const PxU32 k=kk-1;PxArray<double> v=dr[k];const double original=dot(v,v);
            for(PxU32 pass=0;pass<2;++pass)for(PxU32 j=0;j<rank;++j)
            {const double a=dot(q[j],v);R[j][rank]+=a;for(PxU32 i=0;i<n;++i)v[i]-=a*q[j][i];}
            const double length=std::sqrt(dot(v,v));
            if(!(length>1.e-8*std::sqrt(original)) || !std::isfinite(length))
            {for(PxU32 j=0;j<rank;++j)R[j][rank]=0;continue;}
            R[rank][rank]=length;for(PxU32 i=0;i<n;++i)v[i]/=length;
            q[rank]=v;rhs[rank]=dot(v,residual);selected[rank]=k;++rank;
        }
        if(!rank)return false;
        for(PxU32 jj=rank;jj>0;--jj)
        {const PxU32 j=jj-1;double v=rhs[j];for(PxU32 k=j+1;k<rank;++k)v-=R[j][k]*gamma[k];gamma[j]=v/R[j][j];}
        candidate=output;
        for(PxU32 j=0;j<rank;++j)for(PxU32 i=0;i<n;++i)candidate[i]-=gamma[j]*dy[selected[j]][i];
        for(double v:candidate)if(!std::isfinite(v))return false;
        return true;
    }
};
} }
#endif
