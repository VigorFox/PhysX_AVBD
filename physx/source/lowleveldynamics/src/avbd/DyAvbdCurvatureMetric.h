// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_CURVATURE_METRIC_H
#define DY_AVBD_CURVATURE_METRIC_H
#include "DyAvbdCommonProgram.h"
namespace physx { namespace Dy {
// Modify only nonpositive modes of an ASSEMBLED physical block. Positive
// blocks retain their exact curvature, including cancellation between rows.
// The old SPD metric supplies units and a positive model in indefinite modes.
inline bool avbdRegularizeLocalCurvature(double* K,const double* M,PxU32 n)
{
    if(n!=3 && n!=6)return false;
    double zero[6]={},test[6]={};if(avbdSolveSmallBlock(K,zero,test,n))return true;
    double L[6][6]={},inverse[6][6]={},B[6][6]={},Q[6][6]={};
    for(PxU32 i=0;i<n;++i)for(PxU32 j=0;j<=i;++j)
    {
        double value=M[6*i+j];for(PxU32 k=0;k<j;++k)value-=L[i][k]*L[j][k];
        if(i==j){if(!(value>0) || !std::isfinite(value))return false;L[i][j]=std::sqrt(value);}
        else L[i][j]=value/L[j][j];
    }
    for(PxU32 col=0;col<n;++col)for(PxU32 i=0;i<n;++i)
    {double value=i==col?1:0;for(PxU32 j=0;j<i;++j)value-=L[i][j]*inverse[j][col];inverse[i][col]=value/L[i][i];}
    for(PxU32 i=0;i<n;++i)
    {
        Q[i][i]=1;
        for(PxU32 j=0;j<=i;++j)
        {double value=0;for(PxU32 a=0;a<n;++a)for(PxU32 b=0;b<n;++b)value+=inverse[i][a]*K[6*a+b]*inverse[j][b];B[i][j]=B[j][i]=value;}
    }
    bool converged=false;
    for(PxU32 sweep=0;sweep<64 && !converged;++sweep)
    {
        converged=true;
        for(PxU32 p=0;p<n;++p)for(PxU32 q=p+1;q<n;++q)
        {
            const double off=B[p][q],a=B[p][p],b=B[q][q];
            if(std::abs(off)<=16*std::numeric_limits<double>::epsilon()*std::sqrt(std::abs(a)*std::abs(b)))continue;
            converged=false;const double tau=(b-a)/(2*off);
            const double t=std::copysign(1.,tau)/(std::abs(tau)+std::hypot(1.,tau)),c=1/std::sqrt(1+t*t),s=t*c;
            B[p][p]=a-t*off;B[q][q]=b+t*off;B[p][q]=B[q][p]=0;
            for(PxU32 k=0;k<n;++k)
            {
                if(k!=p && k!=q){const double x=B[k][p],y=B[k][q];B[k][p]=B[p][k]=c*x-s*y;B[k][q]=B[q][k]=s*x+c*y;}
                const double x=Q[k][p],y=Q[k][q];Q[k][p]=c*x-s*y;Q[k][q]=s*x+c*y;
            }
        }
    }
    if(!converged)return false;
    for(PxU32 mode=0;mode<n;++mode)if(B[mode][mode]<=0)
    {
        double v[6]={};for(PxU32 i=0;i<n;++i)for(PxU32 j=0;j<=i;++j)v[i]+=L[i][j]*Q[j][mode];
        for(PxU32 i=0;i<n;++i)for(PxU32 j=0;j<=i;++j)
        {const double value=K[6*i+j]+(1-B[mode][mode])*v[i]*v[j];K[6*i+j]=K[6*j+i]=value;}
    }
    return avbdSolveSmallBlock(K,zero,test,n);
}
} }
#endif
