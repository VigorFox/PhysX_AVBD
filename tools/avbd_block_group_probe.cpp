#include "avbd/DyKernelTable.h"
#include "avbd/DyAvbdBlockGroup.h"
#include <cstdio>
#include <cmath>
#include <limits>
using namespace physx;
using namespace physx::Dy;

static PxU32 randomState=7867;
static double randomValue()
{
    randomState=1664525u*randomState+1013904223u;
    return double(randomState>>8)/16777216.0-.5;
}
// Independent dense elimination is only a test oracle, never a runtime path.
static bool denseSolve(double matrix[24][24],double rhs[24],PxU32 n)
{
    for(PxU32 k=0;k<n;++k)
    {
        PxU32 pivot=k;
        for(PxU32 i=k+1;i<n;++i)if(std::abs(matrix[i][k])>std::abs(matrix[pivot][k]))pivot=i;
        if(!(std::abs(matrix[pivot][k])>1.e-15))return false;
        for(PxU32 j=k;j<n;++j){const double v=matrix[k][j];matrix[k][j]=matrix[pivot][j];matrix[pivot][j]=v;}
        const double v=rhs[k];rhs[k]=rhs[pivot];rhs[pivot]=v;
        for(PxU32 i=k+1;i<n;++i)
        {
            const double scale=matrix[i][k]/matrix[k][k];
            for(PxU32 j=k;j<n;++j)matrix[i][j]-=scale*matrix[k][j];
            rhs[i]-=scale*rhs[k];
        }
    }
    for(PxU32 i=n;i-->0;)
    {for(PxU32 j=i+1;j<n;++j)rhs[i]-=matrix[i][j]*rhs[j];rhs[i]/=matrix[i][i];}
    return true;
}
int main()
{
    KernelTable table=KernelTable::makeAvbdCpuScalar();
    PxU32 failures=0,cases=0,products=0,unresolved=0;double maxError=0,maxResidual=0,maxOperatorError=0;
    for(PxU32 restriction=0;restriction<2;++restriction)
    for(PxU32 count=2;count<=4;++count)for(PxU32 trial=0;trial<100;++trial)
    {
        const PxU32 n=6*count,rank=trial%13;
        PxReal diagonal[4][6][6]={},gradient[4][6]={},step[4][6]={};
        AvbdGroupFactor factors[12]={};
        double dense[24][24]={},oracle[24][24]={},answer[24]={},rhsScale=0;
        for(PxU32 i=0;i<n;++i)dense[i][i]=.5+double(i%6)*.2;
        for(PxU32 f=0;f<rank;++f)
        {
            factors[f].curvature=PxReal(.1+double(trial%9));
            for(PxU32 i=0;i<n;++i)factors[f].jacobian[i/6][i%6]=PxReal(randomValue());
            for(PxU32 i=0;i<n;++i)for(PxU32 j=0;j<n;++j)
                dense[i][j]+=double(factors[f].curvature)*double(factors[f].jacobian[i/6][i%6])*double(factors[f].jacobian[j/6][j%6]);
        }
        for(PxU32 i=0;i<n;++i)
        {
            gradient[i/6][i%6]=PxReal(randomValue());answer[i]=-double(gradient[i/6][i%6]);
            rhsScale=PxMax(rhsScale,std::abs(answer[i]));
            for(PxU32 j=0;j<n;++j)
            {
                if(i/6==j/6){diagonal[i/6][i%6][j%6]=PxReal(dense[i][j]);dense[i][j]=diagonal[i/6][i%6][j%6];}
                oracle[i][j]=dense[i][j];
            }
        }
        if(restriction)
        {
            // Restrict the independently built FULL matrix, not the answer.
            // Check all 64 per-body public lock combinations across cases.
            for(PxU32 b=0;b<count;++b)
            {
                const PxU32 mask=(trial+17*b)%64;
                avbdKernelRestrict6(diagonal[b],gradient[b],mask);
                for(PxU32 d=0;d<6;++d)if(mask&(1u<<d))
                {
                    const PxU32 i=6*b+d;answer[i]=0;
                    for(PxU32 j=0;j<n;++j)dense[i][j]=dense[j][i]=0;
                    dense[i][i]=1;
                    for(PxU32 f=0;f<rank;++f)factors[f].jacobian[b][d]=0;
                }
            }
            for(PxU32 i=0;i<n;++i)for(PxU32 j=0;j<n;++j)oracle[i][j]=dense[i][j];
            rhsScale=0;for(PxU32 i=0;i<n;++i)rhsScale=PxMax(rhsScale,std::abs(answer[i]));
        }
        bool solved=false;AvbdLinearResponseResult result={};
        AvbdGroupSolveArgs args={count,diagonal,gradient,factors,rank,step,&solved,&result};
        AvbdGroupOperator op={args,{}};
        double delta[24],modelOracle=0;
        for(PxU32 i=0;i<n;++i)delta[i]=randomValue();
        for(PxU32 i=0;i<n;++i)
        {
            modelOracle+=double(gradient[i/6][i%6])*delta[i];
            for(PxU32 j=0;j<n;++j)modelOracle+=.5*delta[i]*dense[i][j]*delta[j];
        }
        if(std::abs(avbdGroupModelChange(args,delta)-modelOracle)>1.e-11*PxMax(1.,std::abs(modelOracle)))++failures;
        for(PxU32 column=0;column<n;++column)
        {
            double basis[24]={},actual[24];basis[column]=1;op.apply(basis,actual);
            for(PxU32 i=0;i<n;++i)maxOperatorError=PxMax(maxOperatorError,std::abs(actual[i]-dense[i][column]));
        }
        table.launch(AVBD_RIGID_GROUP,&args);++cases;products+=result.operatorCalls;
        if(!denseSolve(oracle,answer,n)){++failures;continue;}
        if(!solved)
        {
            // n finite-precision CG steps need not resolve every SPD input.
            // Production must report exhaustion so the candidate is not used.
            ++unresolved;
            if(result.converged || result.iterations!=n || result.residual<=double(PX_EPS_F32)*rhsScale)++failures;
            // Diagnose the SAME matrix with a larger algebraic budget. This
            // does not change the production budget or scene physics.
            double rhs[24],x[24],work[96];
            for(PxU32 b=0;b<count;++b)for(PxU32 j=0;j<6;++j)op.inverse[b][j][j]=1;
            for(PxU32 i=0;i<n;++i)rhs[i]=-double(gradient[i/6][i%6]);
            const auto extra=avbdLinearResponse(op,n,rhs,x,work,4*n,1.e-11,0.);
            if(!extra.converged)++failures;
            for(PxU32 i=0;i<n;++i)if(std::abs(x[i]-answer[i])>2.e-9)++failures;
            continue;
        }
        if(!result.converged)++failures;
        if(!(result.residual<=double(PX_EPS_F32)*rhsScale))++failures;
        for(PxU32 i=0;i<n;++i)
        {
            maxError=PxMax(maxError,std::abs(double(step[i/6][i%6])-answer[i]));
            double residual=gradient[i/6][i%6];
            for(PxU32 j=0;j<n;++j)residual+=dense[i][j]*double(step[j/6][j%6]);
            maxResidual=PxMax(maxResidual,std::abs(residual)/rhsScale);
        }
    }
    if(maxOperatorError>1.e-12 || maxError>2.e-6 || maxResidual>2.e-6)++failures;
    // A shared contact's relative mode must cancel during common translation.
    // Retain unequal endpoint conditioning ONLY on the diagonal. A projected
    // ascent must be rejected even if the original Newton proposal descended.
    {
        PxReal diag[4][6][6]={},grad[4][6]={};AvbdGroupFactor contact[3]={};
        for(PxU32 b=0;b<2;++b)for(PxU32 j=0;j<6;++j)diag[b][j][j]=1;
        for(PxU32 axis=0;axis<3;++axis)
        {
            contact[axis].curvature=100;
            contact[axis].jacobian[0][axis]=1;contact[axis].jacobian[1][axis]=-1;
            diag[0][axis][axis]+=100;diag[1][axis][axis]+=150;
        }
        grad[0][0]=grad[1][0]=-30;
        AvbdGroupSolveArgs a={2,diag,grad,contact,3,NULL,NULL,NULL};
        double common[24]={},ascent[24]={};common[0]=common[6]=1;ascent[6]=-1;
        if(std::abs(avbdGroupModelChange(a,common)+34)>1.e-12 || avbdGroupModelChange(a,ascent)<=0)++failures;
    }
    PxReal diagonal[4][6][6]={},gradient[4][6]={},step[4][6]={};
    for(PxU32 b=0;b<4;++b)for(PxU32 j=0;j<6;++j)diagonal[b][j][j]=1;
    bool solved=false;AvbdLinearResponseResult result={};
    AvbdGroupSolveArgs args={4,diagonal,gradient,NULL,0,step,&solved,&result};
    table.launch(AVBD_RIGID_GROUP,&args);
    if(!solved || !result.converged || result.operatorCalls!=0)++failures;
    gradient[0][0]=std::numeric_limits<PxReal>::infinity();table.launch(AVBD_RIGID_GROUP,&args);
    if(solved || result.converged)++failures;
    args.count=1;table.launch(AVBD_RIGID_GROUP,&args);if(solved)++failures;
    std::printf("[BLOCK_GROUP] cases=%u unresolved=%u products=%u operatorError=%.9g solutionError=%.9g floatResidual=%.9g failures=%u\n",
        cases,unresolved,products,maxOperatorError,maxError,maxResidual,failures);
    return failures?1:0;
}
