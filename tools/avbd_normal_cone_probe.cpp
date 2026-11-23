#include "avbd/DyAvbdNormalCone.h"
#include "avbd/DyAvbdOgcActivation.h"
#include <cstdio>
#include <cmath>
#include <limits>
using namespace physx;
using namespace physx::Dy;

struct Matrix2
{
    double a,b,c;
    void apply(const double* x,double* y) const
    {y[0]=a*x[0]+b*x[1];y[1]=b*x[0]+c*x[1];}
};

static AvbdNormalConeResult solve(Matrix2 W,const PxU8* bilateral,
    const double* demand,double* force,PxU32 budget=24)
{
    double work[12],diagonal[2]={W.a,W.c};PxU8 active[2];
    return avbdSolveNormalCone(W,2,bilateral,demand,diagonal,force,work,active,budget,24,1.e-12);
}

int main()
{
    unsigned failed=0;double f[2];const PxU8 mixed[2]={0,1},normal[2]={0,0};
    const double demand[2]={.01,0};
    const auto loaded=solve({.5,-.5,5./6},mixed,demand,f);
    if(!loaded.converged || std::abs(f[0]-.05)>1.e-12 || std::abs(f[1]-.03)>1.e-12)++failed;
    std::printf("[NORMAL_CONE] loaded force=%.12g joint=%.12g residual=%.9g calls=%u\n",f[0],f[1],loaded.residual,loaded.operatorCalls);
    const double repeated[2]={1,1};
    auto r=solve({1,1,1},normal,repeated,f);
    if(!r.converged || std::abs(f[0]+f[1]-1)>1.e-12 || f[0]<0 || f[1]<0)++failed;
    r=solve({1,1,1},mixed,repeated,f);
    if(!r.converged || std::abs(f[0]+f[1]-1)>1.e-12 || f[0]<0)++failed;
    const double impossible[2]={1,0};
    if(solve({1,-1,1},normal,impossible,f).converged)++failed;
    if(solve({1,0,1},normal,repeated,f,0).converged)++failed;
    const double nanInput[2]={std::numeric_limits<double>::quiet_NaN(),0};
    if(solve({1,0,1},normal,nanInput,f).converged)++failed;
    const double infInput[2]={std::numeric_limits<double>::infinity(),0};
    if(solve({1,0,1},normal,infInput,f).converged)++failed;
    // A finite prox at zero gap does not imply a fixed point of the material.
    double hard=100,barrier=100;
    for(unsigned i=0;i<80;++i)
    {
        hard=avbdOGCNormalResolvent(hard,1.e4,0.,1.e5);
        barrier=avbdOGCNormalResolvent(barrier,1.e4,.04,1.e5);
    }
    if(hard!=100 || !(barrier>hard))++failed;
    std::printf("[NORMAL_CONE] closedHard=%.9g incompatibleBarrier=%.9g failures=%u\n",hard,barrier,failed);
    return failed?1:0;
}
