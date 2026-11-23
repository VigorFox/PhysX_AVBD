#include "avbd/DyKernelTable.h"
#include <cmath>
#include <cstdio>
#include <limits>
using namespace physx;
using namespace physx::Dy;
struct Operators
{
    bool duplicate=false,invalid=false;
    static void H(void* p,const double* x,double* y)
    {
        if(static_cast<Operators*>(p)->invalid)
        {for(unsigned i=0;i<12;++i)y[i]=std::numeric_limits<double>::quiet_NaN();return;}
        for(unsigned i=0;i<12;++i)y[i]=x[i];
        y[0]=x[0]+.5*x[6];y[6]=.5*x[0]+2*x[6];
    }
    static void J(void* p,const double* x,double* y)
    {y[0]=x[0];y[1]=static_cast<Operators*>(p)->duplicate?x[0]:x[6]-x[0];}
    static void JT(void* p,const double* x,double* y)
    {
        for(unsigned i=0;i<12;++i)y[i]=0;
        if(static_cast<Operators*>(p)->duplicate)y[0]=x[0]+x[1];
        else {y[0]=x[0]-x[1];y[6]=x[1];}
    }
    static void P(void*,const double* x,double* y)
    {for(unsigned i=0;i<12;++i)y[i]=x[i];y[6]*=.5;}
};
int main()
{
    KernelTable table=KernelTable::makeAvbdCpuScalar();Operators op;
    const double inf=std::numeric_limits<double>::infinity();
    double base[12]={},offset[2]={},C[2]={},lo[2]={0,-inf},hi[2]={inf,inf},diagonal[2]={1,1.5};
    base[0]=-.025;base[6]=-.065;
    double x[12],force[2]={},work[12*14];PxU8 active[2];AvbdMixedKktResult result;
    AvbdMixedKktArgs a={&op,&Operators::H,&Operators::J,&Operators::JT,&Operators::P,
        12,2,32,128,base,offset,C,lo,hi,diagonal,x,force,work,active,1.e-11,&result};
    unsigned failures=0;
    auto solve=[&](double f0,double f1,double x0,double x6)
    {
        table.launch(AVBD_MIXED_KKT,&a);
        if(!result.converged || std::abs(force[0]-f0)>1.e-10 || std::abs(force[1]-f1)>1.e-10 ||
           std::abs(x[0]-x0)>1.e-10 || std::abs(x[6]-x6)>1.e-10)++failures;
    };
    solve(.09,.065,0,0);
    hi[0]=.03;solve(.03,.0275,-.015,-.015);
    hi[0]=0;solve(0,.00875,-.0225,-.0225);
    hi[0]=inf;lo[1]=-.01;hi[1]=.01;solve(.02125,.01,0,-.0275);
    lo[1]=-inf;hi[1]=inf;C[0]=.2;solve(.05,.04,-.01,-.01);C[0]=0;
    a.krylovBudget=0;table.launch(AVBD_MIXED_KKT,&a);if(result.converged)++failures;a.krylovBudget=128;
    a.activeBudget=0;table.launch(AVBD_MIXED_KKT,&a);if(result.converged)++failures;a.activeBudget=32;
    op.invalid=true;table.launch(AVBD_MIXED_KKT,&a);if(result.converged)++failures;op.invalid=false;
    op.duplicate=true;lo[0]=-inf;offset[1]=.02;
    table.launch(AVBD_MIXED_KKT,&a);if(result.converged)++failures;
    a.primalSize=a.rowCount=0;a.work=NULL;table.launch(AVBD_MIXED_KKT,&a);if(!result.converged)++failures;
    std::printf("[MIXED_KKT] Table/full-material/normal-cap/zero-response/drive-cap/compliance/invalid failures=%u\n",failures);
    return failures?1:0;
}
