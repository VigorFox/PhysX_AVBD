#include "avbd/DyKernelTable.h"
#include <cstdio>
#include <cmath>
#include <limits>
using namespace physx;
using namespace physx::Dy;

struct Material
{
    static void apply(void*,const double* x,double* y)
    {
        for(unsigned i=0;i<12;++i)y[i]=x[i];
        y[0]=x[0]+.5*x[6];y[6]=.5*x[0]+2*x[6];
    }
    static void inverse(void*,const double* x,double* y)
    {for(unsigned i=0;i<12;++i)y[i]=x[i];y[6]*=.5;}
};
struct Response
{
    KernelTable table;
    unsigned calls=0;
    bool resolved=true;
    static void apply(void* ptr,const double* force,double* y)
    {
        Response& a=*static_cast<Response*>(ptr);
        double rhs[12]={},x[12],work[48];rhs[0]=force[0]-force[1];rhs[6]=force[1];
        AvbdLinearResponseResult result;
        AvbdLinearResponseArgs args={NULL,&Material::apply,&Material::inverse,12,48,rhs,x,work,1.e-13,0.,&result};
        a.table.launch(AVBD_LINEAR_RESPONSE,&args);a.calls+=result.operatorCalls;
        a.resolved=a.resolved && result.converged;
        if(!result.converged){y[0]=y[1]=std::numeric_limits<double>::quiet_NaN();return;}
        y[0]=x[0];y[1]=x[6]-x[0];
    }
    static void invalid(void*,const double*,double* y)
    {y[0]=y[1]=std::numeric_limits<double>::quiet_NaN();}
};
int main()
{
    unsigned failures=0;Response op;op.table=KernelTable::makeAvbdCpuScalar();
    double rhs[12]={},x[12],work[48];rhs[0]=-.025;rhs[6]=-.065;
    AvbdLinearResponseResult linear;
    AvbdLinearResponseArgs a={NULL,&Material::apply,&Material::inverse,12,48,rhs,x,work,1.e-13,0.,&linear};
    op.table.launch(AVBD_LINEAR_RESPONSE,&a);
    if(!linear.converged || std::abs(x[0]+.01)>1.e-12 || std::abs(x[6]+.03)>1.e-12)++failures;
    a.budget=0;op.table.launch(AVBD_LINEAR_RESPONSE,&a);if(linear.converged)++failures;
    a.budget=48;rhs[0]=std::numeric_limits<double>::infinity();
    op.table.launch(AVBD_LINEAR_RESPONSE,&a);if(linear.converged)++failures;
    rhs[0]=0;rhs[6]=0;a.budget=0;op.table.launch(AVBD_LINEAR_RESPONSE,&a);
    if(!linear.converged || linear.operatorCalls!=0)++failures;
    AvbdLinearResponseArgs empty={NULL,NULL,NULL,0,0,NULL,NULL,NULL,1.e-13,0.,&linear};
    op.table.launch(AVBD_LINEAR_RESPONSE,&empty);
    if(!linear.converged || linear.residual!=0 || linear.operatorCalls!=0)++failures;
    const PxU8 eq[2]={0,1};const double demand[2]={.01,.02},diagonal[2]={1,2};
    double force[2],scratch[12];PxU8 active[2];AvbdNormalConeResult result;
    AvbdNormalConeArgs n={&op,&Response::apply,2,24,24,eq,demand,diagonal,force,scratch,active,1.e-12,&result};
    op.table.launch(AVBD_NORMAL_CONE,&n);
    if(!result.converged || !op.resolved || std::abs(force[0]-.09)>1.e-12 || std::abs(force[1]-.065)>1.e-12)++failures;
    std::printf("[LINEAR_RESPONSE] sharedLoad=%.12g joint=%.12g residual=%.9g materialCalls=%u\n",force[0],force[1],result.residual,op.calls);
    n.apply=&Response::invalid;op.table.launch(AVBD_NORMAL_CONE,&n);if(result.converged)++failures;
    std::printf("[LINEAR_RESPONSE] table/fullH/budget/invalid failures=%u\n",failures);
    return failures?1:0;
}
