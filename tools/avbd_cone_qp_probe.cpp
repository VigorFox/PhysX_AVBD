#include "../experiments/avbd_contact_lab/cone_qp.h"
#include <cstdio>
#include <vector>
using namespace physx;
using namespace physx::Dy;
struct Operator
{
    static void H(void*,const double* x,double* y){for(unsigned i=0;i<6;++i)y[i]=x[i];}
    static void J(void*,const double* x,double* y){for(unsigned i=0;i<4;++i)y[i]=x[i];}
    static void T(void*,const double* x,double* y){for(unsigned i=0;i<6;++i)y[i]=i<4?x[i]:0;}
};
int main()
{
    double base[6]={-1,2,0,0,0,0},offset[4]={0,0,0,-.2},diag[4]={1,1,1,1};
    double x[6],force[4];std::vector<double> work(avbdConeQpWorkSize(6,4));AvbdMixedKktResult r;
    AvbdConeQpArgs a={NULL,Operator::H,Operator::J,Operator::T,Operator::H,6,4,1,80,128,
        base,offset,diag,x,force,work.data(),1.e-10,&r};
    avbdSolveConeQp(a);
    unsigned failures=!r.converged || std::abs(x[0]-.5)>1.e-8 || std::abs(x[1]-.5)>1.e-8 || std::abs(x[3]-.2)>1.e-8;
    std::printf("[CONE_QP] converged=%u iterations=%u products=%u error=%.9g x=%.9g,%.9g,%.9g force=%.9g,%.9g failures=%u\n",
        unsigned(r.converged),r.activeIterations,r.materialProducts,r.residual,x[0],x[1],x[3],force[0],force[1],failures);
    a.iterations=0;avbdSolveConeQp(a);if(r.converged)++failures;
    return failures?1:0;
}
