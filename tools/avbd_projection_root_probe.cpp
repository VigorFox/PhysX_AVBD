// Independent analytic projection roots, including unresolved sub-ULP shifts.
#include "avbd/DyAvbdCommonProgram.h"
#include "avbd_probe_foundation.h"
#include <cstdio>
using namespace physx;using namespace physx::Dy;
int main()
{
    AvbdProbeFoundation foundation;unsigned failures=0,cases=0;
    auto check=[&](double actual,double expected)
    {++cases;if(!std::isfinite(actual) || std::abs(actual-expected)>1.e-12*std::max(1.,std::abs(expected)))++failures;};
    const double inf=std::numeric_limits<double>::infinity();
    for(double sign:{-1.,1.})for(double magnitude:{1.,1.e10,1.e20})
    {
        const double f=sign*magnitude;
        check(avbdScalarProjectionRoot(f,.125,-inf,inf),.125);
        check(avbdScalarProjectionRoot(f,-.25,-inf,inf),-.25);
        check(avbdScalarProjectionRoot(f,.125,f,inf),0);
        check(avbdScalarProjectionRoot(f,-.25,-inf,f),0);
    }
    check(avbdScalarProjectionRoot(3,9,0,4),3);
    check(avbdScalarProjectionRoot(3,-9,0,4),-1);
    check(avbdScalarProjectionRoot(3,9,2,2),1);
    const double lo[3]={0,-inf,-inf},hi[3]={inf,inf,inf};
    double f[3]={1.e20,1.e18,-1.e18},u[3]={.125,-.25,.5},root[3];
    avbdCoulombRoot(f,u,2,.5,lo,hi,true,root);
    for(unsigned q=0;q<3;++q)check(root[q],2*u[q]);
    f[0]=4;f[1]=6;f[2]=8;u[0]=0;u[1]=u[2]=0;
    avbdCoulombRoot(f,u,1,.5,lo,hi,true,root);
    check(root[1],4.8);check(root[2],6.4);
    f[0]=0;f[1]=1.e-5;f[2]=-2.e-5;u[1]=1.e20;u[2]=-1.e20;
    avbdCoulombRoot(f,u,1,0,lo,hi,true,root);
    check(root[1],f[1]);check(root[2],f[2]);
    // The implicit compliant AL map retains its own projected-normal capacity.
    AvbdCommonProgram p;p.dimensions.pushBack(3);p.sourceBlocks.pushBack(0);p.base.resize(3,0);
    AvbdCommonHBlock h={};h.values[0]=h.values[7]=h.values[14]=1;p.H.pushBack(h);
    p.sourceRows.pushBack(0);p.mu.pushBack(.5);p.coulomb.pushBack(1);
    for(unsigned q=0;q<3;++q){p.offset.pushBack(q?.25:-.125);p.lower.pushBack(lo[q]);p.upper.pushBack(hi[q]);p.compliance.pushBack(0);}
    p.finish();double x[3]={},penalty[3]={2,3,3},projected[3],stiffness[3],residual[3];
    f[0]=1.e20;f[1]=1.e18;f[2]=-1.e18;
    p.projected(0,penalty,x,f,projected,stiffness,true,NULL,residual);
    for(unsigned q=0;q<3;++q)check(residual[q],penalty[q]*p.offset[q]);
    p.compliance[0]=1.e-20;
    p.projected(0,penalty,x,f,projected,stiffness,true,NULL,residual);check(residual[0],1.75);
    f[0]=1;f[1]=3;f[2]=4;p.compliance[0]=0;p.offset[0]=-.5;p.offset[1]=p.offset[2]=0;
    p.projected(0,penalty,x,f,projected,stiffness,true,NULL,residual);
    check(residual[0],-1);check(residual[1],2.4);check(residual[2],3.2);
    // Zero capacity with a huge proposed tangent must retain a small force.
    p.offset[0]=0;p.offset[1]=1.e20;p.offset[2]=-1.e20;f[0]=0;f[1]=1.e-5;f[2]=-2.e-5;
    p.projected(0,penalty,x,f,projected,stiffness,true,NULL,residual);
    check(residual[1],f[1]);check(residual[2],f[2]);
    std::printf("[PROJECTION_ROOT] cases=%u failed=%u status=%s\n",cases,failures,failures?"FAIL":"PASS");
    return failures?1:0;
}
