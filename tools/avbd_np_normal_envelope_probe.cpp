#include "avbd/DyAvbdOgcActivation.h"
#include <cstdio>
#include <cmath>
#include <initializer_list>
using namespace physx::Dy;
int main()
{
    unsigned failures=0,checked=0;
    double maxGradient=0,maxCurvature=0,maxStationarity=0;
    for(double rho:{1.,100.,10000.,1000000.})
    for(double lambda:{0.,-1.,-100.,-10000.})
    for(double d:{-.1,-.001,0.,.001,.01,.019,.025,.039,.051,.101})
    {
        const double r=.04,k=1.e5,h=1.e-7;
        const auto value=avbdOGCNormalEnvelope(d,lambda,rho,r,k);
        const auto lo=avbdOGCNormalEnvelope(d-h,lambda,rho,r,k);
        const auto hi=avbdOGCNormalEnvelope(d+h,lambda,rho,r,k);
        const double dg=std::abs((hi.energy-lo.energy)/(2*h)+value.force)/(1+value.force);
        const double dh=std::abs((lo.force-hi.force)/(2*h)-value.hessian)/(1+value.hessian);
        double stationarity=0;
        if(value.force>0)
        {
            const double z=value.force<=k*r/2?r-value.force/k:k*r*r/(4*value.force);
            stationarity=std::abs(value.force+lambda+rho*(d-z))/(1+value.force);
            const auto phi=avbdOGCActivationFullT(z,r,k,r/2);
            if(std::abs(phi.force-value.force)>1.e-7*(1+value.force)) ++failures;
        }
        maxGradient=std::fmax(maxGradient,dg);maxCurvature=std::fmax(maxCurvature,dh);
        maxStationarity=std::fmax(maxStationarity,stationarity);
        if(!std::isfinite(value.energy) || dg>1.e-5 || dh>1.e-5 || stationarity>1.e-10) ++failures;
        ++checked;
    }
    const auto closed=avbdOGCNormalEnvelope(0.,0.,10000.,.04,100000.);
    const auto hard=avbdOGCNormalEnvelope(-.01,-3.,10000.,0.,100000.);
    if(!(closed.force>0 && std::isfinite(closed.energy)) || std::abs(hard.force-103)>1.e-12) ++failures;
    // Release is C1, not C2. Check its one-sided quadratic law instead of
    // comparing a central second difference with one branch's curvature.
    const double release=.05,h=1.e-6,rho=10000.,kc=100000.,effective=rho*kc/(rho+kc);
    const auto left=avbdOGCNormalEnvelope(release-h,-100.,rho,.04,kc);
    const auto right=avbdOGCNormalEnvelope(release+h,-100.,rho,.04,kc);
    if(std::abs(left.force-effective*h)>1.e-9 ||
        std::abs(left.energy-.5*effective*h*h)>1.e-13 || right.force!=0 || right.energy!=0) ++failures;
    std::printf("[NP_NORMAL_ENVELOPE] cases=%u gradient=%.9g curvature=%.9g stationarity=%.9g zeroGapForce=%.9g failures=%u\n",
        checked,maxGradient,maxCurvature,maxStationarity,closed.force,failures);
    return failures?1:0;
}
