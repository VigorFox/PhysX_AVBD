#include "avbd/DyAvbdNormalBounds.h"
#include "avbd_probe_foundation.h"
#include <cstdio>
#include <random>
using namespace physx;using namespace physx::Dy;
int main()
{
    AvbdProbeFoundation foundation;std::mt19937 rng(12813);std::uniform_real_distribution<double> random(-1,1);
    unsigned failed=0,certified=0,fallback=0;
    for(unsigned trial=0;trial<100;++trial)
    {
        AvbdCommonProgram p;for(unsigned b=0;b<4;++b)
        {p.dimensions.pushBack(b%2?6:3);p.sourceBlocks.pushBack(b);}
        p.base.resize(18,0);PxArray<double> step(18,0);
        for(double& v:step)v=random(rng)*(trial%3==0?1.e8:.1);
        for(unsigned r=0;r<257;++r)
        {
            p.sourceRows.pushBack(r);p.mu.pushBack(.7);p.coulomb.pushBack(1);
            for(unsigned q=0;q<3;++q)
            {p.offset.pushBack(q?random(rng):trial%3==0?random(rng)*1.e-10:trial%3==1?10.:random(rng));
                p.lower.pushBack(q?-DBL_MAX:0);p.upper.pushBack(DBL_MAX);p.compliance.pushBack(0);}
            // The support changes inside nominal 32-row intervals.
            for(unsigned b=0;b<1+(r/17)%4;++b)
            {AvbdCommonJBlock j={};j.row=r;j.column=b;for(unsigned q=0;q<3;++q)for(unsigned d=0;d<p.dimensions[b];++d)j.values[6*q+d]=random(rng);p.J.pushBack(j);}
        }
        if(!p.finish()){++failed;continue;}
        PxArray<AvbdNormalBound> bounds;avbdCompileNormalBounds(p,bounds);unsigned covered=0;
        for(const auto& b:bounds)
        {
            failed+=b.first!=covered || b.count>32;covered+=b.count;
            if(!b.nonnegative(step.begin(),p.starts)){++fallback;continue;}++certified;
            for(unsigned r=b.first;r<b.first+b.count;++r)
            {double u[3];p.rowCoordinates(r,step.begin(),u);failed+=!std::isfinite(u[0]) || !std::isfinite(u[1]) || !std::isfinite(u[2]) || u[0]<0;}
        }
        failed+=covered!=p.mu.size();
        // An offset change (e.g. SOC) must rebuild its bounds.
        for(unsigned r=0;r<p.mu.size();++r)p.offset[3*r]=-1.e20;
        avbdCompileNormalBounds(p,bounds);for(const auto& b:bounds)failed+=b.nonnegative(step.begin(),p.starts);
    }
    AvbdNormalBound cancellation;cancellation.count=1;cancellation.arity=2;cancellation.columns[0]=0;cancellation.columns[1]=1;
    cancellation.offset=-1.e-10;cancellation.low[0]=cancellation.high[0]=1;cancellation.low[6]=cancellation.high[6]=-1;
    PxArray<PxU32> starts;starts.pushBack(0);starts.pushBack(3);starts.pushBack(6);double step[6]={1.e8,0,0,1.e8,0,0};
    failed+=cancellation.nonnegative(step,starts);
    AvbdNormalBound overflow;overflow.count=1;overflow.arity=1;overflow.columns[0]=0;overflow.offset=1;
    overflow.coordinateCoefficient=1.e300;step[0]=1.e10;failed+=overflow.nonnegative(step,starts);
    failed+=certified==0 || fallback==0;
    std::printf("[NORMAL_BOUNDS] certified=%u uncertain=%u supportCoverage=complete offsetRebuilt=1 cancellationRejected=1 failed=%u status=%s\n",certified,fallback,failed,failed?"FAIL":"PASS");
    return failed?1:0;
}
