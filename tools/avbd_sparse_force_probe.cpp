// Sparse-force products and lazy residual units against direct full equations.
#include "avbd/DyAvbdCommonResidual.h"
#include "avbd_probe_foundation.h"
#include <random>
#include <thread>
#include <cstdio>
using namespace physx;using namespace physx::Dy;
int main()
{
    AvbdProbeFoundation foundation;std::mt19937 rng(42128);std::uniform_real_distribution<double> random(-1,1);
    unsigned failures=0,checks=0;double error=0;size_t evaluated=0,available=0;
    for(unsigned test=0;test<8;++test)
    {
        AvbdCommonProgram p;
        for(unsigned b=0;b<4;++b)
        {
            p.dimensions.pushBack(b%2?6:3);p.sourceBlocks.pushBack(b);AvbdCommonHBlock h={};h.row=h.column=b;
            for(unsigned d=0;d<p.dimensions[b];++d){h.values[6*d+d]=2+std::abs(random(rng));p.base.pushBack(random(rng));}p.H.pushBack(h);
        }
        PxArray<double> force;
        for(unsigned r=0;r<4000;++r)
        {
            const bool bilateral=test && r%31==0,friction=!bilateral && r%2==0;
            p.sourceRows.pushBack(r);p.mu.pushBack(friction?.6:0);p.coulomb.pushBack(friction?1:0);
            for(unsigned q=0;q<3;++q)
            {
                p.offset.pushBack(q?random(rng):test && r%127==0?-.1:3+random(rng));
                p.lower.pushBack(q?(friction?-DBL_MAX:0):(bilateral?-2:0));
                p.upper.pushBack(q?(friction?DBL_MAX:0):(bilateral?2:DBL_MAX));
                p.compliance.pushBack(test && r%19==0?.03:0);
                force.pushBack(test && r%41==0?(q?random(rng):std::abs(random(rng))):0);
            }
            for(unsigned b=0;b<4;++b)if((r+b)%3!=0)
            {AvbdCommonJBlock j={};j.row=r;j.column=b;
                for(unsigned q=0;q<3;++q)for(unsigned d=0;d<p.dimensions[b];++d)j.values[6*q+d]=random(rng);p.J.pushBack(j);}
        }
        if(!p.finish())return 2;
        PxArray<double> oracle(p.base.size(),0),actual(p.base.size()),partial(p.base.size()),weights(p.mu.size(),0);
        for(const auto& j:p.J)
            for(unsigned q=0;q<3;++q)for(unsigned d=0;d<p.dimensions[j.column];++d)
            {
                oracle[p.starts[j.column]+d]+=j.values[6*q+d]*force[3*j.row+q];
                weights[j.row]+=j.values[6*q+d]*j.values[6*q+d]/p.H[j.column].values[6*d+d];
            }
        for(unsigned r=0;r<p.mu.size();++r)for(unsigned q=0;q<3;++q)weights[r]+=p.compliance[3*r+q];
        p.transpose(force.begin(),actual.begin());
        std::thread workers[4];for(unsigned b=0;b<4;++b)workers[b]=std::thread([&,b](){p.transpose(force.begin(),partial.begin(),b,b+1);});
        for(auto& worker:workers)worker.join();
        for(unsigned d=0;d<actual.size();++d)
        {const double e=std::max(std::abs(actual[d]-oracle[d]),std::abs(partial[d]-oracle[d]))/(1+std::abs(oracle[d]));error=std::max(error,e);failures+=e>1.e-12;++checks;}
        AvbdCommonResidual metric(p);double expected=0;
        for(unsigned b=0;b<4;++b)for(unsigned d=0;d<p.dimensions[b];++d)
        {const double v=oracle[p.starts[b]+d]+p.base[p.starts[b]+d];expected+=v*v/p.H[b].values[6*d+d];}
        for(unsigned r=0;r<p.mu.size();++r)
        {double u[3],root[3];for(unsigned q=0;q<3;++q)u[q]=p.offset[3*r+q]+p.compliance[3*r+q]*force[3*r+q];
            const double w=weights[r]>0?weights[r]:1.;avbdCoulombRoot(force.begin()+3*r,u,1/w,p.mu[r],p.lower.begin()+3*r,p.upper.begin()+3*r,p.coulomb[r]!=0,root);
            for(double v:root)expected+=v*v*w;}
        const double residualError=std::abs(metric.evaluate(p,force.begin())-expected)/(1+std::abs(expected));
        error=std::max(error,residualError);failures+=residualError>1.e-12;++checks;
        for(double v:metric.compliance)evaluated+=v>=0;available+=metric.compliance.size();
        // The candidate J/offset changes while residual units stay frozen.
        AvbdCommonProgram candidate=p;for(auto& j:candidate.J)for(double& v:j.values)v*=1.07;
        PxArray<double> step(p.base.size()),tolerance(p.base.size());for(double& v:step)v=.02*random(rng);
        for(unsigned trial=0;trial<3;++trial)
        {
            for(double& v:tolerance)v=trial==0?0:trial==1?.001:100;
            if(test==0 && trial==1)candidate.offset[0]=-1;
            bool reference=true;
            for(unsigned r=0;r<p.mu.size();++r)
            {
                double u[3],root[3],limit[3]={};for(unsigned q=0;q<3;++q)u[q]=candidate.offset[3*r+q]+p.compliance[3*r+q]*force[3*r+q];
                for(unsigned e=candidate.jOffsets[r];e<candidate.jOffsets[r+1];++e)
                {const auto& j=candidate.J[e];for(unsigned q=0;q<3;++q)for(unsigned d=0;d<p.dimensions[j.column];++d)
                    {if(trial)u[q]+=j.values[6*q+d]*step[p.starts[j.column]+d];limit[q]+=std::abs(j.values[6*q+d])*tolerance[p.starts[j.column]+d];}}
                const double w=weights[r]>0?weights[r]:1.;avbdCoulombRoot(force.begin()+3*r,u,1/w,p.mu[r],p.lower.begin()+3*r,p.upper.begin()+3*r,p.coulomb[r]!=0,root);
                for(unsigned q=0;q<3;++q)reference=reference && std::abs(root[q])*w<=limit[q];
            }
            failures+=metric.smallRoot(candidate,force.begin(),tolerance.begin(),trial?step.begin():NULL)!=reference;++checks;
        }
    }
    std::printf("[SPARSE_FORCE] checks=%u relativeError=%.12g unitsEvaluated=%zu available=%zu failures=%u status=%s\n",checks,error,evaluated,available,failures,failures?"FAIL":"PASS");
    return failures?1:0;
}
