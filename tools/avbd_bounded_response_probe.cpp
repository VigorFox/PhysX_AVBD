// Bounded production Table kernels; the test pool is not part of the SDK.
#include "avbd/DyAvbdBoundedResponse.h"
#include "avbd/DyAvbdContactWarmIndex.h"
#include "avbd_probe_foundation.h"
#include "avbd_bounded_probe_pool.h"
#include <cstdio>
using namespace physx;using namespace physx::Dy;
static AvbdCommonProgram make(unsigned n,bool star=false,bool fixed=false)
{
    AvbdCommonProgram p;const double inf=std::numeric_limits<double>::infinity();
    for(unsigned b=0;b<n;++b)
    {
        const unsigned dim=b%2?6:3;p.dimensions.pushBack(dim);p.sourceBlocks.pushBack(b);
        AvbdCommonHBlock h={};h.row=h.column=b;
        for(unsigned d=0;d<dim;++d){h.values[6*d+d]=2+.1*d;p.base.pushBack(0);}p.H.pushBack(h);
        if(b)
        {
            h={};h.row=b;h.column=star?0:b-1;h.values[0]=h.values[7]=h.values[14]=star?-.001:-.25;
            p.H.pushBack(h);std::swap(h.row,h.column);p.H.pushBack(h);
            unsigned r=p.mu.size();p.sourceRows.pushBack(r);p.mu.pushBack(.5);p.coulomb.pushBack(!fixed);
            for(unsigned q=0;q<3;++q)
            {p.offset.pushBack(0);p.lower.pushBack(fixed?(q?0:1):(q?-inf:0));p.upper.pushBack(fixed?(q?0:1):inf);p.compliance.pushBack(0);}
            AvbdCommonJBlock j={};j.row=r;j.column=b;j.values[0]=j.values[7]=j.values[14]=1;p.J.pushBack(j);
            j.column=star?0:b-1;j.values[0]=j.values[7]=j.values[14]=-1;p.J.pushBack(j);
        }
    }
    p.finish();return p;
}
static bool fullResidual(const AvbdCommonProgram& p,const AvbdBoundedResponse& solved,double& station,double& rootError)
{
    PxArray<double> hx(p.base.size()),reaction(p.base.size()),jx(p.offset.size());
    p.hessian(solved.step.begin(),hx.begin());p.transpose(solved.force.begin(),reaction.begin());p.jacobian(solved.step.begin(),jx.begin());
    station=rootError=0;
    for(double v:solved.step)if(!std::isfinite(v))return false;
    for(double v:solved.force)if(!std::isfinite(v))return false;
    for(unsigned d=0;d<hx.size();++d)
    {const double v=hx[d]-p.base[d]-reaction[d];if(!std::isfinite(v))return false;station=std::max(station,std::abs(v));}
    for(unsigned r=0;r<p.mu.size();++r)
    {
        double u[3],root[3];for(unsigned q=0;q<3;++q)u[q]=p.offset[3*r+q]+jx[3*r+q]+p.compliance[3*r+q]*solved.force[3*r+q];
        avbdCoulombRoot(solved.force.begin()+3*r,u,1,p.mu[r],p.lower.begin()+3*r,p.upper.begin()+3*r,p.coulomb[r]!=0,root);
        for(double v:root){if(!std::isfinite(v))return false;rootError=std::max(rootError,std::abs(v));}
    }
    return station<1.e-7 && rootError<1.e-8;
}
static bool check(AvbdCommonProgram& p,AvbdBoundedProbePool& pool,const char* name,bool fixed=false)
{
    PxArray<double> target(p.base.size()),force(p.offset.size(),0),jx(force.size()),reaction(target.size());
    for(unsigned b=0;b<p.dimensions.size();++b)for(unsigned d=0;d<p.dimensions[b];++d)target[p.starts[b]+d]=.01*(1+d);
    for(unsigned r=0;r<p.mu.size();++r){force[3*r]=1+.03*(r%3);if(fixed)force[3*r]=1;else force[3*r+1]=.1*force[3*r];}
    p.hessian(target.begin(),p.base.begin());p.transpose(force.begin(),reaction.begin());p.jacobian(target.begin(),jx.begin());
    for(unsigned d=0;d<target.size();++d)p.base[d]-=reaction[d];
    for(unsigned i=0;i<jx.size();++i)p.offset[i]=-jx[i];
    const auto table=KernelTable::makeAvbdCpuScalar();AvbdBoundedResponse a(p),b(p);a.solve(table);pool.solve(b);
    double error=0,difference=0;for(unsigned d=0;d<target.size();++d)
    {error=std::max(error,std::max(std::abs(a.step[d]-target[d]),std::abs(b.step[d]-target[d])));difference=std::max(difference,std::abs(a.step[d]-b.step[d]));}
    // Independent full operators and original roots, rather than only the
    // executor's reduction tree or a known target displacement.
    double station=0,rootError=0,parallelStation=0,parallelRoot=0;
    const bool roots=fullResidual(p,a,station,rootError) && fullResidual(p,b,parallelStation,parallelRoot);
    bool bounded=a.sweeps<=AvbdBoundedResponse::SweepLimit && b.sweeps<=AvbdBoundedResponse::SweepLimit;
    for(const auto* solved:{&a,&b})for(const auto& node:solved->nodes)
        bounded=bounded && node.count<=unsigned(node.level?AvbdBoundedResponse::FanIn:AvbdBoundedResponse::LeafWidth);
    // A corrupted threaded result must fail even when the serial solve passes.
    const double saved=b.step[0];b.step[0]+=1;double badStation=0,badRoot=0;
    const bool rejectsCorruption=!fullResidual(p,b,badStation,badRoot);b.step[0]=saved;
    const bool ok=a.status==AvbdBoundedResponse::Converged && b.status==AvbdBoundedResponse::Converged && error<1.e-7 && bounded && roots && rejectsCorruption;
    std::printf("[BOUNDED_CASE] name=%s blocks=%u rows=%u colors=%u depth=%u sweeps=%u error=%.9g threadDifference=%.9g status=%s\n",
        name,p.dimensions.size(),p.mu.size(),a.colorCount,a.maxDepth,a.sweeps,error,difference,ok?"PASS":"FAIL");return ok;
}
static bool exchangeCases(AvbdBoundedProbePool& pool)
{
    unsigned failures=0,maximumSweeps=0;double worst=0;
    for(unsigned trial=0;trial<12;++trial)
    {
        auto p=make(64);p.J.clear();p.sourceRows.clear();p.mu.clear();p.coulomb.clear();
        p.offset.clear();p.lower.clear();p.upper.clear();p.compliance.clear();
        const double epsilon=std::pow(10.,-3.-3.*(trial%3)),inf=std::numeric_limits<double>::infinity();
        const bool limited=trial%2!=0;PxArray<double> targetForce,seed;
        for(unsigned b=0;b<p.dimensions.size();++b)for(unsigned side=0;side<2;++side)
        {
            const unsigned r=p.mu.size();p.mu.pushBack(.5);p.coulomb.pushBack(1);p.sourceRows.pushBack(r);
            AvbdCommonJBlock j={};j.row=r;j.column=b;j.values[0]=1;j.values[1]=side?0:epsilon;j.values[7]=j.values[14]=1;p.J.pushBack(j);
            const double normal=limited?(side?.6:.4):(side?1:0);
            for(unsigned q=0;q<3;++q)
            {
                p.offset.pushBack(q?0:(limited?(side?-.01:0):(side?0:1.e-6)));
                p.lower.pushBack(q?-inf:0);p.upper.pushBack(!q && side && limited?.6:inf);p.compliance.pushBack(0);
                targetForce.pushBack(q?(q==1?.2*normal:0):normal);seed.pushBack(q?(q==1?.1:0):.5);
            }
        }
        p.finish();p.transpose(targetForce.begin(),p.base.begin());for(double& v:p.base)v=-v;
        AvbdBoundedResponse response(p,trial<6?seed.begin():NULL);pool.solve(response);double error=0;
        for(double x:response.step)error=std::max(error,std::abs(x));
        PxArray<double> reaction(p.base.size()),hx(p.base.size()),coordinates(p.offset.size());
        p.hessian(response.step.begin(),hx.begin());p.transpose(response.force.begin(),reaction.begin());p.jacobian(response.step.begin(),coordinates.begin());
        double station=0,rootError=0;
        for(unsigned i=0;i<hx.size();++i)station=std::max(station,std::abs(hx[i]-p.base[i]-reaction[i]));
        for(unsigned r=0;r<p.mu.size();++r)
        {double u[3],root[3];for(unsigned q=0;q<3;++q)u[q]=p.offset[3*r+q]+coordinates[3*r+q];
            avbdCoulombRoot(response.force.begin()+3*r,u,1,p.mu[r],p.lower.begin()+3*r,p.upper.begin()+3*r,true,root);
            for(double v:root)rootError=std::max(rootError,std::abs(v));}
        bool ok=response.status==AvbdBoundedResponse::Converged && error<1.e-8 && station<1.e-8 && rootError<1.e-9;
        failures+=!ok;worst=std::max(worst,error);maximumSweeps=std::max(maximumSweeps,response.sweeps);
        if(!ok)std::printf("[BOUNDED_EXCHANGE_CASE] trial=%u stepError=%.9g root=%.9g sweeps=%u status=FAIL\n",trial,error,rootError,response.sweeps);
        // Exact eligibility is essential: nearby tangents or physical tangent
        // compliance cannot use this force-preserving numerical coordinate.
        auto perturbed=p;perturbed.J[1].values[7]+=1.e-12;
        failures+=perturbed.sameTangent(0,1);perturbed=p;perturbed.compliance[1]=1.e-8;failures+=perturbed.sameTangent(0,1);
    }
    std::printf("[BOUNDED_EXCHANGE] cases=12 bodies=64 epsilon=1e-3,1e-6,1e-9 coldWarm=1 finiteForceLimit=1 maxSweeps=%u stepError=%.9g failed=%u status=%s\n",
        maximumSweeps,worst,failures,failures?"FAIL":"PASS");return failures==0;
}
static bool incompatibleStickCases(AvbdBoundedProbePool& pool)
{
    unsigned failed=0;double worstBalance=0,maximumCorrection=0;
    for(unsigned trial=0;trial<8;++trial)
    {
        AvbdCommonProgram p;p.dimensions.pushBack(6);p.sourceBlocks.pushBack(0);p.base.resize(6,0);p.base[1]=-1;
        AvbdCommonHBlock h={};for(unsigned i=0;i<6;++i)h.values[6*i+i]=2;p.H.pushBack(h);
        const double inf=std::numeric_limits<double>::infinity(),epsilon=trial%2?1.e-8:1.e-6,perturb=trial%2?1.e-6:0;
        p.base[3]=.25*perturb;
        double seed[6]={.5,0,0,.5,0,0};
        for(unsigned r=0;r<2;++r)
        {
            p.sourceRows.pushBack(r);p.mu.pushBack(.5);p.coulomb.pushBack(1);
            for(unsigned q=0;q<3;++q)
            {p.offset.pushBack(q==1?(r?-epsilon:epsilon):0);p.lower.pushBack(q?-inf:0);p.upper.pushBack(!q && trial%4>=2?.5:inf);p.compliance.pushBack(0);}
            AvbdCommonJBlock j={};j.row=r;j.values[1]=1;j.values[5]=r?-1:1;
            j.values[6]=1;j.values[9]=r?0:perturb;j.values[14]=1;j.values[16]=r?1:-1;p.J.pushBack(j);
        }
        p.finish();AvbdBoundedResponse response(p,trial<4?seed:NULL);
        const auto table=KernelTable::makeAvbdCpuScalar();bool measured=false;
        // Independently verify the local correction balances its actual full
        // wrench. The QR remainder must not be silently rounded to zero.
        while(response.status==AvbdBoundedResponse::Running && response.sweeps==0)
        {
            const bool exchange=response.waves[response.wave].kind==AvbdBoundedResponse::Exchange && !response.waves[response.wave].check;
            const auto beforeX=response.step,beforeF=response.force;
            for(unsigned begin=0;begin<response.count();begin+=AvbdBoundedResponse::BatchWidth)
            {AvbdBoundedBatchArgs args={&response,begin,std::min(begin+unsigned(AvbdBoundedResponse::BatchWidth),response.count()),false};table.launch(AVBD_BOUNDED_BATCH,&args);}
            if(exchange)
            {
                PxArray<double> df=response.force,dx=response.step,hx(6),reaction(6);
                for(unsigned i=0;i<df.size();++i)df[i]-=beforeF[i];for(unsigned i=0;i<6;++i)dx[i]-=beforeX[i];
                p.hessian(dx.begin(),hx.begin());p.transpose(df.begin(),reaction.begin());
                for(unsigned i=0;i<6;++i)
                {worstBalance=std::max(worstBalance,std::abs(hx[i]-reaction[i]));maximumCorrection=std::max(maximumCorrection,std::abs(dx[i]));}measured=true;
            }
            response.advance();
        }
        pool.solve(response);double motion=0;for(double x:response.step)motion=std::max(motion,std::abs(x));
        bool ok=measured && response.status==AvbdBoundedResponse::Converged && motion<1.e-8 &&
            std::abs(response.force[1]+.25)<1.e-8 && std::abs(response.force[4]-.25)<1.e-8;
        failed+=!ok;if(!ok)std::printf("[BOUNDED_STICK_CASE] trial=%u sweeps=%u motion=%.9g forces=%.9g,%.9g status=FAIL\n",trial,response.sweeps,motion,response.force[1],response.force[4]);
    }
    failed+=worstBalance>1.e-12 || maximumCorrection<1.e-9;
    std::printf("[BOUNDED_STICK_EXCHANGE] cases=8 coldWarm=1 boundedNormal=1 maximumWrenchError=%.9g maximumCorrection=%.9g failed=%u status=%s\n",worstBalance,maximumCorrection,failed,failed?"FAIL":"PASS");
    return failed==0;
}
static bool batchCoverage()
{
    // Coverage is a scheduler contract, not numerical/trajectory equality.
    bool passed=true;
    for(unsigned workers:{0u,1u,2u,4u,8u,16u,32u,128u})
    for(unsigned count:{0u,1u,31u,32u,33u,127u,129u,288u,1023u,1024u,1025u,32769u,1000001u})
    {
        unsigned cursor=0,jobs=0;
        while(cursor<count)
        {
            const AvbdCpuBatch p(count-cursor,workers);
            passed=passed && p.tasks>0 && p.tasks<=32 && p.width>0 && p.width<=1024 && p.width%32==0;
            const unsigned start=cursor,end=cursor+p.items;unsigned tiles=0;
            for(unsigned j=0;j<p.tasks;++j)
            {
                const unsigned stop=start+p.offset(j+1);
                passed=passed && cursor==start+p.offset(j) && stop-cursor<=1024;
                passed=passed && stop>cursor;
                while(cursor<stop){cursor+=std::min(32u,stop-cursor);++tiles;}
            }
            passed=passed && cursor==end && tiles==p.tiles;++jobs;
        }
        passed=passed && cursor==count && (!count || jobs<=count);
    }
    for(unsigned n:{PX_MAX_U32,PX_MAX_U32-31,PX_MAX_U32-1024})
    {const AvbdCpuBatch p(n,32);passed=passed && p.items==32768 && p.tasks==32 && p.tiles==1024;}
    const AvbdCpuBatch small(288,4);passed=passed && small.tasks==4;
    std::printf("[BOUNDED_CPU_PACK] coverage=exact taskItemsMax=1024 inFlightMax=32 status=%s\n",passed?"PASS":"FAIL");
    return passed;
}

int main()
{
    AvbdProbeFoundation foundation;AvbdBoundedProbePool pool;unsigned failed=0;
    failed+=!batchCoverage();
    {
        struct Entry {const void* cm;};int keys[130]={};PxArray<Entry> history;
        for(unsigned i=0;i<10001;++i)history.pushBack({&keys[(i*37)%129]});
        const AvbdContactWarmIndex index(history);PxArray<unsigned> seen(history.size(),0);bool ok=true;
        for(unsigned key=0;key<129;++key)
        {
            unsigned visits=0;
            for(unsigned i=index.first(&keys[key]);i!=PX_MAX_U32 && visits<=history.size();i=index.next[i],++visits)
            {ok=ok && history[i].cm==&keys[key] && ++seen[i]==1;}
            ok=ok && visits<=history.size();
        }
        for(unsigned n:seen)ok=ok && n==1;
        ok=ok && index.first(&keys[129])==PX_MAX_U32;
        history.clear();const AvbdContactWarmIndex empty(history);ok=ok && empty.first(&keys[0])==PX_MAX_U32;
        failed+=!ok;std::printf("[CONTACT_HISTORY_INDEX] complete=1 unique=1 missing=1 empty=1 status=%s\n",ok?"PASS":"FAIL");
    }
    // A reused certificate must solve each new RHS independently. Poison the
    // old numerical scratch to expose stale history/inverse/reduction reads.
    {
        auto source=make(129);AvbdCommonProgram p;
        p.dimensions=source.dimensions;p.sourceBlocks=source.sourceBlocks;p.H=source.H;p.base=source.base;
        if(!p.finish())return 1;
        AvbdBoundedResponse r(p);bool ok=true;
        for(unsigned pass=0;pass<4;++pass)
        {
            PxArray<double> target(p.base.size());
            for(unsigned i=0;i<target.size();++i)target[i]=double(int((i+pass)%7)-3)*(pass==2?1.e-8:.1*(pass+1));
            p.hessian(target.begin(),p.base.begin());
            if(pass)
            {
                for(auto* values:{&r.step,&r.inverse,&r.previousResidual,&r.previousOutput,&r.input,&r.output})
                    for(double& value:*values)value=std::numeric_limits<double>::quiet_NaN();
                for(unsigned k=0;k<4;++k)for(auto* values:{&r.dr[k],&r.dy[k]})
                    for(double& value:*values)value=std::numeric_limits<double>::quiet_NaN();
                ok=r.restartHOnly() && ok;
            }
            pool.solve(r);double station=0,root=0;
            ok=ok && r.status==AvbdBoundedResponse::Converged && fullResidual(p,r,station,root);
            for(unsigned i=0;i<target.size();++i)ok=ok && std::abs(r.step[i]-target[i])<1.e-8;
        }
        p.base[0]=std::numeric_limits<double>::infinity();ok=ok && !r.restartHOnly();
        auto mixed=make(2);AvbdBoundedResponse contact(mixed);ok=ok && !contact.restartHOnly();
        failed+=!ok;std::printf("[BOUNDED_RESTART] rhs=4 coldHistory=1 rejectsNonFinite=1 status=%s\n",ok?"PASS":"FAIL");
    }
    failed+=!exchangeCases(pool);
    failed+=!incompatibleStickCases(pool);
    auto small=make(16);failed+=!check(small,pool,"coupled_coulomb");
    auto four=make(4);for(unsigned i=2;i<4;++i)
    {AvbdCommonJBlock j={};j.row=0;j.column=i;j.values[0]=j.values[7]=j.values[14]=i==2?1:-1;four.J.pushBack(j);}four.finish();
    failed+=!check(four,pool,"four_support_row");
    auto star=make(1025,true,true);failed+=!check(star,pool,"high_degree_star",true);
    auto chain=make(1025,false,true);failed+=!check(chain,pool,"long_chain",true);
    // Exact equilibrium of a long compressed chain: all x coordinates agree,
    // and summing H gives (1.5*n+.5)*x=1. This is a convergence gate now.
    auto hard=make(257);hard.base[0]=1;AvbdBoundedResponse longHard(hard);longHard.solve(KernelTable::makeAvbdCpuScalar());
    AvbdBoundedResponse longThreaded(hard);pool.solve(longThreaded);double chainError=0;
    for(unsigned b=0;b<hard.dimensions.size();++b)
    {chainError=std::max(chainError,std::abs(longHard.step[hard.starts[b]]-1./386.));
        chainError=std::max(chainError,std::abs(longThreaded.step[hard.starts[b]]-1./386.));}
    PxArray<double> hx(hard.base.size()),reaction(hard.base.size()),coordinates(hard.offset.size());
    hard.hessian(longHard.step.begin(),hx.begin());hard.transpose(longHard.force.begin(),reaction.begin());hard.jacobian(longHard.step.begin(),coordinates.begin());
    double chainRoot=0;for(unsigned i=0;i<hx.size();++i)chainRoot=std::max(chainRoot,std::abs(hx[i]-hard.base[i]-reaction[i]));
    for(unsigned r=0;r<hard.mu.size();++r)
    {double root[3];avbdCoulombRoot(longHard.force.begin()+3*r,coordinates.begin()+3*r,1,hard.mu[r],hard.lower.begin()+3*r,hard.upper.begin()+3*r,true,root);
        for(double value:root)chainRoot=std::max(chainRoot,std::abs(value));}
    const bool chainPassed=longHard.status==AvbdBoundedResponse::Converged && longThreaded.status==AvbdBoundedResponse::Converged && chainError<1.e-7 && chainRoot<1.e-8;
    failed+=!chainPassed;
    std::printf("[BOUNDED_HARD_CHAIN] blocks=257 rows=256 sweeps=%u threads=1,4 stepError=%.9g fullRoot=%.9g status=%s\n",
        longHard.sweeps,chainError,chainRoot,chainPassed?"PASS":"FAIL");
    // A stiff two-body contact separates displacement accuracy from force
    // balance. Its exact equilibrium is x=0, f_n=1; check original operators,
    // including every coordinate on both supports, at several stiffnesses.
    for(double stiffness:{1.e2,1.e5,1.e8})
    {
        auto p=make(2);for(auto& h:p.H)
        {for(double& v:h.values)v=0;for(unsigned d=0;d<std::min(p.dimensions[h.row],p.dimensions[h.column]);++d)
            h.values[6*d+d]=h.row==h.column?stiffness:-.5*stiffness;}
        p.base[0]=1;p.base[p.starts[1]]=-1;
        bool passed=true;unsigned sweeps=0,balanced=0;double maximumStation=0,maximumRoot=0;
        for(unsigned threads:{1u,4u})
        {
            AvbdBoundedResponse r(p);if(threads==1)r.solve(KernelTable::makeAvbdCpuScalar());else pool.solve(r);
            PxArray<double> hx(p.base.size()),jf(p.base.size()),u(p.offset.size());
            p.hessian(r.step.begin(),hx.begin());p.transpose(r.force.begin(),jf.begin());p.jacobian(r.step.begin(),u.begin());
            double station=0,motion=0,root[3]={};
            for(unsigned i=0;i<hx.size();++i){station=std::max(station,std::abs(hx[i]-p.base[i]-jf[i]));motion=std::max(motion,std::abs(r.step[i]));}
            avbdCoulombRoot(r.force.begin(),u.begin(),1,p.mu[0],p.lower.begin(),p.upper.begin(),true,root);
            double defect=0;for(double v:root)defect=std::max(defect,std::abs(v));
            passed=passed && r.status==AvbdBoundedResponse::Converged && station<1.e-10 && defect<1.e-10 && motion<1.e-9;
            sweeps=std::max(sweeps,r.sweeps);balanced=std::max(balanced,r.equilibriumSweeps);
            maximumStation=std::max(maximumStation,station);maximumRoot=std::max(maximumRoot,defect);
        }
        failed+=!passed;std::printf("[BOUNDED_STIFF_CONTACT] stiffness=%.9g sweeps=%u balanceSweeps=%u fullStation=%.9g root=%.9g threads=1,4 status=%s\n",
            stiffness,sweeps,balanced,maximumStation,maximumRoot,passed?"PASS":"FAIL");
    }
    // Match the existing publication certificate's infinity-norm residual,
    // including a tiny RHS that the mixed root's unit scale would hide.
    for(double magnitude:{1.e-8,1.,1.e8})
    {
        auto source=make(129);AvbdCommonProgram linear;
        linear.dimensions=source.dimensions;linear.sourceBlocks=source.sourceBlocks;linear.H=source.H;
        linear.base.resize(source.base.size());linear.finish();PxArray<double> target(linear.base.size()),hx(linear.base.size());
        for(unsigned d=0;d<target.size();++d)target[d]=magnitude*(int(d%7)-3);
        linear.hessian(target.begin(),linear.base.begin());
        bool passed=true;unsigned sweeps=0;double maximum=0;
        for(unsigned threads:{1u,4u})
        {
            AvbdBoundedResponse solved(linear);if(threads==1)solved.solve(KernelTable::makeAvbdCpuScalar());else pool.solve(solved);
            linear.hessian(solved.step.begin(),hx.begin());double residual=0,scale=0;
            for(unsigned d=0;d<hx.size();++d){residual=std::max(residual,std::abs(hx[d]-linear.base[d]));scale=std::max(scale,std::abs(linear.base[d]));}
            passed=passed && solved.status==AvbdBoundedResponse::Converged && residual<=std::max(1.e-14,1.e-12*scale);
            maximum=std::max(maximum,residual);sweeps=std::max(sweeps,solved.sweeps);
        }
        failed+=!passed;std::printf("[BOUNDED_FULL_H_CERTIFICATE] magnitude=%.9g sweeps=%u fullResidual=%.9g threads=1,4 status=%s\n",magnitude,sweeps,maximum,passed?"PASS":"FAIL");
    }
    // A different SPD metric may change the iteration, never the original
    // coupled displacement/force root. This catches solving M instead of K.
    auto split=make(17);split.metricH=split.H;
    for(auto& h:split.metricH)if(h.row==h.column)for(unsigned d=0;d<6;++d)h.values[6*d+d]*=4;
    failed+=!split.finish();failed+=!check(split,pool,"operator_metric_split");
    // Two bodies must move together from a closed contact, with one shared dual.
    auto closed=make(2);for(auto& h:closed.H)if(h.row!=h.column)h.values[0]=-1;
    closed.base[0]=2;closed.base[3]=-1;closed.coulomb[0]=0;closed.lower[1]=closed.lower[2]=closed.upper[1]=closed.upper[2]=0;
    AvbdBoundedResponse c(closed);pool.solve(c);
    bool ok=c.status==AvbdBoundedResponse::Converged && std::abs(c.step[0]-.5)<1.e-8 && std::abs(c.step[3]-.5)<1.e-8 && std::abs(c.force[0]-1.5)<1.e-8;
    failed+=!ok;std::printf("[BOUNDED_CLOSED_CHAIN] sweeps=%u a=%.9g b=%.9g force=%.9g status=%s\n",c.sweeps,c.step[0],c.step[3],c.force[0],ok?"PASS":"FAIL");
    // A support-free impossible row must reject within the budget, not pass.
    auto impossible=make(1);impossible.mu.pushBack(0);impossible.coulomb.pushBack(0);impossible.sourceRows.pushBack(0);
    for(unsigned q=0;q<3;++q){impossible.offset.pushBack(q?0:-1);impossible.lower.pushBack(0);impossible.upper.pushBack(q?0:std::numeric_limits<double>::infinity());impossible.compliance.pushBack(0);}impossible.finish();
    AvbdBoundedResponse bad(impossible);bad.solve(KernelTable::makeAvbdCpuScalar());
    ok=bad.status==AvbdBoundedResponse::BudgetExhausted && bad.sweeps<=AvbdBoundedResponse::SweepLimit;
    failed+=!ok;std::printf("[BOUNDED_BUDGET] sweeps=%u status=%s\n",bad.sweeps,ok?"PASS":"FAIL");
    // No hidden row truncation or switch to all-island Jacobi.
    // Opposite unbounded reactions cannot hide incompatible hard anchors.
    auto conflict=make(1);double seed[6]={1.e20,0,0,-1.e20,0,0};
    for(unsigned r=0;r<2;++r)
    {
        conflict.mu.pushBack(0);conflict.coulomb.pushBack(0);conflict.sourceRows.pushBack(r);
        for(unsigned q=0;q<3;++q)
        {conflict.offset.pushBack(q?0:(r?-.1:0));conflict.lower.pushBack(q?0:-std::numeric_limits<double>::infinity());
         conflict.upper.pushBack(q?0:std::numeric_limits<double>::infinity());conflict.compliance.pushBack(0);}
        AvbdCommonJBlock j={};j.row=r;j.column=0;j.values[0]=1;conflict.J.pushBack(j);
    }
    conflict.finish();bool conflictPassed=true;
    for(unsigned threads:{1u,4u})
    {AvbdBoundedResponse r(conflict,seed);if(threads==1)r.solve(KernelTable::makeAvbdCpuScalar());else pool.solve(r);
     conflictPassed=conflictPassed && r.status==AvbdBoundedResponse::BudgetExhausted && r.sweeps<=AvbdBoundedResponse::SweepLimit && r.error>=.05;}
    failed+=!conflictPassed;
    std::printf("[BOUNDED_CONFLICT] opposingForce=1e20 gap=0.1 threads=1,4 status=%s\n",conflictPassed?"PASS":"FAIL");
    auto arity=make(5);for(unsigned i=0;i<5;++i){AvbdCommonJBlock j={};j.row=0;j.column=i;j.values[0]=1;arity.J.pushBack(j);}arity.finish();
    auto capacity=[&](AvbdCommonProgram& p)
    {
        AvbdBoundedResponse r(p);if(r.status==AvbdBoundedResponse::Unsupported)return true;
        pool.solve(r);double station=0,root=0;
        return r.status==AvbdBoundedResponse::Converged && r.sweeps<=AvbdBoundedResponse::SweepLimit && fullResidual(p,r,station,root);
    };
    // Capacity can grow in future: reject explicitly OR solve the full input.
    failed+=!capacity(arity);
    auto clique=make(33);for(unsigned a=0;a<33;++a)for(unsigned b=0;b<33;++b)if(a!=b){AvbdCommonHBlock h={};h.row=a;h.column=b;h.values[0]=.0001;clique.H.pushBack(h);}clique.finish();
    failed+=!capacity(clique);
    auto triangular=make(2);for(unsigned i=0;i<triangular.H.size();++i)if(triangular.H[i].row>triangular.H[i].column)
    {triangular.H.remove(i);break;}triangular.finish();AvbdBoundedResponse unsafeColor(triangular);
    failed+=unsafeColor.status!=AvbdBoundedResponse::Unsupported;
    auto indefinite=make(1);indefinite.H[0].values[0]=-1;AvbdBoundedResponse invalid(indefinite);invalid.solve(KernelTable::makeAvbdCpuScalar());
    failed+=invalid.status!=AvbdBoundedResponse::NumericalFailure;
    failed+=pool.peak.load()<2;
    std::printf("[BOUNDED_RESPONSE] threads=1,4 peak=%u leaf=32 fanIn=8 matrix=6 batch=32 sweepLimit=1024 failed=%u status=%s\n",pool.peak.load(),failed,failed?"FAIL":"PASS");return failed?1:0;
}
