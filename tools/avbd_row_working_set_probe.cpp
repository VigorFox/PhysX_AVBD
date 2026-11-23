// Exercise the production compact view and complete-row Table verification.
#include "avbd/DyAvbdRowWorkingSet.h"
#include "avbd_probe_foundation.h"
#include "avbd_bounded_probe_pool.h"
#include <cfloat>
#include <cstdio>
using namespace physx;using namespace physx::Dy;
static AvbdCommonProgram program(unsigned rows,bool impact,bool paired=false)
{
    AvbdCommonProgram p;p.dimensions.pushBack(paired?3:6);p.sourceBlocks.pushBack(0);p.base.resize(paired?9:6,0);
    AvbdCommonHBlock h={};for(unsigned d=0;d<p.dimensions[0];++d)h.values[6*d+d]=2;p.H.pushBack(h);
    p.base[0]=impact?-2:.02;p.base[1]=impact?.4:0;
    if(paired)
    {p.dimensions.pushBack(6);p.sourceBlocks.pushBack(1);h={};h.row=h.column=1;
        for(unsigned d=0;d<6;++d)h.values[6*d+d]=2;p.H.pushBack(h);p.base[3]=-p.base[0];p.base[4]=-p.base[1];}
    for(unsigned r=0;r<rows;++r)
    {
        p.sourceRows.pushBack(r);p.mu.pushBack(.5);p.coulomb.pushBack(1);
        for(unsigned q=0;q<3;++q)
        {p.offset.pushBack(q?0:(r==0 && impact?.25:10+r*.001));p.compliance.pushBack(0);
            p.lower.pushBack(q?-DBL_MAX:0);p.upper.pushBack(DBL_MAX);}
        AvbdCommonJBlock j={};j.row=r;j.column=0;j.values[0]=j.values[7]=j.values[14]=1;p.J.pushBack(j);
        if(paired){j.column=1;j.values[0]=j.values[7]=j.values[14]=-1;p.J.pushBack(j);}
    }
    p.finish();return p;
}
static bool solve(AvbdCommonProgram& p,AvbdBoundedProbePool& pool,bool threaded)
{
    AvbdRowWorkingSet rows;PxArray<double> force(p.offset.size(),0),step;
    rows.initialize(p,force.begin(),force.begin());bool solved=false;unsigned maximum=0;
    while(rows.sweeps<AvbdBoundedResponse::SweepLimit)
    {
        if(!rows.compile(p,force.begin()))return false;
        AvbdBoundedResponse r(rows.active,rows.seed.begin(),&p,&rows.map,AvbdBoundedResponse::SweepLimit-rows.sweeps,
            step.empty()?NULL:step.begin(),&rows.bounds);
        maximum=std::max(maximum,r.input.size());
        if(threaded)pool.solve(r);else r.solve(KernelTable::makeAvbdCpuScalar());
        rows.sweeps+=r.sweeps;rows.scatter(r.force,force);step=r.step;
        if(r.status==AvbdBoundedResponse::RowsRequired && rows.sweeps<AvbdBoundedResponse::SweepLimit && rows.grow(r.requests))continue;
        solved=r.status==AvbdBoundedResponse::Converged;break;
    }
    // Independent original H/J and the original Coulomb root, not the compact
    // solver's own success flag or an equality with another execution.
    PxArray<double> h(p.base.size()),reaction(h.size()),j(force.size());
    p.hessian(step.begin(),h.begin());p.transpose(force.begin(),reaction.begin());p.jacobian(step.begin(),j.begin());
    double station=0,rootError=0;
    for(unsigned d=0;d<h.size();++d)station=std::max(station,std::abs(h[d]-p.base[d]-reaction[d]));
    for(unsigned r=0;r<p.mu.size();++r)
    {double u[3],v[3];for(unsigned q=0;q<3;++q)u[q]=p.offset[3*r+q]+j[3*r+q];
        avbdCoulombRoot(force.begin()+3*r,u,1,p.mu[r],p.lower.begin()+3*r,p.upper.begin()+3*r,true,v);
        for(double x:v)rootError=std::max(rootError,std::abs(x));}
    const bool impact=p.base[0]<0;
    const bool paired=p.dimensions.size()==2;
    double momentum=0;if(paired)for(unsigned d=0;d<3;++d)momentum=std::max(momentum,std::abs(reaction[d]+reaction[3+d]));
    const bool ok=solved && station<1.e-8 && rootError<1.e-8 && momentum<1.e-8 && maximum<=p.base.size()+3 &&
        rows.expansions<=AvbdRowWorkingSet::ExpansionLimit && std::abs(step[0]-(impact?(paired?-.125:-.25):.01))<1.e-8;
    std::printf("[ROW_WORKING_SET] blocks=%u full=%u compact=%u historyCoordinates=%u sweeps=%u expansions=%u threads=%u station=%.9g root=%.9g momentum=%.9g status=%s\n",
        p.dimensions.size(),p.mu.size(),rows.active.mu.size(),maximum,rows.sweeps,rows.expansions,threaded?4:1,station,rootError,momentum,ok?"PASS":"FAIL");
    return ok;
}
static bool periodicQuery(AvbdBoundedProbePool& pool)
{
    // An ill-conditioned H-only subset must discover an omitted contact before
    // it spends the entire shared budget trying to converge the subset.
    auto p=program(1,true);p.dimensions.clear();p.sourceBlocks.clear();p.base.clear();p.H.clear();
    const unsigned bodies=128;
    for(unsigned b=0;b<bodies;++b)
    {
        p.dimensions.pushBack(6);p.sourceBlocks.pushBack(b);
        for(unsigned d=0;d<6;++d)p.base.pushBack(b==0 && d==0?-1:0);
        AvbdCommonHBlock h={};h.row=h.column=b;
        for(unsigned d=0;d<6;++d)h.values[6*d+d]=d?1:2.00001;p.H.pushBack(h);
        if(b){h={};h.row=b;h.column=b-1;h.values[0]=-1;p.H.pushBack(h);std::swap(h.row,h.column);p.H.pushBack(h);}
    }
    p.finish();AvbdRowWorkingSet rows;rows.initialize(p,NULL,NULL);rows.compile(p,NULL);
    AvbdBoundedResponse r(rows.active,NULL,&p,&rows.map,64);pool.solve(r);
    const bool ok=r.status==AvbdBoundedResponse::RowsRequired && r.sweeps<64 && r.requests[0];
    std::printf("[ROW_WORKING_SET_PERIODIC] sweeps=%u budget=64 omittedContactFound=%u status=%s\n",r.sweeps,r.requests[0],ok?"PASS":"FAIL");
    return ok;
}
int main()
{
    AvbdProbeFoundation foundation;AvbdBoundedProbePool pool;unsigned failed=0;
    for(bool paired:{false,true})for(bool impact:{false,true})for(bool threaded:{false,true})
    {auto p=program(10001,impact,paired);failed+=!solve(p,pool,threaded);}
    auto p=program(3,false);PxArray<double> old(p.offset.size(),0),seed(old.size(),0);old[3]=1;
    p.coulomb[2]=0;p.lower[6]=-1;AvbdRowWorkingSet rows;rows.initialize(p,seed.begin(),old.begin());
    bool retained=rows.keep[1] && rows.keep[2] && !rows.keep[0];failed+=!retained;
    std::printf("[ROW_WORKING_SET_HISTORY] positiveGapReaction=retained bilateral=retained status=%s\n",retained?"PASS":"FAIL");
    // A one-sweep budget may not silently publish the omitted violating row.
    auto missing=program(3,true);rows.initialize(missing,NULL,NULL);rows.compile(missing,NULL);
    AvbdBoundedResponse incomplete(rows.active,NULL,&missing,&rows.map,1);pool.solve(incomplete);
    bool reject=incomplete.status==AvbdBoundedResponse::RowsRequired && incomplete.sweeps==1 && incomplete.requests[0];
    failed+=!reject;std::printf("[ROW_WORKING_SET_BUDGET] unresolvedOmission=rejected status=%s\n",reject?"PASS":"FAIL");
    failed+=!periodicQuery(pool);
    return failed?1:0;
}
