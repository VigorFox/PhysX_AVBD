// Physical mechanisms on the shared local Table kernel, not a second simulator.
#include "avbd/DyAvbdLocalResponse.h"
#include "avbd_probe_foundation.h"
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <cstdio>
using namespace physx;using namespace physx::Dy;
struct Pool
{
    std::mutex mutex;std::condition_variable wake,done;std::vector<std::thread> workers;
    unsigned epoch=0,finished=0;bool stop=false,dual=false;AvbdLocalSweepArgs args={};
    Pool(){for(unsigned id=0;id<4;++id)workers.emplace_back([this,id]{unsigned seen=0;
        for(;;){std::unique_lock<std::mutex> lock(mutex);wake.wait(lock,[&]{return stop || epoch!=seen;});if(stop)return;
            seen=epoch;auto a=args;const bool d=dual;lock.unlock();
            const unsigned size=a.end-a.begin,start=a.begin+size*id/4,end=a.begin+size*(id+1)/4;
            if(d)a.response->dual(a.step,a.force,start,end);else a.response->primal(a.force,a.step,start,end);
            lock.lock();if(++finished==4)done.notify_one();}});}
    ~Pool(){{std::lock_guard<std::mutex> lock(mutex);stop=true;}wake.notify_all();for(auto& w:workers)w.join();}
    void run(void* value,bool d){std::unique_lock<std::mutex> lock(mutex);args=*static_cast<AvbdLocalSweepArgs*>(value);
        dual=d;finished=0;++epoch;wake.notify_all();done.wait(lock,[&]{return finished==4;});}
};
static Pool* pool=NULL;
static void primal(void* a){pool->run(a,false);}static void dual(void* a){pool->run(a,true);}
static AvbdCommonProgram make(unsigned n)
{
    AvbdCommonProgram p;unsigned start=0;
    for(unsigned b=0;b<n;++b)
    {
        const unsigned dim=b%2?6:3;p.dimensions.pushBack(dim);p.sourceBlocks.pushBack(b);
        AvbdCommonHBlock h={};h.row=h.column=b;
        for(unsigned d=0;d<dim;++d){h.values[6*d+d]=2+.1*d;p.base.pushBack(0);++start;}p.H.pushBack(h);
        if(b){h={};h.row=b;h.column=b-1;h.values[0]=h.values[7]=h.values[14]=-.25;p.H.pushBack(h);std::swap(h.row,h.column);p.H.pushBack(h);}
    }
    return p;
}
int main()
{
    AvbdProbeFoundation foundation;Pool workers;pool=&workers;unsigned failed=0,cases=0;double worst=0;
    for(unsigned threaded=0;threaded<3;++threaded)for(unsigned trial=0;trial<24;++trial)
    {
        auto p=make(16);const double inf=std::numeric_limits<double>::infinity();
        for(unsigned r=0;r<16;++r)
        {
            p.sourceRows.pushBack(r);p.mu.pushBack(.5);p.coulomb.pushBack(trial%3!=0);
            for(unsigned q=0;q<3;++q)
            {p.offset.pushBack(0);p.lower.pushBack(q?-inf:0);p.upper.pushBack(inf);p.compliance.pushBack(trial%3==0?.001:0);}
            AvbdCommonJBlock j={};j.row=r;j.column=r;j.values[0]=j.values[7]=j.values[14]=1;p.J.pushBack(j);
            if(r){j.column=r-1;j.values[0]=j.values[7]=j.values[14]=-1;p.J.pushBack(j);}
        }
        if(!p.finish())return 2;PxArray<double> x(p.base.size()),f(p.offset.size()),jx(f.size()),jtf(x.size());
        for(unsigned b=0;b<p.dimensions.size();++b)for(unsigned d=0;d<p.dimensions[b];++d)x[p.starts[b]+d]=.01*(1+trial)*(1+d);
        for(unsigned r=0;r<p.mu.size();++r){f[3*r]=1+.03*r;f[3*r+1]=(trial%3==2?-.5:.1)*f[3*r];f[3*r+2]=0;}
        p.hessian(x.begin(),p.base.begin());p.transpose(f.begin(),jtf.begin());p.jacobian(x.begin(),jx.begin());
        for(unsigned d=0;d<x.size();++d)p.base[d]-=jtf[d];
        for(unsigned r=0;r<p.mu.size();++r)for(unsigned q=0;q<3;++q)
            p.offset[3*r+q]=-jx[3*r+q]-p.compliance[3*r+q]*f[3*r+q]+(trial%3==2 && q==1?.02:0);
        auto table=KernelTable::makeAvbdCpuScalar();if(threaded==1){table.bound[AVBD_LOCAL_PRIMAL]=primal;table.bound[AVBD_LOCAL_DUAL]=dual;}
        AvbdLocalResponse response(p,table);PxArray<double> dx,force;AvbdCoulombNcpResult result={};
        if(threaded==2){force=f;for(double& v:force)v*=trial%2?4.:.4;}
        bool ok=response.solve(dx,force,result,threaded==2);double error=0;
        for(unsigned d=0;d<x.size();++d)error=std::max(error,std::abs(x[d]-dx[d]));
        p.jacobian(dx.begin(),jx.begin());double rootError=0;
        for(unsigned r=0;r<p.mu.size();++r)
        {double u[3],root[3];for(unsigned q=0;q<3;++q)u[q]=p.offset[3*r+q]+jx[3*r+q]+p.compliance[3*r+q]*force[3*r+q];
            avbdCoulombRoot(force.begin()+3*r,u,1,p.mu[r],p.lower.begin()+3*r,p.upper.begin()+3*r,p.coulomb[r]!=0,root);
            for(double v:root)rootError=std::max(rootError,std::abs(v));}
        ok=ok && error<1.e-7 && rootError<1.e-8;failed+=!ok;++cases;worst=std::max(worst,error);
        if(!ok)std::printf("[LOCAL_BLOCK_CASE] threads=%u trial=%u sweeps=%u stepError=%.9g root=%.9g status=FAIL\n",threaded?4:1,trial,result.iterations,error,rootError);
    }
    // Both endpoints must move together; individually feasible GS is stuck.
    auto p=make(2);p.H[0].values[0]=p.H[1].values[0]=2;
    for(auto& h:p.H)if(h.row!=h.column)h.values[0]=-1;
    p.base[0]=2;p.base[3]=-1;p.sourceRows.pushBack(0);p.mu.pushBack(0);p.coulomb.pushBack(0);
    for(unsigned q=0;q<3;++q){p.offset.pushBack(0);p.lower.pushBack(0);p.upper.pushBack(q?0:std::numeric_limits<double>::infinity());p.compliance.pushBack(0);}
    for(unsigned b=0;b<2;++b){AvbdCommonJBlock j={};j.column=b;j.values[0]=b?1:-1;p.J.pushBack(j);}
    p.finish();auto table=KernelTable::makeAvbdCpuScalar();AvbdLocalResponse response(p,table);PxArray<double> x,f;AvbdCoulombNcpResult r;
    const bool closed=response.solve(x,f,r) && std::abs(x[0]-.5)<1.e-8 && std::abs(x[3]-.5)<1.e-8 && std::abs(f[0]-1.5)<1.e-8;
    failed+=!closed;std::printf("[LOCAL_BLOCK_CLOSED_CHAIN] a=%.12g b=%.12g force=%.12g sweeps=%u status=%s\n",x[0],x[3],f[0],r.iterations,closed?"PASS":"FAIL");
    std::printf("[LOCAL_BLOCK] cases=%u threads=1,4 grouped=1 dimensions=3,6 maximumStepError=%.9g failed=%u status=%s\n",cases,worst,failed,failed?"FAIL":"PASS");
    return failed?1:0;
}
