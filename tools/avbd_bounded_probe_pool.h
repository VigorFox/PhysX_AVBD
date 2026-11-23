// Test-only fixed-batch pool; SDK integration must use its Host dispatcher.
#ifndef AVBD_BOUNDED_PROBE_POOL_H
#define AVBD_BOUNDED_PROBE_POOL_H
#include "avbd/DyAvbdBoundedResponse.h"
#include "avbd/DyAvbdCpuBatch.h"
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <atomic>
using namespace physx;using namespace physx::Dy;
struct AvbdBoundedProbePool
{
    std::mutex mutex;std::condition_variable wake,done;std::vector<std::thread> workers;
    unsigned epoch=0,finished=0;AvbdCpuBatch batch;bool stop=false;AvbdBoundedBatchArgs args={};
    const KernelTable table=KernelTable::makeAvbdCpuScalar();
    std::atomic<unsigned> active{0},peak{0};std::atomic<bool> rejected{false};
    AvbdBoundedProbePool(){for(unsigned id=0;id<4;++id)workers.emplace_back([this,id]{unsigned seen=0;
        for(;;){std::unique_lock<std::mutex> lock(mutex);wake.wait(lock,[&]{return stop || epoch!=seen;});if(stop)return;
            seen=epoch;auto a=args;const auto pack=batch;lock.unlock();
            a.end=a.begin+pack.offset(id+1);a.begin+=pack.offset(id);
            if(a.begin<a.end)
            {unsigned value=++active,old=peak.load();while(old<value && !peak.compare_exchange_weak(old,value)){}
                const unsigned end=a.end;
                while(a.begin<end){a.end=std::min(end,a.begin+unsigned(AvbdBoundedResponse::BatchWidth));
                    a.executed=false;table.launch(AVBD_BOUNDED_BATCH,&a);if(!a.executed)rejected=true;a.begin=a.end;}--active;}
            lock.lock();if(++finished==4)done.notify_one();}});}
    ~AvbdBoundedProbePool(){{std::lock_guard<std::mutex> lock(mutex);stop=true;}wake.notify_all();for(auto& w:workers)w.join();}
    void solve(AvbdBoundedResponse& r)
    {
        while(r.status==AvbdBoundedResponse::Running)
        {
            // One bounded batch has no parallel work to distribute. This is
            // the same kernel and cap, not a different numerical executor.
            if(r.count()<=AvbdBoundedResponse::BatchWidth)
            {AvbdBoundedBatchArgs a={&r,0,r.count(),false};table.launch(AVBD_BOUNDED_BATCH,&a);
                if(!a.executed){r.status=AvbdBoundedResponse::NumericalFailure;return;}r.advance();continue;}
            for(unsigned start=0;start<r.count();)
            {const AvbdCpuBatch pack(r.count()-start,4);
                std::unique_lock<std::mutex> lock(mutex);batch=pack;args={&r,start,start+pack.items,false};finished=0;++epoch;wake.notify_all();done.wait(lock,[&]{return finished==4;});
                if(rejected){r.status=AvbdBoundedResponse::NumericalFailure;return;}start+=pack.items;}
            r.advance();
        }
    }
};
#endif
