// Shared numerical kernel replay; no scene or material implementation.
#include "avbd/DyAvbdLocalResponse.h"
#include "avbd/DyAvbdCommonResponse.h"
#include "avbd/DyAvbdBoundedResponse.h"
#include "avbd_probe_foundation.h"
#include "avbd_bounded_probe_pool.h"
#include <cstdio>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <fstream>
#include <string>
using namespace physx;using namespace physx::Dy;
template<class T> static bool read(FILE* f,T* p,size_t n){return std::fread(p,sizeof(T),n,f)==n;}
static int runCase(int argc,char** argv)
{
    AvbdProbeFoundation foundation;if(argc<3)return 2;FILE* file=std::fopen(argv[1],"rb");if(!file)return 2;
    AvbdCommonProgram p;unsigned counts[4];if(!read(file,counts,4))return 2;
    p.dimensions.resize(counts[0]);if(!read(file,p.dimensions.begin(),counts[0]))return 2;
    unsigned n=0;for(unsigned d:p.dimensions)n+=d;
    p.sourceBlocks.resize(counts[0]);p.sourceRows.resize(counts[1]);
    p.H.resize(counts[2]);p.J.resize(counts[3]);p.base.resize(n);p.mu.resize(counts[1]);p.coulomb.resize(counts[1]);
    if(!read(file,p.base.begin(),n) || !read(file,p.mu.begin(),counts[1]) || !read(file,p.coulomb.begin(),counts[1]))return 2;
    for(auto a:{&p.offset,&p.lower,&p.upper,&p.compliance})
    {a->resize(3*counts[1]);if(!read(file,a->begin(),a->size()))return 2;}
    for(auto& h:p.H)if(!read(file,&h.row,1)||!read(file,&h.column,1)||!read(file,h.values,36))return 2;
    for(auto& j:p.J)if(!read(file,&j.row,1)||!read(file,&j.column,1)||!read(file,j.values,18))return 2;
    PxArray<double> initialForce(p.offset.size(),0),motionTolerance(p.base.size(),1.e-5);
    if(!read(file,initialForce.begin(),initialForce.size()) || !read(file,motionTolerance.begin(),motionTolerance.size()))return 2;
    // Optional trailer keeps historical packets readable while preserving the
    // distinct operator and preconditioning metric in new production captures.
    unsigned metricCount=0;
    if(read(file,&metricCount,1))
    {
        if(metricCount && metricCount!=counts[2])return 2;
        p.metricH.resize(metricCount);
        for(auto& h:p.metricH)if(!read(file,&h.row,1)||!read(file,&h.column,1)||!read(file,h.values,36))return 2;
    }
    else if(!std::feof(file))return 2;
    std::fclose(file);if(!p.finish())return 2;
    const auto table=KernelTable::makeAvbdCpuScalar();
    const bool strict=argc>3 && std::strcmp(argv[3],"strict")==0;
    const bool bounded=argc>3 && std::strcmp(argv[3],"bounded")==0;
    const bool engineering=argc>3 && (std::strcmp(argv[3],"engineering")==0 || std::strcmp(argv[3],"grouped-engineering")==0);
    const bool grouped=argc>3 && (std::strcmp(argv[3],"grouped")==0 || std::strcmp(argv[3],"grouped-engineering")==0);
    const unsigned repeats=argc>4?std::max(1,std::atoi(argv[4])):1;
    const unsigned threads=argc>5?std::atoi(argv[5]):1;
    std::unique_ptr<AvbdBoundedProbePool> pool;if(bounded && threads==4)pool.reset(new AvbdBoundedProbePool);
    PxArray<double> step,force,times;AvbdCoulombNcpResult result={};bool solved=true;unsigned colors=0,products=0;
    PxU64 scheduledWaves=0,scheduledBatches=0;unsigned fusedBlocks=0,exchangeBlocks=0,equilibriumSweeps=0,equilibriumEntries=0;
    const char* executionStatus="legacy";
    for(unsigned repeat=0;repeat<repeats;++repeat)
    {
        step.clear();force=initialForce;const auto begin=std::chrono::steady_clock::now();
        if(bounded)
        {
            AvbdBoundedResponse response(p,force.begin());
            if(pool){pool->solve(response);solved=response.status==AvbdBoundedResponse::Converged && solved;}
            else solved=response.solve(table) && solved;
            step=response.step;force=response.force;result.iterations=response.sweeps;result.residual=response.error;
            colors=response.colorCount;products=0;
            equilibriumSweeps=response.equilibriumSweeps;equilibriumEntries=response.equilibriumEntries;
            scheduledWaves=scheduledBatches=0;fusedBlocks=exchangeBlocks=0;
            for(unsigned b=0;b<p.dimensions.size();++b)fusedBlocks+=response.nodes[response.roots[b]].level==0;
            for(unsigned i=0;i<response.waves.size();++i)
            {
                const auto& w=response.waves[i];const PxU64 repeats=i<response.repeatWave?1:response.sweeps;
                scheduledWaves+=repeats;scheduledBatches+=repeats*((w.count+response.BatchWidth-1)/response.BatchWidth);
                if(w.kind==response.Exchange && w.check)exchangeBlocks+=w.count;
            }
            executionStatus=response.statusName();
            unsigned enabled=0;for(auto value:response.enabled)enabled+=value!=0;
            std::printf("[BOUNDED_DETAIL] activeRows=%u activeDefect=%.9g stationRelative=%.9g result=%s\n",
                enabled,response.chosenMetric.activeError,response.chosenMetric.station/response.chosenMetric.scale,executionStatus);
        }
        else
        {
            AvbdCommonResponse response(p,table,!strict);
            if(engineering)response.local->motionTolerance=motionTolerance;
            solved=response.solve(step,force,result,grouped) && solved;
            colors=response.local?response.local->colorStarts.size()-1:0;products=response.materialProducts;
        }
        times.pushBack(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());
    }
    PxArray<double> ordered=times;std::sort(ordered.begin(),ordered.end());const double ms=ordered[ordered.size()/2];
    file=std::fopen(argv[2],"wb");if(!file)return 2;
    std::fprintf(file,"{\"converged\":%s,\"executionStatus\":\"%s\",\"iterations\":%u,\"residual\":%.17g,\"milliseconds\":%.9g,\"colors\":%u,\"materialProducts\":%u,\"scheduledWaves\":%llu,\"scheduledBatches\":%llu,\"singleLeafBlocks\":%u,\"exchangeBlocks\":%u,\"equilibriumSweeps\":%u,\"equilibriumEntries\":%u,\"step\":[",
        solved?"true":"false",executionStatus,result.iterations,result.residual,ms,colors,products,
        static_cast<unsigned long long>(scheduledWaves),static_cast<unsigned long long>(scheduledBatches),fusedBlocks,exchangeBlocks,equilibriumSweeps,equilibriumEntries);
    for(unsigned i=0;i<step.size();++i)std::fprintf(file,"%s%.17g",i?",":"",step[i]);std::fprintf(file,"],\"force\":[");
    for(unsigned i=0;i<force.size();++i)std::fprintf(file,"%s%.17g",i?",":"",force[i]);std::fprintf(file,"],\"samplesMs\":[");
    for(unsigned i=0;i<times.size();++i)std::fprintf(file,"%s%.9g",i?",":"",times[i]);std::fprintf(file,"]}");std::fclose(file);
    std::printf("[LOCAL_RESPONSE] blocks=%u groups=%u colors=%u sweeps=%u residual=%.9g ms=%.6g status=%s\n",
        counts[0],counts[1],colors,result.iterations,result.residual,ms,solved?"PASS":"FAIL");
    return solved?0:1;
}
int main(int argc,char** argv)
{
    if(argc!=3 || std::strcmp(argv[1],"--batch")!=0)return runCase(argc,argv);
    std::ifstream input(argv[2]);if(!input)return 2;int failed=0;unsigned cases=0;
    for(;;)
    {
        std::string values[5];if(!std::getline(input,values[0]))break;
        for(unsigned i=1;i<5;++i)if(!std::getline(input,values[i]))return 2;
        char* args[6]={argv[0]};for(unsigned i=0;i<5;++i)args[i+1]=&values[i][0];
        failed+=runCase(6,args)!=0;++cases;
    }
    std::printf("[LOCAL_RESPONSE_BATCH] cases=%u failed=%d\n",cases,failed);return failed?1:0;
}
