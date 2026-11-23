// Execution-graph contracts; no copied material or contact solver.
#include "avbd/DyAvbdProgramSchedule.h"
#include "avbd_probe_foundation.h"
#include <cstdio>
#include <vector>
using namespace physx;using namespace physx::Dy;
struct Row {std::vector<unsigned> blocks;bool energy,active;};
int main()
{
    AvbdProbeFoundation foundation;unsigned failed=0,cases=0,maximum=0;
    for(unsigned trial=0;trial<32;++trial)
    {
        std::vector<Row> rows;
        // Five-body closed chain must NOT be cut into four-block packets.
        for(unsigned b=0;b<4;++b)rows.push_back({{b,b+1},false,true});
        // Finite four-support material; zero geometric gradients do not
        // remove these declared read dependencies from the graph.
        for(unsigned b=0;b<20;++b)rows.push_back({{b,(b+3)%24,(b+7+(trial>>2))%24,(b+11+2*(trial>>2))%24},true,true});
        // Read-only supports must not connect every dynamic body.
        for(unsigned b=5;b<24;++b)rows.push_back({{b,24,25},false,true});
        rows.push_back({{0,23},false,false});
        if(trial&1)std::reverse(rows.begin(),rows.end());
        if(trial&2)for(auto& row:rows)std::reverse(row.blocks.begin(),row.blocks.end());
        AvbdRowProgram program;
        if(!program.compile(unsigned(rows.size()),26,[&](unsigned r,const auto& emit){for(unsigned b:rows[r].blocks)emit(b);return true;}))return 2;
        AvbdProgramSchedule schedule;
        schedule.compile(program,[](unsigned b){return b<24;},[&](unsigned r){return rows[r].active;},[&](unsigned r){return rows[r].energy;});
        std::vector<unsigned> group(26,PX_MAX_U32),color(schedule.groups.size(),PX_MAX_U32),writes(rows.size(),0);
        for(unsigned g=0;g<schedule.groups.size();++g)
        {
            for(unsigned b:schedule.groups[g].blocks){failed+=b>=24 || group[b]!=PX_MAX_U32;group[b]=g;}
            for(unsigned r:schedule.groups[g].rows)
            {
                failed+=!rows[r].active;bool incident=false;
                for(unsigned b:rows[r].blocks)for(unsigned member:schedule.groups[g].blocks)incident=incident || b==member;
                failed+=!incident;if(!rows[r].energy)++writes[r];
            }
        }
        for(unsigned c=0;c+1<schedule.colorOffsets.size();++c)
            for(unsigned s=schedule.colorOffsets[c];s<schedule.colorOffsets[c+1];++s)
            {const unsigned g=schedule.colorGroups[s];failed+=color[g]!=PX_MAX_U32;color[g]=c;}
        for(unsigned b=0;b<24;++b)failed+=group[b]==PX_MAX_U32;
        for(unsigned b=1;b<5;++b)failed+=group[b]!=group[0];
        for(unsigned r=0;r<rows.size();++r)if(rows[r].active)
        {
            if(!rows[r].energy)failed+=writes[r]!=1;
            for(unsigned b:rows[r].blocks)if(b<24)
            {
                const auto& subset=schedule.groups[group[b]];bool found=false;
                for(unsigned row:subset.rows)found=found || row==r;failed+=!found;
                for(unsigned other:rows[r].blocks)if(other<24 && group[b]!=group[other])
                    failed+=color[group[b]]==color[group[other]];
            }
        }
        failed+=schedule.groups.size()<2;maximum=std::max(maximum,schedule.maximumBlocks);++cases;
        // Also exercise the CSR incident collector used by the full-component
        // oracle. It must be complete and unique even for duplicate supports.
        AvbdProgramSubset subset;subset.blocks.pushBack(0);subset.blocks.pushBack(1);subset.collectRows(program);
        for(unsigned r=0;r<rows.size();++r)
        {
            bool incident=false;for(unsigned b:rows[r].blocks)incident=incident || b<2;
            unsigned count=0;for(unsigned stored:subset.rows)count+=stored==r;failed+=count!=unsigned(incident);
        }
    }
    std::printf("[PROGRAM_SCHEDULE] cases=%u hardChainPreserved=1 uniqueDual=1 completeBoundary=1 conflictColors=1 maxBlocks=%u failed=%u status=%s\n",
        cases,maximum,failed,failed?"FAIL":"PASS");return failed?1:0;
}
