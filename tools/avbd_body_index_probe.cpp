// Host-identity packing must survive permutation, nulls, aliases and removal.
#include "avbd/DyAvbdInternal.h"
#include "avbd_probe_foundation.h"
#include <vector>
#include <chrono>
#include <cstdio>
using namespace physx;using namespace physx::Dy;
static volatile PxU64 checksum=0;
static PxU32 linear(const PxArray<AvbdSolverBody>& bodies,const PxsBodyCore* key)
{if(key)for(PxU32 i=0;i<bodies.size();++i)if(bodies[i].core==key)return i;return PX_MAX_U32;}
int main(int argc,char**)
{
    AvbdProbeFoundation foundation;unsigned failures=0,checks=0;
    std::vector<PxU64> identities(8192); // Opaque addresses, never dereferenced as bodies.
    PxArray<AvbdSolverBody> bodies;AvbdBodyIndex index;
    for(unsigned count:{1u,17u,126u,788u,4096u})
    {
        bodies.resize(count);
        for(unsigned epoch=0;epoch<4;++epoch)
        {
            for(unsigned i=0;i<count;++i)bodies[i].core=reinterpret_cast<PxsBodyCore*>(&identities[(i*13+epoch*791)%identities.size()]);
            bodies[0].core=NULL;if(count>2)bodies[count-1].core=bodies[1].core;
            index.build(bodies);
            for(const auto& identity:identities)
            {const auto* key=reinterpret_cast<const PxsBodyCore*>(&identity);failures+=index.find(key)!=linear(bodies,key);++checks;}
            failures+=index.find(NULL)!=PX_MAX_U32;
        }
        if(argc>1 && count>=126)
        {
            for(unsigned repeat=0;repeat<3;++repeat)for(unsigned variant=0;variant<2;++variant)
            {
                const auto started=std::chrono::steady_clock::now();
                // Include lookup construction in every indexed packing call.
                for(unsigned frame=0;frame<32;++frame)
                {
                    if(variant)index.build(bodies);
                    for(unsigned q=0;q<12*count;++q)
                    {const auto* key=bodies[(q*17)%count].core;checksum+=variant?index.find(key):linear(bodies,key);}
                }
                const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
                std::printf("[BODY_LOOKUP] bodies=%u queriesPerFrame=%u frames=32 indexed=%u repeat=%u ms=%.6f\n",count,12*count,variant,repeat,ms);
            }
        }
    }
    std::printf("[BODY_INDEX] checks=%u failures=%u bodyBytes=%zu checksum=%llu\n",checks,failures,sizeof(AvbdSolverBody),static_cast<unsigned long long>(checksum));
    return failures?1:0;
}
