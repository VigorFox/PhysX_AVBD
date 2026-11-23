// Packet assembly is checked against the coupled support-space contribution.
// The potential itself has an independent coordinate/finite-difference probe.
#include "avbd/DyAvbdFrameAssembly.h"
#include "avbd/DyAvbdNonlinearGroup.h"
#include "avbd_probe_foundation.h"
#include <thread>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
using namespace physx;using namespace physx::Dy;
static double error(const double* a,const double* b,PxU32 n)
{
    double difference=0,scale=1;
    for(PxU32 i=0;i<n;++i)
    {if(!std::isfinite(a[i]) || !std::isfinite(b[i]))return 1.e30;difference=PxMax(difference,PxAbs(a[i]-b[i]));scale=PxMax(scale,PxAbs(b[i]));}
    return difference/scale;
}
static void endpoints(PxU32 trial,PxU32 n,const PxU32* members,PxU32 count,PxU32 (&ends)[2])
{
    ends[0]=members[n%count];ends[1]=(n+trial)%3==0?23:members[(n+1)%count];
    if((n+trial)%5==0)ends[1]=ends[0];
    if((n+trial)%2)std::swap(ends[0],ends[1]);
}
static double check(PxU32 trial,const AvbdElasticFrameMaterials& materials,
    const KernelTable& table,const PxTransform* bodies)
{
    const PxU32 members[4]={2,7,11,19},count=1+trial%4,n=1+(trial/4)%23;
    AvbdNonlinearGroup actual,expected;actual.count=expected.count=count;
    double energy=0,expectedEnergy=0;
    AvbdFrameAssemblyTarget target={&actual.gradient[0][0],&actual.K[0][0][0],&actual.M[0][0][0],&energy,count,144};
    AvbdFrameAssemblyPacket packet(materials,table,members,target);
    for(PxU32 k=0;k<n;++k)
    {
        PxU32 ends[2];endpoints(trial,k,members,count,ends);
        const PxU32 index=(trial+k)%materials.size();
        const PxTransform poses[2]={bodies[ends[0]],bodies[ends[1]]};
        const auto e=avbdEvaluateElasticFrameContribution(materials[index],poses);
        if(!e.valid || !packet.append(index,ends[0],ends[1],bodies))return 1.e30;
        avbdAccumulateFrameGroup(e,ends,members,expected);expectedEnergy+=e.energy;
        if(k%7==3)
        {
            // A non-frame contribution separates two material runs.
            if(!packet.flush())return 1.e30;
            actual.K[0][0][0]+=3;expected.K[0][0][0]+=3;
            actual.gradient[0][0]+=.2;expected.gradient[0][0]+=.2;energy+=.75;expectedEnergy+=.75;
        }
    }
    if(!packet.flush())return 1.e30;
    return PxMax(error(&energy,&expectedEnergy,1),PxMax(error(&actual.gradient[0][0],&expected.gradient[0][0],24),
        PxMax(error(&actual.K[0][0][0],&expected.K[0][0][0],576),error(&actual.M[0][0][0],&expected.M[0][0][0],576))));
}
int main(int argc,char** argv)
{
    AvbdProbeFoundation foundation;const auto table=KernelTable::makeAvbdCpuScalar();
    AvbdElasticFrameMaterials materials;materials.beginStep(.01);
    PxTransform bodies[24];
    for(PxU32 i=0;i<24;++i)bodies[i]=PxTransform(PxVec3(.1f*i,.02f*i*i,-.03f*i),PxQuat(.13f*i,PxVec3(1,2,3).getNormalized()));
    for(PxU32 i=0;i<13;++i)
    {
        PxConstraintElasticFrameData data={};
        data.frames[0]=PxTransform(PxVec3(.1f,.2f,.01f*i),PxQuat(.06f*i,PxVec3(0,1,0)));
        data.frames[1]=PxTransform(PxVec3(-.2f,.1f,-.01f*i),PxQuat(-.03f*i,PxVec3(1,0,0)));
        for(PxU32 j=0;j<6;++j){data.stiffness[j]=float((i+j)%5?120*(1+j):0);data.damping[j]=float((i+j)%3);}
        const PxTransform poses[2]={bodies[i],bodies[i+1]};
        if(materials.append(data,poses)==PX_MAX_U32)return 2;
    }
    unsigned failures=0;double maxError=0;
    for(PxU32 trial=0;trial<368;++trial)maxError=PxMax(maxError,check(trial,materials,table,bodies));
    // Independent parallel destinations share only immutable material data.
    double parallel[4]={};std::thread workers[4];
    for(PxU32 w=0;w<4;++w)workers[w]=std::thread([&,w]()
    {for(PxU32 trial=w;trial<368;trial+=4)parallel[w]=PxMax(parallel[w],check(trial,materials,table,bodies));});
    for(auto& worker:workers)worker.join();for(double e:parallel)maxError=PxMax(maxError,e);

    // Exercise the complete wire-format bounds, including sixteen poses.
    AvbdNonlinearGroup block,expected;block.count=expected.count=4;double energy=0,expectedEnergy=0;
    AvbdFrameAssemblyArgs args;args.materials=&materials;args.count=8;args.poseCount=16;
    args.target={&block.gradient[0][0],&block.K[0][0][0],&block.M[0][0][0],&energy,4,144};
    for(PxU32 i=0;i<16;++i)args.poses[i]=avbdElasticFrameDouble(bodies[i]);
    const PxU32 members[4]={0,1,2,3};
    for(PxU32 n=0;n<8;++n)
    {
        args.items[n]={n,{PxU8(2*n),PxU8(2*n+1)},{PxU8(n%4),PxU8((n+(n%3?1:0))%4)}};
        const PxTransform poses[2]={bodies[2*n],bodies[2*n+1]};const auto e=avbdEvaluateElasticFrameContribution(materials[n],poses);
        const PxU32 ends[2]={args.items[n].members[0],args.items[n].members[1]};avbdAccumulateFrameGroup(e,ends,members,expected);expectedEnergy+=e.energy;
    }
    table.launch(AVBD_ELASTIC_FRAME_BATCH,&args);failures+=!args.valid;
    maxError=PxMax(maxError,error(&block.K[0][0][0],&expected.K[0][0][0],576));
    maxError=PxMax(maxError,error(&block.M[0][0][0],&expected.M[0][0][0],576));
    maxError=PxMax(maxError,error(&block.gradient[0][0],&expected.gradient[0][0],24));
    maxError=PxMax(maxError,error(&energy,&expectedEnergy,1));
    // Run lanes of ONE packet concurrently: private contribution slots, then
    // exclusive destination entries. This checks alias/cross-block ownership,
    // not equality between trajectories or any particular reduction order.
    AvbdNonlinearGroup laneResult;laneResult.count=4;double laneEnergy=0;
    auto laneArgs=args;laneArgs.target={&laneResult.gradient[0][0],&laneResult.K[0][0][0],
        &laneResult.M[0][0][0],&laneEnergy,4,144};
    AvbdFramePacketScratch scratch;failures+=!avbdPrepareFramePacket(laneArgs,scratch);
    bool valid[8]={};std::thread lanes[8];
    for(PxU32 lane=0;lane<8;++lane)lanes[lane]=std::thread([&,lane]()
        {valid[lane]=avbdEvaluateFramePacketItem(laneArgs,lane,scratch.values[lane]);});
    for(auto& lane:lanes)lane.join();for(bool v:valid)failures+=!v;
    for(PxU32 lane=0;lane<8;++lane)lanes[lane]=std::thread([&,lane]()
    {
        for(PxU32 entry=lane;entry<36*scratch.destinationCount;entry+=8)
            avbdReduceFramePacketEntry(laneArgs,scratch,entry/36,(entry%36)/6,entry%6);
    });
    for(auto& lane:lanes)lane.join();
    maxError=PxMax(maxError,error(&laneResult.K[0][0][0],&expected.K[0][0][0],576));
    maxError=PxMax(maxError,error(&laneResult.M[0][0][0],&expected.M[0][0][0],576));
    maxError=PxMax(maxError,error(&laneResult.gradient[0][0],&expected.gradient[0][0],24));
    AvbdNonlinearGroup rowResult;rowResult.count=4;
    auto rowArgs=laneArgs;rowArgs.target={&rowResult.gradient[0][0],&rowResult.K[0][0][0],
        &rowResult.M[0][0][0],&laneEnergy,4,144};
    for(PxU32 lane=0;lane<8;++lane)lanes[lane]=std::thread([&,lane]()
    {
        for(PxU32 row=lane;row<6*scratch.destinationCount;row+=8)
            avbdReduceFramePacketRow(rowArgs,scratch,row/6,row%6);
    });
    for(auto& lane:lanes)lane.join();
    maxError=PxMax(maxError,error(&rowResult.K[0][0][0],&expected.K[0][0][0],576));
    maxError=PxMax(maxError,error(&rowResult.M[0][0][0],&expected.M[0][0][0],576));
    maxError=PxMax(maxError,error(&rowResult.gradient[0][0],&expected.gradient[0][0],24));
    failures+=scratch.termCount>32 || scratch.destinationCount>32;
    PxU32 pairs=0;for(PxU32 d=0;d<scratch.destinationCount;++d)
        pairs+=scratch.destinations[d].a==scratch.destinations[d].b?21:36;
    std::printf("[FRAME_PACKET_LANES] lanes=8 scratchBytes=%zu terms=%u destinations=%u laneSlots=%u ownedPairs=%u\n",
        sizeof(scratch),scratch.termCount,scratch.destinationCount,36*scratch.destinationCount,pairs);
    for(PxU32 control=0;control<8;++control)
    {
        auto invalid=args;
        switch(control)
        {
        case 0:invalid.count=9;break;
        case 1:invalid.poseCount=17;break;
        case 2:invalid.items[0].material=materials.size();break;
        case 3:invalid.items[0].poses[1]=16;break;
        case 4:invalid.items[0].members[0]=4;break;
        case 5:invalid.items[0].members[0]=invalid.items[0].members[1]=255;break;
        case 6:invalid.target.matrixRowStride=36;break;
        case 7:invalid.poses[0].p.x=std::numeric_limits<double>::quiet_NaN();break;
        }
        table.launch(AVBD_ELASTIC_FRAME_BATCH,&invalid);failures+=invalid.valid;
    }
    // Single-block stride, including a shared endpoint and packet overflow.
    AvbdNonlinearBlock single,singleExpected;double singleEnergy=0,singleExpectedEnergy=0;const PxU32 member=2;
    AvbdFrameAssemblyTarget singleTarget={single.gradient,single.K,single.M,&singleEnergy,1,36};
    AvbdFrameAssemblyPacket singlePacket(materials,table,&member,singleTarget);
    for(PxU32 n=0;n<25;++n)
    {
        const PxU32 ends[2]={n%3?2u:23u,2};const PxTransform poses[2]={bodies[ends[0]],bodies[ends[1]]};
        const auto e=avbdEvaluateElasticFrameContribution(materials[n%13],poses);
        avbdAccumulateFrameBlock(e,ends,member,singleExpected);singleExpectedEnergy+=e.energy;
        failures+=!singlePacket.append(n%13,ends[0],ends[1],bodies);
    }
    failures+=!singlePacket.flush();maxError=PxMax(maxError,error(single.K,singleExpected.K,36));
    maxError=PxMax(maxError,error(single.M,singleExpected.M,36));maxError=PxMax(maxError,error(single.gradient,singleExpected.gradient,6));
    maxError=PxMax(maxError,error(&singleEnergy,&singleExpectedEnergy,1));
    for(PxU32 count:{5u,8u,16u})
    {
        AvbdNonlinearGroupT<16> wide,wideExpected;wide.count=wideExpected.count=count;
        double value=0,referenceValue=0;PxU32 members16[16];for(PxU32 i=0;i<16;++i)members16[i]=i;
        const AvbdFrameAssemblyTarget dst={&wide.gradient[0][0],&wide.K[0][0][0],&wide.M[0][0][0],&value,count,576};
        AvbdFrameAssemblyPacket packed(materials,table,members16,dst);
        for(PxU32 i=0;i<47;++i)
        {
            PxU32 ends[2];endpoints(count,i,members16,count,ends);
            const PxTransform poses[2]={bodies[ends[0]],bodies[ends[1]]};
            const auto e=avbdEvaluateElasticFrameContribution(materials[i%13],poses);
            failures+=!packed.append(i%13,ends[0],ends[1],bodies);
            AvbdNonlinearGroup pair;pair.count=ends[0]==ends[1]?1:2;
            avbdAccumulateFrameGroup(e,ends,ends,pair);referenceValue+=e.energy;
            for(PxU32 a=0;a<pair.count;++a)if(ends[a]<count)
            {
                for(PxU32 r=0;r<6;++r)wideExpected.gradient[ends[a]][r]+=pair.gradient[a][r];
                for(PxU32 b=0;b<pair.count;++b)if(ends[b]<count)for(PxU32 k=0;k<36;++k)
                {wideExpected.K[ends[a]][ends[b]][k]+=pair.K[a][b][k];wideExpected.M[ends[a]][ends[b]][k]+=pair.M[a][b][k];}
            }
        }
        failures+=!packed.flush();maxError=PxMax(maxError,error(&value,&referenceValue,1));
        maxError=PxMax(maxError,error(&wide.gradient[0][0],&wideExpected.gradient[0][0],96));
        maxError=PxMax(maxError,error(&wide.K[0][0][0],&wideExpected.K[0][0][0],9216));
        maxError=PxMax(maxError,error(&wide.M[0][0][0],&wideExpected.M[0][0][0],9216));
    }
    failures+=maxError>1.e-10;
    std::printf("[FRAME_PACKET] cases=368 parallelJobs=4 negativeControls=8 maxError=%.9g failures=%u packetBytes=%zu itemBytes=%zu\n",
        maxError,failures,sizeof(AvbdFrameAssemblyPacket),sizeof(AvbdFrameAssemblyItem));
    if(argc>1 && std::strcmp(argv[1],"--benchmark")==0)
    {
        volatile double checksum=0;
        for(PxU32 repeat=0;repeat<3;++repeat)for(PxU32 mode=0;mode<3;++mode)
        {
            const auto start=std::chrono::steady_clock::now();
            for(PxU32 k=0;k<2000;++k)
            {
                AvbdNonlinearBlock result;double value=0;
                AvbdFrameAssemblyTarget dst={result.gradient,result.K,result.M,&value,1,36};
                AvbdFrameAssemblyPacket packet(materials,table,&member,dst);
                for(PxU32 n=0;n<8;++n)
                {
                    const PxU32 ends[2]={2,3+n};const PxTransform poses[2]={bodies[2],bodies[3+n]};
                    if(mode==2)packet.append(n,ends[0],ends[1],bodies);
                    else
                    {
                        const auto e=avbdEvaluateElasticFrameContribution(materials[n],poses,0);
                        value+=mode==0?avbdEvaluateElasticFrameEnergy(materials[n],poses).energy:e.energy;
                        avbdAccumulateFrameBlock(e,ends,member,result);
                    }
                }
                if(mode==2)packet.flush();checksum+=value+result.gradient[4]+result.K[28]+result.M[28];
            }
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            std::printf("[FRAME_PACKET_COST] repeat=%u mode=%s evaluations=16000 ms=%.6f checksum=%.9g\n",
                repeat,mode==0?"legacy-energy-and-contribution":mode==1?"legacy-fused":"packed-packet",ms,double(checksum));
        }
    }
    return failures?1:0;
}
