#include "avbd/DyAvbdHingeBatch.h"
#include <chrono>
#include <cstdio>
#include <limits>
#include <algorithm>
#include <vector>
using namespace physx;
using namespace physx::Dy;

PX_NOINLINE static void evaluateHinge(AvbdHingeBatchArgs& a,bool simd)
{
    if(simd)AvbdHingeBatchKernel<AvbdNodalReal>::sample(a);
    else AvbdHingeBatchKernel<void>::sample(a);
}

static int replay(const char* path)
{
    struct Record {AvbdNodalShellRest rest[4];AvbdHingeBatchArgs args;};
    FILE* file=std::fopen(path,"rb");
    if(!file)return 2;
    std::vector<Record> records;
    PxU32 count;
    while(std::fread(&count,sizeof(count),1,file)==1)
    {
        if(!count || count>4){std::fclose(file);return 2;}
        records.emplace_back();auto& r=records.back();r.args.count=count;
        for(PxU32 i=0;i<count;++i)
            if(std::fread(&r.rest[i],sizeof(r.rest[i]),1,file)!=1 ||
               std::fread(r.args.positions[i],sizeof(r.args.positions[i]),1,file)!=1 ||
               std::fread(&r.args.bending[i],sizeof(r.args.bending[i]),1,file)!=1 ||
               std::fread(&r.args.corner[i],sizeof(r.args.corner[i]),1,file)!=1)
            {std::fclose(file);return 2;}
    }
    std::fclose(file);
    if(records.empty())return 2;
    PxU32 rows=0;
    for(auto& r:records)for(PxU32 i=0;i<r.args.count;++i){r.args.rest[i]=&r.rest[i];++rows;}
    AvbdNodalReal worst=0;PxU32 failures=0;
    for(auto& r:records)
    {
        evaluateHinge(r.args,false);const auto expected=r.args.output;
        evaluateHinge(r.args,true);const auto actual=r.args.output;
        failures+=expected.valid!=actual.valid;
        if(!expected.valid || !actual.valid)continue;
        AvbdNodalReal error=PxAbs(actual.energy-expected.energy)/(1+PxAbs(expected.energy));
        error=PxMax(error,(actual.gradient-expected.gradient).magnitude()/(1+expected.gradient.magnitude()));
        for(PxU32 j=0;j<3;++j)error=PxMax(error,(actual.metric[j]-expected.metric[j]).magnitude()/(1+expected.metric[j].magnitude()));
        worst=PxMax(worst,error);failures+=error>AvbdNodalReal(2.e-5);
    }
    if(failures){std::printf("[HINGE_REPLAY] status=FAIL failures=%u worst=%.9g\n",failures,double(worst));return 1;}
    double milliseconds[2][7];
    volatile double sink=0;
    for(PxU32 repeat=0;repeat<7;++repeat)for(PxU32 order=0;order<2;++order)
    {
        const PxU32 simd=(repeat+order)%2;
        const auto start=std::chrono::steady_clock::now();
        double sum=0;
        for(PxU32 loop=0;loop<64;++loop)for(auto& r:records)
        {
            evaluateHinge(r.args,simd!=0);
            const auto& v=r.args.output;
            sum+=v.energy+v.gradient.x+v.gradient.y+v.gradient.z+v.metric.column0.dot(AvbdNodalVec3(1))+v.metric.column1.dot(AvbdNodalVec3(1))+v.metric.column2.dot(AvbdNodalVec3(1))+v.areaRatio+v.valid;
        }
        milliseconds[simd][repeat]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        sink=sum;
    }
    std::sort(milliseconds[0],milliseconds[0]+7);std::sort(milliseconds[1],milliseconds[1]+7);
    std::printf("[HINGE_REPLAY] packets=%zu rows=%u occupancy=%.6f scalarMs=%.6f simdMs=%.6f ratio=%.6f checksum=%.9g worstRelative=%.9g status=PASS\n",
        records.size(),rows,double(rows)/(4*records.size()),milliseconds[0][3],milliseconds[1][3],
        milliseconds[1][3]/milliseconds[0][3],double(sink),double(worst));
    return 0;
}

int main(int argc,char** argv)
{
    if(argc==2)return replay(argv[1]);
    using R=AvbdNodalReal;using V=AvbdNodalVec3;
    AvbdNodalShellRest rest[4];AvbdHingeBatchArgs args;
    PxU32 seed=71371,failures=0,checks=0;R worst=0,worstDerivative=0;
    const auto random=[&](){seed=1664525u*seed+1013904223u;return R(seed&0xffffffu)/R(8388608)-1;};
    for(PxU32 trial=0;trial<2000;++trial)
    {
        args.count=1+trial%4;
        const R scale=trial%3==0?R(.1):trial%3==1?R(1):R(10);
        for(PxU32 i=0;i<args.count;++i)
        {
            rest[i].kind=AvbdNodalShellRest::Hinge;rest[i].length=scale;
            rest[i].restCos=PxCos(R(.2));rest[i].restSin=PxSin(R(.2));
            args.rest[i]=&rest[i];args.bending[i]=R(.6);
            args.corner[i]=trial%5==4?PX_MAX_U32:(trial+i)%4;
            args.positions[i][0]=V(-1,.2f,0)*scale;args.positions[i][1]=V(1,0,-.3f)*scale;
            args.positions[i][2]=V(0);args.positions[i][3]=V(0,0,1)*scale;
            for(auto& p:args.positions[i])p+=V(random(),random(),random())*(scale*R(.05));
        }
        for(PxU32 i=args.count;i<4;++i){args.rest[i]=NULL;args.positions[i][0]=V(std::numeric_limits<R>::quiet_NaN());}
        AvbdHingeBatchKernel<void>::sample(args);const auto scalar=args.output;
        AvbdHingeBatchKernel<R>::sample(args);const auto actual=args.output;
        R error=PxAbs(actual.energy-scalar.energy)/(1+PxAbs(scalar.energy));
        error=PxMax(error,(actual.gradient-scalar.gradient).magnitude()/(1+scalar.gradient.magnitude()));
        for(PxU32 c=0;c<3;++c)error=PxMax(error,(actual.metric[c]-scalar.metric[c]).magnitude()/(1+scalar.metric[c].magnitude()));
        worst=PxMax(worst,error);failures+=!actual.valid || !scalar.valid || error>R(2.e-5);
        if(trial%5!=4)
        {
            V original[4];for(PxU32 i=0;i<args.count;++i)original[i]=args.positions[i][args.corner[i]];
            const V direction=V(1,2,3).getNormalized();const R h=scale*R(.001);
            const auto shift=[&](R d){for(PxU32 i=0;i<args.count;++i)args.positions[i][args.corner[i]]=original[i]+direction*d;};
            shift(h);AvbdHingeBatchKernel<R>::sample(args);const R plus=args.output.energy;
            shift(-h);AvbdHingeBatchKernel<R>::sample(args);const R minus=args.output.energy;
            const R numerical=(plus-minus)/(2*h),analytical=scalar.gradient.dot(direction);
            const R defect=PxAbs(numerical-analytical)/(1+PxAbs(analytical));worstDerivative=PxMax(worstDerivative,defect);
            failures+=defect>R(.002);
        }
        checks+=args.count;
    }
    args.count=4;
    for(PxU32 i=0;i<4;++i)
    {
        rest[i]=rest[0];args.rest[i]=&rest[i];args.bending[i]=R(.6);args.corner[i]=i;
        args.positions[i][0]=V(0,1,0);args.positions[i][1]=V(0,-1,0);
        args.positions[i][2]=V(0);args.positions[i][3]=V(1,0,0);
    }
    AvbdHingeBatchKernel<R>::sample(args);failures+=!args.output.valid;
    args.positions[1][0].x=std::numeric_limits<R>::quiet_NaN();
    AvbdHingeBatchKernel<R>::sample(args);failures+=args.output.valid;
    args.positions[1][0]=V(0,1,0);args.positions[2][0]=args.positions[2][2];
    AvbdHingeBatchKernel<R>::sample(args);failures+=args.output.valid;
    args.count=0;AvbdHingeBatchKernel<R>::sample(args);failures+=!args.output.valid || args.output.energy!=0;
    std::printf("[HINGE_SIMD] checks=%u argsBytes=%zu worstRelative=%.9g worstDerivative=%.9g failures=%u status=%s\n",
        checks,sizeof(args),double(worst),double(worstDerivative),failures,failures?"FAIL":"PASS");
    return failures?1:0;
}
