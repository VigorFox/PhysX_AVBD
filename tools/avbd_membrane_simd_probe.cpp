#include "avbd/DyAvbdMembraneBatch.h"
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <vector>
#include <limits>
using namespace physx;using namespace physx::Dy;

PX_NOINLINE static void evaluateMembrane(AvbdMembraneBatchArgs& a,bool simd)
{
    if(simd)AvbdMembraneBatchKernel<AvbdNodalReal>::sample(a);
    else AvbdMembraneBatchKernel<void>::sample(a);
}

static int replay(const char* path)
{
    struct Record {AvbdNodalShellRest rest[4];AvbdMembraneBatchArgs args;};
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
               std::fread(&r.args.mu[i],sizeof(r.args.mu[i]),1,file)!=1 ||
               std::fread(&r.args.lambda[i],sizeof(r.args.lambda[i]),1,file)!=1 ||
               std::fread(&r.args.corner[i],sizeof(r.args.corner[i]),1,file)!=1)
            {std::fclose(file);return 2;}
    }
    std::fclose(file);
    if(records.empty())return 2;
    PxU32 rows=0;
    for(auto& r:records)for(PxU32 i=0;i<r.args.count;++i){r.args.rest[i]=&r.rest[i];++rows;}
    AvbdNodalReal worst=0,worstCancellation=0,worstCubicAbsolute=0;PxU32 failures=0;
    for(auto& r:records)
    {
        evaluateMembrane(r.args,false);const auto expected=r.args.output;
        evaluateMembrane(r.args,true);const auto actual=r.args.output;
        failures+=expected.valid!=actual.valid;
        if(!expected.valid || !actual.valid)continue;
        AvbdNodalReal error=PxAbs(actual.energy-expected.energy)/(1+PxAbs(expected.energy));
        error=PxMax(error,(actual.gradient-expected.gradient).magnitude()/(1+expected.gradient.magnitude()));
        for(PxU32 j=0;j<3;++j)error=PxMax(error,(actual.metric[j]-expected.metric[j]).magnitude()/(1+expected.metric[j].magnitude()));
        for(PxU32 j=0;j<3;++j)error=PxMax(error,(actual.polynomialMetric[j]-expected.polynomialMetric[j]).magnitude()/(1+expected.polynomialMetric[j].magnitude()));
        AvbdNodalReal cubicScale=0;
        for(PxU32 i=0;i<r.args.count;++i)
        {
            AvbdNodalWorkT<AvbdNodalReal> work;
            avbdSampleNodalShell(*r.args.rest[i],r.args.positions[i],r.args.mu[i],r.args.lambda[i],AvbdNodalReal(0),r.args.corner[i],static_cast<const AvbdNodalReal*>(NULL),&work);
            cubicScale+=work.cubic.magnitude();
        }
        // Four large, opposing cubic terms may sum almost to zero. The
        // forward-error scale of their reduction is the sum of magnitudes.
        const auto cubicError=(actual.cubic-expected.cubic).magnitude();
        worstCubicAbsolute=PxMax(worstCubicAbsolute,cubicError);
        worstCancellation=PxMax(worstCancellation,cubicError/(1+expected.cubic.magnitude()));
        error=PxMax(error,cubicError/(1+cubicScale));
        error=PxMax(error,PxAbs(actual.quartic-expected.quartic)/(1+PxAbs(expected.quartic)));
        worst=PxMax(worst,error);failures+=error>AvbdNodalReal(2.e-5);
    }
    if(failures){std::printf("[MEMBRANE_REPLAY] status=FAIL failures=%u worst=%.9g\n",failures,double(worst));return 1;}
    double milliseconds[2][7];
    volatile double sink=0;
    for(PxU32 repeat=0;repeat<7;++repeat)for(PxU32 order=0;order<2;++order)
    {
        const PxU32 simd=(repeat+order)%2;
        const auto start=std::chrono::steady_clock::now();
        double sum=0;
        for(PxU32 loop=0;loop<64;++loop)for(auto& r:records)
        {
            evaluateMembrane(r.args,simd!=0);
            const auto& v=r.args.output;
            sum+=v.energy+v.gradient.x+v.gradient.y+v.gradient.z+v.metric.column0.dot(AvbdNodalVec3(1))+v.metric.column1.dot(AvbdNodalVec3(1))+v.metric.column2.dot(AvbdNodalVec3(1))+v.polynomialMetric.column0.dot(AvbdNodalVec3(1))+v.polynomialMetric.column1.dot(AvbdNodalVec3(1))+v.polynomialMetric.column2.dot(AvbdNodalVec3(1))+v.cubic.dot(AvbdNodalVec3(1))+v.quartic+v.valid;
        }
        milliseconds[simd][repeat]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        sink=sum;
    }
    std::sort(milliseconds[0],milliseconds[0]+7);std::sort(milliseconds[1],milliseconds[1]+7);
    std::printf("[MEMBRANE_REPLAY] packets=%zu rows=%u occupancy=%.6f scalarMs=%.6f simdMs=%.6f ratio=%.6f checksum=%.9g worstRelative=%.9g cancellationRelative=%.9g cubicAbsolute=%.9g status=PASS\n",
        records.size(),rows,double(rows)/(4*records.size()),milliseconds[0][3],milliseconds[1][3],
        milliseconds[1][3]/milliseconds[0][3],double(sink),double(worst),double(worstCancellation),double(worstCubicAbsolute));
    return 0;
}

int main(int argc,char** argv)
{
    if(argc==2)return replay(argv[1]);
    using R=AvbdNodalReal;using V=AvbdNodalVec3;
    AvbdNodalShellRest rest[4];AvbdMembraneBatchArgs args;
    PxU32 seed=43719,failures=0,checks=0;R worst=0,worstDerivative=0,worstPolynomial=0;
    const auto random=[&](){seed=1664525u*seed+1013904223u;return R(seed&0xffffffu)/R(8388608)-1;};
    for(PxU32 trial=0;trial<2000;++trial)
    {
        args.count=1+trial%4;
        const R scale=trial%3==0?R(.1):trial%3==1?R(1):R(10);
        for(PxU32 i=0;i<args.count;++i)
        {
            rest[i].kind=AvbdNodalShellRest::Membrane;rest[i].area=scale*scale/R(2);
            rest[i].shape[0]=V(-1,-1,0)/scale;rest[i].shape[1]=V(1,0,0)/scale;rest[i].shape[2]=V(0,1,0)/scale;
            args.rest[i]=&rest[i];args.mu[i]=R(7);args.lambda[i]=trial%4==0?R(-2):R(3);
            args.corner[i]=(trial+i)%3;
            args.positions[i][0]=V(R(.1),R(.2),R(.3))*scale;
            args.positions[i][1]=args.positions[i][0]+V(R(1.1),R(.02),R(.2))*scale;
            args.positions[i][2]=args.positions[i][0]+V(R(.1),R(.9),R(-.1))*scale;
            for(auto& x:args.positions[i])x+=V(random(),random(),random())*(scale*R(.08));
        }
        for(PxU32 i=args.count;i<4;++i)
        {args.rest[i]=NULL;args.positions[i][0]=V(std::numeric_limits<R>::quiet_NaN());}
        // The unspecialized implementation uses the production scalar rows.
        AvbdMembraneBatchKernel<void>::sample(args);const auto scalar=args.output;
        AvbdMembraneBatchKernel<R>::sample(args);const auto actual=args.output;
        R error=PxAbs(actual.energy-scalar.energy)/(1+PxAbs(scalar.energy));
        error=PxMax(error,(actual.gradient-scalar.gradient).magnitude()/(1+scalar.gradient.magnitude()));
        for(PxU32 j=0;j<3;++j)
        {
            error=PxMax(error,(actual.metric[j]-scalar.metric[j]).magnitude()/(1+scalar.metric[j].magnitude()));
            error=PxMax(error,(actual.polynomialMetric[j]-scalar.polynomialMetric[j]).magnitude()/(1+scalar.polynomialMetric[j].magnitude()));
        }
        error=PxMax(error,(actual.cubic-scalar.cubic).magnitude()/(1+scalar.cubic.magnitude()));
        error=PxMax(error,PxAbs(actual.quartic-scalar.quartic)/(1+PxAbs(scalar.quartic)));
        worst=PxMax(worst,error);failures+=!actual.valid || !scalar.valid || error>R(2.e-5);
        const V direction=V(1,2,3).getNormalized();const R h=scale*R(.001);
        V original[4];for(PxU32 i=0;i<args.count;++i)original[i]=args.positions[i][args.corner[i]];
        const auto shift=[&](const V& dx){for(PxU32 i=0;i<args.count;++i)args.positions[i][args.corner[i]]=original[i]+dx;};
        shift(direction*h);AvbdMembraneBatchKernel<R>::sample(args);const R plus=args.output.energy;
        shift(direction*(-h));AvbdMembraneBatchKernel<R>::sample(args);const R minus=args.output.energy;
        const R numerical=(plus-minus)/(2*h),analytical=scalar.gradient.dot(direction);
        const R defect=PxAbs(numerical-analytical)/(1+PxAbs(analytical));
        worstDerivative=PxMax(worstDerivative,defect);failures+=defect>R(.002);
        const V delta=direction*scale*R(.03);shift(delta);
        AvbdMembraneBatchKernel<R>::sample(args);const R end=args.output.energy;
        const R length2=delta.magnitudeSquared();
        const R predicted=actual.energy+actual.gradient.dot(delta)+R(.5)*delta.dot(actual.polynomialMetric*delta)+
            length2*(actual.cubic.dot(delta)+actual.quartic*length2);
        const R polynomial=PxAbs(predicted-end)/(1+PxAbs(end));
        worstPolynomial=PxMax(worstPolynomial,polynomial);failures+=polynomial>R(2.e-5);
        checks+=args.count;
    }
    args.count=4;
    for(PxU32 i=0;i<4;++i)
    {
        rest[i]=rest[0];args.rest[i]=&rest[i];args.mu[i]=R(7);args.lambda[i]=R(3);args.corner[i]=i%3;
        args.positions[i][0]=V(0);args.positions[i][1]=V(1,0,0);args.positions[i][2]=V(0,1,0);
    }
    AvbdMembraneBatchKernel<R>::sample(args);failures+=!args.output.valid;
    failures+=args.output.energy!=0 || args.output.gradient.magnitudeSquared()!=0;
    for(PxU32 i=0;i<4;++i)
    {
        args.positions[i][0]=V(3,-2,1);args.positions[i][1]=V(3,-1,1);args.positions[i][2]=V(3,-2,2);
    }
    AvbdMembraneBatchKernel<R>::sample(args);
    failures+=!args.output.valid || args.output.energy!=0 || args.output.gradient.magnitudeSquared()!=0;
    for(PxU32 i=0;i<4;++i)
    {args.positions[i][0]=V(0);args.positions[i][1]=V(1,0,0);args.positions[i][2]=V(0,1,0);}
    args.positions[2][2]=V(0,R(1.e-6),0);
    AvbdMembraneBatchKernel<R>::sample(args);const bool thin=args.output.valid;
    AvbdMembraneBatchKernel<void>::sample(args);failures+=!thin || thin!=args.output.valid;
    args.positions[2][2]=V(0,1,0);
    args.positions[1][0].x=std::numeric_limits<R>::quiet_NaN();
    AvbdMembraneBatchKernel<R>::sample(args);failures+=args.output.valid;
    args.positions[1][0]=V(0);args.positions[2][2]=V(0);
    AvbdMembraneBatchKernel<R>::sample(args);failures+=args.output.valid;
    args.positions[2][2]=V(0,1,0);args.mu[3]=0;
    AvbdMembraneBatchKernel<R>::sample(args);failures+=args.output.valid;
    args.count=0;AvbdMembraneBatchKernel<R>::sample(args);failures+=!args.output.valid || args.output.energy!=0;
    std::printf("[MEMBRANE_SIMD] checks=%u argsBytes=%zu worstRelative=%.9g worstDerivative=%.9g worstPolynomial=%.9g failures=%u status=%s\n",
        checks,sizeof(args),double(worst),double(worstDerivative),double(worstPolynomial),failures,failures?"FAIL":"PASS");
    return failures?1:0;
}
