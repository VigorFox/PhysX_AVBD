// Shared cooked payload lifecycle and concurrent immutable access.
#include "avbd/DyAvbdElasticFrame.h"
#include "avbd_probe_foundation.h"
#include <vector>
#include <thread>
#include <limits>
#include <cstdio>
using namespace physx;using namespace physx::Dy;
struct Input
{
    PxConstraintElasticFrameData data;
    PxTransform poses[2],trial[2];
    AvbdElasticFramePrepared expected;
};
static double relative(const double* a,const double* b,PxU32 count)
{
    double error=0,scale=1;for(PxU32 i=0;i<count;++i){error+=(a[i]-b[i])*(a[i]-b[i]);scale+=b[i]*b[i];}
    return std::sqrt(error/scale);
}
int main()
{
    AvbdProbeFoundation foundation;AvbdElasticFrameMaterials pool;
    std::vector<Input> inputs(2048);unsigned failures=0;double worst=0;
    failures+=pool.bytes()!=0 || pool.internCapacityEstimate()!=0;
    for(PxU32 epoch=0;epoch<3;++epoch)
    {
        const double dt=epoch==1?.01:.02;pool.beginStep(dt);
        failures+=!pool.empty() || pool.geometryCount()!=0 || pool.parameterCount()!=0;
        for(PxU32 i=0;i<inputs.size();++i)
        {
            auto& input=inputs[i];auto& data=input.data;
            const PxU32 family=i%3,material=i%7;
            data.frames[0]=PxTransform(PxVec3(.1f*(family+1),-.2f,.3f),PxQuat(.1f*family,PxVec3(0,1,0)));
            data.frames[1]=PxTransform(PxVec3(-.1f,.1f,.04f*family));
            if(epoch==2)data.frames[0].p.x+=.03f; // COM/local-frame edit.
            for(PxU32 k=0;k<6;++k)
            {data.stiffness[k]=float((material+1)*(k+1)*50);data.damping[k]=float((material+1)*(k+1))*.01f;}
            input.poses[0]=PxTransform(PxVec3(float(i%13),.2f,1),PxQuat(.001f*i,PxVec3(1,0,0)));
            input.poses[1]=PxTransform(PxVec3(float(i%13)+.3f,.1f,.9f),PxQuat(-.002f*i,PxVec3(0,0,1)));
            input.trial[0]=input.poses[0];input.trial[1]=input.poses[1];input.trial[1].p.y+=.05f;
            input.trial[0].q=PxQuat(.05f,PxVec3(0,1,0))*input.trial[0].q;
            failures+=!avbdPrepareElasticFrame(input.expected,data,input.poses,dt);
            const PxU32 id=pool.append(data,input.poses);failures+=id!=i;
        }
        failures+=pool.geometryCount()!=3 || pool.parameterCount()!=7 || pool.size()!=inputs.size();
        // Views are acquired after all reallocating append work has finished.
        // Every instance has its own frozen reference, even with shared rest.
        failures+=&pool[0].reference==&pool[21].reference;
        std::vector<double> errors(inputs.size());std::vector<unsigned> invalid(inputs.size());
        std::vector<std::thread> threads;
        for(PxU32 worker=0;worker<4;++worker)threads.emplace_back([&,worker]()
        {
            for(PxU32 i=worker;i<inputs.size();i+=4)
            {
                const auto& in=inputs[i];const auto expected=avbdEvaluateElasticFrameContribution(in.expected,in.trial);
                const auto actual=avbdEvaluateElasticFrameContribution(pool[i],in.trial);
                invalid[i]=!actual.valid || !expected.valid;
                errors[i]=PxMax(relative(&actual.energy,&expected.energy,1),PxMax(relative(actual.gradient,expected.gradient,12),
                    PxMax(relative(actual.hessian,expected.hessian,81),relative(actual.gaussNewton,expected.gaussNewton,81))));
            }
        });
        for(auto& thread:threads)thread.join();
        for(PxU32 i=0;i<inputs.size();++i){worst=PxMax(worst,errors[i]);failures+=invalid[i] || errors[i]>1.e-11;}
        const auto original=avbdEvaluateElasticFrameEnergy(pool[0],inputs[0].trial);
        auto edit=inputs[0].data;edit.stiffness[0]=std::nextafter(edit.stiffness[0],std::numeric_limits<float>::infinity());
        const auto editedId=pool.append(edit,inputs[0].poses);
        failures+=editedId==PX_MAX_U32 || pool.parameterCount()!=8 || pool.geometryCount()!=3;
        const auto retained=avbdEvaluateElasticFrameEnergy(pool[0],inputs[0].trial);
        failures+=relative(&original.energy,&retained.energy,1)>1.e-14;
        edit=inputs[0].data;edit.frames[1].p.z=std::nextafter(edit.frames[1].p.z,1.f);
        failures+=pool.append(edit,inputs[0].poses)==PX_MAX_U32 || pool.geometryCount()!=4;
        const PxU32 before=pool.size();edit.damping[0]=-1;
        failures+=pool.append(edit,inputs[0].poses)!=PX_MAX_U32 || pool.size()!=before;
        edit=inputs[0].data;edit.stiffness[0]=std::numeric_limits<float>::quiet_NaN();
        failures+=pool.append(edit,inputs[0].poses)!=PX_MAX_U32 || pool.size()!=before;
        std::printf("[FRAME_STORAGE] epoch=%u instances=%u geometries=%u parameters=%u bytes=%llu capacity=%llu internEstimate=%llu\n",
            epoch,pool.size(),pool.geometryCount(),pool.parameterCount(),static_cast<unsigned long long>(pool.bytes()),
            static_cast<unsigned long long>(pool.capacityBytes()),static_cast<unsigned long long>(pool.internCapacityEstimate()));
    }
    // A fully heterogeneous asset has a different memory tradeoff. Do not
    // hide it behind the regular-grid sharing statistics.
    pool.beginStep(.02);
    for(PxU32 i=0;i<inputs.size();++i)
    {
        auto data=inputs[i].data;data.frames[0].p.z=float(i)*.001f;data.stiffness[0]=float(i+1);
        failures+=pool.append(data,inputs[i].poses)==PX_MAX_U32;
    }
    failures+=pool.geometryCount()!=inputs.size() || pool.parameterCount()!=inputs.size();
    std::printf("[FRAME_STORAGE_UNIQUE] instances=%u bytes=%llu internEstimate=%llu oldMaterialBytes=%llu\n",pool.size(),
        static_cast<unsigned long long>(pool.bytes()),static_cast<unsigned long long>(pool.internCapacityEstimate()),
        static_cast<unsigned long long>(pool.size())*232);
    // Reordering/rebuilding a new epoch must bind to its new inputs, not old
    // asset positions or references. Invalid dt never creates a partial row.
    pool.beginStep(.04);
    for(PxU32 k=0;k<32;++k)
    {
        const auto& input=inputs[31-k];AvbdElasticFramePrepared expected;
        avbdPrepareElasticFrame(expected,input.data,input.poses,.04);const auto id=pool.append(input.data,input.poses);
        const auto a=avbdEvaluateElasticFrameContribution(pool[id],input.trial),b=avbdEvaluateElasticFrameContribution(expected,input.trial);
        failures+=relative(a.gradient,b.gradient,12)>1.e-11;
    }
    pool.beginStep(0);failures+=pool.append(inputs[0].data,inputs[0].poses)!=PX_MAX_U32 || !pool.empty();
    std::printf("[FRAME_STORAGE_RESULT] comparisons=6144 workers=4 worstRelativeError=%.9g failures=%u\n",worst,failures);
    return failures?1:0;
}
