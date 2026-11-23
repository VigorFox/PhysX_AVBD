// Analytic native material / PSD execution checks, no Scene or solver copy.
#include "avbd/DyAvbdElasticFrame.h"
#include <cstdio>
#include <random>
using namespace physx;
using namespace physx::Dy;

int main()
{
    std::mt19937 random(539);std::normal_distribution<double> normal;
    double worstRoot=0,worstWeakRoot=0,worstEnergy=0;PxU32 failures=0,checks=0;
    for(PxU32 trial=0;trial<80;++trial)
    {
        double Q[9][9]={};for(PxU32 i=0;i<9;++i) Q[i][i]=1;
        // Include separated translational/rotational scales over 16 orders.
        for(PxU32 turn=0;turn<100;++turn)
        {
            PxU32 a=random()%9,b=random()%9;
            if(a==b || (trial%2 && (a<3)!=(b<3))) continue;
            const double angle=normal(random),c=std::cos(angle),s=std::sin(angle);
            for(PxU32 k=0;k<9;++k)
            { const double x=Q[k][a],y=Q[k][b];Q[k][a]=c*x-s*y;Q[k][b]=s*x+c*y; }
        }
        double H[81]={},positive[9][9]={},root[9][12];
        for(PxU32 k=0;k<9;++k)
        {
            const double e=(k%3==0?-1:1)*std::pow(10.0,int(k)*2-8);
            for(PxU32 i=0;i<9;++i) for(PxU32 j=0;j<9;++j)
            { H[9*i+j]+=e*Q[i][k]*Q[j][k];positive[i][j]+=PxMax(0.0,e)*Q[i][k]*Q[j][k]; }
        }
        if(!avbdElasticFramePositiveRoot(H,root)) { ++failures;continue; }
        double error=0,scale=0;
        for(PxU32 i=0;i<9;++i) for(PxU32 j=0;j<9;++j)
        {
            double value=0;for(PxU32 k=0;k<9;++k) value+=root[k][i<6?i:i+3]*root[k][j<6?j:j+3];
            error+=std::pow(value-positive[i][j],2);scale+=positive[i][j]*positive[i][j];
        }
        const double relative=std::sqrt(error/scale);worstRoot=PxMax(worstRoot,relative);
        failures+=relative>1.e-12;++checks;
        if(trial%2)
        {
            double weakError=0,weakScale=0;
            for(PxU32 i=0;i<3;++i) for(PxU32 j=0;j<3;++j)
            {
                double value=0;for(PxU32 k=0;k<9;++k) value+=root[k][i]*root[k][j];
                weakError+=std::pow(value-positive[i][j],2);weakScale+=positive[i][j]*positive[i][j];
            }
            const double weak=std::sqrt(weakError/weakScale);worstWeakRoot=PxMax(worstWeakRoot,weak);
            failures+=weak>1.e-12;++checks;
        }
        for(PxU32 k=0;k<9;++k) for(PxU32 j=0;j<3;++j)
        { failures+=root[k][j]+root[k][6+j]!=0;++checks; }
    }
    for(PxU32 trial=0;trial<60;++trial)
    {
        PxConstraintElasticFrameData data;
        data.frames[0]=PxTransform(PxVec3(.1f,-.2f,.3f));
        data.frames[1]=PxTransform(PxVec3(-.2f,.1f,.05f));
        for(PxU32 i=0;i<6;++i) { data.stiffness[i]=PxReal(100+10*i*i);data.damping[i]=PxReal(.3+.1*i); }
        const PxTransform rest[2]={PxTransform(PxIdentity),PxTransform(PxVec3(.5f,0,0))};
        PxTransform poses[2]={rest[0],rest[1]};
        poses[0].q=PxQuat(PxReal(trial)*.061f,PxVec3(1,2,3).getNormalized());
        poses[1].q=PxQuat(PxReal(trial)*-.047f,PxVec3(3,2,1).getNormalized());
        AvbdElasticFramePrepared material;
        if(!avbdPrepareElasticFrame(material,data,rest,1.0/60)) { ++failures;continue; }
        const auto e=avbdEvaluateElasticFrame(material,poses,true);
        failures+=!e.valid;++checks;
        const auto light=avbdEvaluateElasticFrame(material,poses,true,false);
        failures+=!light.valid || light.energy!=e.energy;++checks;
        for(PxU32 d=0;d<12;++d){failures+=light.gradient[d]!=e.gradient[d];++checks;}
        // The translational restriction is exactly quadratic; sample actual
        // PxReal positions and use their true stored increments.
        const PxTransform original=poses[0];const double step=.0005;
        for(PxU32 axis=0;axis<3;++axis)
        {
            poses[0]=original;poses[0].p[axis]=PxReal(double(original.p[axis])+step);
            const double hi=poses[0].p[axis];const double plus=avbdEvaluateElasticFrame(material,poses,false).energy;
            poses[0]=original;poses[0].p[axis]=PxReal(double(original.p[axis])-step);
            const double lo=poses[0].p[axis];const double minus=avbdEvaluateElasticFrame(material,poses,false).energy;
            const double expected=e.gradient[axis],relative=PxAbs((plus-minus)/(hi-lo)-expected)/PxMax(1.0,PxAbs(expected));
            worstEnergy=PxMax(worstEnergy,relative);failures+=relative>1.e-8;++checks;
        }
    }
    std::printf("[ELASTIC_FRAME] checks=%u failures=%u rootRelativeError=%.12g weakBlockRelativeError=%.12g energyDerivativeRelativeError=%.12g\n",checks,failures,worstRoot,worstWeakRoot,worstEnergy);
    return failures?1:0;
}
