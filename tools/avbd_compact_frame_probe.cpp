// Independent coordinate potential/derivatives versus the production contraction.
#include "avbd/DyAvbdElasticFrame.h"
#include <random>
#include <cstdio>
#include <chrono>
#include <cstring>
using namespace physx;using namespace physx::Dy;
struct Sample
{
    PxConstraintElasticFrameData data;
    PxTransformT<double> bodies[2],local[2],rest[2];
    AvbdElasticFrameReference reference;
    double dh[6],initial[18];
};
static AvbdElasticFrameContribution coordinate(const Sample& s,const PxTransformT<double>* bodies)
{
    AvbdElasticFrameContribution out={};double w[18],bias[18];
    const auto a=bodies[0]*s.local[0],b=bodies[1]*s.local[1];
    const auto c=finiteJointCoordinates(a,b,bodies[0].p,bodies[1].p);
    for(PxU32 r=0;r<18;++r)
    {
        const PxU32 i=finiteJointDriveIndex(r);const double k=s.data.stiffness[i],dh=s.dh[i];
        const double C=c.value[r],delta=C-s.initial[r],f=k*C+dh*delta;
        w[r]=k+dh;bias[r]=dh*s.initial[r];out.energy+=.5*(k*C*C+dh*delta*delta);
        for(PxU32 j=0;j<3;++j)
        {out.gradient[j]+=f*c.linear[r][j];out.gradient[6+j]-=f*c.linear[r][j];
            out.gradient[3+j]+=f*c.angularA[r][j];out.gradient[9+j]-=f*c.angularB[r][j];}
    }
    finiteJointMaterialHessian(a,b,bodies[0].p,bodies[1].p,w,bias,out.hessian,&c,out.gaussNewton);
    out.valid=true;return out;
}
static AvbdElasticFrameContribution compact(const Sample& s,const PxTransformT<double>* bodies,PxU32 body=2)
{
    AvbdElasticFrameContribution out;
    const PxTransformT<double> frames[2]={bodies[0]*s.local[0],bodies[1]*s.local[1]};
    out.energy=avbdElasticFramePotential(s.data.stiffness,s.dh,s.reference,bodies,frames,out.gradient,out.hessian,out.gaussNewton,body);
    return out;
}
static double energy(const Sample& s,const PxTransformT<double>* bodies)
{
    const auto a=bodies[0]*s.local[0],b=bodies[1]*s.local[1];
    const auto c=finiteJointCoordinates<double,false>(a,b,bodies[0].p,bodies[1].p);
    double e=0;for(PxU32 r=0;r<18;++r)
    {const auto i=finiteJointDriveIndex(r);const double delta=c.value[r]-s.initial[r];
        e+=.5*(s.data.stiffness[i]*c.value[r]*c.value[r]+s.dh[i]*delta*delta);}
    return e;
}
static double error(const double* a,const double* b,PxU32 n)
{
    double d=0,scale=1;for(PxU32 i=0;i<n;++i){d+=(a[i]-b[i])*(a[i]-b[i]);scale+=b[i]*b[i];}
    return std::sqrt(d/scale);
}
static void move(PxTransformT<double>* x,PxU32 d,double step)
{
    const PxU32 b=d/6,axis=d%3;
    if(d%6<3)x[b].p[axis]+=step;
    else {PxVec3d a(0);a[axis]=1;x[b].q=(PxQuatT<double>(step,a)*x[b].q).getNormalized();}
}
int main(int argc,char** argv)
{
    std::mt19937 random(20261113);std::normal_distribution<double> normal;
    auto vector=[&](){return PxVec3d(normal(random),normal(random),normal(random));};
    auto rotation=[&](){return PxQuatT<double>(normal(random),normal(random),normal(random),normal(random)).getNormalized();};
    Sample samples[240];double maxEnergy=0,maxGradient=0,maxH=0,maxGN=0,maxFD=0,maxFDH=0,maxMotion=0;
    unsigned failures=0;double restingTorque=0;
    for(PxU32 n=0;n<240;++n)
    {
        auto& s=samples[n];for(PxU32 b=0;b<2;++b)
        {s.bodies[b]=PxTransformT<double>(vector(),rotation());s.rest[b]=PxTransformT<double>(vector(),rotation());
            s.local[b]=PxTransformT<double>(vector()*.2,rotation());}
        if(n%4==0)for(PxU32 b=0;b<2;++b)s.bodies[b]=s.rest[b];
        if(n%4==1)
        {s.bodies[0]=s.rest[0];s.bodies[1]=s.rest[1];move(s.bodies,9,1.e-8);}
        if(n%4==2)
        {const auto qa=(s.bodies[0]*s.local[0]).q;s.bodies[1].q=qa*PxQuatT<double>(3.141592653589793-1.e-9,vector().getNormalized())*s.local[1].q.getConjugate();}
        for(PxU32 d=0;d<6;++d)
        {s.data.stiffness[d]=float(std::pow(10.,int(random()%8)-3));s.dh[d]=n%7==0?0:std::pow(10.,int(random()%7)-3);}
        const auto ra=s.rest[0]*s.local[0],rb=s.rest[1]*s.local[1];
        const PxTransformT<double> restFrames[2]={ra,rb};s.reference=avbdPrepareFrameReference(restFrames);
        const auto initial=finiteJointCoordinates<double,false>(ra,rb,s.rest[0].p,s.rest[1].p);
        for(PxU32 r=0;r<18;++r)s.initial[r]=initial.value[r];
        const auto expected=coordinate(s,s.bodies),actual=compact(s,s.bodies);
        maxEnergy=PxMax(maxEnergy,error(&actual.energy,&expected.energy,1));
        maxGradient=PxMax(maxGradient,error(actual.gradient,expected.gradient,12));
        maxH=PxMax(maxH,error(actual.hessian,expected.hessian,81));maxGN=PxMax(maxGN,error(actual.gaussNewton,expected.gaussNewton,81));
        for(PxU32 endpoint=0;endpoint<2;++endpoint)
        {
            const auto part=compact(s,s.bodies,endpoint);double h[81]={},m[81]={};
            for(PxU32 a=0;a<6;++a)for(PxU32 b=0;b<6;++b)
            {const auto j=9*finiteJointDerivativeIndex(a,endpoint)+finiteJointDerivativeIndex(b,endpoint);h[j]=expected.hessian[j];m[j]=expected.gaussNewton[j];}
            maxH=PxMax(maxH,error(part.hessian,h,81));maxGN=PxMax(maxGN,error(part.gaussNewton,m,81));
            const PxTransformT<double> frames[2]={s.bodies[0]*s.local[0],s.bodies[1]*s.local[1]};
            double packedG[6],packedH[21],packedM[21],expectedH[21],expectedM[21];
            const double packedEnergy=endpoint?
                avbdElasticFramePotentialImpl<1,true>(s.data.stiffness,s.dh,s.reference,s.bodies,frames,packedG,packedH,packedM):
                avbdElasticFramePotentialImpl<0,true>(s.data.stiffness,s.dh,s.reference,s.bodies,frames,packedG,packedH,packedM);
            for(PxU32 a=0;a<6;++a)for(PxU32 b=a;b<6;++b)
            {
                const PxU32 index=9*finiteJointDerivativeIndex(a,endpoint)+finiteJointDerivativeIndex(b,endpoint);
                const double sign=(endpoint && a<3?-1:1)*(endpoint && b<3?-1:1);
                expectedH[avbdFramePackedIndex(a,b)]=sign*expected.hessian[index];
                expectedM[avbdFramePackedIndex(a,b)]=sign*expected.gaussNewton[index];
            }
            maxEnergy=PxMax(maxEnergy,error(&packedEnergy,&expected.energy,1));
            maxGradient=PxMax(maxGradient,error(packedG,expected.gradient+6*endpoint,6));
            maxH=PxMax(maxH,error(packedH,expectedH,21));maxGN=PxMax(maxGN,error(packedM,expectedM,21));
        }
        const PxTransformT<double> transform(vector(),rotation());
        const PxTransformT<double> moved[2]={transform*s.bodies[0],transform*s.bodies[1]};
        const auto objective=compact(s,moved);maxMotion=PxMax(maxMotion,error(&objective.energy,&actual.energy,1));
        for(PxU32 b=0;b<2;++b)
        {PxTransformT<double> flipped[2]={s.bodies[0],s.bodies[1]};flipped[b].q=-flipped[b].q;
            const auto e=compact(s,flipped);maxMotion=PxMax(maxMotion,error(e.gradient,actual.gradient,12));}
        if(n<80)
        {
            double numeric[12],H[144],g[2][12];const double step=2.e-4;
            for(PxU32 j=0;j<12;++j)
            {
                double e[4];for(PxU32 k=0;k<4;++k)
                {PxTransformT<double> x[2]={s.bodies[0],s.bodies[1]};move(x,j,(k<2?int(k)-2:int(k)-1)*step);e[k]=energy(s,x);}
                numeric[j]=(e[0]-8*e[1]+8*e[2]-e[3])/(12*step);
                for(PxU32 k=0;k<2;++k)
                {PxTransformT<double> x[2]={s.bodies[0],s.bodies[1]};move(x,j,k?1.e-5:-1.e-5);
                    const auto e=compact(s,x);for(PxU32 i=0;i<12;++i)g[k][i]=e.gradient[i];}
                for(PxU32 i=0;i<12;++i)H[12*i+j]=(g[1][i]-g[0][i])/2.e-5;
            }
            maxFD=PxMax(maxFD,error(numeric,actual.gradient,12));double full[144],sym[144];
            for(PxU32 i=0;i<12;++i)for(PxU32 j=0;j<12;++j)
            {
                const PxU32 a=i<6?i:i<9?i-6:i-3,b=j<6?j:j<9?j-6:j-3;
                full[12*i+j]=(i>=6 && i<9?-1:1)*(j>=6 && j<9?-1:1)*actual.hessian[9*a+b];
                sym[12*i+j]=.5*(H[12*i+j]+H[12*j+i]);
            }
            maxFDH=PxMax(maxFDH,error(sym,full,144));
        }
    }
    // A frozen high-damping reference must not generate a torque by
    // cancellation of large radial quaternion terms.
    for(PxU32 n=0;n<80;++n)
    {
        Sample s=samples[n];for(PxU32 d=0;d<6;++d){s.data.stiffness[d]=0;s.dh[d]=1.e18*(d+1);}
        const auto a=s.bodies[0]*s.local[0],b=s.bodies[1]*s.local[1];
        const PxTransformT<double> frames[2]={a,b};s.reference=avbdPrepareFrameReference(frames);
        const auto e=compact(s,s.bodies);
        for(PxU32 d=0;d<12;++d)restingTorque=PxMax(restingTorque,PxAbs(e.gradient[d]));
        for(PxU32 endpoint=0;endpoint<2;++endpoint)
        {
            double g[6],h[21],m[21];
            if(endpoint)avbdElasticFramePotentialImpl<1,true>(s.data.stiffness,s.dh,s.reference,s.bodies,frames,g,h,m);
            else avbdElasticFramePotentialImpl<0,true>(s.data.stiffness,s.dh,s.reference,s.bodies,frames,g,h,m);
            for(double f:g)restingTorque=PxMax(restingTorque,PxAbs(f));
        }
        failures+=!std::isfinite(e.energy) || e.energy>1.e-12;
    }
    failures+=restingTorque>1.e-8;
    failures+=maxEnergy>1.e-10 || maxGradient>1.e-10 || maxH>1.e-10 || maxGN>1.e-10 || maxMotion>1.e-10;
    failures+=maxFD>1.e-7 || maxFDH>1.e-7;
    std::printf("[COMPACT_FRAME] cases=240 fdCases=80 energy=%.9g gradient=%.9g K=%.9g GN=%.9g motion=%.9g fdGradient=%.9g fdHessian=%.9g failures=%u\n",
        maxEnergy,maxGradient,maxH,maxGN,maxMotion,maxFD,maxFDH,failures);
    std::printf("[COMPACT_LAYOUT] instance=%zu geometry=%zu parameters=%zu energy=%zu contribution=%zu reference=%zu\n",
        sizeof(AvbdElasticFrameInstance),sizeof(AvbdElasticFrameGeometry),sizeof(AvbdElasticFrameParameters),
        sizeof(AvbdElasticFrameEnergy),sizeof(AvbdElasticFrameContribution),sizeof(AvbdElasticFrameEvaluation));
    std::printf("[COMPACT_FRAME_REST] cases=80 damping=1e18 maxForce=%.9g\n",restingTorque);
    std::printf("[COMPACT_ENDPOINT] bytes=%zu\n",sizeof(AvbdElasticFrameBlockContribution));
    if(argc>1 && std::strcmp(argv[1],"--benchmark")==0)
    {
        volatile double checksum=0;
        for(PxU32 repeat=0;repeat<3;++repeat)for(PxU32 mode=0;mode<2;++mode)
        {
            const auto start=std::chrono::steady_clock::now();
            for(PxU32 k=0;k<100;++k)for(const auto& s:samples)
            {const auto e=mode?compact(s,s.bodies):coordinate(s,s.bodies);checksum+=e.energy+e.gradient[4]+e.hessian[40]+e.gaussNewton[40];}
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            std::printf("[COMPACT_KERNEL_COST] repeat=%u mode=%s evaluations=24000 ms=%.6f checksum=%.9g\n",repeat,mode?"compact":"coordinate",ms,double(checksum));
        }
    }
    return failures?1:0;
}
