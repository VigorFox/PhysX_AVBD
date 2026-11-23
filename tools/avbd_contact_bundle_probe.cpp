// Mechanism qualification only. No production row pruning or new solver.
#include "avbd/DyAvbdNpDomain.h"
#include "avbd/DyAvbdCommonProgram.h"
#include "avbd_probe_foundation.h"
#include <random>
#include <vector>
#include <cstdio>
using namespace physx;using namespace physx::Dy;

int main()
{
    std::setvbuf(stdout,NULL,_IONBF,0);
    AvbdProbeFoundation foundation;std::mt19937 rng(91723);std::normal_distribution<double> rnd;
    unsigned failures=0,normalCases=0;double worstValue=0,worstWrench=0;
    for(unsigned test=0;test<24;++test)
    {
        AvbdNpDomain d;d.normal=PxVec3d(rnd(rng),rnd(rng),rnd(rng)).getNormalized();
        d.radius[0]=.1;d.radius[1]=.2;d.restDistance=.01;d.count[0]=8;d.count[1]=test%2?13:8;
        for(unsigned s=0;s<2;++s)for(unsigned i=0;i<d.count[s];++i)d.points[s].pushBack(PxVec3d(rnd(rng),rnd(rng),rnd(rng)));
        PxTransform start[2]={PxTransform(PxVec3(0)),PxTransform(PxVec3(0,2,0))},end[2];
        for(unsigned s=0;s<2;++s)end[s]=PxTransform(start[s].p+PxVec3(.01f,-.02f,.04f),
            PxQuat(float(rnd(rng)),PxVec3(.3f,1.f,.7f).getNormalized()));
        std::vector<AvbdNpPathFeature> f;
        for(unsigned i=0;i<d.count[0]*d.count[1];++i){f.push_back(avbdNpPathFeature(d,start,end,i));failures+=!f.back().valid;}
        double x[12];for(double& v:x)v=.03*rnd(rng);
        auto jac=[&](unsigned ij,unsigned k,unsigned c)
        {const unsigned s=c/6,q=c%6;return q<3?(s?-1:1)*d.normal[q]:f[ij].angular[k][s][q-3];};
        auto coordinate=[&](unsigned ij,unsigned k)
        {double v=f[ij].gap[k];for(unsigned c=0;c<12;++c)v+=jac(ij,k,c)*x[c];return v;};
        for(unsigned k=0;k<5;++k)
        {
            std::vector<double> u(d.count[0],0),v(d.count[1],0);
            double full[12]={},compressed[12]={},work=0,reducedWork=0,total=0;
            double lower=DBL_MAX,upper=DBL_MAX,minimum=DBL_MAX;
            const double base=coordinate(0,k);
            for(unsigned i=0;i<d.count[0];++i)lower=std::min(lower,coordinate(i*d.count[1],k));
            for(unsigned j=0;j<d.count[1];++j)upper=std::min(upper,coordinate(j,k)-base);
            for(unsigned i=0;i<d.count[0];++i)for(unsigned j=0;j<d.count[1];++j)
            {
                const unsigned ij=i*d.count[1]+j;const double lambda=std::abs(rnd(rng));
                u[i]+=lambda;v[j]+=lambda;total+=lambda;work+=lambda*coordinate(ij,k);
                minimum=std::min(minimum,coordinate(ij,k));
                for(unsigned c=0;c<12;++c)full[c]+=lambda*jac(ij,k,c);
            }
            for(unsigned i=0;i<d.count[0];++i)
            {reducedWork+=u[i]*coordinate(i*d.count[1],k);for(unsigned c=0;c<12;++c)compressed[c]+=u[i]*jac(i*d.count[1],k,c);}
            for(unsigned j=0;j<d.count[1];++j)
            {reducedWork+=v[j]*(coordinate(j,k)-base);for(unsigned c=0;c<12;++c)compressed[c]+=v[j]*(jac(j,k,c)-jac(0,k,c));}
            worstValue=std::max(worstValue,std::abs(minimum-lower-upper));
            worstValue=std::max(worstValue,std::abs(work-reducedWork)/std::max(1.,std::abs(work)));
            for(unsigned c=0;c<12;++c)worstWrench=std::max(worstWrench,std::abs(full[c]-compressed[c])/std::max(1.,total));
            // Lifted interval: max_j(-B_j) <= z <= min_i(A_i).
            failures+=(minimum>1.e-10)!=(lower+upper>1.e-10);++normalCases;
        }
    }
    failures+=worstValue>1.e-11 || worstWrench>1.e-11;
    std::printf("[NORMAL_BUNDLE] cases=%u valueError=%.12g wrenchError=%.12g production=0\n",normalCases,worstValue,worstWrench);

    AvbdCommonProgram p;p.dimensions.pushBack(6);p.sourceBlocks.pushBack(0);p.base.resize(6,0);
    AvbdCommonHBlock h={};for(unsigned i=0;i<6;++i)h.values[6*i+i]=1;p.H.pushBack(h);
    for(unsigned r=0;r<5;++r)
    {
        p.mu.pushBack(.5);p.coulomb.pushBack(1);p.sourceRows.pushBack(r);
        for(unsigned q=0;q<3;++q){p.offset.pushBack(0);p.compliance.pushBack(0);
            p.lower.pushBack(q?-std::numeric_limits<double>::infinity():0);p.upper.pushBack(std::numeric_limits<double>::infinity());}
        AvbdCommonJBlock j={};j.row=r;j.values[0]=1;j.values[3]=.1*r;
        j.values[7]=1;j.values[11]=.2;j.values[14]=1;j.values[16]=-.3;p.J.pushBack(j);
    }
    if(!p.finish()){std::printf("Invalid synthetic packet\n");return 2;}
    for(unsigned r=1;r<5;++r)failures+=!p.sameTangent(0,r);
    double worstRoot=0,worstConservation=0;
    for(unsigned test=0;test<90;++test)
    {
        double normal[5],sum=0;for(double& v:normal){v=test%3==2?0:std::abs(rnd(rng));sum+=v;}
        const double a=rnd(rng),b=rnd(rng),length=std::hypot(a,b);
        const bool sliding=test%3==1;const double magnitude=sliding?.5*sum:.25*sum;
        const double totalT[2]={-magnitude*a/length,-magnitude*b/length};
        double force[15]={},penalty[15],step[6]={},actual[6]={},expected[6]={};
        for(double& v:penalty)v=1;
        for(unsigned r=0;r<5;++r)
        {
            force[3*r]=normal[r];for(unsigned q=1;q<3;++q)
            {force[3*r+q]=sum?normal[r]/sum*totalT[q-1]:0;p.offset[3*r+q]=sliding?(q==1?a:b):0;}
            double projected[3],stiffness[3],root[3];p.projected(r,penalty,step,force,projected,stiffness,true,NULL,root);
            for(double v:root)worstRoot=std::max(worstRoot,std::abs(v));
            for(unsigned c=0;c<6;++c)for(unsigned q=0;q<3;++q)actual[c]+=p.J[r].values[6*q+c]*force[3*r+q];
            for(unsigned c=0;c<6;++c)expected[c]+=p.J[r].values[c]*normal[r];
        }
        for(unsigned c=0;c<6;++c)
        {expected[c]+=p.J[0].values[6+c]*totalT[0]+p.J[0].values[12+c]*totalT[1];worstConservation=std::max(worstConservation,std::abs(expected[c]-actual[c]));}
    }
    failures+=worstRoot>1.e-11 || worstConservation>1.e-11;
    std::printf("[SHARED_TANGENT_ROOT] cases=90 rootError=%.12g wrenchError=%.12g coordinates=15->7 production=0\n",worstRoot,worstConservation);
    // Tiny differences in the lever arm or slip offset prohibit this quotient.
    p.J[1].values[11]+=1.e-12;failures+=p.sameTangent(0,1);p.J[1].values[11]=p.J[0].values[11];
    p.offset[4]=p.offset[1]+1.e-12;failures+=p.sameTangent(0,1);
    // Equal roots do not imply that the sum of projected AL iterates agrees.
    const double separate=std::max(-.5,-.7)+std::max(-1.5,-.7),merged=std::max(-2.,-1.4);
    failures+=std::abs(separate-merged)<.19;
    // Opposite sliding at opposite arms has zero net tangent force but torque 2.
    const double separateTorque=1.*1.+(-1.)*(-1.),collapsedTorque=0.;
    failures+=std::abs(separateTorque-collapsedTorque)!=2.;
    std::printf("[MERGE_COUNTEREXAMPLES] alDifference=%.12g lostTorque=%.12g failures=%u\n",std::abs(separate-merged),separateTorque-collapsedTorque,failures);
    return failures?1:0;
}
