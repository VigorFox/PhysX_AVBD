// Independent KKT and overlapping-coordinate convergence checks.
#include "avbd/DyAvbdNonlinearGroup.h"
#include "avbd/DyKernelTable.h"
#include "avbd_probe_foundation.h"
#include <cstdio>
using namespace physx;using namespace physx::Dy;
// Verify the support scatter against its directional energy, including merged
// aliases and contact cross terms. This does not compare two implementations.
template<class Scalar> unsigned checkRankOne(bool offDiagonal)
{
    AvbdLocalNonlinearGroup b;b.count=16;
    const PxU32 endpoints[4]={1,8,15,8},members[3]={15,1,8};
    Scalar J[16][6]={};double direction[16][6]={},projections[16]={};
    for(unsigned a=0;a<4;++a)for(unsigned d=0;d<6;++d)
        J[endpoints[a]][d]+=Scalar(std::sin(1.+6*a+d));
    for(unsigned i=0;i<16;++i)for(unsigned d=0;d<6;++d)
    {direction[i][d]=std::cos(1.+6*i+d);projections[i]+=double(J[i][d])*direction[i][d];}
    Scalar packed[3][6];for(unsigned a=0;a<3;++a)for(unsigned d=0;d<6;++d)packed[a][d]=J[members[a]][d];
    const double curvature=2.75;b.addRankOne(members,3,packed,curvature,offDiagonal);
    double projection=0,diagonal=0,actual=0;unsigned failures=0;
    for(double x:projections){projection+=x;diagonal+=x*x;}
    for(unsigned i=0;i<16;++i)for(unsigned j=0;j<16;++j)
        for(unsigned d=0;d<6;++d)for(unsigned k=0;k<6;++k)
        {
            const double h=b.K[i][j][6*d+k];actual+=direction[i][d]*h*direction[j][k];
            failures+=!std::isfinite(h) || h!=b.M[i][j][6*d+k];
            if((i!=1 && i!=8 && i!=15) || (j!=1 && j!=8 && j!=15) || (offDiagonal && i==j))
                failures+=h!=0;
        }
    failures+=std::abs(actual-curvature*(projection*projection-(offDiagonal?diagonal:0)))>1.e-10;
    return failures;
}
int main()
{
    AvbdProbeFoundation foundation;auto table=KernelTable::makeAvbdCpuScalar();
    unsigned failures=0,cases=0;double worstBalance=0,worstKkt=0;
    for(bool crossOnly:{false,true})
    {failures+=checkRankOne<float>(crossOnly)+checkRankOne<double>(crossOnly);cases+=2;}
    for(unsigned count:{1u,2u,3u,4u,8u,16u})for(unsigned trial=0;trial<20;++trial)
    {
        AvbdLocalNonlinearGroup b;b.count=count;
        for(unsigned i=0;i<count;++i)
        {
            b.translationMass[i]=(i==0?.2:i==1?17:1.3)*(1+trial);
            for(unsigned d=0;d<6;++d)
            {
                b.gradient[i][d]=std::sin(1.+6*i+d+trial);
                for(unsigned j=0;j<count;++j)for(unsigned k=0;k<6;++k)
                {
                    // Known SPD dense matrix, including translation/rotation coupling.
                    const double a=std::sin(1.+6*i+d),c=std::sin(1.+6*j+k);
                    b.K[i][j][6*d+k]=b.M[i][j][6*d+k]=(i==j && d==k?2.+i+d:0)+a*c;
                }
            }
        }
        if(trial%3==0)b.locked[count-1]=1u<<5;
        // Fail after a substantial partial factor, exercising scratch reuse.
        const bool metric=trial%2!=0;if(metric)b.K[count-1][count-1][28]=-1;
        const auto original=b;AvbdLocalNonlinearGroupArgs args={&b};table.launch(AVBD_LOCAL_NONLINEAR_GROUP,&args);++cases;
        failures+=!b.solved || b.metric!=metric || b.substitutions!=4 || b.factorizations!=(metric?3u:2u);
        double residual[16][6]={},balance[3]={},lambda[3]={},slope=0;
        for(unsigned i=0;i<count;++i)for(unsigned d=0;d<6;++d)
        {
            residual[i][d]=original.gradient[i][d];slope+=original.gradient[i][d]*b.step[i][d];
            for(unsigned j=0;j<count;++j)for(unsigned k=0;k<6;++k)
                residual[i][d]+=original.M[i][j][6*d+k]*b.step[j][k];
            if(d<3)balance[d]+=b.translationMass[i]*b.step[i][d];
        }
        for(unsigned d=0;d<3;++d)lambda[d]=residual[0][d]/b.translationMass[0];
        for(unsigned i=0;i<count;++i)for(unsigned d=0;d<6;++d)
        {
            if(original.locked[i]&(1u<<d)){failures+=b.step[i][d]!=0;continue;}
            const double e=std::abs(residual[i][d]-(d<3?lambda[d]*b.translationMass[i]:0));
            worstKkt=std::max(worstKkt,e);failures+=e>1.e-10;
        }
        for(double x:balance){worstBalance=std::max(worstBalance,std::abs(x));failures+=std::abs(x)>1.e-10;}
        failures+=slope>0;
    }
    // A closed elastic chain: exact manufactured minimizer has zero mass
    // weighted displacement. Alternating pairs must converge without pinning
    // each pair COM; keeping one partition is deliberately insufficient.
    double x[6]={},mass[6]={1,2,1,3,2,1},desired[6]={.1,-.1,.2,-.1,.1,0},hat[6]={};
    double mean=0,total=0;for(unsigned i=0;i<6;++i){mean+=mass[i]*desired[i];total+=mass[i];}
    for(unsigned i=0;i<6;++i)desired[i]-=mean/total;
    const double stiffness=2;
    for(unsigned i=0;i<6;++i)
    {
        double f=0;if(i)f+=stiffness*(desired[i]-desired[i-1]);if(i+1<6)f+=stiffness*(desired[i]-desired[i+1]);
        hat[i]=x[i]=desired[i]+f/mass[i];
    }
    for(unsigned sweep=0;sweep<1024;++sweep)for(unsigned a=sweep%2;a+1<6;a+=2)
    {
        AvbdNonlinearGroup b;b.count=2;
        for(unsigned s=0;s<2;++s)
        {
            const unsigned i=a+s;b.translationMass[s]=mass[i];
            for(unsigned d=0;d<6;++d)b.K[s][s][7*d]=b.M[s][s][7*d]=mass[i];
            b.gradient[s][0]=mass[i]*(x[i]-hat[i]);
            if(i){b.gradient[s][0]+=stiffness*(x[i]-x[i-1]);b.K[s][s][0]+=stiffness;}
            if(i+1<6){b.gradient[s][0]+=stiffness*(x[i]-x[i+1]);b.K[s][s][0]+=stiffness;}
            b.M[s][s][0]=b.K[s][s][0];
        }
        b.K[0][1][0]=b.K[1][0][0]=b.M[0][1][0]=b.M[1][0][0]=-stiffness;
        AvbdNonlinearGroupArgs args={&b};table.launch(AVBD_NONLINEAR_GROUP,&args);failures+=!b.solved;
        for(unsigned s=0;s<2;++s)x[a+s]+=b.step[s][0];
    }
    double error=0,balance=0;
    for(unsigned i=0;i<6;++i){error=std::max(error,std::abs(x[i]-desired[i]));balance+=mass[i]*(x[i]-hat[i]);}
    failures+=error>1.e-9 || std::abs(balance)>1.e-10;++cases;
    std::printf("[LOCAL_REACTION_PROBE] status=%s cases=%u failures=%u balance=%.12g kkt=%.12g chainError=%.12g\n",
        failures?"FAIL":"PASS",cases,failures,worstBalance,worstKkt,error);
    return failures?1:0;
}
