// Second variations in the world-left tangent, independent finite differences.
#include "avbd/DyAvbdElasticFrame.h"
#include "avbd/DyAvbdCurvatureMetric.h"
#include <cstring>
#include <random>
#include <cstdio>
using namespace physx;using namespace physx::Dy;
static void gradient(const PxTransformT<double>* bodies,const PxTransformT<double>* local,
    const double* weight,const double* bias,double* g)
{
    const auto c=finiteJointCoordinates(bodies[0]*local[0],bodies[1]*local[1],bodies[0].p,bodies[1].p);
    for(unsigned d=0;d<9;++d)g[d]=0;
    for(unsigned r=0;r<FINITE_JOINT_ROWS;++r)
    {
        const double f=weight[r]*c.value[r]-bias[r];
        for(unsigned d=0;d<3;++d){g[d]+=f*c.linear[r][d];g[3+d]+=f*c.angularA[r][d];g[6+d]-=f*c.angularB[r][d];}
    }
}
int main()
{
    std::mt19937 rng(28491);std::normal_distribution<double> normal;double worst=0;unsigned failures=0;
    for(unsigned test=0;test<120;++test)
    {
        PxTransformT<double> bodies[2],local[2];double weight[FINITE_JOINT_ROWS],bias[FINITE_JOINT_ROWS];
        for(unsigned b=0;b<2;++b)
        {
            bodies[b]=PxTransformT<double>(PxVec3d(normal(rng),normal(rng),normal(rng)),
                PxQuatT<double>(normal(rng),PxVec3d(normal(rng),normal(rng),normal(rng)).getNormalized()));
            local[b]=PxTransformT<double>(PxVec3d(normal(rng),normal(rng),normal(rng))*.3,
                PxQuatT<double>(normal(rng),PxVec3d(normal(rng),normal(rng),normal(rng)).getNormalized()));
        }
        for(unsigned r=0;r<FINITE_JOINT_ROWS;++r){weight[r]=1+std::abs(normal(rng));bias[r]=normal(rng);}
        double analytic[81],numeric[81];const auto fa=bodies[0]*local[0],fb=bodies[1]*local[1];
        finiteJointMaterialHessian(fa,fb,bodies[0].p,bodies[1].p,weight,bias,analytic);
        for(unsigned column=0;column<9;++column)
        {
            double g[2][9];const double eps=1.e-5;
            for(unsigned side=0;side<2;++side)
            {
                PxTransformT<double> x[2]={bodies[0],bodies[1]};const double h=side?eps:-eps;
                if(column<3)x[0].p[column]+=h;
                else {const unsigned body=column<6?0:1;PxVec3d axis(0);axis[column%3]=1;
                    x[body].q=PxQuatT<double>(h,axis)*x[body].q;}
                gradient(x,local,weight,bias,g[side]);
            }
            for(unsigned row=0;row<9;++row)numeric[9*row+column]=(g[1][row]-g[0][row])/(2*eps);
        }
        double error=0,scale=0;
        // Symmetrization removes the connection term of the moving world-left
        // gradient. Its contraction with the direction is identically zero.
        for(unsigned i=0;i<9;++i)for(unsigned j=0;j<9;++j)
        {const double expected=.5*(numeric[9*i+j]+numeric[9*j+i]);error+=std::pow(analytic[9*i+j]-expected,2);scale+=expected*expected;}
        const double relative=std::sqrt(error/scale);worst=std::max(worst,relative);failures+=relative>1.e-7;
    }
    std::printf("[FRAME_CURVATURE] cases=120 relativeError=%.12g failures=%u\n",worst,failures);
    double worstMetric=0;unsigned metricFailures=0;
    for(unsigned test=0;test<120;++test)
    {
        const unsigned n=test%2?3:6;double L[6][6]={},Q[6][6]={},M[36]={},K[36]={},expected[36]={},before[36];
        double units[6],eigen[6];
        for(unsigned i=0;i<n;++i)
        {
            units[i]=std::pow(10.,int(i)-3);Q[i][i]=1;L[i][i]=units[i];
            for(unsigned j=0;j<i;++j)L[i][j]=.1*normal(rng)*units[i];
            eigen[i]=test%3==0?.2+.1*i:i==0?-.1:i==1?-2:.2+.1*i;
        }
        for(unsigned turn=0;turn<30;++turn)
        {
            const unsigned a=rng()%n,b=rng()%n;if(a==b)continue;
            const double angle=normal(rng),c=std::cos(angle),s=std::sin(angle);
            for(unsigned i=0;i<n;++i){const double x=Q[i][a],y=Q[i][b];Q[i][a]=c*x-s*y;Q[i][b]=s*x+c*y;}
        }
        double V[6][6]={};for(unsigned i=0;i<n;++i)for(unsigned j=0;j<n;++j)for(unsigned k=0;k<n;++k)V[i][j]+=L[i][k]*Q[k][j];
        for(unsigned i=0;i<n;++i)for(unsigned j=0;j<n;++j)for(unsigned k=0;k<n;++k)
        {M[6*i+j]+=L[i][k]*L[j][k];K[6*i+j]+=V[i][k]*eigen[k]*V[j][k];expected[6*i+j]+=V[i][k]*(eigen[k]>0?eigen[k]:1)*V[j][k];}
        std::memcpy(before,K,sizeof K);const bool ok=avbdRegularizeLocalCurvature(K,M,n);
        double error=0;for(unsigned i=0;i<n;++i)for(unsigned j=0;j<n;++j)error=std::max(error,std::abs(K[6*i+j]-expected[6*i+j])/(units[i]*units[j]));
        worstMetric=std::max(worstMetric,error);metricFailures+=!ok || error>1.e-10 || (test%3==0 && std::memcmp(before,K,sizeof K)!=0);
    }
    double singular[36]={},identity[36]={};for(unsigned i=0;i<6;++i){identity[6*i+i]=1;singular[6*i+i]=i?1:0;}
    metricFailures+=!avbdRegularizeLocalCurvature(singular,identity,6) || singular[0]!=1;
    std::printf("[LOCAL_CURVATURE_METRIC] cases=121 unitScaledError=%.12g failures=%u\n",worstMetric,metricFailures);
    return failures || metricFailures?1:0;
}
