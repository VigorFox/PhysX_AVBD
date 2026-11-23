#include "../experiments/avbd_contact_lab/sparse_symmetric.h"
#include "avbd/DyAvbdCoulombNcp.h"
#include "avbd/DyKernelTable.h"
#include <cstdio>
#include "avbd_probe_foundation.h"
#include <random>

struct DenseResponse
{
    int n;std::vector<double> values;
    void apply(const double* x,double* y)
    {for(int i=0;i<n;++i){y[i]=0;for(int j=0;j<n;++j)y[i]+=values[n*i+j]*x[j];}}
};

int main()
{
    AvbdProbeFoundation foundation;
    std::mt19937 rng(271029);std::uniform_real_distribution<double> random(-1,1);
    int failures=0;double worstSolve=0,worstDerivative=0,worstAdjoint=0;
    for(int test=0;test<192;++test)
    {
        const int n=12,m=6,N=n+m;std::vector<double> A(N*N,0),R(n*n),C(m*m),scale(N),expected(N),b(N,0),x(N);
        for(double& v:R){v=random(rng);if(test%2 && random(rng)>.0)v=0;}
        for(double& v:C){v=random(rng);if(test%2 && random(rng)>.0)v=0;}
        for(int i=0;i<n;++i)for(int j=0;j<n;++j)
        {double v=i==j?1:0;for(int k=0;k<n;++k)v+=R[k*n+i]*R[k*n+j];A[i*N+j]=v;}
        for(int i=0;i<m;++i)for(int j=0;j<m;++j)
        {double v=i==j?1:0;for(int k=0;k<m;++k)v+=C[k*m+i]*C[k*m+j];A[(n+i)*N+n+j]=-v;}
        for(int i=0;i<n;++i)for(int j=0;j<m;++j)
        {double v=random(rng);if(test%2 && random(rng)>-.5)v=0;A[i*N+n+j]=A[(n+j)*N+i]=v;}
        for(int i=0;i<N;++i){scale[i]=std::pow(10.,4*random(rng));expected[i]=random(rng)/scale[i];}
        AvbdLabSparseSymmetric matrix(N);
        for(int i=0;i<N;++i)for(int j=0;j<N;++j)
        {A[i*N+j]*=scale[i]*scale[j];b[i]+=A[i*N+j]*expected[j];if(i>=j && A[i*N+j]!=0)matrix.add(i,j,A[i*N+j]);}
        matrix.compile();if(test%2==0)matrix.compile();
        if(!matrix.factor() || !matrix.solve(b.data(),x.data())){++failures;continue;}
        double error=0;
        for(int i=0;i<N;++i)
        {
            double value=-b[i],magnitude=std::abs(b[i]);
            for(int j=0;j<N;++j){value+=A[i*N+j]*x[j];magnitude+=std::abs(A[i*N+j]*x[j]);}
            error=std::max(error,std::abs(value)/(1+magnitude));
        }
        worstSolve=std::max(worstSolve,error);if(error>1.e-12)++failures;
    }
    for(int test=0;test<192;++test)
    {
        const int count=5,n=3*count;DenseResponse response={n,std::vector<double>(n*n,0)};
        std::vector<double> R(n*n),d(n),f(n),a(n),b(n),Da(n),DTb(n),plus(n),minus(n),u(n),v(n),mu(count,.4),C(n,.1),lo(n,-std::numeric_limits<double>::infinity()),hi(n,std::numeric_limits<double>::infinity());
        int kinds[count]={1,1,1,0,0};
        for(double& value:R)value=random(rng);
        for(int i=0;i<n;++i)for(int j=0;j<n;++j)
        {double value=i==j?1.:0.;for(int k=0;k<n;++k)value+=R[k*n+i]*R[k*n+j];response.values[i*n+j]=value;}
        for(int i=0;i<n;++i){d[i]=random(rng);f[i]=random(rng);a[i]=random(rng);b[i]=random(rng);}
        for(int c=0;c<3;++c)lo[3*c]=0;hi[0]=.7;mu[1]=0;lo[9]=-.2;hi[9]=.3;
        std::vector<double> work(physx::Dy::avbdCoulombNcpWorkSize(count)+1,123.);
        physx::Dy::AvbdCoulombNcp::Solver<DenseResponse> solver(response,count,d.data(),mu.data(),work.data(),kinds,lo.data(),hi.data(),C.data());
        if(!solver.valid || work.back()!=123.){++failures;continue;}
        solver.evaluate(f.data(),solver.r.data(),true);solver.apply(a.data(),Da.data());solver.transpose(b.data(),DTb.data());
        const double adjoint=std::abs(physx::Dy::AvbdCoulombNcp::dot(Da.data(),b.data(),n)-physx::Dy::AvbdCoulombNcp::dot(a.data(),DTb.data(),n));
        worstAdjoint=std::max(worstAdjoint,adjoint);if(adjoint>1.e-10)++failures;
        for(int i=0;i<n;++i){plus[i]=f[i]+1.e-6*a[i];minus[i]=f[i]-1.e-6*a[i];}
        solver.evaluate(plus.data(),u.data(),false);solver.evaluate(minus.data(),v.data(),false);
        double derivative=0;
        for(int i=0;i<n;++i)derivative=std::max(derivative,std::abs((u[i]-v[i])/2.e-6-Da[i]));
        worstDerivative=std::max(worstDerivative,derivative);if(derivative>1.e-6)++failures;
    }
    // Production SPD factors must reject lost positive definiteness, never
    // inherit the lab cone's quasi-definite pivot repair.
    for(int test=0;test<192;++test)
    {
        const int n=9;std::vector<double> R(n*n),A(n*n,0),scale(n),expected(n),b(n,0),x(n);
        for(double& v:R)v=random(rng);
        for(int i=0;i<n;++i){scale[i]=std::pow(10.,4*random(rng));expected[i]=random(rng)/scale[i];}
        physx::Dy::AvbdSparseSymmetric matrix(n);
        for(int i=0;i<n;++i)for(int j=0;j<n;++j)
        {
            double v=i==j?1.:0.;for(int k=0;k<n;++k)v+=R[k*n+i]*R[k*n+j];
            A[i*n+j]=v*scale[i]*scale[j];b[i]+=A[i*n+j]*expected[j];if(i>=j)matrix.add(i,j,A[i*n+j]);
        }
        if(matrix.factor() || matrix.solve(b.data(),x.data()))++failures;
        if(!matrix.compile() || !matrix.factor() || !matrix.solve(b.data(),x.data())){++failures;continue;}
        for(int i=0;i<n;++i)
        {
            double value=-b[i],magnitude=std::abs(b[i]);
            for(int j=0;j<n;++j){value+=A[i*n+j]*x[j];magnitude+=std::abs(A[i*n+j]*x[j]);}
            const double error=std::abs(value)/(1+magnitude);worstSolve=std::max(worstSolve,error);if(error>1e-12)++failures;
        }
    }
    physx::Dy::AvbdSparseSymmetric indefinite(2);indefinite.add(0,0,1);indefinite.add(1,1,1);indefinite.add(1,0,2);
    if(!indefinite.compile() || indefinite.factor())++failures;
    physx::Dy::AvbdSparseSymmetric invalid(1);invalid.add(0,1,1);if(invalid.compile() || invalid.factor())++failures;
    physx::Dy::AvbdSparseSymmetric nonfinite(1);nonfinite.add(0,0,std::numeric_limits<double>::infinity());if(nonfinite.compile())++failures;
    AvbdLabSparseSymmetric zero(2);zero.add(0,0,1);zero.compile();if(zero.factor())++failures;
    using namespace physx::Dy;
    const KernelTable table=KernelTable::makeAvbdCpuScalar();
    const double inf=std::numeric_limits<double>::infinity();
    DenseResponse op={6,std::vector<double>(36,0)};
    for(int i=0;i<5;++i)op.values[6*i+i]=1.;
    for(int i=0;i<4;++i)op.values[6*i+i+1]=op.values[6*(i+1)+i]=.1;
    double demand[6],mu[2]={.4,0},lo[6]={0,-inf,-inf,-1,-.3,0},hi[6]={inf,inf,inf,1,.3,0};
    double C[6]={.1,0,0,0,0,0},force[6]={};int kinds[2]={1,0};
    std::vector<double> work(avbdCoulombNcpWorkSize(2)+1,123.);AvbdCoulombNcpResult result;
    AvbdCoulombNcpArgs args={&op,[](void* p,const double* x,double* y){static_cast<DenseResponse*>(p)->apply(x,y);},
        2,64,demand,mu,lo,hi,C,kinds,force,work.data(),&result};
    for(int capped=0;capped<2;++capped)
    {
        double expected[6]={capped?.5:1.,capped?-.2:-.4,0,.2,-.3,0};
        double gap[6]={capped?-1.:0.,1.,0,0,1.,2.};hi[0]=capped?.5:inf;
        op.apply(expected,demand);
        for(int i=0;i<6;++i){demand[i]=gap[i]-demand[i]-C[i]*expected[i];force[i]=0;}
        table.launch(AVBD_COULOMB_NCP,&args);
        if(!result.converged || work.back()!=123.)++failures;
        for(int i=0;i<6;++i)if(std::abs(force[i]-expected[i])>1.e-9)++failures;
    }
    args.iterations=0;std::fill(force,force+6,0.);table.launch(AVBD_COULOMB_NCP,&args);if(result.converged)++failures;
    args.iterations=64;force[0]=std::numeric_limits<double>::quiet_NaN();
    table.launch(AVBD_COULOMB_NCP,&args);if(result.converged)++failures;
    std::fill(force,force+6,0.);C[0]=-1.;table.launch(AVBD_COULOMB_NCP,&args);if(result.converged)++failures;
    args.groups=0;args.work=NULL;table.launch(AVBD_COULOMB_NCP,&args);if(!result.converged)++failures;
    std::printf("[SPARSE_COULOMB] Table/mixed/caps/padding/failure sparseIndefinite=192 sparseSpd=192 derivatives=192 solve=%.9g derivative=%.9g adjoint=%.9g failures=%d\n",
        worstSolve,worstDerivative,worstAdjoint,failures);
    return failures?1:0;
}
