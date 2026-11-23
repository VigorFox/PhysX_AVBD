#include "avbd/DyAvbdCommonResponse.h"
#include "avbd/DyAvbdRigidMetric.h"
#include "avbd/DyAvbdCommonResidual.h"
#include "avbd_probe_foundation.h"
#include <thread>
#include <vector>
#include <limits>
using namespace physx;using namespace physx::Dy;
static unsigned seed=281026;
static double sample(){seed=1664525u*seed+1013904223u;return double(seed>>8)/16777216.-.5;}
int main()
{
    AvbdProbeFoundation foundation;double worst=0,solveError=0,rootError=0,stationError=0;unsigned failed=0,rejected=0;
    const KernelTable table=KernelTable::makeAvbdCpuScalar();
    {
        AvbdCommonProgram p;p.dimensions.pushBack(3);p.sourceBlocks.pushBack(0);
        p.sourceRows.pushBack(0);p.mu.pushBack(.5);p.coulomb.pushBack(1);
        AvbdCommonHBlock h={};h.values[0]=2;h.values[7]=4;h.values[14]=8;p.H.pushBack(h);
        AvbdCommonJBlock j={};j.values[0]=j.values[7]=j.values[14]=1;p.J.pushBack(j);
        const double inf=std::numeric_limits<double>::infinity(),f[3]={2,1,0},tol[3]={1e-5,1e-5,1e-5};
        for(unsigned d=0;d<3;++d)
        {p.base.pushBack(-f[d]);p.offset.pushBack(0);p.compliance.pushBack(0);p.lower.pushBack(d?-inf:0);p.upper.pushBack(inf);}
        failed+=!p.finish();AvbdCommonResidual metric(p);
        failed+=metric.evaluate(p,f)!=0 || !metric.smallRoot(p,f,tol);
        p.offset[0]=.01; // Unchanged pose / zero trial step cannot erase a bad law.
        failed+=!(metric.evaluate(p,f)>0) || metric.smallRoot(p,f,tol);
        p.offset[0]=0;const double partialForce[3]={2,.5,0};
        failed+=std::abs(metric.evaluate(p,partialForce)-.0625)>1.e-12;
        double root[3];const double u[3]={.1,0,0};
        avbdCoulombRoot(f,u,1,.5,p.lower.begin(),p.upper.begin(),true,root);
        failed+=std::abs(root[0]-.1)>1.e-12 || root[1]!=0 || root[2]!=0;
        p.upper[0]=.06;
        avbdCoulombRoot(f,u,1,.5,p.lower.begin(),p.upper.begin(),true,root);
        failed+=std::abs(root[0]-1.94)>1.e-12;
        std::printf("[COMMON_NONLINEAR_METRIC] zeroStepRejected=1 poseForceResidual=1 actualNormalDisk=1 forceCap=1 failed=%u\n",failed);
    }
    unsigned asymmetricLegacy=0;double inertiaError=0;
    unsigned nearParallelIterations=0;
    for(unsigned trial=0;trial<24;++trial)
    {
        AvbdCommonProgram p;p.dimensions.pushBack(3);p.sourceBlocks.pushBack(0);
        AvbdCommonHBlock h={};h.values[0]=h.values[7]=h.values[14]=1;p.H.pushBack(h);
        p.base.pushBack(-3);p.base.pushBack(0);p.base.pushBack(0);
        for(unsigned row=0;row<2;++row)
        {
            const bool friction=trial>=12;
            p.sourceRows.pushBack(row);p.mu.pushBack(friction?.4:0);p.coulomb.pushBack(friction?1:0);
            const bool weak=row==(trial%2);
            AvbdCommonJBlock j={};j.row=row;j.values[0]=1;j.values[2]=weak?0:1.e-8;
            if(friction)j.values[7]=j.values[14]=1;p.J.pushBack(j);
            for(unsigned q=0;q<3;++q)
            {p.offset.pushBack(q?0:1+(weak?2.e-7:0));p.lower.pushBack(q && friction?-std::numeric_limits<double>::infinity():0);
                p.upper.pushBack(q && !friction?0:std::numeric_limits<double>::infinity());p.compliance.pushBack(0);}
        }
        failed+=!p.finish();AvbdCommonResponse response(p,table);PxArray<double> step,f;AvbdCoulombNcpResult result;
        failed+=!response.solve(step,f,result);nearParallelIterations=std::max(nearParallelIterations,result.iterations);
        if(f.size()==6)failed+=std::abs(f[3*(trial%2)])>1.e-7 || std::abs(f[3*(1-trial%2)]-2)>1.e-7;
    }
    std::printf("[COMMON_NEAR_PARALLEL] cases=24 maxIterations=%u failed=%u\n",nearParallelIterations,failed);
    for(unsigned trial=0;trial<2000;++trial)
    {
        const PxVec3 axis=PxVec3(float(sample()),float(sample()),float(sample())).getNormalized();
        const PxQuat q(float(sample()*2.8),axis),target(PxIdentity);
        const PxMat33 R(PxQuat(float(sample()*2.8),PxVec3(0,1,0)));
        const float scale=std::pow(10.f,float(int(trial%10)-3));
        PxMat33 I=R*PxMat33::createDiagonal(PxVec3(2,7,13)*scale)*R.getTranspose();
        I[0][1]=std::nextafter(I[0][1],std::numeric_limits<float>::infinity());
        const auto result=avbdRotationInertialPotential(target,q,I);
        const PxMat33 J(PxVec3(q.w,-q.z,q.y),PxVec3(q.z,q.w,-q.x),PxVec3(-q.y,q.x,q.w));
        const PxMat33 legacy=J.getTranspose()*I*J;bool asym=false;
        double magnitude=0,error=0;
        for(unsigned r=0;r<3;++r)for(unsigned c=0;c<3;++c)
        {
            asym|=legacy[r][c]!=legacy[c][r];failed+=result.gaussNewton[r][c]!=result.gaussNewton[c][r];
            double reference=0;
            for(unsigned a=0;a<3;++a)for(unsigned b=0;b<3;++b)
                reference+=double(J[r][a])*.5*(double(I[b][a])+I[a][b])*double(J[c][b]);
            error=std::max(error,std::abs(reference-result.gaussNewton[r][c]));magnitude=std::max(magnitude,std::abs(reference));
        }
        asymmetricLegacy+=asym;inertiaError=std::max(inertiaError,error/std::max(1e-30,magnitude));
    }
    for(unsigned trial=0;trial<120;++trial)
    {
        AvbdCommonProgram p;unsigned n=4+trial%9,m=2+trial%7,dofs=0;
        std::vector<unsigned> starts(n+1);
        for(unsigned b=0;b<n;++b){starts[b]=dofs;p.dimensions.pushBack((trial+b)%2?3:6);dofs+=p.dimensions[b];p.sourceBlocks.pushBack(2*b);}
        starts[n]=dofs;for(unsigned d=0;d<dofs;++d)p.base.pushBack(0);
        for(unsigned r=0;r<m;++r){p.mu.pushBack(.4);p.coulomb.pushBack(1);p.sourceRows.pushBack(r);
            for(unsigned k=0;k<3;++k){p.offset.pushBack(sample());p.compliance.pushBack(0);p.lower.pushBack(k?-1e20:0);p.upper.pushBack(1e20);}}
        std::vector<double> H(dofs*dofs),J(3*m*dofs),x(dofs),f(3*m);
        for(unsigned repeat=0;repeat<2;++repeat)for(unsigned b=0;b<n;++b)for(unsigned c=0;c<n;++c)
        {
            AvbdCommonHBlock e={};e.row=b;e.column=c;
            for(unsigned d=0;d<p.dimensions[b];++d)for(unsigned k=0;k<p.dimensions[c];++k)
            {double v=sample();e.values[6*d+k]=v;H[(starts[b]+d)*dofs+starts[c]+k]+=v;}p.H.pushBack(e);
        }
        for(unsigned repeat=0;repeat<2;++repeat)for(unsigned r=0;r<m;++r)for(unsigned b=0;b<n;++b)
        {
            AvbdCommonJBlock e={};e.row=r;e.column=b;
            for(unsigned k=0;k<3;++k)for(unsigned d=0;d<p.dimensions[b];++d)
            {double v=sample();e.values[6*k+d]=v;J[(3*r+k)*dofs+starts[b]+d]+=v;}p.J.pushBack(e);
        }
        if(!p.finish()){++failed;continue;}
        for(double& v:x)v=sample();for(double& v:f)v=sample();
        std::vector<double> hx(dofs),jx(3*m),jtf(dofs);
        // Execute the actual destination ranges concurrently, including empty ranges.
        std::vector<std::thread> workers;
        for(unsigned t=0;t<4;++t)workers.emplace_back([&,t]{p.hessian(x.data(),hx.data(),n*t/4,n*(t+1)/4);
            p.jacobian(x.data(),jx.data(),m*t/4,m*(t+1)/4);p.transpose(f.data(),jtf.data(),n*t/4,n*(t+1)/4);});
        for(auto& w:workers)w.join();
        for(unsigned d=0;d<dofs;++d)
        {double h=0,j=0;for(unsigned k=0;k<dofs;++k)h+=H[d*dofs+k]*x[k];
            for(unsigned k=0;k<3*m;++k)j+=J[k*dofs+d]*f[k];worst=std::max(worst,std::abs(h-hx[d]));worst=std::max(worst,std::abs(j-jtf[d]));}
        double left=0,right=0;
        for(unsigned k=0;k<3*m;++k){double j=0;for(unsigned d=0;d<dofs;++d)j+=J[k*dofs+d]*x[d];worst=std::max(worst,std::abs(j-jx[k]));left+=f[k]*jx[k];}
        for(unsigned d=0;d<dofs;++d)right+=x[d]*jtf[d];worst=std::max(worst,std::abs(left-right));
        const unsigned dim=p.dimensions[0];p.dimensions[0]=4;rejected+=!p.finish();p.dimensions[0]=dim;
        p.lower[0]=p.upper[0]+1e20;rejected+=!p.finish();p.lower[0]=0;
        p.J[0].column=n;rejected+=!p.finish();p.J[0].column=0;
        p.H[0].values[0]=std::numeric_limits<double>::quiet_NaN();rejected+=!p.finish();
        // A manufactured mixed 3/6 root of the ORIGINAL full equation.
        // Positive physical compliance makes the dual unique even if J is rank deficient.
        p.H.clear();p.J.clear();
        for(unsigned b=0;b<n;++b)for(unsigned c=0;c<n;++c)
        {
            AvbdCommonHBlock e={};e.row=b;e.column=c;
            for(unsigned d=0;d<p.dimensions[b];++d)for(unsigned k=0;k<p.dimensions[c];++k)
            {double v=.03*x[starts[b]+d]*x[starts[c]+k]+(b==c && d==k?1.:0.);
                e.values[6*d+k]=v;H[(starts[b]+d)*dofs+starts[c]+k]=v;}p.H.pushBack(e);
        }
        for(unsigned r=0;r<m;++r)for(unsigned b=0;b<n;++b)
        {AvbdCommonJBlock e={};e.row=r;e.column=b;for(unsigned k=0;k<3;++k)for(unsigned d=0;d<p.dimensions[b];++d)
            e.values[6*k+d]=J[(3*r+k)*dofs+starts[b]+d];p.J.pushBack(e);}
        for(unsigned k=0;k<3*m;++k){f[k]=k%3? .1:.8;p.lower[k]=k%3?-std::numeric_limits<double>::infinity():0;
            p.upper[k]=std::numeric_limits<double>::infinity();p.compliance[k]=.01;
            double value=.01*f[k];for(unsigned d=0;d<dofs;++d)value+=J[k*dofs+d]*x[d];p.offset[k]=-value;}
        for(unsigned d=0;d<dofs;++d){double value=0;for(unsigned k=0;k<dofs;++k)value+=H[d*dofs+k]*x[k];
            for(unsigned k=0;k<3*m;++k)value-=J[k*dofs+d]*f[k];p.base[d]=value;}
        failed+=!p.finish();AvbdCommonResponse response(p,table);PxArray<double> step,force;AvbdCoulombNcpResult result;
        failed+=!response.solve(step,force,result);
        for(unsigned d=0;d<dofs;++d)solveError=std::max(solveError,std::abs(step[d]-x[d]));
        for(unsigned k=0;k<3*m;++k)solveError=std::max(solveError,std::abs(force[k]-f[k]));
        double xx=0;for(double v:x)xx+=v*v;const double alpha=.03/(1+.03*xx);
        for(unsigned r=0;r<m;++r)
        {
            double trace=.03,u[3]={},v[3];
            for(unsigned k=0;k<3;++k)
            {const unsigned row=3*r+k;double dot=0;u[k]=p.offset[row]+.01*force[row];
                for(unsigned d=0;d<dofs;++d){double j=J[row*dofs+d];dot+=j*x[d];trace+=j*j;u[k]+=j*step[d];}trace-=alpha*dot*dot;}
            for(unsigned k=0;k<3;++k)v[k]=force[3*r+k]-u[k]/trace;
            v[0]=std::max(0.,v[0]);const double length=std::hypot(v[1],v[2]),radius=.4*std::max(0.,force[3*r]);
            if(length>radius){v[1]*=radius/length;v[2]*=radius/length;}
            for(unsigned k=0;k<3;++k)rootError=std::max(rootError,std::abs(force[3*r+k]-v[k]));
        }
        for(unsigned d=0;d<dofs;++d)
        {double value=-p.base[d];for(unsigned k=0;k<dofs;++k)value+=H[d*dofs+k]*step[k];
            for(unsigned k=0;k<3*m;++k)value-=J[k*dofs+d]*force[k];stationError=std::max(stationError,std::abs(value));}
    }
    bool pass=!failed && rejected==480 && worst<1e-12 && rootError<1e-10 && stationError<1e-10 && asymmetricLegacy>0 && inertiaError<1e-6;
    std::printf("[COMMON_INERTIA] cases=2000 legacyAsymmetric=%u relativeError=%.17g\n",asymmetricLegacy,inertiaError);
    std::printf("[COMMON_PROGRAM_OPERATOR] mixed3x6=120 threads=4 rejected=%u error=%.17g solveError=%.17g rootError=%.17g stationError=%.17g failed=%u status=%s\n",rejected,worst,solveError,rootError,stationError,failed,pass?"PASS":"FAIL");return pass?0:1;
}
