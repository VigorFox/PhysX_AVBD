#include "avbd/DyAvbdNodalShell.h"
#include "avbd/DyKernelTable.h"
#include "avbd_probe_foundation.h"
#include <cstdio>
#include <limits>
using namespace physx;using namespace physx::Dy;
template<class T> unsigned verify()
{
    typedef PxVec3T<T> V;typedef PxMat33T<T> M;
    unsigned failures=0;T worst=0;
    {
        const V normal=V(T(.25),T(-.4),T(.7)).getNormalized();
        const M K=avbdNodalOuter(normal,normal)*T(24);
        const M A=M(PxIdentity)*T(3)+K,B=K*T(-1),C=M(PxIdentity)*T(7)+K;
        const V ga=normal*T(5),gb=-ga;V da(0),db(0);
        failures+=!avbdSolveNodalPair(A,B,C,ga,gb,da,db);
        const T tol=T(128)*std::numeric_limits<T>::epsilon();
        failures+=(A*da+B*db+ga).magnitude()>tol || (B.getTranspose()*da+C*db+gb).magnitude()>tol;
        failures+=(da*T(3)+db*T(7)).magnitude()>tol || ga.dot(da)+gb.dot(db)>=0;
        failures+=avbdSolveNodalPair(M(PxZero),B,C,ga,gb,da,db);
    }
    failures+=avbdNodalTrianglePath(V(0,0,1),V(0,0,-1));
    failures+=!avbdNodalTrianglePath(V(0,0,1),V(1,0,0));
    failures+=avbdNodalTrianglePath(V(0),V(0));
    {
        const V x[]={V(0),V(1,0,0),V(0,1,0)};
        const V translation(2,1,3);
        const V y[]={translation,translation+V(0,1,0),translation+V(-1,0,0)};
        const V collapse[]={translation,translation-V(1,0,0),translation-V(0,1,0)};
        failures+=!avbdMembranePath(x,y) || avbdMembranePath(x,collapse);
    }
    {
        AvbdNodalShellRestT<T> rest;
        rest.area=T(.5);rest.shape[0]=V(-1,-1,0);rest.shape[1]=V(1,0,0);rest.shape[2]=V(0,1,0);
        const V x[]={V(0),V(T(1.12),T(.04),T(.2)),V(T(.08),T(.94),T(-.1))};
        const V old[]={V(0),V(T(.01),T(-.02),0),V(0,T(.005),0)};
        const T mobility[]={T(0),T(.003),T(.0007)},mu=T(700),lambda=T(300);
        V force[3];
        if(!avbdMembraneTrialForce(rest,x,mobility,old,mu,lambda,force))++failures;
        else
        {
            const V a=x[1]-x[0],b=x[2]-x[0];
            V strain((a.dot(a)-1)/2,(b.dot(b)-1)/2,a.dot(b)),net(0),torque(0);
            M J[3];
            for(PxU32 i=0;i<3;++i)
            {
                const T s=rest.shape[i].x,t=rest.shape[i].y;
                J[i]=M(a*s,b*t,a*t+b*s);
                strain+=J[i].getTranspose()*(force[i]*mobility[i]-old[i]);
                net+=force[i];torque+=x[i].cross(force[i]);
            }
            const T trace=lambda*(strain.x+strain.y);
            const V stress=V(2*mu*strain.x+trace,2*mu*strain.y+trace,mu*strain.z)*rest.area;
            const T tolerance=T(128)*std::numeric_limits<T>::epsilon()*(1+stress.magnitude());
            for(PxU32 i=0;i<3;++i)failures+=(force[i]+J[i]*stress).magnitude()>tolerance;
            failures+=net.magnitude()>tolerance || torque.magnitude()>tolerance;
        }
    }
    for(unsigned kind=0;kind<2;++kind)
    {
        AvbdNodalShellRestT<T> rest;rest.kind=kind;rest.area=T(.5);rest.length=1;
        if(kind==rest.Hinge){rest.restCos=T(.8);rest.restSin=T(.6);}
        rest.shape[0]=V(-1,-1,0);rest.shape[1]=V(1,0,0);rest.shape[2]=V(0,1,0);
        V x[4]={V(T(.1),T(.2),T(.03)),V(T(1.2),T(.1),T(.2)),V(T(.1),T(1.1),T(-.1)),V(T(.8),T(1.3),T(.4))};
        const T eps=sizeof(T)==4?T(.001):T(1.e-5),tolerance=sizeof(T)==4?T(.002):T(2.e-8);
        if(kind==rest.Hinge)
        {
            // Edge along +x, first normal +z and second normal (0,-.6,.8).
            const V folded[4]={V(0,1,0),V(0,T(-.8),T(-.6)),V(0),V(1,0,0)};
            for(PxU32 c=0;c<4;++c)
            {
                const auto hold=avbdSampleNodalShell(rest,folded,T(7),T(4),T(2),c);
                failures+=!hold.valid || PxAbs(hold.energy)>tolerance || hold.gradient.magnitude()>tolerance;
            }
        }
        V net(0),torque(0);const T mu=7,lambda=4,bend=2;
        const auto energy=[&](){return avbdSampleNodalShell(rest,x,mu,lambda,bend,PX_MAX_U32).energy;};
        for(unsigned c=0;c<rest.arity();++c)
        {
            AvbdNodalWorkT<T> work;
            const auto sample=avbdSampleNodalShell(rest,x,mu,lambda,bend,c,static_cast<const T*>(NULL),&work);
            failures+=!sample.valid;net+=sample.gradient;torque+=x[c].cross(sample.gradient);
            if(kind==rest.Membrane)
            {
                const V saved=x[c];x[c]+=V(T(.073),T(-.041),T(.129));
                const V dx=x[c]-saved;const T r2=dx.magnitudeSquared();
                const T change=dx.dot(sample.gradient+(sample.metric*dx+dx*work.geometric)*T(.5))+
                    r2*(work.cubic.dot(dx)+work.quartic*r2);
                failures+=!work.polynomial || PxAbs(change-(energy()-sample.energy))>
                    T(64)*std::numeric_limits<T>::epsilon()*(1+sample.energy);
                x[c]=saved;
            }
            for(unsigned axis=0;axis<3;++axis)
            {
                const T old=x[c][axis];x[c][axis]=old+eps;const T plus=energy();
                x[c][axis]=old-eps;const T minus=energy();x[c][axis]=old;
                const T finite=(plus-minus)/(2*eps),error=PxAbs(finite-sample.gradient[axis])/(1+PxAbs(finite));
                worst=PxMax(worst,error);failures+=error>tolerance;
            }
            const M A=sample.metric+M(PxIdentity);V step(0);
            failures+=!avbdSolveNodal3(A,sample.gradient,step);
            failures+=(A*step+sample.gradient).magnitude()>tolerance*(1+sample.gradient.magnitude());
            failures+=step.dot(sample.gradient)>tolerance;
        }
        const T weights[4]={T(.3),T(-.7),T(.2),T(.1)};
        const auto relative=avbdSampleNodalShell(rest,x,mu,lambda,bend,PX_MAX_U32,weights);
        V independent(0);for(PxU32 c=0;c<rest.arity();++c)
            independent+=avbdSampleNodalShell(rest,x,mu,lambda,bend,c).gradient*weights[c];
        failures+=!relative.valid || (relative.gradient-independent).magnitude()>tolerance;
        for(PxU32 axis=0;axis<3;++axis)
        {
            V saved[4];for(PxU32 c=0;c<rest.arity();++c){saved[c]=x[c];x[c][axis]+=eps*weights[c];}
            const T plus=energy();for(PxU32 c=0;c<rest.arity();++c)x[c][axis]=saved[c][axis]-eps*weights[c];
            const T minus=energy();for(PxU32 c=0;c<rest.arity();++c)x[c]=saved[c];
            const T finite=(plus-minus)/(2*eps);
            failures+=PxAbs(finite-relative.gradient[axis])/(1+PxAbs(finite))>tolerance;
        }
        failures+=net.magnitude()>tolerance || torque.magnitude()>tolerance;
        const T before=energy();for(unsigned c=0;c<rest.arity();++c)x[c]=V(-x[c].y,x[c].x,x[c].z)+V(3,-2,1);
        failures+=PxAbs(energy()-before)>tolerance*(1+before);
    }
    std::printf("[NODAL_SHELL_KERNEL] scalarBytes=%u worstDerivative=%.9g failures=%u\n",unsigned(sizeof(T)),double(worst),failures);
    return failures;
}
int main()
{
    AvbdProbeFoundation foundation;unsigned failures=verify<float>()+verify<double>();
    const auto table=KernelTable::makeAvbdCpuScalar();
    AvbdNodalMat33 A=AvbdNodalMat33::createDiagonal(AvbdNodalVec3(2,3,4));
    AvbdNodalVec3 gradient(2,6,12),step(0);AvbdNodalSolveArgs args={&A,&gradient,&step};
    table.launch(AVBD_NODAL_SOLVE3,&args);failures+=!args.solved || (step-AvbdNodalVec3(-1,-2,-3)).magnitude()>1.e-5f;
    A[0][0]=-1;table.launch(AVBD_NODAL_SOLVE3,&args);failures+=args.solved;
    return failures?1:0;
}
