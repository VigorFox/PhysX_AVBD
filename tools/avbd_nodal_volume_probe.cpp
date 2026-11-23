#include "avbd/DyAvbdNodalVolume.h"
#include "avbd/DyKernelTable.h"
#include "avbd_probe_foundation.h"
#include <cstdio>
#include <limits>
using namespace physx;using namespace physx::Dy;

template<class T> AvbdNodalVolumeRestT<T> cookRest(const PxVec3T<T>* x)
{
    const PxMat33T<T> D(x[1]-x[0],x[2]-x[0],x[3]-x[0]);
    const auto inverse=D.getInverse();AvbdNodalVolumeRestT<T> rest;
    for(PxU32 i=0;i<3;++i)rest.shape[i+1]=PxVec3T<T>(inverse[0][i],inverse[1][i],inverse[2][i]);
    rest.shape[0]=-rest.shape[1]-rest.shape[2]-rest.shape[3];
    rest.volume=PxAbs(D.getDeterminant())/T(6);return rest;
}

template<class T> unsigned verify()
{
    typedef PxVec3T<T> V;typedef PxMat33T<T> M;
    const V reference[4]={V(0),V(T(1.2),T(.1),0),V(T(.2),T(.9),T(.1)),V(T(.1),T(.2),T(1.1))};
    const auto rest=cookRest(reference);
    const T mu=7,lambda=4,eps=sizeof(T)==4?T(.0005):T(1.e-5);
    const T tolerance=sizeof(T)==4?T(.002):T(2.e-8);
    unsigned failures=0;T worstGradient=0,worstHessian=0;
    V x[4];for(PxU32 i=0;i<4;++i)x[i]=reference[i];
    for(PxU32 c=0;c<4;++c)
    {
        const auto atRest=avbdSampleNodalVolume(rest,x,mu,lambda,c);
        failures+=!atRest.valid || PxAbs(atRest.energy)>tolerance || atRest.gradient.magnitude()>tolerance;
    }
    for(PxU32 sampleIndex=0;sampleIndex<8;++sampleIndex)
    {
        const T amount=T(.03)*(T(sampleIndex)-T(3));
        for(PxU32 i=0;i<4;++i)
            x[i]=V(reference[i].x*(1+amount),reference[i].y*(1-amount/T(2)),reference[i].z)+
                V(T(.02*i),T(-.013*i*i),T(.017*(i%2)));
        V net(0),torque(0);
        for(PxU32 c=0;c<4;++c)
        {
            const auto actual=avbdSampleNodalVolume(rest,x,mu,lambda,c);
            failures+=!actual.valid;net+=actual.gradient;torque+=x[c].cross(actual.gradient);
            for(PxU32 axis=0;axis<3;++axis)
            {
                const T old=x[c][axis];x[c][axis]=old+eps;
                const auto plus=avbdSampleNodalVolume(rest,x,mu,lambda,c);
                x[c][axis]=old-eps;const auto minus=avbdSampleNodalVolume(rest,x,mu,lambda,c);x[c][axis]=old;
                const T fd=(plus.energy-minus.energy)/(2*eps);
                const T ge=PxAbs(fd-actual.gradient[axis])/(1+PxAbs(fd));
                const V hd=(plus.gradient-minus.gradient)/(2*eps);
                const T he=(hd-actual.metric[axis]).magnitude()/(1+hd.magnitude());
                worstGradient=PxMax(worstGradient,ge);worstHessian=PxMax(worstHessian,he);
                failures+=ge>tolerance || he>tolerance;
            }
            V step(0);failures+=!avbdSolveNodal3(actual.metric,actual.gradient,step);
            failures+=(actual.metric*step+actual.gradient).magnitude()>tolerance*(1+actual.gradient.magnitude());
        }
        const T weights[4]={T(.3),T(-.7),T(.2),T(.1)};
        const auto relative=avbdSampleNodalVolume(rest,x,mu,lambda,PX_MAX_U32,weights);
        V independent(0);for(PxU32 c=0;c<4;++c)independent+=avbdSampleNodalVolume(rest,x,mu,lambda,c).gradient*weights[c];
        failures+=!relative.valid || (relative.gradient-independent).magnitude()>tolerance;
        for(PxU32 axis=0;axis<3;++axis)
        {
            V saved[4];for(PxU32 c=0;c<4;++c){saved[c]=x[c];x[c][axis]+=eps*weights[c];}
            const auto plus=avbdSampleNodalVolume(rest,x,mu,lambda,PX_MAX_U32,weights);
            for(PxU32 c=0;c<4;++c)x[c][axis]=saved[c][axis]-eps*weights[c];
            const auto minus=avbdSampleNodalVolume(rest,x,mu,lambda,PX_MAX_U32,weights);
            for(PxU32 c=0;c<4;++c)x[c]=saved[c];
            const T finite=(plus.energy-minus.energy)/(2*eps);
            failures+=PxAbs(finite-relative.gradient[axis])/(1+PxAbs(finite))>tolerance;
            const V hd=(plus.gradient-minus.gradient)/(2*eps);
            failures+=(hd-relative.metric[axis]).magnitude()/(1+hd.magnitude())>tolerance;
        }
        failures+=net.magnitude()>tolerance || torque.magnitude()>tolerance;
        // A rotated/translated rest pose has zero strain-rate work. For a
        // deformed pose test the exact, finite one-node line-search work.
        const M startMetric(PxIdentity);
        const auto damped=[&](PxU32 c,AvbdNodalWorkT<T>* terms=NULL)
        {return avbdSampleNodalVolume(rest,x,mu,lambda,c,static_cast<const T*>(NULL),&startMetric,T(.6),terms);};
        V dampedNet(0),dampedTorque(0);
        for(PxU32 c=0;c<4;++c)
        {
            AvbdNodalWorkT<T> terms;
            const auto sample=damped(c,&terms);const V saved=x[c],dx(T(.073),T(-.041),T(.129));
            dampedNet+=sample.gradient;dampedTorque+=x[c].cross(sample.gradient);
            const T r2=dx.magnitudeSquared();
            const T predicted=dx.dot(sample.gradient+(sample.metric*dx+dx*terms.geometric)*T(.5))+
                r2*(terms.cubic.dot(dx)+terms.quartic*r2);
            x[c]+=dx;
            failures+=!sample.valid || PxAbs(predicted-(damped(c).energy-sample.energy))>tolerance;
            x[c]=saved;
        }
        failures+=dampedNet.magnitude()>tolerance || dampedTorque.magnitude()>tolerance;
        const T viscousBefore=damped(PX_MAX_U32).energy;
        const T before=avbdSampleNodalVolume(rest,x,mu,lambda,PX_MAX_U32).energy;
        for(PxU32 c=0;c<4;++c)x[c]=V(-x[c].y,x[c].x,x[c].z)+V(3,-2,1);
        failures+=PxAbs(avbdSampleNodalVolume(rest,x,mu,lambda,PX_MAX_U32).energy-before)>tolerance;
        failures+=PxAbs(damped(PX_MAX_U32).energy-viscousBefore)>tolerance;
        for(PxU32 i=0;i<4;++i)x[i]=V(-reference[i].y,reference[i].x,reference[i].z)+V(3,-2,1);
        failures+=PxAbs(damped(0).energy)>tolerance || damped(0).gradient.magnitude()>tolerance;
    }
    // Recover the small-strain uniaxial modulus lambda+2*mu, independently of
    // the implementation's volume coefficient. This catches an unshifted lambda.
    const T strain=T(.001);
    for(PxU32 i=0;i<4;++i){x[i]=reference[i];x[i].x*=1+strain;}
    T work=0;for(PxU32 i=0;i<4;++i)work+=avbdSampleNodalVolume(rest,x,mu,lambda,i).gradient.x*reference[i].x;
    failures+=PxAbs(work/(rest.volume*strain)-(lambda+2*mu))>T(.01);
    // One-point determinant is affine: a positive-to-positive local update
    // stays positive; an inverted endpoint must be rejected.
    for(PxU32 i=0;i<4;++i)x[i]=reference[i];
    x[3]=x[0]-(reference[3]-reference[0]);
    failures+=avbdSampleNodalVolume(rest,x,mu,lambda,3).valid;
    V end[4];for(PxU32 i=0;i<4;++i){x[i]=reference[i];end[i]=reference[i]+V(2,-1,3);}
    failures+=!avbdNodalVolumePath(rest,x,end);
    // Both endpoints have J>0, yet this 180-degree interpolation collapses
    // internally. Endpoint-only admission would incorrectly publish it.
    for(PxU32 i=0;i<4;++i)end[i]=V(-x[i].x,-x[i].y,x[i].z);
    failures+=!avbdSampleNodalVolume(rest,end,mu,lambda,0).valid;
    failures+=avbdNodalVolumePath(rest,x,end);
    for(PxU32 i=0;i<4;++i){x[i]=reference[i];end[i]=x[i];end[i].y*=T(.025);}
    failures+=avbdNodalVolumePath(rest,x,end);
    for(PxU32 i=0;i<4;++i){x[i]=end[i];end[i].y*=T(3);}
    failures+=!avbdNodalVolumePath(rest,x,end);
    for(PxU32 i=0;i<4;++i)end[i].y=x[i].y*T(.8);
    failures+=avbdNodalVolumePath(rest,x,end);
    x[3]=V(std::numeric_limits<T>::quiet_NaN());
    failures+=avbdSampleNodalVolume(rest,x,mu,lambda,3).valid;
    std::printf("[NODAL_VOLUME_KERNEL] scalarBytes=%u worstGradient=%.9g worstHessian=%.9g failures=%u\n",
        unsigned(sizeof(T)),double(worstGradient),double(worstHessian),failures);
    return failures;
}

template<class T> static bool checkPlaneStep()
{
    using V=PxVec3T<T>;using M=PxMat33T<T>;
    const M H(V(4,1,0),V(1,2,0),V(0,0,3));
    const V normal(0,1,0),g(-4,2,-3);const T eps=T(1.e-5);
    AvbdNodalPlaneRangeT<T> ranges[2],upper;
    ranges[0].include(normal,T(.5),T(0));
    upper.include(normal,T(-.25),T(.05));ranges[0].merge(upper);
    AvbdNodalPlaneRangeT<T> opposite;
    opposite.include(-normal,T(.25),T(.025));ranges[0].merge(opposite);
    if(PxAbs(ranges[0].high-T(.1))>eps)return false;
    V step;
    if(!avbdSolveNodal3(H,g,step,&ranges[0]) || (step-V(1,0,1)).magnitude()>eps)return false;
    // Independent lower and upper faces at two coupled endpoints. Choose the
    // gradient from a known optimum and compressive KKT multipliers.
    const M B=M(PxIdentity)*T(.35);
    ranges[1].include(normal,T(-1),T(.1));
    const V x(1,0,.2),y(-.75,.1,-.3);
    const V ga=-(H*x+B*y)+normal*T(2),gb=-(B*x+H*y)-normal*T(3);
    V a,b;
    if(!avbdSolveNodalPair(H,B,H,ga,gb,a,b,ranges) || (a-x).magnitude()>eps || (b-y).magnitude()>eps)return false;
    // Only one endpoint touches the plane. Its normal reaction must also
    // move the other endpoint through the off-diagonal stiffness.
    ranges[1]=AvbdNodalPlaneRangeT<T>();
    if(!avbdSolveNodalPair(H,B,H,ga,-(B*x+H*y),a,b,ranges) ||
        (a-x).magnitude()>eps || (b-y).magnitude()>eps)return false;
    // On release the same solver must keep the free inward-to-outward motion.
    const V release(.1,.075,0);
    if(!avbdSolveNodal3(H,-(H*release),step,&ranges[0]) || (step-release).magnitude()>eps)return false;
    // A nonparallel corner goes through the full geometric line search.
    ranges[0].include(V(1,0,0),T(1),T(0));
    V freeStep;
    return !ranges[0].parallel && avbdSolveNodal3(H,g,freeStep) &&
        avbdSolveNodal3(H,g,step,&ranges[0]) && (freeStep-step).magnitude()<eps;
}

int main()
{
    AvbdProbeFoundation foundation;unsigned failures=verify<float>()+verify<double>();
    const bool plane=checkPlaneStep<float>() && checkPlaneStep<double>();
    failures+=!plane;
    std::printf("[NODAL_PLANE_STEP] tangentReleaseAndCoupledBounds=%u\n",unsigned(plane));
    const AvbdNodalVec3 x[4]={AvbdNodalVec3(0),AvbdNodalVec3(1,0,0),AvbdNodalVec3(0,1,0),AvbdNodalVec3(0,0,1)};
    const auto rest=cookRest(x);AvbdNodalShellSampleT<AvbdNodalReal> sample;
    AvbdNodalVolumeArgs args={&rest,x,7,4,0,&sample};
    KernelTable::makeAvbdCpuScalar().launch(AVBD_NODAL_VOLUME,&args);
    failures+=!sample.valid || sample.gradient.magnitude()>AvbdNodalReal(1.e-5);
    std::printf("[NODAL_VOLUME_TABLE] scalarBytes=%u status=%s\n",unsigned(sizeof(AvbdNodalReal)),failures?"FAIL":"PASS");
    return failures?1:0;
}
