// Mechanism counterexamples, not scene qualification or a second simulator.
#include "avbd/DyAvbdCommonResponse.h"
#include "avbd/DyAvbdNpDomain.h"
#include "avbd/DyAvbdCommonResidual.h"
#include "avbd/DyAvbdRigidMetric.h"
#include "avbd_probe_foundation.h"
#include <cstdio>
using namespace physx;using namespace physx::Dy;

static AvbdCommonProgram program(PxU32 blocks)
{
    AvbdCommonProgram p;
    for(PxU32 b=0;b<blocks;++b)
    {
        p.dimensions.pushBack(6);p.sourceBlocks.pushBack(b);
        AvbdCommonHBlock h={};h.row=h.column=b;
        for(PxU32 d=0;d<6;++d){p.base.pushBack(0);h.values[6*d+d]=1;}
        p.H.pushBack(h);
    }
    return p;
}
static PxU32 normal(AvbdCommonProgram& p)
{
    const PxU32 r=p.mu.size();p.sourceRows.pushBack(r);p.mu.pushBack(0);p.coulomb.pushBack(0);
    for(PxU32 q=0;q<3;++q)
    {p.offset.pushBack(0);p.lower.pushBack(0);p.upper.pushBack(q?0:std::numeric_limits<double>::infinity());p.compliance.pushBack(0);}
    return r;
}
int main()
{
    AvbdProbeFoundation foundation;const KernelTable table=KernelTable::makeAvbdCpuScalar();bool pass=true;
    AvbdNpDomain domain;domain.normal=PxVec3d(0,1,0);domain.radius[0]=domain.radius[1]=0;
    domain.points[0].pushBack(PxVec3d(-1,-1,0));domain.points[0].pushBack(PxVec3d(1,-1,0));
    domain.points[1].pushBack(PxVec3d(0));
    const PxTransform initial[2]={PxTransform(PxVec3(0,1,0)),PxTransform(PxIdentity)};
    for(PxU32 order=0;order<2;++order)
    {
        PxVec3d arms[2];avbdNpGapGradient(domain,initial,arms);
        auto one=program(1);one.base[1]=-1.e-4;
        AvbdCommonJBlock j={};j.row=normal(one);j.values[1]=1;j.values[5]=arms[0].cross(domain.normal).z;one.J.pushBack(j);
        pass=one.finish() && pass;AvbdCommonResponse response(one,table);PxArray<double> dx,f;AvbdCoulombNcpResult result;
        const bool solved=response.solve(dx,f,result);
        if(!solved){pass=false;continue;}
        PxTransform candidate[2]={initial[0],initial[1]};candidate[0].p.y+=PxReal(dx[1]);
        candidate[0].q=PxQuat(0,0,PxReal(.5*dx[5]),1).getNormalized();
        const double singleGap=avbdNpGap(domain,candidate);
        const bool singleCertified=avbdNpDomainPath(domain,initial,candidate);
        // The other active feature has the opposite moment arm. Both enter
        // the SAME 6x6 material operator and original bounded NCP kernel.
        auto bundle=program(1);bundle.base[1]=-1.e-4;
        const bool rotates[2]={true,false};avbdNpCompileFeatures(domain,initial,rotates,0);
        for(PxU32 v=0;v<domain.count[0]*domain.count[1];++v)
        {
            PxVec3d featureArms[2];const double gap=avbdNpFeatureGradient(domain,initial,v,featureArms);
            j.row=normal(bundle);bundle.offset[3*j.row]=gap;
            j.values[5]=featureArms[0].cross(domain.normal).z;bundle.J.pushBack(j);
        }
        pass=bundle.finish() && pass;AvbdCommonResponse bundleResponse(bundle,table);
        const bool bundleSolved=bundleResponse.solve(dx,f,result);
        const double dy=dx.size()>=6?dx[1]:1,angle=dx.size()>=6?dx[5]:1;
        const double force0=f.size()>=6?f[0]:0,force1=f.size()>=6?f[3]:0;
        const bool ok=solved && singleGap<-9.e-5 && !singleCertified && bundleSolved &&
            std::abs(dy)<1.e-12 && std::abs(angle)<1.e-12 && std::abs(force0-5.e-5)<1.e-12 && std::abs(force1-5.e-5)<1.e-12;
        pass=ok && pass;
        std::printf("[STAGNATION_FACE] order=%u frozenSingleSolved=%u singleGap=%.17g singleCertified=%u bundleDy=%.17g bundleAngle=%.17g force0=%.17g force1=%.17g status=%s\n",
            order,unsigned(solved),singleGap,unsigned(singleCertified),dy,angle,force0,force1,ok?"PASS":"FAIL");
        const PxVec3d point=domain.points[0][0];domain.points[0][0]=domain.points[0][1];domain.points[0][1]=point;
    }
    // A closed dynamic contact can also trap individually feasible GS. Keep
    // the off-diagonal material coupling: this is not a block-diagonal oracle.
    auto p=program(2);p.H[0].values[0]=p.H[1].values[0]=2;
    AvbdCommonHBlock cross={};cross.row=0;cross.column=1;cross.values[0]=-1;p.H.pushBack(cross);
    cross.row=1;cross.column=0;p.H.pushBack(cross);p.base[0]=2;p.base[6]=-1;
    const PxU32 row=normal(p);
    for(PxU32 b=0;b<2;++b){AvbdCommonJBlock j={};j.row=row;j.column=b;j.values[0]=b?1:-1;p.J.pushBack(j);}
    pass=p.finish() && pass;AvbdCommonResponse response(p,table);PxArray<double> dx,f;AvbdCoulombNcpResult result;
    const bool solved=response.solve(dx,f,result);double a=0,b=0;
    for(PxU32 sweep=0;sweep<16;++sweep){a=std::min((2+b)/2,b);b=std::max((-1+a)/2,a);}
    const bool ok=solved && a==0 && b==0 && std::abs(dx[0]-.5)<1.e-12 && std::abs(dx[6]-.5)<1.e-12 && std::abs(f[0]-1.5)<1.e-12;
    pass=ok && pass;
    std::printf("[STAGNATION_GROUP] singleA=%.17g singleB=%.17g groupA=%.17g groupB=%.17g force=%.17g status=%s\n",
        a,b,solved?dx[0]:0,solved?dx[6]:0,solved?f[0]:0,ok?"PASS":"FAIL");
    // Positive GN metric at negative true curvature: the old squared-root
    // merit rejects ALL sixteen energy-descending backtracks.
    auto energy=[](double x){return .25*(x*x-1)*(x*x-1)+.05*(x-.01)*(x-.01);};
    auto gradient=[](double x){return x*(x*x-1)+.1*(x-.01);};
    auto from=program(1);from.base[0]=-gradient(.01);from.H[0].values[0]=.1002;
    from.materialEnergies.pushBack(energy(.01));pass=from.finish() && pass;
    AvbdCommonResidual merit(from);double none=0;const double old=merit.evaluate(from,&none);
    unsigned accepted=0,rejectedByOldMerit=0;
    for(PxU32 i=0;i<16;++i)
    {
        const double move=from.base[0]/.1002*std::ldexp(1.,-int(i));double delta[6]={move,0,0,0,0,0};
        auto to=from;to.base[0]=-gradient(.01+move);to.materialEnergies[0]=energy(.01+move);double work,slope;
        accepted+=AvbdCommonResidual::workDescent(from,to,&none,delta,work,slope);
        rejectedByOldMerit+=merit.evaluate(to,&none)>old;
    }
    auto uphill=from;uphill.materialEnergies[0]=energy(-.005);double wrong[6]={-.015,0,0,0,0,0},work,slope;
    const bool rejectsUphill=!AvbdCommonResidual::workDescent(from,uphill,&none,wrong,work,slope);
    const bool meritOk=accepted==16 && rejectedByOldMerit==16 && rejectsUphill;pass=pass && meritOk;
    std::printf("[STAGNATION_WORK] accepted=%u oldMeritRejected=%u uphillRejected=%u status=%s\n",
        accepted,rejectedByOldMerit,unsigned(rejectsUphill),meritOk?"PASS":"FAIL");
    // Over-majorized material: unit metric has twice the actual curvature.
    // The spectral proposal uses displacement units, so changing metres to
    // millimetres cannot hide a weak rotational mode behind stiff translation.
    double secantStep[2]={2.e-4,3.e-5},secantCorrection[2]={1.e-4,1.5e-5},secantTolerance[2]={1.e-4,1.e-5};
    const double spectral=AvbdCommonResidual::secantStart(secantStep,secantCorrection,secantTolerance,2);
    secantStep[0]*=1000;secantCorrection[0]*=1000;secantTolerance[0]*=1000;
    const double rescaled=AvbdCommonResidual::secantStart(secantStep,secantCorrection,secantTolerance,2);
    auto quadratic=program(1);quadratic.base[0]=1;quadratic.materialEnergies.pushBack(1);
    pass=quadratic.finish() && pass;auto minimum=quadratic;minimum.base[0]=0;minimum.materialEnergies[0]=0;
    double extended[6]={spectral,0,0,0,0,0};
    const bool extensionDescent=AvbdCommonResidual::workDescent(quadratic,minimum,&none,extended,work,slope);
    auto overshot=quadratic;overshot.base[0]=-2;overshot.materialEnergies[0]=4;extended[0]=6;
    const bool rejectsOvershoot=!AvbdCommonResidual::workDescent(quadratic,overshot,&none,extended,work,slope);
    double s1=1,c1=2,t1=1;
    const bool noNegative=AvbdCommonResidual::secantStart(&s1,&c1,&t1,1)==1;
    c1=0;const bool noExact=AvbdCommonResidual::secantStart(&s1,&c1,&t1,1)==1;
    s1=0;const bool noZero=AvbdCommonResidual::secantStart(&s1,&c1,&t1,1)==1;
    const bool secantOk=std::abs(spectral-2)<1.e-14 && spectral==rescaled && extensionDescent && rejectsOvershoot && noNegative && noExact && noZero;
    pass=pass && secantOk;
    std::printf("[STAGNATION_SPECTRAL] scale=%.9g unitInvariant=%u descent=%u overshootRejected=%u status=%s\n",
        spectral,unsigned(spectral==rescaled),unsigned(extensionDescent),unsigned(rejectsOvershoot),secantOk?"PASS":"FAIL");
    auto shifted=from;shifted.materialEnergies.pushBack(1.e16);auto reduced=shifted;
    reduced.materialEnergies[0]=energy(.02);double move[6]={.01,0,0,0,0,0};
    const bool shiftOk=AvbdCommonResidual::workDescent(shifted,reduced,&none,move,work,slope) &&
        reduced.materialEnergies[0]+1.e16==shifted.materialEnergies[0]+1.e16;
    pass=pass && shiftOk;
    std::printf("[STAGNATION_ENERGY_SUM] globalDifferenceErased=1 localWork=%.17g status=%s\n",work,shiftOk?"PASS":"FAIL");
    auto screening=program(1);screening.base[0]=-1;
    AvbdCommonJBlock first={};first.row=normal(screening);first.values[0]=first.values[1]=1;screening.J.pushBack(first);
    AvbdCommonJBlock second={};second.row=normal(screening);second.values[1]=-1;
    screening.offset[3*second.row]=.25;screening.J.pushBack(second);
    pass=screening.finish() && pass;AvbdCommonResponse screened(screening,table);
    const bool screenedSolved=screened.solve(dx,f,result);
    const bool screenOk=screenedSolved && screened.activeGroups==2 && std::abs(dx[0]+.25)<1.e-12 &&
        std::abs(dx[1]-.25)<1.e-12 && std::abs(f[0]-.75)<1.e-12 && std::abs(f[3]-.5)<1.e-12;
    pass=pass && screenOk;
    std::printf("[STAGNATION_SCREEN] activeGroups=%u newlyClosingConstraint=%s status=%s\n",
        screened.activeGroups,screenOk?"included":"unresolved",screenOk?"PASS":"FAIL");
    AvbdNpDomain pathDomain;pathDomain.normal=PxVec3d(0,1,0);pathDomain.radius[0]=pathDomain.radius[1]=0;
    pathDomain.points[0].pushBack(PxVec3d(0,-1,0));pathDomain.points[1].pushBack(PxVec3d(0));
    PxTransform pathStart[2]={PxTransform(PxVec3(0,.9f,0),PxQuat(-.5f,PxVec3(0,0,1))),PxTransform(PxIdentity)};
    PxTransform pathEnd[2]={PxTransform(PxVec3(0,.9f,0),PxQuat(.5f,PxVec3(0,0,1))),PxTransform(PxIdentity)};
    const bool moving[2]={true,true};avbdNpCompileFeatures(pathDomain,pathStart,moving,0);
    const bool endpointsValid=avbdNpGap(pathDomain,pathStart)>0 && avbdNpGap(pathDomain,pathEnd)>0;
    const bool crossingRejected=!avbdNpDomainPath(pathDomain,pathStart,pathEnd);
    pathEnd[0].p.y=1.4f;const bool liftedAccepted=avbdNpDomainPath(pathDomain,pathStart,pathEnd);
    double derivativeError=0,endpointError=0;
    for(PxU32 test=0;test<24;++test)
    {
        const PxVec3 axis=PxVec3(1.f+test,.3f+test%3,.7f).getNormalized();
        pathEnd[0].q=PxQuat(.02f*test,axis);pathEnd[1].q=PxQuat(-.015f*test,axis);
        pathDomain.points[1][0]=PxVec3d(.17,.23,-.11);
        const auto value=avbdNpPathFeature(pathDomain,pathStart,pathEnd,0);pass=pass && value.valid;
        PxVec3d arms[2];endpointError=std::max(endpointError,std::abs(value.gap[4]-avbdNpFeatureGradient(pathDomain,pathEnd,0,arms)));
        for(PxU32 side=0;side<2;++side)for(PxU32 d=0;d<3;++d)
        {
            PxVec3 unit(0);unit[d]=1;PxTransform plus[2]={pathEnd[0],pathEnd[1]},minus[2]={pathEnd[0],pathEnd[1]};
            plus[side].q=PxQuat(.002f,unit)*plus[side].q;minus[side].q=PxQuat(-.002f,unit)*minus[side].q;
            const auto a=avbdNpPathFeature(pathDomain,pathStart,plus,0,false),b=avbdNpPathFeature(pathDomain,pathStart,minus,0,false);
            for(PxU32 k=0;k<5;++k)derivativeError=std::max(derivativeError,
                std::abs((a.gap[k]-b.gap[k])/.004-value.angular[k][side][d]));
        }
    }
    const bool pathOk=endpointsValid && crossingRejected && liftedAccepted && derivativeError<2.e-4 && endpointError<1.e-12;
    pass=pass && pathOk;
    std::printf("[STAGNATION_PATH] endpointOnlyMissesCrossing=%u crossingRejected=%u liftAccepted=%u derivativeError=%.9g endpointError=%.9g status=%s\n",
        unsigned(endpointsValid),unsigned(crossingRejected),unsigned(liftedAccepted),derivativeError,endpointError,pathOk?"PASS":"FAIL");
    const PxVec3 center(0,10,0),local(0,0,1);const PxQuat q0(.01f,PxVec3(1,0,0)),q1(.0100001f,PxVec3(1,0,0));
    const double oldSlip=double((center+q1.rotate(local)).y-(center+q0.rotate(local)).y);
    const double directSlip=(avbdNpPointMotion(center,q1,center,PxQuat(PxIdentity),local)-
        avbdNpPointMotion(center,q0,center,PxQuat(PxIdentity),local)).y;
    const PxQuatd d=q1==q0?PxQuatd(PxIdentity):avbdNpRotation(q1)*avbdNpRotation(q0).getConjugate();
    const double angle=2*std::atan2(d.x,d.w),linear=-avbdNpRotation(q0).rotate(avbdOgcDouble(local)).z*angle;
    const double newError=std::abs((directSlip-linear)/linear),oldError=std::abs((oldSlip-linear)/linear);
    const bool slipOk=newError<1.e-5 && oldError>.1;pass=pass && slipOk;
    std::printf("[STAGNATION_SLIP] old=%.17g direct=%.17g linear=%.17g oldRelativeError=%.9g newRelativeError=%.9g status=%s\n",
        oldSlip,directSlip,linear,oldError,newError,slipOk?"PASS":"FAIL");
    PxU32 oldZeroMoves=0;double maxOldAngle=0;bool zeroStable=true;
    for(PxU32 i=0;i<1000;++i)
    {
        const PxQuat stored(PxReal(.003*i),PxVec3(1.f+float(i%7),.3f,.7f).getNormalized());
        PxQuat renormalized=stored;renormalized.normalize();
        const double drift=avbdAcceptedRotationDisplacementDouble(stored,renormalized).magnitude();
        oldZeroMoves+=drift>1.e-12;maxOldAngle=std::max(maxOldAngle,drift);
        zeroStable=zeroStable && avbdStoredRotationIncrement(stored,PxVec3d(0))==stored;
    }
    const bool rotationOk=zeroStable && oldZeroMoves>0;pass=pass && rotationOk;
    std::printf("[STAGNATION_ROTATION] cases=1000 oldZeroMoves=%u maxOldAngle=%.9g zeroStable=%u status=%s\n",
        oldZeroMoves,maxOldAngle,unsigned(zeroStable),rotationOk?"PASS":"FAIL");
    return pass?0:1;
}
