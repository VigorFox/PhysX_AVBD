// Mechanism checks for the production full-domain path coefficients/minimum.
#include "avbd/DyAvbdNpDomain.h"
#include "avbd/DyKernelTable.h"
#include "avbd_probe_foundation.h"
#include <random>
#include <cstdio>
#include <fstream>
using namespace physx;using namespace physx::Dy;
// Independent oracle: reconstruct the Bernstein curve using quaternion norm
// products, then compare with directly rotated points along the actual path.
static unsigned pathOracle(std::mt19937& rng)
{
    std::normal_distribution<double> random;unsigned failures=0,samples=0,derivatives=0;
    double pathError=0,gradientError=0;
    const double choose2[3]={1,2,1},choose4[5]={1,4,6,4,1},choose5[6]={1,5,10,10,5,1};
    for(unsigned test=0;test<80;++test)
    {
        AvbdNpDomain domain;domain.normal=PxVec3d(random(rng),random(rng),random(rng)).getNormalized();
        domain.radius[0]=.1;domain.radius[1]=.2;domain.restDistance=.01;
        PxTransform start[2],end[2];PxQuatd a[2],b[2];double norms[2][3];
        for(unsigned s=0;s<2;++s)
        {
            domain.count[s]=1;domain.points[s].pushBack(PxVec3d(random(rng),random(rng),random(rng)));
            const PxVec3 axis=PxVec3(float(random(rng)),float(random(rng)),float(random(rng))).getNormalized();
            start[s]=PxTransform(PxVec3(float(random(rng)),float(random(rng)),float(random(rng))),PxQuat(float(random(rng)),axis));
            end[s]=PxTransform(PxVec3(float(random(rng)),float(random(rng)),float(random(rng))),PxQuat(float(random(rng)),axis));
            a[s]=avbdNpRotation(start[s].q);b[s]=avbdNpRotation(end[s].q);if(a[s].dot(b[s])<0)b[s]=-b[s];
            norms[s][0]=a[s].dot(a[s]);norms[s][1]=a[s].dot(b[s]);norms[s][2]=b[s].dot(b[s]);
        }
        const AvbdNpPathPolynomial polynomial(domain,start,end);const auto f=polynomial.evaluate(domain,0);
        failures+=!f.valid;
        double product[5]={},control[6];
        for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)
            product[i+j]+=choose2[i]*choose2[j]/choose4[i+j]*norms[0][i]*norms[1][j];
        auto physical=[&](double t)
        {
            PxVec3d point[2];
            for(unsigned s=0;s<2;++s)point[s]=avbdOgcDouble(start[s].p)*(1-t)+avbdOgcDouble(end[s].p)*t+
                (a[s]*(1-t)+b[s]*t).getNormalized().rotate(domain.points[s][0]);
            return domain.normal.dot(point[0]-point[1])-domain.radius[0]-domain.radius[1]-domain.restDistance;
        };
        control[0]=physical(0)*product[0];
        for(unsigned k=1;k<=5;++k)control[k]=f.gap[k-1]*double(k)/5*product[k-1];
        for(unsigned sample=1;sample<=9;++sample)
        {
            const double t=double(sample)/10;double value=0,denominator=1;
            for(unsigned k=0;k<=5;++k)value+=choose5[k]*std::pow(t,int(k))*std::pow(1-t,int(5-k))*control[k];
            for(unsigned s=0;s<2;++s){const auto q=a[s]*(1-t)+b[s]*t;denominator*=q.dot(q);}
            const double expected=physical(t),error=std::abs(value/denominator-expected)/(1+std::abs(expected));
            pathError=std::max(pathError,error);failures+=error>1.e-11;++samples;
        }
        for(unsigned s=0;s<2;++s)for(unsigned d=0;d<3;++d)
        {
            PxVec3 axis(0);axis[d]=1;const float epsilon=.002f;
            PxTransform plus[2]={end[0],end[1]},minus[2]={end[0],end[1]};
            plus[s].q=PxQuat(epsilon,axis)*end[s].q;minus[s].q=PxQuat(-epsilon,axis)*end[s].q;
            const auto p=avbdNpPathFeature(domain,start,plus,0,false),m=avbdNpPathFeature(domain,start,minus,0,false);
            for(unsigned k=0;k<5;++k)
            {
                const double expected=f.angular[k][s][d],observed=(p.gap[k]-m.gap[k])/(2*epsilon);
                const double error=std::abs(observed-expected)/(1+std::abs(expected));
                gradientError=std::max(gradientError,error);failures+=error>2.e-4;++derivatives;
            }
        }
    }
    std::printf("[NP_PATH_ORACLE] samples=%u derivatives=%u pathError=%.12g gradientError=%.12g failures=%u\n",samples,derivatives,pathError,gradientError,failures);
    return failures;
}
static unsigned transportedOracle(std::mt19937& rng)
{
    std::normal_distribution<double> random;unsigned failures=0,samples=0;double worst=0;
    const double choose2[3]={1,2,1},choose4[5]={1,4,6,4,1},choose5[6]={1,5,10,10,5,1};
    for(unsigned test=0;test<80;++test)
    {
        AvbdNpDomain d;d.normal=PxVec3d(random(rng),random(rng),random(rng)).getNormalized();
        d.radius[0]=.1;d.radius[1]=.2;d.restDistance=.01;d.count[0]=d.count[1]=1;
        PxTransform start[2],end[2];PxQuatd a[2],b[2];double norms[2][3];
        for(unsigned s=0;s<2;++s)
        {
            d.points[s].pushBack(PxVec3d(random(rng),random(rng),random(rng)));
            const PxVec3 axis=PxVec3(float(random(rng)),float(random(rng)),float(random(rng))).getNormalized();
            start[s]=PxTransform(PxVec3(float(random(rng)),float(random(rng)),float(random(rng))),PxQuat(float(random(rng)),axis));
            end[s]=PxTransform(PxVec3(float(random(rng)),float(random(rng)),float(random(rng))),PxQuat(float(random(rng)),axis));
            a[s]=avbdNpRotation(start[s].q);b[s]=avbdNpRotation(end[s].q);if(a[s].dot(b[s])<0)b[s]=-b[s];
            norms[s][0]=a[s].dot(a[s]);norms[s][1]=a[s].dot(b[s]);norms[s][2]=b[s].dot(b[s]);
        }
        double product[5]={};for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)
            product[i+j]+=choose2[i]*choose2[j]/choose4[i+j]*norms[0][i]*norms[1][j];
        for(unsigned frame=0;frame<2;++frame)
        {
            const auto p=avbdNpTransportedPolynomial(d,start,end,frame);const auto f=p.evaluate(d,0);
            failures+=!f.valid;const PxVec3d local=a[frame].rotateInv(d.normal);
            auto physical=[&](double t)
            {
                PxVec3d point[2];PxQuatd q[2];
                for(unsigned s=0;s<2;++s){q[s]=(a[s]*(1-t)+b[s]*t).getNormalized();
                    point[s]=avbdOgcDouble(start[s].p)*(1-t)+avbdOgcDouble(end[s].p)*t+q[s].rotate(d.points[s][0]);}
                return q[frame].rotate(local).dot(point[0]-point[1])-.31;
            };
            double control[6];control[0]=physical(0)*product[0];
            for(unsigned k=1;k<=5;++k)control[k]=f.gap[k-1]*double(k)/5*product[k-1];
            for(unsigned sample=0;sample<=10;++sample)
            {
                const double t=double(sample)/10;double value=0,denominator=1;
                for(unsigned k=0;k<=5;++k)value+=choose5[k]*std::pow(t,int(k))*std::pow(1-t,int(5-k))*control[k];
                for(unsigned s=0;s<2;++s){const auto q=a[s]*(1-t)+b[s]*t;denominator*=q.dot(q);}
                const double expected=physical(t),error=std::abs(value/denominator-expected)/(1+std::abs(expected));
                worst=std::max(worst,error);failures+=error>1.e-11;++samples;
            }
        }
    }
    // Actual failed cloth pair: its overhanging corner crosses the OLD axis,
    // while the sheet remains separated from the finite box throughout.
    AvbdNpDomain d;d.normal=PxVec3d(0,.9999642907810106,.008450867578).getNormalized();
    d.radius[0]=d.radius[1]=0;d.count[0]=d.count[1]=8;
    for(unsigned i=0;i<8;++i){d.points[0].pushBack(PxVec3d(i&1?.400000006:-.400000006,i&2?.00999999978:-.00999999978,i&4?.400000006:-.400000006));
        d.points[1].pushBack(PxVec3d(i&1?1.149999976:-1.149999976,i&2?1:-1,i&4?1.149999976:-1.149999976));}
    PxTransform start[2]={PxTransform(PxVec3(.400035739f,2.06694961f,1.20017719f),PxQuat(.00422547013f,-.0000122279389f,.000113091679f,.999991059f)),PxTransform(PxVec3(0,1,0))};
    PxTransform end[2]={PxTransform(PxVec3(.400452226f,2.04114914f,1.20164216f),PxQuat(.0492620207f,-.000379098463f,.00114850421f,.998785257f)),start[1]};
    const auto fixed=avbdNpAxisQuery(d,start,end,PX_MAX_U32),moving=avbdNpCompleteQuery(d,start,end);
    failures+=fixed.accepted || !moving.accepted || moving.axisFrame!=0;
    // Swap endpoint ordering: the same certificate must exist on side one.
    AvbdNpDomain swapped=d;swapped.normal=-d.normal;swapped.points[0]=d.points[1];swapped.points[1]=d.points[0];
    const PxTransform swappedStart[2]={start[1],start[0]},swappedEnd[2]={end[1],end[0]};
    const auto second=avbdNpCompleteQuery(swapped,swappedStart,swappedEnd);failures+=!second.accepted || second.axisFrame!=1;
    d.fixedAxis=true;failures+=avbdNpCompleteQuery(d,start,end).accepted;d.fixedAxis=false;
    // A true terminal overlap cannot be certified by changing the axis.
    PxTransform overlap[2]={end[0],end[1]};overlap[0].p=overlap[1].p;
    failures+=avbdNpCompleteQuery(d,start,overlap).accepted;
    failures+=avbdNpCompleteQuery(d,overlap,start).accepted;
    AvbdNpDomain crossing;crossing.normal=PxVec3d(-1,0,0);crossing.radius[0]=crossing.radius[1]=.1;
    crossing.points[0].pushBack(PxVec3d(0));crossing.points[1].pushBack(PxVec3d(0));
    PxTransform crossStart[2]={PxTransform(PxVec3(-1,0,0)),PxTransform(PxIdentity)};
    PxTransform crossEnd[2]={PxTransform(PxVec3(1,0,0),PxQuat(3.f,PxVec3(0,0,1))),PxTransform(PxIdentity)};
    // Both ends are separated on the transported axis, but the midpoint
    // coincides. Endpoint-only SAT would incorrectly accept this path.
    failures+=crossEnd[0].q.rotate(PxVec3(-1,0,0)).dot(crossEnd[0].p)<=.2f;
    failures+=avbdNpCompleteQuery(crossing,crossStart,crossEnd).accepted;
    // A thin rod turns through a sphere. Both endpoints separate on the
    // selected diagonal, but the middle of the path physically intersects.
    AvbdNpDomain rod;rod.normal=PxVec3d(0,-1,0);rod.radius[0]=0;rod.radius[1]=.1;
    for(unsigned i=0;i<8;++i)rod.points[0].pushBack(PxVec3d(i&1?1:-1,i&2?.01:-.01,i&4?.01:-.01));
    rod.points[1].pushBack(PxVec3d(0));
    PxTransform rodStart[2]={PxTransform(PxIdentity),PxTransform(PxVec3(.6f,.6f,0),PxQuat(PxPi/4,PxVec3(0,0,1)))};
    PxTransform rodEnd[2]={PxTransform(PxVec3(0),PxQuat(PxPi/2,PxVec3(0,0,1))),rodStart[1]};
    const auto rodQuery=avbdNpCompleteQuery(rod,rodStart,rodEnd);
    const PxVec3d diagonal=PxVec3d(-1,-1,0).getNormalized();
    for(unsigned k=0;k<2;++k)
    {const auto* pose=k?rodEnd:rodStart;failures+=avbdNpSupportRange(rod,0,pose[0],&diagonal).lo-
        avbdNpSupportRange(rod,1,pose[1],&diagonal).hi<=.03;}
    failures+=rodQuery.accepted || !rodQuery.axisCandidates || rodQuery.axesTested<3;
    std::printf("[NP_TRANSPORTED_AXIS] samples=%u error=%.12g overhang=%u swapped=%u boundaryCases=7 failures=%u\n",
        samples,worst,unsigned(moving.accepted),unsigned(second.accepted),failures);
    return failures;
}
static unsigned capturedQuery(const char* path)
{
    std::ifstream input(path);unsigned expected=0,fixed=0;AvbdNpDomain d;
    input>>expected>>fixed>>d.normal.x>>d.normal.y>>d.normal.z>>d.restDistance;d.fixedAxis=fixed!=0;
    PxTransform start[2],end[2];
    for(unsigned s=0;s<2;++s)
    {
        unsigned count=0;input>>d.radius[s]>>count;if(count<1 || count>256)return 1;
        for(auto* pose:{&start[s],&end[s]})input>>pose->p.x>>pose->p.y>>pose->p.z>>pose->q.x>>pose->q.y>>pose->q.z>>pose->q.w;
        d.count[s]=count;
        for(unsigned i=0;i<count;++i){PxVec3d p;input>>p.x>>p.y>>p.z;d.points[s].pushBack(p);}
    }
    if(!input)return 1;
    bool accepted=false;AvbdNpQuery query;AvbdNpDomainArgs args={&d,start,end,&accepted,&query};
    KernelTable::makeAvbdCpuScalar().launch(AVBD_NP_DOMAIN,&args);
    bool ok=query.valid && accepted==(expected!=0) && query.axesTested<=4 && query.axisCandidates<=15 &&
        query.pointVisits<=62*(d.points[0].size()+d.points[1].size());
    AvbdNpDomain swapped=d;swapped.normal=-d.normal;
    for(unsigned s=0;s<2;++s){swapped.points[s]=d.points[1-s];swapped.radius[s]=d.radius[1-s];}
    const PxTransform reverseStart[2]={start[1],start[0]},reverseEnd[2]={end[1],end[0]};
    ok=ok && avbdNpCompleteQuery(swapped,reverseStart,reverseEnd).accepted==(expected!=0);
    // Independently reconstruct actual world supports, rather than reusing
    // the polynomial coefficients. Sampling is a test oracle, not admission.
    if(accepted)for(unsigned k=0;k<=100;++k)
    {
        const double t=double(k)/100;PxQuatd q[2];PxVec3d p[2];
        for(unsigned s=0;s<2;++s)
        {
            const auto a=avbdNpRotation(start[s].q);auto b=avbdNpRotation(end[s].q);if(a.dot(b)<0)b=-b;
            q[s]=(a*(1-t)+b*t).getNormalized();p[s]=avbdOgcDouble(start[s].p)*(1-t)+avbdOgcDouble(end[s].p)*t;
        }
        const auto n=query.axisFrame<2?q[query.axisFrame].rotate(avbdNpRotation(start[query.axisFrame].q).rotateInv(query.proofAxis)):query.proofAxis;
        double lo=DBL_MAX,hi=-DBL_MAX;
        for(const auto& v:d.points[0])lo=PxMin(lo,n.dot(p[0]+q[0].rotate(v)));
        for(const auto& v:d.points[1])hi=PxMax(hi,n.dot(p[1]+q[1].rotate(v)));
        ok=ok && lo-hi-d.radius[0]-d.radius[1]-d.restDistance>=-1.e-10;
    }
    PxVec3d history;
    if(input>>history.x>>history.y>>history.z)
    {
        ok=ok && !accepted && avbdNpGap(d,start)<0;
        d.normal=history;
        const auto recovered=avbdNpCompleteQuery(d,start,end);
        ok=ok && recovered.accepted && recovered.startGap>0;
        // A formerly valid proof direction cannot authorize a changed pair.
        end[0].p-=PxVec3(float(history.x),float(history.y),float(history.z))*1.e-4f;
        ok=ok && !avbdNpCompleteQuery(d,start,end).accepted;
        std::printf("[NP_PROOF_HISTORY] clearance=%.12g staleRejected=1 status=%s\n",recovered.startGap,ok?"PASS":"FAIL");
    }
    std::printf("[NP_CAPTURED_QUERY] file=%s accepted=%u axes=%u visits=%u status=%s\n",path,unsigned(accepted),query.axesTested,query.pointVisits,ok?"PASS":"FAIL");
    return ok?0:1;
}
int main(int argc,char** argv)
{
    AvbdProbeFoundation foundation;std::mt19937 rng(98135);std::normal_distribution<double> random;
    unsigned failures=0,cases=0;double worst=0;PxU64 exhaustiveCalls=0,factorCalls=0;
    for(unsigned test=0;test<20;++test)
    {
        AvbdNpDomain domain;domain.fixedAxis=true;domain.normal=PxVec3d(random(rng),random(rng),random(rng)).getNormalized();
        domain.radius[0]=.1;domain.radius[1]=.2;domain.restDistance=.01;
        const unsigned sizes[2]={test%2?32u:64u,test%2?48u:96u};
        for(unsigned s=0;s<2;++s){domain.count[s]=sizes[s];for(unsigned i=0;i<sizes[s];++i)domain.points[s].pushBack(PxVec3d(random(rng),random(rng),random(rng)));}
        PxTransform start[2]={PxTransform(PxVec3(0,0,0)),PxTransform(PxVec3(0,2,0))},end[2];
        for(unsigned s=0;s<2;++s)end[s]=PxTransform(start[s].p+PxVec3(.03f,-.02f,.01f),PxQuat(float(random(rng)),PxVec3(1.f,.3f,.7f).getNormalized()));
        const PxVec3d dp[2]={PxVec3d(.03,-.04,.07),PxVec3d(-.02,.02,-.03)},dt[2]={PxVec3d(.2,-.1,.3),PxVec3d(-.1,.2,.1)};
        auto linear=[&](const AvbdNpPathFeature& f,unsigned k)
        {return f.gap[k]+domain.normal.dot(dp[0]-dp[1])+f.angular[k][0].dot(dt[0])+f.angular[k][1].dot(dt[1]);};
        const AvbdNpPathPolynomial polynomial(domain,start,end);
        const auto base=polynomial.evaluate(domain,0);++factorCalls;double fullGap=DBL_MAX;
        double lo[5],hi[5],reference[5],rawReference[5];unsigned a[5]={},b[5]={};
        for(unsigned k=0;k<5;++k)lo[k]=hi[k]=reference[k]=rawReference[k]=DBL_MAX;
        for(unsigned i=0;i<sizes[0];++i)
        {const auto f=polynomial.evaluate(domain,i*sizes[1]);++factorCalls;failures+=!f.valid;
            for(unsigned k=0;k<5;++k)if(linear(f,k)<lo[k]){lo[k]=linear(f,k);a[k]=i;}}
        for(unsigned j=0;j<sizes[1];++j)
        {const auto f=polynomial.evaluate(domain,j);++factorCalls;failures+=!f.valid;
            for(unsigned k=0;k<5;++k)if(linear(f,k)<hi[k]){hi[k]=linear(f,k);b[k]=j;}}
        for(unsigned i=0;i<sizes[0];++i)for(unsigned j=0;j<sizes[1];++j)
        {const auto f=polynomial.evaluate(domain,i*sizes[1]+j);++exhaustiveCalls;failures+=!f.valid;
            for(unsigned k=0;k<5;++k){reference[k]=std::min(reference[k],linear(f,k));
                rawReference[k]=std::min(rawReference[k],f.gap[k]);fullGap=std::min(fullGap,f.gap[k]);}}
        failures+=std::abs(fullGap-avbdNpPathGap(domain,start,end))>1.e-11;
        AvbdNpQuery query;bool accepted=false;
        AvbdNpDomainArgs queryArgs={&domain,start,end,&accepted,&query};
        KernelTable::makeAvbdCpuScalar().launch(AVBD_NP_DOMAIN,&queryArgs);
        double queryGap=DBL_MAX;
        for(unsigned k=0;k<5;++k)
        {
            queryGap=std::min(queryGap,query.gap[k]);
            const auto witness=polynomial.evaluate(domain,query.point[k][0]*sizes[1]+query.point[k][1]);
            failures+=std::abs(witness.gap[k]-query.gap[k])>1.e-11 || std::abs(query.gap[k]-rawReference[k])>1.e-11;
        }
        failures+=!query.valid || std::abs(queryGap-fullGap)>1.e-11;
        for(unsigned k=0;k<5;++k)
        {
            const auto witness=polynomial.evaluate(domain,a[k]*sizes[1]+b[k]);++factorCalls;
            const double error=std::max(std::abs(lo[k]+hi[k]-linear(base,k)-reference[k]),std::abs(linear(witness,k)-reference[k]));
            worst=std::max(worst,error);failures+=error>1.e-11;++cases;
        }
        // Fixed endpoints can select an interior support index. The minimum
        // must respect both the first index and the complete retained range.
        for(unsigned s=0;s<2;++s){domain.first[s]=sizes[s]/2;domain.count[s]=sizes[s]-domain.first[s];}
        double subsetGap=DBL_MAX;
        for(unsigned f=0;f<domain.count[0]*domain.count[1];++f)
        {const auto path=polynomial.evaluate(domain,f);for(unsigned k=0;k<5;++k)subsetGap=std::min(subsetGap,path.gap[k]);}
        failures+=std::abs(subsetGap-avbdNpPathGap(domain,start,end))>1.e-11;
    }
    std::printf("[NP_FACTORIZATION] cases=%u fullCalls=%llu factorCalls=%llu error=%.12g failures=%u completeMinima=40 sceneQualification=0\n",
        cases,static_cast<unsigned long long>(exhaustiveCalls),static_cast<unsigned long long>(factorCalls),worst,failures);
    // Ragged tiles and alternate reduction trees must retain every support
    // minimum and a matching witness. The oracle projects the whole point set.
    for(PxU32 count:{1u,31u,32u,33u,255u,256u})
    {
        PxVec3d coefficients[5];PxArray<PxVec3d> points;
        for(auto& c:coefficients)c=PxVec3d(random(rng),random(rng),random(rng));
        for(PxU32 i=0;i<count;++i)points.pushBack(PxVec3d(random(rng),random(rng),random(rng)));
        AvbdNpControlMinimum tiles[8],minimum;
        const PxU32 tileCount=(count+31)/32;
        for(PxU32 t=0;t<tileCount;++t)tiles[t]=avbdNpControlTile(coefficients,points.begin(),32*t,PxMin(32u,count-32*t));
        for(PxU32 t=tileCount;t>0;--t)minimum.merge(tiles[t-1]);
        for(PxU32 k=0;k<5;++k)
        {
            double full=DBL_MAX;for(const auto& p:points)full=PxMin(full,coefficients[k].dot(p));
            failures+=!minimum.valid || std::abs(minimum.gap[k]-full)>1.e-12 ||
                std::abs(coefficients[k].dot(points[minimum.point[k]])-full)>1.e-12;
        }
        failures+=avbdNpControlTile(coefficients,points.begin(),0,33).valid;
    }
    std::printf("[NP_CONTROL_TILES] sizes=1,31,32,33,255,256 width=32 reductions=5 failures=%u\n",failures);
    failures+=pathOracle(rng);
    failures+=transportedOracle(rng);
    for(int i=1;i<argc;++i)failures+=capturedQuery(argv[i]);
    {
        AvbdNpDomain d;d.normal=PxVec3d(0,1,0);d.radius[0]=d.radius[1]=0;
        d.points[0].pushBack(PxVec3d(0));d.points[1].pushBack(PxVec3d(0));
        PxTransform start[2]={PxTransform(PxVec3(0,1,0)),PxTransform(PxIdentity)};
        PxTransform end[2]={PxTransform(PxVec3(0,-1,0)),PxTransform(PxIdentity)};
        failures+=!avbdNpCompleteQuery(d,start,start).accepted;
        failures+=avbdNpCompleteQuery(d,start,end).accepted;
        d.points[0].resize(257,PxVec3d(0));failures+=avbdNpCompleteQuery(d,start,start).valid;
        std::printf("[NP_BOUNDED_QUERY] witnesses=100 capacityAndCrossing=3 failures=%u\n",failures);
    }
    return failures?1:0;
}
