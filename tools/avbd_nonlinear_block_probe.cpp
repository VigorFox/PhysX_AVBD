// Independent small-system and directional-curvature checks of the Table kernel.
#include "avbd/DyAvbdNonlinearBlock.h"
#include "avbd/DyAvbdNonlinearGroup.h"
#include "avbd/DyAvbdLocalWork.h"
#include "avbd/DyKernelTable.h"
#include "avbd_probe_foundation.h"
#include <cstdio>
#include <limits>
#include <vector>
using namespace physx;using namespace physx::Dy;
int main()
{
    AvbdProbeFoundation foundation;
    const auto table=KernelTable::makeAvbdCpuScalar();unsigned failures=0,cases=0;
    double worst=0;
    for(unsigned n:{3u,6u})for(unsigned locked=0;locked<(1u<<n);++locked)
    {
        AvbdNonlinearBlock b;b.dimension=n;b.locked=locked;double expected[6]={};
        for(unsigned i=0;i<n;++i)expected[i]=(locked&(1u<<i))?0:.1*(i+1);
        for(unsigned i=0;i<n;++i)for(unsigned j=0;j<n;++j)
        {b.K[6*i+j]=b.M[6*i+j]=(i==j?2.+i:0)+.1*(i+1)*(j+1);b.gradient[i]-=b.K[6*i+j]*expected[j];}
        AvbdNonlinearBlockArgs a={&b};table.launch(AVBD_NONLINEAR_BLOCK,&a);++cases;
        failures+=!b.solved;
        for(unsigned i=0;i<n;++i){const double error=std::abs(b.step[i]-expected[i]);worst=std::max(worst,error);failures+=error>1.e-12;}
    }
    // Curvatures can cancel across materials. Projecting each material first
    // gives a different positive direction even though the total is SPD.
    AvbdElasticFrameEvaluation e[2]={};const unsigned ends[2]={7,9};AvbdNonlinearBlock assembled;
    for(unsigned i=0;i<6;++i){assembled.K[7*i]=assembled.M[7*i]=1;assembled.gradient[i]=1;}
    e[0].hessian[0]=-3;e[1].hessian[0]=5;e[1].root[0][0]=std::sqrt(5.);
    avbdAccumulateFrameBlock(e[0],ends,7,assembled);avbdAccumulateFrameBlock(e[1],ends,7,assembled);
    AvbdNonlinearBlockArgs a={&assembled};table.launch(AVBD_NONLINEAR_BLOCK,&a);++cases;
    failures+=!assembled.solved || std::abs(assembled.step[0]+1./3)>1.e-12;
    const double projectedDirection=-1./6;
    failures+=std::abs(assembled.step[0]-projectedDirection)<.1;

    // Indefinite modes use the supplied physical metric, with no absolute cap.
    for(unsigned n:{3u,6u})for(double units:{.001,1.,1000.})
    {
        AvbdNonlinearBlock b;b.dimension=n;
        for(unsigned i=0;i<n;++i)
        {b.K[7*i]=units*(i==0?-.5:i==1?0:2);b.M[7*i]=units;b.gradient[i]=units;}
        AvbdNonlinearBlockArgs args={&b};table.launch(AVBD_NONLINEAR_BLOCK,&args);++cases;
        failures+=!b.solved;
        for(unsigned i=0;i<n;++i)failures+=std::abs(b.step[i]+(i<2?1:.5))>1.e-10;
    }
    // Real frame: verify each assembled body block by second differences of
    // energy on a world-left exponential curve, independently of its Hessian.
    PxConstraintElasticFrameData data={};data.frames[0]=PxTransform(PxVec3(.2f,.1f,0));
    data.frames[1]=PxTransform(PxVec3(-.1f,.3f,.1f));
    for(unsigned i=0;i<6;++i){data.stiffness[i]=float(50+20*i);data.damping[i]=float(i);}
    PxTransform poses[2]={PxTransform(PxVec3(.1f,0,0),PxQuat(.2f,PxVec3(1,0,0))),
        PxTransform(PxVec3(.4f,.1f,.2f),PxQuat(-.4f,PxVec3(0,1,0)))};
    AvbdElasticFramePrepared material;failures+=!avbdPrepareElasticFrame(material,data,poses,.02);
    const auto initial=finiteJointCoordinates<double,false>(avbdElasticFrameDouble(poses[0])*avbdElasticFrameDouble(data.frames[0]),
        avbdElasticFrameDouble(poses[1])*avbdElasticFrameDouble(data.frames[1]),
        avbdElasticFrameDouble(poses[0]).p,avbdElasticFrameDouble(poses[1]).p);
    poses[1].p.y+=.1f;
    const auto sample=avbdEvaluateElasticFrame(material,poses,true,true,false);
    for(unsigned body=0;body<2;++body)for(unsigned test=0;test<6;++test)
    {
        AvbdElasticFrameContribution local;
        const AvbdElasticFrameMaterial view=material;
        AvbdElasticFrameArgs request={&view,poses,AvbdElasticFrameRequest::Contribution,&local,body};
        table.launch(AVBD_ELASTIC_FRAME_MATERIAL,&request);failures+=!local.valid;
        AvbdNonlinearBlock b;avbdAccumulateFrameBlock(local,ends,ends[body],b);
        AvbdNonlinearBlock full;avbdAccumulateFrameBlock(sample,ends,ends[body],full,true);
        for(unsigned k=0;k<36;++k)
            failures+=std::abs(full.K[k]-b.K[k])>1.e-10*std::max(1.,std::abs(full.K[k])) ||
                std::abs(full.M[k]-b.M[k])>1.e-10*std::max(1.,std::abs(full.M[k]));
        double direction[6]={};direction[test]=1;direction[(test+2)%6]=-.3;
        const double h=.003;double energies[2],coordinates[2][FINITE_JOINT_ROWS];
        for(unsigned side=0;side<2;++side)
        {
            auto trial0=avbdElasticFrameDouble(poses[0]),trial1=avbdElasticFrameDouble(poses[1]);
            auto& trial=body?trial1:trial0;const double sign=side?1:-1;
            trial.p+=PxVec3d(direction[0],direction[1],direction[2])*(sign*h);
            const PxVec3d angle=PxVec3d(direction[3],direction[4],direction[5])*(sign*h);
            if(angle.magnitude()>0)trial.q=PxQuatT<double>(angle.magnitude(),angle.getNormalized())*trial.q;
            const auto c=finiteJointCoordinates(trial0*avbdElasticFrameDouble(data.frames[0]),
                trial1*avbdElasticFrameDouble(data.frames[1]),trial0.p,trial1.p);
            energies[side]=0;
            for(unsigned r=0;r<FINITE_JOINT_ROWS;++r)
            {coordinates[side][r]=c.value[r];const unsigned d=finiteJointDriveIndex(r);const double dx=c.value[r]-initial.value[r];
             energies[side]+=.5*(data.stiffness[d]*c.value[r]*c.value[r]+material.parameters.dampingOverDt[d]*dx*dx);}
        }
        double analytic=0;for(unsigned i=0;i<6;++i)for(unsigned j=0;j<6;++j)analytic+=direction[i]*b.K[6*i+j]*direction[j];
        const double numeric=(energies[0]+energies[1]-2*sample.energy)/(h*h);
        failures+=std::abs(numeric-analytic)>1.e-4*std::max(1.,std::abs(analytic));++cases;
        double gn=0,expected=0;
        for(unsigned i=0;i<6;++i)for(unsigned j=0;j<6;++j)gn+=direction[i]*b.M[6*i+j]*direction[j];
        for(unsigned r=0;r<FINITE_JOINT_ROWS;++r)
        {const unsigned d=finiteJointDriveIndex(r);const double J=(coordinates[1][r]-coordinates[0][r])/(2*h);
            expected+=(data.stiffness[d]+material.parameters.dampingOverDt[d])*J*J;}
        failures+=std::abs(gn-expected)>1.e-4*std::max(1.,std::abs(expected));
    }
    // Repeated physical endpoints must retain cross terms. Moving both
    // authored frames on one body cannot deform that body's internal frame.
    {
        const PxTransform same[2]={poses[0],poses[0]};const unsigned repeated[2]={7,7};
        const auto e=avbdEvaluateElasticFrame(material,same,true,true,false);
        const auto energy=avbdEvaluateElasticFrame(material,same,false);
        AvbdNonlinearBlock b;avbdAccumulateFrameBlock(e,repeated,7,b,true);
        failures+=!e.valid || !energy.valid || std::abs(e.energy-energy.energy)>1.e-10;
        for(unsigned k=0;k<36;++k)failures+=std::abs(b.K[k])>1.e-9 || std::abs(b.M[k])>1.e-9;
        for(unsigned k=0;k<6;++k)failures+=std::abs(b.gradient[k])>1.e-9;
        ++cases;
    }
    for(double mass:{.01,1.,100.})for(double rho:{1.,1.e4,1.e8})
    {
        const double dt=.01,w=1/mass,tau=avbdInertialDualStep(rho,w,dt);
        failures+=!(tau>0 && tau<=rho && tau<=1/(dt*dt*w));
        failures+=std::abs(avbdInertialDualStep(7*rho,w/7,dt)-7*tau)>1.e-12*7*tau;++cases;
    }
    AvbdNonlinearBlock invalid;invalid.gradient[0]=std::numeric_limits<double>::quiet_NaN();
    AvbdNonlinearBlockArgs bad={&invalid};table.launch(AVBD_NONLINEAR_BLOCK,&bad);failures+=invalid.solved;++cases;
    // Coupled Rigid6 elimination: independent known solutions, locks, units,
    // and an indefinite model requiring its supplied GN metric.
    for(unsigned count=1;count<=4;++count)for(double units:{.001,1.,1000.})for(unsigned indefinite=0;indefinite<2;++indefinite)
    {
        AvbdNonlinearGroup g;g.count=count;double expected[4][6]={};
        for(unsigned i=0;i<count;++i)
        {g.locked[i]=1u<<i;for(unsigned r=0;r<6;++r)expected[i][r]=(g.locked[i]&(1u<<r))?0:.02*(6*i+r+1);}
        for(unsigned i=0;i<count;++i)for(unsigned j=0;j<count;++j)for(unsigned r=0;r<6;++r)for(unsigned c=0;c<6;++c)
        {
            const double a=units*((i==j && r==c?2.:0)+.01*(6*i+r+1)*(6*j+c+1));
            g.K[i][j][6*r+c]=g.M[i][j][6*r+c]=a;g.gradient[i][r]-=a*expected[j][c];
        }
        if(indefinite)g.K[0][0][35]=-units;
        AvbdNonlinearGroupArgs request={&g};table.launch(AVBD_NONLINEAR_GROUP,&request);++cases;
        failures+=!g.solved || g.factorizations!=(indefinite?2u:1u) || g.metric!=bool(indefinite);
        for(unsigned i=0;i<count;++i)for(unsigned r=0;r<6;++r)
            failures+=std::abs(g.step[i][r]-expected[i][r])>1.e-10;
    }
    // Two-endpoint mixed directions independently check the material cross
    // blocks through energy differences, including a common translation.
    for(unsigned trial=0;trial<8;++trial)
    {
        AvbdNonlinearGroup g;g.count=2;avbdAccumulateFrameGroup(sample,ends,ends,g);
        double direction[2][6]={};
        if(!trial){direction[0][0]=direction[1][0]=1;}
        else for(unsigned s=0;s<2;++s)for(unsigned d=0;d<6;++d)direction[s][d]=std::sin(double(3*s+2*d+trial))*.4;
        double energies[2]={};const double h=.003;
        for(unsigned side=0;side<2;++side)
        {
            PxTransformT<double> p[2]={avbdElasticFrameDouble(poses[0]),avbdElasticFrameDouble(poses[1])};
            const double t=(side?1:-1)*h;
            for(unsigned s=0;s<2;++s)
            {
                p[s].p+=PxVec3d(direction[s][0],direction[s][1],direction[s][2])*t;
                const PxVec3d angle=PxVec3d(direction[s][3],direction[s][4],direction[s][5])*t;
                if(angle.magnitude()>0)p[s].q=PxQuatT<double>(angle.magnitude(),angle.getNormalized())*p[s].q;
            }
            const auto c=finiteJointCoordinates(p[0]*avbdElasticFrameDouble(data.frames[0]),p[1]*avbdElasticFrameDouble(data.frames[1]),p[0].p,p[1].p);
            for(unsigned r=0;r<FINITE_JOINT_ROWS;++r)
            {const unsigned d=finiteJointDriveIndex(r);const double dx=c.value[r]-initial.value[r];
                energies[side]+=.5*(data.stiffness[d]*c.value[r]*c.value[r]+material.parameters.dampingOverDt[d]*dx*dx);}
        }
        double analytic=0;
        for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j)for(unsigned r=0;r<6;++r)for(unsigned c=0;c<6;++c)
            analytic+=direction[i][r]*g.K[i][j][6*r+c]*direction[j][c];
        const double numeric=(energies[0]+energies[1]-2*sample.energy)/(h*h);
        failures+=std::abs(numeric-analytic)>1.e-4*std::max(1.,std::abs(analytic));++cases;
    }
    // Odd/even trees, multiple independent roots, complete signed sums and
    // boolean admission. Trial reduction cannot overwrite the assembled model.
    for(unsigned rows:{257u,512u,513u,1025u,4097u})
    {
        AvbdLocalWork work;
        for(unsigned body=0;body<3;++body)work.add(body,body*(rows+9),body*(rows+9)+rows);
        work.partials.resize(work.nodes.size());std::vector<unsigned> visits(3*rows,0);
        for(auto id:work.leaves)
        {
            const auto& n=work.nodes[id];auto& p=work.partials[id];failures+=n.end-n.begin>256;
            p.scalar=n.state!=1;p.valid=n.state!=2;
            for(unsigned slot=n.begin;slot<n.end;++slot)
            {
                const unsigned r=slot-n.state*(rows+9);++visits[n.state*rows+r];
                p.model.K[0]+=r+1;p.model.M[7]+=1;p.model.gradient[2]+=(r%2?-1.:1.)*(r+1);p.energy+=r+1;
            }
        }
        for(auto visitsForRow:visits)failures+=visitsForRow!=1;
        for(const auto& level:work.levels)for(auto id:level)
        {
            const auto& n=work.nodes[id];AvbdLocalReductionArgs args={&work.partials[n.left],&work.partials[n.right],&work.partials[id],false};
            table.launch(AVBD_LOCAL_REDUCTION,&args);
        }
        const double sum=double(rows)*(rows+1)/2,alternating=rows%2?double(rows+1)/2:-double(rows)/2;
        for(const auto& s:work.states)
        {
            const auto& p=work.partials[s.root];failures+=p.model.K[0]!=sum || p.model.M[7]!=rows || p.model.gradient[2]!=alternating;
            failures+=p.energy!=sum || p.scalar!=(s.body!=1) || p.valid!=(s.body!=2);++cases;
        }
        for(auto id:work.leaves){work.partials[id].energy*=2;work.partials[id].valid=false;}
        for(const auto& level:work.levels)for(auto id:level)
        {
            const auto& n=work.nodes[id];AvbdLocalReductionArgs args={&work.partials[n.left],&work.partials[n.right],&work.partials[id],true};
            table.launch(AVBD_LOCAL_REDUCTION,&args);
        }
        for(const auto& s:work.states)
        {const auto& p=work.partials[s.root];failures+=p.model.K[0]!=sum || p.energy!=2*sum || p.valid;++cases;}
    }
    std::printf("[NONLINEAR_BLOCK] cases=%u maxSolveError=%.9g failures=%u\n",cases,worst,failures);
    return failures?1:0;
}
