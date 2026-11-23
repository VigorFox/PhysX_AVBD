#include "avbd/DyAvbdMaterialAssembly.h"
#include "avbd/DyKernelTable.h"
#include <cstdio>
#include <cmath>
using namespace physx;
using namespace physx::Dy;
static PxU32 seed=20261028;
static double randomValue(){seed=1664525u*seed+1013904223u;return double(seed>>8)/16777216.-.5;}
static PxU32 member(PxU32 id,const PxU32* blocks,PxU32 count)
{for(PxU32 b=0;b<count;++b)if(id==blocks[b])return b;return count;}
int main()
{
    const KernelTable table=KernelTable::makeAvbdCpuScalar();
    unsigned failures=0,negativeControls=0;double maxMatrix=0,maxGradient=0,maxDifference=0;
    for(PxU32 trial=0;trial<1200;++trial)
    {
        const PxU32 count=1+trial%4,blocks[4]={7,2,19,11},row=37;
        const bool frame=(trial/4)%2==0;const PxU32 rank=frame?9:1,supportCount=frame?2:4;
        const PxU32 rows[1]={row},offsets[2]={0,rank};
        PxU32 supports[4];PxVec3 arms[4];PxVec3d points[4];
        for(PxU32 s=0;s<supportCount;++s)
        {
            supports[s]=blocks[(s+trial/8)%count];
            if(trial%13==0 && s==0)supports[s]=PX_MAX_U32; // Prescribed support.
            arms[s]=PxVec3(PxReal(randomValue()),PxReal(randomValue()),PxReal(randomValue()));
        }
        // Independent support-space derivatives, BEFORE any body assembly.
        double jacobian[9][4][6]={},gradient[4][6]={},curvature[9]={};
        AvbdElasticFrameEvaluation elastic={};AvbdTetrahedronEvaluation tet={};
        AvbdTetrahedronMaterial material={1,1000,0};
        if(frame)
        {
            PxConstraintElasticFrameData data={};
            data.frames[0]=PxTransform(PxVec3(.1f,.2f,-.1f));data.frames[1]=PxTransform(PxVec3(-.2f,.1f,.3f));
            for(PxU32 k=0;k<6;++k){data.stiffness[k]=float(100*(k+1));data.damping[k]=float(k);}
            PxTransform poses[2]={PxTransform(PxIdentity),PxTransform(PxVec3(.5f,.2f,.1f),PxQuat(float(randomValue()),PxVec3(0,1,0)))};
            AvbdElasticFramePrepared m;avbdPrepareElasticFrame(m,data,poses,.01);
            poses[1].p+=PxVec3(.03f,-.01f,.02f);
            const AvbdElasticFrameMaterial view=m;
            AvbdElasticFrameArgs args={&view,poses,AvbdElasticFrameRequest::Reference,&elastic};table.launch(AVBD_ELASTIC_FRAME_MATERIAL,&args);
            failures+=!elastic.valid;
            for(PxU32 r=0;r<rank;++r)
            {
                curvature[r]=1;
                for(PxU32 s=0;s<2;++s)for(PxU32 d=0;d<6;++d)jacobian[r][s][d]=elastic.root[r][6*s+d];
            }
            for(PxU32 s=0;s<2;++s)for(PxU32 d=0;d<6;++d)gradient[s][d]=elastic.gradient[6*s+d];
        }
        else
        {
            points[0]=PxVec3d(0,0,0);points[1]=PxVec3d(1.1+.2*randomValue(),.1,0);
            points[2]=PxVec3d(0,1.05,.1);points[3]=PxVec3d(.1,0,1.1);
            AvbdTetrahedronArgs args={&material,points,&tet};table.launch(AVBD_TETRAHEDRON_MATERIAL,&args);
            failures+=!tet.valid;curvature[0]=tet.curvature;
            for(PxU32 s=0;s<4;++s)
            {
                const PxVec3d angular=PxVec3d(arms[s].x,arms[s].y,arms[s].z).cross(tet.jacobians[s]);
                for(PxU32 d=0;d<3;++d)
                {
                    jacobian[0][s][d]=tet.jacobians[s][d];jacobian[0][s][d+3]=angular[d];
                    gradient[s][d]=tet.gradient*tet.jacobians[s][d];gradient[s][d+3]=tet.gradient*angular[d];
                }
            }
        }
        AvbdGroupFactor factors[9]={};AvbdRigidAssembly assembled[4]={};
        PxReal diagonal[4][6][6]={},g[4][6]={};double expected[24][24]={},expectedG[24]={};
        for(PxU32 b=0;b<count;++b)
        {
            auto& a=assembled[b];a.coupledRows=rows;a.count=1;a.member=b;a.factors=factors;
            a.factorOffsets=offsets;a.consistent=true;a.lockedMask=(trial/8+17*b)%64;
            for(PxU32 d=0;d<6;++d){a.hessian[d][d]=1;expected[6*b+d][6*b+d]=1;}
        }
        AvbdRigidAssemblyT<double> precise[4]={};AvbdGroupFactor preciseFactors[9]={};
        for(PxU32 b=0;b<count;++b)
        {precise[b].coupledRows=rows;precise[b].count=1;precise[b].member=b;precise[b].factors=preciseFactors;
            precise[b].factorOffsets=offsets;precise[b].consistent=true;precise[b].lockedMask=assembled[b].lockedMask;}
        if(frame)
        {avbdAssembleElasticFrame(row,elastic,supports,blocks,count,assembled);avbdAssembleElasticFrame(row,elastic,supports,blocks,count,precise);}
        else
        {avbdAssembleTetrahedron(row,tet,supports,arms,blocks,count,assembled);avbdAssembleTetrahedron(row,tet,supports,arms,blocks,count,precise);}
        // T^T H T in support coordinates, including EACH support pair on the
        // same body. This reference does not use emitted assembly factors.
        for(PxU32 s=0;s<supportCount;++s)
        {
            const PxU32 a=member(supports[s],blocks,count);if(a==count)continue;
            for(PxU32 d=0;d<6;++d)if(!(assembled[a].lockedMask&(1u<<d)))
            {
                expectedG[6*a+d]+=gradient[s][d];
                for(PxU32 t=0;t<supportCount;++t)
                {
                    const PxU32 b=member(supports[t],blocks,count);if(b==count)continue;
                    for(PxU32 k=0;k<6;++k)if(!(assembled[b].lockedMask&(1u<<k)))
                        for(PxU32 r=0;r<rank;++r)expected[6*a+d][6*b+k]+=curvature[r]*jacobian[r][s][d]*jacobian[r][t][k];
                }
            }
        }
        for(PxU32 b=0;b<count;++b)
        {
            for(PxU32 d=0;d<6;++d)if(!(precise[b].lockedMask&(1u<<d)))
                failures+=std::abs(precise[b].gradient[d]-expectedG[6*b+d])>1.e-12*PxMax(1.,std::abs(expectedG[6*b+d]));
            auto& a=assembled[b];failures+=!a.consistent;avbdKernelRestrict6(a.hessian,a.gradient,a.lockedMask);
            PxMemCopy(diagonal[b],a.hessian,sizeof(a.hessian));PxMemCopy(g[b],a.gradient,sizeof(a.gradient));
        }
        AvbdGroupSolveArgs args={count,diagonal,g,factors,rank,NULL,NULL,NULL};AvbdGroupOperator op={args,{}};
        double error=0,scale=1,gradientError=0,gradientScale=1;
        for(PxU32 c=0;c<6*count;++c)
        {
            double unit[24]={},product[24];unit[c]=1;op.apply(unit,product);
            for(PxU32 r=0;r<6*count;++r){error=PxMax(error,std::abs(product[r]-expected[r][c]));scale=PxMax(scale,std::abs(expected[r][c]));}
            gradientError=PxMax(gradientError,std::abs(double(g[c/6][c%6])-expectedG[c]));gradientScale=PxMax(gradientScale,std::abs(expectedG[c]));
        }
        maxMatrix=PxMax(maxMatrix,error/scale);maxGradient=PxMax(maxGradient,gradientError/gradientScale);
        failures+=error/scale>2.e-6 || gradientError/gradientScale>2.e-6;
        if(!frame)
        {
            double direction[24];for(PxU32 i=0;i<6*count;++i)direction[i]=(assembled[i/6].lockedMask&(1u<<(i%6)))?0:randomValue();
            PxVec3d plus[4],minus[4];const double h=1.e-6;double predicted=0;
            for(PxU32 i=0;i<6*count;++i)predicted+=expectedG[i]*direction[i];
            for(PxU32 s=0;s<4;++s)
            {
                const PxU32 b=member(supports[s],blocks,count);PxVec3d delta(0);
                if(b<count)delta=PxVec3d(direction[6*b],direction[6*b+1],direction[6*b+2])+
                    PxVec3d(direction[6*b+3],direction[6*b+4],direction[6*b+5]).cross(PxVec3d(arms[s].x,arms[s].y,arms[s].z));
                plus[s]=points[s]+delta*h;minus[s]=points[s]-delta*h;
            }
            const double finite=(avbdEvaluateTetrahedron(material,plus).energy-avbdEvaluateTetrahedron(material,minus).energy)/(2*h);
            const double difference=std::abs(finite-predicted)/PxMax(1.,std::abs(predicted));maxDifference=PxMax(maxDifference,difference);failures+=difference>1.e-6;
            if(count==1 && trial%13!=0)
            {
                // Old independent support outer products oppose a common
                // translation although the determinant is translation invariant.
                double wrong=0;for(PxU32 s=0;s<4;++s)wrong+=curvature[0]*jacobian[0][s][0]*jacobian[0][s][0];
                negativeControls+=wrong>1;
            }
        }
    }
    failures+=negativeControls==0;
    std::printf("[MATERIAL_ASSEMBLY] cases=1200 matrixError=%.9g gradientError=%.9g finiteDifferenceError=%.9g oldDuplicateSupportFailures=%u failures=%u\n",
        maxMatrix,maxGradient,maxDifference,negativeControls,failures);
    return failures?1:0;
}
