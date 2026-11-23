// Verify candidate evaluations without publishing poses or cloning body state.
#include "avbd/DyAvbdInternal.h"
#include "foundation/PxPhysicsVersion.h"
#include "extensions/PxDefaultAllocator.h"
#include <cstdio>
using namespace physx;
using namespace physx::Dy;
struct Errors : PxErrorCallback
{
    unsigned count=0;
    void reportError(PxErrorCode::Enum,const char* message,const char*,int) PX_OVERRIDE
    { ++count;std::printf("ERROR: %s\n",message); }
};

int main()
{
    PxDefaultAllocator allocator;Errors errors;
    PxFoundation* foundation=PxCreateFoundation(PX_PHYSICS_VERSION,allocator,errors);
    bool pass=true;PxReal maximum=0;
    {
        PxArray<AvbdSolverBody> bodies; bodies.resize(3);
        for(PxU32 i=0;i<bodies.size();++i)
        {
            AvbdSolverBody& b=bodies[i];
            b.p=PxVec3(.2f*i,-.4f,.1f);b.q=PxQuat(.13f*i,PxVec3(1,2,3).getNormalized());
            b.pHat=b.p+PxVec3(.01f,.02f,-.03f);b.qHat=PxQuat(.07f,PxVec3(0,1,0))*b.q;
            b.pStart=b.p-PxVec3(.03f,.02f,.01f);b.qStart=PxQuat(.04f,PxVec3(1,0,0))*b.q;
            b.pLin=b.p+PxVec3(.01f);b.qLin=PxQuat(.09f,PxVec3(0,0,1))*b.q;
            b.linVel=b.angVel=b.outputLinVel=b.outputAngVel=PxVec3(0);
            b.invMass=b.massInvH2=1;b.invInertia=PxVec3(1);b.inertiaWorldInvH2=PxMat33(PxIdentity);
            b.disableGravity=b.isKinematic=b.hasStaticTouch=b.hasContact=0;
            b.ogcRegion.anchorPosition=b.p;b.ogcRegion.anchorRotation=b.q;
            b.ogcRegion.supportRadius=1;b.ogcRegion.bound=PX_MAX_F32;b.ogcRegion.planeCount=32;
            b.ogcRegion.planes.resize(32); // Exercise a certificate exceeding inline capacity.
            PxMemZero(b.ogcRegion.planes.begin(),32*sizeof(AvbdOgcHalfSpace));
            b.ogcRegionValid=b.ogcExceeded=0;b.core=NULL;b.rigid=NULL;b.artic=NULL;b.linkIndex=0;
        }
        for(PxU32 sample=0;sample<128;++sample)
        {
            PxTransform candidates[3];
            for(PxU32 i=0;i<3;++i)
                candidates[i]=PxTransform(bodies[i].p+PxVec3(.0003f*sample,.01f,-.04f),
                    PxQuat(.011f*sample,PxVec3(2,1,-1).getNormalized())*bodies[i].q);
            const PxU32 ids[2]={2,0};
            for(PxU32 mode=0;mode<3;++mode)
            {
                const PxU32 count=mode==0?1:mode==1?2:3;
                const AvbdRigidPoseView view={bodies,mode==2?NULL:ids,candidates,count};
                PxArray<AvbdSolverBody> copied(bodies);
                for(PxU32 k=0;k<count;++k)
                { const PxU32 i=mode==2?k:ids[k];copied[i].p=candidates[k].p;copied[i].q=candidates[k].q; }
                for(PxU32 k=0;k<6;++k)
                {
                    AvbdRowBuilder row;PxMemZero(&row,sizeof(row));
                    row.header().bodyA=0;row.header().bodyB=2;row.scalar().n=PxVec3(1,2,-1).getNormalized();
                    row.scalar().jLinA=row.scalar().n;row.scalar().jLinB=-row.scalar().n;
                    row.scalar().jAngA=PxVec3(.4f,-.2f,.1f);row.scalar().jAngB=-row.scalar().jAngA;
                    row.scalar().rA=row.scalar().ogcRA=PxVec3(.2f,.3f,.4f);
                    row.scalar().rB=row.scalar().ogcRB=PxVec3(-.1f,.05f,-.02f);
                    row.scalar().C0=-.02f;row.scalar().CLin=.03f;row.scalar().contactOffset=-.04f;
                    row.scalar().driveStiffness=17;row.scalar().driveDampingOverDt=29;row.scalar().driveVelocityTarget=.08f;
                    if(k==1) row.header().flags=eAVBD_ROW_CONTACT|eAVBD_ROW_UNILATERAL;
                    if(k==2) row.header().flags=eAVBD_ROW_POINT_LOCK;
                    if(k==3 || k==4) row.header().flags=eAVBD_ROW_DRIVE;
                    if(k==4) row.header().kernelId=eAVBD_ROW_KERNEL_FINITE_SPRING;
                    if(k==5) row.scalar().integratedAngular=1;
                    const PxReal expected=avbdPrimalC(row,copied,1),actual=avbdPrimalC(row,view,1);
                    const PxReal error=PxAbs(expected-actual);maximum=PxMax(maximum,error);
                    pass&=PxIsFinite(actual) && error<2.e-6f;
                }
                for(PxU32 i=0;i<3;++i)
                {
                    const AvbdBodyPose pose=view[i];
                    pass&=pose.p==copied[i].p && pose.q==copied[i].q;
                    pass&=&pose.pStart==&bodies[i].pStart && &pose.qHat==&bodies[i].qHat;
                    pass&=pose.ogcRegion.planes.begin()==bodies[i].ogcRegion.planes.begin();
                    pass&=bodies[i].p==PxVec3(.2f*i,-.4f,.1f);
                }
            }
        }
        std::printf("[POSE_VIEW] cases=2304 bodyBytes=%u candidateBytes=%u blockBytes=%u particleBytes=%u maxResidualError=%.9g status=%s\n",
            unsigned(sizeof(AvbdSolverBody)),unsigned(sizeof(PxTransform)),unsigned(sizeof(AvbdBlock)),
            unsigned(sizeof(AvbdSoftParticle)),maximum,pass?"PASS":"FAIL");
    }
    foundation->release();return pass && !errors.count?0:1;
}
