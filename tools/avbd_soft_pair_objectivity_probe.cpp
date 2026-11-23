// Counterexample for the C01b full-pair, fixed-world-plane force prototype.
// This is a diagnostic of that proposal, not a passing physical gate.
#include "avbd/DyAvbdContactPotential.h"
#include "avbd/DyAvbdContactGeometry.h"
#include "avbd_probe_foundation.h"
#include <cstdio>

using namespace physx;
using namespace physx::Dy;

int main()
{
    AvbdProbeFoundation foundation;
    const PxVec3 a[3]={PxVec3(0,0,0),PxVec3(1,0,0),PxVec3(0,1,0)};
    const PxVec3 shift(0.25f,0.125f,0.1f);
    const PxVec3 b[3]={a[0]+shift,a[1]+shift,a[2]+shift};
    const AvbdTrianglePairPlane plane=avbdTrianglePairPlane(a,b,PxVec3(0,0,-1));
    PxVec3 force(0.0f),moment(0.0f);
    PxReal energy=0.0f,rotatedEnergy=0.0f;
    const PxQuat rotation(0.01f,PxVec3(0,1,0));
    for(PxU32 i=0;i<3;++i) for(PxU32 j=0;j<3;++j)
    {
        const AvbdContactPotential value=avbdUnilateralContactPotential(
            plane.normal.dot(a[i]-b[j])-0.2f,100.0f,0.0f,-PX_MAX_F32);
        const PxVec3 fa=-value.gradient*plane.normal,fb=-fa;
        force+=fa+fb;
        moment+=a[i].cross(fa)+b[j].cross(fb);
        energy+=value.energy;
        rotatedEnergy+=avbdUnilateralContactPotential(
            plane.normal.dot(rotation.rotate(a[i])-rotation.rotate(b[j]))-0.2f,
            100.0f,0.0f,-PX_MAX_F32).energy;
    }
    const bool pass=moment.magnitude()<1.0e-4f && PxAbs(rotatedEnergy-energy)<1.0e-4f;
    std::printf("[SOFT_PAIR_OBJECTIVITY] proposal=full_pair_fixed_world_plane status=%s netForce=%.9g netMoment=%.9g energy=%.9g rotatedEnergy=%.9g\n",
        pass?"PASS":"FAIL",double(force.magnitude()),double(moment.magnitude()),double(energy),double(rotatedEnergy));
    return pass?0:1;
}
