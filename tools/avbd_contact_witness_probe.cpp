// Test the NP endpoint convention against independently authored surfaces.
#include "PxsContactGeometry.h"
#include <cmath>
#include <cstdio>
using namespace physx;
int main()
{
    PxsContactGeometry geometry;geometry.manifoldPointShape=1;
    const PxVec3 point(.5f,0,0),normal(0,1,0),center(0,.5f,0);
    PxVec3 witnesses[2];unsigned failures=0,samples=0;
    failures+=!geometry.normalWitnesses(point,normal,.49f,witnesses);
    const PxVec3 local=witnesses[0]-center,legacy=point-center;
    // A is a thin rotating slab, B its stationary support. The real bottom
    // corner is (.5,-.01,0); the shared-point proxy is half a metre below COM.
    failures+=(local-PxVec3(.5f,-.01f,0)).magnitude()>1.e-6f;
    failures+=(local.cross(normal)-legacy.cross(normal)).magnitude()>1.e-6f;
    double worst=0,legacyError=0,derivativeError=0;
    for(unsigned i=0;i<=200;++i)
    {
        const double angle=double(i)*3.14159265358979323846/200;
        const PxQuat q(PxReal(angle),PxVec3(0,0,1));
        const PxVec3 arm=q.rotate(local);
        const double actual=normal.dot(center+arm-witnesses[1]);
        const double oracle=.5+.5*std::sin(angle)-.01*std::cos(angle);
        worst=PxMax(worst,std::abs(actual-oracle));
        derivativeError=PxMax(derivativeError,std::abs(double(arm.cross(normal).z)-(.5*std::cos(angle)+.01*std::sin(angle))));
        legacyError=PxMax(legacyError,std::abs(double(normal.dot(center+q.rotate(legacy)-point)+.49f)-oracle));
        // Reverse pair order: same two authored witnesses and signed gap.
        geometry.manifoldPointShape=0;PxVec3 reversed[2];
        failures+=!geometry.normalWitnesses(point,-normal,.49f,reversed);
        failures+=(reversed[0]-witnesses[1]).magnitude()>1.e-6f || (reversed[1]-witnesses[0]).magnitude()>1.e-6f;
        ++samples;
    }
    failures+=worst>3.e-7 || derivativeError>3.e-7 || legacyError<.9;
    // Unsupported generator convention must leave its public point intact.
    geometry.manifoldPointShape=PX_MAX_U32;
    failures+=geometry.normalWitnesses(point,normal,.49f,witnesses);
    failures+=(witnesses[0]-point).magnitude()>1.e-6f || (witnesses[1]-point).magnitude()>1.e-6f;
    // Signed separation is not clamped by this coordinate conversion.
    geometry.manifoldPointShape=1;geometry.normalWitnesses(point,normal,-.02f,witnesses);
    failures+=PxAbs(normal.dot(witnesses[0]-witnesses[1])+.02f)>1.e-6f;
    std::printf("[CONTACT_WITNESS] samples=%u surfaceError=%.9g derivativeError=%.9g legacyError=%.9g status=%s\n",
        samples,worst,derivativeError,legacyError,failures?"FAIL":"PASS");
    return failures?1:0;
}
