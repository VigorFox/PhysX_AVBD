// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef DY_AVBD_TETRAHEDRON_H
#define DY_AVBD_TETRAHEDRON_H

#include "foundation/PxVec3.h"
#include "foundation/PxMath.h"
#include <cmath>
#include <cfloat>

namespace physx { namespace Dy {

// Rest-only payload. Supports/anchors remain in the common row program.
struct AvbdTetrahedronMaterial
{
    double inverseRestDeterminant;
    double energyScale;
    PxU32 supportStart;
};

struct AvbdTetrahedronEvaluation
{
    double ratio, energy, gradient, curvature;
    PxVec3d jacobians[4];
    bool valid;
};

PX_FORCE_INLINE AvbdTetrahedronEvaluation avbdEvaluateTetrahedron(
    const AvbdTetrahedronMaterial& material, const PxVec3d* points)
{
    AvbdTetrahedronEvaluation r = {};
    const PxVec3d a=points[1]-points[0], b=points[2]-points[0], c=points[3]-points[0];
    r.ratio=a.dot(b.cross(c))*material.inverseRestDeterminant;
    r.valid=PxIsFinite(r.ratio) && r.ratio>0.0 && material.energyScale>0.0;
    if(!r.valid) return r;
    const double d=r.ratio-1.0;
    const double h=d-(r.ratio>0.5 ? std::log1p(d) : std::log(r.ratio));
    r.energy=material.energyScale*d*d*h;
    r.gradient=material.energyScale*(2*d*h+d*d*d/r.ratio);
    r.curvature=material.energyScale*(2*h+4*d*d/r.ratio+d*d/(r.ratio*r.ratio));
    r.jacobians[1]=b.cross(c)*material.inverseRestDeterminant;
    r.jacobians[2]=c.cross(a)*material.inverseRestDeterminant;
    r.jacobians[3]=a.cross(b)*material.inverseRestDeterminant;
    r.jacobians[0]=-r.jacobians[1]-r.jacobians[2]-r.jacobians[3];
    r.valid=PxIsFinite(r.energy) && PxIsFinite(r.gradient) && PxIsFinite(r.curvature);
    return r;
}

// The determinant along four linear point paths is a cubic Bezier polynomial.
// Positive Bernstein coefficients certify the entire segment, not just its end.
// curveError bounds deviation of each rigid anchor from its endpoint chord.
PX_FORCE_INLINE bool avbdTetrahedronPathAdmissible(const AvbdTetrahedronMaterial& m,
    const PxVec3d* p, const PxVec3d* q, const double* curveError)
{
    PxVec3d e[2][3];
    double bound[3], length[3];
    PxU32 moved=0;
    bool curved=false;
    for(PxU32 i=0;i<4;++i)
    {
        moved+=p[i]!=q[i];
        curved|=curveError[i]!=0.0;
    }
    double magnitudeProduct=1;
    for(PxU32 i=0;i<3;++i)
    {
        e[0][i]=p[i+1]-p[0]; e[1][i]=q[i+1]-q[0];
        bound[i]=curveError[0]+curveError[i+1];
        length[i]=curved ? PxMax(e[0][i].magnitude(),e[1][i].magnitude()) : 0.0;
        magnitudeProduct*=PxMax(PxAbs(e[0][i].x)+PxAbs(e[0][i].y)+PxAbs(e[0][i].z),
                                PxAbs(e[1][i].x)+PxAbs(e[1][i].y)+PxAbs(e[1][i].z));
    }
    // Forward error envelope for edge subtraction, triple products, summing
    // at most three Bernstein terms, division by three and rest scaling.
    // The L1 product bounds every absolute monomial sum; 64 eps covers this
    // operation chain. This is arithmetic uncertainty, not a volume tolerance.
    const double roundoff=64*DBL_EPSILON*magnitudeProduct*PxAbs(m.inverseRestDeterminant);
    if(!curved && moved<=1)
    {
        // A determinant is affine in any one of its four points. For a
        // straight one-point path, endpoint positivity certifies the segment.
        const double start=e[0][0].dot(e[0][1].cross(e[0][2]))*m.inverseRestDeterminant;
        const double end=e[1][0].dot(e[1][1].cross(e[1][2]))*m.inverseRestDeterminant;
        return PxIsFinite(start) && PxIsFinite(end) && start>roundoff && end>roundoff;
    }
    double coeff[4]={0.0,0.0,0.0,0.0};
    for(PxU32 a=0;a<2;++a) for(PxU32 b=0;b<2;++b) for(PxU32 c=0;c<2;++c)
        coeff[a+b+c]+=e[a][0].dot(e[b][1].cross(e[c][2]))*m.inverseRestDeterminant;
    coeff[1]/=3.0; coeff[2]/=3.0;
    const double error=roundoff+(1+64*DBL_EPSILON)*PxAbs(m.inverseRestDeterminant)*(
        bound[0]*length[1]*length[2]+length[0]*bound[1]*length[2]+length[0]*length[1]*bound[2]+
        bound[0]*bound[1]*length[2]+bound[0]*length[1]*bound[2]+length[0]*bound[1]*bound[2]+
        bound[0]*bound[1]*bound[2]);
    for(PxU32 i=0;i<4;++i) if(!PxIsFinite(coeff[i]) || !(coeff[i]>error)) return false;
    return true;
}

} }
#endif
