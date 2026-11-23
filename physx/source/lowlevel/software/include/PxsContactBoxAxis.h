// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef PXS_CONTACT_BOX_AXIS_H
#define PXS_CONTACT_BOX_AXIS_H
#include "PxsContactGeometry.h"
#include <cfloat>
namespace physx
{
// NP candidate direction only. A consumer must still certify its complete
// geometry. In particular, a near-contact float GJK normal need not separate
// two disjoint boxes. Test the full SAT directions without a contact margin.
PX_INLINE void pxsContactBoxSeparatingAxis(PxsContactGeometry& geometry)
{
    if(geometry.shapes[0].kind!=PxsContactShapeGeometry::eBOX ||
        geometry.shapes[1].kind!=PxsContactShapeGeometry::eBOX)return;
    PxVec3d axes[2][3];
    for(PxU32 s=0;s<2;++s)
    {
        const PxQuat& stored=geometry.shapes[s].shapeToWorld.q;
        const PxQuatd q=PxQuatd(stored.x,stored.y,stored.z,stored.w).getNormalized();
        for(PxU32 d=0;d<3;++d){PxVec3d basis(0);basis[d]=1;axes[s][d]=q.rotate(basis);}
    }
    const PxVec3 a=geometry.shapes[0].shapeToWorld.p,b=geometry.shapes[1].shapeToWorld.p;
    const PxVec3d center(double(a.x)-b.x,double(a.y)-b.y,double(a.z)-b.z);
    double best=-DBL_MAX;PxVec3d selected(0);
    auto consider=[&](PxVec3d axis)
    {
        const double length=axis.magnitude();if(!(length>0))return;axis/=length;
        if(axis.dot(center)<0)axis=-axis;
        double gap=axis.dot(center);
        for(PxU32 s=0;s<2;++s)for(PxU32 d=0;d<3;++d)
            gap-=PxAbs(axis.dot(axes[s][d]))*geometry.shapes[s].halfExtents[d];
        if(gap>best){best=gap;selected=axis;}
    };
    if(geometry.axisValid)consider(geometry.separatingAxis);
    for(PxU32 s=0;s<2;++s)for(PxU32 d=0;d<3;++d)consider(axes[s][d]);
    for(PxU32 i=0;i<3;++i)for(PxU32 j=0;j<3;++j)consider(axes[0][i].cross(axes[1][j]));
    geometry.separatingAxis=selected;geometry.axisValid=selected.isFinite() && selected.magnitudeSquared()>0;
}
}
#endif
