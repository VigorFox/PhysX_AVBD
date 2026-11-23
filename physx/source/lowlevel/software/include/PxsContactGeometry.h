// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef PXS_CONTACT_GEOMETRY_H
#define PXS_CONTACT_GEOMETRY_H

#include "foundation/PxMat33.h"
#include "foundation/PxTransform.h"
#include "foundation/PxUserAllocated.h"

namespace physx
{
struct PxsDeformableGeometry;
struct PxsDeformableContactOutput;
// CPU NP output, borrowed for the duration of a simulation step. A hull's
// immutable vertex storage belongs to its cooked mesh, never to the solver.
// These are complete support sets, not reduced contact-manifold witnesses.
struct PxsContactShapeGeometry
{
    enum Kind { eNONE, eVERTICES, eBOX, eSPHERE, eCAPSULE, ePLANE };
    const PxVec3* vertices;
    PxMat33 vertexToShape;
    PxTransform shapeToWorld;
    PxTransform shapeToBody;
    PxVec3 halfExtents;
    PxReal radius;
    PxU32 count;
    Kind kind;
    PxsContactShapeGeometry() : vertices(NULL), vertexToShape(PxIdentity),
        shapeToWorld(PxIdentity), shapeToBody(PxIdentity), halfExtents(0), radius(0), count(0), kind(eNONE) {}
};

struct PxsContactGeometry : public PxUserAllocated
{
    PxsContactShapeGeometry shapes[2];
    // Borrowed authored geometry, published only by NP in this output epoch.
    // Presence is NOT a complete contact result: feature rows remain required.
    const PxsDeformableGeometry* deformable[2];
    PxsDeformableContactOutput* deformableOutput;
    PxVec3d separatingAxis;
    PxReal contactDistance;
    // PCM polytope contacts report one surface point, not a shared material
    // point. PX_MAX_U32 means the generator's endpoint convention is unknown.
    PxU32 manifoldPointShape;
    bool axisValid;
    PxsContactGeometry() : deformable{NULL,NULL},deformableOutput(NULL),separatingAxis(0),contactDistance(0),manifoldPointShape(PX_MAX_U32),axisValid(false) {}
    PX_INLINE bool normalWitnesses(const PxVec3& point,const PxVec3& normal,PxReal separation,PxVec3 (&out)[2]) const
    {
        out[0]=out[1]=point;
        if(manifoldPointShape>1)return false;
        if(manifoldPointShape==1)out[0]+=normal*separation;
        else out[1]-=normal*separation;
        return true;
    }
};
}
#endif
