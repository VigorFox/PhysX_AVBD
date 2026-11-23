// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef SNIPPET_VOLUME_MATERIAL_H
#define SNIPPET_VOLUME_MATERIAL_H
#include "PxPhysicsAPI.h"
#include <cmath>
namespace JointSoftBodySnippet {
using namespace physx;
// Authoring adapter only: four explicit actors and immutable rest data.
// No dynamic poses, derivatives, multipliers or solver are cached here.
class VolumeMaterial : public PxConstraintConnector
{
    PxConstraint* mConstraint;
    PxConstraintTetrahedronData mData;
    PxVec3 mActorLocalPoints[4];
    VolumeMaterial() : mConstraint(NULL) {}
    static bool prep(PxConstraintTetrahedronData& data,const void* block)
    { data=*static_cast<const PxConstraintTetrahedronData*>(block);return true; }
public:
    static VolumeMaterial* create(PxPhysics& physics,PxRigidActor* const* actors,PxReal restDet,PxReal energy)
    {
        VolumeMaterial* result=new VolumeMaterial;
        for(PxU32 i=0;i<4;++i)
        {
            result->mData.localPoints[i]=PxVec3(0);
            result->mActorLocalPoints[i]=actors[i]->is<PxRigidBody>()->getCMassLocalPose().p;
        }
        result->mData.restDeterminant=restDet;result->mData.energyScale=energy;
        PxConstraintShaderTable shaders={NULL,NULL,PxConstraintFlag::eCOLLISION_ENABLED};shaders.tetrahedronPrep=prep;
        result->mConstraint=physics.createConstraint(actors,4,*result,shaders,sizeof(result->mData));
        if(!result->mConstraint) { delete result;return NULL; }return result;
    }
    void release() { mConstraint->release(); }
    double elasticEnergy() const
    {
        PxRigidActor* actors[4];mConstraint->getActors(actors,4);PxVec3d p[4];
        for(PxU32 i=0;i<4;++i)
        {
            const PxRigidBody* b=actors[i]->is<PxRigidBody>();
            const PxVec3 point=(actors[i]->getGlobalPose()*b->getCMassLocalPose()).transform(mData.localPoints[i]);
            p[i]=PxVec3d(point.x,point.y,point.z);
        }
        const double j=(p[1]-p[0]).dot((p[2]-p[0]).cross(p[3]-p[0]))/double(mData.restDeterminant);
        return j>0 ? double(mData.energyScale)*(j-1)*(j-1)*(j-1-std::log(j)) : double(PX_MAX_F32);
    }
    void* prepareData() PX_OVERRIDE { return &mData; }
    void onConstraintRelease() PX_OVERRIDE { delete this; }
    void onComShift(PxU32 i) PX_OVERRIDE
    {
        // Keep the authored material point fixed in the actor frame when the
        // public mass frame changes; only its COM-local coordinates change.
        PxRigidActor* actors[4];mConstraint->getActors(actors,4);
        mData.localPoints[i]=actors[i]->is<PxRigidBody>()->getCMassLocalPose().transformInv(mActorLocalPoints[i]);
        mConstraint->markDirty();
    }
    void onOriginShift(const PxVec3&) PX_OVERRIDE {}
    void* getExternalReference(PxU32& id) PX_OVERRIDE { id=PxConcreteType::eFIRST_USER_EXTENSION+3;return this; }
    bool updatePvdProperties(pvdsdk::PvdDataStream&,const PxConstraint*,PxPvdUpdateType::Enum) const PX_OVERRIDE { return false; }
    void updateOmniPvdProperties() const PX_OVERRIDE {}
    PxBase* getSerializable() PX_OVERRIDE { return NULL; }
    PxConstraintSolverPrep getPrep() const PX_OVERRIDE { return NULL; }
    const void* getConstantBlock() const PX_OVERRIDE { return &mData; }
};
}
#endif
