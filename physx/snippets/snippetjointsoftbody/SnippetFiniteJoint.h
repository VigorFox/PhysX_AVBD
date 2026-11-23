// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef SNIPPET_FINITE_JOINT_H
#define SNIPPET_FINITE_JOINT_H
#include "PxPhysicsAPI.h"
#include "SnippetFiniteJointMath.h"
namespace JointSoftBodySnippet
{
using namespace physx;

struct FiniteJointData
{
	PxTransform c2b[2];
	PxD6JointDrive drives[6];
};
inline bool finiteJointMaterialPrep(PxConstraintElasticFrameData& output,const void* block)
{
    const FiniteJointData& data=*static_cast<const FiniteJointData*>(block);
    for(PxU32 i=0;i<2;++i) output.frames[i]=data.c2b[i];
    for(PxU32 i=0;i<6;++i)
    {
        if(data.drives[i].forceLimit!=PX_MAX_F32 ||
            (data.drives[i].flags&(PxD6JointDriveFlag::eACCELERATION|PxD6JointDriveFlag::eOUTPUT_FORCE))) return false;
        output.stiffness[i]=data.drives[i].stiffness;output.damping[i]=data.drives[i].damping;
    }
    return true;
}
inline PxU32 finiteJointPrep(Px1DConstraint* rows,PxVec3p& offset,PxU32 capacity,
	PxConstraintInvMassScale& massScale,const void* block,const PxTransform& a,const PxTransform& b,
	bool,PxVec3p& anchorA,PxVec3p& anchorB)
{
	if(capacity<FINITE_JOINT_ROWS) return 0;
	const FiniteJointData& data=*static_cast<const FiniteJointData*>(block);
	const PxTransform fa=a*data.c2b[0],fb=b*data.c2b[1];
	const FiniteJointCoordinates<PxReal> coordinates=finiteJointCoordinates(fa,fb,a.p,b.p);
	anchorA=fa.p;anchorB=fb.p;offset=fb.p-a.p;massScale=PxConstraintInvMassScale(1,1,1,1);
	for(PxU32 i=0;i<FINITE_JOINT_ROWS;++i)
	{
		const PxD6JointDrive& drive=data.drives[finiteJointDriveIndex(i)];
		Px1DConstraint& row=rows[i];PxMemZero(&row,sizeof(row));
		row.flags=Px1DConstraintFlag::eSPRING|Px1DConstraintFlag::eHAS_DRIVE_LIMIT;
		if(drive.flags&PxD6JointDriveFlag::eACCELERATION) row.flags|=Px1DConstraintFlag::eACCELERATION_SPRING;
		if(drive.flags&PxD6JointDriveFlag::eOUTPUT_FORCE) row.flags|=Px1DConstraintFlag::eOUTPUT_FORCE;
		row.solveHint=PxConstraintSolveHint::eNONE;
		row.mods.spring.stiffness=drive.stiffness;row.mods.spring.damping=drive.damping;
		row.minImpulse=-drive.forceLimit;row.maxImpulse=drive.forceLimit;
		if(i>=6) row.flags|=Px1DConstraintFlag::eANGULAR_CONSTRAINT;
		row.linear0=row.linear1=coordinates.linear[i];
		row.angular0=coordinates.angularA[i];row.angular1=coordinates.angularB[i];
		row.geometricError=coordinates.value[i];
	}
	return FINITE_JOINT_ROWS;
}
// A standard PhysX constraint object owns the single constant block. No
// simulation state, neighbor graph, dual, or solver lives in this connector.
class FiniteJoint : public PxConstraintConnector
{
	PxConstraint* constraint;
	PxRigidDynamic* actors[2];
	PxTransform local[2];
	FiniteJointData data;
	FiniteJoint(PxRigidDynamic& a,const PxTransform& fa,PxRigidDynamic& b,const PxTransform& fb) : constraint(NULL)
	{
		actors[0]=&a;actors[1]=&b;local[0]=fa;local[1]=fb;
		for(PxU32 i=0;i<2;++i) data.c2b[i]=actors[i]->getCMassLocalPose().transformInv(local[i]);
	}
public:
	static FiniteJoint* create(PxPhysics& physics,PxRigidDynamic& a,const PxTransform& fa,PxRigidDynamic& b,const PxTransform& fb)
	{
		FiniteJoint* joint=new FiniteJoint(a,fa,b,fb);
        PxConstraintShaderTable shaders={finiteJointPrep,NULL,PxConstraintFlag::Enum(0)};
        shaders.elasticFramePrep=finiteJointMaterialPrep;
		joint->constraint=physics.createConstraint(&a,&b,*joint,shaders,sizeof(FiniteJointData));
		if(!joint->constraint) { delete joint;return NULL; }
		joint->constraint->setFlag(PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES,true);return joint;
	}
	void release() { constraint->release(); }
	void setDrive(PxD6Drive::Enum i,const PxD6JointDrive& drive)
	{
		// A finite limit on an authored axis has no equivalent set of
		// independent limits on its redundant smooth strain coordinates.
		if(drive.forceLimit!=PX_MAX_F32 || (drive.flags&(PxD6JointDriveFlag::eACCELERATION|PxD6JointDriveFlag::eOUTPUT_FORCE)))
		{
			PxGetFoundation().error(PxErrorCode::eINVALID_PARAMETER,PX_FL,
				"FiniteJoint material requires unlimited force drives without acceleration mode or force reporting.");
			return;
		}
		data.drives[i]=drive;constraint->markDirty();
	}
	PxD6JointDrive getDrive(PxD6Drive::Enum i) const { return data.drives[i]; }
	void getActors(PxRigidActor*& a,PxRigidActor*& b) const { a=actors[0];b=actors[1]; }
	PxTransform getLocalPose(PxJointActorIndex::Enum i) const { return local[i]; }
	PxVec3 rotationError() const { return finiteJointRotation((actors[0]->getGlobalPose()*local[0]).q,(actors[1]->getGlobalPose()*local[1]).q); }
	double elasticEnergy() const
	{
		const PxTransform a=actors[0]->getGlobalPose()*local[0],b=actors[1]->getGlobalPose()*local[1];
		const auto c=finiteJointCoordinates(a,b,a.p,b.p);
		double energy=0;for(PxU32 i=0;i<FINITE_JOINT_ROWS;++i)
			energy+=.5*double(data.drives[finiteJointDriveIndex(i)].stiffness)*c.value[i]*c.value[i];
		return energy;
	}
	PxReal getTwistAngle() const { return rotationError().x; }
	PxReal getSwingYAngle() const { return rotationError().y; }
	PxReal getSwingZAngle() const { return rotationError().z; }
	void* prepareData() PX_OVERRIDE { return &data; }
	void onConstraintRelease() PX_OVERRIDE { delete this; }
	void onComShift(PxU32 i) PX_OVERRIDE { data.c2b[i]=actors[i]->getCMassLocalPose().transformInv(local[i]);constraint->markDirty(); }
	void onOriginShift(const PxVec3&) PX_OVERRIDE {}
	void* getExternalReference(PxU32& id) PX_OVERRIDE { id=PxConcreteType::eFIRST_USER_EXTENSION+1;return this; }
	bool updatePvdProperties(pvdsdk::PvdDataStream&,const PxConstraint*,PxPvdUpdateType::Enum) const PX_OVERRIDE { return false; }
	void updateOmniPvdProperties() const PX_OVERRIDE {}
	PxBase* getSerializable() PX_OVERRIDE { return NULL; }
	PxConstraintSolverPrep getPrep() const PX_OVERRIDE { return finiteJointPrep; }
	const void* getConstantBlock() const PX_OVERRIDE { return &data; }
};
}
#endif
