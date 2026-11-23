// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.  

#include "PxConstraint.h"
#include "NpConstraint.h"
#include "NpPhysics.h"
#include "NpRigidDynamic.h"
#include "NpRigidStatic.h"
#include "NpArticulationLink.h"
#include "ScConstraintSim.h"
#include "ScConstraintInteraction.h"
#include "PxsSimulationController.h"
#include "DyContext.h"

using namespace physx;
using namespace Sc;

///////////////////////////////////////////////////////////////////////////////

PX_IMPLEMENT_OUTPUT_ERROR

///////////////////////////////////////////////////////////////////////////////

static PX_FORCE_INLINE PxConstraintFlags scGetFlags(const ConstraintCore& core)
{
	return core.getFlags() & (~(PxConstraintFlag::eGPU_COMPATIBLE));
}

static NpScene* getSceneFromActors(const PxRigidActor* actor0, const PxRigidActor* actor1)
{
	NpScene* s0 = NULL;
	NpScene* s1 = NULL;

	if(actor0 && (!(actor0->getActorFlags().isSet(PxActorFlag::eDISABLE_SIMULATION))))
		s0 = static_cast<NpScene*>(actor0->getScene());
	if(actor1 && (!(actor1->getActorFlags().isSet(PxActorFlag::eDISABLE_SIMULATION))))
		s1 = static_cast<NpScene*>(actor1->getScene());

#if PX_CHECKED
	if ((s0 && s1) && (s0 != s1))
		outputError<PxErrorCode::eINVALID_PARAMETER>(__LINE__, "Adding constraint to scene: Actors belong to different scenes, undefined behavior expected!");
#endif

	if ((!actor0 || s0) && (!actor1 || s1))
		return s0 ? s0 : s1;
	else
		return NULL;
}

void NpConstraint::setConstraintFunctions(PxConstraintConnector& n, const PxConstraintShaderTable& shaders)
{
    if(mCore.getTetrahedronPrep() || shaders.tetrahedronPrep || mCore.getElasticFramePrep() || shaders.elasticFramePrep)
    {
        outputError<PxErrorCode::eINVALID_OPERATION>(__LINE__, "Multi-body shader replacement is not supported; recreate the constraint.");
        return;
    }
	mCore.setConstraintFunctions(n, shaders);
	
	//update mConnectorArray, since mActor0 or mActor1 should be in external reference
	bool bNeedUpdate = false;
	if(mActor0)
	{
		NpActor& npActor = NpActor::getFromPxActor(*mActor0);
		if(npActor.findConnector(NpConnectorType::eConstraint, this) == 0xffffffff)
		{
			bNeedUpdate = true;	
			npActor.addConnector(NpConnectorType::eConstraint, this, "PxConstraint: Add to rigid actor 0: Constraint already added");
		}
	}

	if(mActor1)
	{
		NpActor& npActor = NpActor::getFromPxActor(*mActor1);
		if(npActor.findConnector(NpConnectorType::eConstraint, this) == 0xffffffff)
		{
			bNeedUpdate = true;	
			npActor.addConnector(NpConnectorType::eConstraint, this, "PxConstraint: Add to rigid actor 1: Constraint already added");
		}
	}

	if(bNeedUpdate)
	{
		NpScene* newScene = ::getSceneFromActors(mActor0, mActor1);
		NpScene* oldScene = getNpScene();

		if (oldScene != newScene)
		{
			if(oldScene)
				oldScene->removeFromConstraintList(*this);

			if(newScene)
				newScene->addToConstraintList(*this);
		}
	}
}

PxConstraintGPUIndex NpConstraint::getGPUIndex() const
{
	NpScene* scene = getNpScene();

	NP_READ_CHECK(scene);

	if (scene)
	{
		PX_ASSERT(mCore.getSim());

		return mCore.getSim()->getGPUIndex();
	}
	else
	{
		PxGetFoundation().error(PxErrorCode::eINVALID_OPERATION, PX_FL, "Constraint::getGPUIndex(): constraint has to be part of a scene to return a valid index.");

		return PX_INVALID_CONSTRAINT_GPU_INDEX;
	}
}

void NpConstraint::addConnectors(PxRigidActor* actor0, PxRigidActor* actor1)
{
    if(mCore.getTetrahedronPrep())
        for(PxU32 i=0;i<2;++i) if(mExtraActors[i])
            NpActor::getFromPxActor(*mExtraActors[i]).addConnector(NpConnectorType::eConstraint, this, "Duplicate constraint support");
	if(actor0)
		NpActor::getFromPxActor(*actor0).addConnector(NpConnectorType::eConstraint, this, "PxConstraint: Add to rigid actor 0: Constraint already added");
	if(actor1)
		NpActor::getFromPxActor(*actor1).addConnector(NpConnectorType::eConstraint, this, "PxConstraint: Add to rigid actor 1: Constraint already added");
}

void NpConstraint::removeConnectors(const char* errorMsg0, const char* errorMsg1)
{
    if(mCore.getTetrahedronPrep())
        for(PxU32 i=0;i<2;++i) if(mExtraActors[i])
            NpActor::getFromPxActor(*mExtraActors[i]).removeConnector(*mExtraActors[i], NpConnectorType::eConstraint, this, errorMsg0);
	if(mActor0)
		NpActor::getFromPxActor(*mActor0).removeConnector(*mActor0, NpConnectorType::eConstraint, this, errorMsg0);
	if(mActor1)
		NpActor::getFromPxActor(*mActor1).removeConnector(*mActor1, NpConnectorType::eConstraint, this, errorMsg1);
}

NpConstraint::NpConstraint(PxRigidActor* actor0, PxRigidActor* actor1, PxConstraintConnector& connector, const PxConstraintShaderTable& shaders, PxU32 dataSize, PxRigidActor* actor2, PxRigidActor* actor3) :
	PxConstraint(PxConcreteType::eCONSTRAINT, PxBaseFlag::eOWNS_MEMORY),
	NpBase		(NpType::eCONSTRAINT),
	mActor0		(actor0),
	mActor1		(actor1),
	mCore		(connector, shaders, dataSize)
{
    mExtraActors[0]=actor2; mExtraActors[1]=actor3;
    mMissingActorMask=0;
    mCore.setExtraBodies(actor2 ? &NpActor::getFromPxActor(*actor2).getScRigidCore() : NULL,
                         actor3 ? &NpActor::getFromPxActor(*actor3).getScRigidCore() : NULL);
	scSetFlags(shaders.flag);

	addConnectors(actor0, actor1);

	NpScene* s = getSceneFromActors();
	if (s)
	{
		if(s->isAPIWriteForbidden())
		{
			outputError<PxErrorCode::eINVALID_OPERATION>(__LINE__, "PxConstraint creation not allowed while simulation is running. Call will be ignored.");
			return;
		}

		s->addToConstraintList(*this);
	}
}

NpConstraint::~NpConstraint()
{
	if(getBaseFlags()&PxBaseFlag::eOWNS_MEMORY)
		mCore.getPxConnector()->onConstraintRelease();

	NpFactory::getInstance().onConstraintRelease(this);
}

static const char* gRemoveConnectorMsg = "PxConstraint::release(): internal error, mConnectorArray not created.";

bool NpConstraint::checkSupportWriteAllowed() const
{
    if(!mCore.getTetrahedronPrep()) return true;
    // A partially inserted constraint has no owning scene yet, but its actor
    // connectors can already belong to a running scene.
    PxRigidActor* actors[4]={mActor0,mActor1,mExtraActors[0],mExtraActors[1]};
    for(PxU32 i=0;i<4;++i) if(actors[i])
    {
        const NpScene* scene=static_cast<const NpScene*>(actors[i]->getScene());
        if(scene && scene->isAPIWriteForbidden())
        {
            outputError<PxErrorCode::eINVALID_OPERATION>(__LINE__, "Material support mutation is not allowed while any support scene is running.");
            return false;
        }
    }
    return true;
}

void NpConstraint::release()
{
	if(!checkSupportWriteAllowed()) return;
	NpScene* npScene = getNpScene();
	NP_WRITE_CHECK(npScene);

	PX_CHECK_SCENE_API_WRITE_FORBIDDEN(npScene, "PxConstraint::release() not allowed while simulation is running. Call will be ignored.")

	NpPhysics::getInstance().notifyDeletionListenersUserRelease(this, NULL);

	removeConnectors(gRemoveConnectorMsg, gRemoveConnectorMsg);

	if(npScene)
		npScene->removeFromConstraintList(*this);

	NpDestroyConstraint(this);
}

// PX_SERIALIZATION
void NpConstraint::resolveReferences(PxDeserializationContext& context)
{
	context.translatePxBase(mActor0);
	context.translatePxBase(mActor1);
}

NpConstraint* NpConstraint::createObject(PxU8*& address, PxDeserializationContext& context)
{
	NpConstraint* obj = PX_PLACEMENT_NEW(address, NpConstraint(PxBaseFlags(0)));
	address += sizeof(NpConstraint);	
	obj->importExtraData(context);
	obj->resolveReferences(context);
	return obj;
}
// ~PX_SERIALIZATION

PxScene* NpConstraint::getScene() const
{
	return getNpScene();
}

void NpConstraint::getActors(PxRigidActor*& actor0, PxRigidActor*& actor1) const
{
	NP_READ_CHECK(getNpScene());
	actor0 = mActor0;
	actor1 = mActor1;
}

static PX_INLINE void scSetBodies(ConstraintCore& core, NpActor* r0, NpActor* r1)
{
	Sc::RigidCore* scR0 = r0 ? &r0->getScRigidCore() : NULL;
	Sc::RigidCore* scR1 = r1 ? &r1->getScRigidCore() : NULL;
	core.setBodies(scR0, scR1);
}

void NpConstraint::setActors(PxRigidActor* actor0, PxRigidActor* actor1)
{
    if(mCore.getTetrahedronPrep())
    {
        outputError<PxErrorCode::eINVALID_OPERATION>(__LINE__, "setActors: supply all four material supports.");
        return;
    }
	NpScene* npScene = getNpScene();
	NP_WRITE_CHECK(npScene);
	PX_CHECK_AND_RETURN((actor0 && actor0->getConcreteType()!=PxConcreteType::eRIGID_STATIC) || (actor1 && actor1->getConcreteType()!=PxConcreteType::eRIGID_STATIC), "PxConstraint: at least one actor must be non-static");

	PX_CHECK_SCENE_API_WRITE_FORBIDDEN(npScene, "PxConstraint::setActors() not allowed while simulation is running. Call will be ignored.")

	if(mActor0 == actor0 && mActor1 == actor1)
		return;

	removeConnectors(	"PxConstraint: Add to rigid actor 0: Constraint already added",
						"PxConstraint: Add to rigid actor 1: Constraint already added");

	addConnectors(actor0, actor1);

	mActor0 = actor0;
	mActor1 = actor1;

	NpScene* newScene = ::getSceneFromActors(actor0, actor1);
	NpScene* oldScene = getNpScene();

	if(oldScene != newScene)
	{
		if(oldScene)
			oldScene->removeFromConstraintList(*this);

		scSetBodies(mCore, NpActor::getNpActor(actor0), NpActor::getNpActor(actor1));

		if(newScene)
			newScene->addToConstraintList(*this);
	}
	else
	{
		// If the constraint remains in the same scene, only a "light" update should be
		// needed. This is especially important in the context of direct GPU API, to make
		// sure the GPU index stays the same (users might have it cached).

		scSetBodies(mCore, NpActor::getNpActor(actor0), NpActor::getNpActor(actor1));
	}

	UPDATE_PVD_PROPERTY
}

PxConstraintFlags NpConstraint::getFlags() const
{
	NP_READ_CHECK(getNpScene());

	return scGetFlags(mCore);
}

void NpConstraint::setFlags(PxConstraintFlags flags)
{
	if(!checkSupportWriteAllowed()) return;
    if(mCore.getTetrahedronPrep() && !validateMaterialFlags(flags))
    {
        outputError<PxErrorCode::eINVALID_OPERATION>(__LINE__, "Unsupported four-body constraint flags.");
        return;
    }
	NpScene* npScene = getNpScene();
	NP_WRITE_CHECK(npScene);
	PX_CHECK_AND_RETURN(!(flags & PxConstraintFlag::eBROKEN), "PxConstraintFlag::eBROKEN is a read only flag");
	PX_CHECK_AND_RETURN(!(flags & PxConstraintFlag::eGPU_COMPATIBLE), "PxConstraintFlag::eGPU_COMPATIBLE is an internal flag and is illegal to set via the API");

	PX_CHECK_SCENE_API_WRITE_FORBIDDEN(npScene, "PxConstraint::setFlags() not allowed while simulation is running. Call will be ignored.")

	scSetFlags(flags);
}

void NpConstraint::setFlag(PxConstraintFlag::Enum flag, bool value)
{
	if(!checkSupportWriteAllowed()) return;
    const PxConstraintFlags proposed=value ? scGetFlags(mCore)|flag : scGetFlags(mCore)&~flag;
    if(mCore.getTetrahedronPrep() && !validateMaterialFlags(proposed))
    {
        outputError<PxErrorCode::eINVALID_OPERATION>(__LINE__, "Unsupported four-body constraint flags.");
        return;
    }
	NpScene* npScene = getNpScene();
	NP_WRITE_CHECK(npScene);
	PX_CHECK_AND_RETURN(flag != PxConstraintFlag::eBROKEN, "PxConstraintFlag::eBROKEN is a read only flag");
	PX_CHECK_AND_RETURN(flag != PxConstraintFlag::eGPU_COMPATIBLE, "PxConstraintFlag::eGPU_COMPATIBLE is an internal flag and is illegal to set via the API");

	PX_CHECK_SCENE_API_WRITE_FORBIDDEN(npScene, "PxConstraint::setFlag() not allowed while simulation is running. Call will be ignored.")

	const PxConstraintFlags f = scGetFlags(mCore);
	scSetFlags(value ? f|flag : f&~flag);
}

void NpConstraint::getForce(PxVec3& linear, PxVec3& angular) const
{
	if(mCore.getTetrahedronPrep() || mCore.getElasticFramePrep())
	{
		linear=angular=PxVec3(0);
		outputError<PxErrorCode::eINVALID_OPERATION>(__LINE__, "Aggregate force reporting is not supported for typed materials.");
		return;
	}
	NP_READ_CHECK(getNpScene());

	PX_CHECK_SCENE_API_READ_FORBIDDEN_EXCEPT_COLLIDE(getNpScene(), "PxConstraint::getForce() not allowed while simulation is running (except during PxScene::collide()).");

	mCore.getForce(linear, angular);
}

void NpConstraint::markDirty()
{
	if(!checkSupportWriteAllowed()) return;
#ifdef NEW_DIRTY_SHADERS_CODE
	if(mCore.getFlags() & PxConstraintFlag::eALWAYS_UPDATE)
		return;

	if(!mCore.isDirty())
	{
		NpScene* npScene = getNpScene();
		if(npScene)
			npScene->addDirtyConstraint(this);
		mCore.setDirty();
	}
#else
	mCore.setDirty();
#endif
}

void NpConstraint::updateConstants(PxsSimulationController& simController)
{
	if(!mCore.isDirty() && !(mCore.getFlags() & PxConstraintFlag::eALWAYS_UPDATE))
		return;

	PX_ASSERT(!isAPIWriteForbidden());

	Sc::ConstraintSim* sim = mCore.getSim();
	if(sim)
	{
		Dy::Constraint& LLC = sim->getLowLevelConstraint();
		PxMemCopy(LLC.constantBlock, mCore.getPxConnector()->prepareData(), LLC.constantBlockSize);
		simController.updateJoint(sim->getInteraction()->getEdgeIndex(), &LLC);
	}

	mCore.clearDirty();

#if PX_SUPPORT_PVD
	NpScene* npScene = getNpScene();
	//Changed to use the visual scenes update system which respects
	//the debugger's connection type flag.
	if(npScene)
		npScene->getScenePvdClientInternal().updatePvdProperties(this);
#endif
}

void NpConstraint::setBreakForce(PxReal linear, PxReal angular)
{
	if(!checkSupportWriteAllowed()) return;
	if((mCore.getTetrahedronPrep() || mCore.getElasticFramePrep()) && (!PxIsFinite(linear) || !PxIsFinite(angular) ||
        linear!=PX_MAX_F32 || angular!=PX_MAX_F32))
    {
        outputError<PxErrorCode::eINVALID_OPERATION>(__LINE__, "Multi-body break force is not supported.");
        return;
    }
	NpScene* npScene = getNpScene();
	NP_WRITE_CHECK(npScene);

	PX_CHECK_SCENE_API_WRITE_FORBIDDEN(npScene, "PxConstraint::setBreakForce() not allowed while simulation is running. Call will be ignored.")

	mCore.setBreakForce(linear, angular);
	markDirty();
	UPDATE_PVD_PROPERTY
}

void NpConstraint::getBreakForce(PxReal& linear, PxReal& angular) const
{
	NP_READ_CHECK(getNpScene());
	mCore.getBreakForce(linear, angular);
}

void NpConstraint::setMinResponseThreshold(PxReal threshold)
{
	if(!checkSupportWriteAllowed()) return;
	if((mCore.getTetrahedronPrep() || mCore.getElasticFramePrep()) && threshold!=0)
	{
		outputError<PxErrorCode::eINVALID_OPERATION>(__LINE__, "Row response thresholds are not supported for four-body materials.");
		return;
	}
	NpScene* npScene = getNpScene();
	NP_WRITE_CHECK(npScene);
	PX_CHECK_AND_RETURN(PxIsFinite(threshold) && threshold>=0, "PxConstraint::setMinResponseThreshold: threshold must be non-negative");

	PX_CHECK_SCENE_API_WRITE_FORBIDDEN(npScene, "PxConstraint::setMinResponseThreshold() not allowed while simulation is running. Call will be ignored.")

	mCore.setMinResponseThreshold(threshold);
	UPDATE_PVD_PROPERTY
}

PxReal NpConstraint::getMinResponseThreshold() const
{
	NP_READ_CHECK(getNpScene());
	return mCore.getMinResponseThreshold();
}

bool NpConstraint::isValid() const
{
	NP_READ_CHECK(getNpScene());
    if(mCore.getTetrahedronPrep())
    {
        if(mMissingActorMask) return false;
        PxRigidActor* actors[4]={mActor0,mActor1,mExtraActors[0],mExtraActors[1]};
        for(PxU32 i=0;i<4;++i)
            if(actors[i] && actors[i]->getConcreteType()==PxConcreteType::eRIGID_DYNAMIC) return true;
        return false;
    }
	const bool isValid0 = mActor0 && mActor0->getConcreteType()!=PxConcreteType::eRIGID_STATIC;
	const bool isValid1 = mActor1 && mActor1->getConcreteType()!=PxConcreteType::eRIGID_STATIC;
	return isValid0 || isValid1;
}

void* NpConstraint::getExternalReference(PxU32& typeID)
{
	NP_READ_CHECK(getNpScene());
	return mCore.getPxConnector()->getExternalReference(typeID);
}

void NpConstraint::comShift(PxRigidActor* actor)
{
    if(mCore.getTetrahedronPrep())
    {
        PxRigidActor* actors[4]={mActor0,mActor1,mExtraActors[0],mExtraActors[1]};
        for(PxU32 i=0;i<4;++i) if(actors[i]==actor) mCore.getPxConnector()->onComShift(i);
        return;
    }
	PX_ASSERT(actor == mActor0 || actor == mActor1);
	PxConstraintConnector* connector = mCore.getPxConnector();
	if(actor == mActor0)
		connector->onComShift(0);
	if(actor == mActor1)
		connector->onComShift(1);
}

void NpConstraint::actorDeleted(PxRigidActor* actor)
{
    if(mCore.getTetrahedronPrep())
    {
        PxRigidActor** slots[4]={&mActor0,&mActor1,&mExtraActors[0],&mExtraActors[1]};
        for(PxU32 i=0;i<4;++i) if(*slots[i]==actor) { *slots[i]=NULL; mMissingActorMask|=PxU8(1u<<i); }
        mCore.setExtraBodies(mExtraActors[0] ? &NpActor::getFromPxActor(*mExtraActors[0]).getScRigidCore() : NULL,
                             mExtraActors[1] ? &NpActor::getFromPxActor(*mExtraActors[1]).getScRigidCore() : NULL);
        return;
    }
	// the actor cannot be deleted without also removing it from the scene,
	// which means that the joint will also have been removed from the scene,
	// so we can just reset the actor here.
	PX_ASSERT(actor == mActor0 || actor == mActor1);

	if(actor == mActor0)
		mActor0 = NULL;
	else
		mActor1 = NULL;
}

NpScene* NpConstraint::getSceneFromActors() const
{
    if(!mCore.getTetrahedronPrep()) return ::getSceneFromActors(mActor0, mActor1);
    if(mMissingActorMask) return NULL;
    PxRigidActor* actors[4]={mActor0,mActor1,mExtraActors[0],mExtraActors[1]};
    NpScene* scene=NULL;
    for(PxU32 i=0;i<4;++i) if(actors[i])
    {
        NpScene* next=static_cast<NpScene*>(actors[i]->getScene());
        if(!next || (actors[i]->getActorFlags() & PxActorFlag::eDISABLE_SIMULATION)) return NULL;
        if(scene && scene!=next)
        {
            outputError<PxErrorCode::eINVALID_PARAMETER>(__LINE__, "Material supports cannot span scenes.");
            return NULL;
        }
        scene=next;
    }
    if(scene && (!scene->getScScene().getDynamicsContext()->supportsTetrahedralConstraints() ||
        (scene->getFlagsFast() & PxSceneFlag::eENABLE_GPU_DYNAMICS)))
    {
        outputError<PxErrorCode::eINVALID_OPERATION>(__LINE__, "The selected backend does not support four-body materials.");
        return NULL;
    }
    return scene;
}

PxU32 NpConstraint::getNbActors() const
{
    NP_READ_CHECK(getNpScene());
    return mCore.getTetrahedronPrep() ? 4u : 2u;
}

PxU32 NpConstraint::getActors(PxRigidActor** actors, PxU32 capacity) const
{
    NP_READ_CHECK(getNpScene());
    if(!actors) return 0;
    PxRigidActor* all[4]={mActor0,mActor1,mExtraActors[0],mExtraActors[1]};
    const PxU32 count=PxMin(capacity,getNbActors());
    for(PxU32 i=0;i<count;++i) actors[i]=all[i];
    return count;
}

bool NpConstraint::validateActors(PxRigidActor* const* actors, PxU32 count)
{
    if(!actors || count!=4) return false;
    NpScene* scene=NULL;
    bool dynamic=false;
    for(PxU32 i=0;i<count;++i)
    {
        for(PxU32 j=0;j<i;++j) if(actors[i]==actors[j]) return false;
        if(!actors[i]) continue;
        const PxType type=actors[i]->getConcreteType();
        if(type!=PxConcreteType::eRIGID_STATIC && type!=PxConcreteType::eRIGID_DYNAMIC) return false;
        dynamic|=type==PxConcreteType::eRIGID_DYNAMIC;
        NpScene* next=static_cast<NpScene*>(actors[i]->getScene());
        if(!next) continue;
        if(next->isAPIWriteForbidden() || (scene && scene!=next) ||
           !next->getScScene().getDynamicsContext()->supportsTetrahedralConstraints() ||
           (next->getFlagsFast() & PxSceneFlag::eENABLE_GPU_DYNAMICS)) return false;
        scene=next;
    }
    return dynamic;
}

bool NpConstraint::validateMaterialFlags(PxConstraintFlags flags)
{
    const PxU16 allowed=PxConstraintFlag::eCOLLISION_ENABLED|PxConstraintFlag::eALWAYS_UPDATE|PxConstraintFlag::eDISABLE_CONSTRAINT;
    return flags.isSet(PxConstraintFlag::eCOLLISION_ENABLED) && !(PxU16(flags)&~allowed);
}

bool NpConstraint::setActors(PxRigidActor* const* actors, PxU32 count)
{
	if(!checkSupportWriteAllowed()) return false;
    NpScene* scene=getNpScene();
    NP_WRITE_CHECK(scene);
    if(!mCore.getTetrahedronPrep())
    {
        if(!actors || count!=2 || (scene && scene->isAPIWriteForbidden())) return false;
        setActors(actors[0],actors[1]);
        return mActor0==actors[0] && mActor1==actors[1];
    }
    if((scene && scene->isAPIWriteForbidden()) || !validateActors(actors,count))
    {
        outputError<PxErrorCode::eINVALID_OPERATION>(__LINE__, "setActors: invalid material support set or simulation is running.");
        return false;
    }
    // Copy before mutation; the caller may have passed overlapping storage.
    PxRigidActor* copy[4]={actors[0],actors[1],actors[2],actors[3]};
    if(scene) scene->removeFromConstraintList(*this);
    removeConnectors(gRemoveConnectorMsg,gRemoveConnectorMsg);
    mActor0=copy[0]; mActor1=copy[1]; mExtraActors[0]=copy[2]; mExtraActors[1]=copy[3];
    mMissingActorMask=0;
    mCore.setExtraBodies(copy[2] ? &NpActor::getFromPxActor(*copy[2]).getScRigidCore() : NULL,
                         copy[3] ? &NpActor::getFromPxActor(*copy[3]).getScRigidCore() : NULL);
    addConnectors(mActor0,mActor1);
    markDirty();
    NpScene* next=getSceneFromActors();
    if(next) next->addToConstraintList(*this);
    return true;
}
