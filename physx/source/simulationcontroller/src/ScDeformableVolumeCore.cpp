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

#include "foundation/PxPreprocessor.h"
#include "ScDeformableVolumeCore.h"
#include "ScPhysics.h"

#if PX_SUPPORT_GPU_PHYSX
#include "ScDeformableSurfaceCore.h"
#include "ScDeformableVolumeSim.h"
#include "DyDeformableVolume.h"
#include "GuTetrahedronMesh.h"
#include "GuBV4.h"
#include "geometry/PxTetrahedronMesh.h"
#endif

using namespace physx;

static PX_FORCE_INLINE void requestCpuAvbdWake(
	Dy::DeformableBodyCore& core)
{
	core.cpuAvbdSleeping = false;
	core.cpuAvbdWakeRequested = true;
}

Sc::DeformableVolumeCore::DeformableVolumeCore() :
	ActorCore(PxActorType::eDEFORMABLE_VOLUME, PxActorFlag::eVISUALIZATION, PX_DEFAULT_CLIENT, 0),
	mGpuMemStat(0)
{
	const PxTolerancesScale& scale = Physics::getInstance().getTolerancesScale();

	// Dy::DeformableCore
	mCore.sleepThreshold = 5e-5f * scale.speed * scale.speed;
	mCore.solverIterationCounts = (1 << 8) | 4;
	mCore.wakeCounter = Physics::sWakeCounterOnCreation;
	mCore.dirty = true;
	
	// Dy::DeformableVolumeCore
	mCore.freezeThreshold = 5e-6f * scale.speed * scale.speed;
}


Sc::DeformableVolumeCore::~DeformableVolumeCore() { }

void Sc::DeformableVolumeCore::
initializeCpuAvbdSimulationRestPositions(
	const PxVec4* positions,
	PxU32 positionCount)
{
	if(!mCpuAvbdSimulationRestPositions.empty() ||
		!positions || positionCount == 0)
		return;

	mCpuAvbdSimulationRestPositions.resize(positionCount);
	for(PxU32 i = 0; i < positionCount; ++i)
		mCpuAvbdSimulationRestPositions[i] =
			positions[i].getXYZ();
}

void Sc::DeformableVolumeCore::
clearCpuAvbdSimulationRestPositions()
{
	mCpuAvbdSimulationRestPositions.clear();
}

/////////////////////////////////////////////////////////////////////////////////////////
// PxActor API
/////////////////////////////////////////////////////////////////////////////////////////

void Sc::DeformableVolumeCore::setActorFlags(PxActorFlags flags)
{
	mCore.actorFlags = flags;
	mCore.dirty = true;
	requestCpuAvbdWake(mCore);
}

/////////////////////////////////////////////////////////////////////////////////////////
// PxDeformableBody API
/////////////////////////////////////////////////////////////////////////////////////////

void Sc::DeformableVolumeCore::setBodyFlags(PxDeformableBodyFlags flags)
{
#if PX_SUPPORT_GPU_PHYSX
	Sc::DeformableVolumeSim* sim = getSim();
	if (sim)
	{
		const bool wasDisabledSelfCollision = mCore.bodyFlags & PxDeformableBodyFlag::eDISABLE_SELF_COLLISION;
		const bool isDisabledSelfCollision = flags & PxDeformableBodyFlag::eDISABLE_SELF_COLLISION;

		if (wasDisabledSelfCollision != isDisabledSelfCollision)
		{
			if (isDisabledSelfCollision)
				sim->disableSelfCollision();
			else
				sim->enableSelfCollision();
		}
	}
#endif

	mCore.bodyFlags = flags;
	mCore.dirty = true;
	requestCpuAvbdWake(mCore);
}

void Sc::DeformableVolumeCore::setLinearDamping(const PxReal v)
{
	mCore.linearDamping = v;
	mCore.dirty = true;
	requestCpuAvbdWake(mCore);
}

void Sc::DeformableVolumeCore::setMaxLinearVelocity(const PxReal v)
{
	mCore.maxLinearVelocity = v;
	mCore.dirty = true;
	requestCpuAvbdWake(mCore);
}

void Sc::DeformableVolumeCore::setMaxPenetrationBias(const PxReal v)
{
	mCore.maxPenetrationBias = v;
	mCore.dirty = true;
	requestCpuAvbdWake(mCore);
}

void Sc::DeformableVolumeCore::setSolverIterationCounts(const PxU16 c)
{
	mCore.solverIterationCounts = c;
	mCore.dirty = true;
	requestCpuAvbdWake(mCore);
}

void Sc::DeformableVolumeCore::setSleepThreshold(const PxReal v)
{
	mCore.sleepThreshold = v;
	mCore.dirty = true;
	requestCpuAvbdWake(mCore);
}

void Sc::DeformableVolumeCore::setSettlingThreshold(const PxReal v)
{
	mCore.settlingThreshold = v;
	mCore.dirty = true;
	requestCpuAvbdWake(mCore);
}

void Sc::DeformableVolumeCore::setSettlingDamping(const PxReal v)
{
	mCore.settlingDamping = v;
	mCore.dirty = true;
	requestCpuAvbdWake(mCore);
}

void Sc::DeformableVolumeCore::setSelfCollisionFilterDistance(const PxReal v)
{
	mCore.selfCollisionFilterDistance = v;
	mCore.dirty = true;
	requestCpuAvbdWake(mCore);
}

void Sc::DeformableVolumeCore::setWakeCounter(const PxReal v)
{
	setWakeCounterInternal(v);
}

void Sc::DeformableVolumeCore::setWakeCounterInternal(const PxReal v)
{
	mCore.wakeCounter = v;
	mCore.dirty = true;
	if(v > 0.0f)
	{
		mCore.cpuAvbdSleeping = false;
		mCore.cpuAvbdWakeRequested = true;
	}

#if PX_SUPPORT_GPU_PHYSX
	Sc::DeformableVolumeSim* sim = getSim();
	if (sim)
	{
		sim->onSetWakeCounter();
	}
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////
// PxDeformableVolume API
/////////////////////////////////////////////////////////////////////////////////////////

void Sc::DeformableVolumeCore::setVolumeFlags(PxDeformableVolumeFlags flags)
{
	mCore.volumeFlags = flags;
	mCore.dirty = true;
	requestCpuAvbdWake(mCore);
}

void Sc::DeformableVolumeCore::setSelfCollisionStressTolerance(const PxReal v)
{
	mCore.selfCollisionStressTolerance = v;
	mCore.dirty = true;
	requestCpuAvbdWake(mCore);
}

void Sc::DeformableVolumeCore::setKinematicTargets(const PxVec4* positions)
{
	mCore.kinematicTarget = positions;
	mCore.dirty = true;
	requestCpuAvbdWake(mCore);
}

PxU32 Sc::DeformableVolumeCore::getGpuIndex() const
{
#if PX_SUPPORT_GPU_PHYSX
	const Sc::DeformableVolumeSim* sim = getSim();
	return sim ? sim->getGpuIndex() : 0xffffffff;
#else
	return 0xffffffff;
#endif
}

PxU32 Sc::DeformableVolumeCore::addRigidAttachment(Sc::BodyCore* core, PxU32 particleId, const PxVec3& actorSpacePose, bool doConversion)
{
	PxU32 handle = 0xFFFFFFFF;
#if PX_SUPPORT_GPU_PHYSX
	Sc::DeformableVolumeSim* sim = getSim();
	if(sim)
		handle = sim->getScene().addRigidAttachment(core, *sim, particleId, actorSpacePose, doConversion);
#else
	PX_UNUSED(core);
	PX_UNUSED(particleId);
	PX_UNUSED(actorSpacePose);
	PX_UNUSED(doConversion);
#endif
	return handle;
}

void Sc::DeformableVolumeCore::removeRigidAttachment(Sc::BodyCore* core, PxU32 handle)
{
#if PX_SUPPORT_GPU_PHYSX
	Sc::DeformableVolumeSim* sim = getSim();
	if (sim)
	{
		sim->getScene().removeRigidAttachment(core, *sim, handle);
		setWakeCounterInternal(ScInternalWakeCounterResetValue);
	}
#else
	PX_UNUSED(core);
	PX_UNUSED(handle);
#endif
}


void Sc::DeformableVolumeCore::addTetRigidFilter(Sc::BodyCore* core, PxU32 tetIdx)
{
#if PX_SUPPORT_GPU_PHYSX
	Sc::DeformableVolumeSim* sim = getSim();
	if (sim)
		sim->getScene().addTetRigidFilter(core, *sim, tetIdx);
#else
	PX_UNUSED(core);
	PX_UNUSED(tetIdx);
#endif
}

void Sc::DeformableVolumeCore::removeTetRigidFilter(Sc::BodyCore* core, PxU32 tetIdx)
{
#if PX_SUPPORT_GPU_PHYSX
	Sc::DeformableVolumeSim* sim = getSim();
	if (sim)
	{
		sim->getScene().removeTetRigidFilter(core, *sim, tetIdx);
	}
#else
	PX_UNUSED(core);
	PX_UNUSED(tetIdx);
#endif
}

PxU32 Sc::DeformableVolumeCore::addTetRigidAttachment(Sc::BodyCore* core, PxU32 tetIdx, const PxVec4& barycentric,
	const PxVec3& actorSpacePose, bool doConversion)
{
	PxU32 handle = 0xFFFFFFFF;
#if PX_SUPPORT_GPU_PHYSX
	Sc::DeformableVolumeSim* sim = getSim();
	if (sim)
		handle = sim->getScene().addTetRigidAttachment(core, *sim, tetIdx, barycentric, actorSpacePose, doConversion);
#else
	PX_UNUSED(core);
	PX_UNUSED(tetIdx);
	PX_UNUSED(barycentric);
	PX_UNUSED(actorSpacePose);
	PX_UNUSED(doConversion);
#endif
	return handle;
}

void Sc::DeformableVolumeCore::addSoftBodyFilter(Sc::DeformableVolumeCore& core, PxU32 tetIdx0, PxU32 tetIdx1)
{
#if PX_SUPPORT_GPU_PHYSX
	Sc::DeformableVolumeSim* sim = getSim();
	if (sim)
		sim->getScene().addSoftBodyFilter(core, tetIdx0, *sim, tetIdx1);
#else
	PX_UNUSED(core);
	PX_UNUSED(tetIdx0);
	PX_UNUSED(tetIdx1);
#endif
}

void Sc::DeformableVolumeCore::removeSoftBodyFilter(Sc::DeformableVolumeCore& core, PxU32 tetIdx0, PxU32 tetIdx1)
{
#if PX_SUPPORT_GPU_PHYSX
	Sc::DeformableVolumeSim* sim = getSim();
	if (sim)
		sim->getScene().removeSoftBodyFilter(core, tetIdx0, *sim, tetIdx1);
#else
	PX_UNUSED(core);
	PX_UNUSED(tetIdx0);
	PX_UNUSED(tetIdx1);
#endif
}

void Sc::DeformableVolumeCore::addSoftBodyFilters(Sc::DeformableVolumeCore& core, PxU32* tetIndices0, PxU32* tetIndices1, PxU32 tetIndicesSize)
{
#if PX_SUPPORT_GPU_PHYSX
	Sc::DeformableVolumeSim* sim = getSim();
	if (sim)
		sim->getScene().addSoftBodyFilters(core, *sim, tetIndices0, tetIndices1, tetIndicesSize);
#else
	PX_UNUSED(core);
	PX_UNUSED(tetIndices0);
	PX_UNUSED(tetIndices1);
	PX_UNUSED(tetIndicesSize);
#endif
}

void Sc::DeformableVolumeCore::removeSoftBodyFilters(Sc::DeformableVolumeCore& core, PxU32* tetIndices0, PxU32* tetIndices1, PxU32 tetIndicesSize)
{
#if PX_SUPPORT_GPU_PHYSX
	Sc::DeformableVolumeSim* sim = getSim();
	if (sim)
		sim->getScene().removeSoftBodyFilters(core, *sim, tetIndices0, tetIndices1, tetIndicesSize);
#else
	PX_UNUSED(core);
	PX_UNUSED(tetIndices0);
	PX_UNUSED(tetIndices1);
	PX_UNUSED(tetIndicesSize);
#endif
}


PxU32 Sc::DeformableVolumeCore::addSoftBodyAttachment(Sc::DeformableVolumeCore& core, PxU32 tetIdx0, const PxVec4& triBarycentric0, PxU32 tetIdx1, const PxVec4& tetBarycentric1,
	bool doConversion)
{
	PxU32 handle = 0xFFFFFFFF;
#if PX_SUPPORT_GPU_PHYSX
	Sc::DeformableVolumeSim* sim = getSim();
	if (sim)
		handle = sim->getScene().addSoftBodyAttachment(core, tetIdx0, triBarycentric0, *sim, tetIdx1, tetBarycentric1, doConversion);
#else
	PX_UNUSED(core);
	PX_UNUSED(tetIdx0);
	PX_UNUSED(triBarycentric0);
	PX_UNUSED(tetIdx1);
	PX_UNUSED(tetBarycentric1);
	PX_UNUSED(doConversion);
#endif
	return handle;
}

void Sc::DeformableVolumeCore::removeSoftBodyAttachment(Sc::DeformableVolumeCore& core, PxU32 handle)
{
#if PX_SUPPORT_GPU_PHYSX
	Sc::DeformableVolumeSim* sim = getSim();
	setWakeCounterInternal(ScInternalWakeCounterResetValue);
	core.setWakeCounterInternal(ScInternalWakeCounterResetValue);
	if (sim)
		sim->getScene().removeSoftBodyAttachment(core, *sim, handle);
#else
	PX_UNUSED(core);
	PX_UNUSED(handle);
#endif
}


void Sc::DeformableVolumeCore::addClothFilter(Sc::DeformableSurfaceCore& core, PxU32 triIdx, PxU32 tetIdx)
{
#if PX_SUPPORT_GPU_PHYSX
	Sc::DeformableVolumeSim* sim = getSim();

	if (sim)
		sim->getScene().addClothFilter(core, triIdx, *sim, tetIdx);
#else
	PX_UNUSED(core);
	PX_UNUSED(triIdx);
	PX_UNUSED(tetIdx);
#endif
}

void Sc::DeformableVolumeCore::removeClothFilter(Sc::DeformableSurfaceCore& core, PxU32 triIdx, PxU32 tetIdx)
{
#if PX_SUPPORT_GPU_PHYSX
	Sc::DeformableVolumeSim* sim = getSim();
	if (sim)
		sim->getScene().removeClothFilter(core, triIdx, *sim, tetIdx);
#else
	PX_UNUSED(core);
	PX_UNUSED(triIdx);
	PX_UNUSED(tetIdx);
#endif
}

PxU32 Sc::DeformableVolumeCore::addClothAttachment(Sc::DeformableSurfaceCore& core, PxU32 triIdx, const PxVec4& triBarycentric, PxU32 tetIdx, const PxVec4& tetBarycentric,
	bool doConversion)
{
	PxU32 handle = 0xFFFFFFFF;
#if PX_SUPPORT_GPU_PHYSX
	Sc::DeformableVolumeSim* sim = getSim();
	if (sim)
		handle = sim->getScene().addClothAttachment(core, triIdx, triBarycentric, *sim, tetIdx, tetBarycentric, doConversion);
#else
	PX_UNUSED(core);
	PX_UNUSED(triIdx);
	PX_UNUSED(triBarycentric);
	PX_UNUSED(tetIdx);
	PX_UNUSED(tetBarycentric);
	PX_UNUSED(doConversion);
#endif
	return handle;
}

void Sc::DeformableVolumeCore::removeClothAttachment(Sc::DeformableSurfaceCore& core, PxU32 handle)
{
#if PX_SUPPORT_GPU_PHYSX
	Sc::DeformableVolumeSim* sim = getSim();
	setWakeCounterInternal(ScInternalWakeCounterResetValue);
	core.setWakeCounterInternal(ScInternalWakeCounterResetValue);
	if (sim)
		sim->getScene().removeClothAttachment(core, *sim, handle);
#else
	PX_UNUSED(core);
	PX_UNUSED(handle);
#endif
}

//---------------------------------------------------------------------------------
// Internal API
//---------------------------------------------------------------------------------

void Sc::DeformableVolumeCore::addMaterial(const PxU16 handle)
{
	mCore.materialHandles.pushBack(handle);
	mCore.dirty = true;
}

void Sc::DeformableVolumeCore::clearMaterials()
{
	mCore.materialHandles.clear();
	mCore.dirty = true;
}

PxActor* Sc::DeformableVolumeCore::getPxActor() const
{
	return PxPointerOffset<PxActor*>(const_cast<DeformableVolumeCore*>(this), gOffsetTable.scCore2PxActor[getActorCoreType()]);
}

void Sc::DeformableVolumeCore::attachShapeCore(ShapeCore* shapeCore)
{
#if PX_SUPPORT_GPU_PHYSX
	Sc::DeformableVolumeSim* sim = getSim();
	if (sim)
	{
		sim->attachShapeCore(shapeCore);
		mCore.dirty = true;
	}
#else
	PX_UNUSED(shapeCore);
#endif
}

void Sc::DeformableVolumeCore::attachSimulationMesh(PxTetrahedronMesh* simulationMesh, PxDeformableVolumeAuxData* simulationState)
{
#if PX_SUPPORT_GPU_PHYSX
	Sc::DeformableVolumeSim* sim = getSim();
	if (sim)
	{
		sim->attachSimulationMesh(simulationMesh, simulationState);
		mCore.dirty = true;
	}
#else
	PX_UNUSED(simulationMesh);
	PX_UNUSED(simulationState);
#endif
}

void Sc::DeformableVolumeCore::onShapeChange(ShapeCore& shape, ShapeChangeNotifyFlags notifyFlags)
{
#if PX_SUPPORT_GPU_PHYSX
	PX_UNUSED(shape);
	DeformableVolumeSim* sim = getSim();
	if (!sim)
		return;
	ShapeSimBase& s = sim->getShapeSim();

	if (notifyFlags & ShapeChangeNotifyFlag::eGEOMETRY)
		s.onVolumeOrTransformChange();
	if (notifyFlags & ShapeChangeNotifyFlag::eRESET_FILTERING)
		s.onResetFiltering();
	if (notifyFlags & ShapeChangeNotifyFlag::eSHAPE2BODY)
		s.onVolumeOrTransformChange();
	if (notifyFlags & ShapeChangeNotifyFlag::eFILTERDATA)
		s.onFilterDataChange();
	if (notifyFlags & ShapeChangeNotifyFlag::eCONTACTOFFSET)
		s.onContactOffsetChange();
	if (notifyFlags & ShapeChangeNotifyFlag::eRESTOFFSET)
		s.onRestOffsetChange();
#else
	PX_UNUSED(shape);
	PX_UNUSED(notifyFlags);
#endif
}

Sc::DeformableVolumeSim* Sc::DeformableVolumeCore::getSim() const
{
#if PX_SUPPORT_GPU_PHYSX
	return static_cast<Sc::DeformableVolumeSim*>(ActorCore::getSim());
#else
	return NULL;
#endif
}
