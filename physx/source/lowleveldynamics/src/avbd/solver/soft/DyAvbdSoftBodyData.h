/*
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of NVIDIA CORPORATION nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef DY_AVBD_SOFT_BODY_DATA_H
#define DY_AVBD_SOFT_BODY_DATA_H

#include "foundation/PxArray.h"
#include "foundation/PxMath.h"
#include "avbd/solver/soft/DyAvbdSoftBodyTypes.h"

namespace physx
{
namespace Dy
{

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
	defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
	#define DY_AVBD_SOFT_BODY_DATA_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
	#define DY_AVBD_SOFT_BODY_DATA_API PX_UNIX_EXPORT
#else
	#define DY_AVBD_SOFT_BODY_DATA_API
#endif

enum class AvbdSoftAttachmentTargetKind : PxU8
{
	eDYNAMIC_RIGID,
	eARTICULATION_LINK,
	eDYNAMIC_SOFT,
	eUNSUPPORTED
};

struct AvbdSoftAttachment
{
	AvbdSoftPoint point;
	AvbdSoftPoint targetPoint;
	PxU32 rigidBodyIdx;
	PxU32 sourceHandle;
	AvbdSoftAttachmentTargetKind targetKind;
	PxVec3 localOffset;
	PxVec3 alLambda;
	PxReal k;
	PxReal kMax;

	AvbdSoftAttachment()
		: rigidBodyIdx(0), sourceHandle(PX_MAX_U32),
		  targetKind(AvbdSoftAttachmentTargetKind::eDYNAMIC_RIGID),
		  localOffset(0.0f), alLambda(0.0f), k(1e3f), kMax(1e5f)
	{
	}
};

enum class AvbdSoftPinTargetKind : PxU8
{
	eWORLD_FIXED,
	ePRESCRIBED_RIGID,
	eDEFORMABLE_KINEMATIC,
	eUNSUPPORTED
};

struct AvbdKinematicPin
{
	AvbdSoftPoint point;
	PxU32 sourceHandle;
	AvbdSoftPinTargetKind targetKind;
	PxVec3 worldTarget;
	PxVec3 previousWorldTarget;
	PxVec3 alLambda;
	PxReal k;
	PxReal kMax;

	AvbdKinematicPin()
		: sourceHandle(PX_MAX_U32),
		  targetKind(AvbdSoftPinTargetKind::eWORLD_FIXED),
		  worldTarget(0.0f), previousWorldTarget(0.0f),
		  alLambda(0.0f), k(1e4f), kMax(1e6f)
	{
	}
};

enum class AvbdSoftObjectiveOwner : PxU8
{
	eKINEMATIC_PIN_POSITION_AL,
	eDEFORMABLE_KINEMATIC_POSITION_AL,
	eKINEMATIC_ATTACHMENT_POSITION_AL,
	eRIGID_ATTACHMENT_POSITION_AL,
	eARTICULATION_ATTACHMENT_POSITION_AL,
	eSOFT_PAIR_ATTACHMENT_POSITION_AL,
	eUNSUPPORTED
};

PX_FORCE_INLINE AvbdSoftObjectiveOwner avbdGetAttachmentObjectiveOwner(
	const AvbdSoftAttachment& attachment, bool particleIsValid)
{
	if(!particleIsValid)
		return AvbdSoftObjectiveOwner::eUNSUPPORTED;
	switch(attachment.targetKind)
	{
	case AvbdSoftAttachmentTargetKind::eDYNAMIC_RIGID:
		return AvbdSoftObjectiveOwner::eRIGID_ATTACHMENT_POSITION_AL;
	case AvbdSoftAttachmentTargetKind::eARTICULATION_LINK:
		return AvbdSoftObjectiveOwner::eARTICULATION_ATTACHMENT_POSITION_AL;
	case AvbdSoftAttachmentTargetKind::eDYNAMIC_SOFT:
		if(attachment.targetPoint.particleCount == 0 ||
			attachment.targetPoint.particleCount > 4)
			return AvbdSoftObjectiveOwner::eUNSUPPORTED;
		for(PxU32 endpoint = 0;
			endpoint < attachment.targetPoint.particleCount; endpoint++)
		{
			if(attachment.targetPoint.particleIndices[endpoint] == PX_MAX_U32 ||
				!PxIsFinite(attachment.targetPoint.weights[endpoint]))
				return AvbdSoftObjectiveOwner::eUNSUPPORTED;
		}
		return AvbdSoftObjectiveOwner::eSOFT_PAIR_ATTACHMENT_POSITION_AL;
	case AvbdSoftAttachmentTargetKind::eUNSUPPORTED:
	default:
		return AvbdSoftObjectiveOwner::eUNSUPPORTED;
	}
}

PX_FORCE_INLINE bool avbdIsAttachmentPositionOwner(
	AvbdSoftObjectiveOwner owner)
{
	return owner == AvbdSoftObjectiveOwner::eRIGID_ATTACHMENT_POSITION_AL ||
		owner == AvbdSoftObjectiveOwner::eARTICULATION_ATTACHMENT_POSITION_AL ||
		owner == AvbdSoftObjectiveOwner::eSOFT_PAIR_ATTACHMENT_POSITION_AL;
}

PX_FORCE_INLINE AvbdSoftObjectiveOwner avbdGetPinObjectiveOwner(
	const AvbdKinematicPin& pin, bool particleIsValid)
{
	if(!particleIsValid)
		return AvbdSoftObjectiveOwner::eUNSUPPORTED;
	switch(pin.targetKind)
	{
	case AvbdSoftPinTargetKind::eWORLD_FIXED:
		return AvbdSoftObjectiveOwner::eKINEMATIC_PIN_POSITION_AL;
	case AvbdSoftPinTargetKind::eDEFORMABLE_KINEMATIC:
		return AvbdSoftObjectiveOwner::eDEFORMABLE_KINEMATIC_POSITION_AL;
	case AvbdSoftPinTargetKind::ePRESCRIBED_RIGID:
		return AvbdSoftObjectiveOwner::eKINEMATIC_ATTACHMENT_POSITION_AL;
	case AvbdSoftPinTargetKind::eUNSUPPORTED:
	default:
		return AvbdSoftObjectiveOwner::eUNSUPPORTED;
	}
}

PX_FORCE_INLINE bool avbdIsPinPositionOwner(AvbdSoftObjectiveOwner owner)
{
	return owner == AvbdSoftObjectiveOwner::eKINEMATIC_PIN_POSITION_AL ||
		owner == AvbdSoftObjectiveOwner::eDEFORMABLE_KINEMATIC_POSITION_AL ||
		owner == AvbdSoftObjectiveOwner::eKINEMATIC_ATTACHMENT_POSITION_AL;
}

struct AvbdCompiledSoftObjective
{
	AvbdSoftObjectiveOwner owner;
	PxU32 runtimeStateIndex;
	AvbdSoftPoint point;
	AvbdSoftPoint targetPoint;
	PxU32 rigidBodyIdx;

	AvbdCompiledSoftObjective()
		: owner(AvbdSoftObjectiveOwner::eUNSUPPORTED),
		  runtimeStateIndex(PX_MAX_U32), rigidBodyIdx(PX_MAX_U32)
	{
	}
};
struct AvbdParticleObjectiveAdjacency
{
	PxArray<PxU32> objectiveIndices;
};

// =============================================================================
// AvbdSoftBody -- explicit compiled/material/runtime ownership
// =============================================================================

struct AvbdSoftBodyMaterialData
{
	PxReal youngsModulus;
	PxReal poissonsRatio;
	PxReal density;
	PxReal damping;
	PxReal bendingStiffness;
	PxReal bendingDamping;
	PxReal thickness;
	PxReal dynamicFriction;
	bool coRotationalVolumeModel;

	PxReal mu;
	PxReal lambda;
	PxReal neoHookeanAlpha;

	AvbdSoftBodyMaterialData()
		: youngsModulus(1e5f), poissonsRatio(0.3f),
		  density(1000.0f), damping(0.0f),
		  bendingStiffness(0.0f), bendingDamping(0.0f),
		  thickness(0.01f), dynamicFriction(0.5f),
		  coRotationalVolumeModel(true),
		  mu(0.0f), lambda(0.0f), neoHookeanAlpha(1.0f)
	{
	}

	DY_AVBD_SOFT_BODY_DATA_API void computeLameParameters();
};

struct AvbdSoftBodyRuntimeState
{
	PxArray<AvbdSoftAttachment> attachments;
	PxArray<AvbdKinematicPin> pins;
	PxArray<AvbdCompiledSoftObjective> compiledObjectives;
	PxArray<AvbdParticleObjectiveAdjacency> objectiveAdjacency;

	DY_AVBD_SOFT_BODY_DATA_API void compileObjectiveProgram(
		PxU32 particleStart, PxU32 particleCount);

	DY_AVBD_SOFT_BODY_DATA_API bool isObjectiveProgramCurrent(
		PxU32 particleStart, PxU32 particleCount) const;
};

#undef DY_AVBD_SOFT_BODY_DATA_API

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_DATA_H
