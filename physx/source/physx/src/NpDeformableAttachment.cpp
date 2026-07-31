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

#include "foundation/PxAllocator.h"
#include "foundation/PxPreprocessor.h"

#include "NpCheck.h"
#include "NpScene.h"
#include "NpShape.h"
#include "NpRigidDynamic.h"
#include "NpRigidStatic.h"
#include "NpArticulationLink.h"
#include "NpDeformableAttachment.h"
#include "NpDeformableSurface.h"
#include "NpDeformableVolume.h"

using namespace physx;

NpInternalAttachmentType::Enum getInternalAttachmentType(const PxDeformableAttachmentData& data, PxU32 actorIndex[2])
{
	// Make sure both actors are not NULL
	if (data.actor[0] == NULL && data.actor[1] == NULL)
		return NpInternalAttachmentType::eUNDEFINED;

	// Actor 0 is always assumed to be a deformable (deformable volume has the highest priority)
	// If Actor 0 is a null actor or rigid actor, the actors are swapped using the Actor Index
	// If Actor 0 is a deformable surface and Actor 1 is a deformable volume, the actors are swapped using the Actor Index
	actorIndex[0] = 0;
	actorIndex[1] = 1;

	if ((data.actor[0] == NULL || data.actor[0]->is<PxRigidActor>()) ||
		(data.actor[0]->is<PxDeformableSurface>() && data.actor[1] != NULL && data.actor[1]->is<PxDeformableVolume>()))
	{
		// Swap the actor index.
		PxSwap(actorIndex[0], actorIndex[1]);
	}

	// Further order according to element type, TET > TRI > VTX
	PxType actorType0 = data.actor[actorIndex[0]]->getConcreteType();
	PxType actorType1 = data.actor[actorIndex[1]] ? data.actor[actorIndex[1]]->getConcreteType() : PxConcreteType::eUNDEFINED;
	if (actorType0 == actorType1 && data.type[actorIndex[0]] < data.type[actorIndex[1]])
	{
		// Swap again to order according to first type: PxDeformableAttachmentTargetType::eTETRAHEDRON first, then eTRIANGLE, last eVERTEX.
		PxSwap(actorIndex[0], actorIndex[1]);
	}

	switch (data.actor[actorIndex[0]]->getConcreteType())
	{
		case PxConcreteType::eDEFORMABLE_SURFACE:
		{
			if (data.actor[actorIndex[1]] == NULL)
			{
				if (data.type[actorIndex[1]] == PxDeformableAttachmentTargetType::eWORLD)
				{
					if (data.type[actorIndex[0]] == PxDeformableAttachmentTargetType::eVERTEX)
						return NpInternalAttachmentType::eSURFACE_VTX_GLOBAL_POSE;
					
					if (data.type[actorIndex[0]] == PxDeformableAttachmentTargetType::eTRIANGLE)
						return NpInternalAttachmentType::eSURFACE_TRI_GLOBAL_POSE;
				}
			}
			else if (data.actor[actorIndex[1]]->is<PxRigidActor>())
			{
				if (data.type[actorIndex[1]] == PxDeformableAttachmentTargetType::eRIGID)
				{
					if (data.type[actorIndex[0]] == PxDeformableAttachmentTargetType::eVERTEX)
						return NpInternalAttachmentType::eSURFACE_VTX_RIGID_BODY;
					
					if (data.type[actorIndex[0]] == PxDeformableAttachmentTargetType::eTRIANGLE)
						return NpInternalAttachmentType::eSURFACE_TRI_RIGID_BODY;
				}
			}
			else if (data.actor[actorIndex[1]]->is<PxDeformableSurface>())
			{
				if (data.type[actorIndex[0]] == PxDeformableAttachmentTargetType::eTRIANGLE)
				{
					if (data.type[actorIndex[1]] == PxDeformableAttachmentTargetType::eTRIANGLE)
						return NpInternalAttachmentType::eSURFACE_TRI_SURFACE_TRI;

					if (data.type[actorIndex[1]] == PxDeformableAttachmentTargetType::eVERTEX)
						return NpInternalAttachmentType::eSURFACE_TRI_SURFACE_VTX;
				}
				else if (data.type[actorIndex[0]] == PxDeformableAttachmentTargetType::eVERTEX)
				{
					if (data.type[actorIndex[1]] == PxDeformableAttachmentTargetType::eVERTEX)
						return NpInternalAttachmentType::eSURFACE_VTX_SURFACE_VTX;
				}
			}

			break;
		}
		case PxConcreteType::eDEFORMABLE_VOLUME:
		{
			if (data.actor[actorIndex[1]] == NULL)
			{
				if (data.type[actorIndex[1]] == PxDeformableAttachmentTargetType::eWORLD)
				{
					if (data.type[actorIndex[0]] == PxDeformableAttachmentTargetType::eVERTEX)
						return NpInternalAttachmentType::eVOLUME_VTX_GLOBAL_POSE;
					
					if (data.type[actorIndex[0]] == PxDeformableAttachmentTargetType::eTETRAHEDRON)
						return NpInternalAttachmentType::eVOLUME_TET_GLOBAL_POSE;
				}
			}
			else if (data.actor[actorIndex[1]]->is<PxRigidActor>())
			{
				if (data.type[actorIndex[1]] == PxDeformableAttachmentTargetType::eRIGID)
				{
					if (data.type[actorIndex[0]] == PxDeformableAttachmentTargetType::eVERTEX)
						return NpInternalAttachmentType::eVOLUME_VTX_RIGID_BODY;
					
					if (data.type[actorIndex[0]] == PxDeformableAttachmentTargetType::eTETRAHEDRON)
						return NpInternalAttachmentType::eVOLUME_TET_RIGID_BODY;
				}
			}
			else if (data.actor[actorIndex[1]]->is<PxDeformableSurface>())
			{
				if (data.type[actorIndex[0]] == PxDeformableAttachmentTargetType::eTETRAHEDRON)
				{
					if (data.type[actorIndex[1]] == PxDeformableAttachmentTargetType::eVERTEX)
						return NpInternalAttachmentType::eVOLUME_TET_SURFACE_VTX;

					if (data.type[actorIndex[1]] == PxDeformableAttachmentTargetType::eTRIANGLE)
						return NpInternalAttachmentType::eVOLUME_TET_SURFACE_TRI;
				}
				else if (data.type[actorIndex[0]] == PxDeformableAttachmentTargetType::eVERTEX)
				{
					if (data.type[actorIndex[1]] == PxDeformableAttachmentTargetType::eVERTEX)
						return NpInternalAttachmentType::eVOLUME_VTX_SURFACE_VTX;
				}
			}
			else if (data.actor[actorIndex[1]]->is<PxDeformableVolume>())
			{
				if (data.type[actorIndex[0]] == PxDeformableAttachmentTargetType::eTETRAHEDRON)
				{
					if (data.type[actorIndex[1]] == PxDeformableAttachmentTargetType::eTETRAHEDRON)
						return NpInternalAttachmentType::eVOLUME_TET_VOLUME_TET;

					if (data.type[actorIndex[1]] == PxDeformableAttachmentTargetType::eVERTEX)
						return NpInternalAttachmentType::eVOLUME_TET_VOLUME_VTX;
				}
				else if (data.type[actorIndex[0]] == PxDeformableAttachmentTargetType::eVERTEX)
				{
					if (data.type[actorIndex[1]] == PxDeformableAttachmentTargetType::eVERTEX)
						return NpInternalAttachmentType::eVOLUME_VTX_VOLUME_VTX;
				}
			}

			break;
		}
	}

	return NpInternalAttachmentType::eUNDEFINED;
}

bool NpDeformableAttachment::parseAttachment(const PxDeformableAttachmentData& data, AttachmentInfo& info)
{
	NpInternalAttachmentType::Enum& internalAttachmentType = info.internalAttachmentType;
	PxU32* actorIndex = info.actorIndex;

	internalAttachmentType = getInternalAttachmentType(data, actorIndex);

	switch (internalAttachmentType)
	{
		case NpInternalAttachmentType::eSURFACE_TRI_RIGID_BODY:
		case NpInternalAttachmentType::eSURFACE_TRI_GLOBAL_POSE:
		case NpInternalAttachmentType::eSURFACE_VTX_RIGID_BODY:
		case NpInternalAttachmentType::eSURFACE_VTX_GLOBAL_POSE:
		case NpInternalAttachmentType::eVOLUME_TET_RIGID_BODY:
		case NpInternalAttachmentType::eVOLUME_TET_GLOBAL_POSE:
		case NpInternalAttachmentType::eVOLUME_VTX_RIGID_BODY:
		case NpInternalAttachmentType::eVOLUME_VTX_GLOBAL_POSE:
		{
			{
				PX_CHECK_AND_RETURN_VAL
				(
					data.indices[actorIndex[0]].count == data.coords[actorIndex[1]].count,
					"PxDeformableAttachment: Number of attachment points are not equal for both actors.",
					NULL
				);
			}
			
			if (internalAttachmentType == NpInternalAttachmentType::eSURFACE_TRI_RIGID_BODY ||
				internalAttachmentType == NpInternalAttachmentType::eSURFACE_TRI_GLOBAL_POSE ||
				internalAttachmentType == NpInternalAttachmentType::eVOLUME_TET_RIGID_BODY ||
				internalAttachmentType == NpInternalAttachmentType::eVOLUME_TET_GLOBAL_POSE)
			{
				PX_CHECK_AND_RETURN_VAL
				(
					data.indices[actorIndex[0]].count == data.coords[actorIndex[0]].count,
					"PxDeformableAttachment: Number of attachment points are not equal for both actors.",
					NULL
				);
			}

			break;
		}

		case NpInternalAttachmentType::eSURFACE_TRI_SURFACE_TRI:
		case NpInternalAttachmentType::eSURFACE_TRI_SURFACE_VTX:
		case NpInternalAttachmentType::eSURFACE_VTX_SURFACE_VTX:
		case NpInternalAttachmentType::eVOLUME_TET_VOLUME_TET:
		case NpInternalAttachmentType::eVOLUME_TET_VOLUME_VTX:
		case NpInternalAttachmentType::eVOLUME_TET_SURFACE_TRI:
		case NpInternalAttachmentType::eVOLUME_TET_SURFACE_VTX:
		case NpInternalAttachmentType::eVOLUME_VTX_VOLUME_VTX:
		case NpInternalAttachmentType::eVOLUME_VTX_SURFACE_VTX:
		{
			const PxU32 endpoint0 = actorIndex[0];
			const PxU32 endpoint1 = actorIndex[1];
			const bool endpoint0Element =
				data.type[endpoint0] ==
					PxDeformableAttachmentTargetType::eTRIANGLE ||
				data.type[endpoint0] ==
					PxDeformableAttachmentTargetType::eTETRAHEDRON;
			const bool endpoint1Element =
				data.type[endpoint1] ==
					PxDeformableAttachmentTargetType::eTRIANGLE ||
				data.type[endpoint1] ==
					PxDeformableAttachmentTargetType::eTETRAHEDRON;
			PX_CHECK_AND_RETURN_VAL
			(
				data.indices[endpoint0].count ==
					data.indices[endpoint1].count &&
				(!endpoint0Element ||
				 data.indices[endpoint0].count ==
					data.coords[endpoint0].count) &&
				(!endpoint1Element ||
				 data.indices[endpoint1].count ==
					data.coords[endpoint1].count),
				"PxDeformableAttachment: Number of attachment points are not equal for both actors.",
				NULL
			);
			PX_UNUSED(endpoint0Element);
			PX_UNUSED(endpoint1Element);

			break;
		}

		default:
		{
			PX_CHECK_AND_RETURN_VAL(false, "PxDeformableAttachment: No matching actor pairs found for attachment.", NULL);
		}
	}

	return true;
}

void NpDeformableAttachment::addAttachment()
{
	if (!getSceneFromActors() ||  mEnabled)
		return;

	const bool cpuAvbdWorldElement =
		mCpuAvbdRoute == CpuAvbdRoute::eWORLD_ELEMENT;
	if(mCpuAvbdRoute == CpuAvbdRoute::eWORLD_PIN ||
		cpuAvbdWorldElement)
	{
		NpScene* scene = getSceneFromActors();
		PxActor* actor = mActor[mActorIndex[0]];
		const PxU32 softIndex = mActorIndex[0];
		const bool surfaceType =
			(mInternalAttachmentType &
				NpInternalAttachmentType::eSURFACE_TYPE) != 0;
		const PxU32 worldIndex = mActorIndex[1];
		mHandles.resize(mIndices[mActorIndex[0]].size());
		for(PxU32 i = 0; i < mHandles.size(); i++)
		{
			const PxVec3 worldTarget =
				mPose[worldIndex].transform(
					mCoords[worldIndex][i].getXYZ());
			if(surfaceType)
			{
				mHandles[i] = cpuAvbdWorldElement
					? scene->getScScene().
						addAvbdCpuDeformableSurfaceWorldElementAttachment(
							*getDeformableSurfaceCore(actor),
							mIndices[softIndex][i],
							mCoords[softIndex][i], worldTarget)
					: scene->getScScene().
						addAvbdCpuDeformableSurfaceWorldPin(
							*getDeformableSurfaceCore(actor),
							mIndices[softIndex][i], worldTarget);
			}
			else
			{
				mHandles[i] = cpuAvbdWorldElement
					? scene->getScScene().
						addAvbdCpuDeformableVolumeWorldElementAttachment(
							*getDeformableVolumeCore(actor),
							mIndices[softIndex][i],
							mCoords[softIndex][i], worldTarget)
					: scene->getScScene().
						addAvbdCpuDeformableVolumeWorldPin(
							*getDeformableVolumeCore(actor),
							mIndices[softIndex][i], worldTarget);
			}
			if(mHandles[i] == PX_MAX_U32)
			{
				for(PxU32 j = 0; j < i; j++)
				{
					if(surfaceType)
						scene->getScScene().
							removeAvbdCpuDeformableSurfaceWorldPin(
								*getDeformableSurfaceCore(actor),
								mHandles[j]);
					else
						scene->getScScene().
							removeAvbdCpuDeformableVolumeWorldPin(
								*getDeformableVolumeCore(actor),
								mHandles[j]);
				}
				mHandles.clear();
				PX_CHECK_AND_RETURN(
					false,
					"PxDeformableAttachment: Failed to add CPU "
					"AVBD deformable-to-world attachment.");
			}
		}
		mEnabled = true;
		return;
	}

	const bool cpuAvbdStaticElement =
		mCpuAvbdRoute == CpuAvbdRoute::eSTATIC_RIGID_ELEMENT;
	if(mCpuAvbdRoute == CpuAvbdRoute::eSTATIC_RIGID ||
		cpuAvbdStaticElement)
	{
		NpScene* scene = getSceneFromActors();
		PxActor* softActor = mActor[mActorIndex[0]];
		NpRigidStatic* staticActor =
			static_cast<NpRigidStatic*>(
				mActor[mActorIndex[1]]->is<PxRigidStatic>());
		Sc::StaticCore& staticCore = staticActor->getCore();
		const bool surfaceType =
			(mInternalAttachmentType &
				NpInternalAttachmentType::eSURFACE_TYPE) != 0;
		const PxU32 rigidIndex = mActorIndex[1];
		mHandles.resize(mIndices[mActorIndex[0]].size());
		for(PxU32 i = 0; i < mHandles.size(); i++)
		{
			const PxVec3 actorLocalTarget =
				mPose[rigidIndex].transform(
					mCoords[rigidIndex][i].getXYZ());
			if(surfaceType)
				mHandles[i] = cpuAvbdStaticElement
					? scene->getScScene().
						addAvbdCpuDeformableSurfaceStaticElementAttachment(
							*getDeformableSurfaceCore(softActor),
							staticCore,
							mIndices[mActorIndex[0]][i],
							mCoords[mActorIndex[0]][i],
							actorLocalTarget)
					: scene->getScScene().
						addAvbdCpuDeformableSurfaceStaticAttachment(
							*getDeformableSurfaceCore(softActor),
							staticCore,
							mIndices[mActorIndex[0]][i],
							actorLocalTarget);
			else
				mHandles[i] = cpuAvbdStaticElement
					? scene->getScScene().
						addAvbdCpuDeformableVolumeStaticElementAttachment(
							*getDeformableVolumeCore(softActor),
							staticCore,
							mIndices[mActorIndex[0]][i],
							mCoords[mActorIndex[0]][i],
							actorLocalTarget)
					: scene->getScScene().
						addAvbdCpuDeformableVolumeStaticAttachment(
							*getDeformableVolumeCore(softActor),
							staticCore,
							mIndices[mActorIndex[0]][i],
							actorLocalTarget);
			if(mHandles[i] == PX_MAX_U32)
			{
				for(PxU32 j = 0; j < i; j++)
				{
					if(surfaceType)
						scene->getScScene().
							removeAvbdCpuDeformableSurfaceStaticAttachment(
								*getDeformableSurfaceCore(
									softActor),
								mHandles[j]);
					else
						scene->getScScene().
							removeAvbdCpuDeformableVolumeStaticAttachment(
								*getDeformableVolumeCore(
									softActor),
								mHandles[j]);
				}
				mHandles.clear();
				PX_CHECK_AND_RETURN(
					false,
					"PxDeformableAttachment: Failed to add CPU "
					"AVBD deformable-to-static attachment.");
			}
		}
		mEnabled = true;
		return;
	}

	if(mCpuAvbdRoute == CpuAvbdRoute::eSOFT_PAIR)
	{
		NpScene* scene = getSceneFromActors();
		PxActor* actor0 = mActor[mActorIndex[0]];
		PxActor* actor1 = mActor[mActorIndex[1]];
		Sc::ActorCore* core0 = actor0->is<PxDeformableSurface>()
			? static_cast<Sc::ActorCore*>(
				getDeformableSurfaceCore(actor0))
			: static_cast<Sc::ActorCore*>(
				getDeformableVolumeCore(actor0));
		Sc::ActorCore* core1 = actor1->is<PxDeformableSurface>()
			? static_cast<Sc::ActorCore*>(
				getDeformableSurfaceCore(actor1))
			: static_cast<Sc::ActorCore*>(
				getDeformableVolumeCore(actor1));
		const PxU32 endpoint0 = mActorIndex[0];
		const PxU32 endpoint1 = mActorIndex[1];
		const bool element0 =
			mType[endpoint0] ==
				PxDeformableAttachmentTargetType::eTRIANGLE ||
			mType[endpoint0] ==
				PxDeformableAttachmentTargetType::eTETRAHEDRON;
		const bool element1 =
			mType[endpoint1] ==
				PxDeformableAttachmentTargetType::eTRIANGLE ||
			mType[endpoint1] ==
				PxDeformableAttachmentTargetType::eTETRAHEDRON;
		mHandles.resize(mIndices[endpoint0].size());
		for(PxU32 i = 0; i < mHandles.size(); i++)
		{
			const PxVec4 barycentric0 = element0
				? mCoords[endpoint0][i] : PxVec4(0.0f);
			const PxVec4 barycentric1 = element1
				? mCoords[endpoint1][i] : PxVec4(0.0f);
			mHandles[i] = scene->getScScene().
				addAvbdCpuDeformablePairAttachment(
					*core0, element0, mIndices[endpoint0][i],
					barycentric0,
					*core1, element1, mIndices[endpoint1][i],
					barycentric1);
			if(mHandles[i] == PX_MAX_U32)
			{
				for(PxU32 j = 0; j < i; j++)
					scene->getScScene().
						removeAvbdCpuDeformablePairAttachment(
							*core0, mHandles[j]);
				mHandles.clear();
				PX_CHECK_AND_RETURN(
					false,
					"PxDeformableAttachment: Failed to add CPU "
					"AVBD deformable-pair attachment.");
			}
		}
		mEnabled = true;
		return;
	}

	const bool cpuAvbdKinematicElement =
		mCpuAvbdRoute ==
			CpuAvbdRoute::eKINEMATIC_RIGID_ELEMENT;
	const bool cpuAvbdKinematicAttachment =
		mCpuAvbdRoute == CpuAvbdRoute::eKINEMATIC_RIGID ||
		cpuAvbdKinematicElement;
	const bool cpuAvbdArticulationElement =
		mCpuAvbdRoute ==
			CpuAvbdRoute::eARTICULATION_LINK_ELEMENT;
	const bool cpuAvbdArticulationAttachment =
		mCpuAvbdRoute == CpuAvbdRoute::eARTICULATION_LINK ||
		cpuAvbdArticulationElement;
	const bool cpuAvbdRigidElement =
		mCpuAvbdRoute == CpuAvbdRoute::eDYNAMIC_RIGID_ELEMENT;
	if(mCpuAvbdRoute == CpuAvbdRoute::eDYNAMIC_RIGID ||
		cpuAvbdRigidElement ||
		cpuAvbdKinematicAttachment ||
		cpuAvbdArticulationAttachment)
	{
		NpScene* scene = getSceneFromActors();
		PxActor* softActor = mActor[mActorIndex[0]];
		PxRigidActor* rigidActor =
			mActor[mActorIndex[1]]->is<PxRigidActor>();
		Sc::BodyCore* rigidCore = getBodyCore(rigidActor);
		const bool surfaceType =
			(mInternalAttachmentType &
				NpInternalAttachmentType::eSURFACE_TYPE) != 0;
		const PxU32 rigidIndex = mActorIndex[1];
		mHandles.resize(mIndices[mActorIndex[0]].size());
		for(PxU32 i = 0; i < mHandles.size(); i++)
		{
			const PxVec3 actorLocalTarget =
				mPose[rigidIndex].transform(
					mCoords[rigidIndex][i].getXYZ());
			if(surfaceType)
			{
				if(cpuAvbdArticulationAttachment)
					mHandles[i] = cpuAvbdArticulationElement
						? scene->getScScene().
							addAvbdCpuDeformableSurfaceArticulationElementAttachment(
								*getDeformableSurfaceCore(
									softActor),
								*rigidCore,
								mIndices[mActorIndex[0]][i],
								mCoords[mActorIndex[0]][i],
								actorLocalTarget)
						: scene->getScScene().
							addAvbdCpuDeformableSurfaceArticulationAttachment(
								*getDeformableSurfaceCore(
									softActor),
								*rigidCore,
								mIndices[mActorIndex[0]][i],
								actorLocalTarget);
				else if(cpuAvbdKinematicAttachment)
					mHandles[i] = cpuAvbdKinematicElement
						? scene->getScScene().
							addAvbdCpuDeformableSurfaceKinematicElementAttachment(
								*getDeformableSurfaceCore(
									softActor),
								*rigidCore,
								mIndices[mActorIndex[0]][i],
								mCoords[mActorIndex[0]][i],
								actorLocalTarget)
						: scene->getScScene().
							addAvbdCpuDeformableSurfaceKinematicAttachment(
								*getDeformableSurfaceCore(
									softActor),
								*rigidCore,
								mIndices[mActorIndex[0]][i],
								actorLocalTarget);
				else
					mHandles[i] = cpuAvbdRigidElement
						? scene->getScScene().
							addAvbdCpuDeformableSurfaceRigidElementAttachment(
								*getDeformableSurfaceCore(softActor),
								*rigidCore,
								mIndices[mActorIndex[0]][i],
								mCoords[mActorIndex[0]][i],
								actorLocalTarget)
						: scene->getScScene().
							addAvbdCpuDeformableSurfaceRigidAttachment(
								*getDeformableSurfaceCore(softActor),
								*rigidCore,
								mIndices[mActorIndex[0]][i],
								actorLocalTarget);
			}
			else
			{
				if(cpuAvbdArticulationAttachment)
					mHandles[i] = cpuAvbdArticulationElement
						? scene->getScScene().
							addAvbdCpuDeformableVolumeArticulationElementAttachment(
								*getDeformableVolumeCore(
									softActor),
								*rigidCore,
								mIndices[mActorIndex[0]][i],
								mCoords[mActorIndex[0]][i],
								actorLocalTarget)
						: scene->getScScene().
							addAvbdCpuDeformableVolumeArticulationAttachment(
								*getDeformableVolumeCore(
									softActor),
								*rigidCore,
								mIndices[mActorIndex[0]][i],
								actorLocalTarget);
				else if(cpuAvbdKinematicAttachment)
					mHandles[i] = cpuAvbdKinematicElement
						? scene->getScScene().
							addAvbdCpuDeformableVolumeKinematicElementAttachment(
								*getDeformableVolumeCore(
									softActor),
								*rigidCore,
								mIndices[mActorIndex[0]][i],
								mCoords[mActorIndex[0]][i],
								actorLocalTarget)
						: scene->getScScene().
							addAvbdCpuDeformableVolumeKinematicAttachment(
								*getDeformableVolumeCore(
									softActor),
								*rigidCore,
								mIndices[mActorIndex[0]][i],
								actorLocalTarget);
				else
					mHandles[i] = cpuAvbdRigidElement
						? scene->getScScene().
							addAvbdCpuDeformableVolumeRigidElementAttachment(
								*getDeformableVolumeCore(softActor),
								*rigidCore,
								mIndices[mActorIndex[0]][i],
								mCoords[mActorIndex[0]][i],
								actorLocalTarget)
						: scene->getScScene().
							addAvbdCpuDeformableVolumeRigidAttachment(
								*getDeformableVolumeCore(softActor),
								*rigidCore,
								mIndices[mActorIndex[0]][i],
								actorLocalTarget);
			}
			if(mHandles[i] == PX_MAX_U32)
			{
				for(PxU32 j = 0; j < i; j++)
				{
					if(surfaceType)
					{
						if(cpuAvbdArticulationAttachment)
							scene->getScScene().
								removeAvbdCpuDeformableSurfaceArticulationAttachment(
									*getDeformableSurfaceCore(
										softActor),
									mHandles[j]);
						else if(cpuAvbdKinematicAttachment)
							scene->getScScene().
								removeAvbdCpuDeformableSurfaceKinematicAttachment(
									*getDeformableSurfaceCore(
										softActor),
									mHandles[j]);
						else
							scene->getScScene().
								removeAvbdCpuDeformableSurfaceRigidAttachment(
									*getDeformableSurfaceCore(
										softActor),
									mHandles[j]);
					}
					else if(cpuAvbdArticulationAttachment)
						scene->getScScene().
							removeAvbdCpuDeformableVolumeArticulationAttachment(
								*getDeformableVolumeCore(softActor),
								mHandles[j]);
					else if(cpuAvbdKinematicAttachment)
						scene->getScScene().
							removeAvbdCpuDeformableVolumeKinematicAttachment(
								*getDeformableVolumeCore(softActor),
								mHandles[j]);
					else
						scene->getScScene().
							removeAvbdCpuDeformableVolumeRigidAttachment(
								*getDeformableVolumeCore(softActor),
								mHandles[j]);
				}
				mHandles.clear();
				PX_CHECK_AND_RETURN(
					false,
					"PxDeformableAttachment: Failed to add CPU "
					"AVBD deformable-to-rigid attachment.");
			}
		}
		mEnabled = true;
		return;
	}

#if PX_SUPPORT_GPU_PHYSX
	switch (mInternalAttachmentType)
	{
		case NpInternalAttachmentType::eSURFACE_TRI_RIGID_BODY:
		case NpInternalAttachmentType::eSURFACE_TRI_GLOBAL_POSE:
		case NpInternalAttachmentType::eSURFACE_VTX_RIGID_BODY:
		case NpInternalAttachmentType::eSURFACE_VTX_GLOBAL_POSE:
		case NpInternalAttachmentType::eVOLUME_TET_RIGID_BODY:
		case NpInternalAttachmentType::eVOLUME_TET_GLOBAL_POSE:
		case NpInternalAttachmentType::eVOLUME_VTX_RIGID_BODY:
		case NpInternalAttachmentType::eVOLUME_VTX_GLOBAL_POSE:
		{
			PxActor* actor0 = mActor[mActorIndex[0]];

			PxRigidActor* actor1 = mActor[mActorIndex[1]] == NULL ? NULL : mActor[mActorIndex[1]]->is<PxRigidActor>();
			Sc::BodyCore* core1 = mActor[mActorIndex[1]] == NULL ? NULL : getBodyCore(actor1);

			mHandles.resize(mIndices[mActorIndex[0]].size());
			for (PxU32 i = 0; i < mIndices[mActorIndex[0]].size(); i++)
			{
				PxU32 id = mIndices[mActorIndex[0]][i];

				PxVec3 actor1Pose = mCoords[mActorIndex[1]][i].getXYZ();

				if (actor1 && actor1->getConcreteType() == PxConcreteType::eRIGID_STATIC)
				{
					NpRigidStatic* stat = static_cast<NpRigidStatic*>(actor1);
					actor1Pose = stat->getGlobalPose().transform(actor1Pose);
				}

				actor1Pose = mPose[mActorIndex[1]].transform(actor1Pose);

				if (mInternalAttachmentType & NpInternalAttachmentType::eSURFACE_TYPE)
				{
					if (mInternalAttachmentType == NpInternalAttachmentType::eSURFACE_TRI_RIGID_BODY ||
						mInternalAttachmentType == NpInternalAttachmentType::eSURFACE_TRI_GLOBAL_POSE)
					{
						const PxVec4& barycentric = mCoords[mActorIndex[0]][i];
						mHandles[i] = getDeformableSurfaceCore(actor0)->addTriRigidAttachment(core1, id, barycentric, actor1Pose);
					}
					else if (mInternalAttachmentType == NpInternalAttachmentType::eSURFACE_VTX_RIGID_BODY ||
							 mInternalAttachmentType == NpInternalAttachmentType::eSURFACE_VTX_GLOBAL_POSE)
					{
						mHandles[i] = getDeformableSurfaceCore(actor0)->addRigidAttachment(core1, id, actor1Pose);
					}
				}
				else if (mInternalAttachmentType & NpInternalAttachmentType::eVOLUME_TYPE)
				{
					if (mInternalAttachmentType == NpInternalAttachmentType::eVOLUME_TET_RIGID_BODY ||
						mInternalAttachmentType == NpInternalAttachmentType::eVOLUME_TET_GLOBAL_POSE)
					{
						const PxVec4& barycentric = mCoords[mActorIndex[0]][i];
						mHandles[i] = getDeformableVolumeCore(actor0)->addTetRigidAttachment(core1, id, barycentric, actor1Pose, false);
					}
					else if (mInternalAttachmentType == NpInternalAttachmentType::eVOLUME_VTX_RIGID_BODY ||
							 mInternalAttachmentType == NpInternalAttachmentType::eVOLUME_VTX_GLOBAL_POSE)
					{
						mHandles[i] = getDeformableVolumeCore(actor0)->addRigidAttachment(core1, id, actor1Pose, false);
					}
				}
			}

			break;
		}

		case NpInternalAttachmentType::eSURFACE_TRI_SURFACE_TRI:
		case NpInternalAttachmentType::eVOLUME_TET_VOLUME_TET:
		case NpInternalAttachmentType::eVOLUME_TET_SURFACE_TRI:
		{
			PxActor* actor0 = mActor[mActorIndex[0]];
			PxActor* actor1 = mActor[mActorIndex[1]];

			mHandles.resize(mIndices[mActorIndex[0]].size());
			for (PxU32 i = 0; i < mIndices[mActorIndex[0]].size(); i++)
			{
				PxU32 id0 = mIndices[mActorIndex[0]][i];
				const PxVec4& barycentric0 = mCoords[mActorIndex[0]][i];

				PxU32 id1 = mIndices[mActorIndex[1]][i];
				const PxVec4& barycentric1 = mCoords[mActorIndex[1]][i];

				if (mInternalAttachmentType == NpInternalAttachmentType::eSURFACE_TRI_SURFACE_TRI)
				{
					mHandles[i] = getDeformableSurfaceCore(actor0)->addClothAttachment(getDeformableSurfaceCore(actor1), id1, barycentric1, id0, barycentric0);
				}
				else if (mInternalAttachmentType == NpInternalAttachmentType::eVOLUME_TET_VOLUME_TET)
				{
					mHandles[i] = getDeformableVolumeCore(actor0)->addSoftBodyAttachment(*getDeformableVolumeCore(actor1), id1, barycentric1, id0, barycentric0, false);
				}
				else if (mInternalAttachmentType == NpInternalAttachmentType::eVOLUME_TET_SURFACE_TRI)
				{
					mHandles[i] = getDeformableVolumeCore(actor0)->addClothAttachment(*getDeformableSurfaceCore(actor1), id1, barycentric1, id0, barycentric0, false);
				}
			}

			break;
		}

		default:
		{
			PX_ASSERT(0);
			break;
		}
	}

	mEnabled = true;
#else
	PX_CHECK_AND_RETURN(
		false,
		"PxDeformableAttachment: This attachment type requires GPU "
		"PhysX or a supported CPU AVBD attachment path.");
#endif
}

void NpDeformableAttachment::removeAttachment()
{
	if (!mEnabled)
		return;

	if(mCpuAvbdRoute == CpuAvbdRoute::eWORLD_PIN ||
		mCpuAvbdRoute == CpuAvbdRoute::eWORLD_ELEMENT)
	{
		NpScene* scene = getSceneFromActors();
		PxActor* actor = mActor[mActorIndex[0]];
		const bool surfaceType =
			(mInternalAttachmentType &
				NpInternalAttachmentType::eSURFACE_TYPE) != 0;
		if(scene)
		{
			for(PxU32 i = 0; i < mHandles.size(); i++)
			{
				if(surfaceType)
					scene->getScScene().
						removeAvbdCpuDeformableSurfaceWorldPin(
							*getDeformableSurfaceCore(actor),
							mHandles[i]);
				else
					scene->getScScene().
						removeAvbdCpuDeformableVolumeWorldPin(
							*getDeformableVolumeCore(actor),
							mHandles[i]);
			}
		}
		mHandles.clear();
		mEnabled = false;
		return;
	}

	if(mCpuAvbdRoute == CpuAvbdRoute::eSTATIC_RIGID ||
		mCpuAvbdRoute == CpuAvbdRoute::eSTATIC_RIGID_ELEMENT)
	{
		NpScene* scene = getSceneFromActors();
		PxActor* softActor = mActor[mActorIndex[0]];
		const bool surfaceType =
			(mInternalAttachmentType &
				NpInternalAttachmentType::eSURFACE_TYPE) != 0;
		if(scene)
		{
			for(PxU32 i = 0; i < mHandles.size(); i++)
			{
				if(surfaceType)
					scene->getScScene().
						removeAvbdCpuDeformableSurfaceStaticAttachment(
							*getDeformableSurfaceCore(softActor),
							mHandles[i]);
				else
					scene->getScScene().
						removeAvbdCpuDeformableVolumeStaticAttachment(
							*getDeformableVolumeCore(softActor),
							mHandles[i]);
			}
		}
		mHandles.clear();
		mEnabled = false;
		return;
	}

	if(mCpuAvbdRoute == CpuAvbdRoute::eSOFT_PAIR)
	{
		NpScene* scene = getSceneFromActors();
		PxActor* actor0 = mActor[mActorIndex[0]];
		Sc::ActorCore* core0 = actor0->is<PxDeformableSurface>()
			? static_cast<Sc::ActorCore*>(
				getDeformableSurfaceCore(actor0))
			: static_cast<Sc::ActorCore*>(
				getDeformableVolumeCore(actor0));
		if(scene)
		{
			for(PxU32 i = 0; i < mHandles.size(); i++)
				scene->getScScene().
					removeAvbdCpuDeformablePairAttachment(
						*core0, mHandles[i]);
		}
		mHandles.clear();
		mEnabled = false;
		return;
	}

	const bool cpuAvbdKinematicAttachment =
		mCpuAvbdRoute == CpuAvbdRoute::eKINEMATIC_RIGID ||
		mCpuAvbdRoute ==
			CpuAvbdRoute::eKINEMATIC_RIGID_ELEMENT;
	const bool cpuAvbdArticulationAttachment =
		mCpuAvbdRoute == CpuAvbdRoute::eARTICULATION_LINK ||
		mCpuAvbdRoute ==
			CpuAvbdRoute::eARTICULATION_LINK_ELEMENT;
	if(mCpuAvbdRoute == CpuAvbdRoute::eDYNAMIC_RIGID ||
		mCpuAvbdRoute == CpuAvbdRoute::eDYNAMIC_RIGID_ELEMENT ||
		cpuAvbdKinematicAttachment ||
		cpuAvbdArticulationAttachment)
	{
		NpScene* scene = getSceneFromActors();
		PxActor* softActor = mActor[mActorIndex[0]];
		const bool surfaceType =
			(mInternalAttachmentType &
				NpInternalAttachmentType::eSURFACE_TYPE) != 0;
		if(scene)
		{
			for(PxU32 i = 0; i < mHandles.size(); i++)
			{
				if(surfaceType)
				{
					if(cpuAvbdArticulationAttachment)
						scene->getScScene().
							removeAvbdCpuDeformableSurfaceArticulationAttachment(
								*getDeformableSurfaceCore(softActor),
								mHandles[i]);
					else if(cpuAvbdKinematicAttachment)
						scene->getScScene().
							removeAvbdCpuDeformableSurfaceKinematicAttachment(
								*getDeformableSurfaceCore(softActor),
								mHandles[i]);
					else
						scene->getScScene().
							removeAvbdCpuDeformableSurfaceRigidAttachment(
								*getDeformableSurfaceCore(softActor),
								mHandles[i]);
				}
				else if(cpuAvbdArticulationAttachment)
					scene->getScScene().
						removeAvbdCpuDeformableVolumeArticulationAttachment(
							*getDeformableVolumeCore(softActor),
							mHandles[i]);
				else if(cpuAvbdKinematicAttachment)
					scene->getScScene().
						removeAvbdCpuDeformableVolumeKinematicAttachment(
							*getDeformableVolumeCore(softActor),
							mHandles[i]);
				else
					scene->getScScene().
						removeAvbdCpuDeformableVolumeRigidAttachment(
							*getDeformableVolumeCore(softActor),
							mHandles[i]);
			}
		}
		mHandles.clear();
		mEnabled = false;
		return;
	}

#if PX_SUPPORT_GPU_PHYSX
	switch (mInternalAttachmentType)
	{
		case NpInternalAttachmentType::eSURFACE_VTX_RIGID_BODY:
		case NpInternalAttachmentType::eSURFACE_TRI_RIGID_BODY:
		case NpInternalAttachmentType::eSURFACE_VTX_GLOBAL_POSE:
		case NpInternalAttachmentType::eVOLUME_VTX_RIGID_BODY:
		case NpInternalAttachmentType::eVOLUME_TET_RIGID_BODY:
		case NpInternalAttachmentType::eVOLUME_VTX_GLOBAL_POSE:
		{
			PxActor* actor0 = mActor[mActorIndex[0]];

			PxRigidActor* actor1 = mActor[mActorIndex[1]] == NULL ? NULL : mActor[mActorIndex[1]]->is<PxRigidActor>();
			Sc::BodyCore* core1 = mActor[mActorIndex[1]] == NULL ? NULL : getBodyCore(actor1);

			for (PxU32 i = 0; i < mHandles.size(); i++)
			{
				if (mInternalAttachmentType & NpInternalAttachmentType::eSURFACE_TYPE)
				{
					if (mInternalAttachmentType == NpInternalAttachmentType::eSURFACE_TRI_RIGID_BODY)
					{
						getDeformableSurfaceCore(actor0)->removeTriRigidAttachment(core1, mHandles[i]);
					}
					else if (mInternalAttachmentType == NpInternalAttachmentType::eSURFACE_VTX_RIGID_BODY || mInternalAttachmentType == NpInternalAttachmentType::eSURFACE_VTX_GLOBAL_POSE)
					{
						getDeformableSurfaceCore(actor0)->removeRigidAttachment(core1, mHandles[i]);
					}
				}
				else if (mInternalAttachmentType & NpInternalAttachmentType::eVOLUME_TYPE)
				{
					getDeformableVolumeCore(actor0)->removeRigidAttachment(core1, mHandles[i]);
				}
			}

			break;
		}

		case NpInternalAttachmentType::eSURFACE_TRI_SURFACE_TRI:
		case NpInternalAttachmentType::eVOLUME_TET_VOLUME_TET:
		case NpInternalAttachmentType::eVOLUME_TET_SURFACE_TRI:
		{
			PxActor* actor0 = mActor[mActorIndex[0]];
			PxActor* actor1 = mActor[mActorIndex[1]];

			for (PxU32 i = 0; i < mHandles.size(); i++)
			{
				if (mInternalAttachmentType == NpInternalAttachmentType::eSURFACE_TRI_SURFACE_TRI)
				{
					getDeformableSurfaceCore(actor0)->removeClothAttachment(getDeformableSurfaceCore(actor1), mHandles[i]);
				}
				else if (mInternalAttachmentType == NpInternalAttachmentType::eVOLUME_TET_VOLUME_TET)
				{
					getDeformableVolumeCore(actor0)->removeSoftBodyAttachment(*getDeformableVolumeCore(actor1), mHandles[i]);
				}
				else if (mInternalAttachmentType == NpInternalAttachmentType::eVOLUME_TET_SURFACE_TRI)
				{
					getDeformableVolumeCore(actor0)->removeClothAttachment(*getDeformableSurfaceCore(actor1), mHandles[i]);
				}
			}

			break;
		}

		default:
		{
			PX_ASSERT(0);
			break;
		}
	}

	mHandles.clear();
	mEnabled = false;
#else
	PX_ASSERT(0);
#endif
}

bool NpDeformableAttachment::isCpuAvbdWorldVertexAttachment() const
{
	const PxActor* actor = mActor[mActorIndex[0]];
	if(mInternalAttachmentType ==
		NpInternalAttachmentType::eSURFACE_VTX_GLOBAL_POSE)
	{
		const PxDeformableSurface* surface =
			actor ? actor->is<PxDeformableSurface>() : NULL;
		return surface &&
			surface->getDeformableSurfaceBackend() ==
				PxDeformableSurfaceBackend::eCPU_AVBD;
	}
	if(mInternalAttachmentType ==
		NpInternalAttachmentType::eVOLUME_VTX_GLOBAL_POSE)
	{
		const PxDeformableVolume* volume =
			actor ? actor->is<PxDeformableVolume>() : NULL;
		return volume &&
			volume->getDeformableVolumeBackend() ==
				PxDeformableVolumeBackend::eCPU_AVBD;
	}
	return false;
}

bool NpDeformableAttachment::isCpuAvbdWorldElementAttachment() const
{
	const PxActor* actor = mActor[mActorIndex[0]];
	if(mInternalAttachmentType ==
		NpInternalAttachmentType::eSURFACE_TRI_GLOBAL_POSE)
	{
		const PxDeformableSurface* surface =
			actor ? actor->is<PxDeformableSurface>() : NULL;
		return surface &&
			surface->getDeformableSurfaceBackend() ==
				PxDeformableSurfaceBackend::eCPU_AVBD;
	}
	if(mInternalAttachmentType ==
		NpInternalAttachmentType::eVOLUME_TET_GLOBAL_POSE)
	{
		const PxDeformableVolume* volume =
			actor ? actor->is<PxDeformableVolume>() : NULL;
		return volume &&
			volume->getDeformableVolumeBackend() ==
				PxDeformableVolumeBackend::eCPU_AVBD;
	}
	return false;
}

bool NpDeformableAttachment::isCpuAvbdStaticVertexAttachment() const
{
	const PxActor* softActor = mActor[mActorIndex[0]];
	const PxRigidStatic* staticActor = mActor[mActorIndex[1]]
		? mActor[mActorIndex[1]]->is<PxRigidStatic>()
		: NULL;
	if(!staticActor)
		return false;
	if(mInternalAttachmentType ==
		NpInternalAttachmentType::eSURFACE_VTX_RIGID_BODY)
	{
		const PxDeformableSurface* surface =
			softActor
				? softActor->is<PxDeformableSurface>()
				: NULL;
		return surface &&
			surface->getDeformableSurfaceBackend() ==
				PxDeformableSurfaceBackend::eCPU_AVBD;
	}
	if(mInternalAttachmentType ==
		NpInternalAttachmentType::eVOLUME_VTX_RIGID_BODY)
	{
		const PxDeformableVolume* volume =
			softActor
				? softActor->is<PxDeformableVolume>()
				: NULL;
		return volume &&
			volume->getDeformableVolumeBackend() ==
				PxDeformableVolumeBackend::eCPU_AVBD;
	}
	return false;
}

bool NpDeformableAttachment::isCpuAvbdStaticElementAttachment() const
{
	const PxActor* softActor = mActor[mActorIndex[0]];
	const PxRigidStatic* staticActor = mActor[mActorIndex[1]]
		? mActor[mActorIndex[1]]->is<PxRigidStatic>()
		: NULL;
	if(!staticActor)
		return false;
	if(mInternalAttachmentType ==
		NpInternalAttachmentType::eSURFACE_TRI_RIGID_BODY)
	{
		const PxDeformableSurface* surface =
			softActor
				? softActor->is<PxDeformableSurface>()
				: NULL;
		return surface &&
			surface->getDeformableSurfaceBackend() ==
				PxDeformableSurfaceBackend::eCPU_AVBD;
	}
	if(mInternalAttachmentType ==
		NpInternalAttachmentType::eVOLUME_TET_RIGID_BODY)
	{
		const PxDeformableVolume* volume =
			softActor
				? softActor->is<PxDeformableVolume>()
				: NULL;
		return volume &&
			volume->getDeformableVolumeBackend() ==
				PxDeformableVolumeBackend::eCPU_AVBD;
	}
	return false;
}

bool NpDeformableAttachment::isCpuAvbdRigidVertexAttachment() const
{
	const PxActor* softActor = mActor[mActorIndex[0]];
	const PxRigidDynamic* rigidActor = mActor[mActorIndex[1]]
		? mActor[mActorIndex[1]]->is<PxRigidDynamic>()
		: NULL;
	if(!rigidActor ||
		(rigidActor->getRigidBodyFlags() &
			PxRigidBodyFlag::eKINEMATIC))
		return false;
	if(mInternalAttachmentType ==
		NpInternalAttachmentType::eSURFACE_VTX_RIGID_BODY)
	{
		const PxDeformableSurface* surface =
			softActor
				? softActor->is<PxDeformableSurface>()
				: NULL;
		return surface &&
			surface->getDeformableSurfaceBackend() ==
				PxDeformableSurfaceBackend::eCPU_AVBD;
	}
	if(mInternalAttachmentType ==
		NpInternalAttachmentType::eVOLUME_VTX_RIGID_BODY)
	{
		const PxDeformableVolume* volume =
			softActor
				? softActor->is<PxDeformableVolume>()
				: NULL;
		return volume &&
			volume->getDeformableVolumeBackend() ==
				PxDeformableVolumeBackend::eCPU_AVBD;
	}
	return false;
}

bool NpDeformableAttachment::isCpuAvbdRigidElementAttachment() const
{
	const PxActor* softActor = mActor[mActorIndex[0]];
	const PxRigidDynamic* rigidActor = mActor[mActorIndex[1]]
		? mActor[mActorIndex[1]]->is<PxRigidDynamic>()
		: NULL;
	if(!rigidActor ||
		(rigidActor->getRigidBodyFlags() &
			PxRigidBodyFlag::eKINEMATIC))
		return false;
	if(mInternalAttachmentType ==
		NpInternalAttachmentType::eSURFACE_TRI_RIGID_BODY)
	{
		const PxDeformableSurface* surface =
			softActor
				? softActor->is<PxDeformableSurface>()
				: NULL;
		return surface &&
			surface->getDeformableSurfaceBackend() ==
				PxDeformableSurfaceBackend::eCPU_AVBD;
	}
	if(mInternalAttachmentType ==
		NpInternalAttachmentType::eVOLUME_TET_RIGID_BODY)
	{
		const PxDeformableVolume* volume =
			softActor
				? softActor->is<PxDeformableVolume>()
				: NULL;
		return volume &&
			volume->getDeformableVolumeBackend() ==
				PxDeformableVolumeBackend::eCPU_AVBD;
	}
	return false;
}

bool NpDeformableAttachment::
	isCpuAvbdKinematicVertexAttachment() const
{
	const PxActor* softActor = mActor[mActorIndex[0]];
	const PxRigidDynamic* rigidActor = mActor[mActorIndex[1]]
		? mActor[mActorIndex[1]]->is<PxRigidDynamic>()
		: NULL;
	if(!rigidActor ||
		!(rigidActor->getRigidBodyFlags() &
			PxRigidBodyFlag::eKINEMATIC))
		return false;
	if(mInternalAttachmentType ==
		NpInternalAttachmentType::eSURFACE_VTX_RIGID_BODY)
	{
		const PxDeformableSurface* surface =
			softActor
				? softActor->is<PxDeformableSurface>()
				: NULL;
		return surface &&
			surface->getDeformableSurfaceBackend() ==
				PxDeformableSurfaceBackend::eCPU_AVBD;
	}
	if(mInternalAttachmentType ==
		NpInternalAttachmentType::eVOLUME_VTX_RIGID_BODY)
	{
		const PxDeformableVolume* volume =
			softActor
				? softActor->is<PxDeformableVolume>()
				: NULL;
		return volume &&
			volume->getDeformableVolumeBackend() ==
				PxDeformableVolumeBackend::eCPU_AVBD;
	}
	return false;
}

bool NpDeformableAttachment::
	isCpuAvbdKinematicElementAttachment() const
{
	const PxActor* softActor = mActor[mActorIndex[0]];
	const PxRigidDynamic* rigidActor = mActor[mActorIndex[1]]
		? mActor[mActorIndex[1]]->is<PxRigidDynamic>()
		: NULL;
	if(!rigidActor ||
		!(rigidActor->getRigidBodyFlags() &
			PxRigidBodyFlag::eKINEMATIC))
		return false;
	if(mInternalAttachmentType ==
		NpInternalAttachmentType::eSURFACE_TRI_RIGID_BODY)
	{
		const PxDeformableSurface* surface =
			softActor
				? softActor->is<PxDeformableSurface>()
				: NULL;
		return surface &&
			surface->getDeformableSurfaceBackend() ==
				PxDeformableSurfaceBackend::eCPU_AVBD;
	}
	if(mInternalAttachmentType ==
		NpInternalAttachmentType::eVOLUME_TET_RIGID_BODY)
	{
		const PxDeformableVolume* volume =
			softActor
				? softActor->is<PxDeformableVolume>()
				: NULL;
		return volume &&
			volume->getDeformableVolumeBackend() ==
				PxDeformableVolumeBackend::eCPU_AVBD;
	}
	return false;
}

bool NpDeformableAttachment::
	isCpuAvbdArticulationVertexAttachment() const
{
	const PxActor* softActor = mActor[mActorIndex[0]];
	const PxArticulationLink* articulationLink =
		mActor[mActorIndex[1]]
			? mActor[mActorIndex[1]]->is<PxArticulationLink>()
			: NULL;
	if(!articulationLink)
		return false;
	if(mInternalAttachmentType ==
		NpInternalAttachmentType::eSURFACE_VTX_RIGID_BODY)
	{
		const PxDeformableSurface* surface =
			softActor
				? softActor->is<PxDeformableSurface>()
				: NULL;
		return surface &&
			surface->getDeformableSurfaceBackend() ==
				PxDeformableSurfaceBackend::eCPU_AVBD;
	}
	if(mInternalAttachmentType ==
		NpInternalAttachmentType::eVOLUME_VTX_RIGID_BODY)
	{
		const PxDeformableVolume* volume =
			softActor
				? softActor->is<PxDeformableVolume>()
				: NULL;
		return volume &&
			volume->getDeformableVolumeBackend() ==
				PxDeformableVolumeBackend::eCPU_AVBD;
	}
	return false;
}

bool NpDeformableAttachment::
	isCpuAvbdArticulationElementAttachment() const
{
	const PxActor* softActor = mActor[mActorIndex[0]];
	const PxArticulationLink* articulationLink =
		mActor[mActorIndex[1]]
			? mActor[mActorIndex[1]]->is<PxArticulationLink>()
			: NULL;
	if(!articulationLink)
		return false;
	if(mInternalAttachmentType ==
		NpInternalAttachmentType::eSURFACE_TRI_RIGID_BODY)
	{
		const PxDeformableSurface* surface =
			softActor
				? softActor->is<PxDeformableSurface>()
				: NULL;
		return surface &&
			surface->getDeformableSurfaceBackend() ==
				PxDeformableSurfaceBackend::eCPU_AVBD;
	}
	if(mInternalAttachmentType ==
		NpInternalAttachmentType::eVOLUME_TET_RIGID_BODY)
	{
		const PxDeformableVolume* volume =
			softActor
				? softActor->is<PxDeformableVolume>()
				: NULL;
		return volume &&
			volume->getDeformableVolumeBackend() ==
				PxDeformableVolumeBackend::eCPU_AVBD;
	}
	return false;
}

bool NpDeformableAttachment::isCpuAvbdSoftPairAttachment() const
{
	const PxActor* actor0 = mActor[mActorIndex[0]];
	const PxActor* actor1 = mActor[mActorIndex[1]];
	if(!actor0 || !actor1 || actor0 == actor1)
		return false;
	const PxDeformableSurface* surface0 =
		actor0->is<PxDeformableSurface>();
	const PxDeformableSurface* surface1 =
		actor1->is<PxDeformableSurface>();
	const PxDeformableVolume* volume0 =
		actor0->is<PxDeformableVolume>();
	const PxDeformableVolume* volume1 =
		actor1->is<PxDeformableVolume>();
	const bool endpoint0Cpu =
		(surface0 &&
		 surface0->getDeformableSurfaceBackend() ==
			PxDeformableSurfaceBackend::eCPU_AVBD) ||
		(volume0 &&
		 volume0->getDeformableVolumeBackend() ==
			PxDeformableVolumeBackend::eCPU_AVBD);
	const bool endpoint1Cpu =
		(surface1 &&
		 surface1->getDeformableSurfaceBackend() ==
			PxDeformableSurfaceBackend::eCPU_AVBD) ||
		(volume1 &&
		 volume1->getDeformableVolumeBackend() ==
			PxDeformableVolumeBackend::eCPU_AVBD);
	if(!endpoint0Cpu || !endpoint1Cpu)
		return false;

	switch(mInternalAttachmentType)
	{
	case NpInternalAttachmentType::eSURFACE_VTX_SURFACE_VTX:
	case NpInternalAttachmentType::eSURFACE_TRI_SURFACE_VTX:
	case NpInternalAttachmentType::eSURFACE_TRI_SURFACE_TRI:
	case NpInternalAttachmentType::eVOLUME_VTX_SURFACE_VTX:
	case NpInternalAttachmentType::eVOLUME_TET_SURFACE_VTX:
	case NpInternalAttachmentType::eVOLUME_TET_SURFACE_TRI:
	case NpInternalAttachmentType::eVOLUME_VTX_VOLUME_VTX:
	case NpInternalAttachmentType::eVOLUME_TET_VOLUME_VTX:
	case NpInternalAttachmentType::eVOLUME_TET_VOLUME_TET:
		return true;
	default:
		return false;
	}
}

NpScene* NpDeformableAttachment::getSceneFromActors()
{
	const PxActor* actor[2] = { mActor[mActorIndex[0]], mActor[mActorIndex[1]] };

	for (PxU32 i = 0; i < 2; i++)
	{
		if (actor[i] && (actor[i]->getScene() == NULL))
			return NULL;
	}

	if (actor[0] && actor[1])
	{
		if (actor[0]->getScene() != actor[1]->getScene())
		{
			PX_CHECK_MSG(false, "PxDeformableAttachment: Actors belong to different scenes, undefined behavior expected!");

			return NULL;
		}
	}

	return static_cast<NpScene*>(actor[0]->getScene());
}

NpDeformableAttachment::NpDeformableAttachment(const PxDeformableAttachmentData& data, const AttachmentInfo& info)
	: PxDeformableAttachment(PxConcreteType::eDEFORMABLE_ATTACHMENT, PxBaseFlag::eOWNS_MEMORY | PxBaseFlag::eIS_RELEASABLE), NpBase(NpType::eDEFORMABLE_ATTACHMENT)
{
	mInternalAttachmentType = info.internalAttachmentType;
	mEnabled = false;

	for (PxU32 i = 0; i < 2; i++)
	{
		mActor[i] = data.actor[i];
		mType[i]  = data.type[i];
		mPose[i]  = data.pose[i];

		mIndices[i].resize(data.indices[i].count);
		for (PxU32 j = 0; j < data.indices[i].count; j++)
		{
			mIndices[i][j] = data.indices[i].at(j);
		}

		mCoords[i].resize(data.coords[i].count);
		for (PxU32 j = 0; j < data.coords[i].count; j++)
		{
			mCoords[i][j] = data.coords[i].at(j);
		}

		mActorIndex[i] = info.actorIndex[i];

		// Add connector
		if (mActor[i])
			NpActor::getFromPxActor(*mActor[i]).addConnector(NpConnectorType::eAttachment, this, "PxDeformableAttachment: Attachment already added");
	}
	if(isCpuAvbdSoftPairAttachment())
		mCpuAvbdRoute = CpuAvbdRoute::eSOFT_PAIR;
	else if(isCpuAvbdWorldVertexAttachment())
		mCpuAvbdRoute = CpuAvbdRoute::eWORLD_PIN;
	else if(isCpuAvbdWorldElementAttachment())
		mCpuAvbdRoute = CpuAvbdRoute::eWORLD_ELEMENT;
	else if(isCpuAvbdStaticVertexAttachment())
		mCpuAvbdRoute = CpuAvbdRoute::eSTATIC_RIGID;
	else if(isCpuAvbdStaticElementAttachment())
		mCpuAvbdRoute = CpuAvbdRoute::eSTATIC_RIGID_ELEMENT;
	else if(isCpuAvbdArticulationVertexAttachment())
		mCpuAvbdRoute = CpuAvbdRoute::eARTICULATION_LINK;
	else if(isCpuAvbdArticulationElementAttachment())
		mCpuAvbdRoute = CpuAvbdRoute::eARTICULATION_LINK_ELEMENT;
	else if(isCpuAvbdKinematicVertexAttachment())
		mCpuAvbdRoute = CpuAvbdRoute::eKINEMATIC_RIGID;
	else if(isCpuAvbdKinematicElementAttachment())
		mCpuAvbdRoute = CpuAvbdRoute::eKINEMATIC_RIGID_ELEMENT;
	else if(isCpuAvbdRigidVertexAttachment())
		mCpuAvbdRoute = CpuAvbdRoute::eDYNAMIC_RIGID;
	else if(isCpuAvbdRigidElementAttachment())
		mCpuAvbdRoute = CpuAvbdRoute::eDYNAMIC_RIGID_ELEMENT;

	NpScene* s = getSceneFromActors();
	if (s)
	{
		PX_CHECK_SCENE_API_WRITE_FORBIDDEN(s, "PxDeformableAttachment creation not allowed while simulation is running. Call will be ignored.");

		s->addToAttachmentList(*this);
	}

	setNpScene(s);
}

NpDeformableAttachment::~NpDeformableAttachment()
{
	NpFactory::getInstance().onAttachmentRelease(this);
}

void NpDeformableAttachment::release()
{
	NpScene* npScene = getNpScene();
	NP_WRITE_CHECK(npScene);

	PX_CHECK_SCENE_API_WRITE_FORBIDDEN(npScene, "PxDeformableAttachment: Illegal to call release while simulation is running.");

	for (PxU32 i = 0; i < 2; i++)
	{
		// Remove connector
		if (mActor[i])
		{
			PxDeformableAttachment* buffer[1];

			if (NpActor::getFromPxActor(*mActor[i]).getConnectors(NpConnectorType::eAttachment, buffer, 1))
				NpActor::getFromPxActor(*mActor[i]).removeConnector(*mActor[i], NpConnectorType::eAttachment, this, "PxDeformableAttachment: Attachment already released");
		}
	}
	
	if (npScene)
	{
		npScene->removeFromAttachmentList(*this);
	}
	
	NpDestroyAttachment(this);
}

void NpDeformableAttachment::getActors(PxActor*& actor0, PxActor*& actor1) const
{
	NP_READ_CHECK(getNpScene());

	actor0 = mActor[0];
	actor1 = mActor[1];
}

void NpDeformableAttachment::updatePose(const PxTransform& pose)
{
	NP_WRITE_CHECK(getNpScene());
	PX_CHECK_AND_RETURN(
		mCpuAvbdRoute != CpuAvbdRoute::eSOFT_PAIR,
		"PxDeformableAttachment: updatePose is not defined for a "
		"CPU AVBD deformable-pair attachment.");

	if(mCpuAvbdRoute == CpuAvbdRoute::eWORLD_PIN ||
		mCpuAvbdRoute == CpuAvbdRoute::eWORLD_ELEMENT)
	{
		const PxU32 worldIndex = mActorIndex[1];
		if(mEnabled)
		{
			NpScene* scene = getSceneFromActors();
			PxActor* actor = mActor[mActorIndex[0]];
			const bool surfaceType =
				(mInternalAttachmentType &
					NpInternalAttachmentType::eSURFACE_TYPE) != 0;
			PX_CHECK_AND_RETURN(
				scene != NULL,
				"PxDeformableAttachment: CPU AVBD world "
				"attachment has no Scene.");
			for(PxU32 i = 0; i < mHandles.size(); i++)
			{
				const PxVec3 worldTarget = pose.transform(
					mCoords[worldIndex][i].getXYZ());
				const bool updated = surfaceType
					? scene->getScScene().
						updateAvbdCpuDeformableSurfaceWorldPin(
							*getDeformableSurfaceCore(actor),
							mHandles[i], worldTarget)
					: scene->getScScene().
						updateAvbdCpuDeformableVolumeWorldPin(
							*getDeformableVolumeCore(actor),
							mHandles[i], worldTarget);
				PX_CHECK_AND_RETURN(
					updated,
					"PxDeformableAttachment: Failed to update CPU "
					"AVBD world attachment pose.");
				PX_UNUSED(updated);
			}
		}
		mPose[worldIndex] = pose;
		return;
	}

	if(mCpuAvbdRoute == CpuAvbdRoute::eSTATIC_RIGID ||
		mCpuAvbdRoute == CpuAvbdRoute::eSTATIC_RIGID_ELEMENT)
	{
		const PxU32 rigidIndex = mActorIndex[1];
		if(mEnabled)
		{
			NpScene* scene = getSceneFromActors();
			PxActor* softActor = mActor[mActorIndex[0]];
			const bool surfaceType =
				(mInternalAttachmentType &
					NpInternalAttachmentType::eSURFACE_TYPE) != 0;
			PX_CHECK_AND_RETURN(
				scene != NULL,
				"PxDeformableAttachment: CPU AVBD static "
				"attachment has no Scene.");
			for(PxU32 i = 0; i < mHandles.size(); i++)
			{
				const PxVec3 actorLocalTarget = pose.transform(
					mCoords[rigidIndex][i].getXYZ());
				const bool updated = surfaceType
					? scene->getScScene().
						updateAvbdCpuDeformableSurfaceStaticAttachment(
							*getDeformableSurfaceCore(softActor),
							mHandles[i], actorLocalTarget)
					: scene->getScScene().
						updateAvbdCpuDeformableVolumeStaticAttachment(
							*getDeformableVolumeCore(softActor),
							mHandles[i], actorLocalTarget);
				if(!updated)
				{
					for(PxU32 j = 0; j < i; j++)
					{
						const PxVec3 oldActorLocalTarget =
							mPose[rigidIndex].transform(
								mCoords[rigidIndex][j].
									getXYZ());
						if(surfaceType)
							scene->getScScene().
								updateAvbdCpuDeformableSurfaceStaticAttachment(
									*getDeformableSurfaceCore(
										softActor),
									mHandles[j],
									oldActorLocalTarget);
						else
							scene->getScScene().
								updateAvbdCpuDeformableVolumeStaticAttachment(
									*getDeformableVolumeCore(
										softActor),
									mHandles[j],
									oldActorLocalTarget);
					}
					PX_CHECK_AND_RETURN(
						false,
						"PxDeformableAttachment: Failed to update "
						"CPU AVBD static attachment pose.");
				}
			}
		}
		mPose[rigidIndex] = pose;
		return;
	}

	const bool cpuAvbdKinematicAttachment =
		mCpuAvbdRoute == CpuAvbdRoute::eKINEMATIC_RIGID ||
		mCpuAvbdRoute ==
			CpuAvbdRoute::eKINEMATIC_RIGID_ELEMENT;
	const bool cpuAvbdArticulationAttachment =
		mCpuAvbdRoute == CpuAvbdRoute::eARTICULATION_LINK ||
		mCpuAvbdRoute ==
			CpuAvbdRoute::eARTICULATION_LINK_ELEMENT;
	if(mCpuAvbdRoute == CpuAvbdRoute::eDYNAMIC_RIGID ||
		mCpuAvbdRoute == CpuAvbdRoute::eDYNAMIC_RIGID_ELEMENT ||
		cpuAvbdKinematicAttachment ||
		cpuAvbdArticulationAttachment)
	{
		const PxU32 rigidIndex = mActorIndex[1];
		if(mEnabled)
		{
			NpScene* scene = getSceneFromActors();
			PxActor* softActor = mActor[mActorIndex[0]];
			const bool surfaceType =
				(mInternalAttachmentType &
					NpInternalAttachmentType::eSURFACE_TYPE) != 0;
			PX_CHECK_AND_RETURN(
				scene != NULL,
				"PxDeformableAttachment: CPU AVBD rigid "
				"attachment has no Scene.");
			for(PxU32 i = 0; i < mHandles.size(); i++)
			{
				const PxVec3 actorLocalTarget = pose.transform(
					mCoords[rigidIndex][i].getXYZ());
				bool updated;
				if(surfaceType)
				{
					if(cpuAvbdArticulationAttachment)
						updated = scene->getScScene().
							updateAvbdCpuDeformableSurfaceArticulationAttachment(
								*getDeformableSurfaceCore(
									softActor),
								mHandles[i], actorLocalTarget);
					else if(cpuAvbdKinematicAttachment)
						updated = scene->getScScene().
							updateAvbdCpuDeformableSurfaceKinematicAttachment(
								*getDeformableSurfaceCore(
									softActor),
								mHandles[i], actorLocalTarget);
					else
						updated = scene->getScScene().
							updateAvbdCpuDeformableSurfaceRigidAttachment(
								*getDeformableSurfaceCore(
									softActor),
								mHandles[i], actorLocalTarget);
				}
				else
				{
					if(cpuAvbdArticulationAttachment)
						updated = scene->getScScene().
							updateAvbdCpuDeformableVolumeArticulationAttachment(
								*getDeformableVolumeCore(
									softActor),
								mHandles[i], actorLocalTarget);
					else if(cpuAvbdKinematicAttachment)
						updated = scene->getScScene().
							updateAvbdCpuDeformableVolumeKinematicAttachment(
								*getDeformableVolumeCore(
									softActor),
								mHandles[i], actorLocalTarget);
					else
						updated = scene->getScScene().
							updateAvbdCpuDeformableVolumeRigidAttachment(
								*getDeformableVolumeCore(
									softActor),
								mHandles[i], actorLocalTarget);
				}
				if(!updated)
				{
					for(PxU32 j = 0; j < i; j++)
					{
						const PxVec3 oldActorLocalTarget =
							mPose[rigidIndex].transform(
								mCoords[rigidIndex][j].
									getXYZ());
						if(surfaceType)
						{
							if(cpuAvbdArticulationAttachment)
								scene->getScScene().
									updateAvbdCpuDeformableSurfaceArticulationAttachment(
										*getDeformableSurfaceCore(
											softActor),
										mHandles[j],
										oldActorLocalTarget);
							else if(cpuAvbdKinematicAttachment)
								scene->getScScene().
									updateAvbdCpuDeformableSurfaceKinematicAttachment(
										*getDeformableSurfaceCore(
											softActor),
										mHandles[j],
										oldActorLocalTarget);
							else
								scene->getScScene().
									updateAvbdCpuDeformableSurfaceRigidAttachment(
										*getDeformableSurfaceCore(
											softActor),
										mHandles[j],
										oldActorLocalTarget);
						}
						else if(cpuAvbdArticulationAttachment)
							scene->getScScene().
								updateAvbdCpuDeformableVolumeArticulationAttachment(
									*getDeformableVolumeCore(
										softActor),
									mHandles[j],
									oldActorLocalTarget);
						else if(cpuAvbdKinematicAttachment)
							scene->getScScene().
								updateAvbdCpuDeformableVolumeKinematicAttachment(
									*getDeformableVolumeCore(
										softActor),
									mHandles[j],
									oldActorLocalTarget);
						else
							scene->getScScene().
								updateAvbdCpuDeformableVolumeRigidAttachment(
									*getDeformableVolumeCore(
										softActor),
									mHandles[j],
									oldActorLocalTarget);
					}
					PX_CHECK_AND_RETURN(
						false,
						"PxDeformableAttachment: Failed to update "
						"CPU AVBD rigid attachment pose.");
				}
			}
		}
		mPose[rigidIndex] = pose;
		return;
	}

#if PX_SUPPORT_GPU_PHYSX
	switch (mInternalAttachmentType)
	{
		case NpInternalAttachmentType::eSURFACE_VTX_RIGID_BODY:
		case NpInternalAttachmentType::eSURFACE_TRI_RIGID_BODY:
		case NpInternalAttachmentType::eSURFACE_VTX_GLOBAL_POSE:
		case NpInternalAttachmentType::eVOLUME_VTX_RIGID_BODY:
		case NpInternalAttachmentType::eVOLUME_TET_RIGID_BODY:
		case NpInternalAttachmentType::eVOLUME_VTX_GLOBAL_POSE:
		{
			PxActor* actor0 = mActor[mActorIndex[0]];

			PxRigidActor* actor1 = mActor[mActorIndex[1]] == NULL ? NULL : mActor[mActorIndex[1]]->is<PxRigidActor>();
			Sc::BodyCore* core1 = mActor[mActorIndex[1]] == NULL ? NULL : getBodyCore(actor1);

			for (PxU32 i = 0; i < mHandles.size(); i++)
			{
				if (mInternalAttachmentType & NpInternalAttachmentType::eSURFACE_TYPE)
				{
					if (mInternalAttachmentType == NpInternalAttachmentType::eSURFACE_TRI_RIGID_BODY)
					{
						getDeformableSurfaceCore(actor0)->removeTriRigidAttachment(core1, mHandles[i]);
					}
					else
					{
						getDeformableSurfaceCore(actor0)->removeRigidAttachment(core1, mHandles[i]);
					}
				}
				else if (mInternalAttachmentType & NpInternalAttachmentType::eVOLUME_TYPE)
				{
					getDeformableVolumeCore(actor0)->removeRigidAttachment(core1, mHandles[i]);
				}
			}

			for (PxU32 i = 0; i < mHandles.size(); i++)
			{
				PxU32 id = mIndices[mActorIndex[0]][i];
				PxVec3 actor1Pose = mCoords[mActorIndex[1]][i].getXYZ();

				if (actor1 && actor1->getConcreteType() == PxConcreteType::eRIGID_STATIC)
				{
					NpRigidStatic* stat = static_cast<NpRigidStatic*>(actor1);
					actor1Pose = stat->getGlobalPose().transform(actor1Pose);
				}

				actor1Pose = pose.transform(actor1Pose);

				if (mInternalAttachmentType & NpInternalAttachmentType::eSURFACE_TYPE)
				{
					if (mInternalAttachmentType == NpInternalAttachmentType::eSURFACE_TRI_RIGID_BODY)
					{
						const PxVec4& barycentric = mCoords[mActorIndex[0]][i];

						mHandles[i] = getDeformableSurfaceCore(actor0)->addTriRigidAttachment(core1, id, barycentric, actor1Pose);
					}
					else
					{
						mHandles[i] = getDeformableSurfaceCore(actor0)->addRigidAttachment(core1, id, actor1Pose);
					}
				}
				else if (mInternalAttachmentType & NpInternalAttachmentType::eVOLUME_TYPE)
				{
					if (mInternalAttachmentType == NpInternalAttachmentType::eVOLUME_TET_RIGID_BODY)
					{
						const PxVec4& barycentric = mCoords[mActorIndex[0]][i];

						mHandles[i] = getDeformableVolumeCore(actor0)->addTetRigidAttachment(core1, id, barycentric, actor1Pose, false);
					}
					else
					{
						mHandles[i] = getDeformableVolumeCore(actor0)->addRigidAttachment(core1, id, actor1Pose, false);
					}
				}
			}

			mPose[mActorIndex[1]] = pose;

			break;
		}

		default:
		{
			PX_CHECK_AND_RETURN(false, "PxDeformableAttachment: Updating of pose is not supported for this attachment type.");
		}
	}
#else
	PX_CHECK_AND_RETURN(
		false,
		"PxDeformableAttachment: Updating this attachment pose "
		"requires GPU PhysX.");
#endif
}
