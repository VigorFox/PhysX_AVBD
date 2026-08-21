// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions in the PhysX SDK
// license are met.

#include "avbd/solver/soft/DyAvbdSoftBodyData.h"

namespace physx
{
namespace Dy
{

void AvbdSoftBodyMaterialData::computeLameParameters()
{
	mu = youngsModulus / (2.0f * (1.0f + poissonsRatio));
	lambda = youngsModulus * poissonsRatio /
		((1.0f + poissonsRatio) * (1.0f - 2.0f * poissonsRatio));
	const PxReal lambdaSafe = PxAbs(lambda) < 1e-6f ? 1e-6f : lambda;
	neoHookeanAlpha = 1.0f + mu / lambdaSafe;
}

void AvbdSoftBodyRuntimeState::compileObjectiveProgram(
	PxU32 particleStart, PxU32 particleCount)
{
	compiledObjectives.clear();
	objectiveAdjacency.resize(particleCount);
	for(PxU32 i = 0; i < particleCount; i++)
		objectiveAdjacency[i].objectiveIndices.clear();

	for(PxU32 ai = 0; ai < attachments.size(); ai++)
	{
		const AvbdSoftAttachment& attachment = attachments[ai];
		const bool pointIsValid = avbdIsSoftPointValid(
			attachment.point, particleStart, particleCount);
		AvbdCompiledSoftObjective objective;
		objective.owner = avbdGetAttachmentObjectiveOwner(
			attachment, pointIsValid);
		objective.runtimeStateIndex = ai;
		objective.point = attachment.point;
		objective.targetPoint = attachment.targetPoint;
		objective.rigidBodyIdx = attachment.rigidBodyIdx;
		const PxU32 objectiveIndex = compiledObjectives.size();
		compiledObjectives.pushBack(objective);
		if(pointIsValid)
		{
			for(PxU32 endpoint = 0;
				endpoint < attachment.point.particleCount; endpoint++)
			{
				const PxU32 particleIndex =
					attachment.point.particleIndices[endpoint];
				bool firstOccurrence = true;
				for(PxU32 previous = 0; previous < endpoint; previous++)
				{
					if(attachment.point.particleIndices[previous] ==
						particleIndex)
					{
						firstOccurrence = false;
						break;
					}
				}
				if(firstOccurrence)
					objectiveAdjacency[particleIndex - particleStart].
						objectiveIndices.pushBack(objectiveIndex);
			}
		}
	}

	for(PxU32 pi = 0; pi < pins.size(); pi++)
	{
		const AvbdKinematicPin& pin = pins[pi];
		const bool pointIsValid = avbdIsSoftPointValid(
			pin.point, particleStart, particleCount);
		AvbdCompiledSoftObjective objective;
		objective.owner = avbdGetPinObjectiveOwner(pin, pointIsValid);
		objective.runtimeStateIndex = pi;
		objective.point = pin.point;
		objective.rigidBodyIdx = PX_MAX_U32;
		const PxU32 objectiveIndex = compiledObjectives.size();
		compiledObjectives.pushBack(objective);
		if(pointIsValid)
		{
			for(PxU32 endpoint = 0;
				endpoint < pin.point.particleCount; endpoint++)
			{
				const PxU32 particleIndex = pin.point.particleIndices[endpoint];
				bool firstOccurrence = true;
				for(PxU32 previous = 0; previous < endpoint; previous++)
				{
					if(pin.point.particleIndices[previous] == particleIndex)
					{
						firstOccurrence = false;
						break;
					}
				}
				if(firstOccurrence)
					objectiveAdjacency[particleIndex - particleStart].
						objectiveIndices.pushBack(objectiveIndex);
			}
		}
	}
}

bool AvbdSoftBodyRuntimeState::isObjectiveProgramCurrent(
	PxU32 particleStart, PxU32 particleCount) const
{
	if(objectiveAdjacency.size() != particleCount ||
		compiledObjectives.size() != attachments.size() + pins.size())
		return false;

	for(PxU32 ai = 0; ai < attachments.size(); ai++)
	{
		const AvbdSoftAttachment& attachment = attachments[ai];
		const AvbdCompiledSoftObjective& objective = compiledObjectives[ai];
		const bool pointIsValid = avbdIsSoftPointValid(
			attachment.point, particleStart, particleCount);
		const AvbdSoftObjectiveOwner expectedOwner =
			avbdGetAttachmentObjectiveOwner(attachment, pointIsValid);
		if(objective.owner != expectedOwner ||
			objective.runtimeStateIndex != ai ||
			!(objective.point == attachment.point) ||
			!(objective.targetPoint == attachment.targetPoint) ||
			objective.rigidBodyIdx != attachment.rigidBodyIdx)
			return false;
	}

	for(PxU32 pi = 0; pi < pins.size(); pi++)
	{
		const AvbdKinematicPin& pin = pins[pi];
		const AvbdCompiledSoftObjective& objective =
			compiledObjectives[attachments.size() + pi];
		const bool pointIsValid = avbdIsSoftPointValid(
			pin.point, particleStart, particleCount);
		const AvbdSoftObjectiveOwner expectedOwner =
			avbdGetPinObjectiveOwner(pin, pointIsValid);
		if(objective.owner != expectedOwner ||
			objective.runtimeStateIndex != pi ||
			!(objective.point == pin.point) ||
			objective.rigidBodyIdx != PX_MAX_U32)
			return false;
	}
	return true;
}

} // namespace Dy
} // namespace physx
