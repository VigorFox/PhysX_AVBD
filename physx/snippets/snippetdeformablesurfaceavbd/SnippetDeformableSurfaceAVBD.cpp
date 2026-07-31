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

#include "PxPhysicsAPI.h"
#include "cooking/PxCooking.h"
#include "extensions/PxDeformableVolumeExt.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetDeformableAVBDSkinning.h"
#include "SnippetDeformableSurfaceAVBD.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace physx;

#ifndef AVBD_SURFACE_SNIPPET_NAME
#define AVBD_SURFACE_SNIPPET_NAME "SnippetDeformableSurfaceAVBD"
#endif

#ifndef AVBD_SURFACE_DEFAULT_CASE
#define AVBD_SURFACE_DEFAULT_CASE "surface-lifecycle"
#endif

#ifndef AVBD_SURFACE_ENABLE_SKINNING
#define AVBD_SURFACE_ENABLE_SKINNING 0
#endif

namespace
{

struct SurfaceSkinningMetrics
{
	PxU32 initialized;
	PxU32 evaluatedFrames;
	PxU32 finiteFrames;
	PxU32 vertices;
	PxU32 triangles;
	PxReal maxDisplacement;

	SurfaceSkinningMetrics()
		: initialized(0), evaluatedFrames(0), finiteFrames(0),
		  vertices(0), triangles(0), maxDisplacement(0.0f)
	{
	}
};

static SurfaceSkinningMetrics gSurfaceSkinningMetrics;

struct Metrics
{
	PxU32 actorCreated;
	PxU32 shapeAttached;
	PxU32 hostBuffersInitialized;
	PxU32 actorAdded;
	PxU32 actorRemoved;
	PxU32 actorReadded;
	PxU32 fetchFailures;
	PxU32 nonFiniteSamples;
	PxU32 pinnedStable;
	PxU32 dynamicMoved;
	PxU32 boundsFinite;
	PxU32 groundAdded;
	PxU32 groundContactObserved;
	PxU32 groundPenetrationBounded;
	PxU32 groundSettled;
	PxU32 surfaceSlept;
	PxU32 initialSleepObserved;
	PxU32 velocityWakeIssued;
	PxU32 velocityWakeObserved;
	PxU32 movedAfterVelocityWake;
	PxU32 bufferMutationIssued;
	PxU32 bufferMutationApplied;
	PxU32 bufferPinHeld;
	PxU32 bufferInvMassRestored;
	PxU32 bufferRestoredMoved;
	PxU32 dynamicBoxAdded;
	PxU32 dynamicBoxInitiallySleeping;
	PxU32 dynamicBoxWoke;
	PxU32 dynamicBoxWakeFrame;
	PxU32 dynamicBoxFinalSleeping;
	PxU32 kinematicBoxAdded;
	PxU32 kinematicTargetIssued;
	PxU32 kinematicTargetReached;
	PxU32 kinematicSurfaceWoke;
	PxU32 kinematicSurfaceMoved;
	PxU32 kinematicContactObserved;
	PxU32 secondSurfaceCreated;
	PxU32 secondSurfaceAdded;
	PxU32 secondSurfaceInitiallySleeping;
	PxU32 secondSurfaceWoke;
	PxU32 secondSurfaceWakeFrame;
	PxU32 secondSurfaceMoved;
	PxU32 secondSurfaceFinalSleeping;
	PxU32 mixedVolumeCreated;
	PxU32 mixedVolumeAdded;
	PxU32 mixedVolumeInitiallySleeping;
	PxU32 mixedVolumeWoke;
	PxU32 mixedVolumeWakeFrame;
	PxU32 mixedVolumeMoved;
	PxU32 mixedVolumeFinalSleeping;
	PxU32 selfCollisionEnabled;
	PxU32 selfCollisionPreventedCrossing;
	PxU32 selfCollisionDisableIssued;
	PxU32 selfCollisionDisabledCrossed;
	PxU32 selfCollisionFilterApplied;
	PxU32 selfCollisionFilterExcludedPair;
	PxU32 materialFrictionLowApplied;
	PxU32 materialFrictionHighApplied;
	PxU32 materialFrictionResponseObserved;
	PxU32 attachmentCreated;
	PxU32 attachmentPinned;
	PxU32 attachmentReleased;
	PxU32 attachmentMovedAfterRelease;
	PxU32 rigidAttachmentActorAdded;
	PxU32 rigidAttachmentInitiallySleeping;
	PxU32 rigidAttachmentCreated;
	PxU32 rigidAttachmentRigidWoke;
	PxU32 rigidAttachmentRigidMoved;
	PxU32 rigidAttachmentRigidRotated;
	PxU32 rigidAttachmentHeldAcrossReadd;
	PxU32 rigidAttachmentReleased;
	PxU32 rigidAttachmentSeparatedAfterRelease;
	PxU32 articulationCreated;
	PxU32 articulationAdded;
	PxU32 articulationInitiallySleeping;
	PxU32 articulationWoke;
	PxU32 articulationJointSubspaceHeld;
	PxU32 articulationRootStable;
	PxU32 elementFilterCreated;
	PxU32 elementFilterHeldAcrossReadd;
	PxU32 elementFilterSuppressedContact;
	PxU32 elementFilterReleased;
	PxU32 elementFilterContactRestored;
	PxU32 partialFilterExactOwnership;
	PxU32 partialFilterUnfilteredContactHeld;
	PxU32 bendingMaterialPairCreated;
	PxU32 bendingZeroControlHeld;
	PxU32 bendingResponseObserved;
	PxU32 bendingMembraneIsolated;
	PxU32 flatteningFlagApplied;
	PxU32 flatteningControlHeld;
	PxU32 flatteningResponseObserved;
	PxU32 flatteningRetargetObserved;
	PxU32 flatteningMembraneIsolated;
	PxU32 motionMaxVelocityBounded;
	PxU32 motionSettlingApplied;
	PxU32 motionSettlingSlept;
	PxU32 motionControlStayedAwake;
	PxU32 depenetrationLimitApplied;
	PxU32 depenetrationFirstStepBounded;
	PxU32 depenetrationControlSeparated;
	PxU32 depenetrationGradualRecovery;
	PxU32 speculativeCcdFlagApplied;
	PxU32 speculativeCcdPreventedTunneling;
	PxU32 speculativeCcdNegativeControlTunneled;
	PxU32 movingSphereTargetIssued;
	PxU32 movingSphereCcdResponseObserved;
	PxU32 movingSphereNegativeControlHeld;
	PxU32 dynamicSphereSweepLaunched;
	PxU32 dynamicSphereSweepResponseObserved;
	PxU32 dynamicSphereSweepNegativeControlTunneled;
	PxU32 dynamicSphereSweepTwoSidedResponseObserved;
	PxU32 cleanupComplete;
	PxReal maxPinnedDrift;
	PxReal maxDynamicDisplacement;
	PxReal initialDynamicCentroidY;
	PxReal finalDynamicCentroidY;
	PxReal minY;
	PxReal finalMinY;
	PxReal maxSpeed;
	PxReal finalMaxSpeed;
	PxReal wakeCentroidY;
	PxReal maxWakeCentroidRise;
	PxReal bufferPinnedDrift;
	PxReal bufferRestoredDisplacement;
	PxReal dynamicBoxInitialY;
	PxReal dynamicBoxMinY;
	PxReal dynamicBoxFinalY;
	PxReal dynamicBoxMaxDrop;
	PxReal dynamicBoxMaxLinearSpeed;
	PxReal dynamicBoxFinalLinearSpeed;
	PxReal dynamicBoxMaxAngularSpeed;
	PxReal dynamicBoxFinalAngularSpeed;
	PxReal kinematicMaxPoseError;
	PxReal kinematicSurfaceDisplacement;
	PxReal kinematicFinalY;
	PxReal secondSurfaceInitialCentroidY;
	PxReal secondSurfaceFinalCentroidY;
	PxReal secondSurfaceMaxDisplacement;
	PxReal secondSurfaceMinY;
	PxReal secondSurfaceFinalMinY;
	PxReal secondSurfaceMaxSpeed;
	PxReal secondSurfaceFinalMaxSpeed;
	PxReal mixedVolumeInitialCentroidY;
	PxReal mixedVolumeFinalCentroidY;
	PxReal mixedVolumeMaxDisplacement;
	PxReal mixedVolumeMinY;
	PxReal mixedVolumeFinalMinY;
	PxReal mixedVolumeMaxSpeed;
	PxReal mixedVolumeFinalMaxSpeed;
	PxReal selfCollisionMinEnabledSeparation;
	PxReal selfCollisionMinDisabledSeparation;
	PxReal selfCollisionFilterMinSeparation;
	PxReal materialFrictionLowDisplacement;
	PxReal materialFrictionHighDisplacement;
	PxReal materialFrictionHighFinalSpeed;
	PxReal attachmentPinMaxDrift;
	PxReal attachmentReleasedMaxDisplacement;
	PxReal rigidAttachmentMaxDrift;
	PxReal rigidAttachmentMaxRigidDisplacement;
	PxReal rigidAttachmentMaxRigidSpeed;
	PxReal rigidAttachmentMaxAngularDisplacement;
	PxReal rigidAttachmentMaxAngularSpeed;
	PxReal rigidAttachmentReleasedSeparation;
	PxReal articulationRootMaxDisplacement;
	PxReal articulationChildMaxForbiddenDisplacement;
	PxReal articulationChildMaxAngularDisplacement;
	PxReal elementFilterMinY;
	PxReal elementFilterFinalMinY;
	PxReal partialFilterFilteredMinY;
	PxReal partialFilterUnfilteredMinY;
	PxReal bendingInitialPlaneError;
	PxReal bendingFinalPlaneError;
	PxReal bendingZeroControlDisplacement;
	PxReal bendingStiffDisplacement;
	PxReal bendingMaxEdgeStrain;
	PxReal flatteningInitialPlaneError;
	PxReal flatteningMinimumPlaneError;
	PxReal flatteningFinalPlaneError;
	PxReal flatteningControlDisplacement;
	PxReal flatteningTargetDisplacement;
	PxReal flatteningMaxEdgeStrain;
	PxReal motionMaxVelocityFirstStepDisplacement;
	PxReal motionMaxVelocityFirstStepSpeed;
	PxReal motionSettlingFinalSpeed;
	PxReal motionControlFinalSpeed;
	PxReal depenetrationLimitedFirstStepRise;
	PxReal depenetrationControlFirstStepRise;
	PxReal depenetrationLimitedFinalRise;
	PxReal depenetrationLimitedMaxSpeed;
	PxReal speculativeCcdPositiveMinY;
	PxReal speculativeCcdPositiveMinSeparation;
	PxReal speculativeCcdNegativeMaxY;
	PxReal movingSpherePositiveDisplacement;
	PxReal movingSphereNegativeDisplacement;
	PxReal movingSpherePositiveMinSeparation;
	PxReal dynamicSphereSweepPositiveSoftDisplacement;
	PxReal dynamicSphereSweepNegativeSoftDisplacement;
	PxReal dynamicSphereSweepPositiveRigidDrop;
	PxReal dynamicSphereSweepNegativeRigidDrop;
	PxReal dynamicSphereSweepPositiveMinSeparation;

	Metrics()
		: actorCreated(0), shapeAttached(0), hostBuffersInitialized(0),
		  actorAdded(0), actorRemoved(0), actorReadded(0),
		  fetchFailures(0), nonFiniteSamples(0), pinnedStable(0),
		  dynamicMoved(0), boundsFinite(0), groundAdded(0),
		  groundContactObserved(0), groundPenetrationBounded(0),
		  groundSettled(0), surfaceSlept(0),
		  initialSleepObserved(0), velocityWakeIssued(0),
		  velocityWakeObserved(0), movedAfterVelocityWake(0),
		  bufferMutationIssued(0), bufferMutationApplied(0),
		  bufferPinHeld(0), bufferInvMassRestored(0),
		  bufferRestoredMoved(0),
		  dynamicBoxAdded(0), dynamicBoxInitiallySleeping(0),
		  dynamicBoxWoke(0), dynamicBoxWakeFrame(PX_MAX_U32),
		  dynamicBoxFinalSleeping(0),
		  kinematicBoxAdded(0), kinematicTargetIssued(0),
		  kinematicTargetReached(0), kinematicSurfaceWoke(0),
		  kinematicSurfaceMoved(0), kinematicContactObserved(0),
		  secondSurfaceCreated(0), secondSurfaceAdded(0),
		  secondSurfaceInitiallySleeping(0), secondSurfaceWoke(0),
		  secondSurfaceWakeFrame(PX_MAX_U32),
		  secondSurfaceMoved(0), secondSurfaceFinalSleeping(0),
		  mixedVolumeCreated(0), mixedVolumeAdded(0),
		  mixedVolumeInitiallySleeping(0), mixedVolumeWoke(0),
		  mixedVolumeWakeFrame(PX_MAX_U32), mixedVolumeMoved(0),
		  mixedVolumeFinalSleeping(0),
		  selfCollisionEnabled(0),
		  selfCollisionPreventedCrossing(0),
		  selfCollisionDisableIssued(0),
		  selfCollisionDisabledCrossed(0),
		  selfCollisionFilterApplied(0),
		  selfCollisionFilterExcludedPair(0),
		  materialFrictionLowApplied(0),
		  materialFrictionHighApplied(0),
		  materialFrictionResponseObserved(0),
		  attachmentCreated(0), attachmentPinned(0),
		  attachmentReleased(0),
		  attachmentMovedAfterRelease(0),
		  rigidAttachmentActorAdded(0),
		  rigidAttachmentInitiallySleeping(0),
		  rigidAttachmentCreated(0),
		  rigidAttachmentRigidWoke(0),
		  rigidAttachmentRigidMoved(0),
		  rigidAttachmentRigidRotated(0),
		  rigidAttachmentHeldAcrossReadd(0),
		  rigidAttachmentReleased(0),
		  rigidAttachmentSeparatedAfterRelease(0),
		  articulationCreated(0), articulationAdded(0),
		  articulationInitiallySleeping(0), articulationWoke(0),
		  articulationJointSubspaceHeld(0),
		  articulationRootStable(0),
		  elementFilterCreated(0),
		  elementFilterHeldAcrossReadd(0),
		  elementFilterSuppressedContact(0),
		  elementFilterReleased(0),
		  elementFilterContactRestored(0),
		  partialFilterExactOwnership(0),
		  partialFilterUnfilteredContactHeld(0),
		  bendingMaterialPairCreated(0),
		  bendingZeroControlHeld(0),
		  bendingResponseObserved(0),
		  bendingMembraneIsolated(0),
		  flatteningFlagApplied(0),
		  flatteningControlHeld(0),
		  flatteningResponseObserved(0),
		  flatteningRetargetObserved(0),
		  flatteningMembraneIsolated(0),
		  motionMaxVelocityBounded(0),
		  motionSettlingApplied(0),
		  motionSettlingSlept(0),
		  motionControlStayedAwake(0),
		  depenetrationLimitApplied(0),
		  depenetrationFirstStepBounded(0),
		  depenetrationControlSeparated(0),
		  depenetrationGradualRecovery(0),
		  speculativeCcdFlagApplied(0),
		  speculativeCcdPreventedTunneling(0),
		  speculativeCcdNegativeControlTunneled(0),
		  movingSphereTargetIssued(0),
		  movingSphereCcdResponseObserved(0),
		  movingSphereNegativeControlHeld(0),
		  dynamicSphereSweepLaunched(0),
		  dynamicSphereSweepResponseObserved(0),
		  dynamicSphereSweepNegativeControlTunneled(0),
		  dynamicSphereSweepTwoSidedResponseObserved(0),
		  cleanupComplete(0),
		  maxPinnedDrift(0.0f), maxDynamicDisplacement(0.0f),
		  initialDynamicCentroidY(0.0f), finalDynamicCentroidY(0.0f),
		  minY(PX_MAX_F32), finalMinY(PX_MAX_F32),
		  maxSpeed(0.0f), finalMaxSpeed(0.0f),
		  wakeCentroidY(0.0f), maxWakeCentroidRise(0.0f),
		  bufferPinnedDrift(0.0f),
		  bufferRestoredDisplacement(0.0f),
		  dynamicBoxInitialY(0.0f), dynamicBoxMinY(PX_MAX_F32),
		  dynamicBoxFinalY(PX_MAX_F32), dynamicBoxMaxDrop(0.0f),
		  dynamicBoxMaxLinearSpeed(0.0f),
		  dynamicBoxFinalLinearSpeed(0.0f),
		  dynamicBoxMaxAngularSpeed(0.0f),
		  dynamicBoxFinalAngularSpeed(0.0f),
		  kinematicMaxPoseError(0.0f),
		  kinematicSurfaceDisplacement(0.0f),
		  kinematicFinalY(PX_MAX_F32),
		  secondSurfaceInitialCentroidY(0.0f),
		  secondSurfaceFinalCentroidY(0.0f),
		  secondSurfaceMaxDisplacement(0.0f),
		  secondSurfaceMinY(PX_MAX_F32),
		  secondSurfaceFinalMinY(PX_MAX_F32),
		  secondSurfaceMaxSpeed(0.0f),
		  secondSurfaceFinalMaxSpeed(0.0f),
		  mixedVolumeInitialCentroidY(0.0f),
		  mixedVolumeFinalCentroidY(0.0f),
		  mixedVolumeMaxDisplacement(0.0f),
		  mixedVolumeMinY(PX_MAX_F32),
		  mixedVolumeFinalMinY(PX_MAX_F32),
		  mixedVolumeMaxSpeed(0.0f),
		  mixedVolumeFinalMaxSpeed(0.0f),
		  selfCollisionMinEnabledSeparation(PX_MAX_F32),
		  selfCollisionMinDisabledSeparation(PX_MAX_F32),
		  selfCollisionFilterMinSeparation(PX_MAX_F32),
		  materialFrictionLowDisplacement(0.0f),
		  materialFrictionHighDisplacement(0.0f),
		  materialFrictionHighFinalSpeed(0.0f),
		  attachmentPinMaxDrift(0.0f),
		  attachmentReleasedMaxDisplacement(0.0f),
		  rigidAttachmentMaxDrift(0.0f),
		  rigidAttachmentMaxRigidDisplacement(0.0f),
		  rigidAttachmentMaxRigidSpeed(0.0f),
		  rigidAttachmentMaxAngularDisplacement(0.0f),
		  rigidAttachmentMaxAngularSpeed(0.0f),
		  rigidAttachmentReleasedSeparation(0.0f),
		  articulationRootMaxDisplacement(0.0f),
		  articulationChildMaxForbiddenDisplacement(0.0f),
		  articulationChildMaxAngularDisplacement(0.0f),
		  elementFilterMinY(PX_MAX_F32),
		  elementFilterFinalMinY(PX_MAX_F32),
		  partialFilterFilteredMinY(PX_MAX_F32),
		  partialFilterUnfilteredMinY(PX_MAX_F32),
		  bendingInitialPlaneError(0.0f),
		  bendingFinalPlaneError(0.0f),
		  bendingZeroControlDisplacement(0.0f),
		  bendingStiffDisplacement(0.0f),
		  bendingMaxEdgeStrain(0.0f),
		  flatteningInitialPlaneError(0.0f),
		  flatteningMinimumPlaneError(PX_MAX_F32),
		  flatteningFinalPlaneError(0.0f),
		  flatteningControlDisplacement(0.0f),
		  flatteningTargetDisplacement(0.0f),
		  flatteningMaxEdgeStrain(0.0f),
		  motionMaxVelocityFirstStepDisplacement(0.0f),
		  motionMaxVelocityFirstStepSpeed(0.0f),
		  motionSettlingFinalSpeed(0.0f),
		  motionControlFinalSpeed(0.0f),
		  depenetrationLimitedFirstStepRise(0.0f),
		  depenetrationControlFirstStepRise(0.0f),
		  depenetrationLimitedFinalRise(0.0f),
		  depenetrationLimitedMaxSpeed(0.0f),
		  speculativeCcdPositiveMinY(PX_MAX_F32),
		  speculativeCcdPositiveMinSeparation(PX_MAX_F32),
		  speculativeCcdNegativeMaxY(-PX_MAX_F32),
		  movingSpherePositiveDisplacement(0.0f),
		  movingSphereNegativeDisplacement(0.0f),
		  movingSpherePositiveMinSeparation(PX_MAX_F32),
		  dynamicSphereSweepPositiveSoftDisplacement(0.0f),
		  dynamicSphereSweepNegativeSoftDisplacement(0.0f),
		  dynamicSphereSweepPositiveRigidDrop(0.0f),
		  dynamicSphereSweepNegativeRigidDrop(0.0f),
		  dynamicSphereSweepPositiveMinSeparation(PX_MAX_F32)
	{
	}
};

static const PxU32 sGridWidth = 6;
static const PxU32 sGridHeight = 6;

static PxU32 vertexIndex(PxU32 x, PxU32 y)
{
	return y * sGridWidth + x;
}

static bool buildSurfaceMesh(
	PxPhysics& physics,
	const PxTolerancesScale& scale,
	std::vector<PxVec3>& vertices,
	std::vector<PxU32>& triangles,
	PxTriangleMesh*& triangleMesh,
	bool groundCase,
	bool selfCollisionCase,
	bool partialElementFilterCase)
{
	vertices.clear();
	triangles.clear();
	if(partialElementFilterCase)
	{
		// Two disconnected elements make the public triangle ownership
		// unambiguous: triangle 0 is filtered from the plane while
		// triangle 1 must retain contact.
		vertices.push_back(PxVec3(-2.0f, 2.0f, -0.5f));
		vertices.push_back(PxVec3(-1.5f, 2.0f, 0.5f));
		vertices.push_back(PxVec3(-1.0f, 2.0f, -0.5f));
		vertices.push_back(PxVec3(1.0f, 2.0f, -0.5f));
		vertices.push_back(PxVec3(1.5f, 2.0f, 0.5f));
		vertices.push_back(PxVec3(2.0f, 2.0f, -0.5f));
		for(PxU32 i = 0; i < 6; ++i)
			triangles.push_back(i);
	}
	else if(selfCollisionCase)
	{
		// Two disconnected, upward-wound triangles in one actor. The
		// dynamic upper triangle can only be kept above the pinned lower
		// triangle by same-body self collision.
		vertices.push_back(PxVec3(-1.0f, 0.0f, -1.0f));
		vertices.push_back(PxVec3(0.0f, 0.0f, 1.0f));
		vertices.push_back(PxVec3(1.0f, 0.0f, -1.0f));
		vertices.push_back(PxVec3(-1.0f, 0.04f, -1.0f));
		vertices.push_back(PxVec3(0.0f, 0.04f, 1.0f));
		vertices.push_back(PxVec3(1.0f, 0.04f, -1.0f));
		for(PxU32 i = 0; i < 6; ++i)
			triangles.push_back(i);
	}
	else
	{
	vertices.reserve(sGridWidth * sGridHeight);
	triangles.reserve(
		6 * (sGridWidth - 1) * (sGridHeight - 1));

	const PxReal spacing = 0.35f;
	for(PxU32 y = 0; y < sGridHeight; ++y)
	{
		for(PxU32 x = 0; x < sGridWidth; ++x)
		{
			const PxReal localX =
				(PxReal(x) -
					0.5f * PxReal(sGridWidth - 1)) * spacing;
			const PxReal localY =
				(PxReal(y) -
					0.5f * PxReal(sGridHeight - 1)) * spacing;
			vertices.push_back(groundCase
				? PxVec3(localX, 2.0f, localY)
				: PxVec3(localX, 4.5f - PxReal(y) * spacing, 0.0f));
		}
	}

	for(PxU32 y = 0; y + 1 < sGridHeight; ++y)
	{
		for(PxU32 x = 0; x + 1 < sGridWidth; ++x)
		{
			const PxU32 v00 = vertexIndex(x, y);
			const PxU32 v10 = vertexIndex(x + 1, y);
			const PxU32 v01 = vertexIndex(x, y + 1);
			const PxU32 v11 = vertexIndex(x + 1, y + 1);
			triangles.push_back(v00);
			triangles.push_back(v01);
			triangles.push_back(v10);
			triangles.push_back(v01);
			triangles.push_back(v11);
			triangles.push_back(v10);
		}
	}
	}

	PxTriangleMeshDesc meshDesc;
	meshDesc.points.count = PxU32(vertices.size());
	meshDesc.points.stride = sizeof(PxVec3);
	meshDesc.points.data = vertices.data();
	meshDesc.triangles.count = PxU32(triangles.size() / 3);
	meshDesc.triangles.stride = 3 * sizeof(PxU32);
	meshDesc.triangles.data = triangles.data();

	PxCookingParams cookingParams(scale);
	cookingParams.buildGPUData = false;
	cookingParams.buildTriangleAdjacencies = true;
	triangleMesh = PxCreateTriangleMesh(
		cookingParams, meshDesc,
		physics.getPhysicsInsertionCallback());
	return triangleMesh != NULL;
}

struct BendingSurfaceFixture
{
	PxTriangleMesh* triangleMesh;
	PxDeformableSurfaceMaterial* material;
	PxDeformableSurface* surface;
	PxVec4* positions;
	PxVec4* velocities;
	PxVec4* restPositions;
	PxVec3 initialPositions[4];
	PxVec3 restVertices[4];

	BendingSurfaceFixture()
		: triangleMesh(NULL), material(NULL), surface(NULL),
		  positions(NULL), velocities(NULL), restPositions(NULL)
	{
	}
};

static bool createBendingSurfaceFixture(
	PxPhysics& physics,
	PxScene& scene,
	const PxTolerancesScale& scale,
	const PxVec3& offset,
	PxReal bendingStiffness,
	bool curvedRestShape,
	bool flatteningEnabled,
	BendingSurfaceFixture& fixture)
{
	fixture.restVertices[0] = offset + PxVec3(0.0f, 0.0f, 0.0f);
	fixture.restVertices[1] = offset + PxVec3(1.0f, 0.0f, 0.0f);
	fixture.restVertices[2] = offset + PxVec3(0.0f, 0.0f, 1.0f);
	fixture.restVertices[3] = offset + PxVec3(1.0f, 0.0f, 1.0f);
	const PxVec3 foldedVertex =
		offset + PxVec3(0.5f, PxSqrt(0.5f), 0.5f);
	if(curvedRestShape)
		fixture.restVertices[3] = foldedVertex;
	for(PxU32 i = 0; i < 4; ++i)
		fixture.initialPositions[i] = fixture.restVertices[i];

	// Rotate only the free vertex by 90 degrees around the shared
	// diagonal. Both triangle edge lengths remain at their rest values,
	// so any restoring response is attributable to dihedral bending.
	if(!curvedRestShape)
		fixture.initialPositions[3] = foldedVertex;

	const PxU32 triangles[] =
	{
		0, 2, 1,
		2, 3, 1
	};
	PxTriangleMeshDesc meshDesc;
	meshDesc.points.count = 4;
	meshDesc.points.stride = sizeof(PxVec3);
	meshDesc.points.data = fixture.restVertices;
	meshDesc.triangles.count = 2;
	meshDesc.triangles.stride = 3 * sizeof(PxU32);
	meshDesc.triangles.data = triangles;

	PxCookingParams cookingParams(scale);
	cookingParams.buildGPUData = false;
	cookingParams.buildTriangleAdjacencies = true;
	fixture.triangleMesh = PxCreateTriangleMesh(
		cookingParams, meshDesc,
		physics.getPhysicsInsertionCallback());
	if(!fixture.triangleMesh)
		return false;

	fixture.material = physics.createDeformableSurfaceMaterial(
		2.0e4f, 0.3f, 0.0f, 0.02f, bendingStiffness);
	if(!fixture.material)
		return false;
	fixture.surface = physics.createDeformableSurface(
		PxDeformableSurfaceBackend::eCPU_AVBD);
	if(!fixture.surface)
		return false;

	PxDeformableSurfaceMaterial* materials[] = {fixture.material};
	const PxShapeFlags shapeFlags =
		PxShapeFlag::eSIMULATION_SHAPE |
		PxShapeFlag::eSCENE_QUERY_SHAPE;
	PxShape* shape = physics.createShape(
		PxTriangleMeshGeometry(fixture.triangleMesh),
		materials, 1, true, shapeFlags);
	if(!shape)
		return false;
	const bool attached = fixture.surface->attachShape(*shape);
	shape->release();
	if(!attached)
		return false;

	fixture.positions =
		fixture.surface->getPositionInvMassBufferH();
	fixture.velocities =
		fixture.surface->getVelocityBufferH();
	fixture.restPositions =
		fixture.surface->getRestPositionBufferH();
	if(!fixture.positions || !fixture.velocities ||
		!fixture.restPositions)
		return false;
	for(PxU32 i = 0; i < 4; ++i)
	{
		fixture.positions[i] = PxVec4(
			fixture.initialPositions[i], i == 3 ? 1.0f : 0.0f);
		fixture.velocities[i] = PxVec4(0.0f);
		fixture.restPositions[i] =
			PxVec4(fixture.restVertices[i], 0.0f);
	}
	fixture.surface->setActorFlag(
		PxActorFlag::eDISABLE_GRAVITY, true);
	fixture.surface->setDeformableBodyFlag(
		PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
	fixture.surface->setDeformableSurfaceFlag(
		PxDeformableSurfaceFlag::eENABLE_FLATTENING,
		flatteningEnabled);
	fixture.surface->setLinearDamping(2.0f);
	fixture.surface->setSolverIterationCounts(16);
	fixture.surface->setSleepThreshold(1.0e-8f);
	fixture.surface->markDirty(
		PxDeformableSurfaceDataFlag::eALL);
	return scene.addActor(*fixture.surface) &&
		fixture.surface->getScene() == &scene;
}

static void releaseBendingSurfaceFixture(
	BendingSurfaceFixture& fixture)
{
	if(fixture.surface)
	{
		if(fixture.surface->getScene())
			fixture.surface->getScene()->removeActor(*fixture.surface);
		fixture.surface->release();
		fixture.surface = NULL;
	}
	PX_RELEASE(fixture.material);
	PX_RELEASE(fixture.triangleMesh);
}

struct MotionSurfaceFixture
{
	PxTriangleMesh* triangleMesh;
	PxDeformableSurfaceMaterial* material;
	PxDeformableSurface* surface;
	PxVec4* positions;
	PxVec4* velocities;
	PxVec4* restPositions;
	PxVec3 initialPositions[4];

	MotionSurfaceFixture()
		: triangleMesh(NULL), material(NULL), surface(NULL),
		  positions(NULL), velocities(NULL), restPositions(NULL)
	{
	}
};

static bool createMotionSurfaceFixture(
	PxPhysics& physics,
	PxScene& scene,
	const PxTolerancesScale& scale,
	const PxVec3& offset,
	const PxVec3& initialVelocity,
	PxReal maxLinearVelocity,
	PxReal settlingThreshold,
	PxReal settlingDamping,
	PxReal sleepThreshold,
	PxReal wakeCounter,
	MotionSurfaceFixture& fixture)
{
	const PxVec3 vertices[] =
	{
		offset + PxVec3(0.0f, 0.0f, 0.0f),
		offset + PxVec3(1.0f, 0.0f, 0.0f),
		offset + PxVec3(0.0f, 0.0f, 1.0f),
		offset + PxVec3(1.0f, 0.0f, 1.0f)
	};
	const PxU32 triangles[] =
	{
		0, 2, 1,
		2, 3, 1
	};
	PxTriangleMeshDesc meshDesc;
	meshDesc.points.count = 4;
	meshDesc.points.stride = sizeof(PxVec3);
	meshDesc.points.data = vertices;
	meshDesc.triangles.count = 2;
	meshDesc.triangles.stride = 3 * sizeof(PxU32);
	meshDesc.triangles.data = triangles;

	PxCookingParams cookingParams(scale);
	cookingParams.buildGPUData = false;
	cookingParams.buildTriangleAdjacencies = true;
	fixture.triangleMesh = PxCreateTriangleMesh(
		cookingParams, meshDesc,
		physics.getPhysicsInsertionCallback());
	if(!fixture.triangleMesh)
		return false;

	fixture.material = physics.createDeformableSurfaceMaterial(
		2.0e4f, 0.3f, 0.0f, 0.02f, 0.0f);
	fixture.surface = physics.createDeformableSurface(
		PxDeformableSurfaceBackend::eCPU_AVBD);
	if(!fixture.material || !fixture.surface)
		return false;

	PxDeformableSurfaceMaterial* materials[] = {fixture.material};
	const PxShapeFlags shapeFlags =
		PxShapeFlag::eSIMULATION_SHAPE |
		PxShapeFlag::eSCENE_QUERY_SHAPE;
	PxShape* shape = physics.createShape(
		PxTriangleMeshGeometry(fixture.triangleMesh),
		materials, 1, true, shapeFlags);
	if(!shape)
		return false;
	const bool attached = fixture.surface->attachShape(*shape);
	shape->release();
	if(!attached)
		return false;

	fixture.positions =
		fixture.surface->getPositionInvMassBufferH();
	fixture.velocities =
		fixture.surface->getVelocityBufferH();
	fixture.restPositions =
		fixture.surface->getRestPositionBufferH();
	if(!fixture.positions || !fixture.velocities ||
		!fixture.restPositions)
		return false;
	for(PxU32 i = 0; i < 4; ++i)
	{
		fixture.initialPositions[i] = vertices[i];
		fixture.positions[i] = PxVec4(vertices[i], 1.0f);
		fixture.velocities[i] = PxVec4(initialVelocity, 0.0f);
		fixture.restPositions[i] = PxVec4(vertices[i], 0.0f);
	}
	fixture.surface->setActorFlag(
		PxActorFlag::eDISABLE_GRAVITY, true);
	fixture.surface->setDeformableBodyFlag(
		PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
	fixture.surface->setLinearDamping(0.0f);
	fixture.surface->setMaxLinearVelocity(maxLinearVelocity);
	fixture.surface->setSettlingThreshold(settlingThreshold);
	fixture.surface->setSettlingDamping(settlingDamping);
	fixture.surface->setSleepThreshold(sleepThreshold);
	fixture.surface->setWakeCounter(wakeCounter);
	fixture.surface->setSolverIterationCounts(8);
	fixture.surface->markDirty(
		PxDeformableSurfaceDataFlag::eALL);
	return scene.addActor(*fixture.surface) &&
		fixture.surface->getScene() == &scene;
}

static void releaseMotionSurfaceFixture(
	MotionSurfaceFixture& fixture)
{
	if(fixture.surface)
	{
		if(fixture.surface->getScene())
			fixture.surface->getScene()->removeActor(*fixture.surface);
		fixture.surface->release();
		fixture.surface = NULL;
	}
	PX_RELEASE(fixture.material);
	PX_RELEASE(fixture.triangleMesh);
}

static PxConvexMesh* createAvbdTestConvexMesh(
	PxPhysics& physics, const PxTolerancesScale& scale,
	const PxVec3* vertices, PxU32 vertexCount)
{
	if(!vertices || vertexCount < 4)
		return NULL;
	PxConvexMeshDesc convexDesc;
	convexDesc.points.count = vertexCount;
	convexDesc.points.stride = sizeof(PxVec3);
	convexDesc.points.data = vertices;
	convexDesc.flags = PxConvexFlag::eCOMPUTE_CONVEX;
	PxCookingParams cookingParams(scale);
	cookingParams.buildGPUData = false;
	return PxCreateConvexMesh(
		cookingParams, convexDesc,
		physics.getPhysicsInsertionCallback());
}

static PxTriangleMesh* createAvbdRigidTriangleMesh(
	PxPhysics& physics, const PxTolerancesScale& scale,
	bool reverseFeatureCase)
{
	const PxVec3 reverseVertices[] =
	{
		PxVec3(0.0f, 0.3f, 0.0f),
		PxVec3(-0.3f, -0.3f, -0.3f),
		PxVec3(0.3f, -0.3f, -0.3f),
		PxVec3(0.0f, -0.3f, 0.35f)
	};
	const PxU32 reverseTriangles[] =
	{
		0, 2, 1,
		0, 3, 2,
		0, 1, 3,
		1, 2, 3
	};
	const PxVec3 kinematicVertices[] =
	{
		PxVec3(-2.0f, 0.0f, -2.0f),
		PxVec3(-2.0f, 0.0f, 2.0f),
		PxVec3(2.0f, 0.0f, 2.0f),
		PxVec3(2.0f, 0.0f, -2.0f)
	};
	const PxU32 kinematicTriangles[] =
	{
		0, 1, 2,
		0, 2, 3
	};
	PxTriangleMeshDesc meshDesc;
	meshDesc.points.count = 4;
	meshDesc.points.stride = sizeof(PxVec3);
	meshDesc.points.data = reverseFeatureCase
		? reverseVertices : kinematicVertices;
	meshDesc.triangles.count =
		reverseFeatureCase ? 4u : 2u;
	meshDesc.triangles.stride = 3 * sizeof(PxU32);
	meshDesc.triangles.data = reverseFeatureCase
		? reverseTriangles : kinematicTriangles;
	PxCookingParams cookingParams(scale);
	cookingParams.buildGPUData = false;
	cookingParams.buildTriangleAdjacencies = true;
	return PxCreateTriangleMesh(
		cookingParams, meshDesc,
		physics.getPhysicsInsertionCallback());
}

static PxTriangleMesh* createAvbdRotationalRigidTriangleMesh(
	PxPhysics& physics, const PxTolerancesScale& scale)
{
	// A narrow one-sided blade. Its +Y face and active boundary features
	// sweep through the soft target only between the two endpoint poses.
	const PxVec3 vertices[] =
	{
		PxVec3(-1.0f, 0.0f, -0.1f),
		PxVec3(-1.0f, 0.0f, 0.1f),
		PxVec3(1.0f, 0.0f, 0.1f),
		PxVec3(1.0f, 0.0f, -0.1f)
	};
	const PxU32 triangles[] =
	{
		0, 1, 2,
		0, 2, 3
	};
	PxTriangleMeshDesc meshDesc;
	meshDesc.points.count = 4;
	meshDesc.points.stride = sizeof(PxVec3);
	meshDesc.points.data = vertices;
	meshDesc.triangles.count = 2;
	meshDesc.triangles.stride = 3 * sizeof(PxU32);
	meshDesc.triangles.data = triangles;
	PxCookingParams cookingParams(scale);
	cookingParams.buildGPUData = false;
	cookingParams.buildTriangleAdjacencies = true;
	return PxCreateTriangleMesh(
		cookingParams, meshDesc,
		physics.getPhysicsInsertionCallback());
}

static PxHeightField* createAvbdRigidHeightField(
	PxPhysics& physics, bool reverseFeatureCase)
{
	PxHeightFieldSample samples[9];
	for(PxU32 sampleIndex = 0;
		sampleIndex < 9; ++sampleIndex)
	{
		samples[sampleIndex].height =
			reverseFeatureCase ? PxI16(-3) : PxI16(0);
		samples[sampleIndex].materialIndex0 =
			PxBitAndByte(0);
		samples[sampleIndex].materialIndex1 =
			PxBitAndByte(0);
		samples[sampleIndex].setTessFlag();
	}
	if(reverseFeatureCase)
		samples[4].height = PxI16(3);
	PxHeightFieldDesc heightFieldDesc;
	heightFieldDesc.nbRows = 3;
	heightFieldDesc.nbColumns = 3;
	heightFieldDesc.samples.data = samples;
	heightFieldDesc.samples.stride =
		sizeof(PxHeightFieldSample);
	return PxCreateHeightField(
		heightFieldDesc,
		physics.getPhysicsInsertionCallback());
}

static PxVec3 closestPointOnTriangleForConvexGate(
	const PxVec3& point, const PxVec3& a,
	const PxVec3& b, const PxVec3& c)
{
	const PxVec3 ab = b - a;
	const PxVec3 ac = c - a;
	const PxVec3 ap = point - a;
	const PxReal d1 = ab.dot(ap);
	const PxReal d2 = ac.dot(ap);
	if(d1 <= 0.0f && d2 <= 0.0f)
		return a;
	const PxVec3 bp = point - b;
	const PxReal d3 = ab.dot(bp);
	const PxReal d4 = ac.dot(bp);
	if(d3 >= 0.0f && d4 <= d3)
		return b;
	const PxReal vc = d1 * d4 - d3 * d2;
	if(vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
	{
		const PxReal denominator = d1 - d3;
		return denominator > 1.0e-20f
			? a + ab * (d1 / denominator) : a;
	}
	const PxVec3 cp = point - c;
	const PxReal d5 = ab.dot(cp);
	const PxReal d6 = ac.dot(cp);
	if(d6 >= 0.0f && d5 <= d6)
		return c;
	const PxReal vb = d5 * d2 - d1 * d6;
	if(vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
	{
		const PxReal denominator = d2 - d6;
		return denominator > 1.0e-20f
			? a + ac * (d2 / denominator) : a;
	}
	const PxReal va = d3 * d6 - d5 * d4;
	if(va <= 0.0f && d4 - d3 >= 0.0f &&
		d5 - d6 >= 0.0f)
	{
		const PxReal edgeTerm = d4 - d3;
		const PxReal denominator =
			edgeTerm + d5 - d6;
		return denominator > 1.0e-20f
			? b + (c - b) * (edgeTerm / denominator) : b;
	}
	const PxReal denominator = va + vb + vc;
	if(PxAbs(denominator) <= 1.0e-20f)
		return a;
	const PxReal inverseDenominator = 1.0f / denominator;
	return a + ab * (vb * inverseDenominator) +
		ac * (vc * inverseDenominator);
}

static PxVec3 getMotionSurfaceCentroid(
	const MotionSurfaceFixture& fixture)
{
	PxVec3 centroid(0.0f);
	for(PxU32 i = 0; i < 4; ++i)
		centroid += fixture.positions[i].getXYZ();
	return centroid * 0.25f;
}

static PxReal getMotionSurfaceMaxSpeed(
	const MotionSurfaceFixture& fixture)
{
	PxReal maxSpeed = 0.0f;
	for(PxU32 i = 0; i < 4; ++i)
		maxSpeed = PxMax(
			maxSpeed, fixture.velocities[i].getXYZ().magnitude());
	return maxSpeed;
}

static bool buildVolumeMesh(
	PxPhysics& physics,
	const PxTolerancesScale& scale,
	PxDeformableVolumeMesh*& volumeMesh)
{
	const PxVec3 vertices[] =
	{
		PxVec3(0.0f, 0.0f, 0.0f),
		PxVec3(1.0f, 0.0f, 0.0f),
		PxVec3(0.0f, 1.0f, 0.0f),
		PxVec3(0.0f, 0.0f, 1.0f)
	};
	const PxU32 triangles[] =
	{
		0, 2, 1,
		0, 1, 3,
		0, 3, 2,
		1, 2, 3
	};
	PxSimpleTriangleMesh surfaceMesh;
	surfaceMesh.points.count = 4;
	surfaceMesh.points.data = vertices;
	surfaceMesh.points.stride = sizeof(PxVec3);
	surfaceMesh.triangles.count = 4;
	surfaceMesh.triangles.data = triangles;
	surfaceMesh.triangles.stride = 3 * sizeof(PxU32);

	PxCookingParams cookingParams(scale);
	// The mixed Surface/Volume gate deliberately uses CPU-only cooking.
	cookingParams.buildGPUData = false;
	cookingParams.meshWeldTolerance = 0.001f;
	cookingParams.meshPreprocessParams =
		PxMeshPreprocessingFlag::eWELD_VERTICES;
	volumeMesh =
		PxDeformableVolumeExt::createDeformableVolumeMeshNoVoxels(
			cookingParams, surfaceMesh,
			physics.getPhysicsInsertionCallback(), 1.5f, true);
	return volumeMesh != NULL;
}

static bool buildPairFilterVolumeMesh(
	PxPhysics& physics,
	const PxTolerancesScale& scale,
	PxDeformableVolumeMesh*& volumeMesh)
{
	const PxVec3 collisionVertices[] =
	{
		PxVec3(-2.4f, 0.0f, -0.3f),
		PxVec3(-2.0f, 0.0f, 0.5f),
		PxVec3(-1.6f, 0.0f, -0.3f),
		PxVec3(-2.0f, 0.8f, 0.0f),
		PxVec3(1.6f, 0.0f, -0.3f),
		PxVec3(2.0f, 0.0f, 0.5f),
		PxVec3(2.4f, 0.0f, -0.3f),
		PxVec3(2.0f, 0.8f, 0.0f)
	};
	const PxU32 collisionTetrahedra[] =
	{
		0, 1, 2, 3,
		4, 5, 6, 7
	};
	PxTetrahedronMeshDesc collisionMeshDesc;
	collisionMeshDesc.points.count = 8;
	collisionMeshDesc.points.data = collisionVertices;
	collisionMeshDesc.points.stride = sizeof(PxVec3);
	collisionMeshDesc.tetrahedrons.count = 2;
	collisionMeshDesc.tetrahedrons.data = collisionTetrahedra;
	collisionMeshDesc.tetrahedrons.stride = 4 * sizeof(PxU32);

	PxVec3 simulationVertices[10];
	for(PxU32 i = 0; i < 8; ++i)
		simulationVertices[i] = collisionVertices[i];
	simulationVertices[8] =
		(collisionVertices[0] + collisionVertices[1] +
		 collisionVertices[2] + collisionVertices[3]) * 0.25f;
	simulationVertices[9] =
		(collisionVertices[4] + collisionVertices[5] +
		 collisionVertices[6] + collisionVertices[7]) * 0.25f;
	const PxU32 simulationTetrahedra[] =
	{
		0, 1, 2, 8,
		0, 1, 8, 3,
		0, 8, 2, 3,
		8, 1, 2, 3,
		4, 5, 6, 9,
		4, 5, 9, 7,
		4, 9, 6, 7,
		9, 5, 6, 7
	};
	PxTetrahedronMeshDesc simulationMeshDesc;
	simulationMeshDesc.points.count = 10;
	simulationMeshDesc.points.data = simulationVertices;
	simulationMeshDesc.points.stride = sizeof(PxVec3);
	simulationMeshDesc.tetrahedrons.count = 8;
	simulationMeshDesc.tetrahedrons.data = simulationTetrahedra;
	simulationMeshDesc.tetrahedrons.stride = 4 * sizeof(PxU32);

	PxCookingParams cookingParams(scale);
	cookingParams.buildGPUData = false;
	PxDeformableVolumeSimulationDataDesc simulationDataDesc;
	volumeMesh = PxCreateDeformableVolumeMesh(
		cookingParams, simulationMeshDesc,
		collisionMeshDesc, simulationDataDesc,
		physics.getPhysicsInsertionCallback());
	return volumeMesh != NULL;
}

struct PairFilterVolumeFixture
{
	PxDeformableVolume* volume;
	PxVec4* positions;
	PxVec4* velocities;
	std::vector<PxVec4> initialPositions;

	PairFilterVolumeFixture()
		: volume(NULL), positions(NULL), velocities(NULL)
	{
	}
};

static bool createPairFilterVolumeFixture(
	PxPhysics& physics,
	PxScene& scene,
	PxDeformableVolumeMesh& volumeMesh,
	PxDeformableVolumeMaterial& material,
	bool fixedTarget,
	PairFilterVolumeFixture& fixture)
{
	fixture.volume = physics.createDeformableVolume(
		PxDeformableVolumeBackend::eCPU_AVBD);
	if(!fixture.volume)
		return false;
	PxDeformableVolumeMaterial* materials[] = {&material};
	const PxShapeFlags shapeFlags =
		PxShapeFlag::eSIMULATION_SHAPE |
		PxShapeFlag::eSCENE_QUERY_SHAPE;
	PxShape* shape = physics.createShape(
		PxTetrahedronMeshGeometry(
			volumeMesh.getCollisionMesh()),
		materials, 1, true, shapeFlags);
	if(!shape)
		return false;
	const bool attached = fixture.volume->attachShape(*shape);
	shape->release();
	if(!attached ||
		!fixture.volume->attachSimulationMesh(
			*volumeMesh.getSimulationMesh(),
			*volumeMesh.getDeformableVolumeAuxData()))
		return false;

	fixture.positions =
		fixture.volume->getSimPositionInvMassBufferH();
	fixture.velocities =
		fixture.volume->getSimVelocityBufferH();
	PxVec4* collisionPositions =
		fixture.volume->getPositionInvMassBufferH();
	PxVec4* collisionRestPositions =
		fixture.volume->getRestPositionBufferH();
	const PxTetrahedronMesh* simulationMesh =
		fixture.volume->getSimulationMesh();
	if(!fixture.positions || !fixture.velocities ||
		!collisionPositions || !collisionRestPositions ||
		!simulationMesh)
		return false;
	const PxVec3* vertices = simulationMesh->getVertices();
	const PxU32 vertexCount = simulationMesh->getNbVertices();
	fixture.initialPositions.resize(vertexCount);
	for(PxU32 i = 0; i < vertexCount; ++i)
	{
		PxVec3 position = vertices[i];
		PxReal invMass = 1.0f;
		if(fixedTarget)
		{
			const PxReal centerX =
				position.x < 0.0f ? -2.0f : 2.0f;
			PxVec3 local =
				position - PxVec3(centerX, 0.0f, 0.0f);
			local = PxVec3(
				3.0f * local.x,
				-2.0f * local.y,
				-2.0f * local.z);
			position =
				PxVec3(centerX, 0.0f, 0.0f) + local;
			invMass = 0.0f;
		}
		else
			// Match the Surface pair-filter fixture: start just inside the
			// 0.05 contact shell so the first position correction does not
			// become an artificial launch velocity.
			position.y += 0.049f;
		fixture.positions[i] = PxVec4(position, invMass);
		fixture.velocities[i] = PxVec4(
			PxVec3(0.0f), invMass);
		fixture.initialPositions[i] =
			fixture.positions[i];
	}
	if(!fixedTarget)
	{
		PxDeformableVolumeExt::updateMass(
			*fixture.volume, 100.0f, 50.0f,
			fixture.positions);
		for(PxU32 i = 0; i < vertexCount; ++i)
		{
			fixture.velocities[i].w =
				fixture.positions[i].w;
			fixture.initialPositions[i].w =
				fixture.positions[i].w;
		}
	}
	PxDeformableVolumeExt::updateEmbeddedCollisionMesh(
		*fixture.volume, fixture.positions,
		collisionPositions);
	const PxU32 collisionVertexCount =
		fixture.volume->getCollisionMesh()->getNbVertices();
	for(PxU32 i = 0; i < collisionVertexCount; ++i)
		collisionRestPositions[i] = collisionPositions[i];
	fixture.volume->setSolverIterationCounts(16, 1);
	fixture.volume->setLinearDamping(2.0f);
	if(fixedTarget)
		fixture.volume->setActorFlag(
			PxActorFlag::eDISABLE_GRAVITY, true);
	fixture.volume->markDirty(
		PxDeformableVolumeDataFlag::eALL);
	return scene.addActor(*fixture.volume) &&
		fixture.volume->getScene() == &scene;
}

struct PairFilterSurfaceFixture
{
	PxTriangleMesh* mesh;
	PxDeformableSurfaceMaterial* material;
	PxDeformableSurface* surface;
	PxVec4* positions;
	PxVec4* velocities;
	std::vector<PxVec4> initialPositions;

	PairFilterSurfaceFixture()
		: mesh(NULL), material(NULL), surface(NULL),
		  positions(NULL), velocities(NULL)
	{
	}
};

static bool createPairFilterSurfaceQueryFixture(
	PxPhysics& physics,
	PxScene& scene,
	const PxTolerancesScale& scale,
	PairFilterSurfaceFixture& fixture)
{
	const PxVec3 vertices[] =
	{
		// Keep each query triangle strictly inside the transformed Volume
		// top face.  The previous points straddled its sloped side faces, so
		// minY measured valid draping rather than shell penetration.
		// Start just inside the 0.05 contact shell.  A deeper initial
		// overlap would turn the first position correction into a launch
		// velocity and stop this fixture from isolating filter ownership.
		PxVec3(-2.4f, 0.049f, 0.333333f),
		PxVec3(-1.6f, 0.049f, 0.333333f),
		PxVec3(-2.0f, 0.049f, -0.466667f),
		PxVec3(1.6f, 0.049f, 0.333333f),
		PxVec3(2.4f, 0.049f, 0.333333f),
		PxVec3(2.0f, 0.049f, -0.466667f)
	};
	const PxU32 triangles[] =
	{
		0, 1, 2,
		3, 4, 5
	};
	PxTriangleMeshDesc meshDesc;
	meshDesc.points.count = 6;
	meshDesc.points.data = vertices;
	meshDesc.points.stride = sizeof(PxVec3);
	meshDesc.triangles.count = 2;
	meshDesc.triangles.data = triangles;
	meshDesc.triangles.stride = 3 * sizeof(PxU32);
	PxCookingParams cookingParams(scale);
	cookingParams.buildGPUData = false;
	fixture.mesh = PxCreateTriangleMesh(
		cookingParams, meshDesc,
		physics.getPhysicsInsertionCallback());
	if(!fixture.mesh)
		return false;
	fixture.material =
		physics.createDeformableSurfaceMaterial(
			2.0e4f, 0.3f, 0.08f, 0.02f, 0.0f);
	fixture.surface = physics.createDeformableSurface(
		PxDeformableSurfaceBackend::eCPU_AVBD);
	if(!fixture.material || !fixture.surface)
		return false;
	PxDeformableSurfaceMaterial* materials[] =
		{fixture.material};
	const PxShapeFlags shapeFlags =
		PxShapeFlag::eSIMULATION_SHAPE |
		PxShapeFlag::eSCENE_QUERY_SHAPE;
	PxShape* shape = physics.createShape(
		PxTriangleMeshGeometry(fixture.mesh),
		materials, 1, true, shapeFlags);
	if(!shape)
		return false;
	const bool attached =
		fixture.surface->attachShape(*shape);
	shape->release();
	if(!attached)
		return false;
	fixture.positions =
		fixture.surface->getPositionInvMassBufferH();
	fixture.velocities =
		fixture.surface->getVelocityBufferH();
	PxVec4* restPositions =
		fixture.surface->getRestPositionBufferH();
	const PxVec3* cookedVertices = fixture.mesh->getVertices();
	const PxU32 vertexCount = fixture.mesh->getNbVertices();
	if(!fixture.positions || !fixture.velocities ||
		!restPositions || !cookedVertices || vertexCount != 6)
		return false;
	fixture.initialPositions.resize(vertexCount);
	for(PxU32 i = 0; i < vertexCount; ++i)
	{
		// Triangle cooking may remap vertices.  Actor buffers are indexed
		// by the cooked mesh, so source-order positions would attach the
		// prepared triangle topology to the wrong points.
		fixture.positions[i] =
			PxVec4(cookedVertices[i], 1.0f);
		fixture.velocities[i] =
			PxVec4(PxVec3(0.0f), 1.0f);
		restPositions[i] =
			PxVec4(cookedVertices[i], 0.0f);
		fixture.initialPositions[i] = fixture.positions[i];
	}
	fixture.surface->setSolverIterationCounts(16);
	fixture.surface->setLinearDamping(2.0f);
	fixture.surface->markDirty(
		PxDeformableSurfaceDataFlag::eALL);
	return scene.addActor(*fixture.surface) &&
		fixture.surface->getScene() == &scene;
}

static PxReal dynamicCentroidY(
	const PxVec4* positions, PxU32 vertexCount)
{
	PxReal sum = 0.0f;
	PxU32 dynamicCount = 0;
	for(PxU32 i = 0; i < vertexCount; ++i)
	{
		if(positions[i].w > 0.0f)
		{
			sum += positions[i].y;
			++dynamicCount;
		}
	}
	return dynamicCount ? sum / PxReal(dynamicCount) : 0.0f;
}

static PxReal dynamicCentroidX(
	const PxVec4* positions, PxU32 vertexCount)
{
	PxReal sum = 0.0f;
	PxU32 dynamicCount = 0;
	for(PxU32 i = 0; i < vertexCount; ++i)
	{
		if(positions[i].w > 0.0f)
		{
			sum += positions[i].x;
			++dynamicCount;
		}
	}
	return dynamicCount ? sum / PxReal(dynamicCount) : 0.0f;
}

static bool sampleSurface(
	const PxVec4* positions,
	const PxVec4* velocities,
	const std::vector<PxVec3>& initialPositions,
	Metrics& metrics,
	PxU32 ignoredPinnedIndex)
{
	bool finite = true;
	for(PxU32 i = 0; i < PxU32(initialPositions.size()); ++i)
	{
		const PxVec3 position = positions[i].getXYZ();
		const PxVec3 velocity = velocities[i].getXYZ();
		if(!position.isFinite() || !velocity.isFinite() ||
			!PxIsFinite(positions[i].w))
		{
			++metrics.nonFiniteSamples;
			finite = false;
			continue;
		}
		metrics.minY = PxMin(metrics.minY, position.y);
		metrics.maxSpeed = PxMax(
			metrics.maxSpeed, velocity.magnitude());
		const PxReal displacement =
			(position - initialPositions[i]).magnitude();
		if(positions[i].w == 0.0f && i != ignoredPinnedIndex)
			metrics.maxPinnedDrift = PxMax(
				metrics.maxPinnedDrift, displacement);
		else
			metrics.maxDynamicDisplacement = PxMax(
				metrics.maxDynamicDisplacement, displacement);
	}
	return finite;
}

static bool tetrahedronCentroidPositiveX(
	const PxTetrahedronMesh& mesh,
	PxU32 tetrahedronIndex,
	bool& positiveX)
{
	if(tetrahedronIndex >= mesh.getNbTetrahedrons())
		return false;
	const bool has16BitIndices =
		mesh.getTetrahedronMeshFlags() &
			PxTetrahedronMeshFlag::e16_BIT_INDICES;
	const PxU16* tetrahedra16 = has16BitIndices
		? static_cast<const PxU16*>(mesh.getTetrahedrons())
		: NULL;
	const PxU32* tetrahedra32 = has16BitIndices
		? NULL
		: static_cast<const PxU32*>(mesh.getTetrahedrons());
	const PxVec3* vertices = mesh.getVertices();
	if(!vertices || (!tetrahedra16 && !tetrahedra32))
		return false;
	PxVec3 centroid(0.0f);
	for(PxU32 i = 0; i < 4; ++i)
	{
		const PxU32 vertexIndex = has16BitIndices
			? tetrahedra16[4 * tetrahedronIndex + i]
			: tetrahedra32[4 * tetrahedronIndex + i];
		if(vertexIndex >= mesh.getNbVertices())
			return false;
		centroid += vertices[vertexIndex];
	}
	positiveX = centroid.x > 0.0f;
	return true;
}

static bool findTriangleForXSign(
	const PxTriangleMesh& mesh,
	bool positiveX,
	PxU32& triangleIndex)
{
	const bool has16BitIndices =
		mesh.getTriangleMeshFlags() &
			PxTriangleMeshFlag::e16_BIT_INDICES;
	const PxU16* triangles16 = has16BitIndices
		? static_cast<const PxU16*>(mesh.getTriangles())
		: NULL;
	const PxU32* triangles32 = has16BitIndices
		? NULL
		: static_cast<const PxU32*>(mesh.getTriangles());
	const PxVec3* vertices = mesh.getVertices();
	if(!vertices || (!triangles16 && !triangles32))
		return false;
	for(PxU32 candidate = 0;
		candidate < mesh.getNbTriangles(); ++candidate)
	{
		PxVec3 centroid(0.0f);
		for(PxU32 i = 0; i < 3; ++i)
		{
			const PxU32 vertexIndex = has16BitIndices
				? triangles16[3 * candidate + i]
				: triangles32[3 * candidate + i];
			if(vertexIndex >= mesh.getNbVertices())
				return false;
			centroid += vertices[vertexIndex];
		}
		if((centroid.x > 0.0f) == positiveX)
		{
			triangleIndex = candidate;
			return true;
		}
	}
	return false;
}

static bool runVolumePairFilterCase(
	const Snippets::HeadlessOptions& options,
	Snippets::TrackingErrorCallback& errorCallback,
	Metrics& metrics,
	bool volumeTargetCase)
{
	PxDefaultAllocator allocator;
	PxFoundation* foundation = PxCreateFoundation(
		PX_PHYSICS_VERSION, allocator, errorCallback);
	if(!foundation)
		return false;

	PxPhysics* physics = NULL;
	PxDefaultCpuDispatcher* dispatcher = NULL;
	PxScene* scene = NULL;
	PxDeformableVolumeMesh* volumeMesh = NULL;
	PxDeformableVolumeMaterial* volumeMaterial = NULL;
	PxDeformableElementFilter* elementFilter = NULL;
	PairFilterVolumeFixture query;
	PairFilterVolumeFixture volumeTarget;
	PairFilterSurfaceFixture surfaceQuery;
	bool extensionsInitialized = false;
	bool success = false;

	do
	{
		const PxTolerancesScale scale;
		physics = PxCreatePhysics(
			PX_PHYSICS_VERSION, *foundation, scale, true, NULL);
		if(!physics)
			break;
		extensionsInitialized = PxInitExtensions(*physics, NULL);
		if(!extensionsInitialized)
			break;

		PxSceneDesc sceneDesc(scale);
		// A low, steady load isolates pair ownership from the current
		// first-order soft-contact shell's high-load penetration response.
		sceneDesc.gravity = PxVec3(0.0f, -1.0f, 0.0f);
		sceneDesc.solverType = PxSolverType::eAVBD;
		sceneDesc.flags |= PxSceneFlag::eDISABLE_SLEEPING;
		sceneDesc.filterShader = PxDefaultSimulationFilterShader;
		dispatcher = PxDefaultCpuDispatcherCreate(
			options.dispatcherThreads);
		if(!dispatcher)
			break;
		sceneDesc.cpuDispatcher = dispatcher;
		scene = physics->createScene(sceneDesc);
		if(!scene)
			break;

		if(!buildPairFilterVolumeMesh(
			*physics, scale, volumeMesh))
			break;
		volumeMaterial =
			physics->createDeformableVolumeMaterial(
				2.0e5f, 0.3f, 0.2f, 0.01f);
		if(!volumeMaterial)
			break;
		// This exact-ownership fixture predates public material-model
		// dispatch and was authored against the former CPU Neo-Hookean
		// path. Keep that constitutive response explicit so changing the
		// public default to co-rotational cannot weaken the filter proof.
		volumeMaterial->setMaterialModel(
			PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);
		if(volumeTargetCase)
		{
			if(!createPairFilterVolumeFixture(
				*physics, *scene, *volumeMesh,
				*volumeMaterial, false, query))
				break;
		}
		else if(!createPairFilterSurfaceQueryFixture(
			*physics, *scene, scale, surfaceQuery))
			break;
		metrics.actorCreated = 1;
		metrics.shapeAttached = 1;
		metrics.hostBuffersInitialized = 1;
		metrics.actorAdded = 1;

		if(!createPairFilterVolumeFixture(
			*physics, *scene, *volumeMesh,
			*volumeMaterial, true, volumeTarget))
			break;

		const PxTetrahedronMesh* collisionMesh =
			volumeTarget.volume->getCollisionMesh();
		if(!collisionMesh ||
			collisionMesh->getNbTetrahedrons() != 2)
			break;
		const PxU32 targetElement = 0;
		bool selectedPositiveX = false;
		if(!tetrahedronCentroidPositiveX(
			*collisionMesh, targetElement, selectedPositiveX))
			break;

		const PxU32 groupElementCount = 1;
		PxU32 queryElement = targetElement;
		PxU32 actorElement[2] =
			{targetElement, queryElement};
		PxDeformableElementFilterData filterData;
		if(volumeTargetCase)
		{
			// Put the fixed target first so Volume--Volume prep must
			// normalize the only dynamic contact-query direction.
			filterData.actor[0] = volumeTarget.volume;
			filterData.actor[1] = query.volume;
		}
		else
		{
			if(!surfaceQuery.mesh ||
				!findTriangleForXSign(
					*surfaceQuery.mesh,
					selectedPositiveX, queryElement))
				break;
			// Deliberately reverse the canonical Volume--Surface public
			// actor order. Prep must preserve the corresponding elements.
			filterData.actor[0] = surfaceQuery.surface;
			filterData.actor[1] = volumeTarget.volume;
			actorElement[0] = queryElement;
			actorElement[1] = targetElement;
		}
		filterData.groupElementCounts[0].data =
			&groupElementCount;
		filterData.groupElementCounts[0].count = 1;
		filterData.groupElementCounts[1].data =
			&groupElementCount;
		filterData.groupElementCounts[1].count = 1;
		filterData.groupElementIndices[0].data =
			&actorElement[0];
		filterData.groupElementIndices[0].count = 1;
		filterData.groupElementIndices[1].data =
			&actorElement[1];
		filterData.groupElementIndices[1].count = 1;
		elementFilter =
			physics->createDeformableElementFilter(filterData);
		if(!elementFilter)
			break;
		metrics.elementFilterCreated = 1;

		PxActor* queryActor = volumeTargetCase
			? static_cast<PxActor*>(query.volume)
			: static_cast<PxActor*>(surfaceQuery.surface);
		PxVec4* queryPositions = volumeTargetCase
			? query.positions : surfaceQuery.positions;
		PxVec4* queryVelocities = volumeTargetCase
			? query.velocities : surfaceQuery.velocities;
		const std::vector<PxVec4>& queryInitialPositions =
			volumeTargetCase
				? query.initialPositions
				: surfaceQuery.initialPositions;
		const PxU32 vertexCount =
			PxU32(queryInitialPositions.size());
		if(!queryActor || !queryPositions || !queryVelocities ||
			vertexCount == 0)
			break;
		metrics.initialDynamicCentroidY =
			dynamicCentroidY(queryPositions, vertexCount);
		const PxU32 churnFrame =
			PxMax<PxU32>(1, 2 * options.frames / 5);
		const PxU32 filterReleaseFrame =
			PxMax<PxU32>(churnFrame + 1, options.frames / 2);
		bool samplesFinite = true;
		PxReal finalSelectedMinY = PX_MAX_F32;
		PxReal finalUnselectedMinY = PX_MAX_F32;
		for(PxU32 frame = 0; frame < options.frames; ++frame)
		{
			if(frame == churnFrame)
			{
				scene->removeActor(*queryActor);
				if(queryActor->getScene() != NULL)
				{
					samplesFinite = false;
					break;
				}
				metrics.actorRemoved = 1;
				if(!scene->addActor(*queryActor) ||
					queryActor->getScene() != scene)
				{
					samplesFinite = false;
					break;
				}
				metrics.actorReadded = 1;
			}
			if(elementFilter &&
				frame > churnFrame &&
				frame < filterReleaseFrame)
				metrics.elementFilterHeldAcrossReadd = 1;
			if(elementFilter && frame == filterReleaseFrame)
			{
				elementFilter->release();
				elementFilter = NULL;
				metrics.elementFilterReleased = 1;
				scene->removeActor(*queryActor);
				if(queryActor->getScene() != NULL)
				{
					samplesFinite = false;
					break;
				}
				for(PxU32 i = 0; i < vertexCount; ++i)
				{
					queryPositions[i] =
						queryInitialPositions[i];
					queryVelocities[i] = PxVec4(
						PxVec3(0.0f),
						queryInitialPositions[i].w);
				}
				if(volumeTargetCase)
					query.volume->markDirty(
						PxDeformableVolumeDataFlags(
							PxU32(
								PxDeformableVolumeDataFlag::
									eSIM_POSITION_INVMASS) |
							PxU32(
								PxDeformableVolumeDataFlag::
									eSIM_VELOCITY)));
				else
					surfaceQuery.surface->markDirty(
						PxDeformableSurfaceDataFlags(
							PxU32(
								PxDeformableSurfaceDataFlag::
									ePOSITION_INVMASS) |
							PxU32(
								PxDeformableSurfaceDataFlag::
									eVELOCITY)));
				if(!scene->addActor(*queryActor) ||
					queryActor->getScene() != scene)
				{
					samplesFinite = false;
					break;
				}
			}

			scene->simulate(options.dt);
			if(!scene->fetchResults(true))
			{
				++metrics.fetchFailures;
				samplesFinite = false;
				break;
			}

			PxReal currentSelectedMinY = PX_MAX_F32;
			PxReal currentUnselectedMinY = PX_MAX_F32;
			PxReal currentMinY = PX_MAX_F32;
			PxReal currentMaxSpeed = 0.0f;
			for(PxU32 i = 0; i < vertexCount; ++i)
			{
				const PxVec4& position = queryPositions[i];
				const PxVec4& velocity = queryVelocities[i];
				if(!position.isFinite() || !velocity.isFinite() ||
					position.w < 0.0f)
				{
					++metrics.nonFiniteSamples;
					samplesFinite = false;
					continue;
				}
				const PxReal speed = velocity.getXYZ().magnitude();
				const PxReal displacement =
					(position.getXYZ() -
					 queryInitialPositions[i].getXYZ()).
						magnitude();
				currentMinY = PxMin(currentMinY, position.y);
				currentMaxSpeed =
					PxMax(currentMaxSpeed, speed);
				metrics.minY =
					PxMin(metrics.minY, position.y);
				metrics.maxSpeed =
					PxMax(metrics.maxSpeed, speed);
				metrics.maxDynamicDisplacement = PxMax(
					metrics.maxDynamicDisplacement,
					displacement);
				const bool particleSelected =
					(queryInitialPositions[i].x > 0.0f) ==
						selectedPositiveX;
				if(particleSelected)
					currentSelectedMinY = PxMin(
						currentSelectedMinY, position.y);
				else
					currentUnselectedMinY = PxMin(
						currentUnselectedMinY, position.y);
			}
			if(elementFilter)
			{
				metrics.partialFilterFilteredMinY = PxMin(
					metrics.partialFilterFilteredMinY,
					currentSelectedMinY);
				metrics.partialFilterUnfilteredMinY = PxMin(
					metrics.partialFilterUnfilteredMinY,
					currentUnselectedMinY);
				metrics.elementFilterMinY =
					metrics.partialFilterFilteredMinY;
				if(currentSelectedMinY < -0.2f)
					metrics.elementFilterSuppressedContact = 1;
				if(currentUnselectedMinY > -0.05f &&
					currentUnselectedMinY < 0.1f)
					metrics.
						partialFilterUnfilteredContactHeld = 1;
			}
			if(frame + 1 == options.frames)
			{
				finalSelectedMinY = currentSelectedMinY;
				finalUnselectedMinY = currentUnselectedMinY;
				metrics.finalMinY = currentMinY;
				metrics.finalMaxSpeed = currentMaxSpeed;
				metrics.finalDynamicCentroidY =
					dynamicCentroidY(
						queryPositions, vertexCount);
			}
		}

		const PxBounds3 bounds = volumeTargetCase
			? query.volume->getWorldBounds()
			: surfaceQuery.surface->getWorldBounds();
		metrics.boundsFinite =
			!bounds.isEmpty() &&
			bounds.minimum.isFinite() &&
			bounds.maximum.isFinite() ? 1u : 0u;
		metrics.pinnedStable = 1;
		metrics.dynamicMoved =
			metrics.maxDynamicDisplacement > 0.2f ? 1u : 0u;
		metrics.elementFilterFinalMinY = metrics.finalMinY;
		metrics.elementFilterContactRestored =
			metrics.elementFilterReleased &&
			finalSelectedMinY > -0.05f &&
			finalUnselectedMinY > -0.05f &&
			PxAbs(
				finalSelectedMinY -
				finalUnselectedMinY) < 0.05f &&
			metrics.elementFilterFinalMinY < 0.1f &&
			metrics.finalMaxSpeed < 0.1f ? 1u : 0u;
		metrics.partialFilterExactOwnership =
			metrics.elementFilterSuppressedContact &&
			metrics.partialFilterUnfilteredContactHeld &&
			metrics.partialFilterUnfilteredMinY > -0.05f
				? 1u : 0u;
		success = samplesFinite &&
			metrics.actorCreated && metrics.shapeAttached &&
			metrics.hostBuffersInitialized &&
			metrics.actorAdded && metrics.actorRemoved &&
			metrics.actorReadded &&
			metrics.fetchFailures == 0 &&
			metrics.nonFiniteSamples == 0 &&
			metrics.pinnedStable && metrics.dynamicMoved &&
			metrics.boundsFinite &&
			metrics.elementFilterCreated &&
			metrics.elementFilterHeldAcrossReadd &&
			metrics.elementFilterSuppressedContact &&
			metrics.elementFilterReleased &&
			metrics.elementFilterContactRestored &&
			metrics.partialFilterExactOwnership;
	}
	while(false);

	if(elementFilter)
	{
		elementFilter->release();
		elementFilter = NULL;
	}
	if(surfaceQuery.surface)
	{
		if(surfaceQuery.surface->getScene())
			surfaceQuery.surface->getScene()->removeActor(
				*surfaceQuery.surface);
		surfaceQuery.surface->release();
		surfaceQuery.surface = NULL;
	}
	if(volumeTarget.volume)
	{
		if(volumeTarget.volume->getScene())
			volumeTarget.volume->getScene()->removeActor(
				*volumeTarget.volume);
		volumeTarget.volume->release();
		volumeTarget.volume = NULL;
	}
	if(query.volume)
	{
		if(query.volume->getScene())
			query.volume->getScene()->removeActor(*query.volume);
		query.volume->release();
		query.volume = NULL;
	}
	PX_RELEASE(surfaceQuery.material);
	PX_RELEASE(surfaceQuery.mesh);
	PX_RELEASE(volumeMaterial);
	PX_RELEASE(volumeMesh);
	PX_RELEASE(scene);
	PX_RELEASE(dispatcher);
	if(extensionsInitialized)
		PxCloseExtensions();
	PX_RELEASE(physics);
	PX_RELEASE(foundation);
	metrics.cleanupComplete = 1;
	return success;
}

static bool runBendingCase(
	const Snippets::HeadlessOptions& options,
	Snippets::TrackingErrorCallback& errorCallback,
	Metrics& metrics)
{
	PxDefaultAllocator allocator;
	PxFoundation* foundation = PxCreateFoundation(
		PX_PHYSICS_VERSION, allocator, errorCallback);
	if(!foundation)
		return false;

	PxPhysics* physics = NULL;
	PxDefaultCpuDispatcher* dispatcher = NULL;
	PxScene* scene = NULL;
	BendingSurfaceFixture zeroBending;
	BendingSurfaceFixture stiffBending;
	bool extensionsInitialized = false;
	bool success = false;

	do
	{
		const PxTolerancesScale scale;
		physics = PxCreatePhysics(
			PX_PHYSICS_VERSION, *foundation, scale, true, NULL);
		if(!physics)
			break;
		extensionsInitialized = PxInitExtensions(*physics, NULL);
		if(!extensionsInitialized)
			break;

		PxSceneDesc sceneDesc(scale);
		sceneDesc.gravity = PxVec3(0.0f);
		sceneDesc.solverType = PxSolverType::eAVBD;
		sceneDesc.filterShader = PxDefaultSimulationFilterShader;
		dispatcher = PxDefaultCpuDispatcherCreate(
			options.dispatcherThreads);
		if(!dispatcher)
			break;
		sceneDesc.cpuDispatcher = dispatcher;
		scene = physics->createScene(sceneDesc);
		if(!scene)
			break;

		if(!createBendingSurfaceFixture(
				*physics, *scene, scale,
				PxVec3(-2.0f, 0.0f, 0.0f),
				0.0f, false, false, zeroBending) ||
			!createBendingSurfaceFixture(
				*physics, *scene, scale,
				PxVec3(2.0f, 0.0f, 0.0f),
				10.0f, false, false, stiffBending))
			break;

		metrics.actorCreated = 1;
		metrics.shapeAttached = 1;
		metrics.hostBuffersInitialized = 1;
		metrics.actorAdded = 1;
		stiffBending.material->setBendingDamping(0.5f);
		metrics.bendingMaterialPairCreated =
			zeroBending.material->getBendingStiffness() == 0.0f &&
			stiffBending.material->getBendingStiffness() > 0.0f &&
			PxAbs(
				stiffBending.material->getBendingDamping() -
					0.5f) <= 1.0e-6f
				? 1u : 0u;
		metrics.initialDynamicCentroidY =
			stiffBending.positions[3].y;
		metrics.bendingInitialPlaneError = PxAbs(
			stiffBending.positions[3].y -
			stiffBending.restPositions[3].y);

		const PxU32 churnFrame = PxMax<PxU32>(
			1, options.frames / 3);
		bool samplesFinite = true;
		for(PxU32 frame = 0; frame < options.frames; ++frame)
		{
			if(frame == churnFrame)
			{
				scene->removeActor(*stiffBending.surface);
				if(stiffBending.surface->getScene() != NULL)
					break;
				metrics.actorRemoved = 1;
				if(!scene->addActor(*stiffBending.surface) ||
					stiffBending.surface->getScene() != scene)
					break;
				metrics.actorReadded = 1;
			}

			scene->simulate(options.dt);
			if(!scene->fetchResults(true))
			{
				++metrics.fetchFailures;
				break;
			}

			for(PxU32 i = 0; i < 4; ++i)
			{
				const PxVec4& zeroPosition =
					zeroBending.positions[i];
				const PxVec4& zeroVelocity =
					zeroBending.velocities[i];
				const PxVec4& stiffPosition =
					stiffBending.positions[i];
				const PxVec4& stiffVelocity =
					stiffBending.velocities[i];
				if(!zeroPosition.isFinite() ||
					!zeroVelocity.isFinite() ||
					!stiffPosition.isFinite() ||
					!stiffVelocity.isFinite())
				{
					++metrics.nonFiniteSamples;
					samplesFinite = false;
					continue;
				}

				if(i < 3)
				{
					metrics.maxPinnedDrift = PxMax(
						metrics.maxPinnedDrift,
						(zeroPosition.getXYZ() -
							zeroBending.initialPositions[i]).
								magnitude());
					metrics.maxPinnedDrift = PxMax(
						metrics.maxPinnedDrift,
						(stiffPosition.getXYZ() -
							stiffBending.initialPositions[i]).
								magnitude());
				}
				else
				{
					metrics.bendingZeroControlDisplacement =
						PxMax(
							metrics.
								bendingZeroControlDisplacement,
							(zeroPosition.getXYZ() -
								zeroBending.initialPositions[i]).
									magnitude());
					metrics.bendingStiffDisplacement = PxMax(
						metrics.bendingStiffDisplacement,
						(stiffPosition.getXYZ() -
							stiffBending.initialPositions[i]).
								magnitude());
					metrics.maxDynamicDisplacement =
						metrics.bendingStiffDisplacement;
				}
				metrics.maxSpeed = PxMax(
					metrics.maxSpeed,
					PxMax(
						zeroVelocity.getXYZ().magnitude(),
						stiffVelocity.getXYZ().magnitude()));
				metrics.minY = PxMin(
					metrics.minY, stiffPosition.y);
			}

			const PxVec3 stiffFree =
				stiffBending.positions[3].getXYZ();
			for(PxU32 pinnedIndex = 1;
				pinnedIndex <= 2; ++pinnedIndex)
			{
				const PxReal restLength =
					(stiffBending.restVertices[3] -
						stiffBending.
							restVertices[pinnedIndex]).
						magnitude();
				const PxReal currentLength =
					(stiffFree -
						stiffBending.positions[pinnedIndex].
							getXYZ()).magnitude();
				metrics.bendingMaxEdgeStrain = PxMax(
					metrics.bendingMaxEdgeStrain,
					PxAbs(currentLength - restLength) /
						restLength);
			}
		}

		metrics.finalDynamicCentroidY =
			stiffBending.positions[3].y;
		metrics.finalMinY = PX_MAX_F32;
		metrics.finalMaxSpeed = 0.0f;
		for(PxU32 i = 0; i < 4; ++i)
		{
			metrics.finalMinY = PxMin(
				metrics.finalMinY,
				stiffBending.positions[i].y);
			metrics.finalMaxSpeed = PxMax(
				metrics.finalMaxSpeed,
				stiffBending.velocities[i].getXYZ().
					magnitude());
		}
		metrics.bendingFinalPlaneError = PxAbs(
			stiffBending.positions[3].y -
			stiffBending.restPositions[3].y);
		metrics.pinnedStable =
			metrics.maxPinnedDrift <= 1.0e-4f ? 1u : 0u;
		metrics.dynamicMoved =
			metrics.maxDynamicDisplacement >= 1.0e-2f ? 1u : 0u;
		const PxBounds3 zeroBounds =
			zeroBending.surface->getWorldBounds();
		const PxBounds3 stiffBounds =
			stiffBending.surface->getWorldBounds();
		metrics.boundsFinite =
			!zeroBounds.isEmpty() &&
			zeroBounds.minimum.isFinite() &&
			zeroBounds.maximum.isFinite() &&
			!stiffBounds.isEmpty() &&
			stiffBounds.minimum.isFinite() &&
			stiffBounds.maximum.isFinite() ? 1u : 0u;
		metrics.bendingZeroControlHeld =
			metrics.bendingZeroControlDisplacement <= 1.0e-4f
				? 1u : 0u;
		metrics.bendingResponseObserved =
			metrics.bendingStiffDisplacement > 5.0e-2f &&
			metrics.bendingFinalPlaneError <
				metrics.bendingInitialPlaneError - 5.0e-2f
				? 1u : 0u;
		metrics.bendingMembraneIsolated =
			metrics.bendingMaxEdgeStrain < 5.0e-2f
				? 1u : 0u;

		success = samplesFinite &&
			metrics.actorCreated && metrics.shapeAttached &&
			metrics.hostBuffersInitialized && metrics.actorAdded &&
			metrics.actorRemoved && metrics.actorReadded &&
			metrics.fetchFailures == 0 &&
			metrics.nonFiniteSamples == 0 &&
			metrics.pinnedStable && metrics.dynamicMoved &&
			metrics.boundsFinite &&
			metrics.bendingMaterialPairCreated &&
			metrics.bendingZeroControlHeld &&
			metrics.bendingResponseObserved &&
			metrics.bendingMembraneIsolated;
	}
	while(false);

	releaseBendingSurfaceFixture(stiffBending);
	releaseBendingSurfaceFixture(zeroBending);
	PX_RELEASE(scene);
	PX_RELEASE(dispatcher);
	if(extensionsInitialized)
		PxCloseExtensions();
	PX_RELEASE(physics);
	PX_RELEASE(foundation);
	metrics.cleanupComplete = 1;
	return success;
}

static bool runFlatteningCase(
	const Snippets::HeadlessOptions& options,
	Snippets::TrackingErrorCallback& errorCallback,
	Metrics& metrics)
{
	PxDefaultAllocator allocator;
	PxFoundation* foundation = PxCreateFoundation(
		PX_PHYSICS_VERSION, allocator, errorCallback);
	if(!foundation)
		return false;

	PxPhysics* physics = NULL;
	PxDefaultCpuDispatcher* dispatcher = NULL;
	PxScene* scene = NULL;
	BendingSurfaceFixture control;
	BendingSurfaceFixture flattening;
	bool extensionsInitialized = false;
	bool success = false;

	do
	{
		const PxTolerancesScale scale;
		physics = PxCreatePhysics(
			PX_PHYSICS_VERSION, *foundation, scale, true, NULL);
		if(!physics)
			break;
		extensionsInitialized = PxInitExtensions(*physics, NULL);
		if(!extensionsInitialized)
			break;

		PxSceneDesc sceneDesc(scale);
		sceneDesc.gravity = PxVec3(0.0f);
		sceneDesc.solverType = PxSolverType::eAVBD;
		sceneDesc.filterShader = PxDefaultSimulationFilterShader;
		dispatcher = PxDefaultCpuDispatcherCreate(
			options.dispatcherThreads);
		if(!dispatcher)
			break;
		sceneDesc.cpuDispatcher = dispatcher;
		scene = physics->createScene(sceneDesc);
		if(!scene)
			break;

		if(!createBendingSurfaceFixture(
				*physics, *scene, scale,
				PxVec3(-2.0f, 0.0f, 0.0f),
				10.0f, true, false, control) ||
			!createBendingSurfaceFixture(
				*physics, *scene, scale,
				PxVec3(2.0f, 0.0f, 0.0f),
				10.0f, true, true, flattening))
			break;

		metrics.actorCreated = 1;
		metrics.shapeAttached = 1;
		metrics.hostBuffersInitialized = 1;
		metrics.actorAdded = 1;
		const bool initialFlagApplied =
			flattening.surface->getDeformableSurfaceFlags() &
			PxDeformableSurfaceFlag::eENABLE_FLATTENING;
		metrics.initialDynamicCentroidY =
			flattening.positions[3].y;
		metrics.flatteningInitialPlaneError =
			PxAbs(flattening.positions[3].y -
				flattening.positions[0].y);
		metrics.flatteningMinimumPlaneError =
			metrics.flatteningInitialPlaneError;

		const PxU32 churnFrame = PxMax<PxU32>(
			1, options.frames / 3);
		const PxU32 retargetFrame = PxMax<PxU32>(
			churnFrame + 1, (2 * options.frames) / 3);
		bool retargetIssued = false;
		bool samplesFinite = true;
		for(PxU32 frame = 0; frame < options.frames; ++frame)
		{
			if(frame == churnFrame)
			{
				scene->removeActor(*flattening.surface);
				if(flattening.surface->getScene() != NULL)
					break;
				metrics.actorRemoved = 1;
				if(!scene->addActor(*flattening.surface) ||
					flattening.surface->getScene() != scene)
					break;
				metrics.actorReadded = 1;
			}
			if(frame == retargetFrame)
			{
				flattening.surface->setDeformableSurfaceFlag(
					PxDeformableSurfaceFlag::eENABLE_FLATTENING,
					false);
				retargetIssued = true;
			}

			scene->simulate(options.dt);
			if(!scene->fetchResults(true))
			{
				++metrics.fetchFailures;
				break;
			}

			for(PxU32 i = 0; i < 4; ++i)
			{
				const PxVec4& controlPosition =
					control.positions[i];
				const PxVec4& controlVelocity =
					control.velocities[i];
				const PxVec4& targetPosition =
					flattening.positions[i];
				const PxVec4& targetVelocity =
					flattening.velocities[i];
				if(!controlPosition.isFinite() ||
					!controlVelocity.isFinite() ||
					!targetPosition.isFinite() ||
					!targetVelocity.isFinite())
				{
					++metrics.nonFiniteSamples;
					samplesFinite = false;
					continue;
				}

				if(i < 3)
				{
					metrics.maxPinnedDrift = PxMax(
						metrics.maxPinnedDrift,
						(controlPosition.getXYZ() -
							control.initialPositions[i]).
								magnitude());
					metrics.maxPinnedDrift = PxMax(
						metrics.maxPinnedDrift,
						(targetPosition.getXYZ() -
							flattening.initialPositions[i]).
								magnitude());
				}
				else
				{
					metrics.flatteningControlDisplacement =
						PxMax(
							metrics.
								flatteningControlDisplacement,
							(controlPosition.getXYZ() -
								control.initialPositions[i]).
									magnitude());
					metrics.flatteningTargetDisplacement =
						PxMax(
							metrics.
								flatteningTargetDisplacement,
							(targetPosition.getXYZ() -
								flattening.initialPositions[i]).
									magnitude());
					metrics.maxDynamicDisplacement =
						metrics.flatteningTargetDisplacement;
				}
				metrics.maxSpeed = PxMax(
					metrics.maxSpeed,
					PxMax(
						controlVelocity.getXYZ().magnitude(),
						targetVelocity.getXYZ().magnitude()));
				metrics.minY = PxMin(
					metrics.minY, targetPosition.y);
			}

			const PxReal planeError = PxAbs(
				flattening.positions[3].y -
				flattening.positions[0].y);
			metrics.flatteningMinimumPlaneError = PxMin(
				metrics.flatteningMinimumPlaneError,
				planeError);
			const PxVec3 targetFree =
				flattening.positions[3].getXYZ();
			for(PxU32 pinnedIndex = 1;
				pinnedIndex <= 2; ++pinnedIndex)
			{
				const PxReal restLength =
					(flattening.restVertices[3] -
						flattening.restVertices[pinnedIndex]).
							magnitude();
				const PxReal currentLength =
					(targetFree -
						flattening.positions[pinnedIndex].
							getXYZ()).magnitude();
				metrics.flatteningMaxEdgeStrain = PxMax(
					metrics.flatteningMaxEdgeStrain,
					PxAbs(currentLength - restLength) /
						restLength);
			}
		}

		metrics.finalDynamicCentroidY =
			flattening.positions[3].y;
		metrics.finalMinY = PX_MAX_F32;
		metrics.finalMaxSpeed = 0.0f;
		for(PxU32 i = 0; i < 4; ++i)
		{
			metrics.finalMinY = PxMin(
				metrics.finalMinY,
				flattening.positions[i].y);
			metrics.finalMaxSpeed = PxMax(
				metrics.finalMaxSpeed,
				flattening.velocities[i].getXYZ().
					magnitude());
		}
		metrics.flatteningFinalPlaneError = PxAbs(
			flattening.positions[3].y -
				flattening.positions[0].y);
		metrics.pinnedStable =
			metrics.maxPinnedDrift <= 1.0e-4f ? 1u : 0u;
		metrics.dynamicMoved =
			metrics.flatteningTargetDisplacement > 5.0e-2f
				? 1u : 0u;
		const PxBounds3 controlBounds =
			control.surface->getWorldBounds();
		const PxBounds3 flatteningBounds =
			flattening.surface->getWorldBounds();
		metrics.boundsFinite =
			!controlBounds.isEmpty() &&
			controlBounds.minimum.isFinite() &&
			controlBounds.maximum.isFinite() &&
			!flatteningBounds.isEmpty() &&
			flatteningBounds.minimum.isFinite() &&
			flatteningBounds.maximum.isFinite() ? 1u : 0u;
		metrics.flatteningFlagApplied =
			initialFlagApplied && retargetIssued &&
			!(flattening.surface->getDeformableSurfaceFlags() &
				PxDeformableSurfaceFlag::eENABLE_FLATTENING)
				? 1u : 0u;
		metrics.flatteningControlHeld =
			metrics.flatteningControlDisplacement <= 1.0e-4f
				? 1u : 0u;
		metrics.flatteningResponseObserved =
			metrics.flatteningTargetDisplacement > 5.0e-2f &&
			metrics.flatteningMinimumPlaneError <
				metrics.flatteningInitialPlaneError - 5.0e-2f
				? 1u : 0u;
		metrics.flatteningRetargetObserved =
			metrics.flatteningFinalPlaneError >
				metrics.flatteningMinimumPlaneError + 5.0e-2f
				? 1u : 0u;
		metrics.flatteningMembraneIsolated =
			metrics.flatteningMaxEdgeStrain < 5.0e-2f
				? 1u : 0u;

		success = samplesFinite &&
			metrics.actorCreated && metrics.shapeAttached &&
			metrics.hostBuffersInitialized && metrics.actorAdded &&
			metrics.actorRemoved && metrics.actorReadded &&
			metrics.fetchFailures == 0 &&
			metrics.nonFiniteSamples == 0 &&
			metrics.pinnedStable && metrics.dynamicMoved &&
			metrics.boundsFinite &&
			metrics.flatteningFlagApplied &&
			metrics.flatteningControlHeld &&
			metrics.flatteningResponseObserved &&
			metrics.flatteningRetargetObserved &&
			metrics.flatteningMembraneIsolated;
	}
	while(false);

	releaseBendingSurfaceFixture(flattening);
	releaseBendingSurfaceFixture(control);
	PX_RELEASE(scene);
	PX_RELEASE(dispatcher);
	if(extensionsInitialized)
		PxCloseExtensions();
	PX_RELEASE(physics);
	PX_RELEASE(foundation);
	metrics.cleanupComplete = 1;
	return success;
}

static bool runMotionControlsCase(
	const Snippets::HeadlessOptions& options,
	Snippets::TrackingErrorCallback& errorCallback,
	Metrics& metrics)
{
	PxDefaultAllocator allocator;
	PxFoundation* foundation = PxCreateFoundation(
		PX_PHYSICS_VERSION, allocator, errorCallback);
	if(!foundation)
		return false;

	PxPhysics* physics = NULL;
	PxDefaultCpuDispatcher* dispatcher = NULL;
	PxScene* scene = NULL;
	MotionSurfaceFixture control;
	MotionSurfaceFixture settling;
	bool extensionsInitialized = false;
	bool success = false;

	do
	{
		if(options.frames < 30)
			break;
		const PxTolerancesScale scale;
		physics = PxCreatePhysics(
			PX_PHYSICS_VERSION, *foundation, scale, true, NULL);
		if(!physics)
			break;
		extensionsInitialized = PxInitExtensions(*physics, NULL);
		if(!extensionsInitialized)
			break;

		PxSceneDesc sceneDesc(scale);
		sceneDesc.gravity = PxVec3(0.0f);
		sceneDesc.solverType = PxSolverType::eAVBD;
		sceneDesc.filterShader = PxDefaultSimulationFilterShader;
		dispatcher = PxDefaultCpuDispatcherCreate(
			options.dispatcherThreads);
		if(!dispatcher)
			break;
		sceneDesc.cpuDispatcher = dispatcher;
		scene = physics->createScene(sceneDesc);
		if(!scene)
			break;

		if(!createMotionSurfaceFixture(
				*physics, *scene, scale,
				PxVec3(-3.0f, 0.0f, 0.0f),
				PxVec3(10.0f, 0.0f, 0.0f),
				1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
				control) ||
			!createMotionSurfaceFixture(
				*physics, *scene, scale,
				PxVec3(3.0f, 0.0f, 0.0f),
				PxVec3(0.08f, 0.0f, 0.0f),
				PX_MAX_F32, 0.1f, 10.0f, 0.05f, 0.2f,
				settling))
			break;

		metrics.actorCreated = 1;
		metrics.shapeAttached = 1;
		metrics.hostBuffersInitialized = 1;
		metrics.actorAdded = 1;
		const PxVec3 initialControlCentroid =
			getMotionSurfaceCentroid(control);
		metrics.initialDynamicCentroidY =
			initialControlCentroid.y;
		PxVec3 controlPhaseStart = initialControlCentroid;
		bool samplesFinite = true;
		bool controlStayedAwake = true;

		for(PxU32 frame = 0; frame < options.frames; ++frame)
		{
			if(frame == 1)
			{
				scene->removeActor(*control.surface);
				if(control.surface->getScene() != NULL)
					break;
				metrics.actorRemoved = 1;
				if(!scene->addActor(*control.surface) ||
					control.surface->getScene() != scene)
					break;
				metrics.actorReadded = 1;
			}

			scene->simulate(options.dt);
			if(!scene->fetchResults(true))
			{
				++metrics.fetchFailures;
				break;
			}

			for(PxU32 i = 0; i < 4; ++i)
			{
				if(!control.positions[i].isFinite() ||
					!control.velocities[i].isFinite() ||
					!settling.positions[i].isFinite() ||
					!settling.velocities[i].isFinite())
				{
					++metrics.nonFiniteSamples;
					samplesFinite = false;
				}
				metrics.minY = PxMin(
					metrics.minY,
					PxMin(
						control.positions[i].y,
						settling.positions[i].y));
			}

			const PxVec3 controlCentroid =
				getMotionSurfaceCentroid(control);
			const PxReal controlSpeed =
				getMotionSurfaceMaxSpeed(control);
			const PxReal settlingSpeed =
				getMotionSurfaceMaxSpeed(settling);
			metrics.maxSpeed = PxMax(
				metrics.maxSpeed,
				PxMax(controlSpeed, settlingSpeed));
			metrics.maxDynamicDisplacement = PxMax(
				metrics.maxDynamicDisplacement,
				(controlCentroid - initialControlCentroid).
					magnitude());

			if(frame == 0)
			{
				metrics.motionMaxVelocityFirstStepDisplacement =
					(controlCentroid - initialControlCentroid).
						magnitude();
				metrics.motionMaxVelocityFirstStepSpeed =
					controlSpeed;
				metrics.motionMaxVelocityBounded =
					metrics.
						motionMaxVelocityFirstStepDisplacement <=
							options.dt * 1.01f &&
					controlSpeed <= 1.01f ? 1u : 0u;
				if(settlingSpeed <= 0.07f)
					metrics.motionSettlingApplied = 1;

				for(PxU32 i = 0; i < 4; ++i)
					control.velocities[i] =
						PxVec4(
							PxVec3(0.08f, 0.0f, 0.0f),
							control.velocities[i].w);
				control.surface->setMaxLinearVelocity(
					PX_MAX_F32);
				control.surface->setSettlingThreshold(0.1f);
				control.surface->setSettlingDamping(0.0f);
				control.surface->setSleepThreshold(0.05f);
				control.surface->setWakeCounter(0.2f);
				control.surface->markDirty(
					PxDeformableSurfaceDataFlag::eVELOCITY);
				controlPhaseStart =
					getMotionSurfaceCentroid(control);
			}
			else
			{
				if(control.surface->isSleeping())
					controlStayedAwake = false;
				metrics.dynamicMoved =
					(controlCentroid - controlPhaseStart).
						magnitude() > 0.01f ? 1u :
						metrics.dynamicMoved;
			}
			if(settling.surface->isSleeping())
				metrics.motionSettlingSlept = 1;
		}

		metrics.motionControlStayedAwake =
			controlStayedAwake ? 1u : 0u;
		metrics.motionControlFinalSpeed =
			getMotionSurfaceMaxSpeed(control);
		metrics.motionSettlingFinalSpeed =
			getMotionSurfaceMaxSpeed(settling);
		metrics.finalMaxSpeed = PxMax(
			metrics.motionControlFinalSpeed,
			metrics.motionSettlingFinalSpeed);
		metrics.finalDynamicCentroidY =
			getMotionSurfaceCentroid(control).y;
		metrics.finalMinY = PX_MAX_F32;
		for(PxU32 i = 0; i < 4; ++i)
			metrics.finalMinY = PxMin(
				metrics.finalMinY,
				PxMin(
					control.positions[i].y,
					settling.positions[i].y));
		metrics.pinnedStable = 1;
		const PxBounds3 controlBounds =
			control.surface->getWorldBounds();
		const PxBounds3 settlingBounds =
			settling.surface->getWorldBounds();
		metrics.boundsFinite =
			!controlBounds.isEmpty() &&
			controlBounds.minimum.isFinite() &&
			controlBounds.maximum.isFinite() &&
			!settlingBounds.isEmpty() &&
			settlingBounds.minimum.isFinite() &&
			settlingBounds.maximum.isFinite() ? 1u : 0u;
		success = samplesFinite &&
			metrics.actorCreated && metrics.shapeAttached &&
			metrics.hostBuffersInitialized && metrics.actorAdded &&
			metrics.actorRemoved && metrics.actorReadded &&
			metrics.fetchFailures == 0 &&
			metrics.nonFiniteSamples == 0 &&
			metrics.pinnedStable && metrics.dynamicMoved &&
			metrics.boundsFinite &&
			metrics.motionMaxVelocityBounded &&
			metrics.motionSettlingApplied &&
			metrics.motionSettlingSlept &&
			metrics.motionControlStayedAwake &&
			metrics.motionSettlingFinalSpeed <= 1.0e-6f &&
			metrics.motionControlFinalSpeed >= 0.07f;
	}
	while(false);

	releaseMotionSurfaceFixture(settling);
	releaseMotionSurfaceFixture(control);
	PX_RELEASE(scene);
	PX_RELEASE(dispatcher);
	if(extensionsInitialized)
		PxCloseExtensions();
	PX_RELEASE(physics);
	PX_RELEASE(foundation);
	metrics.cleanupComplete = 1;
	return success;
}

static bool runMaxDepenetrationVelocityCase(
	const Snippets::HeadlessOptions& options,
	Snippets::TrackingErrorCallback& errorCallback,
	Metrics& metrics)
{
	const PxReal limitedVelocity = 0.12f;
	PxDefaultAllocator allocator;
	PxFoundation* foundation = PxCreateFoundation(
		PX_PHYSICS_VERSION, allocator, errorCallback);
	if(!foundation)
		return false;

	PxPhysics* physics = NULL;
	PxDefaultCpuDispatcher* dispatcher = NULL;
	PxScene* scene = NULL;
	PxMaterial* rigidMaterial = NULL;
	PxRigidStatic* ground = NULL;
	MotionSurfaceFixture limited;
	MotionSurfaceFixture control;
	bool extensionsInitialized = false;
	bool success = false;

	do
	{
		if(options.frames < 8)
			break;
		const PxTolerancesScale scale;
		physics = PxCreatePhysics(
			PX_PHYSICS_VERSION, *foundation, scale, true, NULL);
		if(!physics)
			break;
		extensionsInitialized = PxInitExtensions(*physics, NULL);
		if(!extensionsInitialized)
			break;

		PxSceneDesc sceneDesc(scale);
		sceneDesc.gravity = PxVec3(0.0f);
		sceneDesc.solverType = PxSolverType::eAVBD;
		sceneDesc.filterShader = PxDefaultSimulationFilterShader;
		dispatcher = PxDefaultCpuDispatcherCreate(
			options.dispatcherThreads);
		if(!dispatcher)
			break;
		sceneDesc.cpuDispatcher = dispatcher;
		scene = physics->createScene(sceneDesc);
		if(!scene)
			break;

		rigidMaterial = physics->createMaterial(
			0.0f, 0.0f, 0.0f);
		if(!rigidMaterial)
			break;
		ground = PxCreatePlane(
			*physics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f),
			*rigidMaterial);
		if(!ground || !scene->addActor(*ground))
			break;

		if(!createMotionSurfaceFixture(
				*physics, *scene, scale,
				PxVec3(-3.0f, -0.05f, 0.0f),
				PxVec3(0.0f), PX_MAX_F32,
				0.0f, 0.0f, 0.0f, 1.0f, limited) ||
			!createMotionSurfaceFixture(
				*physics, *scene, scale,
				PxVec3(3.0f, -0.05f, 0.0f),
				PxVec3(0.0f), PX_MAX_F32,
				0.0f, 0.0f, 0.0f, 1.0f, control))
			break;

		limited.surface->setMaxDepenetrationVelocity(
			limitedVelocity);
		control.surface->setMaxDepenetrationVelocity(
			PX_MAX_F32);
		metrics.depenetrationLimitApplied =
			PxAbs(
				limited.surface->getMaxDepenetrationVelocity() -
				limitedVelocity) <= 1.0e-6f &&
			control.surface->getMaxDepenetrationVelocity() >
				1.0e20f ? 1u : 0u;
		metrics.actorCreated = 1;
		metrics.shapeAttached = 1;
		metrics.hostBuffersInitialized = 1;
		metrics.actorAdded = 1;
		metrics.groundAdded = 1;

		const PxReal limitedInitialY =
			getMotionSurfaceCentroid(limited).y;
		const PxReal controlInitialY =
			getMotionSurfaceCentroid(control).y;
		bool finite = true;
		for(PxU32 frame = 0; frame < options.frames; ++frame)
		{
			scene->simulate(options.dt);
			if(!scene->fetchResults(true))
			{
				++metrics.fetchFailures;
				break;
			}
			for(PxU32 i = 0; i < 4; ++i)
			{
				if(!limited.positions[i].isFinite() ||
					!limited.velocities[i].isFinite() ||
					!control.positions[i].isFinite() ||
					!control.velocities[i].isFinite())
				{
					++metrics.nonFiniteSamples;
					finite = false;
				}
			}
			const PxReal limitedRise =
				getMotionSurfaceCentroid(limited).y -
				limitedInitialY;
			const PxReal controlRise =
				getMotionSurfaceCentroid(control).y -
				controlInitialY;
			const PxReal limitedSpeed =
				getMotionSurfaceMaxSpeed(limited);
			metrics.depenetrationLimitedMaxSpeed = PxMax(
				metrics.depenetrationLimitedMaxSpeed,
				limitedSpeed);
			if(frame == 0)
			{
				metrics.depenetrationLimitedFirstStepRise =
					limitedRise;
				metrics.depenetrationControlFirstStepRise =
					controlRise;
				metrics.depenetrationFirstStepBounded =
					limitedRise >= -1.0e-6f &&
					limitedRise <=
						limitedVelocity * options.dt * 1.05f &&
					limitedSpeed <= limitedVelocity * 1.05f
						? 1u : 0u;
				metrics.depenetrationControlSeparated =
					controlRise >
						limitedRise + 5.0e-3f
						? 1u : 0u;
			}
			metrics.depenetrationLimitedFinalRise = limitedRise;
		}
		metrics.depenetrationGradualRecovery =
			metrics.depenetrationLimitedFinalRise >
				metrics.depenetrationLimitedFirstStepRise +
					4.0e-3f ? 1u : 0u;
		metrics.dynamicMoved =
			metrics.depenetrationLimitedFinalRise > 1.0e-2f
				? 1u : 0u;
		metrics.pinnedStable = 1;
		metrics.boundsFinite =
			limited.surface->getWorldBounds().isValid() &&
			control.surface->getWorldBounds().isValid() ? 1u : 0u;
		metrics.initialDynamicCentroidY = limitedInitialY;
		metrics.finalDynamicCentroidY =
			getMotionSurfaceCentroid(limited).y;
		metrics.maxDynamicDisplacement =
			metrics.depenetrationLimitedFinalRise;
		metrics.minY = limitedInitialY;
		metrics.finalMinY = metrics.finalDynamicCentroidY;
		metrics.maxSpeed =
			metrics.depenetrationLimitedMaxSpeed;
		metrics.finalMaxSpeed =
			getMotionSurfaceMaxSpeed(limited);

		success = finite &&
			metrics.depenetrationLimitApplied &&
			metrics.depenetrationFirstStepBounded &&
			metrics.depenetrationControlSeparated &&
			metrics.depenetrationGradualRecovery &&
			metrics.fetchFailures == 0 &&
			metrics.nonFiniteSamples == 0 &&
			metrics.boundsFinite;
	}
	while(false);

	releaseMotionSurfaceFixture(control);
	releaseMotionSurfaceFixture(limited);
	PX_RELEASE(ground);
	PX_RELEASE(rigidMaterial);
	PX_RELEASE(scene);
	PX_RELEASE(dispatcher);
	if(extensionsInitialized)
		PxCloseExtensions();
	PX_RELEASE(physics);
	PX_RELEASE(foundation);
	metrics.cleanupComplete = 1;
	return success;
}

static PxReal getCapsuleSignedSeparation(
	const PxVec3& point, const PxTransform& capsulePose,
	PxReal radius, PxReal halfHeight)
{
	const PxVec3 localPoint = capsulePose.transformInv(point);
	const PxVec3 medialPoint(
		PxClamp(localPoint.x, -halfHeight, halfHeight),
		0.0f, 0.0f);
	return (localPoint - medialPoint).magnitude() - radius;
}

static bool runSpeculativeCcdCase(
	const Snippets::HeadlessOptions& options,
	Snippets::TrackingErrorCallback& errorCallback,
	Metrics& metrics)
{
	const bool planeCase =
		options.caseName == "surface-plane-speculative-ccd";
	const bool sphereCase =
		options.caseName == "surface-sphere-speculative-ccd";
	const bool capsuleCase =
		options.caseName == "surface-capsule-speculative-ccd";
	const bool convexCase =
		options.caseName == "surface-convex-speculative-ccd";
	const bool finiteGeometryCase =
		sphereCase || capsuleCase || convexCase;
	PxDefaultAllocator allocator;
	PxFoundation* foundation = PxCreateFoundation(
		PX_PHYSICS_VERSION, allocator, errorCallback);
	if(!foundation)
		return false;

	PxPhysics* physics = NULL;
	PxDefaultCpuDispatcher* dispatcher = NULL;
	PxScene* scene = NULL;
	PxMaterial* rigidMaterial = NULL;
	PxConvexMesh* convexMesh = NULL;
	PxConvexMeshGeometry convexGeometry;
	PxRigidStatic* obstacles[8] =
		{NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
	PxU32 obstacleCount = 0;
	MotionSurfaceFixture speculative;
	MotionSurfaceFixture discrete;
	bool extensionsInitialized = false;
	bool success = false;

	do
	{
		const PxTolerancesScale scale;
		physics = PxCreatePhysics(
			PX_PHYSICS_VERSION, *foundation, scale, true, NULL);
		if(!physics)
			break;
		extensionsInitialized = PxInitExtensions(*physics, NULL);
		if(!extensionsInitialized)
			break;

		PxSceneDesc sceneDesc(scale);
		sceneDesc.gravity = PxVec3(0.0f);
		sceneDesc.solverType = PxSolverType::eAVBD;
		sceneDesc.filterShader = PxDefaultSimulationFilterShader;
		dispatcher = PxDefaultCpuDispatcherCreate(
			options.dispatcherThreads);
		if(!dispatcher)
			break;
		sceneDesc.cpuDispatcher = dispatcher;
		scene = physics->createScene(sceneDesc);
		if(!scene)
			break;

		rigidMaterial = physics->createMaterial(
			0.0f, 0.0f, 0.0f);
		if(!rigidMaterial)
			break;
		if(convexCase)
		{
			const PxVec3 convexVertices[] =
			{
				PxVec3(-0.15f, -0.15f, -0.15f),
				PxVec3(0.15f, -0.15f, -0.15f),
				PxVec3(0.15f, -0.15f, 0.15f),
				PxVec3(-0.15f, -0.15f, 0.15f),
				PxVec3(-0.15f, 0.15f, -0.15f),
				PxVec3(0.15f, 0.15f, -0.15f),
				PxVec3(0.15f, 0.15f, 0.15f),
				PxVec3(-0.15f, 0.15f, 0.15f)
			};
			convexMesh = createAvbdTestConvexMesh(
				*physics, scale, convexVertices,
				sizeof(convexVertices) /
					sizeof(convexVertices[0]));
			if(!convexMesh)
				break;
			convexGeometry = PxConvexMeshGeometry(convexMesh);
		}
		if(finiteGeometryCase)
		{
			const PxReal obstacleX[] =
				{-2.0f, -1.0f, -2.0f, -1.0f,
				  1.0f,  2.0f,  1.0f,  2.0f};
			const PxReal obstacleZ[] =
				{0.0f, 0.0f, 1.0f, 1.0f,
				 0.0f, 0.0f, 1.0f, 1.0f};
			for(PxU32 i = 0;
				i < sizeof(obstacleX) /
					sizeof(obstacleX[0]); ++i)
			{
				const PxTransform obstaclePose(PxVec3(
					obstacleX[i], 0.25f, obstacleZ[i]));
				PxRigidStatic* obstacle = convexCase
					? PxCreateStatic(
						*physics, obstaclePose,
						convexGeometry, *rigidMaterial)
					: capsuleCase
					? PxCreateStatic(
						*physics, obstaclePose,
						PxCapsuleGeometry(0.3f, 0.2f),
						*rigidMaterial)
					: PxCreateStatic(
						*physics, obstaclePose,
						PxSphereGeometry(0.3f),
						*rigidMaterial);
				if(!obstacle || !scene->addActor(*obstacle))
				{
					PX_RELEASE(obstacle);
					break;
				}
				obstacles[obstacleCount++] = obstacle;
			}
			if(obstacleCount !=
				sizeof(obstacleX) / sizeof(obstacleX[0]))
				break;
		}
		else
		{
			PxRigidStatic* obstacle = planeCase
				? PxCreatePlane(
					*physics,
					PxPlane(0.0f, 1.0f, 0.0f, -0.5f),
					*rigidMaterial)
				: PxCreateStatic(
					*physics,
					PxTransform(PxVec3(0.0f, 0.5f, 0.5f)),
					PxBoxGeometry(4.0f, 0.05f, 2.0f),
					*rigidMaterial);
			if(!obstacle || !scene->addActor(*obstacle))
			{
				PX_RELEASE(obstacle);
				break;
			}
			obstacles[obstacleCount++] = obstacle;
		}

		const PxVec3 initialVelocity(
			0.0f, finiteGeometryCase ? -80.0f : -120.0f, 0.0f);
		if(!createMotionSurfaceFixture(
				*physics, *scene, scale,
				PxVec3(-2.0f, 1.2f, 0.0f),
				initialVelocity, PX_MAX_F32,
				0.0f, 0.0f, 0.0f, 1.0f, speculative) ||
			!createMotionSurfaceFixture(
				*physics, *scene, scale,
				PxVec3(1.0f, 1.2f, 0.0f),
				initialVelocity, PX_MAX_F32,
				0.0f, 0.0f, 0.0f, 1.0f, discrete))
			break;

		speculative.surface->setDeformableBodyFlag(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD, true);
		metrics.speculativeCcdFlagApplied =
			speculative.surface->getDeformableBodyFlags().isSet(
				PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD) &&
			!discrete.surface->getDeformableBodyFlags().isSet(
				PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD)
				? 1u : 0u;
		metrics.actorCreated = 1;
		metrics.shapeAttached = 1;
		metrics.hostBuffersInitialized = 1;
		metrics.actorAdded = 1;
		scene->removeActor(*speculative.surface);
		if(speculative.surface->getScene() != NULL)
			break;
		metrics.actorRemoved = 1;
		if(!scene->addActor(*speculative.surface) ||
			speculative.surface->getScene() != scene)
			break;
		metrics.actorReadded = 1;

		const PxVec3 initialCentroid =
			getMotionSurfaceCentroid(speculative);
		const PxVec3 obstacleCenters[] =
		{
			PxVec3(-2.0f, 0.25f, 0.0f),
			PxVec3(-1.0f, 0.25f, 0.0f),
			PxVec3(-2.0f, 0.25f, 1.0f),
			PxVec3(-1.0f, 0.25f, 1.0f),
			PxVec3(1.0f, 0.25f, 0.0f),
			PxVec3(2.0f, 0.25f, 0.0f),
			PxVec3(1.0f, 0.25f, 1.0f),
			PxVec3(2.0f, 0.25f, 1.0f)
		};
		const PxU32 simulatedFrames = options.frames;
		bool finite = true;
		bool discreteBoundsFinite = false;
		for(PxU32 frame = 0; frame < simulatedFrames; ++frame)
		{
			scene->simulate(options.dt);
			if(!scene->fetchResults(true))
			{
				++metrics.fetchFailures;
				break;
			}
			for(PxU32 i = 0; i < 4; ++i)
			{
				if(!speculative.positions[i].isFinite() ||
					!speculative.velocities[i].isFinite() ||
					(frame < 3 &&
						(!discrete.positions[i].isFinite() ||
						 !discrete.velocities[i].isFinite())))
				{
					++metrics.nonFiniteSamples;
					finite = false;
				}
			}
			for(PxU32 i = 0; i < 4; ++i)
			{
				metrics.speculativeCcdPositiveMinY =
					PxMin(
						metrics.speculativeCcdPositiveMinY,
						speculative.positions[i].y);
				if(finiteGeometryCase)
				{
					for(PxU32 obstacleIndex = 0;
						obstacleIndex <
							sizeof(obstacleCenters) /
								sizeof(obstacleCenters[0]);
						++obstacleIndex)
					{
						PxReal separation = 0.0f;
						if(convexCase)
						{
							const PxReal distanceSq =
								PxGeometryQuery::pointDistance(
									speculative.positions[i].
										getXYZ(),
									convexGeometry,
									PxTransform(
										obstacleCenters[
											obstacleIndex]));
							separation =
								distanceSq >= 0.0f
									? PxSqrt(distanceSq)
									: -PX_MAX_F32;
						}
						else
							separation = capsuleCase
								? getCapsuleSignedSeparation(
									speculative.positions[i].
										getXYZ(),
									PxTransform(
										obstacleCenters[
											obstacleIndex]),
									0.3f, 0.2f)
								: (speculative.positions[i].getXYZ() -
									obstacleCenters[obstacleIndex]).
										magnitude() - 0.3f;
						metrics.
							speculativeCcdPositiveMinSeparation =
								PxMin(
									metrics.
										speculativeCcdPositiveMinSeparation,
									separation);
					}
				}
			}
			if(frame < 3)
			{
				PxReal negativeFrameMaxY = -PX_MAX_F32;
				for(PxU32 i = 0; i < 4; ++i)
					negativeFrameMaxY = PxMax(
						negativeFrameMaxY,
						discrete.positions[i].y);
				if(!planeCase || frame == 0)
					metrics.speculativeCcdNegativeMaxY =
						negativeFrameMaxY;
				discreteBoundsFinite =
					discrete.surface->getWorldBounds().isValid();
				if(frame == 2 &&
					discrete.surface->getScene() == scene)
					scene->removeActor(*discrete.surface);
			}
		}

		metrics.speculativeCcdPreventedTunneling =
			(finiteGeometryCase
				? metrics.speculativeCcdPositiveMinSeparation >=
					(convexCase ? 1.0e-3f : -0.05f) &&
					metrics.speculativeCcdPositiveMinSeparation <
						PX_MAX_F32
				: metrics.speculativeCcdPositiveMinY >=
					(planeCase ? 0.49f : 0.54f))
				? 1u : 0u;
		metrics.speculativeCcdNegativeControlTunneled =
			metrics.speculativeCcdNegativeMaxY <=
				(planeCase ? 0.45f : 0.44f)
				? 1u : 0u;
		metrics.pinnedStable = 1;
		metrics.maxPinnedDrift = 0.0f;
		const PxVec3 finalCentroid =
			getMotionSurfaceCentroid(speculative);
		metrics.initialDynamicCentroidY = initialCentroid.y;
		metrics.finalDynamicCentroidY = finalCentroid.y;
		metrics.maxDynamicDisplacement =
			(finalCentroid - initialCentroid).magnitude();
		metrics.dynamicMoved =
			metrics.maxDynamicDisplacement > 1.0e-2f ? 1u : 0u;
		metrics.minY = PxMin(
			metrics.speculativeCcdPositiveMinY,
			metrics.speculativeCcdNegativeMaxY);
		metrics.finalMinY = finalCentroid.y;
		metrics.maxSpeed = PxMax(
			getMotionSurfaceMaxSpeed(speculative),
			getMotionSurfaceMaxSpeed(discrete));
		metrics.finalMaxSpeed =
			getMotionSurfaceMaxSpeed(speculative);
		const PxBounds3 speculativeBounds =
			speculative.surface->getWorldBounds();
		metrics.boundsFinite =
			speculativeBounds.isValid() &&
			discreteBoundsFinite ? 1u : 0u;

		success = finite &&
			metrics.speculativeCcdFlagApplied &&
			metrics.speculativeCcdPreventedTunneling &&
			(planeCase ||
				metrics.speculativeCcdNegativeControlTunneled) &&
			metrics.fetchFailures == 0 &&
			metrics.nonFiniteSamples == 0 &&
			metrics.boundsFinite;
	}
	while(false);

	releaseMotionSurfaceFixture(discrete);
	releaseMotionSurfaceFixture(speculative);
	for(PxU32 i = 0; i < obstacleCount; ++i)
		PX_RELEASE(obstacles[i]);
	PX_RELEASE(convexMesh);
	PX_RELEASE(rigidMaterial);
	PX_RELEASE(scene);
	PX_RELEASE(dispatcher);
	if(extensionsInitialized)
		PxCloseExtensions();
	PX_RELEASE(physics);
	PX_RELEASE(foundation);
	metrics.cleanupComplete = 1;
	return success;
}

static bool runMovingKinematicFiniteSpeculativeCcdCase(
	const Snippets::HeadlessOptions& options,
	Snippets::TrackingErrorCallback& errorCallback,
	Metrics& metrics)
{
	const bool rotationalCapsuleCase =
		options.caseName ==
			"surface-rotating-kinematic-capsule-speculative-ccd";
	const bool rotationalConvexCase =
		options.caseName ==
			"surface-rotating-kinematic-convex-speculative-ccd";
	const bool rotationalFiniteCase =
		rotationalCapsuleCase || rotationalConvexCase;
	const bool capsuleCase =
		rotationalCapsuleCase ||
		options.caseName ==
			"surface-moving-kinematic-capsule-speculative-ccd";
	const bool convexCase =
		rotationalConvexCase ||
		options.caseName ==
			"surface-moving-kinematic-convex-speculative-ccd";
	PxDefaultAllocator allocator;
	PxFoundation* foundation = PxCreateFoundation(
		PX_PHYSICS_VERSION, allocator, errorCallback);
	if(!foundation)
		return false;

	PxPhysics* physics = NULL;
	PxDefaultCpuDispatcher* dispatcher = NULL;
	PxScene* scene = NULL;
	PxMaterial* rigidMaterial = NULL;
	PxRigidDynamic* spheres[2] = {NULL, NULL};
	PxConvexMesh* convexMesh = NULL;
	PxConvexMeshGeometry convexGeometry;
	MotionSurfaceFixture speculative;
	MotionSurfaceFixture discrete;
	bool extensionsInitialized = false;
	bool success = false;

	do
	{
		const PxTolerancesScale scale;
		physics = PxCreatePhysics(
			PX_PHYSICS_VERSION, *foundation, scale, true, NULL);
		if(!physics)
			break;
		extensionsInitialized = PxInitExtensions(*physics, NULL);
		if(!extensionsInitialized)
			break;

		PxSceneDesc sceneDesc(scale);
		sceneDesc.gravity = PxVec3(0.0f);
		sceneDesc.solverType = PxSolverType::eAVBD;
		sceneDesc.filterShader = PxDefaultSimulationFilterShader;
		dispatcher = PxDefaultCpuDispatcherCreate(
			options.dispatcherThreads);
		if(!dispatcher)
			break;
		sceneDesc.cpuDispatcher = dispatcher;
		scene = physics->createScene(sceneDesc);
		if(!scene)
			break;

		rigidMaterial = physics->createMaterial(
			0.0f, 0.0f, 0.0f);
		if(!rigidMaterial)
			break;
		if(convexCase)
		{
			const PxVec3 translationConvexVertices[] =
			{
				PxVec3(-0.8f, -0.3f, -0.8f),
				PxVec3(0.8f, -0.3f, -0.8f),
				PxVec3(0.8f, -0.3f, 0.8f),
				PxVec3(-0.8f, -0.3f, 0.8f),
				PxVec3(-0.8f, 0.3f, -0.8f),
				PxVec3(0.8f, 0.3f, -0.8f),
				PxVec3(0.8f, 0.3f, 0.8f),
				PxVec3(-0.8f, 0.3f, 0.8f)
			};
			const PxVec3 rotationalConvexVertices[] =
			{
				PxVec3(-1.0f, -0.1f, -0.1f),
				PxVec3(1.0f, -0.1f, -0.1f),
				PxVec3(1.0f, -0.1f, 0.1f),
				PxVec3(-1.0f, -0.1f, 0.1f),
				PxVec3(-1.0f, 0.1f, -0.1f),
				PxVec3(1.0f, 0.1f, -0.1f),
				PxVec3(1.0f, 0.1f, 0.1f),
				PxVec3(-1.0f, 0.1f, 0.1f)
			};
			const PxVec3* convexVertices =
				rotationalConvexCase
					? rotationalConvexVertices
					: translationConvexVertices;
			const PxU32 convexVertexCount = 8;
			convexMesh = createAvbdTestConvexMesh(
				*physics, scale, convexVertices,
				convexVertexCount);
			if(!convexMesh)
				break;
			convexGeometry = PxConvexMeshGeometry(convexMesh);
		}

		const PxVec3 surfaceOffsets[2] =
		{
			PxVec3(-2.0f, 0.0f, 0.0f),
			PxVec3(1.0f, 0.0f, 0.0f)
		};
		if(!createMotionSurfaceFixture(
				*physics, *scene, scale, surfaceOffsets[0],
				PxVec3(0.0f), PX_MAX_F32,
				0.0f, 0.0f, 0.0f, 1.0f, speculative) ||
			!createMotionSurfaceFixture(
				*physics, *scene, scale, surfaceOffsets[1],
				PxVec3(0.0f), PX_MAX_F32,
				0.0f, 0.0f, 0.0f, 0.0f, discrete))
			break;

		speculative.surface->setDeformableBodyFlag(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD, true);
		metrics.speculativeCcdFlagApplied =
			speculative.surface->getDeformableBodyFlags().isSet(
				PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD) &&
			!discrete.surface->getDeformableBodyFlags().isSet(
				PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD)
				? 1u : 0u;

		const PxReal radius =
			rotationalCapsuleCase ? 0.1f : 0.8f;
		const PxReal halfHeight =
			rotationalCapsuleCase ? 1.0f :
				(capsuleCase ? 0.3f : 0.0f);
		const PxReal startY =
			rotationalFiniteCase ? 0.0f : 1.1f;
		const PxReal targetY =
			rotationalFiniteCase ? 0.0f : -1.1f;
		const PxReal rotationalAngle = 1.0471975512f;
		const PxQuat rotationalStart(
			-rotationalAngle, PxVec3(0.0f, 0.0f, 1.0f));
		const PxQuat rotationalEnd(
			rotationalAngle, PxVec3(0.0f, 0.0f, 1.0f));
		for(PxU32 i = 0; i < 2; ++i)
		{
			const PxVec3 center = rotationalFiniteCase
				? PxVec3(
					surfaceOffsets[i].x - 0.9f,
					startY, surfaceOffsets[i].z)
				: PxVec3(
					surfaceOffsets[i].x + 0.5f,
					startY,
					surfaceOffsets[i].z + 0.5f);
			spheres[i] =
				physics->createRigidDynamic(PxTransform(
					center,
					rotationalFiniteCase
						? rotationalStart : PxQuat(PxIdentity)));
			const bool shapeCreated = spheres[i] &&
				(convexCase
					? PxRigidActorExt::createExclusiveShape(
						*spheres[i], convexGeometry,
						*rigidMaterial)
					: capsuleCase
					? PxRigidActorExt::createExclusiveShape(
						*spheres[i],
						PxCapsuleGeometry(radius, halfHeight),
						*rigidMaterial)
					: PxRigidActorExt::createExclusiveShape(
						*spheres[i], PxSphereGeometry(radius),
						*rigidMaterial));
			if(!spheres[i] || !shapeCreated)
				break;
			spheres[i]->setRigidBodyFlag(
				PxRigidBodyFlag::eKINEMATIC, true);
			if(!scene->addActor(*spheres[i]))
				break;
		}
		if(!spheres[0] || !spheres[1] ||
			spheres[0]->getScene() != scene ||
			spheres[1]->getScene() != scene)
			break;

		metrics.actorCreated = 1;
		metrics.shapeAttached = 1;
		metrics.hostBuffersInitialized = 1;
		metrics.actorAdded = 1;
		scene->removeActor(*speculative.surface);
		if(speculative.surface->getScene() != NULL)
			break;
		metrics.actorRemoved = 1;
		if(!scene->addActor(*speculative.surface) ||
			speculative.surface->getScene() != scene)
			break;
		metrics.actorReadded = 1;

		for(PxU32 i = 0; i < 2; ++i)
		{
			const PxTransform startPose =
				spheres[i]->getGlobalPose();
			spheres[i]->setKinematicTarget(
				rotationalFiniteCase
					? PxTransform(startPose.p, rotationalEnd)
					: PxTransform(PxVec3(
						startPose.p.x, targetY, startPose.p.z)));
		}
		metrics.movingSphereTargetIssued = 1;

		const PxVec3 positiveInitial =
			getMotionSurfaceCentroid(speculative);
		const PxVec3 negativeInitial =
			getMotionSurfaceCentroid(discrete);
		PxReal rotationalEndpointMinSeparation = PX_MAX_F32;
		PxReal rotationalMidMinSeparation = PX_MAX_F32;
		if(rotationalFiniteCase)
		{
			const PxVec3 center =
				spheres[0]->getGlobalPose().p;
			const PxTransform endpointPoses[2] =
			{
				PxTransform(center, rotationalStart),
				PxTransform(center, rotationalEnd)
			};
			const PxTransform midPose(center);
			for(PxU32 vertex = 0; vertex < 4; ++vertex)
			{
				for(PxU32 endpoint = 0; endpoint < 2;
					++endpoint)
				{
					const PxReal separation =
						rotationalCapsuleCase
							? getCapsuleSignedSeparation(
								speculative.initialPositions[vertex],
								endpointPoses[endpoint],
								radius, halfHeight)
							: [&]()
							{
								const PxReal distanceSq =
									PxGeometryQuery::pointDistance(
										speculative.
											initialPositions[vertex],
										convexGeometry,
										endpointPoses[endpoint]);
								return distanceSq >= 0.0f &&
									PxIsFinite(distanceSq)
									? PxSqrt(distanceSq)
									: -PX_MAX_F32;
							}();
					rotationalEndpointMinSeparation = PxMin(
						rotationalEndpointMinSeparation,
						separation);
				}
				const PxReal midSeparation =
					rotationalCapsuleCase
						? getCapsuleSignedSeparation(
							speculative.initialPositions[vertex],
							midPose, radius, halfHeight)
						: [&]()
						{
							const PxReal distanceSq =
								PxGeometryQuery::pointDistance(
									speculative.
										initialPositions[vertex],
									convexGeometry, midPose);
							return distanceSq >= 0.0f &&
								PxIsFinite(distanceSq)
								? PxSqrt(distanceSq)
								: PX_MAX_F32;
						}();
				rotationalMidMinSeparation = PxMin(
					rotationalMidMinSeparation,
					midSeparation);
			}
		}
		const bool rotationalSweepIsolated =
			!rotationalFiniteCase ||
			(rotationalEndpointMinSeparation > 0.05f &&
			 (rotationalCapsuleCase
				? rotationalMidMinSeparation < -0.05f
				: rotationalMidMinSeparation <= 1.0e-5f));
		bool finite = true;
		bool boundsFinite = true;
		for(PxU32 frame = 0; frame < options.frames; ++frame)
		{
			scene->simulate(options.dt);
			if(!scene->fetchResults(true))
			{
				++metrics.fetchFailures;
				break;
			}

			if(frame < 3)
			{
				const PxVec3 positiveCentroid =
					getMotionSurfaceCentroid(speculative);
				const PxVec3 negativeCentroid =
					getMotionSurfaceCentroid(discrete);
				metrics.movingSpherePositiveDisplacement =
					PxMax(
						metrics.movingSpherePositiveDisplacement,
						(positiveCentroid - positiveInitial).
							magnitude());
				metrics.movingSphereNegativeDisplacement =
					PxMax(
						metrics.movingSphereNegativeDisplacement,
						(negativeCentroid - negativeInitial).
							magnitude());

				const PxTransform positiveRigidPose =
					spheres[0]->getGlobalPose();
				for(PxU32 i = 0; i < 4; ++i)
				{
					const PxVec3 position =
						speculative.positions[i].getXYZ();
					const PxVec3 velocity =
						speculative.velocities[i].getXYZ();
					const PxVec3 negativePosition =
						discrete.positions[i].getXYZ();
					const PxVec3 negativeVelocity =
						discrete.velocities[i].getXYZ();
					if(!position.isFinite() ||
						!velocity.isFinite() ||
						!negativePosition.isFinite() ||
						!negativeVelocity.isFinite())
					{
						++metrics.nonFiniteSamples;
						finite = false;
					}
					metrics.movingSpherePositiveMinSeparation =
						PxMin(
							metrics.
								movingSpherePositiveMinSeparation,
							convexCase
								? [&]()
								{
									const PxReal distanceSq =
										PxGeometryQuery::pointDistance(
											position, convexGeometry,
											positiveRigidPose);
									return distanceSq >= 0.0f
										? PxSqrt(distanceSq)
										: -PX_MAX_F32;
								}()
								: capsuleCase
								? getCapsuleSignedSeparation(
									position, positiveRigidPose,
									radius, halfHeight)
								: (position -
									positiveRigidPose.p).
										magnitude() - radius);
				}
			}

			if(frame < 3)
				boundsFinite = boundsFinite &&
					speculative.surface->getWorldBounds().isValid() &&
					discrete.surface->getWorldBounds().isValid();
			if(frame == 2)
			{
				scene->removeActor(*speculative.surface);
				scene->removeActor(*discrete.surface);
			}
		}

		metrics.movingSphereCcdResponseObserved =
			metrics.movingSpherePositiveDisplacement > 0.02f &&
			metrics.movingSpherePositiveMinSeparation >
				(convexCase ? 1.0e-3f : -0.10f) &&
			metrics.movingSpherePositiveMinSeparation <
				PX_MAX_F32 ? 1u : 0u;
		metrics.movingSphereNegativeControlHeld =
			metrics.movingSphereNegativeDisplacement < 5.0e-3f
				? 1u : 0u;
		metrics.speculativeCcdPreventedTunneling =
			metrics.movingSphereCcdResponseObserved;
		metrics.speculativeCcdNegativeControlTunneled =
			metrics.movingSphereNegativeControlHeld;
		metrics.pinnedStable = 1;
		metrics.dynamicMoved =
			metrics.movingSphereCcdResponseObserved;
		metrics.boundsFinite = boundsFinite ? 1u : 0u;
		metrics.maxDynamicDisplacement =
			metrics.movingSpherePositiveDisplacement;
		metrics.initialDynamicCentroidY = positiveInitial.y;
		metrics.finalDynamicCentroidY =
			getMotionSurfaceCentroid(speculative).y;
		metrics.minY = metrics.finalDynamicCentroidY;
		metrics.finalMinY = metrics.finalDynamicCentroidY;
		metrics.maxSpeed = PxMax(
			getMotionSurfaceMaxSpeed(speculative),
			getMotionSurfaceMaxSpeed(discrete));
		metrics.finalMaxSpeed =
			getMotionSurfaceMaxSpeed(speculative);

		success = finite &&
			metrics.speculativeCcdFlagApplied &&
			metrics.movingSphereTargetIssued &&
			metrics.movingSphereCcdResponseObserved &&
			metrics.movingSphereNegativeControlHeld &&
			rotationalSweepIsolated &&
			metrics.fetchFailures == 0 &&
			metrics.nonFiniteSamples == 0 &&
			metrics.boundsFinite;
		if(rotationalFiniteCase)
		{
			std::printf(
				"[%s] "
				"frames=%u target=kinematic owner=forward "
				"responseObserved=%u negativeControlPassed=%u "
				"endpointMinSeparation=%.9g "
				"midSweepMinSeparation=%.9g "
				"positiveDisplacement=%.9g "
				"negativeDisplacement=%.9g result=%s\n",
				rotationalCapsuleCase
					? "AVBD_CAPSULE_ROTATIONAL_SWEPT"
					: "AVBD_CONVEX_ROTATIONAL_SWEPT",
				options.frames,
				metrics.movingSphereCcdResponseObserved,
				metrics.movingSphereNegativeControlHeld,
				double(rotationalEndpointMinSeparation),
				double(rotationalMidMinSeparation),
				double(metrics.movingSpherePositiveDisplacement),
				double(metrics.movingSphereNegativeDisplacement),
				success ? "PASS" : "FAIL");
		}
	}
	while(false);

	for(PxU32 i = 0; i < 2; ++i)
		PX_RELEASE(spheres[i]);
	PX_RELEASE(convexMesh);
	releaseMotionSurfaceFixture(discrete);
	releaseMotionSurfaceFixture(speculative);
	PX_RELEASE(rigidMaterial);
	PX_RELEASE(scene);
	PX_RELEASE(dispatcher);
	if(extensionsInitialized)
		PxCloseExtensions();
	PX_RELEASE(physics);
	PX_RELEASE(foundation);
	metrics.cleanupComplete = 1;
	return success;
}

static bool runDynamicFiniteRelativeSweptCcdCase(
	const Snippets::HeadlessOptions& options,
	Snippets::TrackingErrorCallback& errorCallback,
	Metrics& metrics)
{
	const bool rotationalCapsuleCase =
		options.caseName ==
			"surface-dynamic-rotating-capsule-relative-swept-ccd";
	const bool rotationalConvexCase =
		options.caseName ==
			"surface-dynamic-rotating-convex-relative-swept-ccd";
	const bool rotationalFiniteCase =
		rotationalCapsuleCase || rotationalConvexCase;
	const bool capsuleCase =
		rotationalCapsuleCase ||
		options.caseName ==
			"surface-dynamic-capsule-relative-swept-ccd";
	const bool convexCase =
		rotationalConvexCase ||
		options.caseName ==
			"surface-dynamic-convex-relative-swept-ccd";
	PxDefaultAllocator allocator;
	PxFoundation* foundation = PxCreateFoundation(
		PX_PHYSICS_VERSION, allocator, errorCallback);
	if(!foundation)
		return false;

	PxPhysics* physics = NULL;
	PxDefaultCpuDispatcher* dispatcher = NULL;
	PxScene* scene = NULL;
	PxMaterial* rigidMaterial = NULL;
	PxRigidDynamic* spheres[2] = {NULL, NULL};
	PxConvexMesh* convexMesh = NULL;
	PxConvexMeshGeometry convexGeometry;
	MotionSurfaceFixture speculative;
	MotionSurfaceFixture discrete;
	PxVec3 rotationalInitialPositions[4];
	PxReal rotationalEndpointMinSeparation = PX_MAX_F32;
	PxReal rotationalMidMinSeparation = PX_MAX_F32;
	PxReal rotationalPositiveAngularTravel = 0.0f;
	PxReal rotationalNegativeAngularTravel = 0.0f;
	bool rotationalSweepIsolated = false;
	bool extensionsInitialized = false;
	bool success = false;

	do
	{
		const PxTolerancesScale scale;
		physics = PxCreatePhysics(
			PX_PHYSICS_VERSION, *foundation, scale, true, NULL);
		if(!physics)
			break;
		extensionsInitialized = PxInitExtensions(*physics, NULL);
		if(!extensionsInitialized)
			break;

		PxSceneDesc sceneDesc(scale);
		sceneDesc.gravity = PxVec3(0.0f);
		sceneDesc.solverType = PxSolverType::eAVBD;
		sceneDesc.filterShader = PxDefaultSimulationFilterShader;
		dispatcher = PxDefaultCpuDispatcherCreate(
			options.dispatcherThreads);
		if(!dispatcher)
			break;
		sceneDesc.cpuDispatcher = dispatcher;
		scene = physics->createScene(sceneDesc);
		if(!scene)
			break;

		rigidMaterial = physics->createMaterial(
			0.0f, 0.0f, 0.0f);
		if(!rigidMaterial)
			break;
		if(convexCase)
		{
			const PxVec3 translationConvexVertices[] =
			{
				PxVec3(-0.8f, -0.3f, -0.8f),
				PxVec3(0.8f, -0.3f, -0.8f),
				PxVec3(0.8f, -0.3f, 0.8f),
				PxVec3(-0.8f, -0.3f, 0.8f),
				PxVec3(-0.8f, 0.3f, -0.8f),
				PxVec3(0.8f, 0.3f, -0.8f),
				PxVec3(0.8f, 0.3f, 0.8f),
				PxVec3(-0.8f, 0.3f, 0.8f)
			};
			const PxVec3 rotationalConvexVertices[] =
			{
				PxVec3(-1.0f, -0.1f, -0.1f),
				PxVec3(1.0f, -0.1f, -0.1f),
				PxVec3(1.0f, -0.1f, 0.1f),
				PxVec3(-1.0f, -0.1f, 0.1f),
				PxVec3(-1.0f, 0.1f, -0.1f),
				PxVec3(1.0f, 0.1f, -0.1f),
				PxVec3(1.0f, 0.1f, 0.1f),
				PxVec3(-1.0f, 0.1f, 0.1f)
			};
			const PxVec3* convexVertices =
				rotationalConvexCase
					? rotationalConvexVertices
					: translationConvexVertices;
			convexMesh = createAvbdTestConvexMesh(
				*physics, scale, convexVertices,
				8);
			if(!convexMesh)
				break;
			convexGeometry = PxConvexMeshGeometry(convexMesh);
		}

		const PxVec3 surfaceOffsets[2] =
		{
			PxVec3(-2.0f, 0.0f, 0.0f),
			PxVec3(1.0f, 0.0f, 0.0f)
		};
		if(!createMotionSurfaceFixture(
				*physics, *scene, scale, surfaceOffsets[0],
				PxVec3(0.0f), PX_MAX_F32,
				0.0f, 0.0f, 0.0f, 1.0f, speculative) ||
			!createMotionSurfaceFixture(
				*physics, *scene, scale, surfaceOffsets[1],
				PxVec3(0.0f), PX_MAX_F32,
				0.0f, 0.0f, 0.0f, 1.0f, discrete))
			break;
		if(rotationalFiniteCase)
			for(PxU32 i = 0; i < 4; ++i)
				rotationalInitialPositions[i] =
					speculative.positions[i].getXYZ();

		speculative.surface->setDeformableBodyFlag(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD, true);
		metrics.speculativeCcdFlagApplied =
			speculative.surface->getDeformableBodyFlags().isSet(
				PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD) &&
			!discrete.surface->getDeformableBodyFlags().isSet(
				PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD)
				? 1u : 0u;

		const PxReal radius =
			rotationalCapsuleCase ? 0.1f : 0.8f;
		const PxReal halfHeight =
			rotationalCapsuleCase ? 1.0f :
				(capsuleCase ? 0.3f : 0.0f);
		const PxReal startY =
			rotationalFiniteCase ? 0.0f : 1.1f;
		const PxReal launchSpeed =
			rotationalFiniteCase ? 0.0f : -132.0f;
		const PxQuat rotationalStart(
			-PxPi * 5.0f / 18.0f,
			PxVec3(0.0f, 0.0f, 1.0f));
		const PxReal rotationalAngularSpeed =
			(PxPi * 10.0f / 18.0f) / options.dt;
		for(PxU32 i = 0; i < 2; ++i)
		{
			const PxVec3 center =
				rotationalFiniteCase
					? PxVec3(
						surfaceOffsets[i].x - 0.9f,
						0.0f, surfaceOffsets[i].z)
					: PxVec3(
						surfaceOffsets[i].x + 0.5f,
						startY,
						surfaceOffsets[i].z + 0.5f);
			spheres[i] =
				physics->createRigidDynamic(
					PxTransform(
						center,
						rotationalFiniteCase
							? rotationalStart
							: PxQuat(PxIdentity)));
			const bool shapeCreated = spheres[i] &&
				(convexCase
					? PxRigidActorExt::createExclusiveShape(
						*spheres[i], convexGeometry,
						*rigidMaterial)
					: capsuleCase
					? PxRigidActorExt::createExclusiveShape(
						*spheres[i],
						PxCapsuleGeometry(radius, halfHeight),
						*rigidMaterial)
					: PxRigidActorExt::createExclusiveShape(
						*spheres[i], PxSphereGeometry(radius),
						*rigidMaterial));
			if(!spheres[i] || !shapeCreated ||
				!PxRigidBodyExt::setMassAndUpdateInertia(
					*spheres[i], 1.0f))
				break;
			spheres[i]->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			PxRigidDynamicLockFlags lockFlags =
				PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
				PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
				PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
				PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y;
			lockFlags |= rotationalFiniteCase
				? PxRigidDynamicLockFlag::eLOCK_LINEAR_Y
				: PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z;
			spheres[i]->setRigidDynamicLockFlags(lockFlags);
			spheres[i]->setSolverIterationCounts(16, 1);
			if(rotationalFiniteCase)
			{
				spheres[i]->setMaxAngularVelocity(200.0f);
				spheres[i]->setAngularVelocity(
					PxVec3(
						0.0f, 0.0f,
						rotationalAngularSpeed));
			}
			else
			{
				spheres[i]->setLinearVelocity(
					PxVec3(0.0f, launchSpeed, 0.0f));
			}
			if(!scene->addActor(*spheres[i]))
				break;
		}
		if(!spheres[0] || !spheres[1] ||
			spheres[0]->getScene() != scene ||
			spheres[1]->getScene() != scene)
			break;
		metrics.dynamicSphereSweepLaunched = 1;

		metrics.actorCreated = 1;
		metrics.shapeAttached = 1;
		metrics.hostBuffersInitialized = 1;
		metrics.actorAdded = 1;
		scene->removeActor(*speculative.surface);
		if(speculative.surface->getScene() != NULL)
			break;
		metrics.actorRemoved = 1;
		if(!scene->addActor(*speculative.surface) ||
			speculative.surface->getScene() != scene)
			break;
		metrics.actorReadded = 1;

		const PxVec3 positiveInitial =
			getMotionSurfaceCentroid(speculative);
		const PxVec3 negativeInitial =
			getMotionSurfaceCentroid(discrete);
		bool finite = true;
		bool boundsFinite = true;
		for(PxU32 frame = 0; frame < options.frames; ++frame)
		{
			scene->simulate(options.dt);
			if(!scene->fetchResults(true))
			{
				++metrics.fetchFailures;
				break;
			}

			if(frame == 0)
			{
				const PxVec3 positiveCentroid =
					getMotionSurfaceCentroid(speculative);
				const PxVec3 negativeCentroid =
					getMotionSurfaceCentroid(discrete);
				const PxTransform positiveRigidPose =
					spheres[0]->getGlobalPose();
				const PxTransform negativeRigidPose =
					spheres[1]->getGlobalPose();
				metrics.
					dynamicSphereSweepPositiveSoftDisplacement =
						(positiveCentroid - positiveInitial).
							magnitude();
				metrics.
					dynamicSphereSweepNegativeSoftDisplacement =
						(negativeCentroid - negativeInitial).
							magnitude();
				if(rotationalFiniteCase)
				{
					auto getAngularTravel = [&](
						const PxQuat& endRotation)
					{
						const PxReal alignment = PxClamp(
							PxAbs(
								rotationalStart.dot(
									endRotation.getNormalized())),
							0.0f, 1.0f);
						return 2.0f * PxAcos(alignment);
					};
					rotationalPositiveAngularTravel =
						getAngularTravel(positiveRigidPose.q);
					rotationalNegativeAngularTravel =
						getAngularTravel(negativeRigidPose.q);
					metrics.dynamicSphereSweepPositiveRigidDrop =
						rotationalPositiveAngularTravel;
					metrics.dynamicSphereSweepNegativeRigidDrop =
						rotationalNegativeAngularTravel;

					const PxTransform startPose(
						PxVec3(
							surfaceOffsets[0].x - 0.9f,
							0.0f, surfaceOffsets[0].z),
						rotationalStart);
					const PxTransform endPose(
						startPose.p,
						negativeRigidPose.q.getNormalized());
					for(PxU32 i = 0; i < 4; ++i)
					{
						const PxReal startSeparation =
							rotationalCapsuleCase
								? getCapsuleSignedSeparation(
									rotationalInitialPositions[i],
									startPose, radius, halfHeight)
								: [&]()
								{
									const PxReal distanceSq =
										PxGeometryQuery::pointDistance(
											rotationalInitialPositions[i],
											convexGeometry, startPose);
									return distanceSq >= 0.0f &&
										PxIsFinite(distanceSq)
										? PxSqrt(distanceSq)
										: -PX_MAX_F32;
								}();
						const PxReal endSeparation =
							rotationalCapsuleCase
								? getCapsuleSignedSeparation(
									rotationalInitialPositions[i],
									endPose, radius, halfHeight)
								: [&]()
								{
									const PxReal distanceSq =
										PxGeometryQuery::pointDistance(
											rotationalInitialPositions[i],
											convexGeometry, endPose);
									return distanceSq >= 0.0f &&
										PxIsFinite(distanceSq)
										? PxSqrt(distanceSq)
										: -PX_MAX_F32;
								}();
						rotationalEndpointMinSeparation = PxMin(
							rotationalEndpointMinSeparation,
							PxMin(startSeparation, endSeparation));
						for(PxU32 sample = 1; sample < 64; ++sample)
						{
							const PxReal time =
								PxReal(sample) / 64.0f;
							const PxTransform samplePose(
								startPose.p,
								PxSlerp(
									time, rotationalStart,
									negativeRigidPose.q.
										getNormalized()).
											getNormalized());
							const PxReal sampleSeparation =
								rotationalCapsuleCase
									? getCapsuleSignedSeparation(
										rotationalInitialPositions[i],
										samplePose,
										radius, halfHeight)
									: [&]()
									{
										const PxReal distanceSq =
											PxGeometryQuery::pointDistance(
												rotationalInitialPositions[i],
												convexGeometry, samplePose);
										return distanceSq >= 0.0f &&
											PxIsFinite(distanceSq)
											? PxSqrt(distanceSq)
											: PX_MAX_F32;
									}();
							rotationalMidMinSeparation = PxMin(
								rotationalMidMinSeparation,
								sampleSeparation);
						}
					}
					rotationalSweepIsolated =
						PxIsFinite(
							rotationalEndpointMinSeparation) &&
						PxIsFinite(rotationalMidMinSeparation) &&
						rotationalEndpointMinSeparation > 0.05f &&
						(rotationalCapsuleCase
							? rotationalMidMinSeparation < -0.05f
							: rotationalMidMinSeparation <= 1.0e-5f);
				}
				else
				{
					metrics.dynamicSphereSweepPositiveRigidDrop =
						startY - positiveRigidPose.p.y;
					metrics.dynamicSphereSweepNegativeRigidDrop =
						startY - negativeRigidPose.p.y;
				}

				for(PxU32 i = 0; i < 4; ++i)
				{
					const PxVec3 position =
						speculative.positions[i].getXYZ();
					const PxVec3 velocity =
						speculative.velocities[i].getXYZ();
					const PxVec3 negativePosition =
						discrete.positions[i].getXYZ();
					const PxVec3 negativeVelocity =
						discrete.velocities[i].getXYZ();
					if(!position.isFinite() ||
						!velocity.isFinite() ||
						!negativePosition.isFinite() ||
						!negativeVelocity.isFinite())
					{
						++metrics.nonFiniteSamples;
						finite = false;
					}
					metrics.
						dynamicSphereSweepPositiveMinSeparation =
							PxMin(
								metrics.
									dynamicSphereSweepPositiveMinSeparation,
								convexCase
									? [&]()
									{
										const PxReal distanceSq =
											PxGeometryQuery::pointDistance(
												position, convexGeometry,
												positiveRigidPose);
										return distanceSq >= 0.0f
											? PxSqrt(distanceSq)
											: -PX_MAX_F32;
									}()
									: capsuleCase
									? getCapsuleSignedSeparation(
										position, positiveRigidPose,
										radius, halfHeight)
									: (position -
										positiveRigidPose.p).
											magnitude() - radius);
				}

				boundsFinite =
					speculative.surface->getWorldBounds().isValid() &&
					discrete.surface->getWorldBounds().isValid();
				metrics.dynamicSphereSweepResponseObserved =
					metrics.
						dynamicSphereSweepPositiveSoftDisplacement >
							0.02f &&
					metrics.
						dynamicSphereSweepPositiveMinSeparation >
							(convexCase ? 1.0e-3f : -0.15f) &&
					metrics.
						dynamicSphereSweepPositiveMinSeparation <
							PX_MAX_F32 ? 1u : 0u;
				metrics.
					dynamicSphereSweepNegativeControlTunneled =
						metrics.
							dynamicSphereSweepNegativeSoftDisplacement <
								5.0e-3f &&
							metrics.
								dynamicSphereSweepNegativeRigidDrop >
								(rotationalFiniteCase
									? 0.8f : 1.5f)
							? 1u : 0u;
				metrics.
					dynamicSphereSweepTwoSidedResponseObserved =
						metrics.
							dynamicSphereSweepPositiveRigidDrop +
								0.05f <
						metrics.
							dynamicSphereSweepNegativeRigidDrop
							? 1u : 0u;

				scene->removeActor(*speculative.surface);
				scene->removeActor(*discrete.surface);
				scene->removeActor(*spheres[0]);
				scene->removeActor(*spheres[1]);
			}
		}

		metrics.speculativeCcdPreventedTunneling =
			metrics.dynamicSphereSweepResponseObserved;
		metrics.speculativeCcdNegativeControlTunneled =
			metrics.dynamicSphereSweepNegativeControlTunneled;
		metrics.pinnedStable = 1;
		metrics.dynamicMoved =
			metrics.dynamicSphereSweepResponseObserved;
		metrics.boundsFinite = boundsFinite ? 1u : 0u;
		metrics.maxDynamicDisplacement =
			metrics.dynamicSphereSweepPositiveSoftDisplacement;
		metrics.initialDynamicCentroidY = positiveInitial.y;
		metrics.finalDynamicCentroidY =
			getMotionSurfaceCentroid(speculative).y;
		metrics.minY = metrics.finalDynamicCentroidY;
		metrics.finalMinY = metrics.finalDynamicCentroidY;
		metrics.maxSpeed = PxMax(
			getMotionSurfaceMaxSpeed(speculative),
			getMotionSurfaceMaxSpeed(discrete));
		metrics.finalMaxSpeed =
			getMotionSurfaceMaxSpeed(speculative);

		success = finite &&
			metrics.speculativeCcdFlagApplied &&
			metrics.dynamicSphereSweepLaunched &&
			metrics.dynamicSphereSweepResponseObserved &&
			metrics.dynamicSphereSweepNegativeControlTunneled &&
			metrics.dynamicSphereSweepTwoSidedResponseObserved &&
			metrics.fetchFailures == 0 &&
			metrics.nonFiniteSamples == 0 &&
			metrics.boundsFinite &&
			(!rotationalFiniteCase ||
			 rotationalSweepIsolated);
		if(rotationalFiniteCase)
		{
			std::printf(
				"[%s] "
				"frames=%u target=dynamic owner=forward "
				"responseObserved=%u negativeControlPassed=%u "
				"twoSidedResponseObserved=%u "
				"endpointMinSeparation=%.9g "
				"midSweepMinSeparation=%.9g "
				"positiveDisplacement=%.9g "
				"negativeDisplacement=%.9g "
				"positiveAngularTravel=%.9g "
				"negativeAngularTravel=%.9g result=%s\n",
				rotationalCapsuleCase
					? "AVBD_CAPSULE_DYNAMIC_ROTATIONAL_SWEPT"
					: "AVBD_CONVEX_DYNAMIC_ROTATIONAL_SWEPT",
				options.frames,
				metrics.dynamicSphereSweepResponseObserved,
				metrics.
					dynamicSphereSweepNegativeControlTunneled,
				metrics.
					dynamicSphereSweepTwoSidedResponseObserved,
				double(rotationalEndpointMinSeparation),
				double(rotationalMidMinSeparation),
				double(
					metrics.
						dynamicSphereSweepPositiveSoftDisplacement),
				double(
					metrics.
						dynamicSphereSweepNegativeSoftDisplacement),
				double(rotationalPositiveAngularTravel),
				double(rotationalNegativeAngularTravel),
				success ? "PASS" : "FAIL");
		}
	}
	while(false);

	for(PxU32 i = 0; i < 2; ++i)
		PX_RELEASE(spheres[i]);
	PX_RELEASE(convexMesh);
	releaseMotionSurfaceFixture(discrete);
	releaseMotionSurfaceFixture(speculative);
	PX_RELEASE(rigidMaterial);
	PX_RELEASE(scene);
	PX_RELEASE(dispatcher);
	if(extensionsInitialized)
		PxCloseExtensions();
	PX_RELEASE(physics);
	PX_RELEASE(foundation);
	metrics.cleanupComplete = 1;
	return success;
}

static bool runFiniteReverseSweptCcdCase(
	const Snippets::HeadlessOptions& options,
	Snippets::TrackingErrorCallback& errorCallback,
	Metrics& metrics)
{
	const bool deformingSoftTarget =
		options.caseName ==
			"surface-deforming-sphere-reverse-swept-ccd" ||
		options.caseName ==
			"surface-deforming-capsule-reverse-swept-ccd" ||
		options.caseName ==
			"surface-deforming-convex-reverse-swept-ccd";
	const bool rotationalCapsuleTarget =
		options.caseName ==
			"surface-rotating-kinematic-capsule-reverse-swept-ccd" ||
		options.caseName ==
			"surface-dynamic-rotating-capsule-reverse-swept-ccd";
	const bool rotationalConvexTarget =
		options.caseName ==
			"surface-rotating-kinematic-convex-reverse-swept-ccd" ||
		options.caseName ==
			"surface-dynamic-rotating-convex-reverse-swept-ccd";
	const bool rotationalFiniteTarget =
		rotationalCapsuleTarget || rotationalConvexTarget;
	const bool capsuleTarget =
		rotationalCapsuleTarget ||
		options.caseName ==
			"surface-deforming-capsule-reverse-swept-ccd" ||
		options.caseName ==
			"surface-static-capsule-reverse-swept-ccd" ||
		options.caseName ==
			"surface-kinematic-capsule-reverse-swept-ccd" ||
		options.caseName ==
			"surface-dynamic-capsule-reverse-swept-ccd";
	const bool convexTarget =
		rotationalConvexTarget ||
		options.caseName ==
			"surface-deforming-convex-reverse-swept-ccd" ||
		options.caseName ==
			"surface-static-convex-reverse-swept-ccd" ||
		options.caseName ==
			"surface-kinematic-convex-reverse-swept-ccd" ||
		options.caseName ==
			"surface-dynamic-convex-reverse-swept-ccd";
	const bool staticTarget =
		deformingSoftTarget ||
		options.caseName ==
			"surface-static-sphere-reverse-swept-ccd" ||
		options.caseName ==
			"surface-static-capsule-reverse-swept-ccd" ||
		options.caseName ==
			"surface-static-convex-reverse-swept-ccd";
	const bool kinematicTarget =
		options.caseName ==
			"surface-kinematic-sphere-reverse-swept-ccd" ||
		options.caseName ==
			"surface-kinematic-capsule-reverse-swept-ccd" ||
		options.caseName ==
			"surface-rotating-kinematic-capsule-reverse-swept-ccd" ||
		options.caseName ==
			"surface-rotating-kinematic-convex-reverse-swept-ccd" ||
		options.caseName ==
			"surface-kinematic-convex-reverse-swept-ccd";
	const bool dynamicTarget =
		options.caseName ==
			"surface-dynamic-sphere-reverse-swept-ccd" ||
		options.caseName ==
			"surface-dynamic-capsule-reverse-swept-ccd" ||
		options.caseName ==
			"surface-dynamic-rotating-capsule-reverse-swept-ccd" ||
		options.caseName ==
			"surface-dynamic-rotating-convex-reverse-swept-ccd" ||
		options.caseName ==
			"surface-dynamic-convex-reverse-swept-ccd";
	PxDefaultAllocator allocator;
	PxFoundation* foundation = PxCreateFoundation(
		PX_PHYSICS_VERSION, allocator, errorCallback);
	if(!foundation)
		return false;

	PxPhysics* physics = NULL;
	PxDefaultCpuDispatcher* dispatcher = NULL;
	PxScene* scene = NULL;
	PxMaterial* rigidMaterial = NULL;
	PxRigidActor* rigidActors[2] = {NULL, NULL};
	PxRigidDynamic* dynamicActors[2] = {NULL, NULL};
	PxConvexMesh* convexMesh = NULL;
	PxConvexMeshGeometry convexGeometry;
	MotionSurfaceFixture speculative;
	MotionSurfaceFixture discrete;
	bool extensionsInitialized = false;
	bool finite = true;
	PxU32 responseObserved = 0;
	PxU32 negativeControlPassed = 0;
	PxU32 twoSidedResponseObserved = dynamicTarget ? 0u : 1u;
	PxU32 vertexSweepExcluded = 0;
	PxReal positiveDisplacement = 0.0f;
	PxReal negativeDisplacement = 0.0f;
	PxReal positiveDrop = 0.0f;
	PxReal negativeDrop = 0.0f;
	PxReal positiveRigidDrop = 0.0f;
	PxReal negativeRigidDrop = 0.0f;
	PxReal faceSeparation = PX_MAX_F32;
	PxReal minimumVertexSweepSeparation = PX_MAX_F32;
	PxReal endpointMinSeparation = PX_MAX_F32;
	PxReal midSweepMinSeparation = PX_MAX_F32;
	PxReal positiveAngularTravel = 0.0f;
	PxReal negativeAngularTravel = 0.0f;
	PxReal responseDelta = 0.0f;
	bool success = false;

	do
	{
		if(!staticTarget && !kinematicTarget && !dynamicTarget)
			break;
		const PxTolerancesScale scale;
		physics = PxCreatePhysics(
			PX_PHYSICS_VERSION, *foundation, scale, true, NULL);
		if(!physics)
			break;
		extensionsInitialized = PxInitExtensions(*physics, NULL);
		if(!extensionsInitialized)
			break;

		PxSceneDesc sceneDesc(scale);
		sceneDesc.gravity = PxVec3(0.0f);
		sceneDesc.solverType = PxSolverType::eAVBD;
		sceneDesc.filterShader = PxDefaultSimulationFilterShader;
		dispatcher = PxDefaultCpuDispatcherCreate(
			options.dispatcherThreads);
		if(!dispatcher)
			break;
		sceneDesc.cpuDispatcher = dispatcher;
		scene = physics->createScene(sceneDesc);
		if(!scene)
			break;

		rigidMaterial = physics->createMaterial(
			0.0f, 0.0f, 0.0f);
		if(!rigidMaterial)
			break;
		const PxVec3 translationConvexVertices[] =
		{
			PxVec3(-0.2f, -0.3f, -0.15f),
			PxVec3(0.2f, -0.3f, -0.15f),
			PxVec3(0.2f, -0.3f, 0.15f),
			PxVec3(-0.2f, -0.3f, 0.15f),
			PxVec3(-0.15f, 0.3f, -0.15f),
			PxVec3(0.15f, 0.3f, -0.15f),
			PxVec3(0.0f, 0.3f, 0.2f)
		};
		const PxVec3 rotationalConvexVertices[] =
		{
			PxVec3(-1.0f, -0.1f, -0.1f),
			PxVec3(1.0f, -0.1f, -0.1f),
			PxVec3(1.0f, -0.1f, 0.1f),
			PxVec3(-1.0f, -0.1f, 0.1f),
			PxVec3(-1.0f, 0.1f, -0.1f),
			PxVec3(1.0f, 0.1f, -0.1f),
			PxVec3(1.0f, 0.1f, 0.1f),
			PxVec3(-1.0f, 0.1f, 0.1f)
		};
		const PxVec3* convexVertices =
			rotationalConvexTarget
				? rotationalConvexVertices
				: translationConvexVertices;
		const PxU32 convexVertexCount =
			rotationalConvexTarget ? 8u : 7u;
		if(convexTarget)
		{
			convexMesh = createAvbdTestConvexMesh(
				*physics, scale, convexVertices,
				convexVertexCount);
			if(!convexMesh)
				break;
			convexGeometry = PxConvexMeshGeometry(convexMesh);
		}
		const PxVec3 surfaceOffsets[2] =
		{
			PxVec3(
				-2.0f,
				deformingSoftTarget
					? 0.45f : (staticTarget ? 1.1f : 0.0f),
				0.0f),
			PxVec3(
				1.0f,
				deformingSoftTarget
					? 0.45f : (staticTarget ? 1.1f : 0.0f),
				0.0f)
		};
		const PxVec3 surfaceVelocity =
			deformingSoftTarget
				? PxVec3(0.0f)
				: staticTarget
				? PxVec3(0.0f, -132.0f, 0.0f)
				: PxVec3(0.0f);
		if(!createMotionSurfaceFixture(
				*physics, *scene, scale, surfaceOffsets[0],
				surfaceVelocity, PX_MAX_F32,
				0.0f, 0.0f, 0.0f, 1.0f, speculative) ||
			!createMotionSurfaceFixture(
				*physics, *scene, scale, surfaceOffsets[1],
				surfaceVelocity, PX_MAX_F32,
				0.0f, 0.0f, 0.0f, 1.0f, discrete))
			break;
		if(deformingSoftTarget)
		{
			for(PxU32 vertex = 0; vertex < 4; ++vertex)
			{
				const PxVec3 velocity =
					(vertex == 0 || vertex == 2)
						? PxVec3(0.0f, -180.0f, 0.0f)
						: PxVec3(0.0f);
				speculative.velocities[vertex] =
					PxVec4(velocity, 0.0f);
				discrete.velocities[vertex] =
					PxVec4(velocity, 0.0f);
			}
			speculative.surface->markDirty(
				PxDeformableSurfaceDataFlag::eVELOCITY);
			discrete.surface->markDirty(
				PxDeformableSurfaceDataFlag::eVELOCITY);
		}
		speculative.surface->setDeformableBodyFlag(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD, true);
		metrics.speculativeCcdFlagApplied =
			speculative.surface->getDeformableBodyFlags().isSet(
				PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD) &&
			!discrete.surface->getDeformableBodyFlags().isSet(
				PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD)
				? 1u : 0u;

		const PxReal radius =
			rotationalCapsuleTarget ? 0.1f : 0.3f;
		const PxReal halfHeight =
			rotationalCapsuleTarget ? 1.0f :
				(capsuleTarget
					? (deformingSoftTarget ? 0.05f : 0.15f)
					: 0.0f);
		const PxReal startY =
			deformingSoftTarget ? 0.0f :
				(rotationalFiniteTarget ? 0.75f :
					(staticTarget ? 0.0f : 1.1f));

		const PxReal endY = staticTarget ? 0.0f : -1.1f;
		const PxQuat rotationalStart(
			PxPi / 6.0f, PxVec3(0.0f, 0.0f, 1.0f));
		const PxQuat rotationalEnd(
			PxPi * 5.0f / 6.0f,
			PxVec3(0.0f, 0.0f, 1.0f));
		const PxReal rotationalAngularSpeed =
			(PxPi * 10.0f / 9.0f) / options.dt;
		const PxVec3 sphereStartCenters[2] =
		{
			PxVec3(
				surfaceOffsets[0].x +
					(rotationalFiniteTarget ? 0.5f : 0.35f),
				startY,
				surfaceOffsets[0].z +
					(rotationalFiniteTarget ? 0.5f : 0.35f)),
			PxVec3(
				surfaceOffsets[1].x +
					(rotationalFiniteTarget ? 0.5f : 0.35f),
				startY,
				surfaceOffsets[1].z +
					(rotationalFiniteTarget ? 0.5f : 0.35f))
		};
		for(PxU32 i = 0; i < 2; ++i)
		{
			if(staticTarget)
			{
				rigidActors[i] = convexTarget
					? PxCreateStatic(
						*physics,
						PxTransform(
							sphereStartCenters[i],
							rotationalFiniteTarget
								? rotationalStart : PxQuat(PxIdentity)),
						convexGeometry, *rigidMaterial)
					: capsuleTarget
					? PxCreateStatic(
						*physics,
						PxTransform(
							sphereStartCenters[i],
							rotationalFiniteTarget
								? rotationalStart : PxQuat(PxIdentity)),
						PxCapsuleGeometry(radius, halfHeight),
						*rigidMaterial)
					: PxCreateStatic(
						*physics, PxTransform(sphereStartCenters[i]),
						PxSphereGeometry(radius), *rigidMaterial);
			}
			else
			{
				dynamicActors[i] = physics->createRigidDynamic(
					PxTransform(
						sphereStartCenters[i],
						rotationalFiniteTarget
							? rotationalStart : PxQuat(PxIdentity)));
				rigidActors[i] = dynamicActors[i];
				if(!dynamicActors[i] ||
					!(convexTarget
						? PxRigidActorExt::createExclusiveShape(
							*dynamicActors[i],
							convexGeometry,
							*rigidMaterial)
						: capsuleTarget
						? PxRigidActorExt::createExclusiveShape(
							*dynamicActors[i],
							PxCapsuleGeometry(radius, halfHeight),
							*rigidMaterial)
						: PxRigidActorExt::createExclusiveShape(
							*dynamicActors[i],
							PxSphereGeometry(radius),
							*rigidMaterial)))
					break;
				if(kinematicTarget)
					dynamicActors[i]->setRigidBodyFlag(
						PxRigidBodyFlag::eKINEMATIC, true);
				else
				{
					if(!PxRigidBodyExt::setMassAndUpdateInertia(
							*dynamicActors[i],
							rotationalFiniteTarget ? 10.0f : 1.0f))
						break;
					dynamicActors[i]->setActorFlag(
						PxActorFlag::eDISABLE_GRAVITY, true);
					PxRigidDynamicLockFlags lockFlags =
						PxRigidDynamicLockFlag::eLOCK_LINEAR_X |
						PxRigidDynamicLockFlag::eLOCK_LINEAR_Z |
						PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
						PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y;
					if(rotationalFiniteTarget)
						lockFlags |=
							PxRigidDynamicLockFlag::eLOCK_LINEAR_Y;
					else
						lockFlags |=
							PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z;
					dynamicActors[i]->setRigidDynamicLockFlags(lockFlags);
					dynamicActors[i]->setSolverIterationCounts(16, 1);
					if(rotationalFiniteTarget)
					{
						dynamicActors[i]->setMaxAngularVelocity(300.0f);
						dynamicActors[i]->setAngularVelocity(
							PxVec3(
								0.0f, 0.0f,
								rotationalAngularSpeed));
					}
					else
						dynamicActors[i]->setLinearVelocity(
							PxVec3(0.0f, -132.0f, 0.0f));
				}
			}
			if(!rigidActors[i] ||
				!scene->addActor(*rigidActors[i]))
				break;
		}
		if(!rigidActors[0] || !rigidActors[1] ||
			rigidActors[0]->getScene() != scene ||
			rigidActors[1]->getScene() != scene)
			break;
		if(kinematicTarget)
		{
			for(PxU32 i = 0; i < 2; ++i)
				dynamicActors[i]->setKinematicTarget(
					rotationalFiniteTarget
						? PxTransform(
							sphereStartCenters[i], rotationalEnd)
						: PxTransform(PxVec3(
							sphereStartCenters[i].x,
							endY,
							sphereStartCenters[i].z)));
			metrics.movingSphereTargetIssued = 1;
		}
		if(dynamicTarget)
			metrics.dynamicSphereSweepLaunched = 1;

		metrics.actorCreated = 1;
		metrics.shapeAttached = 1;
		metrics.hostBuffersInitialized = 1;
		metrics.actorAdded = 1;
		scene->removeActor(*speculative.surface);
		if(speculative.surface->getScene() != NULL)
			break;
		metrics.actorRemoved = 1;
		if(!scene->addActor(*speculative.surface) ||
			speculative.surface->getScene() != scene)
			break;
		metrics.actorReadded = 1;

		const PxVec3 positiveInitial =
			getMotionSurfaceCentroid(speculative);
		const PxVec3 negativeInitial =
			getMotionSurfaceCentroid(discrete);
		auto getCapsuleFaceSeparation = [&](
			const PxVec3* points, const PxTransform& capsulePose)
		{
			PxReal minimumSeparation = PX_MAX_F32;
			const PxVec3 axis =
				capsulePose.q.getBasisVector0();
			for(PxU32 sample = 0; sample <= 128; ++sample)
			{
				const PxReal axisCoordinate =
					-halfHeight +
					2.0f * halfHeight *
						(PxReal(sample) / 128.0f);
				const PxVec3 medialPoint =
					capsulePose.p + axis * axisCoordinate;
				const PxVec3 closest0 =
					closestPointOnTriangleForConvexGate(
						medialPoint, points[0], points[2], points[1]);
				const PxVec3 closest1 =
					closestPointOnTriangleForConvexGate(
						medialPoint, points[2], points[3], points[1]);
				minimumSeparation = PxMin(
					minimumSeparation,
					PxMin(
						(medialPoint - closest0).magnitude() - radius,
						(medialPoint - closest1).magnitude() - radius));
			}
			return minimumSeparation;
		};
		auto getConvexFaceSeparation = [&](
			const PxVec3* points, const PxTransform& convexPose)
		{
			PxReal minimumSeparation = PX_MAX_F32;
			for(PxU32 rigidVertex = 0;
				rigidVertex < convexVertexCount; ++rigidVertex)
			{
				const PxVec3 vertexWorld =
					convexPose.transform(
						convexVertices[rigidVertex]);
				const PxVec3 closest0 =
					closestPointOnTriangleForConvexGate(
						vertexWorld,
						points[0], points[2], points[1]);
				const PxVec3 closest1 =
					closestPointOnTriangleForConvexGate(
						vertexWorld,
						points[2], points[3], points[1]);
				const PxVec3 delta0 = vertexWorld - closest0;
				const PxVec3 delta1 = vertexWorld - closest1;
				const PxReal signed0 =
					(vertexWorld.y >= closest0.y ? 1.0f : -1.0f) *
						delta0.magnitude();
				const PxReal signed1 =
					(vertexWorld.y >= closest1.y ? 1.0f : -1.0f) *
						delta1.magnitude();
				minimumSeparation = PxMin(
					minimumSeparation, PxMin(signed0, signed1));
			}
			return minimumSeparation;
		};
		auto getConvexFaceDistance = [&](
			const PxVec3* points, const PxTransform& convexPose)
		{
			PxReal minimumDistance = PX_MAX_F32;
			for(PxU32 rigidVertex = 0;
				rigidVertex < convexVertexCount; ++rigidVertex)
			{
				const PxVec3 vertexWorld =
					convexPose.transform(
						convexVertices[rigidVertex]);
				const PxVec3 closest0 =
					closestPointOnTriangleForConvexGate(
						vertexWorld,
						points[0], points[2], points[1]);
				const PxVec3 closest1 =
					closestPointOnTriangleForConvexGate(
						vertexWorld,
						points[2], points[3], points[1]);
				minimumDistance = PxMin(
					minimumDistance,
					PxMin(
						(vertexWorld - closest0).magnitude(),
						(vertexWorld - closest1).magnitude()));
			}
			return minimumDistance;
		};
		if(deformingSoftTarget)
		{
			PxVec3 freeStart[4];
			PxVec3 freeEnd[4];
			for(PxU32 vertex = 0; vertex < 4; ++vertex)
			{
				freeStart[vertex] =
					speculative.initialPositions[vertex];
				freeEnd[vertex] =
					freeStart[vertex] +
					speculative.velocities[vertex].getXYZ() *
						options.dt;
			}
			const PxTransform rigidPose =
				rigidActors[0]->getGlobalPose();
			const PxReal startSeparation =
				convexTarget
					? getConvexFaceDistance(freeStart, rigidPose)
					: getCapsuleFaceSeparation(
						freeStart, rigidPose);
			const PxReal endSeparation =
				convexTarget
					? getConvexFaceDistance(freeEnd, rigidPose)
					: getCapsuleFaceSeparation(
						freeEnd, rigidPose);
			endpointMinSeparation =
				PxMin(startSeparation, endSeparation);
			for(PxU32 sample = 0; sample <= 128; ++sample)
			{
				const PxReal time =
					PxReal(sample) / 128.0f;
				PxVec3 samplePoints[4];
				for(PxU32 vertex = 0; vertex < 4; ++vertex)
				{
					samplePoints[vertex] =
						freeStart[vertex] +
						(freeEnd[vertex] -
							freeStart[vertex]) * time;
					const PxReal separation =
						convexTarget
							? [&]()
							{
								const PxReal distanceSq =
									PxGeometryQuery::pointDistance(
										samplePoints[vertex],
										convexGeometry, rigidPose);
								return distanceSq >= 0.0f &&
									PxIsFinite(distanceSq)
									? PxSqrt(distanceSq)
									: -PX_MAX_F32;
							}()
							: getCapsuleSignedSeparation(
								samplePoints[vertex],
								rigidPose, radius, halfHeight);
					minimumVertexSweepSeparation = PxMin(
						minimumVertexSweepSeparation,
						separation);
				}
				if(sample > 0 && sample < 128)
					midSweepMinSeparation = PxMin(
						midSweepMinSeparation,
						convexTarget
							? getConvexFaceDistance(
								samplePoints, rigidPose)
							: getCapsuleFaceSeparation(
								samplePoints, rigidPose));
			}
			vertexSweepExcluded =
				minimumVertexSweepSeparation > 0.10f ? 1u : 0u;
		}
		else if(!rotationalFiniteTarget)
		{
			for(PxU32 vertex = 0; vertex < 4; ++vertex)
			{
				const PxVec3 start = speculative.initialPositions[vertex];
				const PxVec3 center0 = sphereStartCenters[0];
				const PxVec3 center1(
					center0.x,
					staticTarget ? center0.y - 2.2f : endY,
					center0.z);
				const PxVec3 centerSegment = center1 - center0;
				const PxReal denominator =
					centerSegment.magnitudeSquared();
				const PxReal t = denominator > 1.0e-12f
					? PxClamp(
						(start - center0).dot(centerSegment) /
							denominator,
						0.0f, 1.0f)
					: 0.0f;
				const PxVec3 centerAtClosest =
					center0 + centerSegment * t;
				if(convexTarget)
				{
					for(PxU32 sample = 0; sample <= 64; ++sample)
					{
						const PxReal sampleTime =
							PxReal(sample) / 64.0f;
						const PxVec3 sampleCenter =
							center0 + centerSegment * sampleTime;
						const PxReal distanceSq =
							PxGeometryQuery::pointDistance(
								start, convexGeometry,
								PxTransform(sampleCenter));
						if(distanceSq >= 0.0f &&
							PxIsFinite(distanceSq))
							minimumVertexSweepSeparation = PxMin(
								minimumVertexSweepSeparation,
								PxSqrt(distanceSq));
					}
				}
				else
				{
					const PxVec3 medialPoint =
						centerAtClosest + PxVec3(
							PxClamp(start.x - centerAtClosest.x,
								-halfHeight, halfHeight),
							0.0f, 0.0f);
					minimumVertexSweepSeparation = PxMin(
						minimumVertexSweepSeparation,
						(start - medialPoint).magnitude() - radius);
				}
			}
			vertexSweepExcluded =
				minimumVertexSweepSeparation > 0.10f ? 1u : 0u;
		}

		bool boundsFinite = true;
		for(PxU32 frame = 0; frame < options.frames; ++frame)
		{
			scene->simulate(options.dt);
			if(!scene->fetchResults(true))
			{
				++metrics.fetchFailures;
				break;
			}
			if(frame != 0)
				continue;

			const PxVec3 positiveCentroid =
				getMotionSurfaceCentroid(speculative);
			const PxVec3 negativeCentroid =
				getMotionSurfaceCentroid(discrete);
			const PxTransform positivePose =
				rigidActors[0]->getGlobalPose();
			const PxTransform negativePose =
				rigidActors[1]->getGlobalPose();
			const PxVec3 positiveCenter = positivePose.p;
			const PxVec3 negativeCenter = negativePose.p;
			positiveDisplacement =
				(positiveCentroid - positiveInitial).magnitude();
			negativeDisplacement =
				(negativeCentroid - negativeInitial).magnitude();
			responseDelta = 0.0f;
			for(PxU32 vertex = 0; vertex < 4; ++vertex)
			{
				const PxVec3 positiveLocal =
					speculative.positions[vertex].getXYZ() -
						PxVec3(surfaceOffsets[0].x, 0.0f, 0.0f);
				const PxVec3 negativeLocal =
					discrete.positions[vertex].getXYZ() -
						PxVec3(surfaceOffsets[1].x, 0.0f, 0.0f);
				responseDelta = PxMax(
					responseDelta,
					(positiveLocal - negativeLocal).magnitude());
			}
			positiveDrop = positiveInitial.y - positiveCentroid.y;
			negativeDrop = negativeInitial.y - negativeCentroid.y;
			if(rotationalFiniteTarget)
			{
				auto getAngularTravel = [&](
					const PxQuat& endRotation)
				{
					const PxReal alignment = PxClamp(
						PxAbs(
							rotationalStart.dot(
								endRotation.getNormalized())),
						0.0f, 1.0f);
					return 2.0f * PxAcos(alignment);
				};
				positiveAngularTravel =
					getAngularTravel(positivePose.q);
				negativeAngularTravel =
					getAngularTravel(negativePose.q);
				positiveRigidDrop = positiveAngularTravel;
				negativeRigidDrop = negativeAngularTravel;
				const PxTransform startPose(
					sphereStartCenters[0], rotationalStart);
				const PxTransform endPose(
					sphereStartCenters[0], negativePose.q);
				endpointMinSeparation = PxMin(
					rotationalCapsuleTarget
						? getCapsuleFaceSeparation(
							speculative.initialPositions,
							startPose)
						: getConvexFaceSeparation(
							speculative.initialPositions,
							startPose),
					rotationalCapsuleTarget
						? getCapsuleFaceSeparation(
							speculative.initialPositions,
							endPose)
						: getConvexFaceSeparation(
							speculative.initialPositions,
							endPose));
				for(PxU32 sample = 0; sample <= 64; ++sample)
				{
					const PxReal time =
						PxReal(sample) / 64.0f;
					const PxTransform samplePose(
						sphereStartCenters[0],
						PxSlerp(
							time, rotationalStart,
							negativePose.q.getNormalized()).
								getNormalized());
					if(sample > 0 && sample < 64)
						midSweepMinSeparation = PxMin(
							midSweepMinSeparation,
							rotationalCapsuleTarget
								? getCapsuleFaceSeparation(
									speculative.initialPositions,
									samplePose)
								: getConvexFaceSeparation(
									speculative.initialPositions,
									samplePose));
					for(PxU32 vertex = 0; vertex < 4; ++vertex)
					{
						const PxReal separation =
							rotationalCapsuleTarget
								? getCapsuleSignedSeparation(
									speculative.
										initialPositions[vertex],
									samplePose, radius, halfHeight)
								: [&]()
								{
									const PxReal distanceSq =
										PxGeometryQuery::pointDistance(
											speculative.
												initialPositions[vertex],
											convexGeometry, samplePose);
									return distanceSq >= 0.0f &&
										PxIsFinite(distanceSq)
										? PxSqrt(distanceSq)
										: -PX_MAX_F32;
								}();
						minimumVertexSweepSeparation = PxMin(
							minimumVertexSweepSeparation,
							separation);
					}
				}
				vertexSweepExcluded =
					minimumVertexSweepSeparation > 0.10f ? 1u : 0u;
			}
			else
			{
				positiveRigidDrop =
					sphereStartCenters[0].y - positiveCenter.y;
				negativeRigidDrop =
					sphereStartCenters[1].y - negativeCenter.y;
			}
			for(PxU32 vertex = 0; vertex < 4; ++vertex)
			{
				if(!speculative.positions[vertex].isFinite() ||
					!speculative.velocities[vertex].isFinite() ||
					!discrete.positions[vertex].isFinite() ||
					!discrete.velocities[vertex].isFinite())
				{
					finite = false;
					++metrics.nonFiniteSamples;
				}
			}
			const PxVec3 p[4] =
			{
				speculative.positions[0].getXYZ(),
				speculative.positions[1].getXYZ(),
				speculative.positions[2].getXYZ(),
				speculative.positions[3].getXYZ()
			};
			faceSeparation = PX_MAX_F32;
			if(convexTarget)
			{
				if(rotationalConvexTarget)
					faceSeparation =
						getConvexFaceSeparation(p, positivePose);
				else
				{
					for(PxU32 rigidVertex = 0;
						rigidVertex < convexVertexCount;
						++rigidVertex)
					{
						const PxVec3 vertexWorld =
							positivePose.transform(
								convexVertices[rigidVertex]);
						const PxVec3 closest0 =
							closestPointOnTriangleForConvexGate(
								vertexWorld, p[0], p[2], p[1]);
						const PxVec3 closest1 =
							closestPointOnTriangleForConvexGate(
								vertexWorld, p[2], p[3], p[1]);
						faceSeparation = PxMin(
							faceSeparation,
							PxMin(
								(vertexWorld - closest0).
									magnitude(),
								(vertexWorld - closest1).
									magnitude()));
					}
				}
			}
			else
			{
				if(rotationalFiniteTarget)
					faceSeparation =
						getCapsuleFaceSeparation(p, positivePose);
				else
				{
					for(PxU32 axisSample = 0;
						axisSample < (capsuleTarget ? 3u : 1u);
						++axisSample)
					{
						const PxReal axisOffset = capsuleTarget
							? halfHeight *
								(PxReal(axisSample) - 1.0f)
							: 0.0f;
						const PxVec3 medialPoint =
							positiveCenter +
								PxVec3(axisOffset, 0.0f, 0.0f);
						const PxVec3 closest0 =
							closestPointOnTriangleForConvexGate(
								medialPoint, p[0], p[2], p[1]);
						const PxVec3 closest1 =
							closestPointOnTriangleForConvexGate(
								medialPoint, p[2], p[3], p[1]);
						faceSeparation = PxMin(
							faceSeparation,
							PxMin(
								(medialPoint - closest0).magnitude() -
									radius,
								(medialPoint - closest1).magnitude() -
									radius));
					}
				}
			}
			boundsFinite =
				speculative.surface->getWorldBounds().isValid() &&
				discrete.surface->getWorldBounds().isValid();

			if(deformingSoftTarget)
			{
				responseObserved =
					responseDelta > 0.01f &&
					faceSeparation > -0.15f ? 1u : 0u;
				negativeControlPassed =
					negativeDrop > 0.25f ? 1u : 0u;
			}
			else if(staticTarget)
			{
				responseObserved =
					positiveDisplacement > 0.02f &&
					positiveDrop + 0.10f < negativeDrop &&
					faceSeparation > -0.10f ? 1u : 0u;
				negativeControlPassed =
					negativeDrop >
						(convexTarget
							? 1.3f
							: capsuleTarget ? 1.3f : 1.5f)
							? 1u : 0u;
			}
			else if(kinematicTarget)
			{
				responseObserved =
					positiveDisplacement > 0.02f &&
					faceSeparation > -0.10f ? 1u : 0u;
				negativeControlPassed =
					negativeDisplacement < 5.0e-3f ? 1u : 0u;
			}
			else
			{
				responseObserved =
					positiveDisplacement > 0.02f &&
					faceSeparation > -0.15f ? 1u : 0u;
				negativeControlPassed =
					negativeDisplacement < 5.0e-3f &&
					negativeRigidDrop >
						(rotationalFiniteTarget ? 0.8f : 1.5f)
						? 1u : 0u;
				twoSidedResponseObserved =
					positiveRigidDrop + 0.05f <
						negativeRigidDrop ? 1u : 0u;
			}
			scene->removeActor(*speculative.surface);
			scene->removeActor(*discrete.surface);
			scene->removeActor(*rigidActors[0]);
			scene->removeActor(*rigidActors[1]);
		}

		metrics.speculativeCcdPreventedTunneling =
			responseObserved;
		metrics.speculativeCcdNegativeControlTunneled =
			negativeControlPassed;
		metrics.pinnedStable = 1;
		metrics.dynamicMoved = responseObserved;
		metrics.boundsFinite = boundsFinite ? 1u : 0u;
		metrics.maxDynamicDisplacement = positiveDisplacement;
		metrics.initialDynamicCentroidY = positiveInitial.y;
		metrics.finalDynamicCentroidY =
			getMotionSurfaceCentroid(speculative).y;
		metrics.minY = PxMin(
			metrics.finalDynamicCentroidY,
			getMotionSurfaceCentroid(discrete).y);
		metrics.finalMinY = metrics.minY;
		metrics.maxSpeed = PxMax(
			getMotionSurfaceMaxSpeed(speculative),
			getMotionSurfaceMaxSpeed(discrete));
		metrics.finalMaxSpeed = metrics.maxSpeed;
		success = finite &&
			metrics.speculativeCcdFlagApplied &&
			responseObserved &&
			negativeControlPassed &&
			twoSidedResponseObserved &&
			vertexSweepExcluded &&
			(!deformingSoftTarget ||
				(PxIsFinite(endpointMinSeparation) &&
				 endpointMinSeparation > 0.10f &&
				 PxIsFinite(midSweepMinSeparation) &&
				 midSweepMinSeparation <
					(convexTarget ? 0.01f : -0.02f))) &&
			(!rotationalFiniteTarget ||
				(PxIsFinite(endpointMinSeparation) &&
				 endpointMinSeparation > 0.05f &&
				 PxIsFinite(midSweepMinSeparation) &&
				 midSweepMinSeparation < -0.05f &&
				 PxIsFinite(positiveAngularTravel) &&
				 PxIsFinite(negativeAngularTravel))) &&
			metrics.fetchFailures == 0 &&
			metrics.nonFiniteSamples == 0 &&
			metrics.boundsFinite;
	}
	while(false);

	for(PxU32 i = 0; i < 2; ++i)
		PX_RELEASE(rigidActors[i]);
	PX_RELEASE(convexMesh);
	releaseMotionSurfaceFixture(discrete);
	releaseMotionSurfaceFixture(speculative);
	PX_RELEASE(rigidMaterial);
	PX_RELEASE(scene);
	PX_RELEASE(dispatcher);
	if(extensionsInitialized)
		PxCloseExtensions();
	PX_RELEASE(physics);
	PX_RELEASE(foundation);
	metrics.cleanupComplete = 1;
	const char* targetName = staticTarget
		? "static" : (kinematicTarget ? "kinematic" : "dynamic");
	std::printf(
		"[%s] frames=%u target=%s "
		"responseObserved=%u negativeControlPassed=%u "
		"twoSidedResponseObserved=%u vertexSweepExcluded=%u "
		"nonFiniteSamples=%u positiveDisplacement=%.9g "
		"negativeDisplacement=%.9g positiveDrop=%.9g "
		"negativeDrop=%.9g positiveRigidDrop=%.9g "
		"negativeRigidDrop=%.9g faceSeparation=%.9g "
		"minimumVertexSweepSeparation=%.9g result=%s\n",
		convexTarget
			? "AVBD_CONVEX_REVERSE_SWEPT"
			: capsuleTarget
			? "AVBD_CAPSULE_REVERSE_SWEPT"
			: "AVBD_SPHERE_REVERSE_SWEPT",
		options.frames, targetName,
		responseObserved, negativeControlPassed,
		twoSidedResponseObserved, vertexSweepExcluded,
		metrics.nonFiniteSamples,
		double(positiveDisplacement),
		double(negativeDisplacement),
		double(positiveDrop), double(negativeDrop),
		double(positiveRigidDrop), double(negativeRigidDrop),
		double(faceSeparation),
		double(minimumVertexSweepSeparation),
		success ? "PASS" : "FAIL");
	if(rotationalFiniteTarget)
	{
		std::printf(
			"[%s] "
			"frames=%u target=%s owner=reverse "
			"responseObserved=%u negativeControlPassed=%u "
			"twoSidedResponseObserved=%u vertexSweepExcluded=%u "
			"endpointMinSeparation=%.9g "
			"midSweepMinSeparation=%.9g "
			"positiveDisplacement=%.9g "
			"negativeDisplacement=%.9g "
			"positiveAngularTravel=%.9g "
			"negativeAngularTravel=%.9g result=%s\n",
			rotationalCapsuleTarget
				? "AVBD_CAPSULE_ROTATIONAL_REVERSE_SWEPT"
				: "AVBD_CONVEX_ROTATIONAL_REVERSE_SWEPT",
			options.frames, targetName,
			responseObserved, negativeControlPassed,
			twoSidedResponseObserved, vertexSweepExcluded,
			double(endpointMinSeparation),
			double(midSweepMinSeparation),
			double(positiveDisplacement),
			double(negativeDisplacement),
			double(positiveAngularTravel),
			double(negativeAngularTravel),
			success ? "PASS" : "FAIL");
	}
	if(deformingSoftTarget)
	{
		std::printf(
			"[AVBD_DEFORMING_SOFT_REVERSE_SWEPT] "
			"frames=%u geometry=%s target=static owner=reverse "
			"responseObserved=%u negativeControlPassed=%u "
			"vertexSweepExcluded=%u "
			"endpointMinSeparation=%.9g "
			"midSweepMinSeparation=%.9g "
			"minimumVertexSweepSeparation=%.9g "
			"responseDelta=%.9g "
			"positiveDisplacement=%.9g "
			"negativeDisplacement=%.9g result=%s\n",
			options.frames,
			convexTarget
				? "convex"
				: (capsuleTarget ? "capsule" : "sphere"),
			responseObserved, negativeControlPassed,
			vertexSweepExcluded,
			double(endpointMinSeparation),
			double(midSweepMinSeparation),
			double(minimumVertexSweepSeparation),
			double(responseDelta),
			double(positiveDisplacement),
			double(negativeDisplacement),
			success ? "PASS" : "FAIL");
	}
	return success;
}

static bool runTriangleSurfaceSweptCcdCase(
	const Snippets::HeadlessOptions& options,
	Snippets::TrackingErrorCallback& errorCallback,
	Metrics& metrics)
{
	const bool heightFieldCase =
		options.caseName.find("heightfield") !=
			std::string::npos;
	const bool deformingSoftTarget =
		options.caseName ==
			"surface-deforming-triangle-mesh-reverse-swept-ccd" ||
		options.caseName ==
			"surface-deforming-heightfield-reverse-swept-ccd";
	const bool reverseCase =
		options.caseName.find("reverse-swept") !=
			std::string::npos;
	const bool rotationalCase =
		options.caseName.find("rotating-kinematic") !=
			std::string::npos;
	const bool kinematicTarget =
		options.caseName.find("kinematic") !=
			std::string::npos;
	const bool staticTarget =
		deformingSoftTarget ||
		options.caseName.find("static") != std::string::npos;
	PxDefaultAllocator allocator;
	PxFoundation* foundation = PxCreateFoundation(
		PX_PHYSICS_VERSION, allocator, errorCallback);
	if(!foundation)
		return false;

	PxPhysics* physics = NULL;
	PxDefaultCpuDispatcher* dispatcher = NULL;
	PxScene* scene = NULL;
	PxMaterial* rigidMaterial = NULL;
	PxTriangleMesh* triangleMesh = NULL;
	PxHeightField* heightField = NULL;
	PxRigidActor* rigidActors[2] = {NULL, NULL};
	PxRigidDynamic* kinematicActors[2] = {NULL, NULL};
	MotionSurfaceFixture speculative;
	MotionSurfaceFixture discrete;
	bool extensionsInitialized = false;
	bool finite = true;
	PxU32 responseObserved = 0;
	PxU32 negativeControlPassed = 0;
	PxU32 vertexSweepExcluded = reverseCase ? 0u : 1u;
	PxReal positiveDisplacement = 0.0f;
	PxReal negativeDisplacement = 0.0f;
	PxReal positiveDrop = 0.0f;
	PxReal negativeDrop = 0.0f;
	PxReal minimumVertexSweepSeparation = PX_MAX_F32;
	PxReal endpointMinSeparation = PX_MAX_F32;
	PxReal midSweepMinSeparation = PX_MAX_F32;
	PxReal positiveAngularTravel = 0.0f;
	PxReal negativeAngularTravel = 0.0f;
	PxReal responseDelta = 0.0f;
	bool boundsFinite = false;
	bool success = false;

	do
	{
		if(!staticTarget && !kinematicTarget)
			break;
		const PxTolerancesScale scale;
		physics = PxCreatePhysics(
			PX_PHYSICS_VERSION, *foundation, scale, true, NULL);
		if(!physics)
			break;
		extensionsInitialized = PxInitExtensions(*physics, NULL);
		if(!extensionsInitialized)
			break;
		PxSceneDesc sceneDesc(scale);
		sceneDesc.gravity = PxVec3(0.0f);
		sceneDesc.solverType = PxSolverType::eAVBD;
		sceneDesc.filterShader = PxDefaultSimulationFilterShader;
		dispatcher = PxDefaultCpuDispatcherCreate(
			options.dispatcherThreads);
		if(!dispatcher)
			break;
		sceneDesc.cpuDispatcher = dispatcher;
		scene = physics->createScene(sceneDesc);
		if(!scene)
			break;
		rigidMaterial = physics->createMaterial(
			0.0f, 0.0f, 0.0f);
		if(!rigidMaterial)
			break;

		if(heightFieldCase)
		{
			heightField = createAvbdRigidHeightField(
				*physics, rotationalCase ? false : reverseCase);
			if(!heightField)
				break;
		}
		else
		{
			triangleMesh = rotationalCase
				? createAvbdRotationalRigidTriangleMesh(
					*physics, scale)
				: createAvbdRigidTriangleMesh(
					*physics, scale, reverseCase);
			if(!triangleMesh)
				break;
		}

		const PxVec3 surfaceOffsets[2] =
		{
			PxVec3(-2.0f,
				deformingSoftTarget
					? 0.45f : (staticTarget ? 1.1f : 0.0f),
				0.0f),
			PxVec3(1.0f,
				deformingSoftTarget
					? 0.45f : (staticTarget ? 1.1f : 0.0f),
				0.0f)
		};
		const PxVec3 surfaceVelocity =
			staticTarget && !deformingSoftTarget
				? PxVec3(0.0f, -132.0f, 0.0f)
				: PxVec3(0.0f);
		if(!createMotionSurfaceFixture(
				*physics, *scene, scale,
				surfaceOffsets[0], surfaceVelocity,
				PX_MAX_F32, 0.0f, 0.0f, 0.0f, 1.0f,
				speculative) ||
			!createMotionSurfaceFixture(
				*physics, *scene, scale,
				surfaceOffsets[1], surfaceVelocity,
				PX_MAX_F32, 0.0f, 0.0f, 0.0f, 1.0f,
				discrete))
			break;
		if(deformingSoftTarget)
		{
			for(PxU32 vertex = 0; vertex < 4; ++vertex)
			{
				const PxVec3 velocity =
					(vertex == 0 || vertex == 2)
						? PxVec3(0.0f, -180.0f, 0.0f)
						: PxVec3(0.0f);
				speculative.velocities[vertex] =
					PxVec4(velocity, 0.0f);
				discrete.velocities[vertex] =
					PxVec4(velocity, 0.0f);
			}
			speculative.surface->markDirty(
				PxDeformableSurfaceDataFlag::eVELOCITY);
			discrete.surface->markDirty(
				PxDeformableSurfaceDataFlag::eVELOCITY);
		}
		speculative.surface->setDeformableBodyFlag(
			PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD,
			true);
		metrics.speculativeCcdFlagApplied =
			speculative.surface->getDeformableBodyFlags().
				isSet(
					PxDeformableBodyFlag::
						eENABLE_SPECULATIVE_CCD) &&
			!discrete.surface->getDeformableBodyFlags().
				isSet(
					PxDeformableBodyFlag::
						eENABLE_SPECULATIVE_CCD)
				? 1u : 0u;

		const PxReal startY =
			kinematicTarget ? -1.1f : 0.0f;
		const PxReal endY =
			kinematicTarget ? 1.1f : 0.0f;
		const PxReal rotationalStartAngle = 0.5235987756f;
		const PxReal rotationalEndAngle = 2.6179938780f;
		const PxQuat rotationalStart(
			rotationalStartAngle, PxVec3(0.0f, 0.0f, 1.0f));
		const PxQuat rotationalEnd(
			rotationalEndAngle, PxVec3(0.0f, 0.0f, 1.0f));
		const PxReal rotationalCrossingX =
			heightFieldCase ? 1.56124950f : 0.66143781f;
		PxVec3 rigidStartCenters[2];
		for(PxU32 i = 0; i < 2; ++i)
		{
			if(rotationalCase)
			{
				const PxVec3 target = reverseCase
					? surfaceOffsets[i] +
						PxVec3(0.5f, 0.0f, 0.5f)
					: surfaceOffsets[i];
				rigidStartCenters[i] = PxVec3(
					target.x - rotationalCrossingX,
					heightFieldCase ? -1.25f : -0.75f,
					target.z -
						(heightFieldCase ? 0.1f : 0.0f));
			}
			else if(reverseCase)
			{
				rigidStartCenters[i] = heightFieldCase
					? PxVec3(
						surfaceOffsets[i].x + 0.1f,
						startY, 0.1f)
					: PxVec3(
						surfaceOffsets[i].x + 0.4f,
						startY, 0.4f);
			}
			else
			{
				rigidStartCenters[i] = heightFieldCase
					? PxVec3(
						surfaceOffsets[i].x,
						startY, 0.0f)
					: PxVec3(
						surfaceOffsets[i].x + 0.5f,
						startY, 0.5f);
			}

			if(staticTarget)
				rigidActors[i] = physics->createRigidStatic(
					PxTransform(rigidStartCenters[i]));
			else
			{
				kinematicActors[i] =
					physics->createRigidDynamic(
						PxTransform(
							rigidStartCenters[i],
							rotationalCase
								? rotationalStart
								: PxQuat(PxIdentity)));
				rigidActors[i] = kinematicActors[i];
				if(kinematicActors[i])
					kinematicActors[i]->setRigidBodyFlag(
						PxRigidBodyFlag::eKINEMATIC, true);
			}
			if(!rigidActors[i])
				break;
			const bool shapeCreated = heightFieldCase
				? PxRigidActorExt::createExclusiveShape(
					*rigidActors[i],
					PxHeightFieldGeometry(
						heightField,
						PxMeshGeometryFlags(), 0.1f,
						rotationalCase
							? 1.0f
							: (reverseCase ? 0.3f : 1.2f),
						rotationalCase
							? 0.1f
							: (reverseCase ? 0.3f : 1.2f)),
					*rigidMaterial) != NULL
				: PxRigidActorExt::createExclusiveShape(
					*rigidActors[i],
					PxTriangleMeshGeometry(triangleMesh),
					*rigidMaterial) != NULL;
			if(!shapeCreated ||
				!scene->addActor(*rigidActors[i]))
				break;
		}
		if(!rigidActors[0] || !rigidActors[1] ||
			rigidActors[0]->getScene() != scene ||
			rigidActors[1]->getScene() != scene)
			break;
		if(kinematicTarget)
		{
			for(PxU32 i = 0; i < 2; ++i)
				kinematicActors[i]->setKinematicTarget(
					rotationalCase
						? PxTransform(
							rigidStartCenters[i],
							rotationalEnd)
						: PxTransform(PxVec3(
							rigidStartCenters[i].x,
							endY,
							rigidStartCenters[i].z)));
			metrics.movingSphereTargetIssued = 1;
		}

		metrics.actorCreated = 1;
		metrics.shapeAttached = 1;
		metrics.hostBuffersInitialized = 1;
		metrics.actorAdded = 1;
		scene->removeActor(*speculative.surface);
		if(speculative.surface->getScene() != NULL)
			break;
		metrics.actorRemoved = 1;
		if(!scene->addActor(*speculative.surface) ||
			speculative.surface->getScene() != scene)
			break;
		metrics.actorReadded = 1;

		if(rotationalCase)
		{
			const PxVec3 bladeLocal[4] =
			{
				heightFieldCase
					? PxVec3(0.0f, 0.0f, 0.0f)
					: PxVec3(-1.0f, 0.0f, -0.1f),
				heightFieldCase
					? PxVec3(0.0f, 0.0f, 0.2f)
					: PxVec3(-1.0f, 0.0f, 0.1f),
				heightFieldCase
					? PxVec3(2.0f, 0.0f, 0.2f)
					: PxVec3(1.0f, 0.0f, 0.1f),
				heightFieldCase
					? PxVec3(2.0f, 0.0f, 0.0f)
					: PxVec3(1.0f, 0.0f, -0.1f)
			};
			auto getBladeWorld = [&](
				PxReal time, PxVec3 world[4])
			{
				const PxQuat rotation = PxSlerp(
					time, rotationalStart,
					rotationalEnd).getNormalized();
				for(PxU32 vertex = 0; vertex < 4; ++vertex)
					world[vertex] =
						rigidStartCenters[0] +
							rotation.rotate(bladeLocal[vertex]);
			};
			auto getPointBladeDistance = [&](
				const PxVec3& point, PxReal time)
			{
				PxVec3 world[4];
				getBladeWorld(time, world);
				const PxVec3 closest0 =
					closestPointOnTriangleForConvexGate(
						point, world[0], world[1], world[2]);
				const PxVec3 closest1 =
					closestPointOnTriangleForConvexGate(
						point, world[0], world[2], world[3]);
				return PxMin(
					(point - closest0).magnitude(),
					(point - closest1).magnitude());
			};
			for(PxU32 endpoint = 0; endpoint < 2; ++endpoint)
			{
				const PxReal time = PxReal(endpoint);
				for(PxU32 softVertex = 0;
					softVertex < 4; ++softVertex)
					endpointMinSeparation = PxMin(
						endpointMinSeparation,
						getPointBladeDistance(
							speculative.initialPositions[
								softVertex],
							time));
				if(reverseCase)
				{
					PxVec3 world[4];
					getBladeWorld(time, world);
					for(PxU32 rigidVertex = 0;
						rigidVertex < 4; ++rigidVertex)
					{
						const PxVec3 closest0 =
							closestPointOnTriangleForConvexGate(
								world[rigidVertex],
								speculative.initialPositions[0],
								speculative.initialPositions[2],
								speculative.initialPositions[1]);
						const PxVec3 closest1 =
							closestPointOnTriangleForConvexGate(
								world[rigidVertex],
								speculative.initialPositions[2],
								speculative.initialPositions[3],
								speculative.initialPositions[1]);
						endpointMinSeparation = PxMin(
							endpointMinSeparation,
							PxMin(
								(world[rigidVertex] -
									closest0).magnitude(),
								(world[rigidVertex] -
									closest1).magnitude()));
					}
				}
			}
			const PxReal softMinX =
				surfaceOffsets[0].x;
			const PxReal softMaxX =
				surfaceOffsets[0].x + 1.0f;
			const PxReal softMinZ =
				surfaceOffsets[0].z;
			const PxReal softMaxZ =
				surfaceOffsets[0].z + 1.0f;
			for(PxU32 sample = 0; sample <= 256; ++sample)
			{
				const PxReal time = PxReal(sample) / 256.0f;
				if(reverseCase)
				{
					PxVec3 world[4];
					getBladeWorld(time, world);
					for(PxU32 rigidVertex = 0;
						rigidVertex < 4; ++rigidVertex)
					{
						if(world[rigidVertex].x >= softMinX &&
							world[rigidVertex].x <= softMaxX &&
							world[rigidVertex].z >= softMinZ &&
							world[rigidVertex].z <= softMaxZ)
							midSweepMinSeparation = PxMin(
								midSweepMinSeparation,
								surfaceOffsets[0].y -
									world[rigidVertex].y);
					}
					for(PxU32 softVertex = 0;
						softVertex < 4; ++softVertex)
						minimumVertexSweepSeparation = PxMin(
							minimumVertexSweepSeparation,
							getPointBladeDistance(
								speculative.initialPositions[
									softVertex],
								time));
				}
				else
				{
					for(PxU32 softVertex = 0;
						softVertex < 4; ++softVertex)
						midSweepMinSeparation = PxMin(
							midSweepMinSeparation,
							getPointBladeDistance(
								speculative.initialPositions[
									softVertex],
								time));
				}
			}
			vertexSweepExcluded =
				!reverseCase ||
				minimumVertexSweepSeparation > 0.10f
					? 1u : 0u;
		}

		if(reverseCase)
		{
			const PxVec3 rigidTipStart =
				heightFieldCase
					? rigidStartCenters[0] +
						PxVec3(0.3f, 0.3f, 0.3f)
					: rigidStartCenters[0] +
						PxVec3(0.0f, 0.3f, 0.0f);
			const PxVec3 rigidTipEnd =
				rigidTipStart +
					PxVec3(0.0f,
						kinematicTarget
							? endY - startY : 0.0f,
						0.0f);
			for(PxU32 vertex = 0;
				vertex < 4 && !rotationalCase; ++vertex)
			{
				const PxVec3 softStart =
					speculative.initialPositions[vertex];
				const PxVec3 softEnd =
					softStart +
						(deformingSoftTarget
							? ((vertex == 0 || vertex == 2)
								? PxVec3(0.0f, -3.0f, 0.0f)
								: PxVec3(0.0f))
							: staticTarget
							? PxVec3(0.0f, -2.2f, 0.0f)
							: PxVec3(0.0f));
				const PxVec3 relativeEnd =
					rigidTipEnd - (softEnd - softStart);
				const PxVec3 segment =
					relativeEnd - rigidTipStart;
				const PxReal denominator =
					segment.magnitudeSquared();
				const PxReal time =
					denominator > 1.0e-12f
						? PxClamp(
							(softStart - rigidTipStart).
								dot(segment) /
								denominator,
							0.0f, 1.0f)
						: 0.0f;
				minimumVertexSweepSeparation = PxMin(
					minimumVertexSweepSeparation,
					(softStart -
						(rigidTipStart +
							segment * time)).magnitude());
			}
			if(!rotationalCase)
				vertexSweepExcluded =
					minimumVertexSweepSeparation > 0.10f
						? 1u : 0u;
			if(deformingSoftTarget)
			{
				for(PxU32 sample = 0; sample <= 128; ++sample)
				{
					const PxReal time =
						PxReal(sample) / 128.0f;
					PxVec3 points[4];
					for(PxU32 vertex = 0; vertex < 4; ++vertex)
					{
						const PxVec3 displacement =
							(vertex == 0 || vertex == 2)
								? PxVec3(0.0f, -3.0f, 0.0f)
								: PxVec3(0.0f);
						points[vertex] =
							speculative.initialPositions[vertex] +
								displacement * time;
					}
					const PxVec3 closest0 =
						closestPointOnTriangleForConvexGate(
							rigidTipStart,
							points[0], points[2], points[1]);
					const PxVec3 closest1 =
						closestPointOnTriangleForConvexGate(
							rigidTipStart,
							points[2], points[3], points[1]);
					const PxReal distance = PxMin(
						(rigidTipStart - closest0).magnitude(),
						(rigidTipStart - closest1).magnitude());
					if(sample == 0 || sample == 128)
						endpointMinSeparation = PxMin(
							endpointMinSeparation, distance);
					else
						midSweepMinSeparation = PxMin(
							midSweepMinSeparation, distance);
				}
			}
		}

		const PxVec3 positiveInitial =
			getMotionSurfaceCentroid(speculative);
		const PxVec3 negativeInitial =
			getMotionSurfaceCentroid(discrete);
		for(PxU32 frame = 0;
			frame < options.frames; ++frame)
		{
			scene->simulate(options.dt);
			if(!scene->fetchResults(true))
			{
				++metrics.fetchFailures;
				break;
			}
			if(frame != 0)
				continue;
			const PxVec3 positiveCentroid =
				getMotionSurfaceCentroid(speculative);
			const PxVec3 negativeCentroid =
				getMotionSurfaceCentroid(discrete);
			positiveDisplacement =
				(positiveCentroid - positiveInitial).magnitude();
			negativeDisplacement =
				(negativeCentroid - negativeInitial).magnitude();
			responseDelta = 0.0f;
			for(PxU32 vertex = 0; vertex < 4; ++vertex)
			{
				const PxVec3 positiveLocal =
					speculative.positions[vertex].getXYZ() -
						PxVec3(surfaceOffsets[0].x, 0.0f, 0.0f);
				const PxVec3 negativeLocal =
					discrete.positions[vertex].getXYZ() -
						PxVec3(surfaceOffsets[1].x, 0.0f, 0.0f);
				responseDelta = PxMax(
					responseDelta,
					(positiveLocal - negativeLocal).magnitude());
			}
			positiveDrop =
				positiveInitial.y - positiveCentroid.y;
			negativeDrop =
				negativeInitial.y - negativeCentroid.y;
			if(rotationalCase)
			{
				auto getAngularTravel = [&](
					const PxQuat& endRotation)
				{
					const PxReal alignment = PxClamp(
						PxAbs(
							rotationalStart.dot(
								endRotation.getNormalized())),
						0.0f, 1.0f);
					return 2.0f * PxAcos(alignment);
				};
				positiveAngularTravel = getAngularTravel(
					kinematicActors[0]->getGlobalPose().q);
				negativeAngularTravel = getAngularTravel(
					kinematicActors[1]->getGlobalPose().q);
			}
			for(PxU32 vertex = 0; vertex < 4; ++vertex)
			{
				if(!speculative.positions[vertex].isFinite() ||
					!speculative.velocities[vertex].isFinite() ||
					!discrete.positions[vertex].isFinite() ||
					!discrete.velocities[vertex].isFinite())
				{
					finite = false;
					++metrics.nonFiniteSamples;
				}
			}
			boundsFinite =
				speculative.surface->getWorldBounds().isValid() &&
				discrete.surface->getWorldBounds().isValid();
			if(staticTarget)
			{
				if(deformingSoftTarget)
				{
					responseObserved =
						responseDelta > 0.01f ? 1u : 0u;
					negativeControlPassed =
						negativeDrop > 0.25f ? 1u : 0u;
				}
				else
				{
					const PxReal controlSeparation =
						reverseCase ? 0.02f : 0.10f;
					responseObserved =
						positiveDisplacement > 0.02f &&
						positiveDrop + controlSeparation <
							negativeDrop
							? 1u : 0u;
					negativeControlPassed =
						negativeDrop > 1.5f ? 1u : 0u;
				}
			}
			else
			{
				const PxReal responseThreshold =
					reverseCase
						? (rotationalCase ? 2.0e-3f : 0.01f)
						: (rotationalCase ? 2.0e-3f : 0.02f);
				responseObserved =
					positiveDisplacement >
						responseThreshold ? 1u : 0u;
				negativeControlPassed =
					negativeDisplacement < 5.0e-3f
						? 1u : 0u;
			}
			scene->removeActor(*speculative.surface);
			scene->removeActor(*discrete.surface);
			scene->removeActor(*rigidActors[0]);
			scene->removeActor(*rigidActors[1]);
		}

		metrics.speculativeCcdPreventedTunneling =
			responseObserved;
		metrics.speculativeCcdNegativeControlTunneled =
			negativeControlPassed;
		metrics.pinnedStable = 1;
		metrics.dynamicMoved = responseObserved;
		metrics.boundsFinite = boundsFinite ? 1u : 0u;
		metrics.maxDynamicDisplacement =
			positiveDisplacement;
		metrics.initialDynamicCentroidY = positiveInitial.y;
		metrics.finalDynamicCentroidY =
			getMotionSurfaceCentroid(speculative).y;
		metrics.minY = PxMin(
			getMotionSurfaceCentroid(speculative).y,
			getMotionSurfaceCentroid(discrete).y);
		metrics.finalMinY = metrics.minY;
		metrics.maxSpeed = PxMax(
			getMotionSurfaceMaxSpeed(speculative),
			getMotionSurfaceMaxSpeed(discrete));
		metrics.finalMaxSpeed = metrics.maxSpeed;
		success = finite &&
			metrics.speculativeCcdFlagApplied &&
			responseObserved && negativeControlPassed &&
			vertexSweepExcluded &&
			(!deformingSoftTarget ||
				(PxIsFinite(endpointMinSeparation) &&
				 endpointMinSeparation > 0.10f &&
				 PxIsFinite(midSweepMinSeparation) &&
				 midSweepMinSeparation < 0.01f)) &&
			(!rotationalCase ||
				(PxIsFinite(endpointMinSeparation) &&
				 endpointMinSeparation > 0.10f &&
				 PxIsFinite(midSweepMinSeparation) &&
				 (reverseCase
					? midSweepMinSeparation < -0.05f
					: midSweepMinSeparation < 0.01f) &&
				 PxIsFinite(positiveAngularTravel) &&
				 PxIsFinite(negativeAngularTravel) &&
				 PxAbs(
					positiveAngularTravel -
						(rotationalEndAngle -
							rotationalStartAngle)) < 0.002f &&
				 PxAbs(
					negativeAngularTravel -
						(rotationalEndAngle -
							rotationalStartAngle)) < 0.002f)) &&
			metrics.fetchFailures == 0 &&
			metrics.nonFiniteSamples == 0 &&
			metrics.boundsFinite;
	}
	while(false);

	for(PxU32 i = 0; i < 2; ++i)
		PX_RELEASE(rigidActors[i]);
	PX_RELEASE(heightField);
	PX_RELEASE(triangleMesh);
	releaseMotionSurfaceFixture(discrete);
	releaseMotionSurfaceFixture(speculative);
	PX_RELEASE(rigidMaterial);
	PX_RELEASE(scene);
	PX_RELEASE(dispatcher);
	if(extensionsInitialized)
		PxCloseExtensions();
	PX_RELEASE(physics);
	PX_RELEASE(foundation);
	metrics.cleanupComplete = 1;
	std::printf(
		"[%s] frames=%u target=%s geometry=%s "
		"responseObserved=%u negativeControlPassed=%u "
		"vertexSweepExcluded=%u nonFiniteSamples=%u "
		"positiveDisplacement=%.9g negativeDisplacement=%.9g "
		"positiveDrop=%.9g negativeDrop=%.9g "
		"minimumVertexSweepSeparation=%.9g result=%s\n",
		reverseCase
			? "AVBD_TRIANGLE_SURFACE_REVERSE_SWEPT"
			: "AVBD_TRIANGLE_SURFACE_FORWARD_SWEPT",
		options.frames,
		staticTarget ? "static" : "kinematic",
		heightFieldCase ? "heightfield" : "triangle-mesh",
		responseObserved, negativeControlPassed,
		vertexSweepExcluded, metrics.nonFiniteSamples,
		double(positiveDisplacement),
		double(negativeDisplacement),
		double(positiveDrop), double(negativeDrop),
		double(minimumVertexSweepSeparation),
		success ? "PASS" : "FAIL");
	if(rotationalCase)
	{
		std::printf(
			"[AVBD_TRIANGLE_SURFACE_ROTATIONAL_SWEPT] "
			"frames=%u target=kinematic geometry=%s owner=%s "
			"responseObserved=%u negativeControlPassed=%u "
			"vertexSweepExcluded=%u "
			"endpointMinSeparation=%.9g "
			"midSweepMinSeparation=%.9g "
			"minimumVertexSweepSeparation=%.9g "
			"positiveDisplacement=%.9g "
			"negativeDisplacement=%.9g "
			"positiveAngularTravel=%.9g "
			"negativeAngularTravel=%.9g result=%s\n",
			options.frames,
			heightFieldCase ? "heightfield" : "triangle-mesh",
			reverseCase ? "reverse" : "forward",
			responseObserved, negativeControlPassed,
			vertexSweepExcluded,
			double(endpointMinSeparation),
			double(midSweepMinSeparation),
			double(minimumVertexSweepSeparation),
			double(positiveDisplacement),
			double(negativeDisplacement),
			double(positiveAngularTravel),
			double(negativeAngularTravel),
			success ? "PASS" : "FAIL");
	}
	if(deformingSoftTarget)
	{
		std::printf(
			"[AVBD_DEFORMING_SOFT_TRIANGLE_SURFACE_REVERSE_SWEPT] "
			"frames=%u geometry=%s target=static owner=reverse "
			"responseObserved=%u negativeControlPassed=%u "
			"vertexSweepExcluded=%u "
			"endpointMinSeparation=%.9g "
			"midSweepMinSeparation=%.9g "
			"minimumVertexSweepSeparation=%.9g "
			"responseDelta=%.9g "
			"positiveDisplacement=%.9g "
			"negativeDisplacement=%.9g result=%s\n",
			options.frames,
			heightFieldCase ? "heightfield" : "triangle-mesh",
			responseObserved, negativeControlPassed,
			vertexSweepExcluded,
			double(endpointMinSeparation),
			double(midSweepMinSeparation),
			double(minimumVertexSweepSeparation),
			double(responseDelta),
			double(positiveDisplacement),
			double(negativeDisplacement),
			success ? "PASS" : "FAIL");
	}
	return success;
}

static bool runSmoothReverseFeatureCase(
	const Snippets::HeadlessOptions& options,
	Snippets::TrackingErrorCallback& errorCallback,
	Metrics& metrics,
	bool capsuleCase,
	bool convexCase,
	bool triangleMeshCase,
	bool heightFieldCase)
{
	PxDefaultAllocator allocator;
	PxFoundation* foundation = PxCreateFoundation(
		PX_PHYSICS_VERSION, allocator, errorCallback);
	if(!foundation)
		return false;

	PxPhysics* physics = NULL;
	PxDefaultCpuDispatcher* dispatcher = NULL;
	PxScene* scene = NULL;
	PxMaterial* rigidMaterial = NULL;
	PxRigidStatic* rigid = NULL;
	PxConvexMesh* convexMesh = NULL;
	PxConvexMeshGeometry convexGeometry;
	PxTriangleMesh* rigidTriangleMesh = NULL;
	PxTriangleMeshGeometry triangleMeshGeometry;
	PxHeightField* rigidHeightField = NULL;
	PxHeightFieldGeometry heightFieldGeometry;
	MotionSurfaceFixture reverseFeature;
	MotionSurfaceFixture freeControl;
	bool extensionsInitialized = false;
	bool finite = true;
	PxReal positiveDisplacement = 0.0f;
	PxReal positiveDrop = 0.0f;
	PxReal negativeDrop = 0.0f;
	PxReal faceSeparation = -PX_MAX_F32;
	PxReal minimumVertexSeparation = PX_MAX_F32;
	PxU32 faceResponseObserved = 0;
	PxU32 vertexSdfExcluded = 0;
	PxU32 negativeControlPassed = 0;
	bool success = false;

	do
	{
		const PxTolerancesScale scale;
		physics = PxCreatePhysics(
			PX_PHYSICS_VERSION, *foundation, scale, true, NULL);
		if(!physics)
			break;
		extensionsInitialized = PxInitExtensions(*physics, NULL);
		if(!extensionsInitialized)
			break;

		PxSceneDesc sceneDesc(scale);
		sceneDesc.gravity = PxVec3(0.0f);
		sceneDesc.solverType = PxSolverType::eAVBD;
		sceneDesc.filterShader = PxDefaultSimulationFilterShader;
		dispatcher = PxDefaultCpuDispatcherCreate(
			options.dispatcherThreads);
		if(!dispatcher)
			break;
		sceneDesc.cpuDispatcher = dispatcher;
		scene = physics->createScene(sceneDesc);
		if(!scene)
			break;

		rigidMaterial = physics->createMaterial(
			0.0f, 0.0f, 0.0f);
		if(!rigidMaterial)
			break;

		const PxVec3 positiveOffset(-2.0f, 0.34f, 0.0f);
		const PxVec3 negativeOffset(1.0f, 0.34f, 0.0f);
		const PxVec3 initialVelocity(0.0f, -2.0f, 0.0f);
		if(!createMotionSurfaceFixture(
				*physics, *scene, scale, positiveOffset,
				initialVelocity, PX_MAX_F32,
				0.0f, 0.0f, 0.0f, 1.0f,
				reverseFeature) ||
			!createMotionSurfaceFixture(
				*physics, *scene, scale, negativeOffset,
				initialVelocity, PX_MAX_F32,
				0.0f, 0.0f, 0.0f, 1.0f,
				freeControl))
			break;

		const PxReal radius = 0.3f;
		const PxReal halfHeight = capsuleCase ? 0.2f : 0.0f;
		const PxVec3 rigidCenter = triangleMeshCase
			? PxVec3(-1.6f, 0.0f, 0.4f)
			: heightFieldCase
				? PxVec3(-1.9f, 0.0f, 0.1f)
				: PxVec3(-1.5f, 0.0f, 0.5f);
		const PxVec3 convexVertices[] =
		{
			PxVec3(0.0f, 0.3f, 0.0f),
			PxVec3(-0.3f, -0.3f, -0.3f),
			PxVec3(0.3f, -0.3f, -0.3f),
			PxVec3(0.0f, -0.3f, 0.35f)
		};
		if(convexCase)
		{
			convexMesh = createAvbdTestConvexMesh(
				*physics, scale, convexVertices,
				sizeof(convexVertices) /
					sizeof(convexVertices[0]));
			if(!convexMesh)
				break;
			convexGeometry =
				PxConvexMeshGeometry(convexMesh);
		}
		if(triangleMeshCase)
		{
			rigidTriangleMesh =
				createAvbdRigidTriangleMesh(
					*physics, scale, true);
			if(!rigidTriangleMesh)
				break;
			triangleMeshGeometry =
				PxTriangleMeshGeometry(
					rigidTriangleMesh);
		}
		if(heightFieldCase)
		{
			rigidHeightField =
				createAvbdRigidHeightField(
					*physics, true);
			if(!rigidHeightField)
				break;
			heightFieldGeometry =
				PxHeightFieldGeometry(
					rigidHeightField,
					PxMeshGeometryFlags(),
					0.1f, 0.3f, 0.3f);
		}
		rigid = physics->createRigidStatic(
			PxTransform(rigidCenter));
		const bool shapeCreated = rigid &&
			(triangleMeshCase
				? PxRigidActorExt::createExclusiveShape(
					*rigid, triangleMeshGeometry,
					*rigidMaterial) != NULL
				: heightFieldCase
				? PxRigidActorExt::createExclusiveShape(
					*rigid, heightFieldGeometry,
					*rigidMaterial) != NULL
				: convexCase
				? PxRigidActorExt::createExclusiveShape(
					*rigid, convexGeometry,
					*rigidMaterial) != NULL
				: capsuleCase
				? PxRigidActorExt::createExclusiveShape(
					*rigid,
					PxCapsuleGeometry(radius, halfHeight),
					*rigidMaterial) != NULL
				: PxRigidActorExt::createExclusiveShape(
					*rigid, PxSphereGeometry(radius),
					*rigidMaterial) != NULL);
		if(!shapeCreated || !scene->addActor(*rigid))
			break;

		const PxVec3 positiveInitial =
			getMotionSurfaceCentroid(reverseFeature);
		const PxVec3 negativeInitial =
			getMotionSurfaceCentroid(freeControl);
		for(PxU32 frame = 0; frame < options.frames; ++frame)
		{
			scene->simulate(options.dt);
			if(!scene->fetchResults(true))
			{
				++metrics.fetchFailures;
				break;
			}
			if(frame != 0)
				continue;

			const PxVec3 positiveCentroid =
				getMotionSurfaceCentroid(reverseFeature);
			const PxVec3 negativeCentroid =
				getMotionSurfaceCentroid(freeControl);
			positiveDisplacement =
				(positiveCentroid - positiveInitial).magnitude();
			positiveDrop =
				positiveInitial.y - positiveCentroid.y;
			negativeDrop =
				negativeInitial.y - negativeCentroid.y;
			const PxVec3 reverseEdgePoint =
				0.5f * (
					reverseFeature.positions[1].getXYZ() +
					reverseFeature.positions[2].getXYZ());
			if(triangleMeshCase || heightFieldCase)
			{
				const PxVec3 rigidTip = triangleMeshCase
					? rigidCenter + PxVec3(0.0f, 0.3f, 0.0f)
					: rigidCenter + PxVec3(0.3f, 0.3f, 0.3f);
				const PxVec3 closest =
					closestPointOnTriangleForConvexGate(
						rigidTip,
						reverseFeature.positions[0].getXYZ(),
						reverseFeature.positions[2].getXYZ(),
						reverseFeature.positions[1].getXYZ());
				faceSeparation =
					(closest - rigidTip).magnitude();
			}
			else if(convexCase)
				faceSeparation =
					(reverseEdgePoint -
						(rigidCenter + convexVertices[0])).
						magnitude();
			else
			{
				PxVec3 edgeRadial =
					reverseEdgePoint - rigidCenter;
				edgeRadial.x -= PxClamp(
					edgeRadial.x, -halfHeight, halfHeight);
				faceSeparation =
					edgeRadial.magnitude() - radius;
			}
			for(PxU32 vertex = 0; vertex < 4; ++vertex)
			{
				const PxVec3 position =
					reverseFeature.positions[vertex].getXYZ();
				const PxVec3 velocity =
					reverseFeature.velocities[vertex].getXYZ();
				const PxVec3 negativePosition =
					freeControl.positions[vertex].getXYZ();
				const PxVec3 negativeVelocity =
					freeControl.velocities[vertex].getXYZ();
				if(!position.isFinite() ||
					!velocity.isFinite() ||
					!negativePosition.isFinite() ||
					!negativeVelocity.isFinite())
				{
					finite = false;
					++metrics.nonFiniteSamples;
				}
				minimumVertexSeparation = PxMin(
					minimumVertexSeparation,
					[&]()
					{
						if(triangleMeshCase ||
							heightFieldCase)
						{
							const PxTransform rigidPose(
								rigidCenter);
							PxReal minimumDistance =
								PX_MAX_F32;
							const PxU32 triangleCount =
								triangleMeshCase
									? rigidTriangleMesh->
										getNbTriangles()
									: 8u;
							for(PxU32 triangleIndex = 0;
								triangleIndex < triangleCount;
								++triangleIndex)
							{
								PxTriangle triangle;
								if(triangleMeshCase)
									PxMeshQuery::getTriangle(
										triangleMeshGeometry,
										rigidPose,
										triangleIndex,
										triangle);
								else
								{
									const PxU32 heightFieldTriangles[] =
										{0, 1, 2, 3, 6, 7, 8, 9};
									PxMeshQuery::getTriangle(
										heightFieldGeometry,
										rigidPose,
										heightFieldTriangles[
											triangleIndex],
										triangle);
								}
								const PxVec3 closest =
									closestPointOnTriangleForConvexGate(
										position,
										triangle.verts[0],
										triangle.verts[1],
										triangle.verts[2]);
								minimumDistance = PxMin(
									minimumDistance,
									(position - closest).
										magnitude());
							}
							return minimumDistance;
						}
						if(convexCase)
						{
							const PxReal squaredDistance =
								PxGeometryQuery::pointDistance(
									position,
									convexGeometry,
									PxTransform(rigidCenter));
							return squaredDistance >= 0.0f
								? PxSqrt(squaredDistance)
								: -PX_MAX_F32;
						}
						PxVec3 radial = position - rigidCenter;
						radial.x -= PxClamp(
							radial.x,
							-halfHeight, halfHeight);
						return radial.magnitude() - radius;
					}());
			}
			vertexSdfExcluded =
				minimumVertexSeparation > 0.10f ? 1u : 0u;
			negativeControlPassed =
				negativeDrop > 0.02f ? 1u : 0u;
			faceResponseObserved =
				positiveDisplacement > 1.0e-3f &&
				faceSeparation > 0.02f &&
				positiveDrop + 0.01f < negativeDrop
					? 1u : 0u;

			scene->removeActor(*reverseFeature.surface);
			scene->removeActor(*freeControl.surface);
			scene->removeActor(*rigid);
		}

		success =
			finite &&
			faceResponseObserved == 1 &&
			vertexSdfExcluded == 1 &&
			negativeControlPassed == 1 &&
			metrics.fetchFailures == 0 &&
			metrics.nonFiniteSamples == 0;
	}
	while(false);

	PX_RELEASE(rigid);
	PX_RELEASE(rigidHeightField);
	PX_RELEASE(rigidTriangleMesh);
	PX_RELEASE(convexMesh);
	releaseMotionSurfaceFixture(freeControl);
	releaseMotionSurfaceFixture(reverseFeature);
	PX_RELEASE(rigidMaterial);
	PX_RELEASE(scene);
	PX_RELEASE(dispatcher);
	if(extensionsInitialized)
		PxCloseExtensions();
	PX_RELEASE(physics);
	PX_RELEASE(foundation);
	metrics.cleanupComplete = 1;
	const char* reverseFeatureTag = triangleMeshCase
		? "AVBD_TRIANGLE_MESH_REVERSE_FEATURE"
		: heightFieldCase
			? "AVBD_HEIGHTFIELD_REVERSE_FEATURE"
			: convexCase
		? "AVBD_CONVEX_REVERSE_FEATURE"
		: capsuleCase
			? "AVBD_CAPSULE_REVERSE_FEATURE"
			: "AVBD_SPHERE_REVERSE_FEATURE";
	std::printf(
		"[%s] frames=%u "
		"faceResponseObserved=%u vertexSdfExcluded=%u "
		"negativeControlPassed=%u nonFiniteSamples=%u "
		"positiveDisplacement=%.9g positiveDrop=%.9g "
		"negativeDrop=%.9g faceSeparation=%.9g "
		"minimumVertexSeparation=%.9g result=%s\n",
		reverseFeatureTag, options.frames,
		faceResponseObserved, vertexSdfExcluded,
		negativeControlPassed, metrics.nonFiniteSamples,
		double(positiveDisplacement), double(positiveDrop),
		double(negativeDrop), double(faceSeparation),
		double(minimumVertexSeparation),
		success ? "PASS" : "FAIL");
	return success;
}

static bool runSurfaceCase(
	const Snippets::HeadlessOptions& options,
	Snippets::TrackingErrorCallback& errorCallback,
	Metrics& metrics,
	bool groundCase,
	bool sleepWakeCase,
	bool bufferMutationCase,
	bool dynamicRigidCase,
	bool kinematicRigidCase,
	bool softSoftCase,
	bool surfaceVolumeCase,
	bool selfCollisionCase,
	bool materialFrictionCase,
	bool worldPinCase,
	bool rigidAttachmentCase,
	bool staticAttachmentCase,
	bool kinematicAttachmentCase,
	bool articulationAttachmentCase,
	bool elementFilterCase,
	bool partialElementFilterCase,
	bool softSoftElementFilterCase,
	bool skinningCase)
{
	const bool kinematicSphereCase =
		options.caseName == "surface-kinematic-sphere";
	const bool dynamicSphereCase =
		options.caseName == "surface-dynamic-sphere";
	const bool kinematicCapsuleCase =
		options.caseName == "surface-kinematic-capsule";
	const bool dynamicCapsuleCase =
		options.caseName == "surface-dynamic-capsule";
	const bool kinematicConvexCase =
		options.caseName == "surface-kinematic-convex";
	const bool dynamicConvexCase =
		options.caseName == "surface-dynamic-convex";
	const bool kinematicTriangleMeshCase =
		options.caseName == "surface-kinematic-triangle-mesh";
	const bool kinematicHeightFieldCase =
		options.caseName == "surface-kinematic-heightfield";
	const bool kinematicSmoothCase =
		kinematicSphereCase || kinematicCapsuleCase ||
		kinematicConvexCase || kinematicTriangleMeshCase ||
		kinematicHeightFieldCase;
	const bool dynamicSmoothCase =
		dynamicSphereCase || dynamicCapsuleCase ||
		dynamicConvexCase;
	const bool selfCollisionFilterCase =
		options.caseName == "surface-self-collision-filter";
	const bool selfCollisionSweptCase =
		options.caseName ==
			"surface-self-collision-swept-ccd";
	const bool softSoftSweptCase =
		options.caseName ==
			"surface-soft-soft-swept-ccd";
	const bool surfaceSurfaceAttachmentCase =
		options.caseName == "surface-surface-attachment";
	const bool surfaceVolumeAttachmentCase =
		options.caseName == "surface-volume-attachment";
	const bool softPairAttachmentCase =
		surfaceSurfaceAttachmentCase ||
		surfaceVolumeAttachmentCase;
	const bool worldElementAttachmentCase =
		options.caseName == "surface-world-element-attachment";
	const bool rigidElementAttachmentCase =
		options.caseName == "surface-rigid-element-attachment";
	const bool staticElementAttachmentCase =
		options.caseName == "surface-static-element-attachment";
	const bool kinematicElementAttachmentCase =
		options.caseName == "surface-kinematic-element-attachment";
	const bool articulationElementAttachmentCase =
		options.caseName ==
			"surface-articulation-element-attachment";
	const bool elementAttachmentCase =
		worldElementAttachmentCase ||
		rigidElementAttachmentCase ||
		staticElementAttachmentCase ||
		kinematicElementAttachmentCase ||
		articulationElementAttachmentCase ||
		softPairAttachmentCase;
	PxDefaultAllocator allocator;
	PxFoundation* foundation = PxCreateFoundation(
		PX_PHYSICS_VERSION, allocator, errorCallback);
	if(!foundation)
		return false;

	PxPhysics* physics = NULL;
	PxDefaultCpuDispatcher* dispatcher = NULL;
	PxScene* scene = NULL;
	PxTriangleMesh* triangleMesh = NULL;
	PxDeformableSurfaceMaterial* surfaceMaterial = NULL;
	PxDeformableSurface* surface = NULL;
	PxDeformableSurface* secondSurface = NULL;
	PxDeformableVolumeMesh* mixedVolumeMesh = NULL;
	PxDeformableVolumeMaterial* mixedVolumeMaterial = NULL;
	PxDeformableVolume* mixedVolume = NULL;
	PxMaterial* rigidMaterial = NULL;
	PxConvexMesh* rigidConvexMesh = NULL;
	PxTriangleMesh* rigidTriangleMesh = NULL;
	PxHeightField* rigidHeightField = NULL;
	PxRigidStatic* groundPlane = NULL;
	PxRigidDynamic* dynamicBox = NULL;
	PxRigidDynamic* attachmentBox = NULL;
	PxRigidStatic* attachmentStatic = NULL;
	PxArticulationReducedCoordinate* attachmentArticulation = NULL;
	PxArticulationLink* attachmentRoot = NULL;
	PxArticulationLink* attachmentLink = NULL;
	PxRigidBody* attachmentBody = NULL;
	PxDeformableAttachment* worldAttachment = NULL;
	PxDeformableAttachment* rigidAttachment = NULL;
	PxDeformableAttachment* softPairAttachment = NULL;
	PxDeformableElementFilter* elementFilter = NULL;
	PxVec4* secondPositions = NULL;
	PxVec4* secondVelocities = NULL;
	PxVec4* mixedVolumePositions = NULL;
	PxVec4* mixedVolumeVelocities = NULL;
	PxU32 mixedVolumeVertexCount = 0;
	PxU32 softPairTargetVertices[4] =
		{PX_MAX_U32, PX_MAX_U32, PX_MAX_U32, PX_MAX_U32};
	PxU32 softPairTargetVertexCount = 0;
	PxVec3 rigidAttachmentInitialPosition(0.0f);
	const PxVec3 rigidAttachmentLocalOffset =
		articulationAttachmentCase
			? PxVec3(0.0f)
			: PxVec3(0.0f, 0.5f, 0.0f);
	PxTransform articulationRootInitialPose(PxIdentity);
	PxTransform articulationChildInitialPose(PxIdentity);
	bool extensionsInitialized = false;
	bool success = false;
	PxArray<Snippets::AvbdTriangleSkinningBinding>
		skinningBindings;
	PxArray<PxU32> skinningTriangles;
	PxArray<PxVec3> skinningPositions;
	PxArray<PxVec3> skinningNormals;
	PxArray<PxVec3> skinningInitialPositions;

	do
	{
		const PxTolerancesScale scale;
		physics = PxCreatePhysics(
			PX_PHYSICS_VERSION, *foundation, scale, true, NULL);
		if(!physics)
			break;
		extensionsInitialized = PxInitExtensions(*physics, NULL);
		if(!extensionsInitialized)
			break;

		PxSceneDesc sceneDesc(scale);
		sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
		sceneDesc.solverType = PxSolverType::eAVBD;
		if(elementFilterCase || materialFrictionCase)
			sceneDesc.flags |= PxSceneFlag::eDISABLE_SLEEPING;
		sceneDesc.filterShader = PxDefaultSimulationFilterShader;
		dispatcher = PxDefaultCpuDispatcherCreate(
			options.dispatcherThreads);
		if(!dispatcher)
			break;
		sceneDesc.cpuDispatcher = dispatcher;
		scene = physics->createScene(sceneDesc);
		if(!scene)
			break;
		const bool boundedContactCase =
			groundCase || dynamicRigidCase || kinematicRigidCase ||
			softSoftCase ||
			surfaceVolumeCase || worldPinCase ||
			rigidAttachmentCase || staticAttachmentCase ||
			kinematicAttachmentCase ||
			articulationAttachmentCase ||
			elementFilterCase || materialFrictionCase;
		if(boundedContactCase)
		{
			rigidMaterial = materialFrictionCase
				? physics->createMaterial(0.0f, 0.0f, 0.0f)
				: physics->createMaterial(0.6f, 0.5f, 0.0f);
			if(!rigidMaterial)
				break;
			if(!softSoftElementFilterCase &&
				!softPairAttachmentCase)
			{
				const PxReal planeHeight =
					dynamicRigidCase ? -1.0f : 0.0f;
				groundPlane = PxCreatePlane(
					*physics,
					PxPlane(0.0f, 1.0f, 0.0f, -planeHeight),
					*rigidMaterial);
				if(!groundPlane || !scene->addActor(*groundPlane))
					break;
				metrics.groundAdded = 1;
			}
		}

		std::vector<PxVec3> initialPositions;
		std::vector<PxU32> triangles;
		if(!buildSurfaceMesh(
			*physics, scale, initialPositions,
			triangles, triangleMesh,
			groundCase || dynamicRigidCase || kinematicRigidCase ||
				softSoftCase ||
				surfaceVolumeCase || worldPinCase ||
				rigidAttachmentCase || staticAttachmentCase ||
				kinematicAttachmentCase ||
				articulationAttachmentCase ||
				elementFilterCase || materialFrictionCase,
			selfCollisionCase,
			partialElementFilterCase))
			break;
		if(selfCollisionSweptCase)
		{
			for(PxU32 i = 3;
				i < PxU32(initialPositions.size()); ++i)
				initialPositions[i].y = 0.2f;
		}
		if(materialFrictionCase)
		{
			for(PxU32 i = 0;
				i < PxU32(initialPositions.size()); ++i)
				initialPositions[i].y = 0.015f;
		}
		if(softSoftElementFilterCase)
		{
			// Soft-surface OGC is an outward contact shell rather than a
			// swept collision query. Start inside that shell so this gate
			// isolates element ownership instead of high-speed tunnelling.
			for(PxU32 i = 0;
				i < PxU32(initialPositions.size()); ++i)
				initialPositions[i].y = 0.015f;
		}

		surfaceMaterial = physics->createDeformableSurfaceMaterial(
			2.0e4f, 0.3f,
			surfaceVolumeCase ? 0.4f : 0.08f,
			0.02f, 0.2f);
		if(!surfaceMaterial)
			break;
		if(materialFrictionCase)
		{
			surfaceMaterial->setDynamicFriction(0.0f);
			metrics.materialFrictionLowApplied =
				PxAbs(surfaceMaterial->getDynamicFriction()) <=
					1.0e-6f
				? 1u : 0u;
		}

		surface = physics->createDeformableSurface(
			PxDeformableSurfaceBackend::eCPU_AVBD);
		if(!surface)
			break;
		metrics.actorCreated = 1;

		PxDeformableSurfaceMaterial* materials[] = {
			surfaceMaterial
		};
		const PxShapeFlags shapeFlags =
			PxShapeFlag::eSIMULATION_SHAPE |
			PxShapeFlag::eSCENE_QUERY_SHAPE;
		PxShape* shape = physics->createShape(
			PxTriangleMeshGeometry(triangleMesh),
			materials, 1, true, shapeFlags);
		if(!shape)
			break;
		const bool attached = surface->attachShape(*shape);
		shape->release();
		if(!attached)
			break;
		metrics.shapeAttached = 1;

		PxVec4* positions = surface->getPositionInvMassBufferH();
		PxVec4* velocities = surface->getVelocityBufferH();
		PxVec4* restPositions = surface->getRestPositionBufferH();
		if(!positions || !velocities || !restPositions)
			break;
		const PxU32 vertexCount = PxU32(initialPositions.size());
		bool partialFilteredVertex[6] =
			{false, false, false, false, false, false};
		if(partialElementFilterCase)
		{
			if(vertexCount != 6 ||
				triangleMesh->getNbTriangles() != 2)
				break;
			const bool has16BitIndices =
				triangleMesh->getTriangleMeshFlags() &
					PxTriangleMeshFlag::e16_BIT_INDICES;
			const PxU16* triangles16 = has16BitIndices
				? static_cast<const PxU16*>(
					triangleMesh->getTriangles())
				: NULL;
			const PxU32* triangles32 = has16BitIndices
				? NULL
				: static_cast<const PxU32*>(
					triangleMesh->getTriangles());
			bool partialFilterMappingValid = true;
			for(PxU32 i = 0; i < 3; ++i)
			{
				const PxU32 vertex = has16BitIndices
					? triangles16[i] : triangles32[i];
				if(vertex >= vertexCount)
				{
					partialFilterMappingValid = false;
					break;
				}
				partialFilteredVertex[vertex] = true;
			}
			if(!partialFilterMappingValid)
				break;
		}
		for(PxU32 i = 0; i < vertexCount; ++i)
		{
			const bool pinned = selfCollisionCase
				? i < 3
				: (!groundCase && !dynamicRigidCase &&
				   !kinematicRigidCase &&
				   !surfaceVolumeCase &&
				   !worldPinCase &&
				   !rigidAttachmentCase &&
				   !staticAttachmentCase &&
				   !kinematicAttachmentCase &&
				   !articulationAttachmentCase &&
				   !softPairAttachmentCase &&
				   !elementFilterCase &&
				   !materialFrictionCase &&
				   i < sGridWidth);
			positions[i] = PxVec4(
				initialPositions[i], pinned ? 0.0f : 1.0f);
			velocities[i] = materialFrictionCase
				? PxVec4(0.5f, 0.0f, 0.0f, 0.0f)
				: selfCollisionCase && !pinned
				? PxVec4(
					0.0f,
					selfCollisionSweptCase ? -30.0f : -0.5f,
					0.0f, 0.0f)
				: softSoftSweptCase
				? PxVec4(0.0f, -60.0f, 0.0f, 0.0f)
				: PxVec4(0.0f);
			restPositions[i] = PxVec4(initialPositions[i], 0.0f);
		}
		metrics.hostBuffersInitialized = 1;
		if(skinningCase)
		{
			for(PxU32 triangle = 0;
				triangle + 2 < PxU32(triangles.size());
				triangle += 3)
			{
				const PxU32 indices[3] =
				{
					triangles[triangle + 0],
					triangles[triangle + 1],
					triangles[triangle + 2]
				};
				Snippets::appendTriangleSkinningPatch(
					indices, 3, skinningBindings,
					skinningTriangles);
			}
			if(!Snippets::evaluateTriangleSkinning(
				positions, vertexCount, skinningBindings,
				skinningTriangles, skinningPositions,
				skinningNormals))
				break;
			skinningInitialPositions = skinningPositions;
			gSurfaceSkinningMetrics.initialized = 1;
			gSurfaceSkinningMetrics.vertices =
				skinningPositions.size();
			gSurfaceSkinningMetrics.triangles =
				skinningTriangles.size() / 3;
		}
		metrics.initialDynamicCentroidY =
			dynamicCentroidY(positions, vertexCount);
		surface->setLinearDamping(
			materialFrictionCase ? 0.0f :
			(softSoftCase ? 0.8f :
			((dynamicRigidCase || kinematicRigidCase)
				? (dynamicCapsuleCase ? 1.0f : 0.8f) :
				(boundedContactCase ? 0.2f : 0.08f))));
		// The friction gate compares an unconstrained tangential momentum
		// control against a high-friction contact. Give this convergence
		// probe enough position sweeps that under-converged membrane damping
		// cannot masquerade as contact friction.
		surface->setSolverIterationCounts(
			materialFrictionCase ? 32 : 8);
		surface->setSleepThreshold(
			boundedContactCase ? 5.0e-4f : 1.0e-8f);
		if(materialFrictionCase)
		{
			surface->setSettlingThreshold(0.0f);
			surface->setSettlingDamping(0.0f);
		}
		if(dynamicRigidCase)
		{
			// This mixed high-energy fixture relies on first-impact
			// prediction to retain its strict tail-speed bound.
			surface->setDeformableBodyFlag(
				PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD,
				true);
		}
		if(softSoftSweptCase)
		{
			surface->setDeformableBodyFlag(
				PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD,
				true);
			surface->setNbCollisionPairUpdatesPerTimestep(4);
			surface->setNbCollisionSubsteps(2);
		}
		if(kinematicRigidCase || rigidAttachmentCase ||
			staticAttachmentCase ||
			kinematicAttachmentCase || articulationAttachmentCase ||
			softPairAttachmentCase)
			surface->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
		if(selfCollisionCase)
		{
			const PxReal selfCollisionFilterDistance =
				selfCollisionFilterCase ? 0.1f : 0.0f;
			surface->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			surface->setDeformableBodyFlag(
				PxDeformableBodyFlag::eDISABLE_SELF_COLLISION,
				false);
			surface->setSelfCollisionFilterDistance(
				selfCollisionFilterDistance);
			if(selfCollisionSweptCase)
			{
				surface->setDeformableBodyFlag(
					PxDeformableBodyFlag::
						eENABLE_SPECULATIVE_CCD,
					true);
				surface->
					setNbCollisionPairUpdatesPerTimestep(4);
				surface->setNbCollisionSubsteps(2);
			}
			metrics.selfCollisionEnabled =
				(surface->getDeformableBodyFlags() &
					PxDeformableBodyFlag::eDISABLE_SELF_COLLISION)
				? 0u : 1u;
			metrics.selfCollisionFilterApplied =
				PxAbs(surface->getSelfCollisionFilterDistance() -
					selfCollisionFilterDistance) <= 1.0e-6f
				? 1u : 0u;
		}
		surface->markDirty(PxDeformableSurfaceDataFlag::eALL);

		if(!scene->addActor(*surface) || surface->getScene() != scene)
			break;
		metrics.actorAdded = 1;
		const PxU32 attachmentVertex = 0;
		if(elementAttachmentCase &&
			triangleMesh->getNbTriangles() == 0)
			break;
		PxU32 attachmentTriangleVertices[3] = {
			attachmentVertex, attachmentVertex, attachmentVertex};
		if(elementAttachmentCase)
		{
			const bool has16BitIndices =
				triangleMesh->getTriangleMeshFlags() &
					PxTriangleMeshFlag::e16_BIT_INDICES;
			if(has16BitIndices)
			{
				const PxU16* meshTriangles =
					static_cast<const PxU16*>(
						triangleMesh->getTriangles());
				for(PxU32 endpoint = 0; endpoint < 3; endpoint++)
					attachmentTriangleVertices[endpoint] =
						meshTriangles[endpoint];
			}
			else
			{
				const PxU32* meshTriangles =
					static_cast<const PxU32*>(
						triangleMesh->getTriangles());
				for(PxU32 endpoint = 0; endpoint < 3; endpoint++)
					attachmentTriangleVertices[endpoint] =
						meshTriangles[endpoint];
			}
		}
		const PxVec4 attachmentBarycentric(
			0.2f, 0.3f, 0.5f, 0.0f);
		const auto getAttachmentPoint = [&]() -> PxVec3
		{
			if(!elementAttachmentCase)
				return positions[attachmentVertex].getXYZ();
			return
				positions[attachmentTriangleVertices[0]].getXYZ() *
					attachmentBarycentric.x +
				positions[attachmentTriangleVertices[1]].getXYZ() *
					attachmentBarycentric.y +
				positions[attachmentTriangleVertices[2]].getXYZ() *
					attachmentBarycentric.z;
		};
		const PxVec3 attachmentTarget =
			getAttachmentPoint();
		if(worldPinCase)
		{
			PxDeformableAttachmentData attachmentData;
			PxU32 attachmentIndex =
				worldElementAttachmentCase ? 0u : attachmentVertex;
			PxVec4 worldCoordinate(attachmentTarget, 0.0f);
			attachmentData.actor[0] = surface;
			attachmentData.type[0] =
				worldElementAttachmentCase
					? PxDeformableAttachmentTargetType::eTRIANGLE
					: PxDeformableAttachmentTargetType::eVERTEX;
			attachmentData.indices[0].data = &attachmentIndex;
			attachmentData.indices[0].count = 1;
			if(worldElementAttachmentCase)
			{
				attachmentData.coords[0].data =
					&attachmentBarycentric;
				attachmentData.coords[0].count = 1;
			}
			attachmentData.actor[1] = NULL;
			attachmentData.type[1] =
				PxDeformableAttachmentTargetType::eWORLD;
			attachmentData.coords[1].data = &worldCoordinate;
			attachmentData.coords[1].count = 1;
			worldAttachment =
				physics->createDeformableAttachment(
					attachmentData);
			if(!worldAttachment)
				break;
			metrics.attachmentCreated = 1;
		}
		if(rigidAttachmentCase || staticAttachmentCase ||
			kinematicAttachmentCase ||
			articulationAttachmentCase)
		{
			rigidAttachmentInitialPosition =
				attachmentTarget - rigidAttachmentLocalOffset;
			PxRigidActor* attachmentActor = NULL;
			if(articulationAttachmentCase)
			{
				attachmentArticulation =
					physics->createArticulationReducedCoordinate();
				if(!attachmentArticulation)
					break;
				metrics.articulationCreated = 1;
				attachmentArticulation->setArticulationFlag(
					PxArticulationFlag::eFIX_BASE, true);
				attachmentArticulation->setSolverIterationCounts(
					16, 1);
				attachmentArticulation->setSleepThreshold(5.0e-4f);
				articulationRootInitialPose =
					PxTransform(rigidAttachmentInitialPosition);
				articulationChildInitialPose =
					articulationRootInitialPose;
				attachmentRoot =
					attachmentArticulation->createLink(
						NULL, articulationRootInitialPose);
				attachmentLink =
					attachmentArticulation->createLink(
						attachmentRoot,
						articulationChildInitialPose);
				if(!attachmentRoot || !attachmentLink)
					break;
				PxArticulationLink* links[2] = {
					attachmentRoot, attachmentLink};
				for(PxU32 linkIndex = 0;
					linkIndex < 2; ++linkIndex)
				{
					links[linkIndex]->setMass(1.0f);
					links[linkIndex]->
						setMassSpaceInertiaTensor(PxVec3(1.0f));
					links[linkIndex]->setActorFlag(
						PxActorFlag::eDISABLE_GRAVITY, true);
					links[linkIndex]->setLinearDamping(0.5f);
					links[linkIndex]->setAngularDamping(0.5f);
				}
				PxArticulationJointReducedCoordinate* joint =
					attachmentLink->getInboundJoint();
				if(!joint)
					break;
				joint->setJointType(
					PxArticulationJointType::ePRISMATIC);
				joint->setMotion(
					PxArticulationAxis::eX,
					PxArticulationMotion::eFREE);
				joint->setParentPose(PxTransform(PxIdentity));
				joint->setChildPose(PxTransform(PxIdentity));
				if(!scene->addArticulation(
					*attachmentArticulation))
					break;
				attachmentArticulation->putToSleep();
				metrics.articulationAdded = 1;
				metrics.articulationInitiallySleeping =
					attachmentArticulation->isSleeping() ? 1u : 0u;
				attachmentBody = attachmentLink;
				attachmentActor = attachmentLink;
			}
			else if(staticAttachmentCase)
			{
				attachmentStatic = physics->createRigidStatic(
					PxTransform(rigidAttachmentInitialPosition));
				if(!attachmentStatic ||
					!scene->addActor(*attachmentStatic))
					break;
				attachmentActor = attachmentStatic;
				metrics.kinematicBoxAdded = 1;
			}
			else
			{
				attachmentBox = physics->createRigidDynamic(
					PxTransform(rigidAttachmentInitialPosition));
				if(!attachmentBox)
					break;
				if(kinematicAttachmentCase)
					attachmentBox->setRigidBodyFlag(
						PxRigidBodyFlag::eKINEMATIC, true);
				else
				{
					attachmentBox->setMass(1.0f);
					attachmentBox->setMassSpaceInertiaTensor(
						PxVec3(1.0f));
				}
				attachmentBox->setActorFlag(
					PxActorFlag::eDISABLE_GRAVITY, true);
				attachmentBox->setLinearDamping(0.5f);
				attachmentBox->setAngularDamping(0.5f);
				if(!scene->addActor(*attachmentBox))
					break;
				if(!kinematicAttachmentCase)
					attachmentBox->putToSleep();
				else
					metrics.kinematicBoxAdded = 1;
				attachmentBody = attachmentBox;
				attachmentActor = attachmentBox;
			}
			metrics.rigidAttachmentActorAdded = 1;
			metrics.rigidAttachmentInitiallySleeping =
				staticAttachmentCase
					? 1u
					: articulationAttachmentCase
					? metrics.articulationInitiallySleeping
					: (attachmentBox->isSleeping() ? 1u : 0u);

			PxDeformableAttachmentData attachmentData;
			PxU32 attachmentIndex =
				elementAttachmentCase ? 0u : attachmentVertex;
			PxVec4 rigidCoordinate(
				rigidAttachmentLocalOffset, 0.0f);
			attachmentData.actor[0] = surface;
			attachmentData.type[0] =
				elementAttachmentCase
					? PxDeformableAttachmentTargetType::eTRIANGLE
					: PxDeformableAttachmentTargetType::eVERTEX;
			attachmentData.indices[0].data = &attachmentIndex;
			attachmentData.indices[0].count = 1;
			if(elementAttachmentCase)
			{
				attachmentData.coords[0].data =
					&attachmentBarycentric;
				attachmentData.coords[0].count = 1;
			}
			attachmentData.actor[1] = attachmentActor;
			attachmentData.type[1] =
				PxDeformableAttachmentTargetType::eRIGID;
			attachmentData.coords[1].data = &rigidCoordinate;
			attachmentData.coords[1].count = 1;
			rigidAttachment =
				physics->createDeformableAttachment(
					attachmentData);
			if(!rigidAttachment)
				break;
			metrics.rigidAttachmentCreated = 1;
			if(!kinematicAttachmentCase &&
				!staticAttachmentCase)
			{
				for(PxU32 i = 0; i < vertexCount; ++i)
					velocities[i] =
						PxVec4(0.5f, 0.0f, 0.0f, 0.0f);
				surface->markDirty(
					PxDeformableSurfaceDataFlag::eVELOCITY);
			}
		}
		if(elementFilterCase && !softSoftElementFilterCase)
		{
			if(!groundPlane)
				break;
			std::vector<PxU32> filteredTriangles(
				partialElementFilterCase
					? 1u
					: triangles.size() / 3);
			for(PxU32 i = 0;
				i < PxU32(filteredTriangles.size()); ++i)
				filteredTriangles[i] = i;
			const PxU32 groupCount =
				PxU32(filteredTriangles.size());
			PxDeformableElementFilterData filterData;
			filterData.actor[0] = surface;
			filterData.actor[1] = groundPlane;
			filterData.groupElementCounts[0].data = &groupCount;
			filterData.groupElementCounts[0].count = 1;
			filterData.groupElementIndices[0].data =
				filteredTriangles.data();
			filterData.groupElementIndices[0].count =
				groupCount;
			elementFilter =
				physics->createDeformableElementFilter(
					filterData);
			if(!elementFilter)
				break;
			metrics.elementFilterCreated = 1;
		}
		if(dynamicRigidCase || kinematicRigidCase)
		{
			if(!rigidMaterial)
				break;
			if(kinematicConvexCase || dynamicConvexCase)
			{
				const PxVec3 convexVertices[] =
				{
					PxVec3(-0.8f, 0.0f, 0.0f),
					PxVec3(0.8f, 0.0f, 0.0f),
					PxVec3(0.0f, -0.8f, 0.0f),
					PxVec3(0.0f, 0.8f, 0.0f),
					PxVec3(0.0f, 0.0f, -0.8f),
					PxVec3(0.0f, 0.0f, 0.8f)
				};
				rigidConvexMesh =
					createAvbdTestConvexMesh(
						*physics, scale, convexVertices,
						sizeof(convexVertices) /
							sizeof(convexVertices[0]));
				if(!rigidConvexMesh)
					break;
			}
			if(kinematicTriangleMeshCase)
			{
				rigidTriangleMesh =
					createAvbdRigidTriangleMesh(
						*physics, scale, false);
				if(!rigidTriangleMesh)
					break;
			}
			if(kinematicHeightFieldCase)
			{
				rigidHeightField =
					createAvbdRigidHeightField(
						*physics, false);
				if(!rigidHeightField)
					break;
			}
			const PxVec3 kinematicHorizontalOffset =
				kinematicHeightFieldCase
					? PxVec3(-1.2f, 0.0f, -1.2f)
					: PxVec3(0.0f);
			dynamicBox = physics->createRigidDynamic(
				PxTransform(
					kinematicHorizontalOffset + PxVec3(
					0.0f,
					kinematicSmoothCase ? 1.1f :
						dynamicSmoothCase ? 0.3f :
						kinematicRigidCase ? 1.5f : 0.3f,
					0.0f)));
			if(dynamicBox && kinematicRigidCase)
				dynamicBox->setRigidBodyFlag(
					PxRigidBodyFlag::eKINEMATIC, true);
			PxShape* boxShape =
				kinematicTriangleMeshCase
				? physics->createShape(
					PxTriangleMeshGeometry(
						rigidTriangleMesh),
					*rigidMaterial)
				: kinematicHeightFieldCase
				? physics->createShape(
					PxHeightFieldGeometry(
						rigidHeightField,
						PxMeshGeometryFlags(),
						0.1f, 1.2f, 1.2f),
					*rigidMaterial)
				: (kinematicConvexCase || dynamicConvexCase)
				? physics->createShape(
					PxConvexMeshGeometry(rigidConvexMesh),
					*rigidMaterial)
				: (kinematicCapsuleCase || dynamicCapsuleCase)
				? physics->createShape(
					PxCapsuleGeometry(0.8f, 0.3f),
					*rigidMaterial)
				: (kinematicSphereCase || dynamicSphereCase)
				? physics->createShape(
					PxSphereGeometry(0.8f), *rigidMaterial)
				: physics->createShape(
					PxBoxGeometry(
						1.2f,
						kinematicRigidCase ? 0.25f : 0.3f,
						1.2f),
					*rigidMaterial);
			if(!dynamicBox || !boxShape)
			{
				PX_RELEASE(boxShape);
				break;
			}
			const bool boxAttached =
				dynamicBox->attachShape(*boxShape);
			boxShape->release();
			if(!boxAttached)
				break;
			if(kinematicRigidCase)
				dynamicBox->setRigidBodyFlag(
					PxRigidBodyFlag::eKINEMATIC, true);
			else
				PxRigidBodyExt::updateMassAndInertia(
					*dynamicBox, 10.0f);
			dynamicBox->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			dynamicBox->setLinearDamping(
				dynamicConvexCase ? 1.5f : 1.0f);
			dynamicBox->setAngularDamping(
				dynamicConvexCase ? 2.0f : 1.0f);
			if(!scene->addActor(*dynamicBox))
				break;
			if(kinematicRigidCase)
				metrics.kinematicBoxAdded = 1;
			else
			{
				dynamicBox->putToSleep();
				metrics.dynamicBoxAdded = 1;
				metrics.dynamicBoxInitiallySleeping =
					dynamicBox->isSleeping() ? 1u : 0u;
				metrics.dynamicBoxInitialY =
					dynamicBox->getGlobalPose().p.y;
				metrics.dynamicBoxMinY =
					metrics.dynamicBoxInitialY;
			}
		}
		if(softSoftCase || softSoftElementFilterCase)
		{
			secondSurface = physics->createDeformableSurface(
				PxDeformableSurfaceBackend::eCPU_AVBD);
			if(!secondSurface)
				break;
			metrics.secondSurfaceCreated = 1;
			PxShape* secondShape = physics->createShape(
				PxTriangleMeshGeometry(triangleMesh),
				materials, 1, true, shapeFlags);
			if(!secondShape)
				break;
			const bool secondAttached =
				secondSurface->attachShape(*secondShape);
			secondShape->release();
			if(!secondAttached)
				break;
			secondPositions =
				secondSurface->getPositionInvMassBufferH();
			secondVelocities =
				secondSurface->getVelocityBufferH();
			PxVec4* secondRestPositions =
				secondSurface->getRestPositionBufferH();
			if(!secondPositions || !secondVelocities ||
				!secondRestPositions)
				break;
			for(PxU32 i = 0; i < vertexCount; ++i)
			{
				PxVec3 position = initialPositions[i];
				if(softSoftElementFilterCase)
				{
					const PxU32 triangleStart = i - i % 3;
					const PxVec3 centroid =
						(initialPositions[triangleStart] +
						 initialPositions[triangleStart + 1] +
						 initialPositions[triangleStart + 2]) /
						3.0f;
					position =
						centroid +
						2.0f * (position - centroid);
				}
				position.y = softSoftElementFilterCase
					? 0.0f
					: surfaceSurfaceAttachmentCase
						? initialPositions[i].y
						: softSoftSweptCase
							? 1.9f
							: 1.6f;
				secondPositions[i] = PxVec4(
					position,
					softSoftElementFilterCase ? 0.0f : 1.0f);
				secondVelocities[i] = PxVec4(0.0f);
				secondRestPositions[i] =
					PxVec4(position, 0.0f);
			}
			metrics.secondSurfaceInitialCentroidY =
				dynamicCentroidY(
					secondPositions, vertexCount);
			secondSurface->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			secondSurface->setLinearDamping(0.8f);
			secondSurface->setSolverIterationCounts(8);
			secondSurface->setSleepThreshold(5.0e-4f);
			if(softSoftSweptCase)
			{
				secondSurface->setDeformableBodyFlag(
					PxDeformableBodyFlag::
						eENABLE_SPECULATIVE_CCD,
					true);
				secondSurface->
					setNbCollisionPairUpdatesPerTimestep(4);
				secondSurface->setNbCollisionSubsteps(2);
			}
			secondSurface->setWakeCounter(0.0f);
			secondSurface->markDirty(
				PxDeformableSurfaceDataFlag::eALL);
			if(!scene->addActor(*secondSurface) ||
				secondSurface->getScene() != scene)
				break;
			metrics.secondSurfaceAdded = 1;
			metrics.secondSurfaceInitiallySleeping =
				secondSurface->isSleeping() ? 1u : 0u;
		}
		if(softSoftElementFilterCase)
		{
			if(!secondSurface || triangleMesh->getNbTriangles() != 2)
				break;
			const PxU32 groupCount = 1;
			const PxU32 triangleIndex = 0;
			PxDeformableElementFilterData filterData;
			// Deliberately reverse the public actor order relative to the
			// only dynamic query side. Prep must normalize either contact
			// direction to the same physical element pair.
			filterData.actor[0] = secondSurface;
			filterData.actor[1] = surface;
			for(PxU32 i = 0; i < 2; ++i)
			{
				filterData.groupElementCounts[i].data = &groupCount;
				filterData.groupElementCounts[i].count = 1;
				filterData.groupElementIndices[i].data =
					&triangleIndex;
				filterData.groupElementIndices[i].count = 1;
			}
			elementFilter =
				physics->createDeformableElementFilter(
					filterData);
			if(!elementFilter)
				break;
			metrics.elementFilterCreated = 1;
		}
		if(surfaceVolumeCase)
		{
			if(!buildVolumeMesh(
				*physics, scale, mixedVolumeMesh))
				break;
			mixedVolumeMaterial =
				physics->createDeformableVolumeMaterial(
					2.0e5f, 0.3f, 0.2f, 0.01f);
			mixedVolume = physics->createDeformableVolume(
				PxDeformableVolumeBackend::eCPU_AVBD);
			if(!mixedVolumeMaterial || !mixedVolume)
				break;
			metrics.mixedVolumeCreated = 1;

			PxDeformableVolumeMaterial* volumeMaterials[] =
			{
				mixedVolumeMaterial
			};
			PxShape* volumeShape = physics->createShape(
				PxTetrahedronMeshGeometry(
					mixedVolumeMesh->getCollisionMesh()),
				volumeMaterials, 1, true, shapeFlags);
			if(!volumeShape)
				break;
			const bool volumeShapeAttached =
				mixedVolume->attachShape(*volumeShape);
			volumeShape->release();
			if(!volumeShapeAttached ||
				!mixedVolume->attachSimulationMesh(
					*mixedVolumeMesh->getSimulationMesh(),
					*mixedVolumeMesh->
						getDeformableVolumeAuxData()))
				break;

			mixedVolumePositions =
				mixedVolume->getSimPositionInvMassBufferH();
			mixedVolumeVelocities =
				mixedVolume->getSimVelocityBufferH();
			PxVec4* collisionPositions =
				mixedVolume->getPositionInvMassBufferH();
			PxVec4* collisionRestPositions =
				mixedVolume->getRestPositionBufferH();
			if(!mixedVolumePositions || !mixedVolumeVelocities ||
				!collisionPositions || !collisionRestPositions)
				break;
			const PxTetrahedronMesh* simulationMesh =
				mixedVolume->getSimulationMesh();
			mixedVolumeVertexCount =
				simulationMesh->getNbVertices();
			const PxVec3* simulationVertices =
				simulationMesh->getVertices();
			PxVec3 softPairVolumeLocalPoint(0.0f);
			if(surfaceVolumeAttachmentCase)
			{
				const bool has16BitIndices =
					simulationMesh->getTetrahedronMeshFlags() &
						PxTetrahedronMeshFlag::e16_BIT_INDICES;
				const PxU16* tets16 = has16BitIndices
					? static_cast<const PxU16*>(
						simulationMesh->getTetrahedrons())
					: NULL;
				const PxU32* tets32 = has16BitIndices
					? NULL
					: static_cast<const PxU32*>(
						simulationMesh->getTetrahedrons());
				bool targetTetValid = true;
				for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
				{
					const PxU32 vertex =
						has16BitIndices
							? tets16[endpoint]
							: tets32[endpoint];
					if(vertex >= mixedVolumeVertexCount)
					{
						targetTetValid = false;
						break;
					}
					softPairVolumeLocalPoint +=
						simulationVertices[vertex] * 0.25f;
				}
				if(!targetTetValid)
					break;
			}
			const PxVec3 translation =
				surfaceVolumeAttachmentCase
					? attachmentTarget -
						softPairVolumeLocalPoint
					: PxVec3(-0.5f, 0.7f, -0.5f);
			PxReal* cookedInvMass =
				mixedVolume->getDeformableVolumeAuxData()->
					getGridModelInvMass();
			for(PxU32 i = 0; i < mixedVolumeVertexCount; ++i)
			{
				const PxReal invMass = cookedInvMass
					? PxMax(cookedInvMass[i], 0.0f) : 1.0f;
				mixedVolumePositions[i] = PxVec4(
					simulationVertices[i] + translation, invMass);
				mixedVolumeVelocities[i] =
					PxVec4(0.0f, 0.0f, 0.0f, invMass);
			}
			PxDeformableVolumeExt::updateMass(
				*mixedVolume, 100.0f, 50.0f,
				mixedVolumePositions);
			for(PxU32 i = 0; i < mixedVolumeVertexCount; ++i)
				mixedVolumeVelocities[i].w =
					mixedVolumePositions[i].w;
			PxDeformableVolumeExt::updateEmbeddedCollisionMesh(
				*mixedVolume, mixedVolumePositions,
				collisionPositions);
			const PxU32 collisionVertexCount =
				mixedVolume->getCollisionMesh()->getNbVertices();
			for(PxU32 i = 0; i < collisionVertexCount; ++i)
				collisionRestPositions[i] = collisionPositions[i];

			metrics.mixedVolumeInitialCentroidY =
				dynamicCentroidY(
					mixedVolumePositions,
					mixedVolumeVertexCount);
			mixedVolume->setActorFlag(
				PxActorFlag::eDISABLE_GRAVITY, true);
			mixedVolume->setLinearDamping(0.2f);
			mixedVolume->setSolverIterationCounts(8, 1);
			mixedVolume->setSleepThreshold(5.0e-4f);
			mixedVolume->setWakeCounter(0.0f);
			mixedVolume->markDirty(
				PxDeformableVolumeDataFlag::eALL);
			if(!scene->addActor(*mixedVolume) ||
				mixedVolume->getScene() != scene)
				break;
			metrics.mixedVolumeAdded = 1;
			metrics.mixedVolumeInitiallySleeping =
				mixedVolume->isSleeping() ? 1u : 0u;
		}

		const PxVec4 softPairVolumeBarycentric(0.25f);
		if(softPairAttachmentCase)
		{
			PxDeformableAttachmentData attachmentData;
			const PxU32 sourceElement = 0;
			const PxU32 targetElement = 0;
			attachmentData.actor[0] = surface;
			attachmentData.type[0] =
				PxDeformableAttachmentTargetType::eTRIANGLE;
			attachmentData.indices[0].data = &sourceElement;
			attachmentData.indices[0].count = 1;
			attachmentData.coords[0].data =
				&attachmentBarycentric;
			attachmentData.coords[0].count = 1;
			attachmentData.actor[1] =
				surfaceSurfaceAttachmentCase
					? static_cast<PxActor*>(secondSurface)
					: static_cast<PxActor*>(mixedVolume);
			attachmentData.type[1] =
				surfaceSurfaceAttachmentCase
					? PxDeformableAttachmentTargetType::eTRIANGLE
					: PxDeformableAttachmentTargetType::
						eTETRAHEDRON;
			attachmentData.indices[1].data = &targetElement;
			attachmentData.indices[1].count = 1;
			attachmentData.coords[1].data =
				surfaceSurfaceAttachmentCase
					? &attachmentBarycentric
					: &softPairVolumeBarycentric;
			attachmentData.coords[1].count = 1;
			softPairAttachment =
				physics->createDeformableAttachment(
					attachmentData);
			if(!softPairAttachment)
				break;
			metrics.attachmentCreated = 1;

			// The pair objective, rather than coincident collision geometry,
			// owns this test. Filter all element pairs and start from a
			// compatible attachment pose so the gate measures response to an
			// authored load instead of one-frame projection energy.
			const PxU32 wildcardElementCount = 0;
			PxDeformableElementFilterData filterData;
			filterData.actor[0] = surface;
			filterData.actor[1] =
				surfaceSurfaceAttachmentCase
					? static_cast<PxActor*>(secondSurface)
					: static_cast<PxActor*>(mixedVolume);
			for(PxU32 endpoint = 0; endpoint < 2; ++endpoint)
			{
				filterData.groupElementCounts[endpoint].data =
					&wildcardElementCount;
				filterData.groupElementCounts[endpoint].count = 1;
				filterData.groupElementIndices[endpoint].data = NULL;
				filterData.groupElementIndices[endpoint].count = 0;
			}
			elementFilter =
				physics->createDeformableElementFilter(filterData);
			if(!elementFilter)
				break;

			for(PxU32 i = 0; i < vertexCount; ++i)
				velocities[i] =
					PxVec4(0.0f, -0.1f, 0.0f, 0.0f);
			surface->markDirty(
				PxDeformableSurfaceDataFlag::eVELOCITY);
			if(surfaceSurfaceAttachmentCase)
			{
				for(PxU32 i = 0; i < vertexCount; ++i)
					secondVelocities[i] =
						PxVec4(0.0f, 0.1f, 0.0f, 0.0f);
				secondSurface->markDirty(
					PxDeformableSurfaceDataFlag::eVELOCITY);
			}
			else
			{
				for(PxU32 i = 0;
					i < mixedVolumeVertexCount; ++i)
					mixedVolumeVelocities[i] = PxVec4(
						0.0f, 0.1f, 0.0f,
						mixedVolumePositions[i].w);
				mixedVolume->markDirty(
					PxDeformableVolumeDataFlag::eSIM_VELOCITY);
			}

			if(surfaceSurfaceAttachmentCase)
			{
				softPairTargetVertexCount = 3;
				for(PxU32 endpoint = 0; endpoint < 3; endpoint++)
					softPairTargetVertices[endpoint] =
						attachmentTriangleVertices[endpoint];
			}
			else
			{
				const PxTetrahedronMesh* simulationMesh =
					mixedVolume->getSimulationMesh();
				const bool has16BitIndices =
					simulationMesh->getTetrahedronMeshFlags() &
						PxTetrahedronMeshFlag::e16_BIT_INDICES;
				const PxU16* tets16 = has16BitIndices
					? static_cast<const PxU16*>(
						simulationMesh->getTetrahedrons())
					: NULL;
				const PxU32* tets32 = has16BitIndices
					? NULL
					: static_cast<const PxU32*>(
						simulationMesh->getTetrahedrons());
				softPairTargetVertexCount = 4;
				for(PxU32 endpoint = 0; endpoint < 4; endpoint++)
					softPairTargetVertices[endpoint] =
						has16BitIndices
							? tets16[endpoint]
							: tets32[endpoint];
			}
		}
		const auto getSoftPairTargetPoint = [&]() -> PxVec3
		{
			if(surfaceSurfaceAttachmentCase)
			{
				return
					secondPositions[softPairTargetVertices[0]].
						getXYZ() * attachmentBarycentric.x +
					secondPositions[softPairTargetVertices[1]].
						getXYZ() * attachmentBarycentric.y +
					secondPositions[softPairTargetVertices[2]].
						getXYZ() * attachmentBarycentric.z;
			}
			PxVec3 point(0.0f);
			for(PxU32 endpoint = 0;
				endpoint < softPairTargetVertexCount; endpoint++)
				point += mixedVolumePositions[
					softPairTargetVertices[endpoint]].getXYZ() *
					0.25f;
			return point;
		};

		PxReal kinematicCommandY =
			kinematicSmoothCase ? 1.1f : 1.5f;
		PxReal kinematicSurfaceBaselineY = 0.0f;
		bool samplesFinite = true;
		const PxU32 churnFrame = PxMax<PxU32>(
			1, elementFilterCase
				? 2 * options.frames / 5
				: options.frames / 3);
		const PxU32 mutationFrame = PxMax<PxU32>(
			1, options.frames / 4);
		const PxU32 restoreFrame = PxMin<PxU32>(
			options.frames - 1, mutationFrame + 30);
		const PxU32 selfCollisionDisableFrame = PxMax<PxU32>(
			churnFrame + 1, options.frames / 2);
		const PxU32 materialFrictionSwitchFrame =
			PxMax<PxU32>(churnFrame + 1, options.frames / 2);
		const PxU32 attachmentReleaseFrame = PxMax<PxU32>(
			churnFrame + 1, 2 * options.frames / 3);
		const PxU32 elementFilterReleaseFrame = PxMin<PxU32>(
			options.frames - 1,
			PxMax<PxU32>(churnFrame + 1,
				options.frames / 2));
		const PxU32 mutationIndex =
			vertexIndex(sGridWidth / 2, sGridHeight / 2);
		PxVec3 mutationTarget(0.0f);
		PxVec3 restorePosition(0.0f);
		PxReal kinematicAttachmentProgress = 0.0f;
		bool kinematicAttachmentCommandIssued = false;
		PxTransform kinematicAttachmentCommand =
			PxTransform(rigidAttachmentInitialPosition);
		PxVec3 kinematicAttachmentSoftBaseline(0.0f);
		const PxReal materialFrictionInitialCentroidX =
			dynamicCentroidX(positions, vertexCount);
		for(PxU32 frame = 0; frame < options.frames; ++frame)
		{
			if(bufferMutationCase && frame == mutationFrame)
			{
				mutationTarget =
					positions[mutationIndex].getXYZ() +
					PxVec3(0.2f, 0.1f, 0.0f);
				positions[mutationIndex] =
					PxVec4(mutationTarget, 0.0f);
				velocities[mutationIndex] = PxVec4(0.0f);
				restPositions[mutationIndex] =
					PxVec4(mutationTarget, 0.0f);
				surface->markDirty(
					PxDeformableSurfaceDataFlag::eALL);
				metrics.bufferMutationIssued = 1;
				metrics.bufferPinHeld = 1;
			}
			if(bufferMutationCase && frame == restoreFrame)
			{
				restorePosition =
					positions[mutationIndex].getXYZ();
				positions[mutationIndex].w = 1.0f;
				velocities[mutationIndex] =
					PxVec4(0.5f, 0.0f, 0.0f, 0.0f);
				surface->markDirty(
					PxDeformableSurfaceDataFlags(
						PxDeformableSurfaceDataFlag::
							ePOSITION_INVMASS |
						PxDeformableSurfaceDataFlag::eVELOCITY));
				metrics.bufferInvMassRestored = 1;
			}
			if(frame == churnFrame)
			{
				scene->removeActor(*surface);
				if(surface->getScene() != NULL)
					break;
				metrics.actorRemoved = 1;
				if(!scene->addActor(*surface) ||
					surface->getScene() != scene)
					break;
				metrics.actorReadded = 1;
				if(elementFilterCase)
				{
					for(PxU32 i = 0; i < vertexCount; ++i)
						velocities[i] = PxVec4(
							0.0f,
							softSoftElementFilterCase ? 0.0f : -2.0f,
							0.0f, 0.0f);
					surface->markDirty(
						PxDeformableSurfaceDataFlag::eVELOCITY);
				}
			}
			if(elementFilterCase && elementFilter &&
				frame > churnFrame &&
				frame < elementFilterReleaseFrame)
			{
				metrics.elementFilterHeldAcrossReadd = 1;
			}
			if(elementFilterCase && elementFilter &&
				frame == elementFilterReleaseFrame)
			{
				for(PxU32 i = 0; i < vertexCount; ++i)
				{
					positions[i] =
						PxVec4(initialPositions[i], 1.0f);
					velocities[i] = PxVec4(0.0f);
				}
				surface->markDirty(
					PxDeformableSurfaceDataFlags(
						PxDeformableSurfaceDataFlag::
							ePOSITION_INVMASS |
						PxDeformableSurfaceDataFlag::eVELOCITY));
				elementFilter->release();
				elementFilter = NULL;
				metrics.elementFilterReleased = 1;
			}
			if(selfCollisionCase && !selfCollisionFilterCase &&
				frame == selfCollisionDisableFrame)
			{
				surface->setDeformableBodyFlag(
					PxDeformableBodyFlag::eDISABLE_SELF_COLLISION,
					true);
				for(PxU32 i = 0; i < vertexCount; ++i)
				{
					positions[i] = PxVec4(
						initialPositions[i], i < 3 ? 0.0f : 1.0f);
					velocities[i] = i < 3
						? PxVec4(0.0f)
						: PxVec4(
							0.0f,
							selfCollisionSweptCase
								? -30.0f : -0.5f,
							0.0f, 0.0f);
				}
				surface->markDirty(
					PxDeformableSurfaceDataFlags(
						PxDeformableSurfaceDataFlag::
							ePOSITION_INVMASS |
						PxDeformableSurfaceDataFlag::eVELOCITY));
				metrics.selfCollisionDisableIssued = 1;
			}
			if(materialFrictionCase &&
				frame == materialFrictionSwitchFrame)
			{
				metrics.materialFrictionLowDisplacement = PxAbs(
					dynamicCentroidX(positions, vertexCount) -
					materialFrictionInitialCentroidX);
				scene->removeActor(*surface);
				if(surface->getScene() != NULL)
					break;
				surfaceMaterial->setDynamicFriction(2.0f);
				metrics.materialFrictionHighApplied =
					PxAbs(
						surfaceMaterial->getDynamicFriction() -
						2.0f) <= 1.0e-6f
					? 1u : 0u;
				for(PxU32 i = 0; i < vertexCount; ++i)
				{
					positions[i] =
						PxVec4(initialPositions[i], 1.0f);
					velocities[i] =
						PxVec4(0.5f, 0.0f, 0.0f, 0.0f);
				}
				surface->markDirty(
					PxDeformableSurfaceDataFlags(
						PxDeformableSurfaceDataFlag::
							ePOSITION_INVMASS |
						PxDeformableSurfaceDataFlag::eVELOCITY));
				if(!scene->addActor(*surface) ||
					surface->getScene() != scene)
					break;
			}
			if(softPairAttachmentCase && softPairAttachment &&
				frame == attachmentReleaseFrame)
			{
				softPairAttachment->release();
				softPairAttachment = NULL;
				metrics.attachmentReleased = 1;
				for(PxU32 i = 0; i < vertexCount; ++i)
					velocities[i] =
						PxVec4(-0.5f, 0.0f, 0.0f, 0.0f);
				surface->markDirty(
					PxDeformableSurfaceDataFlag::eVELOCITY);
				if(surfaceSurfaceAttachmentCase)
				{
					for(PxU32 i = 0; i < vertexCount; ++i)
						secondVelocities[i] =
							PxVec4(0.5f, 0.0f, 0.0f, 0.0f);
					secondSurface->markDirty(
						PxDeformableSurfaceDataFlag::eVELOCITY);
				}
				else
				{
					for(PxU32 i = 0;
						i < mixedVolumeVertexCount; ++i)
						mixedVolumeVelocities[i] = PxVec4(
							0.5f, 0.0f, 0.0f,
							mixedVolumePositions[i].w);
					mixedVolume->markDirty(
						PxDeformableVolumeDataFlag::eSIM_VELOCITY);
				}
			}
			if(worldPinCase && worldAttachment &&
				frame == attachmentReleaseFrame)
			{
				worldAttachment->release();
				worldAttachment = NULL;
				metrics.attachmentReleased = 1;
			}
			if((rigidAttachmentCase || staticAttachmentCase ||
				kinematicAttachmentCase ||
				articulationAttachmentCase) &&
				rigidAttachment &&
				frame == attachmentReleaseFrame)
			{
				rigidAttachment->release();
				rigidAttachment = NULL;
				metrics.rigidAttachmentReleased = 1;
				for(PxU32 i = 0; i < vertexCount; ++i)
					velocities[i] =
						PxVec4(1.0f, 0.0f, 0.0f, 0.0f);
				surface->markDirty(
					PxDeformableSurfaceDataFlag::eVELOCITY);
				if(rigidAttachmentCase)
					attachmentBox->setLinearVelocity(
						PxVec3(-1.0f, 0.0f, 0.0f));
			}
			if((kinematicAttachmentCase || staticAttachmentCase) &&
				(attachmentBox || attachmentStatic) &&
				!metrics.rigidAttachmentReleased)
			{
				if(!kinematicAttachmentCommandIssued &&
					surface->isSleeping())
				{
					kinematicAttachmentCommandIssued = true;
					metrics.kinematicTargetIssued = 1;
					kinematicAttachmentSoftBaseline =
						getAttachmentPoint();
				}
				if(kinematicAttachmentCommandIssued)
				{
					kinematicAttachmentProgress = PxMin(
						kinematicAttachmentProgress +
							(elementAttachmentCase
								? 0.004f : 0.01f),
						1.0f);
					kinematicAttachmentCommand =
						PxTransform(
							rigidAttachmentInitialPosition +
								PxVec3(
									kinematicAttachmentProgress,
									0.0f, 0.0f),
							PxQuat(
								0.5f *
									kinematicAttachmentProgress,
								PxVec3(0.0f, 0.0f, 1.0f)));
					if(staticAttachmentCase)
						attachmentStatic->setGlobalPose(
							kinematicAttachmentCommand);
					else
						attachmentBox->setKinematicTarget(
							kinematicAttachmentCommand);
				}
			}
			if(kinematicRigidCase && dynamicBox)
			{
				if(!metrics.kinematicTargetIssued &&
					surface->isSleeping())
				{
					metrics.initialSleepObserved = 1;
					metrics.kinematicTargetIssued = 1;
					kinematicSurfaceBaselineY =
						dynamicCentroidY(
							positions, vertexCount);
				}
				if(metrics.kinematicTargetIssued)
				{
					const PxReal nextY = PxMin(
						kinematicCommandY + 0.005f, 2.35f);
					if(nextY > kinematicCommandY)
					{
						kinematicCommandY = nextY;
						dynamicBox->setKinematicTarget(
							PxTransform(
								(kinematicHeightFieldCase
									? PxVec3(-1.2f, nextY, -1.2f)
									: PxVec3(0.0f, nextY, 0.0f))));
					}
				}
			}

			scene->simulate(options.dt);
			if(!scene->fetchResults(true))
			{
				++metrics.fetchFailures;
				break;
			}
			if(skinningCase)
			{
				const bool skinningFinite =
					Snippets::evaluateTriangleSkinning(
						positions, vertexCount,
						skinningBindings, skinningTriangles,
						skinningPositions, skinningNormals);
				gSurfaceSkinningMetrics.evaluatedFrames++;
				if(skinningFinite)
				{
					gSurfaceSkinningMetrics.finiteFrames++;
					for(PxU32 i = 0;
						i < skinningPositions.size(); ++i)
					{
						gSurfaceSkinningMetrics.
							maxDisplacement = PxMax(
								gSurfaceSkinningMetrics.
									maxDisplacement,
								(skinningPositions[i] -
									skinningInitialPositions[i]).
									magnitude());
					}
				}
				else
				{
					++metrics.nonFiniteSamples;
					samplesFinite = false;
				}
			}
			samplesFinite = sampleSurface(
				positions, velocities, initialPositions,
				metrics,
				metrics.bufferMutationIssued ?
					mutationIndex : PX_MAX_U32) &&
				samplesFinite;
			if(softPairAttachmentCase)
			{
				const PxReal separation =
					(getAttachmentPoint() -
						getSoftPairTargetPoint()).magnitude();
				if(!PxIsFinite(separation))
				{
					++metrics.nonFiniteSamples;
					samplesFinite = false;
				}
				else if(!metrics.attachmentReleased &&
					frame >= churnFrame)
				{
					metrics.attachmentPinMaxDrift = PxMax(
						metrics.attachmentPinMaxDrift,
						separation);
					if(separation < 0.05f)
						metrics.attachmentPinned = 1;
					if(frame > churnFrame &&
						separation < 0.05f)
						metrics.rigidAttachmentHeldAcrossReadd =
							1;
				}
				else if(metrics.attachmentReleased)
				{
					metrics.attachmentReleasedMaxDisplacement =
						PxMax(
							metrics.
								attachmentReleasedMaxDisplacement,
							separation);
					if(separation > 0.1f)
						metrics.attachmentMovedAfterRelease = 1;
				}
			}
			if(bufferMutationCase &&
				metrics.bufferMutationIssued &&
				!metrics.bufferInvMassRestored)
			{
				metrics.bufferPinnedDrift = PxMax(
					metrics.bufferPinnedDrift,
					(positions[mutationIndex].getXYZ() -
						mutationTarget).magnitude());
				if(metrics.bufferPinnedDrift > 1.0e-4f)
					metrics.bufferPinHeld = 0;
				else
					metrics.bufferMutationApplied = 1;
			}
			if(bufferMutationCase &&
				metrics.bufferInvMassRestored)
			{
				metrics.bufferRestoredDisplacement = PxMax(
					metrics.bufferRestoredDisplacement,
					(positions[mutationIndex].getXYZ() -
						restorePosition).magnitude());
				if(metrics.bufferRestoredDisplacement > 1.0e-3f)
					metrics.bufferRestoredMoved = 1;
			}
			if(kinematicRigidCase && dynamicBox)
			{
				const PxTransform boxPose =
					dynamicBox->getGlobalPose();
				if(!boxPose.isValid())
				{
					++metrics.nonFiniteSamples;
					samplesFinite = false;
				}
				else
				{
					metrics.kinematicFinalY = boxPose.p.y;
					if(metrics.kinematicTargetIssued)
					{
						const PxReal poseError = PxAbs(
							boxPose.p.y - kinematicCommandY);
						metrics.kinematicMaxPoseError = PxMax(
							metrics.kinematicMaxPoseError,
							poseError);
						if(kinematicCommandY >= 2.35f &&
							poseError <= 1.0e-4f)
							metrics.kinematicTargetReached = 1;
						if(!surface->isSleeping())
							metrics.kinematicSurfaceWoke = 1;
						const PxReal displacement = PxAbs(
							dynamicCentroidY(
								positions, vertexCount) -
							kinematicSurfaceBaselineY);
						metrics.kinematicSurfaceDisplacement =
							PxMax(
								metrics.
									kinematicSurfaceDisplacement,
								displacement);
						if(metrics.
							kinematicSurfaceDisplacement >
								0.02f)
							metrics.kinematicSurfaceMoved = 1;
					}
					for(PxU32 i = 0; i < vertexCount; ++i)
					{
						const PxVec3 localPosition =
							boxPose.transformInv(
								positions[i].getXYZ());
						PxVec3 capsuleRadial = localPosition;
						capsuleRadial.x -= PxClamp(
							capsuleRadial.x, -0.3f, 0.3f);
						const bool nearKinematicShape =
							kinematicCapsuleCase
								? capsuleRadial.magnitude() <= 0.88f
								: kinematicTriangleMeshCase
								? PxAbs(localPosition.x) <= 2.05f &&
									PxAbs(localPosition.z) <= 2.05f &&
									PxAbs(localPosition.y) <= 0.12f
								: kinematicHeightFieldCase
								? localPosition.x >= -0.08f &&
									localPosition.x <= 2.48f &&
									localPosition.z >= -0.08f &&
									localPosition.z <= 2.48f &&
									PxAbs(localPosition.y) <= 0.12f
								: (kinematicSphereCase ||
									kinematicConvexCase)
								? localPosition.magnitude() <= 0.88f
								: PxAbs(localPosition.x) <= 1.28f &&
									PxAbs(localPosition.z) <= 1.28f &&
									PxAbs(localPosition.y) <= 0.35f;
						if(nearKinematicShape)
							metrics.kinematicContactObserved = 1;
					}
					if(kinematicConvexCase &&
						rigidConvexMesh &&
						!metrics.kinematicContactObserved)
					{
						const PxVec3* convexVertices =
							rigidConvexMesh->getVertices();
						const PxU32 convexVertexCount =
							rigidConvexMesh->getNbVertices();
						for(PxU32 rigidVertex = 0;
							rigidVertex < convexVertexCount &&
								!metrics.
									kinematicContactObserved;
							++rigidVertex)
						{
							const PxVec3 worldVertex =
								boxPose.transform(
									convexVertices[
										rigidVertex]);
							for(PxU32 triangle = 0;
								triangle + 2 <
									PxU32(triangles.size());
								triangle += 3)
							{
								const PxU32 v0 =
									triangles[triangle];
								const PxU32 v1 =
									triangles[triangle + 1];
								const PxU32 v2 =
									triangles[triangle + 2];
								if(v0 >= vertexCount ||
									v1 >= vertexCount ||
									v2 >= vertexCount)
									continue;
								const PxVec3 closest =
									closestPointOnTriangleForConvexGate(
										worldVertex,
										positions[v0].getXYZ(),
										positions[v1].getXYZ(),
										positions[v2].getXYZ());
								if((closest - worldVertex).
										magnitudeSquared() <=
									0.08f * 0.08f)
								{
									metrics.
										kinematicContactObserved = 1;
									break;
								}
							}
						}
					}
				}
			}
			if(dynamicRigidCase && dynamicBox)
			{
				const PxReal boxY =
					dynamicBox->getGlobalPose().p.y;
				const PxReal linearSpeed =
					dynamicBox->getLinearVelocity().magnitude();
				const PxReal angularSpeed =
					dynamicBox->getAngularVelocity().magnitude();
				if(!PxIsFinite(boxY) ||
					!PxIsFinite(linearSpeed) ||
					!PxIsFinite(angularSpeed))
				{
					++metrics.nonFiniteSamples;
					samplesFinite = false;
				}
				else
				{
					metrics.dynamicBoxMinY = PxMin(
						metrics.dynamicBoxMinY, boxY);
					metrics.dynamicBoxMaxDrop = PxMax(
						metrics.dynamicBoxMaxDrop,
						metrics.dynamicBoxInitialY - boxY);
					metrics.dynamicBoxMaxLinearSpeed = PxMax(
						metrics.dynamicBoxMaxLinearSpeed,
						linearSpeed);
					metrics.dynamicBoxMaxAngularSpeed = PxMax(
						metrics.dynamicBoxMaxAngularSpeed,
						angularSpeed);
				}
				if(!dynamicBox->isSleeping())
				{
					if(!metrics.dynamicBoxWoke)
						metrics.dynamicBoxWakeFrame = frame;
					metrics.dynamicBoxWoke = 1;
				}
			}
			if((rigidAttachmentCase || staticAttachmentCase ||
				kinematicAttachmentCase ||
				articulationAttachmentCase) &&
				(attachmentBody || attachmentStatic))
			{
				const PxTransform rigidPose =
					staticAttachmentCase
						? attachmentStatic->getGlobalPose()
						: attachmentBody->getGlobalPose();
				const PxReal rigidSpeed =
					staticAttachmentCase
						? 0.0f
						: attachmentBody->getLinearVelocity().
							magnitude();
				const PxReal rigidAngularSpeed =
					staticAttachmentCase
						? 0.0f
						: attachmentBody->getAngularVelocity().
							magnitude();
				const PxReal rigidAngularDisplacement =
					2.0f * PxAcos(PxClamp(
						PxAbs(rigidPose.q.w), 0.0f, 1.0f));
				const PxReal separation =
					(getAttachmentPoint() -
						rigidPose.transform(
							rigidAttachmentLocalOffset)).
						magnitude();
				if(!rigidPose.isValid() ||
					!PxIsFinite(rigidSpeed) ||
					!PxIsFinite(rigidAngularSpeed) ||
					!PxIsFinite(rigidAngularDisplacement) ||
					!PxIsFinite(separation))
				{
					++metrics.nonFiniteSamples;
					samplesFinite = false;
				}
				else
				{
					metrics.rigidAttachmentMaxRigidSpeed =
						PxMax(
							metrics.rigidAttachmentMaxRigidSpeed,
							rigidSpeed);
					metrics.rigidAttachmentMaxAngularSpeed =
						PxMax(
							metrics.
								rigidAttachmentMaxAngularSpeed,
							rigidAngularSpeed);
					if(!metrics.rigidAttachmentReleased)
					{
						metrics.rigidAttachmentMaxRigidDisplacement =
							PxMax(
								metrics.
									rigidAttachmentMaxRigidDisplacement,
								(rigidPose.p -
									rigidAttachmentInitialPosition).
									magnitude());
						if(metrics.
							rigidAttachmentMaxRigidDisplacement >
								0.02f)
							metrics.rigidAttachmentRigidMoved = 1;
						metrics.
							rigidAttachmentMaxAngularDisplacement =
							PxMax(
								metrics.
									rigidAttachmentMaxAngularDisplacement,
								rigidAngularDisplacement);
						if(metrics.
							rigidAttachmentMaxAngularDisplacement >
								0.02f)
							metrics.rigidAttachmentRigidRotated = 1;
						metrics.rigidAttachmentMaxDrift =
							PxMax(
								metrics.rigidAttachmentMaxDrift,
								separation);
						if(frame > churnFrame &&
							separation < 0.05f)
							metrics.
								rigidAttachmentHeldAcrossReadd = 1;
					}
					else
					{
						metrics.
							rigidAttachmentReleasedSeparation =
							PxMax(
								metrics.
									rigidAttachmentReleasedSeparation,
								separation);
						if(metrics.
							rigidAttachmentReleasedSeparation >
								0.2f)
							metrics.
								rigidAttachmentSeparatedAfterRelease =
									1;
					}
				}
				if(articulationAttachmentCase)
				{
					const PxTransform rootPose =
						attachmentRoot->getGlobalPose();
					const PxVec3 childDelta =
						rigidPose.p -
							articulationChildInitialPose.p;
					const PxReal rootDisplacement =
						(rootPose.p -
							articulationRootInitialPose.p).
								magnitude();
					const PxReal forbiddenDisplacement =
						PxSqrt(
							childDelta.y * childDelta.y +
							childDelta.z * childDelta.z);
					const PxQuat childOrientationError =
						rigidPose.q *
							articulationChildInitialPose.q.
								getConjugate();
					const PxReal childAngularDisplacement =
						2.0f * childOrientationError.
							getImaginaryPart().magnitude();
					if(!rootPose.isValid() ||
						!PxIsFinite(rootDisplacement) ||
						!PxIsFinite(forbiddenDisplacement) ||
						!PxIsFinite(childAngularDisplacement))
					{
						++metrics.nonFiniteSamples;
						samplesFinite = false;
					}
					else
					{
						metrics.articulationRootMaxDisplacement =
							PxMax(
								metrics.
									articulationRootMaxDisplacement,
								rootDisplacement);
						metrics.
							articulationChildMaxForbiddenDisplacement =
							PxMax(
								metrics.
									articulationChildMaxForbiddenDisplacement,
								forbiddenDisplacement);
						metrics.
							articulationChildMaxAngularDisplacement =
							PxMax(
								metrics.
									articulationChildMaxAngularDisplacement,
								childAngularDisplacement);
						if(!attachmentArticulation->isSleeping())
							metrics.articulationWoke = 1;
						if(rootDisplacement <= 1.0e-4f)
							metrics.articulationRootStable = 1;
						if(forbiddenDisplacement <= 1.0e-3f &&
							childAngularDisplacement <= 1.0e-3f)
							metrics.
								articulationJointSubspaceHeld = 1;
					}
				}
				else if(kinematicAttachmentCase ||
					staticAttachmentCase)
				{
					if(kinematicAttachmentCommandIssued &&
						!surface->isSleeping())
						metrics.kinematicSurfaceWoke = 1;
					const PxReal softDisplacement =
						(getAttachmentPoint() -
							kinematicAttachmentSoftBaseline).
								magnitude();
					metrics.kinematicSurfaceDisplacement =
						PxMax(
							metrics.
								kinematicSurfaceDisplacement,
							softDisplacement);
					if(softDisplacement > 0.02f)
						metrics.kinematicSurfaceMoved = 1;
					const PxQuat orientationError =
						rigidPose.q *
							kinematicAttachmentCommand.q.
								getConjugate();
					const PxReal poseError =
						(rigidPose.p -
							kinematicAttachmentCommand.p).
								magnitude() +
						2.0f * orientationError.
							getImaginaryPart().magnitude();
					metrics.kinematicMaxPoseError = PxMax(
						metrics.kinematicMaxPoseError,
						poseError);
					if(kinematicAttachmentProgress >= 1.0f &&
						poseError <= 1.0e-4f)
						metrics.kinematicTargetReached = 1;
				}
				else if(!attachmentBox->isSleeping())
					metrics.rigidAttachmentRigidWoke = 1;
			}
			if(softSoftCase && secondSurface &&
				secondPositions && secondVelocities)
			{
				bool secondFinite = true;
				for(PxU32 i = 0; i < vertexCount; ++i)
				{
					secondFinite = secondFinite &&
						secondPositions[i].isFinite() &&
						secondVelocities[i].isFinite();
					metrics.secondSurfaceMinY = PxMin(
						metrics.secondSurfaceMinY,
						secondPositions[i].y);
					metrics.secondSurfaceMaxSpeed = PxMax(
						metrics.secondSurfaceMaxSpeed,
						secondVelocities[i].getXYZ().magnitude());
				}
				if(!secondFinite)
				{
					++metrics.nonFiniteSamples;
					samplesFinite = false;
				}
				const PxReal targetCentroidY =
					dynamicCentroidY(
						secondPositions, vertexCount);
				metrics.secondSurfaceMaxDisplacement = PxMax(
					metrics.secondSurfaceMaxDisplacement,
					PxAbs(targetCentroidY -
						metrics.secondSurfaceInitialCentroidY));
				if(metrics.secondSurfaceMaxDisplacement > 1.0e-3f)
					metrics.secondSurfaceMoved = 1;
				if(!secondSurface->isSleeping())
				{
					if(!metrics.secondSurfaceWoke)
						metrics.secondSurfaceWakeFrame = frame;
					metrics.secondSurfaceWoke = 1;
				}
			}
			if(surfaceVolumeCase && mixedVolume &&
				mixedVolumePositions && mixedVolumeVelocities)
			{
				bool volumeFinite = true;
				for(PxU32 i = 0;
					i < mixedVolumeVertexCount; ++i)
				{
					volumeFinite = volumeFinite &&
						mixedVolumePositions[i].isFinite() &&
						mixedVolumeVelocities[i].isFinite();
					metrics.mixedVolumeMinY = PxMin(
						metrics.mixedVolumeMinY,
						mixedVolumePositions[i].y);
					metrics.mixedVolumeMaxSpeed = PxMax(
						metrics.mixedVolumeMaxSpeed,
						mixedVolumeVelocities[i].
							getXYZ().magnitude());
				}
				if(!volumeFinite)
				{
					++metrics.nonFiniteSamples;
					samplesFinite = false;
				}
				const PxReal targetCentroidY =
					dynamicCentroidY(
						mixedVolumePositions,
						mixedVolumeVertexCount);
				metrics.mixedVolumeMaxDisplacement = PxMax(
					metrics.mixedVolumeMaxDisplacement,
					PxAbs(
						targetCentroidY -
						metrics.mixedVolumeInitialCentroidY));
				if(metrics.mixedVolumeMaxDisplacement > 1.0e-3f)
					metrics.mixedVolumeMoved = 1;
				if(!mixedVolume->isSleeping())
				{
					if(!metrics.mixedVolumeWoke)
						metrics.mixedVolumeWakeFrame = frame;
					metrics.mixedVolumeWoke = 1;
				}
			}
			if(selfCollisionCase)
			{
				PxReal fixedMaxY = -PX_MAX_F32;
				PxReal dynamicMinY = PX_MAX_F32;
				for(PxU32 i = 0; i < 3; ++i)
				{
					fixedMaxY = PxMax(fixedMaxY, positions[i].y);
					dynamicMinY =
						PxMin(dynamicMinY, positions[i + 3].y);
				}
				const PxReal separation = dynamicMinY - fixedMaxY;
				if(selfCollisionFilterCase)
					metrics.selfCollisionFilterMinSeparation =
						PxMin(
							metrics.selfCollisionFilterMinSeparation,
							separation);
				else if(frame < selfCollisionDisableFrame)
					metrics.selfCollisionMinEnabledSeparation =
						PxMin(
							metrics.selfCollisionMinEnabledSeparation,
							separation);
				else
					metrics.selfCollisionMinDisabledSeparation =
						PxMin(
							metrics.selfCollisionMinDisabledSeparation,
							separation);
			}
			if(worldPinCase)
			{
				const PxReal displacement =
					(getAttachmentPoint() -
						attachmentTarget).magnitude();
				if(!metrics.attachmentReleased)
				{
					metrics.attachmentPinMaxDrift = PxMax(
						metrics.attachmentPinMaxDrift,
						displacement);
					if(metrics.attachmentPinMaxDrift <= 1.0e-4f)
						metrics.attachmentPinned = 1;
					else
						metrics.attachmentPinned = 0;
				}
				else
				{
					metrics.attachmentReleasedMaxDisplacement =
						PxMax(
							metrics.
								attachmentReleasedMaxDisplacement,
							displacement);
					if(metrics.
						attachmentReleasedMaxDisplacement >
						1.0e-3f)
						metrics.attachmentMovedAfterRelease = 1;
				}
			}
			if(elementFilterCase &&
				!metrics.elementFilterReleased)
			{
				if(partialElementFilterCase)
				{
					PxReal currentUnfilteredMinY = PX_MAX_F32;
					for(PxU32 i = 0; i < vertexCount; ++i)
					{
						if(partialFilteredVertex[i])
							metrics.
								partialFilterFilteredMinY =
									PxMin(
										metrics.
											partialFilterFilteredMinY,
										positions[i].y);
						else
							currentUnfilteredMinY = PxMin(
								currentUnfilteredMinY,
								positions[i].y);
					}
					metrics.partialFilterUnfilteredMinY =
						PxMin(
							metrics.
								partialFilterUnfilteredMinY,
							currentUnfilteredMinY);
					if(currentUnfilteredMinY > -0.05f &&
						currentUnfilteredMinY < 0.05f)
						metrics.
							partialFilterUnfilteredContactHeld =
								1;
					metrics.elementFilterMinY =
						metrics.partialFilterFilteredMinY;
				}
				else
				{
					for(PxU32 i = 0; i < vertexCount; ++i)
						metrics.elementFilterMinY = PxMin(
							metrics.elementFilterMinY,
							positions[i].y);
				}
				if(metrics.elementFilterMinY < -0.2f)
					metrics.elementFilterSuppressedContact = 1;
			}
			if(sleepWakeCase)
			{
				if(metrics.velocityWakeIssued)
				{
					metrics.velocityWakeObserved =
						!surface->isSleeping() ? 1u :
						metrics.velocityWakeObserved;
					const PxReal wakeCentroidRise =
						dynamicCentroidY(
							positions, vertexCount) -
						metrics.wakeCentroidY;
					metrics.maxWakeCentroidRise = PxMax(
						metrics.maxWakeCentroidRise,
						wakeCentroidRise);
					if(metrics.maxWakeCentroidRise > 1.0e-3f)
						metrics.movedAfterVelocityWake = 1;
				}
				else if(surface->isSleeping())
				{
					metrics.initialSleepObserved = 1;
					metrics.wakeCentroidY =
						dynamicCentroidY(
							positions, vertexCount);
					for(PxU32 i = 0; i < vertexCount; ++i)
					{
						if(positions[i].w > 0.0f)
							velocities[i] =
								PxVec4(0.0f, 1.0f, 0.0f, 0.0f);
					}
					surface->markDirty(
						PxDeformableSurfaceDataFlag::eVELOCITY);
					metrics.velocityWakeIssued = 1;
					metrics.velocityWakeObserved =
						!surface->isSleeping() ? 1u : 0u;
				}
			}
		}

		metrics.finalDynamicCentroidY =
			dynamicCentroidY(positions, vertexCount);
		metrics.finalMinY = PX_MAX_F32;
		metrics.finalMaxSpeed = 0.0f;
		for(PxU32 i = 0; i < vertexCount; ++i)
		{
			metrics.finalMinY = PxMin(
				metrics.finalMinY, positions[i].y);
			metrics.finalMaxSpeed = PxMax(
				metrics.finalMaxSpeed,
				velocities[i].getXYZ().magnitude());
		}
		metrics.pinnedStable =
			metrics.maxPinnedDrift <= 1.0e-4f ? 1u : 0u;
		metrics.dynamicMoved =
			metrics.maxDynamicDisplacement >= 1.0e-2f ? 1u : 0u;
		const PxBounds3 bounds = surface->getWorldBounds();
		metrics.boundsFinite =
			!bounds.isEmpty() && bounds.minimum.isFinite() &&
			bounds.maximum.isFinite() ? 1u : 0u;
		metrics.groundContactObserved =
			groundCase && metrics.minY < 0.1f ? 1u : 0u;
		metrics.surfaceSlept = surface->isSleeping() ? 1u : 0u;
		if(dynamicBox)
		{
			if(kinematicRigidCase)
			{
				metrics.kinematicFinalY =
					dynamicBox->getGlobalPose().p.y;
			}
			else
			{
				metrics.dynamicBoxFinalY =
					dynamicBox->getGlobalPose().p.y;
				metrics.dynamicBoxFinalLinearSpeed =
					dynamicBox->getLinearVelocity().magnitude();
				metrics.dynamicBoxFinalAngularSpeed =
					dynamicBox->getAngularVelocity().magnitude();
				metrics.dynamicBoxFinalSleeping =
					dynamicBox->isSleeping() ? 1u : 0u;
			}
		}
		if(secondSurface && secondPositions && secondVelocities)
		{
			metrics.secondSurfaceFinalCentroidY =
				dynamicCentroidY(secondPositions, vertexCount);
			metrics.secondSurfaceFinalMinY = PX_MAX_F32;
			metrics.secondSurfaceFinalMaxSpeed = 0.0f;
			for(PxU32 i = 0; i < vertexCount; ++i)
			{
				metrics.secondSurfaceFinalMinY = PxMin(
					metrics.secondSurfaceFinalMinY,
					secondPositions[i].y);
				metrics.secondSurfaceFinalMaxSpeed = PxMax(
					metrics.secondSurfaceFinalMaxSpeed,
					secondVelocities[i].getXYZ().magnitude());
			}
			metrics.secondSurfaceFinalSleeping =
				secondSurface->isSleeping() ? 1u : 0u;
		}
		if(mixedVolume && mixedVolumePositions &&
			mixedVolumeVelocities)
		{
			metrics.mixedVolumeFinalCentroidY =
				dynamicCentroidY(
					mixedVolumePositions,
					mixedVolumeVertexCount);
			metrics.mixedVolumeFinalMinY = PX_MAX_F32;
			metrics.mixedVolumeFinalMaxSpeed = 0.0f;
			for(PxU32 i = 0; i < mixedVolumeVertexCount; ++i)
			{
				metrics.mixedVolumeFinalMinY = PxMin(
					metrics.mixedVolumeFinalMinY,
					mixedVolumePositions[i].y);
				metrics.mixedVolumeFinalMaxSpeed = PxMax(
					metrics.mixedVolumeFinalMaxSpeed,
					mixedVolumeVelocities[i].getXYZ().magnitude());
			}
			metrics.mixedVolumeFinalSleeping =
				mixedVolume->isSleeping() ? 1u : 0u;
		}
		metrics.groundPenetrationBounded =
			groundCase && metrics.minY > -0.05f &&
			metrics.finalMinY > -0.05f ? 1u : 0u;
		metrics.groundSettled =
			groundCase && metrics.surfaceSlept &&
			PxAbs(metrics.finalMinY) < 0.05f &&
			metrics.finalMaxSpeed < 1.0e-3f ? 1u : 0u;
		metrics.selfCollisionPreventedCrossing =
			selfCollisionCase && !selfCollisionFilterCase &&
			metrics.selfCollisionMinEnabledSeparation > -0.02f
				? 1u : 0u;
		metrics.selfCollisionDisabledCrossed =
			selfCollisionCase && !selfCollisionFilterCase &&
			metrics.selfCollisionMinDisabledSeparation < -0.05f
				? 1u : 0u;
		metrics.selfCollisionFilterExcludedPair =
			selfCollisionFilterCase &&
			metrics.selfCollisionFilterMinSeparation < -0.05f
				? 1u : 0u;
		if(materialFrictionCase)
		{
			metrics.materialFrictionHighDisplacement = PxAbs(
				dynamicCentroidX(positions, vertexCount) -
				materialFrictionInitialCentroidX);
			metrics.materialFrictionHighFinalSpeed =
				metrics.finalMaxSpeed;
			metrics.materialFrictionResponseObserved =
				metrics.materialFrictionLowDisplacement > 0.2f &&
				metrics.materialFrictionHighDisplacement <
					0.5f *
					metrics.materialFrictionLowDisplacement &&
				metrics.materialFrictionHighFinalSpeed < 0.2f &&
				metrics.finalMinY > -0.05f
				? 1u : 0u;
		}
		metrics.elementFilterFinalMinY = metrics.finalMinY;
		metrics.elementFilterContactRestored =
			elementFilterCase &&
			PxAbs(metrics.elementFilterFinalMinY) < 0.05f &&
			metrics.finalMaxSpeed < 0.1f ? 1u : 0u;
		metrics.partialFilterExactOwnership =
			partialElementFilterCase &&
			metrics.elementFilterSuppressedContact &&
			metrics.partialFilterUnfilteredContactHeld &&
			metrics.partialFilterUnfilteredMinY > -0.05f
				? 1u : 0u;

		success = samplesFinite &&
			metrics.actorCreated && metrics.shapeAttached &&
			metrics.hostBuffersInitialized && metrics.actorAdded &&
			metrics.actorRemoved && metrics.actorReadded &&
			metrics.fetchFailures == 0 &&
			metrics.nonFiniteSamples == 0 &&
			metrics.pinnedStable && metrics.dynamicMoved &&
			metrics.boundsFinite &&
			(!groundCase ||
				(metrics.groundAdded &&
				 metrics.groundContactObserved &&
				 metrics.groundPenetrationBounded &&
				 metrics.groundSettled &&
				 metrics.surfaceSlept)) &&
			(!sleepWakeCase ||
				(metrics.initialSleepObserved &&
				 metrics.velocityWakeIssued &&
				 metrics.velocityWakeObserved &&
				 metrics.movedAfterVelocityWake)) &&
			(!bufferMutationCase ||
				(metrics.bufferMutationIssued &&
				 metrics.bufferMutationApplied &&
				 metrics.bufferPinHeld &&
				 metrics.bufferInvMassRestored &&
				 metrics.bufferRestoredMoved)) &&
			(!dynamicRigidCase ||
				(metrics.dynamicBoxAdded &&
				 metrics.dynamicBoxInitiallySleeping &&
				 metrics.dynamicBoxWoke &&
				 metrics.dynamicBoxMaxDrop > 1.0e-2f)) &&
			(!kinematicRigidCase ||
				(metrics.kinematicBoxAdded &&
				 metrics.initialSleepObserved &&
				 metrics.kinematicTargetIssued &&
				 metrics.kinematicTargetReached &&
				 metrics.kinematicSurfaceWoke &&
				 metrics.kinematicSurfaceMoved &&
				 metrics.kinematicContactObserved &&
				 metrics.kinematicMaxPoseError <= 1.0e-4f &&
				 metrics.kinematicSurfaceDisplacement > 0.02f &&
				 PxIsFinite(metrics.kinematicFinalY) &&
				 PxAbs(metrics.kinematicFinalY - 2.35f) <=
					1.0e-4f &&
				 metrics.maxSpeed < 2.0f &&
				 metrics.finalMaxSpeed < 0.5f)) &&
			(!softSoftCase ||
				(metrics.secondSurfaceCreated &&
				 metrics.secondSurfaceAdded &&
				 metrics.secondSurfaceInitiallySleeping &&
				 metrics.secondSurfaceWoke &&
				 metrics.secondSurfaceMoved)) &&
			(!surfaceVolumeCase ||
				(metrics.mixedVolumeCreated &&
				 metrics.mixedVolumeAdded &&
				 metrics.mixedVolumeInitiallySleeping &&
				 metrics.mixedVolumeWoke &&
				 metrics.mixedVolumeMoved)) &&
			(!selfCollisionCase ||
				(metrics.selfCollisionEnabled &&
				 metrics.selfCollisionFilterApplied &&
				 (selfCollisionFilterCase
					? metrics.selfCollisionFilterExcludedPair
					: (metrics.selfCollisionPreventedCrossing &&
					   metrics.selfCollisionDisableIssued &&
					   metrics.selfCollisionDisabledCrossed)))) &&
			(!materialFrictionCase ||
				(metrics.materialFrictionLowApplied &&
				 metrics.materialFrictionHighApplied &&
				 metrics.materialFrictionResponseObserved)) &&
			(!worldPinCase ||
				(metrics.attachmentCreated &&
				 metrics.attachmentPinned &&
				 metrics.attachmentReleased &&
				 metrics.attachmentMovedAfterRelease)) &&
			(!softPairAttachmentCase ||
				(metrics.attachmentCreated &&
				 metrics.attachmentPinned &&
				 metrics.rigidAttachmentHeldAcrossReadd &&
				 metrics.attachmentReleased &&
				 metrics.attachmentMovedAfterRelease &&
				 metrics.attachmentPinMaxDrift < 0.05f &&
				 metrics.maxSpeed < 5.0f &&
				 (surfaceSurfaceAttachmentCase
					? metrics.secondSurfaceMaxSpeed < 5.0f
					: metrics.mixedVolumeMaxSpeed < 5.0f))) &&
			(!rigidAttachmentCase ||
				(metrics.rigidAttachmentActorAdded &&
				 metrics.rigidAttachmentInitiallySleeping &&
				 metrics.rigidAttachmentCreated &&
				 metrics.rigidAttachmentRigidWoke &&
				 metrics.rigidAttachmentRigidMoved &&
				 metrics.rigidAttachmentRigidRotated &&
				 metrics.rigidAttachmentHeldAcrossReadd &&
				 metrics.rigidAttachmentReleased &&
				 metrics.rigidAttachmentSeparatedAfterRelease &&
				 metrics.rigidAttachmentMaxDrift < 0.05f &&
				 metrics.rigidAttachmentMaxRigidSpeed < 5.0f &&
				 metrics.rigidAttachmentMaxAngularSpeed < 5.0f)) &&
			(!kinematicAttachmentCase ||
				(metrics.rigidAttachmentActorAdded &&
				 metrics.rigidAttachmentInitiallySleeping &&
				 metrics.rigidAttachmentCreated &&
				 metrics.kinematicBoxAdded &&
				 metrics.kinematicTargetIssued &&
				 metrics.kinematicTargetReached &&
				 metrics.kinematicSurfaceWoke &&
				 metrics.kinematicSurfaceMoved &&
				 metrics.kinematicMaxPoseError <= 1.0e-4f &&
				 metrics.kinematicSurfaceDisplacement > 0.02f &&
				 metrics.rigidAttachmentRigidMoved &&
				 metrics.rigidAttachmentRigidRotated &&
				 metrics.rigidAttachmentHeldAcrossReadd &&
				 metrics.rigidAttachmentReleased &&
				 metrics.rigidAttachmentSeparatedAfterRelease &&
				 metrics.rigidAttachmentMaxDrift < 0.05f &&
				 metrics.rigidAttachmentMaxRigidSpeed < 5.0f &&
				 metrics.rigidAttachmentMaxAngularSpeed < 5.0f &&
				 metrics.maxSpeed < 5.0f &&
				 metrics.finalMaxSpeed < 2.0f)) &&
			(!staticAttachmentCase ||
				(metrics.rigidAttachmentActorAdded &&
				 metrics.rigidAttachmentInitiallySleeping &&
				 metrics.rigidAttachmentCreated &&
				 metrics.kinematicBoxAdded &&
				 metrics.kinematicTargetIssued &&
				 metrics.kinematicTargetReached &&
				 metrics.kinematicSurfaceWoke &&
				 metrics.kinematicSurfaceMoved &&
				 metrics.kinematicMaxPoseError <= 1.0e-4f &&
				 metrics.kinematicSurfaceDisplacement > 0.02f &&
				 metrics.rigidAttachmentRigidMoved &&
				 metrics.rigidAttachmentRigidRotated &&
				 metrics.rigidAttachmentHeldAcrossReadd &&
				 metrics.rigidAttachmentReleased &&
				 metrics.rigidAttachmentSeparatedAfterRelease &&
				 metrics.rigidAttachmentMaxDrift < 0.05f &&
				 metrics.maxSpeed < 5.0f &&
				 metrics.finalMaxSpeed < 2.0f)) &&
			(!articulationAttachmentCase ||
				(metrics.articulationCreated &&
				 metrics.articulationAdded &&
				 metrics.articulationInitiallySleeping &&
				 metrics.articulationWoke &&
				 metrics.articulationJointSubspaceHeld &&
				 metrics.articulationRootStable &&
				 metrics.rigidAttachmentActorAdded &&
				 metrics.rigidAttachmentInitiallySleeping &&
				 metrics.rigidAttachmentCreated &&
				 metrics.rigidAttachmentRigidMoved &&
				 metrics.rigidAttachmentHeldAcrossReadd &&
				 metrics.rigidAttachmentReleased &&
				 metrics.rigidAttachmentSeparatedAfterRelease &&
				 metrics.rigidAttachmentMaxDrift < 0.05f &&
				 metrics.rigidAttachmentMaxRigidSpeed < 5.0f &&
				 metrics.articulationRootMaxDisplacement <=
					1.0e-4f &&
				 metrics.
					articulationChildMaxForbiddenDisplacement <=
					1.0e-3f &&
				 metrics.
					articulationChildMaxAngularDisplacement <=
					1.0e-3f &&
				 metrics.maxSpeed < 5.0f &&
				 metrics.finalMaxSpeed < 2.0f)) &&
			(!elementFilterCase ||
				(metrics.elementFilterCreated &&
				 metrics.elementFilterHeldAcrossReadd &&
				 metrics.elementFilterSuppressedContact &&
				 metrics.elementFilterReleased &&
				 metrics.elementFilterContactRestored)) &&
			(!partialElementFilterCase ||
				metrics.partialFilterExactOwnership) &&
			(!skinningCase ||
				(gSurfaceSkinningMetrics.initialized == 1 &&
				 gSurfaceSkinningMetrics.evaluatedFrames ==
					options.frames &&
				 gSurfaceSkinningMetrics.finiteFrames ==
					options.frames &&
				 gSurfaceSkinningMetrics.vertices > vertexCount &&
				 gSurfaceSkinningMetrics.triangles >
					triangleMesh->getNbTriangles() &&
				 PxIsFinite(
					gSurfaceSkinningMetrics.maxDisplacement) &&
				 gSurfaceSkinningMetrics.maxDisplacement >=
					1.0e-2f &&
				 PxAbs(
					gSurfaceSkinningMetrics.maxDisplacement -
					metrics.maxDynamicDisplacement) <=
					1.0e-4f));
	}
	while(false);

	if(worldAttachment)
	{
		worldAttachment->release();
		worldAttachment = NULL;
	}
	if(rigidAttachment)
	{
		rigidAttachment->release();
		rigidAttachment = NULL;
	}
	if(softPairAttachment)
	{
		softPairAttachment->release();
		softPairAttachment = NULL;
	}
	if(elementFilter)
	{
		elementFilter->release();
		elementFilter = NULL;
	}
	if(mixedVolume)
	{
		if(mixedVolume->getScene())
			mixedVolume->getScene()->removeActor(*mixedVolume);
		mixedVolume->release();
		mixedVolume = NULL;
	}
	if(secondSurface)
	{
		if(secondSurface->getScene())
			secondSurface->getScene()->removeActor(
				*secondSurface);
		secondSurface->release();
		secondSurface = NULL;
	}
	if(surface)
	{
		if(surface->getScene())
			surface->getScene()->removeActor(*surface);
		surface->release();
		surface = NULL;
	}
	PX_RELEASE(mixedVolumeMaterial);
	PX_RELEASE(mixedVolumeMesh);
	PX_RELEASE(surfaceMaterial);
	PX_RELEASE(triangleMesh);
	PX_RELEASE(attachmentBox);
	PX_RELEASE(attachmentStatic);
	PX_RELEASE(attachmentArticulation);
	PX_RELEASE(dynamicBox);
	PX_RELEASE(rigidHeightField);
	PX_RELEASE(rigidTriangleMesh);
	PX_RELEASE(rigidConvexMesh);
	PX_RELEASE(groundPlane);
	PX_RELEASE(rigidMaterial);
	skinningBindings.reset();
	skinningTriangles.reset();
	skinningPositions.reset();
	skinningNormals.reset();
	skinningInitialPositions.reset();
	PX_RELEASE(scene);
	PX_RELEASE(dispatcher);
	if(extensionsInitialized)
		PxCloseExtensions();
	PX_RELEASE(physics);
	PX_RELEASE(foundation);
	metrics.cleanupComplete = 1;
	return success;
}

static void printResult(
	const Snippets::HeadlessOptions& options,
	const Snippets::TrackingErrorCallback& errorCallback,
	const Metrics& metrics,
	bool passed)
{
	std::printf(
		"[AVBD_GATE] schema=1 snippet=" AVBD_SURFACE_SNIPPET_NAME " "
		"solver=%s case=%s execution=%s frames=%u "
		"actorCreated=%u shapeAttached=%u "
		"hostBuffersInitialized=%u actorAdded=%u actorRemoved=%u "
		"actorReadded=%u fetchFailures=%u nonFiniteSamples=%u "
		"pinnedStable=%u dynamicMoved=%u boundsFinite=%u "
		"groundAdded=%u groundContactObserved=%u "
		"groundPenetrationBounded=%u groundSettled=%u "
		"surfaceSlept=%u "
		"initialSleepObserved=%u velocityWakeIssued=%u "
		"velocityWakeObserved=%u movedAfterVelocityWake=%u "
		"bufferMutationIssued=%u bufferMutationApplied=%u "
		"bufferPinHeld=%u bufferInvMassRestored=%u "
		"bufferRestoredMoved=%u "
		"dynamicBoxAdded=%u dynamicBoxInitiallySleeping=%u "
		"dynamicBoxWoke=%u dynamicBoxWakeFrame=%u "
		"dynamicBoxFinalSleeping=%u "
		"kinematicBoxAdded=%u kinematicTargetIssued=%u "
		"kinematicTargetReached=%u kinematicSurfaceWoke=%u "
		"kinematicSurfaceMoved=%u kinematicContactObserved=%u "
		"secondSurfaceCreated=%u secondSurfaceAdded=%u "
		"secondSurfaceInitiallySleeping=%u secondSurfaceWoke=%u "
		"secondSurfaceWakeFrame=%u secondSurfaceMoved=%u "
		"secondSurfaceFinalSleeping=%u "
		"mixedVolumeCreated=%u mixedVolumeAdded=%u "
		"mixedVolumeInitiallySleeping=%u mixedVolumeWoke=%u "
		"mixedVolumeWakeFrame=%u mixedVolumeMoved=%u "
		"mixedVolumeFinalSleeping=%u "
		"selfCollisionEnabled=%u "
		"selfCollisionPreventedCrossing=%u "
		"selfCollisionDisableIssued=%u "
		"selfCollisionDisabledCrossed=%u "
		"selfCollisionFilterApplied=%u "
		"selfCollisionFilterExcludedPair=%u "
		"materialFrictionLowApplied=%u "
		"materialFrictionHighApplied=%u "
		"materialFrictionResponseObserved=%u "
		"attachmentCreated=%u attachmentPinned=%u "
		"attachmentReleased=%u attachmentMovedAfterRelease=%u "
		"rigidAttachmentActorAdded=%u "
		"rigidAttachmentInitiallySleeping=%u "
		"rigidAttachmentCreated=%u rigidAttachmentRigidWoke=%u "
		"rigidAttachmentRigidMoved=%u "
		"rigidAttachmentRigidRotated=%u "
		"rigidAttachmentHeldAcrossReadd=%u "
		"rigidAttachmentReleased=%u "
		"rigidAttachmentSeparatedAfterRelease=%u "
		"articulationCreated=%u articulationAdded=%u "
		"articulationInitiallySleeping=%u articulationWoke=%u "
		"articulationJointSubspaceHeld=%u "
		"articulationRootStable=%u "
		"elementFilterCreated=%u "
		"elementFilterHeldAcrossReadd=%u "
		"elementFilterSuppressedContact=%u "
		"elementFilterReleased=%u "
		"elementFilterContactRestored=%u "
		"partialFilterExactOwnership=%u "
		"partialFilterUnfilteredContactHeld=%u "
		"bendingMaterialPairCreated=%u "
		"bendingZeroControlHeld=%u "
		"bendingResponseObserved=%u "
		"bendingMembraneIsolated=%u "
		"flatteningFlagApplied=%u "
		"flatteningControlHeld=%u "
		"flatteningResponseObserved=%u "
		"flatteningRetargetObserved=%u "
		"flatteningMembraneIsolated=%u "
		"motionMaxVelocityBounded=%u "
		"motionSettlingApplied=%u "
		"motionSettlingSlept=%u "
		"motionControlStayedAwake=%u "
		"depenetrationLimitApplied=%u "
		"depenetrationFirstStepBounded=%u "
		"depenetrationControlSeparated=%u "
		"depenetrationGradualRecovery=%u "
		"speculativeCcdFlagApplied=%u "
		"speculativeCcdPreventedTunneling=%u "
		"speculativeCcdNegativeControlTunneled=%u "
		"movingSphereTargetIssued=%u "
		"movingSphereCcdResponseObserved=%u "
		"movingSphereNegativeControlHeld=%u "
		"dynamicSphereSweepLaunched=%u "
		"dynamicSphereSweepResponseObserved=%u "
		"dynamicSphereSweepNegativeControlTunneled=%u "
		"dynamicSphereSweepTwoSidedResponseObserved=%u "
		"maxPinnedDrift=%.9g maxDynamicDisplacement=%.9g "
		"initialDynamicCentroidY=%.9g finalDynamicCentroidY=%.9g "
		"minY=%.9g finalMinY=%.9g maxSpeed=%.9g "
		"finalMaxSpeed=%.9g wakeCentroidY=%.9g "
		"maxWakeCentroidRise=%.9g "
		"bufferPinnedDrift=%.9g "
		"bufferRestoredDisplacement=%.9g "
		"dynamicBoxInitialY=%.9g dynamicBoxMinY=%.9g "
		"dynamicBoxFinalY=%.9g dynamicBoxMaxDrop=%.9g "
		"dynamicBoxMaxLinearSpeed=%.9g "
		"dynamicBoxFinalLinearSpeed=%.9g "
		"dynamicBoxMaxAngularSpeed=%.9g "
		"dynamicBoxFinalAngularSpeed=%.9g "
		"kinematicMaxPoseError=%.9g "
		"kinematicSurfaceDisplacement=%.9g "
		"kinematicFinalY=%.9g "
		"secondSurfaceInitialCentroidY=%.9g "
		"secondSurfaceFinalCentroidY=%.9g "
		"secondSurfaceMaxDisplacement=%.9g "
		"secondSurfaceMinY=%.9g secondSurfaceFinalMinY=%.9g "
		"secondSurfaceMaxSpeed=%.9g "
		"secondSurfaceFinalMaxSpeed=%.9g "
		"mixedVolumeInitialCentroidY=%.9g "
		"mixedVolumeFinalCentroidY=%.9g "
		"mixedVolumeMaxDisplacement=%.9g "
		"mixedVolumeMinY=%.9g mixedVolumeFinalMinY=%.9g "
		"mixedVolumeMaxSpeed=%.9g "
		"mixedVolumeFinalMaxSpeed=%.9g "
		"selfCollisionMinEnabledSeparation=%.9g "
		"selfCollisionMinDisabledSeparation=%.9g "
		"selfCollisionFilterMinSeparation=%.9g "
		"materialFrictionLowDisplacement=%.9g "
		"materialFrictionHighDisplacement=%.9g "
		"materialFrictionHighFinalSpeed=%.9g "
		"attachmentPinMaxDrift=%.9g "
		"attachmentReleasedMaxDisplacement=%.9g "
		"rigidAttachmentMaxDrift=%.9g "
		"rigidAttachmentMaxRigidDisplacement=%.9g "
		"rigidAttachmentMaxRigidSpeed=%.9g "
		"rigidAttachmentMaxAngularDisplacement=%.9g "
		"rigidAttachmentMaxAngularSpeed=%.9g "
		"rigidAttachmentReleasedSeparation=%.9g "
		"articulationRootMaxDisplacement=%.9g "
		"articulationChildMaxForbiddenDisplacement=%.9g "
		"articulationChildMaxAngularDisplacement=%.9g "
		"elementFilterMinY=%.9g "
		"elementFilterFinalMinY=%.9g "
		"partialFilterFilteredMinY=%.9g "
		"partialFilterUnfilteredMinY=%.9g "
		"bendingInitialPlaneError=%.9g "
		"bendingFinalPlaneError=%.9g "
		"bendingZeroControlDisplacement=%.9g "
		"bendingStiffDisplacement=%.9g "
		"bendingMaxEdgeStrain=%.9g "
		"flatteningInitialPlaneError=%.9g "
		"flatteningMinimumPlaneError=%.9g "
		"flatteningFinalPlaneError=%.9g "
		"flatteningControlDisplacement=%.9g "
		"flatteningTargetDisplacement=%.9g "
		"flatteningMaxEdgeStrain=%.9g "
		"motionMaxVelocityFirstStepDisplacement=%.9g "
		"motionMaxVelocityFirstStepSpeed=%.9g "
		"motionSettlingFinalSpeed=%.9g "
		"motionControlFinalSpeed=%.9g "
		"depenetrationLimitedFirstStepRise=%.9g "
		"depenetrationControlFirstStepRise=%.9g "
		"depenetrationLimitedFinalRise=%.9g "
		"depenetrationLimitedMaxSpeed=%.9g "
		"speculativeCcdPositiveMinY=%.9g "
		"speculativeCcdPositiveMinSeparation=%.9g "
		"speculativeCcdNegativeMaxY=%.9g "
		"movingSpherePositiveDisplacement=%.9g "
		"movingSphereNegativeDisplacement=%.9g "
		"movingSpherePositiveMinSeparation=%.9g "
		"dynamicSphereSweepPositiveSoftDisplacement=%.9g "
		"dynamicSphereSweepNegativeSoftDisplacement=%.9g "
		"dynamicSphereSweepPositiveRigidDrop=%.9g "
		"dynamicSphereSweepNegativeRigidDrop=%.9g "
		"dynamicSphereSweepPositiveMinSeparation=%.9g "
		"fatalErrors=%u warningErrors=%u cleanupComplete=%u result=%s\n",
		Snippets::getSolverTypeName(options.solverType),
		options.caseName.c_str(),
		Snippets::getExecutionName(options.execution),
		options.frames,
		metrics.actorCreated, metrics.shapeAttached,
		metrics.hostBuffersInitialized, metrics.actorAdded,
		metrics.actorRemoved, metrics.actorReadded,
		metrics.fetchFailures, metrics.nonFiniteSamples,
		metrics.pinnedStable, metrics.dynamicMoved,
		metrics.boundsFinite,
		metrics.groundAdded, metrics.groundContactObserved,
		metrics.groundPenetrationBounded,
		metrics.groundSettled, metrics.surfaceSlept,
		metrics.initialSleepObserved, metrics.velocityWakeIssued,
		metrics.velocityWakeObserved,
		metrics.movedAfterVelocityWake,
		metrics.bufferMutationIssued,
		metrics.bufferMutationApplied,
		metrics.bufferPinHeld,
		metrics.bufferInvMassRestored,
		metrics.bufferRestoredMoved,
		metrics.dynamicBoxAdded,
		metrics.dynamicBoxInitiallySleeping,
		metrics.dynamicBoxWoke,
		metrics.dynamicBoxWakeFrame,
		metrics.dynamicBoxFinalSleeping,
		metrics.kinematicBoxAdded,
		metrics.kinematicTargetIssued,
		metrics.kinematicTargetReached,
		metrics.kinematicSurfaceWoke,
		metrics.kinematicSurfaceMoved,
		metrics.kinematicContactObserved,
		metrics.secondSurfaceCreated,
		metrics.secondSurfaceAdded,
		metrics.secondSurfaceInitiallySleeping,
		metrics.secondSurfaceWoke,
		metrics.secondSurfaceWakeFrame,
		metrics.secondSurfaceMoved,
		metrics.secondSurfaceFinalSleeping,
		metrics.mixedVolumeCreated,
		metrics.mixedVolumeAdded,
		metrics.mixedVolumeInitiallySleeping,
		metrics.mixedVolumeWoke,
		metrics.mixedVolumeWakeFrame,
		metrics.mixedVolumeMoved,
		metrics.mixedVolumeFinalSleeping,
		metrics.selfCollisionEnabled,
		metrics.selfCollisionPreventedCrossing,
		metrics.selfCollisionDisableIssued,
		metrics.selfCollisionDisabledCrossed,
		metrics.selfCollisionFilterApplied,
		metrics.selfCollisionFilterExcludedPair,
		metrics.materialFrictionLowApplied,
		metrics.materialFrictionHighApplied,
		metrics.materialFrictionResponseObserved,
		metrics.attachmentCreated,
		metrics.attachmentPinned,
		metrics.attachmentReleased,
		metrics.attachmentMovedAfterRelease,
		metrics.rigidAttachmentActorAdded,
		metrics.rigidAttachmentInitiallySleeping,
		metrics.rigidAttachmentCreated,
		metrics.rigidAttachmentRigidWoke,
		metrics.rigidAttachmentRigidMoved,
		metrics.rigidAttachmentRigidRotated,
		metrics.rigidAttachmentHeldAcrossReadd,
		metrics.rigidAttachmentReleased,
		metrics.rigidAttachmentSeparatedAfterRelease,
		metrics.articulationCreated,
		metrics.articulationAdded,
		metrics.articulationInitiallySleeping,
		metrics.articulationWoke,
		metrics.articulationJointSubspaceHeld,
		metrics.articulationRootStable,
		metrics.elementFilterCreated,
		metrics.elementFilterHeldAcrossReadd,
		metrics.elementFilterSuppressedContact,
		metrics.elementFilterReleased,
		metrics.elementFilterContactRestored,
		metrics.partialFilterExactOwnership,
		metrics.partialFilterUnfilteredContactHeld,
		metrics.bendingMaterialPairCreated,
		metrics.bendingZeroControlHeld,
		metrics.bendingResponseObserved,
		metrics.bendingMembraneIsolated,
		metrics.flatteningFlagApplied,
		metrics.flatteningControlHeld,
		metrics.flatteningResponseObserved,
		metrics.flatteningRetargetObserved,
		metrics.flatteningMembraneIsolated,
		metrics.motionMaxVelocityBounded,
		metrics.motionSettlingApplied,
		metrics.motionSettlingSlept,
		metrics.motionControlStayedAwake,
		metrics.depenetrationLimitApplied,
		metrics.depenetrationFirstStepBounded,
		metrics.depenetrationControlSeparated,
		metrics.depenetrationGradualRecovery,
		metrics.speculativeCcdFlagApplied,
		metrics.speculativeCcdPreventedTunneling,
		metrics.speculativeCcdNegativeControlTunneled,
		metrics.movingSphereTargetIssued,
		metrics.movingSphereCcdResponseObserved,
		metrics.movingSphereNegativeControlHeld,
		metrics.dynamicSphereSweepLaunched,
		metrics.dynamicSphereSweepResponseObserved,
		metrics.dynamicSphereSweepNegativeControlTunneled,
		metrics.dynamicSphereSweepTwoSidedResponseObserved,
		double(metrics.maxPinnedDrift),
		double(metrics.maxDynamicDisplacement),
		double(metrics.initialDynamicCentroidY),
		double(metrics.finalDynamicCentroidY),
		double(metrics.minY), double(metrics.finalMinY),
		double(metrics.maxSpeed), double(metrics.finalMaxSpeed),
		double(metrics.wakeCentroidY),
		double(metrics.maxWakeCentroidRise),
		double(metrics.bufferPinnedDrift),
		double(metrics.bufferRestoredDisplacement),
		double(metrics.dynamicBoxInitialY),
		double(metrics.dynamicBoxMinY),
		double(metrics.dynamicBoxFinalY),
		double(metrics.dynamicBoxMaxDrop),
		double(metrics.dynamicBoxMaxLinearSpeed),
		double(metrics.dynamicBoxFinalLinearSpeed),
		double(metrics.dynamicBoxMaxAngularSpeed),
		double(metrics.dynamicBoxFinalAngularSpeed),
		double(metrics.kinematicMaxPoseError),
		double(metrics.kinematicSurfaceDisplacement),
		double(metrics.kinematicFinalY),
		double(metrics.secondSurfaceInitialCentroidY),
		double(metrics.secondSurfaceFinalCentroidY),
		double(metrics.secondSurfaceMaxDisplacement),
		double(metrics.secondSurfaceMinY),
		double(metrics.secondSurfaceFinalMinY),
		double(metrics.secondSurfaceMaxSpeed),
		double(metrics.secondSurfaceFinalMaxSpeed),
		double(metrics.mixedVolumeInitialCentroidY),
		double(metrics.mixedVolumeFinalCentroidY),
		double(metrics.mixedVolumeMaxDisplacement),
		double(metrics.mixedVolumeMinY),
		double(metrics.mixedVolumeFinalMinY),
		double(metrics.mixedVolumeMaxSpeed),
		double(metrics.mixedVolumeFinalMaxSpeed),
		double(metrics.selfCollisionMinEnabledSeparation),
		double(metrics.selfCollisionMinDisabledSeparation),
		double(metrics.selfCollisionFilterMinSeparation),
		double(metrics.materialFrictionLowDisplacement),
		double(metrics.materialFrictionHighDisplacement),
		double(metrics.materialFrictionHighFinalSpeed),
		double(metrics.attachmentPinMaxDrift),
		double(metrics.attachmentReleasedMaxDisplacement),
		double(metrics.rigidAttachmentMaxDrift),
		double(metrics.rigidAttachmentMaxRigidDisplacement),
		double(metrics.rigidAttachmentMaxRigidSpeed),
		double(metrics.rigidAttachmentMaxAngularDisplacement),
		double(metrics.rigidAttachmentMaxAngularSpeed),
		double(metrics.rigidAttachmentReleasedSeparation),
		double(metrics.articulationRootMaxDisplacement),
		double(metrics.articulationChildMaxForbiddenDisplacement),
		double(metrics.articulationChildMaxAngularDisplacement),
		double(metrics.elementFilterMinY),
		double(metrics.elementFilterFinalMinY),
		double(metrics.partialFilterFilteredMinY),
		double(metrics.partialFilterUnfilteredMinY),
		double(metrics.bendingInitialPlaneError),
		double(metrics.bendingFinalPlaneError),
		double(metrics.bendingZeroControlDisplacement),
		double(metrics.bendingStiffDisplacement),
		double(metrics.bendingMaxEdgeStrain),
		double(metrics.flatteningInitialPlaneError),
		double(metrics.flatteningMinimumPlaneError),
		double(metrics.flatteningFinalPlaneError),
		double(metrics.flatteningControlDisplacement),
		double(metrics.flatteningTargetDisplacement),
		double(metrics.flatteningMaxEdgeStrain),
		double(metrics.motionMaxVelocityFirstStepDisplacement),
		double(metrics.motionMaxVelocityFirstStepSpeed),
		double(metrics.motionSettlingFinalSpeed),
		double(metrics.motionControlFinalSpeed),
		double(metrics.depenetrationLimitedFirstStepRise),
		double(metrics.depenetrationControlFirstStepRise),
		double(metrics.depenetrationLimitedFinalRise),
		double(metrics.depenetrationLimitedMaxSpeed),
		double(metrics.speculativeCcdPositiveMinY),
		double(metrics.speculativeCcdPositiveMinSeparation),
		double(metrics.speculativeCcdNegativeMaxY),
		double(metrics.movingSpherePositiveDisplacement),
		double(metrics.movingSphereNegativeDisplacement),
		double(metrics.movingSpherePositiveMinSeparation),
		double(metrics.dynamicSphereSweepPositiveSoftDisplacement),
		double(metrics.dynamicSphereSweepNegativeSoftDisplacement),
		double(metrics.dynamicSphereSweepPositiveRigidDrop),
		double(metrics.dynamicSphereSweepNegativeRigidDrop),
		double(metrics.dynamicSphereSweepPositiveMinSeparation),
		errorCallback.getFatalCount(),
		errorCallback.getWarningCount(),
		metrics.cleanupComplete,
		passed ? "PASS" : "FAIL");
}

} // namespace

#ifdef RENDER_SNIPPET

PxScene* gSurfaceAvbdScene = NULL;
SurfaceAvbdRenderData gSurfaceAvbdRenderData;

namespace
{

static PxDefaultAllocator sVisualAllocator;
static PxDefaultErrorCallback sVisualErrorCallback;
static PxFoundation* sVisualFoundation = NULL;
static PxPhysics* sVisualPhysics = NULL;
static PxDefaultCpuDispatcher* sVisualDispatcher = NULL;
static PxMaterial* sVisualRigidMaterial = NULL;
static PxDeformableSurfaceMaterial* sVisualSurfaceMaterial = NULL;
static PxTriangleMesh* sVisualTriangleMesh = NULL;
static PxDeformableSurface* sVisualSurface = NULL;
static PxRigidStatic* sVisualGround = NULL;
static PxRigidStatic* sVisualObstacle = NULL;
static std::vector<PxVec3> sVisualInitialPositions;
static std::vector<PxU64> sVisualMeshEdges;
static PxArray<Snippets::AvbdTriangleSkinningBinding>
	sVisualSkinningBindings;
static PxArray<PxU32> sVisualSkinningTriangles;
static PxArray<PxVec3> sVisualSkinningPositions;
static PxArray<PxVec3> sVisualSkinningNormals;
static bool sVisualExtensionsInitialized = false;
static bool sVisualPaused = false;

static bool buildVisualSurfaceMesh()
{
	const PxU32 width = 25;
	const PxU32 height = 25;
	const PxReal spacing = 0.2f;
	std::vector<PxU32> triangles;
	sVisualInitialPositions.clear();
	sVisualInitialPositions.reserve(width * height);
	triangles.reserve(6 * (width - 1) * (height - 1));
	sVisualMeshEdges.clear();

	for(PxU32 z = 0; z < height; ++z)
	{
		for(PxU32 x = 0; x < width; ++x)
		{
			const PxReal px =
				(PxReal(x) - 0.5f * PxReal(width - 1)) *
				spacing;
			const PxReal pz =
				(PxReal(z) - 0.5f * PxReal(height - 1)) *
				spacing;
			sVisualInitialPositions.push_back(
				PxVec3(px, 4.2f, pz));
		}
	}

	for(PxU32 z = 0; z + 1 < height; ++z)
	{
		for(PxU32 x = 0; x + 1 < width; ++x)
		{
			const PxU32 v00 = z * width + x;
			const PxU32 v10 = v00 + 1;
			const PxU32 v01 = v00 + width;
			const PxU32 v11 = v01 + 1;
			if((x + z) & 1)
			{
				triangles.push_back(v00);
				triangles.push_back(v01);
				triangles.push_back(v11);
				triangles.push_back(v00);
				triangles.push_back(v11);
				triangles.push_back(v10);
			}
			else
			{
				triangles.push_back(v00);
				triangles.push_back(v01);
				triangles.push_back(v10);
				triangles.push_back(v01);
				triangles.push_back(v11);
				triangles.push_back(v10);
			}
		}
	}
	sVisualMeshEdges.reserve(triangles.size());
	for(PxU32 triangle = 0;
		triangle + 2 < PxU32(triangles.size());
		triangle += 3)
	{
		for(PxU32 edge = 0; edge < 3; ++edge)
		{
			const PxU32 a = triangles[triangle + edge];
			const PxU32 b =
				triangles[triangle + (edge + 1) % 3];
			const PxU32 lo = PxMin(a, b);
			const PxU32 hi = PxMax(a, b);
			sVisualMeshEdges.push_back(
				(PxU64(lo) << 32) | PxU64(hi));
		}
	}
	std::sort(
		sVisualMeshEdges.begin(), sVisualMeshEdges.end());
	sVisualMeshEdges.erase(
		std::unique(
			sVisualMeshEdges.begin(), sVisualMeshEdges.end()),
		sVisualMeshEdges.end());

	PxTriangleMeshDesc meshDesc;
	meshDesc.points.count = PxU32(sVisualInitialPositions.size());
	meshDesc.points.stride = sizeof(PxVec3);
	meshDesc.points.data = sVisualInitialPositions.data();
	meshDesc.triangles.count = PxU32(triangles.size() / 3);
	meshDesc.triangles.stride = 3 * sizeof(PxU32);
	meshDesc.triangles.data = triangles.data();

	PxCookingParams cookingParams(
		sVisualPhysics->getTolerancesScale());
	cookingParams.buildGPUData = false;
	cookingParams.buildTriangleAdjacencies = true;
	sVisualTriangleMesh = PxCreateTriangleMesh(
		cookingParams, meshDesc,
		sVisualPhysics->getPhysicsInsertionCallback());
	return sVisualTriangleMesh != NULL;
}

static bool updateVisualSurfaceSkinning()
{
	if(!AVBD_SURFACE_ENABLE_SKINNING)
		return true;
	if(!sVisualSurface || !sVisualTriangleMesh)
		return false;
	if(!Snippets::evaluateTriangleSkinning(
		sVisualSurface->getPositionInvMassBufferH(),
		sVisualTriangleMesh->getNbVertices(),
		sVisualSkinningBindings, sVisualSkinningTriangles,
		sVisualSkinningPositions, sVisualSkinningNormals))
		return false;
	gSurfaceAvbdRenderData.skinnedPositions =
		sVisualSkinningPositions.begin();
	gSurfaceAvbdRenderData.skinnedNormals =
		sVisualSkinningNormals.begin();
	gSurfaceAvbdRenderData.skinnedTriangles =
		sVisualSkinningTriangles.begin();
	gSurfaceAvbdRenderData.skinnedVertexCount =
		sVisualSkinningPositions.size();
	gSurfaceAvbdRenderData.skinnedTriangleCount =
		sVisualSkinningTriangles.size() / 3;
	return true;
}

static bool initializeVisualSurfaceSkinning()
{
	sVisualSkinningBindings.clear();
	sVisualSkinningTriangles.clear();
	sVisualSkinningPositions.clear();
	sVisualSkinningNormals.clear();
	if(!AVBD_SURFACE_ENABLE_SKINNING)
		return true;
	if(!sVisualTriangleMesh)
		return false;
	const bool has16BitIndices =
		sVisualTriangleMesh->getTriangleMeshFlags() &
			PxTriangleMeshFlag::e16_BIT_INDICES;
	const PxU16* triangles16 = has16BitIndices
		? static_cast<const PxU16*>(
			sVisualTriangleMesh->getTriangles())
		: NULL;
	const PxU32* triangles32 = has16BitIndices
		? NULL
		: static_cast<const PxU32*>(
			sVisualTriangleMesh->getTriangles());
	for(PxU32 triangle = 0;
		triangle < sVisualTriangleMesh->getNbTriangles(); ++triangle)
	{
		PxU32 indices[3];
		for(PxU32 endpoint = 0; endpoint < 3; ++endpoint)
			indices[endpoint] = has16BitIndices
				? PxU32(triangles16[3 * triangle + endpoint])
				: triangles32[3 * triangle + endpoint];
		Snippets::appendTriangleSkinningPatch(
			indices, 2, sVisualSkinningBindings,
			sVisualSkinningTriangles);
	}
	return updateVisualSurfaceSkinning();
}

static void resetVisualSurface()
{
	if(!sVisualSurface ||
		!gSurfaceAvbdRenderData.positionsInvMass ||
		sVisualInitialPositions.empty())
		return;

	PxVec4* positions =
		gSurfaceAvbdRenderData.positionsInvMass;
	PxVec4* velocities = sVisualSurface->getVelocityBufferH();
	PxVec4* restPositions =
		sVisualSurface->getRestPositionBufferH();
	if(!velocities || !restPositions)
		return;

	for(PxU32 i = 0;
		i < PxU32(sVisualInitialPositions.size()); ++i)
	{
		positions[i] =
			PxVec4(sVisualInitialPositions[i], 1.0f);
		velocities[i] = PxVec4(0.0f);
		restPositions[i] =
			PxVec4(sVisualInitialPositions[i], 0.0f);
	}
	sVisualSurface->markDirty(
		PxDeformableSurfaceDataFlag::eALL);
	sVisualSurface->setWakeCounter(0.4f);
	updateVisualSurfaceSkinning();
}

} // namespace

bool initVisualPhysics()
{
	cleanupVisualPhysics();

	sVisualFoundation = PxCreateFoundation(
		PX_PHYSICS_VERSION, sVisualAllocator,
		sVisualErrorCallback);
	if(!sVisualFoundation)
		return false;

	const PxTolerancesScale scale;
	sVisualPhysics = PxCreatePhysics(
		PX_PHYSICS_VERSION, *sVisualFoundation,
		scale, true, NULL);
	if(!sVisualPhysics)
		return false;
	sVisualExtensionsInitialized =
		PxInitExtensions(*sVisualPhysics, NULL);
	if(!sVisualExtensionsInitialized)
		return false;

	PxSceneDesc sceneDesc(scale);
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	sceneDesc.solverType = PxSolverType::eAVBD;
	sceneDesc.filterShader = PxDefaultSimulationFilterShader;
	sVisualDispatcher = PxDefaultCpuDispatcherCreate(2);
	if(!sVisualDispatcher)
		return false;
	sceneDesc.cpuDispatcher = sVisualDispatcher;
	gSurfaceAvbdScene = sVisualPhysics->createScene(sceneDesc);
	if(!gSurfaceAvbdScene)
		return false;

	sVisualRigidMaterial =
		sVisualPhysics->createMaterial(0.7f, 0.6f, 0.0f);
	if(!sVisualRigidMaterial)
		return false;
	sVisualGround = PxCreatePlane(
		*sVisualPhysics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f),
		*sVisualRigidMaterial);
	sVisualObstacle = PxCreateStatic(
		*sVisualPhysics, PxTransform(PxVec3(0.0f, 1.0f, 0.0f)),
		PxBoxGeometry(1.15f, 1.0f, 1.15f),
		*sVisualRigidMaterial);
	if(!sVisualGround || !sVisualObstacle ||
		!gSurfaceAvbdScene->addActor(*sVisualGround) ||
		!gSurfaceAvbdScene->addActor(*sVisualObstacle))
		return false;

	if(!buildVisualSurfaceMesh())
		return false;
	sVisualSurfaceMaterial =
		sVisualPhysics->createDeformableSurfaceMaterial(
			2.0e4f, 0.3f, 0.08f, 0.02f, 0.2f);
	sVisualSurface = sVisualPhysics->createDeformableSurface(
		PxDeformableSurfaceBackend::eCPU_AVBD);
	if(!sVisualSurfaceMaterial || !sVisualSurface)
		return false;

	PxDeformableSurfaceMaterial* materials[] =
		{sVisualSurfaceMaterial};
	const PxShapeFlags shapeFlags =
		PxShapeFlag::eSIMULATION_SHAPE |
		PxShapeFlag::eSCENE_QUERY_SHAPE;
	PxShape* shape = sVisualPhysics->createShape(
		PxTriangleMeshGeometry(sVisualTriangleMesh),
		materials, 1, true, shapeFlags);
	if(!shape)
		return false;
	shape->setContactOffset(0.04f);
	shape->setRestOffset(0.01f);
	const bool attached = sVisualSurface->attachShape(*shape);
	shape->release();
	if(!attached)
		return false;

	gSurfaceAvbdRenderData.positionsInvMass =
		sVisualSurface->getPositionInvMassBufferH();
	gSurfaceAvbdRenderData.triangleMesh =
		sVisualTriangleMesh;
	if(!gSurfaceAvbdRenderData.positionsInvMass ||
		!sVisualSurface->getVelocityBufferH() ||
		!sVisualSurface->getRestPositionBufferH())
		return false;

	sVisualSurface->setLinearDamping(0.2f);
	sVisualSurface->setSolverIterationCounts(12);
	sVisualSurface->setSleepThreshold(5.0e-4f);
	sVisualSurface->setDeformableBodyFlag(
		PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, false);
	resetVisualSurface();
	if(!gSurfaceAvbdScene->addActor(*sVisualSurface))
		return false;
	if(!initializeVisualSurfaceSkinning())
		return false;

	sVisualPaused = false;
	std::printf(
		"%s visual controls: "
		"[P/Space] pause, [R] reset\n",
		AVBD_SURFACE_SNIPPET_NAME);
	return true;
}

void stepVisualPhysics()
{
	if(sVisualPaused || !gSurfaceAvbdScene)
		return;
	gSurfaceAvbdScene->simulate(1.0f / 60.0f);
	if(!gSurfaceAvbdScene->fetchResults(true))
	{
		std::printf(
			"%s: fetchResults failed; visual simulation paused.\n",
			AVBD_SURFACE_SNIPPET_NAME);
		sVisualPaused = true;
	}
	else if(!updateVisualSurfaceSkinning())
	{
		std::printf(
			"%s: CPU skinning update failed; "
			"visual simulation paused.\n",
			AVBD_SURFACE_SNIPPET_NAME);
		sVisualPaused = true;
	}
}

void cleanupVisualPhysics()
{
	gSurfaceAvbdRenderData.positionsInvMass = NULL;
	gSurfaceAvbdRenderData.triangleMesh = NULL;
	gSurfaceAvbdRenderData.skinnedPositions = NULL;
	gSurfaceAvbdRenderData.skinnedNormals = NULL;
	gSurfaceAvbdRenderData.skinnedTriangles = NULL;
	gSurfaceAvbdRenderData.skinnedVertexCount = 0;
	gSurfaceAvbdRenderData.skinnedTriangleCount = 0;
	if(sVisualSurface)
	{
		if(sVisualSurface->getScene())
			sVisualSurface->getScene()->removeActor(
				*sVisualSurface);
		sVisualSurface->release();
		sVisualSurface = NULL;
	}
	if(sVisualObstacle)
	{
		if(sVisualObstacle->getScene())
			sVisualObstacle->getScene()->removeActor(
				*sVisualObstacle);
		sVisualObstacle->release();
		sVisualObstacle = NULL;
	}
	if(sVisualGround)
	{
		if(sVisualGround->getScene())
			sVisualGround->getScene()->removeActor(
				*sVisualGround);
		sVisualGround->release();
		sVisualGround = NULL;
	}
	PX_RELEASE(sVisualSurfaceMaterial);
	PX_RELEASE(sVisualTriangleMesh);
	PX_RELEASE(sVisualRigidMaterial);
	PX_RELEASE(gSurfaceAvbdScene);
	PX_RELEASE(sVisualDispatcher);
	sVisualSkinningBindings.reset();
	sVisualSkinningTriangles.reset();
	sVisualSkinningPositions.reset();
	sVisualSkinningNormals.reset();
	if(sVisualExtensionsInitialized)
	{
		PxCloseExtensions();
		sVisualExtensionsInitialized = false;
	}
	PX_RELEASE(sVisualPhysics);
	PX_RELEASE(sVisualFoundation);
	sVisualInitialPositions.clear();
	sVisualMeshEdges.clear();
	sVisualPaused = false;
}

namespace
{

static bool segmentIntersectsVisualBoxInterior(
	const PxVec3& p0, const PxVec3& p1)
{
	// Shrink the obstacle very slightly so a segment lying on the contact
	// surface is not reported as penetration.  Any returned hit is at least
	// one millimetre inside the rendered box.
	const PxVec3 center(0.0f, 1.0f, 0.0f);
	const PxVec3 halfExtent(1.149f, 0.999f, 1.149f);
	const PxVec3 local0 = p0 - center;
	const PxVec3 direction = p1 - p0;
	PxReal tMinimum = 0.0f;
	PxReal tMaximum = 1.0f;
	for(PxU32 axis = 0; axis < 3; ++axis)
	{
		if(PxAbs(direction[axis]) <= 1.0e-8f)
		{
			if(PxAbs(local0[axis]) >= halfExtent[axis])
				return false;
			continue;
		}
		PxReal t0 =
			(-halfExtent[axis] - local0[axis]) /
			direction[axis];
		PxReal t1 =
			(halfExtent[axis] - local0[axis]) /
			direction[axis];
		if(t0 > t1)
		{
			const PxReal swap = t0;
			t0 = t1;
			t1 = swap;
		}
		tMinimum = PxMax(tMinimum, t0);
		tMaximum = PxMin(tMaximum, t1);
		if(tMinimum > tMaximum)
			return false;
	}
	return tMaximum >= 0.0f && tMinimum <= 1.0f;
}

static PxU32 countVisualBoxInteriorEdgeHits(bool& finite)
{
	finite = false;
	if(!gSurfaceAvbdRenderData.positionsInvMass ||
		!gSurfaceAvbdRenderData.triangleMesh)
		return 0;

	const PxVec4* positions =
		gSurfaceAvbdRenderData.positionsInvMass;
	const PxTriangleMesh* mesh =
		gSurfaceAvbdRenderData.triangleMesh;
	const PxU32 vertexCount = mesh->getNbVertices();
	for(PxU32 vertex = 0; vertex < vertexCount; ++vertex)
	{
		if(!positions[vertex].isFinite())
			return 0;
	}
	finite = true;

	const bool has16BitIndices =
		mesh->getTriangleMeshFlags() &
			PxTriangleMeshFlag::e16_BIT_INDICES;
	const PxU16* triangles16 = has16BitIndices
		? static_cast<const PxU16*>(mesh->getTriangles()) : NULL;
	const PxU32* triangles32 = has16BitIndices
		? NULL
		: static_cast<const PxU32*>(mesh->getTriangles());
	PxU32 hits = 0;
	for(PxU32 triangle = 0;
		triangle < mesh->getNbTriangles(); ++triangle)
	{
		const PxU32 base = 3 * triangle;
		PxU32 indices[3];
		for(PxU32 corner = 0; corner < 3; ++corner)
			indices[corner] = has16BitIndices
				? PxU32(triangles16[base + corner])
				: triangles32[base + corner];
		for(PxU32 edge = 0; edge < 3; ++edge)
		{
			const PxU32 i0 = indices[edge];
			const PxU32 i1 = indices[(edge + 1) % 3];
			if(i0 >= vertexCount || i1 >= vertexCount)
			{
				finite = false;
				return hits;
			}
			if(segmentIntersectsVisualBoxInterior(
				positions[i0].getXYZ(),
				positions[i1].getXYZ()))
				hits++;
		}
	}
	return hits;
}

struct VisualEdgeBounds
{
	PxU32 vertex0;
	PxU32 vertex1;
	PxVec3 minimum;
	PxVec3 maximum;
};

static void closestPointsOnVisualSegments(
	const PxVec3& p0, const PxVec3& p1,
	const PxVec3& q0, const PxVec3& q1,
	PxReal& pWeight1, PxReal& qWeight1,
	PxVec3& pClosest, PxVec3& qClosest)
{
	const PxVec3 dP = p1 - p0;
	const PxVec3 dQ = q1 - q0;
	const PxVec3 r = p0 - q0;
	const PxReal a = dP.dot(dP);
	const PxReal e = dQ.dot(dQ);
	const PxReal epsilon = 1.0e-12f;
	if(a <= epsilon && e <= epsilon)
	{
		pWeight1 = qWeight1 = 0.0f;
		pClosest = p0;
		qClosest = q0;
		return;
	}
	if(a <= epsilon)
	{
		pWeight1 = 0.0f;
		qWeight1 = PxClamp(
			dQ.dot(-r) / e, 0.0f, 1.0f);
	}
	else
	{
		const PxReal c = dP.dot(r);
		if(e <= epsilon)
		{
			qWeight1 = 0.0f;
			pWeight1 = PxClamp(
				-c / a, 0.0f, 1.0f);
		}
		else
		{
			const PxReal b = dP.dot(dQ);
			const PxReal f = dQ.dot(r);
			const PxReal denominator = a * e - b * b;
			pWeight1 = denominator > epsilon
				? PxClamp(
					(b * f - c * e) / denominator,
					0.0f, 1.0f)
				: 0.0f;
			qWeight1 = (b * pWeight1 + f) / e;
			if(qWeight1 < 0.0f)
			{
				qWeight1 = 0.0f;
				pWeight1 = PxClamp(
					-c / a, 0.0f, 1.0f);
			}
			else if(qWeight1 > 1.0f)
			{
				qWeight1 = 1.0f;
				pWeight1 = PxClamp(
					(b - c) / a, 0.0f, 1.0f);
			}
		}
	}
	pClosest = p0 + dP * pWeight1;
	qClosest = q0 + dQ * qWeight1;
}

static PxU32 countVisualSelfEdgeIntersections(bool& finite)
{
	finite = false;
	if(!gSurfaceAvbdRenderData.positionsInvMass ||
		sVisualMeshEdges.empty())
		return 0;
	const PxVec4* positions =
		gSurfaceAvbdRenderData.positionsInvMass;
	const PxU32 vertexCount =
		gSurfaceAvbdRenderData.triangleMesh
			? gSurfaceAvbdRenderData.triangleMesh->getNbVertices()
			: 0;

	std::vector<VisualEdgeBounds> edges;
	edges.reserve(sVisualMeshEdges.size());
	for(PxU32 edgeIndex = 0;
		edgeIndex < PxU32(sVisualMeshEdges.size()); ++edgeIndex)
	{
		const PxU64 key = sVisualMeshEdges[edgeIndex];
		const PxU32 vertex0 = PxU32(key >> 32);
		const PxU32 vertex1 = PxU32(key & 0xffffffffu);
		if(vertex0 >= vertexCount || vertex1 >= vertexCount ||
			!positions[vertex0].isFinite() ||
			!positions[vertex1].isFinite())
			return 0;
		VisualEdgeBounds bounds;
		bounds.vertex0 = vertex0;
		bounds.vertex1 = vertex1;
		bounds.minimum =
			positions[vertex0].getXYZ().minimum(
				positions[vertex1].getXYZ());
		bounds.maximum =
			positions[vertex0].getXYZ().maximum(
				positions[vertex1].getXYZ());
		edges.push_back(bounds);
	}
	finite = true;
	std::sort(
		edges.begin(), edges.end(),
		[](const VisualEdgeBounds& a,
		   const VisualEdgeBounds& b)
		{
			return a.minimum.x < b.minimum.x;
		});

	const PxReal intersectionTolerance = 1.0e-3f;
	const PxReal interiorEpsilon = 1.0e-4f;
	PxU32 intersections = 0;
	for(PxU32 edge0 = 0;
		edge0 < PxU32(edges.size()); ++edge0)
	{
		const VisualEdgeBounds& a = edges[edge0];
		for(PxU32 edge1 = edge0 + 1;
			edge1 < PxU32(edges.size()); ++edge1)
		{
			const VisualEdgeBounds& b = edges[edge1];
			if(b.minimum.x >
				a.maximum.x + intersectionTolerance)
				break;
			if(a.vertex0 == b.vertex0 ||
				a.vertex0 == b.vertex1 ||
				a.vertex1 == b.vertex0 ||
				a.vertex1 == b.vertex1)
				continue;
			if(a.minimum.y >
					b.maximum.y + intersectionTolerance ||
				a.maximum.y <
					b.minimum.y - intersectionTolerance ||
				a.minimum.z >
					b.maximum.z + intersectionTolerance ||
				a.maximum.z <
					b.minimum.z - intersectionTolerance)
				continue;

			PxReal weightA = 0.0f;
			PxReal weightB = 0.0f;
			PxVec3 closestA, closestB;
			closestPointsOnVisualSegments(
				positions[a.vertex0].getXYZ(),
				positions[a.vertex1].getXYZ(),
				positions[b.vertex0].getXYZ(),
				positions[b.vertex1].getXYZ(),
				weightA, weightB, closestA, closestB);
			if(weightA <= interiorEpsilon ||
				weightA >= 1.0f - interiorEpsilon ||
				weightB <= interiorEpsilon ||
				weightB >= 1.0f - interiorEpsilon)
				continue;
			if((closestA - closestB).magnitudeSquared() <
				intersectionTolerance * intersectionTolerance)
				intersections++;
		}
	}
	return intersections;
}

static bool runVisualOgcBoxEdgeCase(
	PxU32 frameCount, Metrics& metrics)
{
	PxU32 maximumInteriorEdgeHits = 0;
	PxU32 maximumSelfEdgeIntersections = 0;
	PxU32 firstHitFrame = PX_MAX_U32;
	PxU32 firstSelfIntersectionFrame = PX_MAX_U32;
	PxU32 nonFiniteFrames = 0;
	bool initialized = initVisualPhysics();
	if(initialized)
	{
		for(PxU32 frame = 0; frame < frameCount; ++frame)
		{
			stepVisualPhysics();
			bool finite = false;
			const PxU32 hits =
				countVisualBoxInteriorEdgeHits(finite);
			if(!finite)
				nonFiniteFrames++;
			bool selfFinite = false;
			const PxU32 selfIntersections =
				countVisualSelfEdgeIntersections(selfFinite);
			if(!selfFinite)
				nonFiniteFrames++;
			if(hits > 0 && firstHitFrame == PX_MAX_U32)
				firstHitFrame = frame;
			if(selfIntersections > 0 &&
				firstSelfIntersectionFrame == PX_MAX_U32)
				firstSelfIntersectionFrame = frame;
			maximumInteriorEdgeHits =
				PxMax(maximumInteriorEdgeHits, hits);
			maximumSelfEdgeIntersections =
				PxMax(
					maximumSelfEdgeIntersections,
					selfIntersections);
		}
	}
	cleanupVisualPhysics();
	metrics.cleanupComplete = 1;
	const bool passed = initialized &&
		nonFiniteFrames == 0 &&
		maximumInteriorEdgeHits == 0 &&
		maximumSelfEdgeIntersections == 0;
	std::printf(
		"[AVBD_OGC_BOX_EDGE] frames=%u "
		"maxInteriorEdgeHits=%u firstHitFrame=%u "
		"maxSelfEdgeIntersections=%u "
		"firstSelfIntersectionFrame=%u "
		"nonFiniteFrames=%u result=%s\n",
		frameCount, maximumInteriorEdgeHits, firstHitFrame,
		maximumSelfEdgeIntersections,
		firstSelfIntersectionFrame,
		nonFiniteFrames, passed ? "PASS" : "FAIL");
	return passed;
}

} // namespace

void keyPress(
	unsigned char key, const PxTransform& /*camera*/)
{
	if(key == ' ')
	{
		sVisualPaused = !sVisualPaused;
		return;
	}

	switch(std::toupper(static_cast<unsigned char>(key)))
	{
	case 'P':
		sVisualPaused = !sVisualPaused;
		break;
	case 'R':
		resetVisualSurface();
		sVisualPaused = false;
		break;
	default:
		break;
	}
}

#endif // RENDER_SNIPPET

#ifndef RENDER_SNIPPET
namespace
{
static bool runVisualOgcBoxEdgeCase(
	PxU32 frameCount, Metrics& metrics)
{
	PX_UNUSED(frameCount);
	metrics.cleanupComplete = 1;
	std::printf(
		"[AVBD_OGC_BOX_EDGE] result=UNSUPPORTED "
		"reason=render-scene-not-built\n");
	return false;
}
} // namespace
#endif

int snippetMain(int argc, const char* const* argv)
{
	Snippets::HeadlessOptions defaults;
	defaults.solverType = PxSolverType::eAVBD;
	defaults.caseName = AVBD_SURFACE_DEFAULT_CASE;
	defaults.frames = 180;
	defaults.dispatcherThreads = 2;

	Snippets::HeadlessOptions options;
	std::string parseError;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, options, parseError))
	{
		std::printf(
			"[AVBD_GATE_CONFIG_ERROR] %s\n", parseError.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	for(int i = 1; i < argc; ++i)
	{
		if(!Snippets::isCommonHeadlessOption(argv[i]))
		{
			std::printf(
				"[AVBD_GATE_CONFIG_ERROR] unknown option: %s\n",
				argv[i]);
			return Snippets::eHEADLESS_CONFIG_ERROR;
		}
	}
	if(!options.headless)
	{
#ifdef RENDER_SNIPPET
		extern void renderLoop();
		renderLoop();
		return 0;
#else
		std::printf(
			"%s: no render support in this build. Pass --headless.\n",
			AVBD_SURFACE_SNIPPET_NAME);
		return Snippets::eHEADLESS_CONFIG_ERROR;
#endif
	}
	if(options.solverType != PxSolverType::eAVBD)
	{
		std::printf(
			"[AVBD_GATE_UNSUPPORTED] reason=surface-is-avbd-only\n");
		return Snippets::eHEADLESS_UNSUPPORTED;
	}
	if(options.caseName != "surface-lifecycle" &&
		options.caseName != "surface-ogc-box-edge" &&
		options.caseName != "surface-ground" &&
		options.caseName != "surface-sleep-wake" &&
		options.caseName != "surface-buffer-mutation" &&
		options.caseName != "surface-dynamic-box" &&
		options.caseName != "surface-dynamic-sphere" &&
		options.caseName != "surface-dynamic-capsule" &&
		options.caseName != "surface-dynamic-convex" &&
		options.caseName != "surface-kinematic-box" &&
		options.caseName != "surface-kinematic-sphere" &&
		options.caseName != "surface-kinematic-capsule" &&
		options.caseName != "surface-kinematic-convex" &&
		options.caseName != "surface-kinematic-triangle-mesh" &&
		options.caseName != "surface-kinematic-heightfield" &&
		options.caseName != "surface-soft-soft-wake" &&
		options.caseName != "surface-soft-soft-swept-ccd" &&
		options.caseName != "surface-volume-wake" &&
		options.caseName != "surface-surface-attachment" &&
		options.caseName != "surface-volume-attachment" &&
		options.caseName != "surface-self-collision" &&
		options.caseName != "surface-self-collision-filter" &&
		options.caseName !=
			"surface-self-collision-swept-ccd" &&
		options.caseName != "surface-material-friction" &&
		options.caseName != "surface-world-pin" &&
		options.caseName != "surface-world-element-attachment" &&
		options.caseName != "surface-rigid-attachment" &&
		options.caseName != "surface-rigid-element-attachment" &&
		options.caseName != "surface-static-attachment" &&
		options.caseName != "surface-static-element-attachment" &&
		options.caseName != "surface-kinematic-attachment" &&
		options.caseName !=
			"surface-kinematic-element-attachment" &&
		options.caseName != "surface-articulation-attachment" &&
		options.caseName !=
			"surface-articulation-element-attachment" &&
		options.caseName != "surface-element-filter" &&
		options.caseName != "surface-partial-element-filter" &&
		options.caseName != "surface-soft-soft-element-filter" &&
		options.caseName != "surface-volume-element-filter" &&
		options.caseName != "volume-volume-element-filter" &&
		options.caseName != "surface-bending" &&
		options.caseName != "surface-flattening" &&
		options.caseName != "surface-motion-controls" &&
		options.caseName !=
			"surface-max-depenetration-velocity" &&
		options.caseName != "surface-speculative-ccd" &&
		options.caseName != "surface-plane-speculative-ccd" &&
		options.caseName != "surface-sphere-speculative-ccd" &&
		options.caseName != "surface-capsule-speculative-ccd" &&
		options.caseName != "surface-convex-speculative-ccd" &&
		options.caseName !=
			"surface-moving-kinematic-sphere-speculative-ccd" &&
		options.caseName !=
			"surface-moving-kinematic-capsule-speculative-ccd" &&
		options.caseName !=
			"surface-rotating-kinematic-capsule-speculative-ccd" &&
		options.caseName !=
			"surface-rotating-kinematic-convex-speculative-ccd" &&
		options.caseName !=
			"surface-moving-kinematic-convex-speculative-ccd" &&
		options.caseName !=
			"surface-dynamic-sphere-relative-swept-ccd" &&
		options.caseName !=
			"surface-dynamic-capsule-relative-swept-ccd" &&
		options.caseName !=
			"surface-dynamic-rotating-capsule-relative-swept-ccd" &&
		options.caseName !=
			"surface-dynamic-rotating-convex-relative-swept-ccd" &&
		options.caseName !=
			"surface-dynamic-convex-relative-swept-ccd" &&
		options.caseName !=
			"surface-static-sphere-reverse-swept-ccd" &&
		options.caseName !=
			"surface-kinematic-sphere-reverse-swept-ccd" &&
		options.caseName !=
			"surface-dynamic-sphere-reverse-swept-ccd" &&
		options.caseName !=
			"surface-static-capsule-reverse-swept-ccd" &&
		options.caseName !=
			"surface-kinematic-capsule-reverse-swept-ccd" &&
		options.caseName !=
			"surface-dynamic-capsule-reverse-swept-ccd" &&
		options.caseName !=
			"surface-rotating-kinematic-capsule-reverse-swept-ccd" &&
		options.caseName !=
			"surface-dynamic-rotating-capsule-reverse-swept-ccd" &&
		options.caseName !=
			"surface-rotating-kinematic-convex-reverse-swept-ccd" &&
		options.caseName !=
			"surface-dynamic-rotating-convex-reverse-swept-ccd" &&
		options.caseName !=
			"surface-static-convex-reverse-swept-ccd" &&
		options.caseName !=
			"surface-kinematic-convex-reverse-swept-ccd" &&
		options.caseName !=
			"surface-dynamic-convex-reverse-swept-ccd" &&
		options.caseName !=
			"surface-deforming-sphere-reverse-swept-ccd" &&
		options.caseName !=
			"surface-deforming-capsule-reverse-swept-ccd" &&
		options.caseName !=
			"surface-deforming-convex-reverse-swept-ccd" &&
		options.caseName !=
			"surface-deforming-triangle-mesh-reverse-swept-ccd" &&
		options.caseName !=
			"surface-deforming-heightfield-reverse-swept-ccd" &&
		options.caseName !=
			"surface-static-triangle-mesh-speculative-ccd" &&
		options.caseName !=
			"surface-kinematic-triangle-mesh-speculative-ccd" &&
		options.caseName !=
			"surface-static-heightfield-speculative-ccd" &&
		options.caseName !=
			"surface-kinematic-heightfield-speculative-ccd" &&
		options.caseName !=
			"surface-static-triangle-mesh-reverse-swept-ccd" &&
		options.caseName !=
			"surface-kinematic-triangle-mesh-reverse-swept-ccd" &&
		options.caseName !=
			"surface-static-heightfield-reverse-swept-ccd" &&
		options.caseName !=
			"surface-kinematic-heightfield-reverse-swept-ccd" &&
		options.caseName !=
			"surface-rotating-kinematic-triangle-mesh-speculative-ccd" &&
		options.caseName !=
			"surface-rotating-kinematic-heightfield-speculative-ccd" &&
		options.caseName !=
			"surface-rotating-kinematic-triangle-mesh-reverse-swept-ccd" &&
		options.caseName !=
			"surface-rotating-kinematic-heightfield-reverse-swept-ccd" &&
			options.caseName != "surface-sphere-reverse-feature" &&
			options.caseName != "surface-capsule-reverse-feature" &&
			options.caseName != "surface-convex-reverse-feature" &&
			options.caseName !=
				"surface-triangle-mesh-reverse-feature" &&
			options.caseName !=
				"surface-heightfield-reverse-feature" &&
			options.caseName != "surface-skinning")
	{
		std::printf(
			"[AVBD_GATE_CONFIG_ERROR] unknown case: %s\n",
			options.caseName.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(!Snippets::applyExecutionEnvironment(options))
	{
		std::printf(
			"[AVBD_GATE_CONFIG_ERROR] "
			"failed to apply execution environment\n");
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}

	Snippets::printHeadlessConfig(
		AVBD_SURFACE_SNIPPET_NAME, options);
	Snippets::TrackingErrorCallback errorCallback;
	Metrics metrics;
	gSurfaceSkinningMetrics = SurfaceSkinningMetrics();
	const bool groundCase =
		options.caseName == "surface-ground" ||
		options.caseName == "surface-sleep-wake";
	const bool sleepWakeCase =
		options.caseName == "surface-sleep-wake";
	const bool bufferMutationCase =
		options.caseName == "surface-buffer-mutation";
	const bool dynamicRigidCase =
		options.caseName == "surface-dynamic-box" ||
		options.caseName == "surface-dynamic-sphere" ||
		options.caseName == "surface-dynamic-capsule" ||
		options.caseName == "surface-dynamic-convex";
	const bool kinematicRigidCase =
		options.caseName == "surface-kinematic-box" ||
		options.caseName == "surface-kinematic-sphere" ||
		options.caseName == "surface-kinematic-capsule" ||
		options.caseName == "surface-kinematic-convex" ||
		options.caseName ==
			"surface-kinematic-triangle-mesh" ||
		options.caseName ==
			"surface-kinematic-heightfield";
	const bool softSoftCase =
		options.caseName == "surface-soft-soft-wake" ||
		options.caseName == "surface-soft-soft-swept-ccd" ||
		options.caseName == "surface-surface-attachment";
	const bool surfaceVolumeCase =
		options.caseName == "surface-volume-wake" ||
		options.caseName == "surface-volume-attachment";
	const bool selfCollisionCase =
		options.caseName == "surface-self-collision" ||
		options.caseName == "surface-self-collision-filter" ||
		options.caseName ==
			"surface-self-collision-swept-ccd";
	const bool materialFrictionCase =
		options.caseName == "surface-material-friction";
	const bool worldPinCase =
		options.caseName == "surface-world-pin" ||
		options.caseName == "surface-world-element-attachment";
	const bool rigidAttachmentCase =
		options.caseName == "surface-rigid-attachment" ||
		options.caseName == "surface-rigid-element-attachment";
	const bool staticAttachmentCase =
		options.caseName == "surface-static-attachment" ||
		options.caseName == "surface-static-element-attachment";
	const bool kinematicAttachmentCase =
		options.caseName == "surface-kinematic-attachment" ||
		options.caseName ==
			"surface-kinematic-element-attachment";
	const bool articulationAttachmentCase =
		options.caseName == "surface-articulation-attachment" ||
		options.caseName ==
			"surface-articulation-element-attachment";
	const bool elementFilterCase =
		options.caseName == "surface-element-filter" ||
		options.caseName == "surface-partial-element-filter" ||
		options.caseName == "surface-soft-soft-element-filter";
	const bool partialElementFilterCase =
		options.caseName == "surface-partial-element-filter" ||
		options.caseName == "surface-soft-soft-element-filter";
	const bool softSoftElementFilterCase =
		options.caseName == "surface-soft-soft-element-filter";
	const bool bendingCase =
		options.caseName == "surface-bending";
	const bool flatteningCase =
		options.caseName == "surface-flattening";
	const bool motionControlsCase =
		options.caseName == "surface-motion-controls";
	const bool maxDepenetrationVelocityCase =
		options.caseName ==
			"surface-max-depenetration-velocity";
	const bool speculativeCcdCase =
		options.caseName == "surface-speculative-ccd" ||
		options.caseName == "surface-plane-speculative-ccd" ||
		options.caseName == "surface-sphere-speculative-ccd" ||
		options.caseName == "surface-capsule-speculative-ccd" ||
		options.caseName == "surface-convex-speculative-ccd";
	const bool movingKinematicSphereSpeculativeCcdCase =
		options.caseName ==
			"surface-moving-kinematic-sphere-speculative-ccd" ||
		options.caseName ==
			"surface-moving-kinematic-capsule-speculative-ccd" ||
		options.caseName ==
			"surface-rotating-kinematic-capsule-speculative-ccd" ||
		options.caseName ==
			"surface-rotating-kinematic-convex-speculative-ccd" ||
		options.caseName ==
			"surface-moving-kinematic-convex-speculative-ccd";
	const bool dynamicSphereRelativeSweptCcdCase =
		options.caseName ==
			"surface-dynamic-sphere-relative-swept-ccd" ||
		options.caseName ==
			"surface-dynamic-capsule-relative-swept-ccd" ||
		options.caseName ==
			"surface-dynamic-rotating-capsule-relative-swept-ccd" ||
		options.caseName ==
			"surface-dynamic-rotating-convex-relative-swept-ccd" ||
		options.caseName ==
			"surface-dynamic-convex-relative-swept-ccd";
	const bool finiteReverseSweptCcdCase =
		options.caseName ==
			"surface-deforming-sphere-reverse-swept-ccd" ||
		options.caseName ==
			"surface-deforming-capsule-reverse-swept-ccd" ||
		options.caseName ==
			"surface-static-sphere-reverse-swept-ccd" ||
		options.caseName ==
			"surface-kinematic-sphere-reverse-swept-ccd" ||
		options.caseName ==
			"surface-dynamic-sphere-reverse-swept-ccd" ||
		options.caseName ==
			"surface-static-capsule-reverse-swept-ccd" ||
		options.caseName ==
			"surface-kinematic-capsule-reverse-swept-ccd" ||
		options.caseName ==
			"surface-dynamic-capsule-reverse-swept-ccd" ||
		options.caseName ==
			"surface-rotating-kinematic-capsule-reverse-swept-ccd" ||
		options.caseName ==
			"surface-dynamic-rotating-capsule-reverse-swept-ccd";
	const bool convexReverseSweptCcdCase =
		options.caseName ==
			"surface-deforming-convex-reverse-swept-ccd" ||
		options.caseName ==
			"surface-static-convex-reverse-swept-ccd" ||
		options.caseName ==
			"surface-kinematic-convex-reverse-swept-ccd" ||
		options.caseName ==
			"surface-dynamic-convex-reverse-swept-ccd" ||
		options.caseName ==
			"surface-rotating-kinematic-convex-reverse-swept-ccd" ||
		options.caseName ==
			"surface-dynamic-rotating-convex-reverse-swept-ccd";
	const bool triangleSurfaceSweptCcdCase =
		options.caseName ==
			"surface-deforming-triangle-mesh-reverse-swept-ccd" ||
		options.caseName ==
			"surface-deforming-heightfield-reverse-swept-ccd" ||
		options.caseName ==
			"surface-static-triangle-mesh-speculative-ccd" ||
		options.caseName ==
			"surface-kinematic-triangle-mesh-speculative-ccd" ||
		options.caseName ==
			"surface-static-heightfield-speculative-ccd" ||
		options.caseName ==
			"surface-kinematic-heightfield-speculative-ccd" ||
		options.caseName ==
			"surface-static-triangle-mesh-reverse-swept-ccd" ||
		options.caseName ==
			"surface-kinematic-triangle-mesh-reverse-swept-ccd" ||
		options.caseName ==
			"surface-static-heightfield-reverse-swept-ccd" ||
		options.caseName ==
			"surface-kinematic-heightfield-reverse-swept-ccd" ||
		options.caseName ==
			"surface-rotating-kinematic-triangle-mesh-speculative-ccd" ||
		options.caseName ==
			"surface-rotating-kinematic-heightfield-speculative-ccd" ||
		options.caseName ==
			"surface-rotating-kinematic-triangle-mesh-reverse-swept-ccd" ||
		options.caseName ==
			"surface-rotating-kinematic-heightfield-reverse-swept-ccd";
	const bool sphereReverseFeatureCase =
		options.caseName == "surface-sphere-reverse-feature";
	const bool capsuleReverseFeatureCase =
		options.caseName == "surface-capsule-reverse-feature";
	const bool convexReverseFeatureCase =
		options.caseName == "surface-convex-reverse-feature";
	const bool triangleMeshReverseFeatureCase =
		options.caseName ==
			"surface-triangle-mesh-reverse-feature";
	const bool heightFieldReverseFeatureCase =
		options.caseName ==
			"surface-heightfield-reverse-feature";
	const bool skinningCase =
		options.caseName == "surface-skinning";
	const bool volumePairFilterCase =
		options.caseName == "surface-volume-element-filter" ||
		options.caseName == "volume-volume-element-filter";
	const bool ogcBoxEdgeCase =
		options.caseName == "surface-ogc-box-edge";
	const bool simulated =
		ogcBoxEdgeCase
			? runVisualOgcBoxEdgeCase(options.frames, metrics)
			: volumePairFilterCase
			? runVolumePairFilterCase(
				options, errorCallback, metrics,
				options.caseName ==
					"volume-volume-element-filter")
			: flatteningCase
			? runFlatteningCase(options, errorCallback, metrics)
			: motionControlsCase
			? runMotionControlsCase(options, errorCallback, metrics)
			: maxDepenetrationVelocityCase
			? runMaxDepenetrationVelocityCase(
				options, errorCallback, metrics)
			: speculativeCcdCase
			? runSpeculativeCcdCase(
				options, errorCallback, metrics)
			: movingKinematicSphereSpeculativeCcdCase
			? runMovingKinematicFiniteSpeculativeCcdCase(
				options, errorCallback, metrics)
			: dynamicSphereRelativeSweptCcdCase
			? runDynamicFiniteRelativeSweptCcdCase(
				options, errorCallback, metrics)
			: finiteReverseSweptCcdCase ||
				convexReverseSweptCcdCase
			? runFiniteReverseSweptCcdCase(
				options, errorCallback, metrics)
			: triangleSurfaceSweptCcdCase
			? runTriangleSurfaceSweptCcdCase(
				options, errorCallback, metrics)
			: sphereReverseFeatureCase
			? runSmoothReverseFeatureCase(
				options, errorCallback, metrics,
				false, false, false, false)
			: capsuleReverseFeatureCase
			? runSmoothReverseFeatureCase(
				options, errorCallback, metrics,
				true, false, false, false)
			: convexReverseFeatureCase
			? runSmoothReverseFeatureCase(
				options, errorCallback, metrics,
				false, true, false, false)
			: triangleMeshReverseFeatureCase
			? runSmoothReverseFeatureCase(
				options, errorCallback, metrics,
				false, false, true, false)
			: heightFieldReverseFeatureCase
			? runSmoothReverseFeatureCase(
				options, errorCallback, metrics,
				false, false, false, true)
			: bendingCase
			? runBendingCase(options, errorCallback, metrics)
			: runSurfaceCase(
				options, errorCallback, metrics,
				groundCase, sleepWakeCase,
				bufferMutationCase, dynamicRigidCase,
				kinematicRigidCase, softSoftCase,
				surfaceVolumeCase, selfCollisionCase,
				materialFrictionCase,
				worldPinCase, rigidAttachmentCase,
				staticAttachmentCase,
				kinematicAttachmentCase,
				articulationAttachmentCase,
				elementFilterCase,
				partialElementFilterCase,
				softSoftElementFilterCase,
				skinningCase);
	const bool passed = simulated &&
		errorCallback.getFatalCount() == 0 &&
		metrics.cleanupComplete != 0;
	if(skinningCase)
	{
		std::printf(
			"[AVBD_CPU_SKINNING] schema=1 snippet=%s "
			"kind=surface vertices=%u triangles=%u "
			"evaluatedFrames=%u finiteFrames=%u "
			"maxDisplacement=%.9g status=%s\n",
			AVBD_SURFACE_SNIPPET_NAME,
			gSurfaceSkinningMetrics.vertices,
			gSurfaceSkinningMetrics.triangles,
			gSurfaceSkinningMetrics.evaluatedFrames,
			gSurfaceSkinningMetrics.finiteFrames,
			double(gSurfaceSkinningMetrics.maxDisplacement),
			passed ? "PASS" : "FAIL");
	}
	printResult(options, errorCallback, metrics, passed);
	return passed ? Snippets::eHEADLESS_PASS
		: Snippets::eHEADLESS_GATE_FAILED;
}
