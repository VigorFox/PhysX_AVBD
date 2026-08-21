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

#include "SnippetDeformableVolumeAVBDValidation.h"

#include "GuIntersectionTriangleBox.h"
#include "extensions/PxTetrahedronMeshExt.h"

#include <cfloat>

using namespace physx;

namespace SnippetDeformableVolumeAVBDValidation
{

DeformableVolumeMetrics::DeformableVolumeMetrics()
	: initialized(0), completedFrames(0), fetchFailures(0),
	  nonFiniteParticleSamples(0), invertedElementSamples(0),
	  firstInversionFrame(PX_MAX_U32), firstInversionBody(PX_MAX_U32),
	  firstInversionElement(PX_MAX_U32), invertedBodiesMask(0),
	  particles(0), softBodies(0), tetElements(0), surfaceTriangles(0),
	  rigidBoxes(0),
	  sceneStatics(0), sceneDynamics(0), sceneDeformableVolumes(0),
	  groundContactFrames(0), rigidContactFrames(0), softContactFrames(0),
	  maxGroundContacts(0), maxRigidContacts(0), maxSoftContacts(0),
	  invalidContactSourceSamples(0), finalInsideParticles(0),
	  cleanupComplete(0), sceneActorCreated(0), sceneShapeAttached(0),
	  sceneSimulationMeshAttached(0), sceneHostBuffersInitialized(0),
	  sceneActorAdded(0), sceneActorRemoved(0), sceneActorReleased(0),
	  sceneBoundsFinite(0), sceneStaticShapeDetached(0),
	  sceneStaticShapeReattached(0), sceneStaticActorRemoved(0),
	  sceneStaticActorReadded(0), sceneDynamicActorAdded(0),
	  sceneDynamicActorRemoved(0),
	  sceneDynamicActorReleased(0), sceneDynamicInitiallySleeping(0),
	  sceneDynamicWokeBySoft(0),
	  sceneDynamicFirstWakeFrame(PX_MAX_U32),
	  sceneDynamicShapeDetached(0),
	  sceneDynamicShapeReattached(0),
	  sceneDynamicActorReadded(0),
	  sceneDynamicReaddedSleeping(0),
	  sceneDynamicRewokeBySoft(0),
	  sceneDynamicSecondWakeFrame(PX_MAX_U32),
	  sceneSecondDynamicActorAdded(0),
	  sceneSecondDynamicActorRemoved(0),
	  sceneSecondDynamicActorReleased(0),
	  sceneSecondDynamicInitiallySleeping(0),
	  sceneSecondDynamicWokeBySoft(0),
	  sceneSecondDynamicFirstWakeFrame(PX_MAX_U32),
	  sceneSecondVolumeActorCreated(0),
	  sceneSecondVolumeHostBuffersInitialized(0),
	  sceneSecondVolumeActorAdded(0),
	  sceneSecondVolumeActorRemoved(0),
	  sceneSecondVolumeActorReleased(0),
	  sceneSecondVolumeBoundsFinite(0),
	  sceneSoftInitiallyAwake(0),
	  sceneSoftFirstSlept(0),
	  sceneSoftFirstSleepFrame(PX_MAX_U32),
	  sceneSoftSleepWakeCounterZero(0),
	  sceneSoftSleepVelocitiesZero(0),
	  sceneSoftStableWhileSleeping(0),
	  sceneSoftCounterWakeIssued(0),
	  sceneSoftWokeByCounter(0),
	  sceneSoftCounterWakeFrame(PX_MAX_U32),
	  sceneSoftSecondSlept(0),
	  sceneSoftSecondSleepFrame(PX_MAX_U32),
	  sceneSoftVelocityWakeIssued(0),
	  sceneSoftWokeByVelocity(0),
	  sceneSoftVelocityWakeFrame(PX_MAX_U32),
	  sceneSoftMovedAfterVelocityWake(0),
	  sceneSoftVelocityStopIssued(0),
	  sceneSoftFinalSlept(0),
	  sceneSoftFinalSleepFrame(PX_MAX_U32),
	  sceneSoftRigidWakeActorAdded(0),
	  sceneSoftWokeByRigid(0),
	  sceneSoftRigidWakeFrame(PX_MAX_U32),
	  sceneSoftMovedAfterRigidWake(0),
	  sceneMixedFirstSlept(0),
	  sceneMixedFirstSleepFrame(PX_MAX_U32),
	  sceneMixedFirstStable(0),
	  sceneMixedSecondStayedAwake(0),
	  sceneMixedSecondMoved(0),
	  sceneSoftChurnRemoveCount(0),
	  sceneSoftChurnReaddCount(0),
	  sceneSoftChurnCycles(0),
	  sceneSoftChurnPostCompactMoveCount(0),
	  sceneSoftChurnStable(0),
	  sceneBufferMutationIssued(0),
	  sceneBufferMutationWoke(0),
	  sceneBufferMutationApplied(0),
	  sceneBufferDriveIssued(0),
	  sceneBufferPinHeld(0),
	  sceneBufferDynamicMoved(0),
	  sceneBufferInvMassRestored(0),
	  sceneBufferRestoredMoved(0),
	  sceneBufferResetIssued(0),
	  sceneWorldPinCreated(0),
	  sceneWorldPinHeld(0),
	  sceneWorldPinActorReadded(0),
	  sceneWorldPinReleased(0),
	  sceneWorldPinMovedAfterRelease(0),
	  sceneRigidAttachmentActorAdded(0),
	  sceneRigidAttachmentInitiallySleeping(0),
	  sceneRigidAttachmentCreated(0),
	  sceneRigidAttachmentRigidWoke(0),
	  sceneRigidAttachmentRigidMoved(0),
	  sceneRigidAttachmentHeldAcrossReadd(0),
	  sceneRigidAttachmentReleased(0),
	  sceneRigidAttachmentSeparatedAfterRelease(0),
	  sceneArticulationCreated(0),
	  sceneArticulationAdded(0),
	  sceneArticulationInitiallySleeping(0),
	  sceneArticulationWoke(0),
	  sceneArticulationJointSubspaceHeld(0),
	  sceneArticulationRootStable(0),
	  sceneElementFilterCreated(0),
	  sceneElementFilterActorReadded(0),
	  sceneElementFilterSuppressedContact(0),
	  sceneElementFilterReleased(0),
	  sceneElementFilterContactRestored(0),
	  scenePartialFilterUnfilteredContactHeld(0),
	  scenePartialFilterExactOwnership(0),
	  sceneKinematicActorAdded(0),
	  sceneKinematicTargetIssued(0),
	  sceneKinematicTargetReached(0),
	  sceneKinematicSoftWoke(0),
	  sceneKinematicSoftMoved(0),
	  sceneKinematicContactObserved(0),
	  sceneVolumeTargetBound(0),
	  sceneVolumeTargetMutated(0),
	  sceneVolumeTargetWoke(0),
	  sceneVolumeTargetReached(0),
	  sceneVolumePartialInactiveIgnored(0),
	  sceneVolumePartialActivated(0),
	  sceneVolumePartialActivatedReached(0),
	  sceneSecondSceneCreated(0),
	  sceneSecondSceneSolverMatched(0),
	  scenePrimarySceneReleased(0),
	  sceneSecondSceneReleased(0),
	  sceneMultiPrimaryStable(0),
	  sceneMultiPrimaryDetachedStable(0),
	  sceneMultiSecondaryUpdatedBeforeRelease(0),
	  sceneMultiSecondaryUpdatedAfterRelease(0),
	  sceneSoftSoftBothSlept(0),
	  sceneSoftSoftDriveIssued(0),
	  sceneSoftSoftDriverWoke(0),
	  sceneSoftSoftTargetWoke(0),
	  sceneSoftSoftTargetWakeFrame(PX_MAX_U32),
	  sceneSoftSoftTargetMoved(0),
	  sceneSoftSoftResetIssued(0),
	  sceneSoftSoftBothFinalSlept(0),
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
	  minDetF(FLT_MAX),
	  maxDetF(-FLT_MAX), minBodyVolumeRatio(FLT_MAX),
	  maxBodyVolumeRatio(-FLT_MAX), minY(FLT_MAX), maxY(-FLT_MAX),
	  maxParticleSpeed(0.0f), finalMinY(FLT_MAX), finalMaxY(-FLT_MAX),
	  finalMaxParticleSpeed(0.0f),
	  maxCentroidDrop(0.0f), sceneDynamicMinY(FLT_MAX),
	  sceneDynamicFinalY(FLT_MAX),
	  sceneDynamicMaxDrop(0.0f),
	  sceneDynamicPreContactMaxDrop(0.0f),
	  sceneDynamicMaxDownSpeed(0.0f),
	  sceneSecondDynamicMinY(FLT_MAX),
	  sceneSecondDynamicFinalY(FLT_MAX),
	  sceneSecondDynamicMaxDrop(0.0f),
	  sceneSecondDynamicPreContactMaxDrop(0.0f),
	  sceneSecondDynamicMaxDownSpeed(0.0f),
	  sceneSecondVolumeMaxCentroidDrop(0.0f),
	  sceneSecondVolumeFinalCentroidY(FLT_MAX),
	  sceneWorldPinMaxDrift(0.0f),
	  sceneWorldPinReleasedMaxDisplacement(0.0f),
	  sceneRigidAttachmentMaxDrift(0.0f),
	  sceneRigidAttachmentMaxRigidDisplacement(0.0f),
	  sceneRigidAttachmentMaxRigidSpeed(0.0f),
	  sceneRigidAttachmentReleasedSeparation(0.0f),
	  sceneArticulationRootMaxDisplacement(0.0f),
	  sceneArticulationChildMaxForbiddenDisplacement(0.0f),
	  sceneArticulationChildMaxAngularDisplacement(0.0f),
	  sceneElementFilterMinY(FLT_MAX),
	  sceneElementFilterFinalMinY(FLT_MAX),
	  scenePartialFilterUnfilteredMinY(FLT_MAX),
	  sceneKinematicMaxPoseError(0.0f),
	  sceneKinematicSoftDisplacement(0.0f),
	  sceneKinematicFinalY(FLT_MAX),
	  sceneVolumeTargetFinalMaxError(FLT_MAX),
	  sceneVolumeTargetMaxDisplacement(0.0f),
	  sceneVolumePartialInactiveDecoyDistance(0.0f),
	  minDynamicSurfaceSeparation(FLT_MAX),
	  finalDynamicSurfaceSeparation(FLT_MAX),
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
	  dynamicSphereSweepPositiveMinSeparation(PX_MAX_F32),
	  solverReadbackMatched(false)
	{
	}

SceneCommonValidationConfig::SceneCommonValidationConfig()
	: expectedFrames(0), expectedSceneVolumeCount(0),
	  expectedSceneDynamicCount(0), centroidDropOptional(false)
{
}

bool isSceneCommonResultValid(
	const DeformableVolumeMetrics& metrics,
	const SceneCommonValidationConfig& config,
	PxU32 fatalErrorCount)
{
	return metrics.initialized == 1 &&
		metrics.completedFrames == config.expectedFrames &&
		metrics.fetchFailures == 0 &&
		metrics.nonFiniteParticleSamples == 0 &&
		metrics.sceneActorCreated == 1 &&
		metrics.sceneShapeAttached == 1 &&
		metrics.sceneSimulationMeshAttached == 1 &&
		metrics.sceneHostBuffersInitialized == 1 &&
		metrics.sceneActorAdded == 1 &&
		metrics.sceneActorRemoved == 1 &&
		metrics.sceneActorReleased == 1 &&
		metrics.sceneBoundsFinite == 1 &&
		metrics.sceneDynamics == config.expectedSceneDynamicCount &&
		metrics.sceneDeformableVolumes ==
			config.expectedSceneVolumeCount &&
		metrics.particles > 0 &&
		metrics.softBodies == config.expectedSceneVolumeCount &&
		metrics.tetElements > 0 && metrics.solverReadbackMatched &&
		(config.centroidDropOptional ||
		 metrics.maxCentroidDrop > 0.0001f) &&
		PxIsFinite(metrics.finalMinY) &&
		PxIsFinite(metrics.finalMaxY) &&
		PxIsFinite(metrics.finalMaxParticleSpeed) &&
		fatalErrorCount == 0 && metrics.cleanupComplete == 1;
}

ComponentValidationConfig::ComponentValidationConfig()
	: expectedFrames(0), fatalErrorCount(0),
	  caseType(eCOMPONENT_CURRENT_ALL)
{
}

bool isComponentResultValid(
	const DeformableVolumeMetrics& metrics,
	const ComponentValidationConfig& config)
{
	const bool denseNoContactCase =
		config.caseType == eCOMPONENT_DENSE_NO_CONTACT;
	const bool manySmallNoContactCase =
		config.caseType == eCOMPONENT_MANY_SMALL_NO_CONTACT;
	const bool noContactCase =
		denseNoContactCase || manySmallNoContactCase;
	bool passed =
		metrics.initialized == 1 &&
		metrics.completedFrames == config.expectedFrames &&
		metrics.fetchFailures == 0 &&
		metrics.nonFiniteParticleSamples == 0 &&
		metrics.invertedElementSamples == 0 &&
		metrics.invalidContactSourceSamples == 0 &&
		metrics.particles > 0 && metrics.softBodies > 0 &&
		metrics.tetElements > 0 && metrics.surfaceTriangles > 0 &&
		metrics.sceneStatics == metrics.rigidBoxes +
			(noContactCase ? 0u : 1u) &&
		metrics.sceneDynamics == 0 &&
		metrics.sceneDeformableVolumes == 0 &&
		metrics.solverReadbackMatched && metrics.cleanupComplete == 1 &&
		PxIsFinite(metrics.minDetF) && metrics.minDetF > 0.0f &&
		PxIsFinite(metrics.maxDetF) && metrics.maxDetF < 20.0f &&
		PxIsFinite(metrics.minBodyVolumeRatio) &&
		metrics.minBodyVolumeRatio > 0.01f &&
		PxIsFinite(metrics.maxBodyVolumeRatio) &&
		metrics.maxBodyVolumeRatio < 20.0f &&
		PxIsFinite(metrics.minY) && metrics.minY > -0.25f &&
		PxIsFinite(metrics.maxY) && metrics.maxY < 100.0f &&
		PxIsFinite(metrics.maxParticleSpeed) &&
		metrics.maxParticleSpeed < 250.0f &&
		config.fatalErrorCount == 0;

	switch(config.caseType)
	{
	case eCOMPONENT_DENSE_NO_CONTACT:
	case eCOMPONENT_MANY_SMALL_NO_CONTACT:
		return passed &&
			metrics.softBodies ==
				(manySmallNoContactCase ? 16u : 1u) &&
			metrics.rigidBoxes == 0 &&
			metrics.groundContactFrames == 0 &&
			metrics.rigidContactFrames == 0 &&
			metrics.softContactFrames == 0 &&
			metrics.maxCentroidDrop <= 1.0e-5f &&
			metrics.maxParticleSpeed <= 1.0e-4f;
	case eCOMPONENT_GROUND:
		return passed && metrics.softBodies == 1 &&
			metrics.rigidBoxes == 0 &&
			metrics.groundContactFrames > 0 &&
			metrics.maxGroundContacts > 0 &&
			metrics.rigidContactFrames == 0 &&
			metrics.softContactFrames == 0 &&
			metrics.maxCentroidDrop > 1.0f;
	case eCOMPONENT_STATIC_BOX:
		return passed && metrics.softBodies == 1 &&
			metrics.rigidBoxes == 1 &&
			metrics.rigidContactFrames > 0 &&
			metrics.maxRigidContacts > 0 &&
			metrics.softContactFrames == 0 &&
			metrics.maxCentroidDrop > 1.0f && metrics.finalMinY > 0.70f;
	case eCOMPONENT_SOFT_SOFT:
		return passed && metrics.softBodies == 2 &&
			metrics.rigidBoxes == 0 &&
			metrics.softContactFrames > 0 &&
			metrics.maxSoftContacts > 0 &&
			metrics.finalInsideParticles == 0 &&
			metrics.maxCentroidDrop > 1.0f;
	case eCOMPONENT_CONE_GROUND:
		return passed && metrics.softBodies == 1 &&
			metrics.rigidBoxes == 0 &&
			metrics.groundContactFrames > 0 &&
			metrics.maxGroundContacts > 0 &&
			metrics.rigidContactFrames == 0 &&
			metrics.softContactFrames == 0 &&
			metrics.maxCentroidDrop > 5.0f;
	case eCOMPONENT_CURRENT_ALL:
		return passed && metrics.softBodies == 5 &&
			metrics.rigidBoxes == 1 &&
			metrics.groundContactFrames > 0 &&
			metrics.rigidContactFrames > 0 &&
			metrics.softContactFrames > 0 &&
			metrics.maxCentroidDrop > 1.0f;
	}

	return false;
}

bool isSoftSleepWakeResultValid(const DeformableVolumeMetrics& metrics)
{
	return metrics.sceneStatics == 0 &&
		metrics.sceneSoftInitiallyAwake == 1 &&
		metrics.sceneSoftFirstSlept == 1 &&
		metrics.sceneSoftFirstSleepFrame < metrics.completedFrames &&
		metrics.sceneSoftSleepWakeCounterZero == 1 &&
		metrics.sceneSoftSleepVelocitiesZero == 1 &&
		metrics.sceneSoftStableWhileSleeping == 1 &&
		metrics.sceneSoftCounterWakeIssued == 1 &&
		metrics.sceneSoftWokeByCounter == 1 &&
		metrics.sceneSoftCounterWakeFrame >
			metrics.sceneSoftFirstSleepFrame &&
		metrics.sceneSoftSecondSlept == 1 &&
		metrics.sceneSoftSecondSleepFrame >
			metrics.sceneSoftCounterWakeFrame &&
		metrics.sceneSoftVelocityWakeIssued == 1 &&
		metrics.sceneSoftWokeByVelocity == 1 &&
		metrics.sceneSoftVelocityWakeFrame >
			metrics.sceneSoftSecondSleepFrame &&
		metrics.sceneSoftMovedAfterVelocityWake == 1 &&
		metrics.sceneSoftVelocityStopIssued == 1 &&
		metrics.sceneSoftFinalSlept == 1 &&
		metrics.sceneSoftFinalSleepFrame >
			metrics.sceneSoftVelocityWakeFrame &&
		metrics.maxParticleSpeed < 3.0f &&
		metrics.finalMaxParticleSpeed < 1.0e-6f;
}

bool isSoftRigidWakeResultValid(const DeformableVolumeMetrics& metrics)
{
	return metrics.sceneStatics == 0 &&
		metrics.sceneSoftInitiallyAwake == 1 &&
		metrics.sceneSoftFirstSlept == 1 &&
		metrics.sceneSoftFirstSleepFrame < metrics.completedFrames &&
		metrics.sceneSoftSleepWakeCounterZero == 1 &&
		metrics.sceneSoftSleepVelocitiesZero == 1 &&
		metrics.sceneSoftStableWhileSleeping == 1 &&
		metrics.sceneSoftRigidWakeActorAdded == 1 &&
		metrics.sceneDynamicActorAdded == 1 &&
		metrics.sceneDynamicInitiallySleeping == 0 &&
		metrics.sceneSoftWokeByRigid == 1 &&
		metrics.sceneSoftRigidWakeFrame >
			metrics.sceneSoftFirstSleepFrame &&
		metrics.sceneSoftVelocityStopIssued == 1 &&
		metrics.sceneSoftFinalSlept == 1 &&
		metrics.sceneSoftFinalSleepFrame >
			metrics.sceneSoftRigidWakeFrame &&
		metrics.rigidContactFrames > 0 && metrics.maxRigidContacts > 0 &&
		metrics.finalMinY > 3.5f && metrics.maxParticleSpeed < 2.0f &&
		metrics.finalMaxParticleSpeed < 1.0e-6f &&
		metrics.sceneDynamicActorRemoved == 1 &&
		metrics.sceneDynamicActorReleased == 1;
}

bool isBufferMutationResultValid(const DeformableVolumeMetrics& metrics)
{
	return metrics.sceneStatics == 0 && metrics.sceneDynamics == 0 &&
		metrics.sceneSoftInitiallyAwake == 1 &&
		metrics.sceneSoftFirstSlept == 1 &&
		metrics.sceneSoftFirstSleepFrame < metrics.completedFrames &&
		metrics.sceneSoftSleepWakeCounterZero == 1 &&
		metrics.sceneSoftSleepVelocitiesZero == 1 &&
		metrics.sceneSoftStableWhileSleeping == 1 &&
		metrics.sceneBufferMutationIssued == 1 &&
		metrics.sceneBufferMutationWoke == 1 &&
		metrics.sceneBufferMutationApplied == 1 &&
		metrics.sceneBufferDriveIssued == 1 &&
		metrics.sceneBufferPinHeld == 1 &&
		metrics.sceneBufferDynamicMoved == 1 &&
		metrics.sceneBufferInvMassRestored == 1 &&
		metrics.sceneBufferRestoredMoved == 1 &&
		metrics.sceneBufferResetIssued == 1 &&
		metrics.sceneSoftFinalSlept == 1 &&
		metrics.sceneSoftFinalSleepFrame >
			metrics.sceneSoftFirstSleepFrame &&
		metrics.sceneSoftFinalSleepFrame < metrics.completedFrames &&
		metrics.finalMinY > 4.0f && metrics.maxParticleSpeed < 3.0f &&
		metrics.finalMaxParticleSpeed < 1.0e-6f;
}

bool isWorldPinResultValid(const DeformableVolumeMetrics& metrics)
{
	return metrics.sceneStatics == 0 && metrics.sceneDynamics == 0 &&
		metrics.sceneWorldPinCreated == 1 &&
		metrics.sceneWorldPinHeld == 1 &&
		metrics.sceneWorldPinActorReadded == 1 &&
		metrics.sceneWorldPinReleased == 1 &&
		metrics.sceneWorldPinMovedAfterRelease == 1 &&
		metrics.sceneWorldPinMaxDrift <= 1.0e-4f &&
		metrics.sceneWorldPinReleasedMaxDisplacement > 1.0e-3f;
}

bool isMixedSleepIslandResultValid(const DeformableVolumeMetrics& metrics)
{
	return metrics.sceneStatics == 0 && metrics.sceneDynamics == 0 &&
		metrics.sceneSecondVolumeActorCreated == 1 &&
		metrics.sceneSecondVolumeHostBuffersInitialized == 1 &&
		metrics.sceneSecondVolumeActorAdded == 1 &&
		metrics.sceneSecondVolumeActorRemoved == 1 &&
		metrics.sceneSecondVolumeActorReleased == 1 &&
		metrics.sceneSecondVolumeBoundsFinite == 1 &&
		metrics.sceneMixedFirstSlept == 1 &&
		metrics.sceneMixedFirstSleepFrame < metrics.completedFrames &&
		metrics.sceneMixedFirstStable == 1 &&
		metrics.sceneMixedSecondStayedAwake == 1 &&
		metrics.sceneMixedSecondMoved == 1 && metrics.finalMinY > 3.5f &&
		metrics.finalMaxParticleSpeed < 0.3f;
}

bool isSoftChurnResultValid(const DeformableVolumeMetrics& metrics)
{
	return metrics.sceneStatics == 0 && metrics.sceneDynamics == 0 &&
		metrics.sceneSecondVolumeActorCreated == 1 &&
		metrics.sceneSecondVolumeHostBuffersInitialized == 1 &&
		metrics.sceneSecondVolumeActorAdded == 1 &&
		metrics.sceneSecondVolumeActorRemoved == 1 &&
		metrics.sceneSecondVolumeActorReleased == 1 &&
		metrics.sceneSecondVolumeBoundsFinite == 1 &&
		metrics.sceneSoftChurnCycles > 0 &&
		metrics.sceneSoftChurnRemoveCount ==
			2 * metrics.sceneSoftChurnCycles &&
		metrics.sceneSoftChurnReaddCount ==
			2 * metrics.sceneSoftChurnCycles &&
		metrics.sceneSoftChurnPostCompactMoveCount ==
			2 * metrics.sceneSoftChurnCycles &&
		metrics.sceneSoftChurnStable == 1 && metrics.finalMinY > 3.5f &&
		metrics.finalMaxParticleSpeed < 1.0e-4f;
}

bool isVolumeKinematicTargetResultValid(
	const DeformableVolumeMetrics& metrics,
	bool fullTarget)
{
	const bool commonTargetPassed =
		metrics.sceneStatics == 0 && metrics.sceneDynamics == 0 &&
		metrics.sceneVolumeTargetBound == 1 &&
		metrics.sceneVolumeTargetMutated == 1 &&
		metrics.sceneVolumeTargetWoke == 1 &&
		metrics.sceneVolumeTargetReached == 1 &&
		PxIsFinite(metrics.sceneVolumeTargetFinalMaxError) &&
		metrics.sceneVolumeTargetFinalMaxError <= 5.0e-3f &&
		metrics.sceneVolumeTargetMaxDisplacement > 0.2f &&
		metrics.maxParticleSpeed < 5.0f &&
		metrics.finalMaxParticleSpeed < 0.5f;
	if(!commonTargetPassed || fullTarget)
		return commonTargetPassed;
	return metrics.sceneVolumePartialInactiveIgnored == 1 &&
		metrics.sceneVolumePartialActivated == 1 &&
		metrics.sceneVolumePartialActivatedReached == 1 &&
		PxIsFinite(metrics.sceneVolumePartialInactiveDecoyDistance) &&
		metrics.sceneVolumePartialInactiveDecoyDistance > 2.0f;
}

bool isMultiSceneIsolationResultValid(
	const DeformableVolumeMetrics& metrics)
{
	return metrics.sceneStatics == 0 && metrics.sceneDynamics == 0 &&
		metrics.sceneSecondSceneCreated == 1 &&
		metrics.sceneSecondSceneSolverMatched == 1 &&
		metrics.scenePrimarySceneReleased == 1 &&
		metrics.sceneSecondSceneReleased == 1 &&
		metrics.sceneSecondVolumeActorCreated == 1 &&
		metrics.sceneSecondVolumeHostBuffersInitialized == 1 &&
		metrics.sceneSecondVolumeActorAdded == 1 &&
		metrics.sceneSecondVolumeActorRemoved == 1 &&
		metrics.sceneSecondVolumeActorReleased == 1 &&
		metrics.sceneSecondVolumeBoundsFinite == 1 &&
		metrics.sceneSoftFirstSlept == 1 &&
		metrics.sceneSoftFirstSleepFrame < 60 &&
		metrics.sceneMultiPrimaryStable == 1 &&
		metrics.sceneMultiPrimaryDetachedStable == 1 &&
		metrics.sceneMultiSecondaryUpdatedBeforeRelease == 1 &&
		metrics.sceneMultiSecondaryUpdatedAfterRelease == 1 &&
		metrics.sceneSecondVolumeMaxCentroidDrop > 0.1f &&
		metrics.finalMinY > 3.5f && metrics.maxParticleSpeed < 0.3f &&
		metrics.finalMaxParticleSpeed < 1.0e-6f;
}

bool isSoftSoftWakeResultValid(const DeformableVolumeMetrics& metrics)
{
	return metrics.sceneStatics == 0 && metrics.sceneDynamics == 0 &&
		metrics.sceneSecondVolumeActorCreated == 1 &&
		metrics.sceneSecondVolumeHostBuffersInitialized == 1 &&
		metrics.sceneSecondVolumeActorAdded == 1 &&
		metrics.sceneSecondVolumeActorRemoved == 1 &&
		metrics.sceneSecondVolumeActorReleased == 1 &&
		metrics.sceneSecondVolumeBoundsFinite == 1 &&
		metrics.sceneSoftFirstSlept == 1 &&
		metrics.sceneSoftSoftBothSlept == 1 &&
		metrics.sceneSoftSoftDriveIssued == 1 &&
		metrics.sceneSoftSoftDriverWoke == 1 &&
		metrics.sceneSoftSoftTargetWoke == 1 &&
		metrics.sceneSoftSoftTargetWakeFrame >
			metrics.sceneSoftFirstSleepFrame &&
		metrics.sceneSoftSoftTargetWakeFrame < metrics.completedFrames &&
		metrics.sceneSoftSoftTargetMoved == 1 &&
		metrics.sceneSoftSoftResetIssued == 1 &&
		metrics.sceneSoftSoftBothFinalSlept == 1 &&
		metrics.finalMinY > 3.5f && metrics.maxParticleSpeed < 10.0f &&
		metrics.finalMaxParticleSpeed < 1.0e-6f;
}

bool isSoftPairAttachmentResultValid(
	const DeformableVolumeMetrics& metrics)
{
	return metrics.sceneStatics == 0 && metrics.sceneDynamics == 0 &&
		metrics.sceneSecondVolumeActorCreated == 1 &&
		metrics.sceneSecondVolumeHostBuffersInitialized == 1 &&
		metrics.sceneSecondVolumeActorAdded == 1 &&
		metrics.sceneSecondVolumeActorRemoved == 1 &&
		metrics.sceneSecondVolumeActorReleased == 1 &&
		metrics.sceneSecondVolumeBoundsFinite == 1 &&
		metrics.sceneRigidAttachmentActorAdded == 1 &&
		metrics.sceneRigidAttachmentCreated == 1 &&
		metrics.sceneRigidAttachmentRigidWoke == 1 &&
		metrics.sceneRigidAttachmentRigidMoved == 1 &&
		metrics.sceneRigidAttachmentHeldAcrossReadd == 1 &&
		metrics.sceneRigidAttachmentReleased == 1 &&
		metrics.sceneRigidAttachmentSeparatedAfterRelease == 1 &&
		metrics.sceneRigidAttachmentMaxDrift < 0.05f &&
		metrics.sceneRigidAttachmentMaxRigidSpeed < 10.0f &&
		metrics.sceneRigidAttachmentMaxRigidDisplacement > 0.02f &&
		metrics.sceneRigidAttachmentReleasedSeparation > 0.2f &&
		metrics.maxParticleSpeed < 10.0f &&
		metrics.finalMaxParticleSpeed < 2.0f;
}

bool isKinematicRigidResultValid(
	const DeformableVolumeMetrics& metrics)
{
	return metrics.sceneStatics == 0 &&
		metrics.sceneKinematicActorAdded == 1 &&
		metrics.sceneSoftFirstSlept == 1 &&
		metrics.sceneKinematicTargetIssued == 1 &&
		metrics.sceneKinematicTargetReached == 1 &&
		metrics.sceneKinematicSoftWoke == 1 &&
		metrics.sceneKinematicSoftMoved == 1 &&
		metrics.sceneKinematicContactObserved == 1 &&
		metrics.sceneKinematicMaxPoseError <= 1.0e-4f &&
		metrics.sceneKinematicSoftDisplacement > 0.02f &&
		PxIsFinite(metrics.sceneKinematicFinalY) &&
		PxAbs(metrics.sceneKinematicFinalY - 4.10f) <= 1.0e-4f &&
		metrics.maxParticleSpeed < 2.0f &&
		metrics.finalMaxParticleSpeed < 0.5f &&
		metrics.sceneDynamicActorRemoved == 1 &&
		metrics.sceneDynamicActorReleased == 1;
}

DynamicSceneValidationConfig::DynamicSceneValidationConfig()
	: initialY(0.0f), smooth(false), capsule(false), convex(false),
	  twoActors(false), multiSoftIsland(false), churn(false)
{
}

bool isDynamicSceneResultValid(
	const DeformableVolumeMetrics& metrics,
	const DynamicSceneValidationConfig& config)
{
	const bool dynamicPassed =
		metrics.sceneStatics == (config.smooth ? 1u : 0u) &&
		metrics.rigidContactFrames > 0 && metrics.maxRigidContacts > 0 &&
		metrics.maxCentroidDrop > 0.5f &&
		metrics.sceneDynamicActorRemoved == 1 &&
		metrics.sceneDynamicActorReleased == 1 &&
		metrics.sceneDynamicInitiallySleeping == 1 &&
		metrics.sceneDynamicWokeBySoft == 1 &&
		metrics.sceneDynamicFirstWakeFrame != PX_MAX_U32 &&
		PxIsFinite(metrics.sceneDynamicMinY) &&
		PxIsFinite(metrics.sceneDynamicFinalY) &&
		(config.convex
			? (metrics.sceneDynamicMaxDrop > 5.0e-4f &&
			   config.initialY - metrics.sceneDynamicFinalY > 5.0e-4f)
			: (metrics.sceneDynamicMaxDrop > 0.05f &&
			   config.initialY - metrics.sceneDynamicFinalY > 0.05f)) &&
		(config.capsule || config.convex ||
		 metrics.sceneDynamicPreContactMaxDrop < 1.0e-4f) &&
		metrics.sceneDynamicMaxDownSpeed >
			(config.convex ? 1.0e-4f : 0.01f) &&
		PxIsFinite(metrics.minDynamicSurfaceSeparation) &&
		metrics.minDynamicSurfaceSeparation > -0.15f &&
		PxIsFinite(metrics.finalDynamicSurfaceSeparation) &&
		metrics.finalDynamicSurfaceSeparation > -0.15f &&
		(!config.smooth ||
		 (config.convex
			? (metrics.finalMinY > -0.15f &&
			   metrics.maxParticleSpeed < 10.0f &&
			   metrics.finalMaxParticleSpeed < 0.75f &&
			   metrics.sceneDynamicMaxDownSpeed < 5.0f &&
			   metrics.sceneDynamicFinalY > 0.90f &&
			   metrics.sceneDynamicFinalY < 1.10f)
			: (metrics.finalMinY > -0.15f &&
			   metrics.maxParticleSpeed < 10.0f &&
			   metrics.finalMaxParticleSpeed < 0.5f &&
			   metrics.sceneDynamicMaxDownSpeed < 5.0f &&
			   metrics.sceneDynamicFinalY > 0.70f &&
			   metrics.sceneDynamicFinalY < 0.90f)));
	if(!dynamicPassed)
		return false;

	if(config.twoActors)
	{
		const bool secondDynamicPassed =
			metrics.sceneSecondDynamicActorAdded == 1 &&
			metrics.sceneSecondDynamicActorRemoved == 1 &&
			metrics.sceneSecondDynamicActorReleased == 1 &&
			metrics.sceneSecondDynamicInitiallySleeping == 1 &&
			metrics.sceneSecondDynamicWokeBySoft == 1 &&
			metrics.sceneSecondDynamicFirstWakeFrame != PX_MAX_U32 &&
			metrics.sceneSecondDynamicFirstWakeFrame ==
				metrics.sceneDynamicFirstWakeFrame &&
			PxIsFinite(metrics.sceneSecondDynamicMinY) &&
			PxIsFinite(metrics.sceneSecondDynamicFinalY) &&
			metrics.sceneSecondDynamicMaxDrop > 0.01f &&
			metrics.sceneSecondDynamicPreContactMaxDrop < 1.0e-4f &&
			metrics.sceneSecondDynamicMaxDownSpeed > 0.01f &&
			metrics.sceneDynamicMaxDownSpeed < 6.0f &&
			metrics.sceneSecondDynamicMaxDownSpeed < 6.0f;
		if(!secondDynamicPassed)
			return false;
		if(!config.multiSoftIsland)
			return true;
		return metrics.sceneSecondVolumeActorCreated == 1 &&
			metrics.sceneSecondVolumeHostBuffersInitialized == 1 &&
			metrics.sceneSecondVolumeActorAdded == 1 &&
			metrics.sceneSecondVolumeActorRemoved == 1 &&
			metrics.sceneSecondVolumeActorReleased == 1 &&
			metrics.sceneSecondVolumeBoundsFinite == 1 &&
			metrics.sceneSecondVolumeMaxCentroidDrop > 0.5f &&
			PxIsFinite(metrics.sceneSecondVolumeFinalCentroidY) &&
			metrics.finalMinY > -10.0f &&
			metrics.finalMaxParticleSpeed < 1.0e-4f;
	}

	if(!config.churn)
		return true;
	return metrics.sceneDynamicShapeDetached == 1 &&
		metrics.sceneDynamicShapeReattached == 1 &&
		metrics.sceneDynamicActorRemoved == 1 &&
		metrics.sceneDynamicActorReadded == 1 &&
		metrics.sceneDynamicReaddedSleeping == 1 &&
		metrics.sceneDynamicRewokeBySoft == 1 &&
		metrics.sceneDynamicSecondWakeFrame >
			metrics.sceneDynamicFirstWakeFrame &&
		metrics.sceneDynamicSecondWakeFrame < metrics.completedFrames;
}

bool isNoStaticSceneResultValid(const DeformableVolumeMetrics& metrics)
{
	return metrics.sceneStatics == 0;
}

bool isGroundSceneResultValid(const DeformableVolumeMetrics& metrics)
{
	return metrics.sceneStatics == 1 && metrics.groundContactFrames > 0 &&
		metrics.maxGroundContacts > 0 && metrics.maxCentroidDrop > 2.0f &&
		metrics.finalMinY > -0.1f;
}

bool isStaticBoxSceneResultValid(
	const DeformableVolumeMetrics& metrics, bool churn)
{
	const bool staticBoxPassed = metrics.sceneStatics == 1 &&
		metrics.rigidContactFrames > 0 && metrics.maxRigidContacts > 0 &&
		metrics.maxCentroidDrop > 2.0f && metrics.finalMinY > 0.7f;
	if(!staticBoxPassed || !churn)
		return staticBoxPassed;

	return metrics.sceneStaticShapeDetached == 1 &&
		metrics.sceneStaticShapeReattached == 1 &&
		metrics.sceneStaticActorRemoved == 1 &&
		metrics.sceneStaticActorReadded == 1;
}

TaskGraphValidationMetrics::TaskGraphValidationMetrics()
	: profiledFrames(0), submittedSolveTasks(0), completedSolveTasks(0),
	  serialSolveTasks(0), pureSoftEligibleIslands(0),
	  pureSoftEligibleParticles(0), submittedPredictionTasks(0),
	  completedPredictionTasks(0), peakActivePredictionTasks(0),
	  serialPredictionStages(0), submittedWriteBackTasks(0),
	  completedWriteBackTasks(0), peakActiveWriteBackTasks(0),
	  serialWriteBackStages(0), submittedCausalLayerTasks(0),
	  completedCausalLayerTasks(0), peakActiveCausalLayerTasks(0),
	  causalLayerFanIns(0), serialCausalLayerFallbacks(0),
	  maxCausalLayerOccupancy(0), submittedWorldPlaneContactTasks(0),
	  completedWorldPlaneContactTasks(0),
	  peakActiveWorldPlaneContactTasks(0), worldPlaneContactFanIns(0),
	  serialWorldPlaneContactFallbacks(0), workspaceGrowthEvents(0),
	  contactWorkspaceGrowthEvents(0), contactSweepScratchGrowthEvents(0),
	  contactOutputGrowthEvents(0)
{
}

TaskGraphValidationConfig::TaskGraphValidationConfig()
	: caseType(eTASK_GRAPH_PURE_SOFT), entryCount(0), dispatcherThreads(0),
	  parallelExecution(false), sequentialExecution(false),
	  requireSteadyStateNoGrowth(false)
{
}

bool isTaskGraphResultValid(
	const DeformableVolumeMetrics& metrics,
	const TaskGraphValidationMetrics& taskGraph,
	const TaskGraphValidationConfig& config)
{
	const PxU64 scheduledTasks =
		taskGraph.submittedSolveTasks + taskGraph.serialSolveTasks;
	const bool solveAuthorityPassed = scheduledTasks > 0 &&
		taskGraph.completedSolveTasks == taskGraph.submittedSolveTasks;

	switch(config.caseType)
	{
	case eTASK_GRAPH_PURE_SOFT:
		return metrics.sceneStatics == 0 && metrics.sceneDynamics == 0 &&
			solveAuthorityPassed &&
			taskGraph.pureSoftEligibleIslands == scheduledTasks &&
			taskGraph.pureSoftEligibleParticles ==
				scheduledTasks * metrics.particles;
	case eTASK_GRAPH_WORLD_PLANE:
		return metrics.sceneStatics == 1 && metrics.sceneDynamics == 0 &&
			solveAuthorityPassed;
	case eTASK_GRAPH_RIGID_SDF:
		return metrics.sceneStatics == 1 && metrics.sceneDynamics == 0 &&
			metrics.speculativeCcdFlagApplied == 1 &&
			solveAuthorityPassed;
	case eTASK_GRAPH_WRITE_BACK:
		break;
	case eTASK_GRAPH_PIPELINE:
		break;
	}

	const PxU64 expectedFrames = taskGraph.profiledFrames;
	const bool parallelStagesRequested = config.parallelExecution &&
		config.dispatcherThreads >= 2;
	const PxU32 expectedStageTaskCount = parallelStagesRequested
		? PxMin(config.dispatcherThreads, config.entryCount)
		: 0u;
	// Prediction, causal solve and write-back partition by body. Contact
	// detection partitions particle ranges and may use the full dispatcher.
	const PxU32 expectedContactTaskConcurrency = parallelStagesRequested
		? config.dispatcherThreads
		: 0u;
	const PxU64 expectedSubmittedSolveTasks =
		config.sequentialExecution ? 0u : expectedFrames;
	const PxU64 expectedSerialSolveTasks =
		config.sequentialExecution ? expectedFrames : 0u;
	const bool volumeHealthPassed = metrics.invertedElementSamples == 0 &&
		PxIsFinite(metrics.minDetF) && metrics.minDetF > 0.0f &&
		PxIsFinite(metrics.maxDetF) && metrics.maxDetF < 20.0f &&
		PxIsFinite(metrics.minBodyVolumeRatio) &&
		metrics.minBodyVolumeRatio > 0.01f &&
		PxIsFinite(metrics.maxBodyVolumeRatio) &&
		metrics.maxBodyVolumeRatio < 20.0f;
	const bool predictionPassed = parallelStagesRequested
		? taskGraph.submittedPredictionTasks ==
				expectedFrames * expectedStageTaskCount &&
			taskGraph.completedPredictionTasks ==
				expectedFrames * expectedStageTaskCount &&
			taskGraph.peakActivePredictionTasks > 0 &&
			taskGraph.peakActivePredictionTasks <= expectedStageTaskCount &&
			taskGraph.serialPredictionStages == 0
		: taskGraph.submittedPredictionTasks == 0 &&
			taskGraph.completedPredictionTasks == 0 &&
			taskGraph.serialPredictionStages ==
				(config.sequentialExecution ? 0u : expectedFrames);
	const bool writeBackPassed = parallelStagesRequested
		? taskGraph.submittedWriteBackTasks ==
				expectedFrames * expectedStageTaskCount &&
			taskGraph.completedWriteBackTasks ==
				expectedFrames * expectedStageTaskCount &&
			taskGraph.peakActiveWriteBackTasks > 0 &&
			taskGraph.peakActiveWriteBackTasks <= expectedStageTaskCount &&
			taskGraph.serialWriteBackStages == 0
		: taskGraph.submittedWriteBackTasks == 0 &&
			taskGraph.completedWriteBackTasks == 0 &&
			taskGraph.serialWriteBackStages ==
				(config.sequentialExecution ? 0u : expectedFrames);
	const bool causalLayerPassed = config.caseType != eTASK_GRAPH_PIPELINE ||
		(parallelStagesRequested
			? taskGraph.submittedCausalLayerTasks > 0 &&
				taskGraph.completedCausalLayerTasks ==
					taskGraph.submittedCausalLayerTasks &&
				taskGraph.peakActiveCausalLayerTasks > 0 &&
				taskGraph.peakActiveCausalLayerTasks <=
					expectedStageTaskCount &&
				taskGraph.causalLayerFanIns > 0 &&
				taskGraph.serialCausalLayerFallbacks == 0 &&
				taskGraph.maxCausalLayerOccupancy > 0
			: taskGraph.submittedCausalLayerTasks == 0 &&
				taskGraph.completedCausalLayerTasks == 0);
	const bool worldPlanePassed = config.caseType != eTASK_GRAPH_PIPELINE ||
		(parallelStagesRequested
			? taskGraph.submittedWorldPlaneContactTasks > 0 &&
				taskGraph.completedWorldPlaneContactTasks ==
					taskGraph.submittedWorldPlaneContactTasks &&
				taskGraph.peakActiveWorldPlaneContactTasks > 0 &&
				taskGraph.peakActiveWorldPlaneContactTasks <=
					expectedContactTaskConcurrency &&
				taskGraph.worldPlaneContactFanIns > 0 &&
				taskGraph.serialWorldPlaneContactFallbacks == 0
			: taskGraph.submittedWorldPlaneContactTasks == 0 &&
				taskGraph.completedWorldPlaneContactTasks == 0);
	const bool steadyStateGrowthPassed =
		!config.requireSteadyStateNoGrowth ||
		(taskGraph.workspaceGrowthEvents == 0 &&
		 taskGraph.contactWorkspaceGrowthEvents == 0 &&
		 taskGraph.contactSweepScratchGrowthEvents == 0 &&
		 taskGraph.contactOutputGrowthEvents == 0);

	return metrics.sceneStatics ==
			(config.caseType == eTASK_GRAPH_PIPELINE ? 1u : 0u) &&
		metrics.sceneDynamics == 0 &&
		metrics.particles >= config.entryCount * 1024u &&
		volumeHealthPassed && metrics.maxCentroidDrop > 0.01f &&
		metrics.maxParticleSpeed > 0.1f &&
		metrics.finalMaxParticleSpeed > 0.01f &&
		taskGraph.submittedSolveTasks == expectedSubmittedSolveTasks &&
		taskGraph.completedSolveTasks == expectedSubmittedSolveTasks &&
		taskGraph.serialSolveTasks == expectedSerialSolveTasks &&
		predictionPassed && writeBackPassed && causalLayerPassed &&
		worldPlanePassed && steadyStateGrowthPassed;
}

VolumeSkinningMetrics::VolumeSkinningMetrics()
	: initialized(0), finiteFrames(0), evaluatedFrames(0), vertices(0),
	  triangles(0), maxDisplacement(0.0f)
{
}

bool isVolumeSkinningResultValid(
	const DeformableVolumeMetrics& metrics,
	const VolumeSkinningMetrics& skinning,
	PxU32 expectedFrames)
{
	return metrics.sceneStatics == 0 && metrics.sceneDynamics == 0 &&
		skinning.initialized == 1 &&
		skinning.evaluatedFrames == expectedFrames &&
		skinning.finiteFrames == expectedFrames && skinning.vertices > 4 &&
		skinning.triangles >= 4 && PxIsFinite(skinning.maxDisplacement) &&
		skinning.maxDisplacement > 0.05f;
}

static bool isSecondVolumeLifecycleValid(
	const DeformableVolumeMetrics& metrics)
{
	return metrics.sceneSecondVolumeActorCreated == 1 &&
		metrics.sceneSecondVolumeHostBuffersInitialized == 1 &&
		metrics.sceneSecondVolumeActorAdded == 1 &&
		metrics.sceneSecondVolumeActorRemoved == 1 &&
		metrics.sceneSecondVolumeActorReleased == 1 &&
		metrics.sceneSecondVolumeBoundsFinite == 1;
}

bool isMotionControlsResultValid(
	const DeformableVolumeMetrics& metrics, PxReal dt)
{
	return metrics.sceneStatics == 0 && metrics.sceneDynamics == 0 &&
		isSecondVolumeLifecycleValid(metrics) &&
		metrics.motionMaxVelocityBounded == 1 &&
		metrics.motionSettlingApplied == 1 &&
		metrics.motionSettlingSlept == 1 &&
		metrics.motionControlStayedAwake == 1 &&
		metrics.motionMaxVelocityFirstStepDisplacement <= dt * 1.01f &&
		metrics.motionMaxVelocityFirstStepSpeed <= 1.02f &&
		metrics.motionSettlingFinalSpeed <= 1.0e-6f &&
		metrics.motionControlFinalSpeed >= 0.07f &&
		metrics.motionControlFinalSpeed <= 0.09f;
}

bool isMaxDepenetrationVelocityResultValid(
	const DeformableVolumeMetrics& metrics, PxReal dt)
{
	return metrics.sceneStatics == 1 && metrics.sceneDynamics == 0 &&
		isSecondVolumeLifecycleValid(metrics) &&
		metrics.depenetrationLimitApplied == 1 &&
		metrics.depenetrationFirstStepBounded == 1 &&
		metrics.depenetrationControlSeparated == 1 &&
		metrics.depenetrationGradualRecovery == 1 &&
		metrics.depenetrationLimitedFirstStepRise >= -1.0e-6f &&
		metrics.depenetrationLimitedFirstStepRise <= dt * 0.12f * 1.25f &&
		metrics.depenetrationLimitedMaxSpeed <= 0.25f;
}

AttachmentValidationConfig::AttachmentValidationConfig()
	: rigid(false), staticTarget(false), kinematic(false),
	  articulation(false)
{
}

bool isAttachmentResultValid(
	const DeformableVolumeMetrics& metrics,
	const AttachmentValidationConfig& config)
{
	if(config.articulation)
	{
		return metrics.sceneStatics == 0 && metrics.sceneDynamics == 0 &&
			metrics.sceneArticulationCreated == 1 &&
			metrics.sceneArticulationAdded == 1 &&
			metrics.sceneArticulationInitiallySleeping == 1 &&
			metrics.sceneArticulationWoke == 1 &&
			metrics.sceneArticulationJointSubspaceHeld == 1 &&
			metrics.sceneArticulationRootStable == 1 &&
			metrics.sceneRigidAttachmentActorAdded == 1 &&
			metrics.sceneRigidAttachmentInitiallySleeping == 1 &&
			metrics.sceneRigidAttachmentCreated == 1 &&
			metrics.sceneRigidAttachmentRigidMoved == 1 &&
			metrics.sceneRigidAttachmentHeldAcrossReadd == 1 &&
			metrics.sceneRigidAttachmentReleased == 1 &&
			metrics.sceneRigidAttachmentSeparatedAfterRelease == 1 &&
			metrics.sceneRigidAttachmentMaxDrift < 0.05f &&
			metrics.sceneRigidAttachmentMaxRigidSpeed < 5.0f &&
			metrics.sceneRigidAttachmentMaxRigidDisplacement > 0.02f &&
			metrics.sceneRigidAttachmentReleasedSeparation > 0.2f &&
			metrics.sceneArticulationRootMaxDisplacement <= 1.0e-4f &&
			metrics.sceneArticulationChildMaxForbiddenDisplacement <=
				1.0e-3f &&
			metrics.sceneArticulationChildMaxAngularDisplacement <=
				1.0e-3f &&
			metrics.maxParticleSpeed < 20.0f &&
			metrics.finalMaxParticleSpeed < 2.0f;
	}

	return metrics.sceneStatics == (config.staticTarget ? 1u : 0u) &&
		metrics.sceneDynamics == (config.staticTarget ? 0u : 1u) &&
		metrics.sceneRigidAttachmentActorAdded == 1 &&
		metrics.sceneRigidAttachmentInitiallySleeping == 1 &&
		metrics.sceneRigidAttachmentCreated == 1 &&
		(!config.rigid || metrics.sceneRigidAttachmentRigidWoke == 1) &&
		metrics.sceneRigidAttachmentRigidMoved == 1 &&
		metrics.sceneRigidAttachmentHeldAcrossReadd == 1 &&
		metrics.sceneRigidAttachmentReleased == 1 &&
		metrics.sceneRigidAttachmentSeparatedAfterRelease == 1 &&
		metrics.sceneRigidAttachmentMaxDrift < 0.05f &&
		metrics.sceneRigidAttachmentMaxRigidSpeed < 5.0f &&
		metrics.sceneRigidAttachmentMaxRigidDisplacement > 0.02f &&
		(!(config.kinematic || config.staticTarget) ||
		 (metrics.sceneKinematicActorAdded == 1 &&
		  metrics.sceneSoftFirstSlept == 1 &&
		  metrics.sceneKinematicTargetIssued == 1 &&
		  metrics.sceneKinematicTargetReached == 1 &&
		  metrics.sceneKinematicSoftWoke == 1 &&
		  metrics.sceneKinematicSoftMoved == 1 &&
		  metrics.sceneKinematicMaxPoseError <= 1.0e-4f &&
		  metrics.sceneKinematicSoftDisplacement > 0.02f &&
		  metrics.maxParticleSpeed < 20.0f &&
		  metrics.finalMaxParticleSpeed < 2.0f)) &&
		metrics.sceneRigidAttachmentReleasedSeparation > 0.2f &&
		(config.staticTarget ||
		 (metrics.sceneDynamicActorRemoved == 1 &&
		  metrics.sceneDynamicActorReleased == 1));
}

ElementFilterValidationConfig::ElementFilterValidationConfig()
	: partial(false), surfaceTolerance(0.0f),
	  minSuppressedDepth(0.0f), contactOffsetLimit(0.0f)
{
}

bool finalizeAndValidateElementFilterResult(
	DeformableVolumeMetrics& metrics,
	const ElementFilterValidationConfig& config)
{
	metrics.sceneElementFilterContactRestored =
		metrics.sceneElementFilterReleased == 1 &&
		PxIsFinite(metrics.sceneElementFilterFinalMinY) &&
		metrics.sceneElementFilterFinalMinY >= -config.surfaceTolerance &&
		metrics.sceneElementFilterFinalMinY <= config.contactOffsetLimit &&
		metrics.finalMaxParticleSpeed < 0.1f ? 1u : 0u;
	metrics.scenePartialFilterExactOwnership =
		config.partial && metrics.sceneElementFilterSuppressedContact == 1 &&
		metrics.scenePartialFilterUnfilteredContactHeld == 1 &&
		PxIsFinite(metrics.scenePartialFilterUnfilteredMinY) &&
		metrics.scenePartialFilterUnfilteredMinY >= -config.surfaceTolerance
			? 1u : 0u;
	return metrics.sceneStatics == 1 && metrics.sceneDynamics == 0 &&
		metrics.sceneElementFilterCreated == 1 &&
		metrics.sceneElementFilterActorReadded == 1 &&
		metrics.sceneElementFilterSuppressedContact == 1 &&
		metrics.sceneElementFilterReleased == 1 &&
		metrics.sceneElementFilterContactRestored == 1 &&
		metrics.sceneElementFilterMinY <= -config.minSuppressedDepth &&
		metrics.sceneElementFilterFinalMinY >= -config.surfaceTolerance &&
		metrics.sceneElementFilterFinalMinY <= config.contactOffsetLimit &&
		(!config.partial ||
		 (metrics.scenePartialFilterExactOwnership == 1 &&
		  metrics.scenePartialFilterUnfilteredMinY >=
			  -config.surfaceTolerance));
}

SurfaceGapMetrics::SurfaceGapMetrics()
	: initialSignedSdf(PX_MAX_F32), minSignedSdf(PX_MAX_F32),
	  finalSignedSdf(PX_MAX_F32), penetrationFrames(0), penetrated(false)
{
}

bool measureVolumeAngularState(
	PxDeformableVolume& volume,
	PxVec3& centroid,
	PxVec3& angularMomentum,
	PxVec3& angularVelocity)
{
	const PxTetrahedronMesh* const simulationMesh =
		volume.getSimulationMesh();
	const PxVec4* const positions = volume.getSimPositionInvMassBufferH();
	const PxVec4* const velocities = volume.getSimVelocityBufferH();
	if(!simulationMesh || !positions || !velocities)
		return false;

	PxReal mass = 0.0f;
	PxVec3 linearMomentum(0.0f);
	centroid = PxVec3(0.0f);
	for(PxU32 vertexIndex = 0;
		vertexIndex < simulationMesh->getNbVertices(); ++vertexIndex)
	{
		const PxReal invMass = positions[vertexIndex].w;
		if(invMass <= 0.0f || !PxIsFinite(invMass))
			continue;
		const PxReal vertexMass = 1.0f / invMass;
		const PxVec3 position = positions[vertexIndex].getXYZ();
		const PxVec3 velocity = velocities[vertexIndex].getXYZ();
		if(!position.isFinite() || !velocity.isFinite())
			return false;
		centroid += position * vertexMass;
		linearMomentum += velocity * vertexMass;
		mass += vertexMass;
	}
	if(!PxIsFinite(mass) || mass <= 0.0f)
		return false;
	centroid *= 1.0f / mass;
	const PxVec3 linearVelocity = linearMomentum * (1.0f / mass);
	PxMat33 inertia(PxZero);
	angularMomentum = PxVec3(0.0f);
	for(PxU32 vertexIndex = 0;
		vertexIndex < simulationMesh->getNbVertices(); ++vertexIndex)
	{
		const PxReal invMass = positions[vertexIndex].w;
		if(invMass <= 0.0f || !PxIsFinite(invMass))
			continue;
		const PxReal vertexMass = 1.0f / invMass;
		const PxVec3 offset = positions[vertexIndex].getXYZ() - centroid;
		const PxVec3 relativeVelocity =
			velocities[vertexIndex].getXYZ() - linearVelocity;
		const PxMat33 outer(offset * offset.x,
			offset * offset.y, offset * offset.z);
		inertia = inertia +
			(PxMat33::createDiagonal(
				PxVec3(offset.magnitudeSquared())) - outer) * vertexMass;
		angularMomentum += offset.cross(relativeVelocity) * vertexMass;
	}
	const PxReal determinant = inertia.getDeterminant();
	if(!PxIsFinite(determinant) || PxAbs(determinant) <= 1.0e-12f)
		return false;
	angularVelocity = inertia.getInverse() * angularMomentum;
	return centroid.isFinite() && angularMomentum.isFinite() &&
		angularVelocity.isFinite();
}

VolumeBodyHealthMetrics::VolumeBodyHealthMetrics()
	: minDetF(FLT_MAX), maxDetF(-FLT_MAX),
	  minVolumeRatio(FLT_MAX), maxVolumeRatio(-FLT_MAX),
	  finalMinDetF(FLT_MAX), finalMaxDetF(-FLT_MAX),
	  finalVolumeRatio(FLT_MAX), minDetFFrame(PX_MAX_U32),
	  maxDetFFrame(PX_MAX_U32), minVolumeRatioFrame(PX_MAX_U32)
{
}

VolumeHealthSample::VolumeHealthSample()
	: nonFiniteParticleSamples(0), invertedElementSamples(0),
	  invertedBodiesMask(0), firstInversionBody(PX_MAX_U32),
	  firstInversionElement(PX_MAX_U32), minY(FLT_MAX), maxY(-FLT_MAX),
	  maxParticleSpeed(0.0f), minDetF(FLT_MAX), maxDetF(-FLT_MAX),
	  minBodyVolumeRatio(FLT_MAX), maxBodyVolumeRatio(-FLT_MAX)
{
}

VolumeHealthMonitor::TetRestState::TetRestState()
	: dmInv(PxIdentity), restVolume(0.0f)
{
}

VolumeHealthMonitor::VolumeRestState::VolumeRestState()
	: volume(NULL), totalRestVolume(0.0f)
{
}

VolumeHealthMonitor::VolumeHealthMonitor()
{
}

static bool getTetIndices(
	const PxTetrahedronMesh& mesh,
	PxU32 tetIndex,
	PxU32 indices[4])
{
	if(tetIndex >= mesh.getNbTetrahedrons() || !mesh.getTetrahedrons())
		return false;
	if(mesh.getTetrahedronMeshFlags() & PxTetrahedronMeshFlag::e16_BIT_INDICES)
	{
		const PxU16* source =
			static_cast<const PxU16*>(mesh.getTetrahedrons()) + 4 * tetIndex;
		for(PxU32 vertex = 0; vertex < 4; ++vertex)
			indices[vertex] = source[vertex];
	}
	else
	{
		const PxU32* source =
			static_cast<const PxU32*>(mesh.getTetrahedrons()) + 4 * tetIndex;
		for(PxU32 vertex = 0; vertex < 4; ++vertex)
			indices[vertex] = source[vertex];
	}
	return indices[0] < mesh.getNbVertices() &&
		indices[1] < mesh.getNbVertices() &&
		indices[2] < mesh.getNbVertices() &&
		indices[3] < mesh.getNbVertices();
}

void VolumeHealthMonitor::reset()
{
	mRestStates.reset();
}

bool VolumeHealthMonitor::empty() const
{
	return mRestStates.empty();
}

PxU32 VolumeHealthMonitor::getBodyCount() const
{
	return mRestStates.size();
}

PxDeformableVolume* VolumeHealthMonitor::getBody(PxU32 bodyIndex) const
{
	PX_ASSERT(bodyIndex < mRestStates.size());
	return mRestStates[bodyIndex].volume;
}

const VolumeBodyHealthMetrics& VolumeHealthMonitor::getBodyMetrics(
	PxU32 bodyIndex) const
{
	PX_ASSERT(bodyIndex < MAX_VOLUME_HEALTH_BODIES);
	return mBodyMetrics[bodyIndex];
}

bool VolumeHealthMonitor::initialize(
	PxDeformableVolume* const* volumes,
	PxU32 volumeCount)
{
	if(!volumes || volumeCount == 0 ||
		volumeCount > MAX_VOLUME_HEALTH_BODIES)
		return false;
	reset();
	for(PxU32 bodyIndex = 0;
		bodyIndex < MAX_VOLUME_HEALTH_BODIES; ++bodyIndex)
		mBodyMetrics[bodyIndex] = VolumeBodyHealthMetrics();
	for(PxU32 bodyIndex = 0; bodyIndex < volumeCount; ++bodyIndex)
	{
		PxDeformableVolume* volume = volumes[bodyIndex];
		const PxTetrahedronMesh* mesh =
			volume ? volume->getSimulationMesh() : NULL;
		const PxVec4* positions =
			volume ? volume->getSimPositionInvMassBufferH() : NULL;
		if(!volume || !mesh || !positions ||
			mesh->getNbTetrahedrons() == 0)
		{
			reset();
			return false;
		}
		mRestStates.resize(mRestStates.size() + 1);
		VolumeRestState& restState = mRestStates[mRestStates.size() - 1];
		restState.volume = volume;
		restState.tets.resize(mesh->getNbTetrahedrons());
		for(PxU32 tetIndex = 0;
			tetIndex < mesh->getNbTetrahedrons(); ++tetIndex)
		{
			PxU32 indices[4];
			if(!getTetIndices(*mesh, tetIndex, indices))
			{
				reset();
				return false;
			}
			const PxVec3 x0 = positions[indices[0]].getXYZ();
			const PxMat33 dm(
				positions[indices[1]].getXYZ() - x0,
				positions[indices[2]].getXYZ() - x0,
				positions[indices[3]].getXYZ() - x0);
			const PxReal determinant = dm.getDeterminant();
			if(!PxIsFinite(determinant) ||
				PxAbs(determinant) <= 1.0e-12f)
			{
				reset();
				return false;
			}
			TetRestState& tetState = restState.tets[tetIndex];
			tetState.dmInv = dm.getInverse();
			tetState.restVolume = PxAbs(determinant) / 6.0f;
			restState.totalRestVolume += tetState.restVolume;
		}
		if(!PxIsFinite(restState.totalRestVolume) ||
			restState.totalRestVolume <= 0.0f)
		{
			reset();
			return false;
		}
	}
	return mRestStates.size() == volumeCount;
}

bool VolumeHealthMonitor::sample(
	PxU32 completedFrame,
	VolumeHealthSample& sample)
{
	sample = VolumeHealthSample();
	if(mRestStates.empty() ||
		mRestStates.size() > MAX_VOLUME_HEALTH_BODIES)
		return false;
	for(PxU32 bodyIndex = 0;
		bodyIndex < mRestStates.size(); ++bodyIndex)
	{
		const VolumeRestState& restState = mRestStates[bodyIndex];
		VolumeBodyHealthMetrics& bodyMetrics = mBodyMetrics[bodyIndex];
		PxDeformableVolume* volume = restState.volume;
		const PxTetrahedronMesh* mesh =
			volume ? volume->getSimulationMesh() : NULL;
		const PxVec4* positions =
			volume ? volume->getSimPositionInvMassBufferH() : NULL;
		const PxVec4* velocities =
			volume ? volume->getSimVelocityBufferH() : NULL;
		if(!volume || !mesh || !positions || !velocities ||
			restState.tets.size() != mesh->getNbTetrahedrons())
			return false;

		for(PxU32 vertexIndex = 0;
			vertexIndex < mesh->getNbVertices(); ++vertexIndex)
		{
			const PxVec3 position = positions[vertexIndex].getXYZ();
			const PxVec3 velocity = velocities[vertexIndex].getXYZ();
			if(!position.isFinite() || !velocity.isFinite() ||
				!PxIsFinite(positions[vertexIndex].w) ||
				!PxIsFinite(velocities[vertexIndex].w))
			{
				sample.nonFiniteParticleSamples++;
				continue;
			}
			sample.minY = PxMin(sample.minY, position.y);
			sample.maxY = PxMax(sample.maxY, position.y);
			sample.maxParticleSpeed = PxMax(
				sample.maxParticleSpeed, velocity.magnitude());
		}

		PxReal currentVolume = 0.0f;
		PxReal frameMinDetF = FLT_MAX;
		PxReal frameMaxDetF = -FLT_MAX;
		for(PxU32 tetIndex = 0;
			tetIndex < mesh->getNbTetrahedrons(); ++tetIndex)
		{
			PxU32 indices[4];
			if(!getTetIndices(*mesh, tetIndex, indices))
				return false;
			const PxVec3 x0 = positions[indices[0]].getXYZ();
			const PxMat33 ds(
				positions[indices[1]].getXYZ() - x0,
				positions[indices[2]].getXYZ() - x0,
				positions[indices[3]].getXYZ() - x0);
			const TetRestState& tetState = restState.tets[tetIndex];
			const PxReal detF = (ds * tetState.dmInv).getDeterminant();
			if(!PxIsFinite(detF) || detF <= 0.0f)
			{
				sample.invertedElementSamples++;
				if(sample.firstInversionBody == PX_MAX_U32)
				{
					sample.firstInversionBody = bodyIndex;
					sample.firstInversionElement = tetIndex;
				}
				sample.invertedBodiesMask |= 1u << bodyIndex;
			}
			if(!PxIsFinite(detF))
				continue;
			sample.minDetF = PxMin(sample.minDetF, detF);
			sample.maxDetF = PxMax(sample.maxDetF, detF);
			frameMinDetF = PxMin(frameMinDetF, detF);
			frameMaxDetF = PxMax(frameMaxDetF, detF);
			currentVolume += detF * tetState.restVolume;
		}
		bodyMetrics.finalMinDetF = frameMinDetF;
		bodyMetrics.finalMaxDetF = frameMaxDetF;
		if(frameMinDetF < bodyMetrics.minDetF)
		{
			bodyMetrics.minDetF = frameMinDetF;
			bodyMetrics.minDetFFrame = completedFrame;
		}
		if(frameMaxDetF > bodyMetrics.maxDetF)
		{
			bodyMetrics.maxDetF = frameMaxDetF;
			bodyMetrics.maxDetFFrame = completedFrame;
		}
		const PxReal volumeRatio =
			currentVolume / restState.totalRestVolume;
		if(!PxIsFinite(volumeRatio))
			sample.nonFiniteParticleSamples++;
		else
		{
			bodyMetrics.finalVolumeRatio = volumeRatio;
			if(volumeRatio < bodyMetrics.minVolumeRatio)
			{
				bodyMetrics.minVolumeRatio = volumeRatio;
				bodyMetrics.minVolumeRatioFrame = completedFrame;
			}
			bodyMetrics.maxVolumeRatio = PxMax(
				bodyMetrics.maxVolumeRatio, volumeRatio);
			sample.minBodyVolumeRatio = PxMin(
				sample.minBodyVolumeRatio, volumeRatio);
			sample.maxBodyVolumeRatio = PxMax(
				sample.maxBodyVolumeRatio, volumeRatio);
		}
	}
	return true;
}

bool isVolumeHealthWithinLimits(
	PxU32 nonFiniteParticleSamples, PxU32 invertedElementSamples,
	PxReal minDetF, PxReal maxDetF, PxReal minBodyVolumeRatio,
	PxReal maxBodyVolumeRatio, PxReal requiredMinDetF,
	PxReal requiredMaxDetF, PxReal requiredMinVolumeRatio,
	PxReal requiredMaxVolumeRatio)
{
	return nonFiniteParticleSamples == 0 && invertedElementSamples == 0 &&
		PxIsFinite(minDetF) && minDetF > requiredMinDetF &&
		PxIsFinite(maxDetF) && maxDetF < requiredMaxDetF &&
		PxIsFinite(minBodyVolumeRatio) &&
		minBodyVolumeRatio > requiredMinVolumeRatio &&
		PxIsFinite(maxBodyVolumeRatio) &&
		maxBodyVolumeRatio < requiredMaxVolumeRatio;
}

OgcSandwichMetrics::OgcSandwichMetrics()
	: generatedRigidContacts(0), nativeIslandSteps(0),
	  initialUpperBottomOffset(PX_MAX_F32),
	  initialLowerTopOffset(PX_MAX_F32), maxUpperCompression(0.0f),
	  maxLowerCompression(0.0f), upperJawMinDetF(PX_MAX_F32),
	  lowerJawMinDetF(PX_MAX_F32), minUpperJawSignedSdf(PX_MAX_F32),
	  minLowerJawSignedSdf(PX_MAX_F32),
	  finalUpperJawSignedSdf(PX_MAX_F32),
	  finalLowerJawSignedSdf(PX_MAX_F32),
	  upperJawTrianglePenetrationFrames(0),
	  lowerJawTrianglePenetrationFrames(0),
	  upperJawTrianglePenetrated(false),
	  lowerJawTrianglePenetrated(false), initialBoxPosition(0.0f),
	  maxBoxLateralOffset(0.0f), maxBoxLateralSpeed(0.0f),
	  maxBoxNormalOffset(0.0f), maxBoxNormalSpeed(0.0f),
	  collisionTelemetryEnabled(false), initialized(false)
{
}

OgcSandwichFrameSample::OgcSandwichFrameSample()
	: generatedRigidContacts(0), boxPosition(0.0f), boxVelocity(0.0f)
{
}

bool isOgcSandwichSurfaceSeparated(const OgcSandwichMetrics& metrics)
{
	return metrics.initialized && metrics.collisionTelemetryEnabled &&
		PxIsFinite(metrics.minUpperJawSignedSdf) &&
		PxIsFinite(metrics.minLowerJawSignedSdf) &&
		!metrics.upperJawTrianglePenetrated &&
		!metrics.lowerJawTrianglePenetrated &&
		metrics.minUpperJawSignedSdf >= -MAX_SURFACE_PENETRATION &&
		metrics.minLowerJawSignedSdf >= -MAX_SURFACE_PENETRATION;
}

bool isOgcSandwichCompressed(
	const OgcSandwichMetrics& metrics, PxReal minJawCompression)
{
	return metrics.maxUpperCompression >= minJawCompression &&
		metrics.maxLowerCompression >= minJawCompression;
}

bool isOgcSandwichContained(
	const OgcSandwichMetrics& metrics, PxReal maxLateralOffset,
	PxReal maxLateralSpeed, PxReal maxNormalOffset, PxReal maxNormalSpeed)
{
	return metrics.maxBoxLateralOffset <= maxLateralOffset &&
		metrics.maxBoxLateralSpeed <= maxLateralSpeed &&
		metrics.maxBoxNormalOffset <= maxNormalOffset &&
		metrics.maxBoxNormalSpeed <= maxNormalSpeed;
}

OgcSandwichMonitor::OgcSandwichMonitor()
	: mUpperJaw(NULL), mLowerJaw(NULL), mBox(NULL),
	  mBoxHalfExtents(0.0f)
{
}

void OgcSandwichMonitor::releaseSources()
{
	mUpperJaw = NULL;
	mLowerJaw = NULL;
	mBox = NULL;
	mUpperJawSurfaceTriangles.reset();
	mLowerJawSurfaceTriangles.reset();
}

const OgcSandwichMetrics& OgcSandwichMonitor::getMetrics() const
{
	return mMetrics;
}

bool OgcSandwichMonitor::measureJawContactPatchOffset(
	PxDeformableVolume& jaw,
	bool lowerFace,
	PxReal& outOffset) const
{
	const PxTetrahedronMesh* const collisionMesh = jaw.getCollisionMesh();
	const PxVec4* const positions = jaw.getPositionInvMassBufferH();
	if(!collisionMesh || !positions || collisionMesh->getNbVertices() == 0)
		return false;

	PxVec3 centroid(0.0f);
	for(PxU32 vertexIndex = 0;
		vertexIndex < collisionMesh->getNbVertices(); ++vertexIndex)
	{
		const PxVec3 position = positions[vertexIndex].getXYZ();
		if(!position.isFinite())
			return false;
		centroid += position;
	}
	centroid *= 1.0f / collisionMesh->getNbVertices();

	const PxReal footprintPadding = 0.06f;
	const PxReal faceHalfHeight = 0.35f;
	PxReal patchCoordinate = 0.0f;
	PxU32 patchVertexCount = 0;
	for(PxU32 vertexIndex = 0;
		vertexIndex < collisionMesh->getNbVertices(); ++vertexIndex)
	{
		const PxVec3 position = positions[vertexIndex].getXYZ();
		const bool insideFootprint =
			PxAbs(position.x - mMetrics.initialBoxPosition.x) <=
				mBoxHalfExtents.x + footprintPadding &&
			PxAbs(position.z - mMetrics.initialBoxPosition.z) <=
				mBoxHalfExtents.z + footprintPadding;
		const bool onFacingHalf = lowerFace ?
			position.y <= centroid.y - faceHalfHeight :
			position.y >= centroid.y + faceHalfHeight;
		if(!insideFootprint || !onFacingHalf)
			continue;
		patchCoordinate += position.y;
		++patchVertexCount;
	}
	if(patchVertexCount == 0)
		return false;
	outOffset = patchCoordinate / patchVertexCount - centroid.y;
	return PxIsFinite(outOffset);
}

bool OgcSandwichMonitor::initialize(
	PxDeformableVolume& upperJaw,
	PxDeformableVolume& lowerJaw,
	PxRigidDynamic& box,
	const PxVec3& boxHalfExtents)
{
	releaseSources();
	mMetrics = OgcSandwichMetrics();
	if(!boxHalfExtents.isFinite() || boxHalfExtents.minElement() <= 0.0f ||
		!box.getScene() || upperJaw.getScene() != box.getScene() ||
		lowerJaw.getScene() != box.getScene() ||
		!upperJaw.getCollisionMesh() || !lowerJaw.getCollisionMesh())
		return false;
	mUpperJaw = &upperJaw;
	mLowerJaw = &lowerJaw;
	mBox = &box;
	mBoxHalfExtents = boxHalfExtents;

	PxTetrahedronMeshExt::extractTetMeshSurface(
		upperJaw.getCollisionMesh(), mUpperJawSurfaceTriangles, NULL, false);
	PxTetrahedronMeshExt::extractTetMeshSurface(
		lowerJaw.getCollisionMesh(), mLowerJawSurfaceTriangles, NULL, false);
	if(mUpperJawSurfaceTriangles.empty() ||
		mLowerJawSurfaceTriangles.empty() ||
		(mUpperJawSurfaceTriangles.size() % 3) != 0 ||
		(mLowerJawSurfaceTriangles.size() % 3) != 0)
	{
		releaseSources();
		return false;
	}

	const PxTransform boxPose = box.getGlobalPose();
	if(!boxPose.isValid())
	{
		releaseSources();
		return false;
	}
	mMetrics.initialBoxPosition = boxPose.p;
	if(!measureCollisionSurfaceBoxGap(upperJaw,
			mUpperJawSurfaceTriangles, boxPose, mBoxHalfExtents,
			mMetrics.minUpperJawSignedSdf,
			mMetrics.upperJawTrianglePenetrated) ||
		!measureCollisionSurfaceBoxGap(lowerJaw,
			mLowerJawSurfaceTriangles, boxPose, mBoxHalfExtents,
			mMetrics.minLowerJawSignedSdf,
			mMetrics.lowerJawTrianglePenetrated) ||
		!measureJawContactPatchOffset(
			upperJaw, true, mMetrics.initialUpperBottomOffset) ||
		!measureJawContactPatchOffset(
			lowerJaw, false, mMetrics.initialLowerTopOffset))
	{
		releaseSources();
		return false;
	}
	mMetrics.finalUpperJawSignedSdf = mMetrics.minUpperJawSignedSdf;
	mMetrics.finalLowerJawSignedSdf = mMetrics.minLowerJawSignedSdf;
	mMetrics.collisionTelemetryEnabled = true;
	mMetrics.initialized = true;
	return true;
}

bool OgcSandwichMonitor::sample(
	PxScene& scene,
	const VolumeHealthMonitor& healthMonitor,
	OgcSandwichFrameSample& sample)
{
	sample = OgcSandwichFrameSample();
	if(!mMetrics.initialized || !mMetrics.collisionTelemetryEnabled ||
		!mBox || !mUpperJaw || !mLowerJaw || mBox->getScene() != &scene)
		return false;

	PxSimulationStatistics statistics;
	scene.getSimulationStatistics(statistics);
	sample.generatedRigidContacts =
		statistics.avbdCpuSoftBodyCollisionGeneratedRigidContacts;
	mMetrics.generatedRigidContacts += sample.generatedRigidContacts;
	mMetrics.nativeIslandSteps +=
		statistics.avbdCpuSoftBodyNativeIslandSteps;

	const PxTransform boxPose = mBox->getGlobalPose();
	const PxVec3 boxVelocity = mBox->getLinearVelocity();
	if(!boxPose.isValid() || !boxVelocity.isFinite())
		return false;
	sample.boxPosition = boxPose.p;
	sample.boxVelocity = boxVelocity;
	PxReal upperSdf = PX_MAX_F32;
	PxReal lowerSdf = PX_MAX_F32;
	bool upperTrianglePenetrated = false;
	bool lowerTrianglePenetrated = false;
	if(!measureCollisionSurfaceBoxGap(*mUpperJaw,
			mUpperJawSurfaceTriangles, boxPose, mBoxHalfExtents, upperSdf,
			upperTrianglePenetrated) ||
		!measureCollisionSurfaceBoxGap(*mLowerJaw,
			mLowerJawSurfaceTriangles, boxPose, mBoxHalfExtents, lowerSdf,
			lowerTrianglePenetrated))
		return false;

	PxReal upperBottomOffset = PX_MAX_F32;
	PxReal lowerTopOffset = PX_MAX_F32;
	if(!measureJawContactPatchOffset(*mUpperJaw, true, upperBottomOffset) ||
		!measureJawContactPatchOffset(*mLowerJaw, false, lowerTopOffset))
		return false;
	mMetrics.maxUpperCompression = PxMax(mMetrics.maxUpperCompression,
		upperBottomOffset - mMetrics.initialUpperBottomOffset);
	mMetrics.maxLowerCompression = PxMax(mMetrics.maxLowerCompression,
		mMetrics.initialLowerTopOffset - lowerTopOffset);

	if(healthMonitor.getBodyCount() != 2 ||
		healthMonitor.getBody(0) != mUpperJaw ||
		healthMonitor.getBody(1) != mLowerJaw)
		return false;
	const VolumeBodyHealthMetrics& upperHealth =
		healthMonitor.getBodyMetrics(0);
	const VolumeBodyHealthMetrics& lowerHealth =
		healthMonitor.getBodyMetrics(1);
	if(!PxIsFinite(upperHealth.minDetF) ||
		!PxIsFinite(lowerHealth.minDetF))
		return false;
	mMetrics.upperJawMinDetF = PxMin(
		mMetrics.upperJawMinDetF, upperHealth.minDetF);
	mMetrics.lowerJawMinDetF = PxMin(
		mMetrics.lowerJawMinDetF, lowerHealth.minDetF);
	mMetrics.minUpperJawSignedSdf = PxMin(
		mMetrics.minUpperJawSignedSdf, upperSdf);
	mMetrics.minLowerJawSignedSdf = PxMin(
		mMetrics.minLowerJawSignedSdf, lowerSdf);
	mMetrics.finalUpperJawSignedSdf = upperSdf;
	mMetrics.finalLowerJawSignedSdf = lowerSdf;
	mMetrics.upperJawTrianglePenetrated =
		mMetrics.upperJawTrianglePenetrated || upperTrianglePenetrated;
	mMetrics.lowerJawTrianglePenetrated =
		mMetrics.lowerJawTrianglePenetrated || lowerTrianglePenetrated;
	mMetrics.upperJawTrianglePenetrationFrames +=
		upperTrianglePenetrated ? 1u : 0u;
	mMetrics.lowerJawTrianglePenetrationFrames +=
		lowerTrianglePenetrated ? 1u : 0u;

	const PxVec3 lateralOffset(
		boxPose.p.x - mMetrics.initialBoxPosition.x, 0.0f,
		boxPose.p.z - mMetrics.initialBoxPosition.z);
	const PxVec3 lateralVelocity(boxVelocity.x, 0.0f, boxVelocity.z);
	mMetrics.maxBoxLateralOffset = PxMax(
		mMetrics.maxBoxLateralOffset, lateralOffset.magnitude());
	mMetrics.maxBoxLateralSpeed = PxMax(
		mMetrics.maxBoxLateralSpeed, lateralVelocity.magnitude());
	mMetrics.maxBoxNormalOffset = PxMax(mMetrics.maxBoxNormalOffset,
		PxAbs(boxPose.p.y - mMetrics.initialBoxPosition.y));
	mMetrics.maxBoxNormalSpeed = PxMax(
		mMetrics.maxBoxNormalSpeed, PxAbs(boxVelocity.y));
	return true;
}

VisualInteractionMetrics::VisualInteractionMetrics()
	: generatedRigidContacts(0), generatedSoftContacts(0),
	  nativeIslandSteps(0), initialUpperJawSignedSdf(PX_MAX_F32),
	  initialLowerJawSignedSdf(PX_MAX_F32),
	  minUpperJawSignedSdf(PX_MAX_F32),
	  minLowerJawSignedSdf(PX_MAX_F32),
	  finalUpperJawSignedSdf(PX_MAX_F32),
	  finalLowerJawSignedSdf(PX_MAX_F32),
	  upperJawTrianglePenetrationFrames(0),
	  lowerJawTrianglePenetrationFrames(0),
	  upperJawTrianglePenetrationFirstFrame(PX_MAX_U32),
	  lowerJawTrianglePenetrationFirstFrame(PX_MAX_U32),
	  upperJawTrianglePenetrated(false),
	  lowerJawTrianglePenetrated(false),
	  collisionTelemetryEnabled(false), initialized(false)
{
}

VisualInteractionFrameSample::VisualInteractionFrameSample()
	: generatedRigidContacts(0), generatedSoftContacts(0),
	  boxPosition(0.0f), boxLinearVelocity(0.0f)
{
}

bool isVisualDynamicSurfaceSeparated(
	const VisualInteractionMetrics& metrics)
{
	return metrics.initialized && metrics.collisionTelemetryEnabled &&
		PxIsFinite(metrics.minUpperJawSignedSdf) &&
		PxIsFinite(metrics.minLowerJawSignedSdf) &&
		!metrics.upperJawTrianglePenetrated &&
		!metrics.lowerJawTrianglePenetrated &&
		metrics.minUpperJawSignedSdf >= -MAX_SURFACE_PENETRATION &&
		metrics.minLowerJawSignedSdf >= -MAX_SURFACE_PENETRATION;
}

bool isVisualStaticSurfaceSeparated(
	const VisualInteractionMetrics& metrics)
{
	return metrics.initialized && metrics.collisionTelemetryEnabled &&
		isSurfaceGapSeparated(metrics.upperJawPedestalGap) &&
		isSurfaceGapSeparated(metrics.lowerJawPedestalGap) &&
		isSurfaceGapSeparated(metrics.upperJawGroundGap) &&
		isSurfaceGapSeparated(metrics.lowerJawGroundGap);
}

VisualInteractionMonitor::VisualInteractionMonitor()
	: mUpperJaw(NULL), mLowerJaw(NULL), mDynamicBox(NULL), mPedestal(NULL),
	  mDynamicBoxHalfExtents(0.0f), mPedestalHalfExtents(0.0f),
	  mGroundHeight(0.0f)
{
}

void VisualInteractionMonitor::releaseSources()
{
	mUpperJaw = NULL;
	mLowerJaw = NULL;
	mDynamicBox = NULL;
	mPedestal = NULL;
	mUpperJawSurfaceTriangles.reset();
	mLowerJawSurfaceTriangles.reset();
}

const VisualInteractionMetrics& VisualInteractionMonitor::getMetrics() const
{
	return mMetrics;
}

bool VisualInteractionMonitor::initialize(
	PxDeformableVolume& upperJaw,
	PxDeformableVolume& lowerJaw,
	PxRigidDynamic& dynamicBox,
	PxRigidStatic& pedestal,
	const PxVec3& dynamicBoxHalfExtents,
	const PxVec3& pedestalHalfExtents,
	PxReal groundHeight)
{
	releaseSources();
	mMetrics = VisualInteractionMetrics();
	PxScene* const scene = dynamicBox.getScene();
	if(!scene || pedestal.getScene() != scene || upperJaw.getScene() != scene ||
		lowerJaw.getScene() != scene || !upperJaw.getCollisionMesh() ||
		!lowerJaw.getCollisionMesh() || !dynamicBoxHalfExtents.isFinite() ||
		!pedestalHalfExtents.isFinite() ||
		dynamicBoxHalfExtents.minElement() <= 0.0f ||
		pedestalHalfExtents.minElement() <= 0.0f ||
		!PxIsFinite(groundHeight))
		return false;
	mUpperJaw = &upperJaw;
	mLowerJaw = &lowerJaw;
	mDynamicBox = &dynamicBox;
	mPedestal = &pedestal;
	mDynamicBoxHalfExtents = dynamicBoxHalfExtents;
	mPedestalHalfExtents = pedestalHalfExtents;
	mGroundHeight = groundHeight;

	PxTetrahedronMeshExt::extractTetMeshSurface(
		upperJaw.getCollisionMesh(), mUpperJawSurfaceTriangles, NULL, false);
	PxTetrahedronMeshExt::extractTetMeshSurface(
		lowerJaw.getCollisionMesh(), mLowerJawSurfaceTriangles, NULL, false);
	if(mUpperJawSurfaceTriangles.empty() ||
		mLowerJawSurfaceTriangles.empty() ||
		(mUpperJawSurfaceTriangles.size() % 3) != 0 ||
		(mLowerJawSurfaceTriangles.size() % 3) != 0)
	{
		releaseSources();
		return false;
	}

	const PxTransform dynamicBoxPose = dynamicBox.getGlobalPose();
	const PxTransform pedestalPose = pedestal.getGlobalPose();
	if(!dynamicBoxPose.isValid() || !pedestalPose.isValid() ||
		!measureCollisionSurfaceBoxGap(upperJaw,
			mUpperJawSurfaceTriangles, dynamicBoxPose,
			mDynamicBoxHalfExtents, mMetrics.initialUpperJawSignedSdf,
			mMetrics.upperJawTrianglePenetrated) ||
		!measureCollisionSurfaceBoxGap(lowerJaw,
			mLowerJawSurfaceTriangles, dynamicBoxPose,
			mDynamicBoxHalfExtents, mMetrics.initialLowerJawSignedSdf,
			mMetrics.lowerJawTrianglePenetrated) ||
		!measureCollisionSurfaceBoxGap(upperJaw,
			mUpperJawSurfaceTriangles, pedestalPose, mPedestalHalfExtents,
			mMetrics.upperJawPedestalGap.initialSignedSdf,
			mMetrics.upperJawPedestalGap.penetrated) ||
		!measureCollisionSurfaceBoxGap(lowerJaw,
			mLowerJawSurfaceTriangles, pedestalPose, mPedestalHalfExtents,
			mMetrics.lowerJawPedestalGap.initialSignedSdf,
			mMetrics.lowerJawPedestalGap.penetrated) ||
		!measureCollisionSurfaceGroundGap(upperJaw,
			mUpperJawSurfaceTriangles, mGroundHeight,
			mMetrics.upperJawGroundGap.initialSignedSdf,
			mMetrics.upperJawGroundGap.penetrated) ||
		!measureCollisionSurfaceGroundGap(lowerJaw,
			mLowerJawSurfaceTriangles, mGroundHeight,
			mMetrics.lowerJawGroundGap.initialSignedSdf,
			mMetrics.lowerJawGroundGap.penetrated))
	{
		releaseSources();
		return false;
	}
	mMetrics.minUpperJawSignedSdf = mMetrics.initialUpperJawSignedSdf;
	mMetrics.minLowerJawSignedSdf = mMetrics.initialLowerJawSignedSdf;
	mMetrics.finalUpperJawSignedSdf = mMetrics.initialUpperJawSignedSdf;
	mMetrics.finalLowerJawSignedSdf = mMetrics.initialLowerJawSignedSdf;
	initializeSurfaceGapMetrics(mMetrics.upperJawPedestalGap,
		mMetrics.upperJawPedestalGap.initialSignedSdf,
		mMetrics.upperJawPedestalGap.penetrated);
	initializeSurfaceGapMetrics(mMetrics.lowerJawPedestalGap,
		mMetrics.lowerJawPedestalGap.initialSignedSdf,
		mMetrics.lowerJawPedestalGap.penetrated);
	initializeSurfaceGapMetrics(mMetrics.upperJawGroundGap,
		mMetrics.upperJawGroundGap.initialSignedSdf,
		mMetrics.upperJawGroundGap.penetrated);
	initializeSurfaceGapMetrics(mMetrics.lowerJawGroundGap,
		mMetrics.lowerJawGroundGap.initialSignedSdf,
		mMetrics.lowerJawGroundGap.penetrated);
	mMetrics.collisionTelemetryEnabled = true;
	mMetrics.initialized = true;
	return true;
}

bool VisualInteractionMonitor::sample(
	PxScene& scene,
	PxU32 completedFrame,
	VisualInteractionFrameSample& sample)
{
	sample = VisualInteractionFrameSample();
	if(!mMetrics.initialized || !mMetrics.collisionTelemetryEnabled ||
		!mUpperJaw || !mLowerJaw || !mDynamicBox || !mPedestal ||
		mDynamicBox->getScene() != &scene || mPedestal->getScene() != &scene)
		return false;

	PxSimulationStatistics sceneStatistics;
	scene.getSimulationStatistics(sceneStatistics);
	sample.generatedRigidContacts =
		sceneStatistics.avbdCpuSoftBodyCollisionGeneratedRigidContacts;
	sample.generatedSoftContacts =
		sceneStatistics.avbdCpuSoftBodyCollisionGeneratedSoftContacts;
	mMetrics.generatedRigidContacts += sample.generatedRigidContacts;
	mMetrics.generatedSoftContacts += sample.generatedSoftContacts;
	mMetrics.nativeIslandSteps +=
		sceneStatistics.avbdCpuSoftBodyNativeIslandSteps;

	const PxTransform boxPose = mDynamicBox->getGlobalPose();
	const PxTransform pedestalPose = mPedestal->getGlobalPose();
	const PxVec3 linearVelocity = mDynamicBox->getLinearVelocity();
	const PxVec3 angularVelocity = mDynamicBox->getAngularVelocity();
	if(!boxPose.isValid() || !pedestalPose.isValid() ||
		!linearVelocity.isFinite() || !angularVelocity.isFinite())
		return false;
	sample.boxPosition = boxPose.p;
	sample.boxLinearVelocity = linearVelocity;

	PxReal upperJawSignedSdf = PX_MAX_F32;
	PxReal lowerJawSignedSdf = PX_MAX_F32;
	bool upperJawTrianglePenetrated = false;
	bool lowerJawTrianglePenetrated = false;
	PxReal upperJawPedestalSignedSdf = PX_MAX_F32;
	PxReal lowerJawPedestalSignedSdf = PX_MAX_F32;
	bool upperJawPedestalTrianglePenetrated = false;
	bool lowerJawPedestalTrianglePenetrated = false;
	PxReal upperJawGroundSignedSdf = PX_MAX_F32;
	PxReal lowerJawGroundSignedSdf = PX_MAX_F32;
	bool upperJawGroundPlanePenetrated = false;
	bool lowerJawGroundPlanePenetrated = false;
	if(!measureCollisionSurfaceBoxGap(*mUpperJaw,
			mUpperJawSurfaceTriangles, boxPose, mDynamicBoxHalfExtents,
			upperJawSignedSdf, upperJawTrianglePenetrated) ||
		!measureCollisionSurfaceBoxGap(*mLowerJaw,
			mLowerJawSurfaceTriangles, boxPose, mDynamicBoxHalfExtents,
			lowerJawSignedSdf, lowerJawTrianglePenetrated) ||
		!measureCollisionSurfaceBoxGap(*mUpperJaw,
			mUpperJawSurfaceTriangles, pedestalPose, mPedestalHalfExtents,
			upperJawPedestalSignedSdf,
			upperJawPedestalTrianglePenetrated) ||
		!measureCollisionSurfaceBoxGap(*mLowerJaw,
			mLowerJawSurfaceTriangles, pedestalPose, mPedestalHalfExtents,
			lowerJawPedestalSignedSdf,
			lowerJawPedestalTrianglePenetrated) ||
		!measureCollisionSurfaceGroundGap(*mUpperJaw,
			mUpperJawSurfaceTriangles, mGroundHeight, upperJawGroundSignedSdf,
			upperJawGroundPlanePenetrated) ||
		!measureCollisionSurfaceGroundGap(*mLowerJaw,
			mLowerJawSurfaceTriangles, mGroundHeight, lowerJawGroundSignedSdf,
			lowerJawGroundPlanePenetrated))
		return false;

	mMetrics.minUpperJawSignedSdf = PxMin(
		mMetrics.minUpperJawSignedSdf, upperJawSignedSdf);
	mMetrics.minLowerJawSignedSdf = PxMin(
		mMetrics.minLowerJawSignedSdf, lowerJawSignedSdf);
	mMetrics.finalUpperJawSignedSdf = upperJawSignedSdf;
	mMetrics.finalLowerJawSignedSdf = lowerJawSignedSdf;
	mMetrics.upperJawTrianglePenetrated =
		mMetrics.upperJawTrianglePenetrated || upperJawTrianglePenetrated;
	mMetrics.lowerJawTrianglePenetrated =
		mMetrics.lowerJawTrianglePenetrated || lowerJawTrianglePenetrated;
	if(upperJawTrianglePenetrated)
	{
		mMetrics.upperJawTrianglePenetrationFrames++;
		if(mMetrics.upperJawTrianglePenetrationFirstFrame == PX_MAX_U32)
			mMetrics.upperJawTrianglePenetrationFirstFrame = completedFrame;
	}
	if(lowerJawTrianglePenetrated)
	{
		mMetrics.lowerJawTrianglePenetrationFrames++;
		if(mMetrics.lowerJawTrianglePenetrationFirstFrame == PX_MAX_U32)
			mMetrics.lowerJawTrianglePenetrationFirstFrame = completedFrame;
	}
	updateSurfaceGapMetrics(mMetrics.upperJawPedestalGap,
		upperJawPedestalSignedSdf, upperJawPedestalTrianglePenetrated);
	updateSurfaceGapMetrics(mMetrics.lowerJawPedestalGap,
		lowerJawPedestalSignedSdf, lowerJawPedestalTrianglePenetrated);
	updateSurfaceGapMetrics(mMetrics.upperJawGroundGap,
		upperJawGroundSignedSdf, upperJawGroundPlanePenetrated);
	updateSurfaceGapMetrics(mMetrics.lowerJawGroundGap,
		lowerJawGroundSignedSdf, lowerJawGroundPlanePenetrated);
	return true;
}

RotationSamplingConfig::RotationSamplingConfig()
	: earlyEndFrame(0), lateBeginFrame(0), windowBeginFrame(0),
	  windowEndFrame(0), checkpointInterval(0), checkpointCount(0),
	  groundEnterHeight(0.01f), groundExitHeight(0.05f)
{
}

RotationMetrics::RotationMetrics()
	: maxOrientationChange(0.0f), maxAngularSpeed(0.0f),
	  finalAngularSpeed(0.0f), earlyMaxAngularSpeed(0.0f),
	  lateMaxAngularSpeed(0.0f), windowMinAngularSpeed(PX_MAX_F32),
	  windowMaxAngularSpeed(0.0f), windowAngularSpeedSum(0.0),
	  windowSampleCount(0), finalLinearSpeed(0.0f),
	  finalLinearVelocity(0.0f), finalAngularVelocity(0.0f),
	  finalCentroid(0.0f), finalLowestCollisionOffset(0.0f),
	  finalRigidRollSlipSpeed(0.0f), finalMinCollisionY(PX_MAX_F32),
	  windowMinLinearSpeed(PX_MAX_F32), windowMaxLinearSpeed(0.0f),
	  windowLinearSpeedSum(0.0), checkpointAngularSpeeds{0.0f},
	  checkpointLinearSpeeds{0.0f}, checkpointCentroidY{0.0f},
	  maxAngularSpeedFrame(PX_MAX_U32), groundContactEpisodes(0),
	  firstGroundContactFrame(PX_MAX_U32),
	  secondGroundContactFrame(PX_MAX_U32),
	  minSecondGroundEpisodeAngularSpeed(PX_MAX_F32),
	  groundContactActive(false), initialized(false)
{
}

bool isRotationResponseObserved(
	const RotationMetrics& metrics, PxReal minOrientationChange,
	PxReal minAngularSpeed)
{
	return metrics.maxOrientationChange >= minOrientationChange &&
		metrics.maxAngularSpeed >= minAngularSpeed &&
		metrics.maxAngularSpeedFrame != PX_MAX_U32;
}

bool isRotationLongRunBounded(
	const RotationMetrics& metrics, PxU32 completedFrames,
	PxU32 minFrames, PxReal maxLateSpeedFloor,
	PxReal maxLateSpeedRatio)
{
	if(completedFrames < minFrames)
		return true;
	const PxReal allowedLateSpeed = PxMax(
		maxLateSpeedFloor,
		metrics.earlyMaxAngularSpeed * maxLateSpeedRatio);
	return metrics.lateMaxAngularSpeed <= allowedLateSpeed &&
		metrics.finalAngularSpeed <= allowedLateSpeed;
}

bool isRollingKinematicsValid(
	const RotationMetrics& metrics, PxReal minOrientationChange,
	PxReal minAngularSpeed, PxReal maxRigidSlipSpeed)
{
	return metrics.groundContactEpisodes >= 1u &&
		metrics.firstGroundContactFrame != PX_MAX_U32 &&
		metrics.groundContactActive &&
		metrics.maxOrientationChange >= minOrientationChange &&
		metrics.maxAngularSpeed >= minAngularSpeed &&
		metrics.finalLinearVelocity.isFinite() &&
		metrics.finalAngularVelocity.isFinite() &&
		PxIsFinite(metrics.finalRigidRollSlipSpeed) &&
		metrics.finalRigidRollSlipSpeed <= maxRigidSlipSpeed;
}

RotationMonitor::RotationMonitor()
	: mVolume(NULL), mAxisVertex0(PX_MAX_U32),
	  mAxisVertex1(PX_MAX_U32), mInitialAxis(0.0f)
{
}

void RotationMonitor::releaseSource()
{
	mVolume = NULL;
	mMetrics.initialized = false;
}

const RotationMetrics& RotationMonitor::getMetrics() const
{
	return mMetrics;
}

bool RotationMonitor::initialize(
	PxDeformableVolume& volume,
	const RotationSamplingConfig& config)
{
	releaseSource();
	mMetrics = RotationMetrics();
	if(config.checkpointCount > ROTATION_CHECKPOINT_CAPACITY ||
		(config.checkpointCount > 0 && config.checkpointInterval == 0) ||
		config.windowEndFrame < config.windowBeginFrame ||
		!PxIsFinite(config.groundEnterHeight) ||
		!PxIsFinite(config.groundExitHeight) ||
		config.groundExitHeight < config.groundEnterHeight)
		return false;
	const PxTetrahedronMesh* collisionMesh = volume.getCollisionMesh();
	const PxVec4* positions = volume.getPositionInvMassBufferH();
	if(!collisionMesh || !positions || collisionMesh->getNbVertices() < 2)
		return false;
	const PxVec3* restVertices = collisionMesh->getVertices();
	PxU32 minX = 0;
	PxU32 maxX = 0;
	for(PxU32 vertexIndex = 1;
		vertexIndex < collisionMesh->getNbVertices(); ++vertexIndex)
	{
		if(restVertices[vertexIndex].x < restVertices[minX].x)
			minX = vertexIndex;
		if(restVertices[vertexIndex].x > restVertices[maxX].x)
			maxX = vertexIndex;
	}
	PxVec3 axis = positions[maxX].getXYZ() - positions[minX].getXYZ();
	if(axis.normalize() <= 1.0e-6f)
		return false;
	mVolume = &volume;
	mAxisVertex0 = minX;
	mAxisVertex1 = maxX;
	mInitialAxis = axis;
	mConfig = config;
	mMetrics.initialized = true;
	return true;
}

bool RotationMonitor::sample(PxU32 completedFrame)
{
	if(!mMetrics.initialized || !mVolume)
		return false;
	PxDeformableVolume& volume = *mVolume;
	const PxTetrahedronMesh* collisionMesh = volume.getCollisionMesh();
	const PxVec4* collisionPositions = volume.getPositionInvMassBufferH();
	const PxTetrahedronMesh* simulationMesh = volume.getSimulationMesh();
	const PxVec4* positions = volume.getSimPositionInvMassBufferH();
	const PxVec4* velocities = volume.getSimVelocityBufferH();
	if(!collisionMesh || !collisionPositions || !simulationMesh ||
		!positions || !velocities ||
		mAxisVertex0 >= collisionMesh->getNbVertices() ||
		mAxisVertex1 >= collisionMesh->getNbVertices())
		return false;

	PxVec3 axis = collisionPositions[mAxisVertex1].getXYZ() -
		collisionPositions[mAxisVertex0].getXYZ();
	if(axis.normalize() <= 1.0e-6f)
		return false;
	const PxReal orientationChange = PxAcos(PxClamp(
		mInitialAxis.dot(axis), -1.0f, 1.0f));
	if(!PxIsFinite(orientationChange))
		return false;
	mMetrics.maxOrientationChange = PxMax(
		mMetrics.maxOrientationChange, orientationChange);

	PxReal mass = 0.0f;
	PxVec3 centroid(0.0f);
	PxVec3 linearMomentum(0.0f);
	for(PxU32 vertexIndex = 0;
		vertexIndex < simulationMesh->getNbVertices(); ++vertexIndex)
	{
		const PxReal invMass = positions[vertexIndex].w;
		if(invMass <= 0.0f || !PxIsFinite(invMass))
			continue;
		const PxReal vertexMass = 1.0f / invMass;
		centroid += positions[vertexIndex].getXYZ() * vertexMass;
		linearMomentum += velocities[vertexIndex].getXYZ() * vertexMass;
		mass += vertexMass;
	}
	if(mass <= 0.0f)
		return false;
	centroid *= 1.0f / mass;
	const PxVec3 linearVelocity = linearMomentum * (1.0f / mass);
	PxMat33 inertia(PxZero);
	PxVec3 angularMomentum(0.0f);
	for(PxU32 vertexIndex = 0;
		vertexIndex < simulationMesh->getNbVertices(); ++vertexIndex)
	{
		const PxReal invMass = positions[vertexIndex].w;
		if(invMass <= 0.0f || !PxIsFinite(invMass))
			continue;
		const PxReal vertexMass = 1.0f / invMass;
		const PxVec3 offset = positions[vertexIndex].getXYZ() - centroid;
		const PxVec3 relativeVelocity =
			velocities[vertexIndex].getXYZ() - linearVelocity;
		const PxMat33 outer(offset * offset.x,
			offset * offset.y, offset * offset.z);
		inertia = inertia +
			(PxMat33::createDiagonal(
				PxVec3(offset.magnitudeSquared())) - outer) * vertexMass;
		angularMomentum += offset.cross(relativeVelocity) * vertexMass;
	}
	const PxReal determinant = inertia.getDeterminant();
	if(!PxIsFinite(determinant) || PxAbs(determinant) <= 1.0e-12f)
		return false;
	const PxVec3 angularVelocity = inertia.getInverse() * angularMomentum;
	const PxReal angularSpeed = angularVelocity.magnitude();
	const PxReal linearSpeed = linearVelocity.magnitude();
	if(!PxIsFinite(angularSpeed) || !PxIsFinite(linearSpeed))
		return false;
	mMetrics.finalAngularSpeed = angularSpeed;
	mMetrics.finalLinearSpeed = linearSpeed;
	mMetrics.finalLinearVelocity = linearVelocity;
	mMetrics.finalAngularVelocity = angularVelocity;
	mMetrics.finalCentroid = centroid;
	PxReal minCollisionY = PX_MAX_F32;
	PxVec3 lowestCollisionPoint(0.0f);
	for(PxU32 vertexIndex = 0;
		vertexIndex < collisionMesh->getNbVertices(); ++vertexIndex)
	{
		const PxVec3 point = collisionPositions[vertexIndex].getXYZ();
		if(point.y < minCollisionY)
		{
			minCollisionY = point.y;
			lowestCollisionPoint = point;
		}
	}
	if(!PxIsFinite(minCollisionY))
		return false;
	mMetrics.finalLowestCollisionOffset = lowestCollisionPoint - centroid;
	const PxVec3 rigidContactVelocity = linearVelocity +
		angularVelocity.cross(mMetrics.finalLowestCollisionOffset);
	mMetrics.finalRigidRollSlipSpeed = PxSqrt(
		rigidContactVelocity.x * rigidContactVelocity.x +
		rigidContactVelocity.z * rigidContactVelocity.z);
	if(!mMetrics.finalLowestCollisionOffset.isFinite() ||
		!PxIsFinite(mMetrics.finalRigidRollSlipSpeed))
		return false;
	mMetrics.finalMinCollisionY = minCollisionY;
	if(!mMetrics.groundContactActive &&
		minCollisionY <= mConfig.groundEnterHeight)
	{
		mMetrics.groundContactActive = true;
		++mMetrics.groundContactEpisodes;
		if(mMetrics.groundContactEpisodes == 1u)
			mMetrics.firstGroundContactFrame = completedFrame;
		else if(mMetrics.groundContactEpisodes == 2u)
			mMetrics.secondGroundContactFrame = completedFrame;
	}
	else if(mMetrics.groundContactActive &&
		minCollisionY > mConfig.groundExitHeight)
		mMetrics.groundContactActive = false;
	if(mMetrics.groundContactEpisodes >= 2u &&
		mMetrics.groundContactActive)
		mMetrics.minSecondGroundEpisodeAngularSpeed = PxMin(
			mMetrics.minSecondGroundEpisodeAngularSpeed, angularSpeed);
	if(completedFrame <= mConfig.earlyEndFrame)
		mMetrics.earlyMaxAngularSpeed = PxMax(
			mMetrics.earlyMaxAngularSpeed, angularSpeed);
	if(completedFrame >= mConfig.lateBeginFrame)
		mMetrics.lateMaxAngularSpeed = PxMax(
			mMetrics.lateMaxAngularSpeed, angularSpeed);
	if(completedFrame >= mConfig.windowBeginFrame &&
		completedFrame <= mConfig.windowEndFrame)
	{
		mMetrics.windowMinAngularSpeed = PxMin(
			mMetrics.windowMinAngularSpeed, angularSpeed);
		mMetrics.windowMaxAngularSpeed = PxMax(
			mMetrics.windowMaxAngularSpeed, angularSpeed);
		mMetrics.windowAngularSpeedSum += angularSpeed;
		mMetrics.windowMinLinearSpeed = PxMin(
			mMetrics.windowMinLinearSpeed, linearSpeed);
		mMetrics.windowMaxLinearSpeed = PxMax(
			mMetrics.windowMaxLinearSpeed, linearSpeed);
		mMetrics.windowLinearSpeedSum += linearSpeed;
		mMetrics.windowSampleCount++;
	}
	if(completedFrame > 0 && mConfig.checkpointInterval > 0 &&
		completedFrame % mConfig.checkpointInterval == 0)
	{
		const PxU32 checkpointIndex =
			completedFrame / mConfig.checkpointInterval - 1;
		if(checkpointIndex < mConfig.checkpointCount)
		{
			mMetrics.checkpointAngularSpeeds[checkpointIndex] = angularSpeed;
			mMetrics.checkpointLinearSpeeds[checkpointIndex] = linearSpeed;
			mMetrics.checkpointCentroidY[checkpointIndex] = centroid.y;
		}
	}
	if(angularSpeed > mMetrics.maxAngularSpeed)
	{
		mMetrics.maxAngularSpeed = angularSpeed;
		mMetrics.maxAngularSpeedFrame = completedFrame;
	}
	return true;
}

SoftContactPhaseMetrics::SoftContactPhaseMetrics()
	: firstSoftContactFrame(PX_MAX_U32), lastSoftContactFrame(PX_MAX_U32),
	  peakSoftContactFrame(PX_MAX_U32), softContactFrames(0),
	  generatedGroundContacts(0), generatedSoftContacts(0),
	  preSoftAngularMomentum(0.0f), preSoftAngularSpeed(0.0f),
	  peakSoftContactAngularMomentum(0.0f),
	  peakSoftContactAngularSpeed(0.0f),
	  lastSoftContactAngularMomentum(0.0f),
	  lastSoftContactAngularSpeed(0.0f),
	  finalPostSoftAngularMomentum(0.0f),
	  finalPostSoftAngularSpeed(0.0f), lastNoSoftAngularMomentum(0.0f),
	  lastNoSoftAngularSpeed(0.0f), preSoftAngularVelocity(0.0f),
	  peakSoftContactAngularVelocity(0.0f),
	  lastNoSoftAngularVelocity(0.0f), hasPreSoftContactSample(false),
	  contactTelemetryEnabled(false), initialized(false)
{
}

SoftContactPhaseFrameSample::SoftContactPhaseFrameSample()
	: generatedGroundContacts(0), generatedSoftContacts(0)
{
}

SoftContactPhaseMonitor::SoftContactPhaseMonitor()
	: mTarget(NULL)
{
}

void SoftContactPhaseMonitor::releaseSource()
{
	mTarget = NULL;
}

const SoftContactPhaseMetrics& SoftContactPhaseMonitor::getMetrics() const
{
	return mMetrics;
}

bool SoftContactPhaseMonitor::initialize(
	PxDeformableVolume& target,
	bool contactTelemetryEnabled)
{
	releaseSource();
	mMetrics = SoftContactPhaseMetrics();
	mTarget = &target;
	mMetrics.contactTelemetryEnabled = contactTelemetryEnabled;
	mMetrics.initialized = true;
	return true;
}

bool SoftContactPhaseMonitor::sample(
	PxScene& scene,
	PxU32 completedFrame,
	SoftContactPhaseFrameSample& sample)
{
	sample = SoftContactPhaseFrameSample();
	if(!mMetrics.initialized || !mTarget)
		return false;
	if(!mMetrics.contactTelemetryEnabled)
		return true;

	PxVec3 centroid(0.0f);
	PxVec3 angularMomentum(0.0f);
	PxVec3 angularVelocity(0.0f);
	if(!measureVolumeAngularState(
			*mTarget, centroid, angularMomentum, angularVelocity))
		return false;
	const PxReal angularMomentumMagnitude = angularMomentum.magnitude();
	const PxReal angularSpeed = angularVelocity.magnitude();
	if(!PxIsFinite(angularMomentumMagnitude) || !PxIsFinite(angularSpeed))
		return false;

	PxSimulationStatistics sceneStatistics;
	scene.getSimulationStatistics(sceneStatistics);
	sample.generatedGroundContacts =
		sceneStatistics.avbdCpuSoftBodyCollisionGeneratedGroundContacts;
	sample.generatedSoftContacts =
		sceneStatistics.avbdCpuSoftBodyCollisionGeneratedSoftContacts;
	mMetrics.generatedGroundContacts += sample.generatedGroundContacts;
	mMetrics.generatedSoftContacts += sample.generatedSoftContacts;
	if(sample.generatedSoftContacts == 0)
	{
		if(mMetrics.firstSoftContactFrame == PX_MAX_U32)
		{
			mMetrics.hasPreSoftContactSample = true;
			mMetrics.lastNoSoftAngularMomentum = angularMomentumMagnitude;
			mMetrics.lastNoSoftAngularSpeed = angularSpeed;
			mMetrics.lastNoSoftAngularVelocity = angularVelocity;
		}
		else
		{
			mMetrics.finalPostSoftAngularMomentum = angularMomentumMagnitude;
			mMetrics.finalPostSoftAngularSpeed = angularSpeed;
		}
		return true;
	}

	if(mMetrics.firstSoftContactFrame == PX_MAX_U32)
	{
		mMetrics.firstSoftContactFrame = completedFrame;
		mMetrics.preSoftAngularMomentum =
			mMetrics.lastNoSoftAngularMomentum;
		mMetrics.preSoftAngularSpeed = mMetrics.lastNoSoftAngularSpeed;
		mMetrics.preSoftAngularVelocity = mMetrics.lastNoSoftAngularVelocity;
	}
	mMetrics.softContactFrames++;
	mMetrics.lastSoftContactFrame = completedFrame;
	mMetrics.lastSoftContactAngularMomentum = angularMomentumMagnitude;
	mMetrics.lastSoftContactAngularSpeed = angularSpeed;
	if(angularSpeed > mMetrics.peakSoftContactAngularSpeed)
	{
		mMetrics.peakSoftContactAngularSpeed = angularSpeed;
		mMetrics.peakSoftContactAngularMomentum = angularMomentumMagnitude;
		mMetrics.peakSoftContactAngularVelocity = angularVelocity;
		mMetrics.peakSoftContactFrame = completedFrame;
	}
	return true;
}

SoftSoftTorqueMetrics::SoftSoftTorqueMetrics()
	: targetSimulationVertices(0), targetCollisionVertices(0),
	  driverSimulationVertices(0), driverCollisionVertices(0),
	  targetDistinctCollisionSimulation(0),
	  driverDistinctCollisionSimulation(0), isolatedConfiguration(0),
	  supportExpansionInstrumentationAvailable(0), softContactFrames(0),
	  firstContactFrame(PX_MAX_U32), firstRotationFrame(PX_MAX_U32),
	  retainedRotationSamples(0), generatedSoftContacts(0),
	  generatedGroundContacts(0), generatedRigidContacts(0),
	  generatedSelfContacts(0), firstContactCentroidLeverArm(0.0f),
	  maxCentroidLeverArm(0.0f), maxAngularMomentum(0.0f),
	  finalAngularMomentum(0.0f), maxAngularSpeed(0.0f),
	  finalAngularSpeed(0.0f), initialized(false)
{
}

SoftSoftTorqueFrameSample::SoftSoftTorqueFrameSample()
	: generatedGroundContacts(0), generatedRigidContacts(0),
	  generatedSoftContacts(0), generatedSelfContacts(0),
	  driverBoundsFinite(false), targetBoundsFinite(false)
{
}

SoftSoftTorqueMonitor::SoftSoftTorqueMonitor()
	: mDriver(NULL), mTarget(NULL)
{
}

void SoftSoftTorqueMonitor::releaseSources()
{
	mDriver = NULL;
	mTarget = NULL;
}

const SoftSoftTorqueMetrics& SoftSoftTorqueMonitor::getMetrics() const
{
	return mMetrics;
}

OgcSandwichValidationConfig::OgcSandwichValidationConfig()
	: gateMinFrames(0), minNativeIslandSteps(0), minJawCompression(0.0f),
	  maxLateralOffset(0.0f), maxLateralSpeed(0.0f),
	  maxNormalOffset(0.0f), maxNormalSpeed(0.0f), minDetF(0.0f),
	  maxDetF(0.0f)
{
}

bool isOgcSandwichResultValid(
	const DeformableVolumeMetrics& metrics,
	const OgcSandwichMetrics& sandwich,
	const OgcSandwichValidationConfig& config)
{
	const bool gateActive = metrics.completedFrames >= config.gateMinFrames;
	const bool interactionObserved = !gateActive ||
		(sandwich.nativeIslandSteps >= config.minNativeIslandSteps &&
		 isOgcSandwichCompressed(sandwich, config.minJawCompression) &&
		 isOgcSandwichSurfaceSeparated(sandwich) &&
		 isOgcSandwichContained(
			sandwich, config.maxLateralOffset, config.maxLateralSpeed,
			config.maxNormalOffset, config.maxNormalSpeed));
	const bool volumeHealthPassed = metrics.invertedElementSamples == 0 &&
		PxIsFinite(metrics.minDetF) && metrics.minDetF >= config.minDetF &&
		PxIsFinite(metrics.maxDetF) && metrics.maxDetF < config.maxDetF;

	return metrics.sceneStatics == 0 && metrics.sceneDynamics == 1 &&
		metrics.sceneDeformableVolumes == 2 &&
		metrics.surfaceTriangles == 2u * 192u && interactionObserved &&
		volumeHealthPassed;
}

VisualShowcaseValidationConfig::VisualShowcaseValidationConfig()
	: rotationGateMinFrames(0), interactionGateMinFrames(0),
	  minSurfaceTriangles(0), primaryMinOrientationChange(0.0f),
	  primaryMinAngularSpeed(0.0f), sphereMinOrientationChange(0.0f),
	  sphereMinAngularSpeed(0.0f), minDetF(0.0f), maxDetF(0.0f),
	  minVolumeRatio(0.0f), maxVolumeRatio(0.0f),
	  sphereLongRunBounded(false)
{
}

bool isVisualShowcaseResultValid(
	const DeformableVolumeMetrics& metrics,
	const RotationMetrics& primaryRotation,
	const RotationMetrics& sphereRotation,
	const VisualInteractionMetrics& interaction,
	const VisualShowcaseValidationConfig& config)
{
	const bool rotationGateActive =
		metrics.completedFrames >= config.rotationGateMinFrames;
	const bool rotationPassed = !rotationGateActive ||
		isRotationResponseObserved(
			primaryRotation, config.primaryMinOrientationChange,
			config.primaryMinAngularSpeed);
	const bool sphereRollPassed = !rotationGateActive ||
		isRotationResponseObserved(
			sphereRotation, config.sphereMinOrientationChange,
			config.sphereMinAngularSpeed);
	const bool dynamicFalling = metrics.sceneDynamicInitiallySleeping == 0 &&
		metrics.sceneDynamicMaxDrop > 0.25f &&
		metrics.sceneDynamicMaxDownSpeed > 0.25f;
	const bool dynamicSurfaceSeparated =
		isVisualDynamicSurfaceSeparated(interaction);
	const bool staticSurfaceSeparated =
		isVisualStaticSurfaceSeparated(interaction);
	const bool interactionGateActive =
		metrics.completedFrames >= config.interactionGateMinFrames;
	const bool mixedInteractionPassed =
		dynamicSurfaceSeparated && staticSurfaceSeparated &&
		(!interactionGateActive ||
		 (interaction.initialized && interaction.collisionTelemetryEnabled &&
		  interaction.nativeIslandSteps > 0 &&
		  metrics.sceneDynamicActorAdded == 1 && dynamicFalling &&
		  metrics.sceneDynamicActorRemoved == 1 &&
		  metrics.sceneDynamicActorReleased == 1));
	const bool volumeHealthPassed = isVolumeHealthWithinLimits(
		metrics.nonFiniteParticleSamples, metrics.invertedElementSamples,
		metrics.minDetF, metrics.maxDetF, metrics.minBodyVolumeRatio,
		metrics.maxBodyVolumeRatio, config.minDetF, config.maxDetF,
		config.minVolumeRatio, config.maxVolumeRatio);

	return metrics.sceneStatics == 2 && metrics.sceneDynamics == 1 &&
		metrics.surfaceTriangles >= config.minSurfaceTriangles &&
		rotationPassed && sphereRollPassed && config.sphereLongRunBounded &&
		staticSurfaceSeparated && mixedInteractionPassed &&
		volumeHealthPassed;
}

SphereLongRollValidationConfig::SphereLongRollValidationConfig()
	: minDetF(0.0f), maxDetF(0.0f), minVolumeRatio(0.0f),
	  maxVolumeRatio(0.0f), rollingKinematicsValid(false),
	  longRunRegressionPassed(false)
{
}

bool isSphereLongRollResultValid(
	const DeformableVolumeMetrics& metrics,
	const VolumeBodyHealthMetrics& sphereHealth,
	const SphereLongRollValidationConfig& config)
{
	const bool rotationGateActive = metrics.completedFrames >= 600;
	const bool sphereRollPassed =
		!rotationGateActive || config.rollingKinematicsValid;
	const bool volumeHealthPassed = metrics.invertedElementSamples == 0 &&
		PxIsFinite(sphereHealth.minDetF) &&
		sphereHealth.minDetF > config.minDetF &&
		PxIsFinite(sphereHealth.maxDetF) &&
		sphereHealth.maxDetF < config.maxDetF &&
		PxIsFinite(sphereHealth.minVolumeRatio) &&
		sphereHealth.minVolumeRatio > config.minVolumeRatio &&
		PxIsFinite(sphereHealth.maxVolumeRatio) &&
		sphereHealth.maxVolumeRatio < config.maxVolumeRatio;

	return metrics.sceneStatics == 1 && metrics.sceneDynamics == 0 &&
		metrics.sceneDeformableVolumes == 2 &&
		metrics.surfaceTriangles == 576 && sphereRollPassed &&
		config.longRunRegressionPassed && volumeHealthPassed;
}

SoftSoftGlancingValidationConfig::SoftSoftGlancingValidationConfig()
	: gateMinFrames(0), minDeltaSpeed(0.0f), minDetF(0.0f),
	  maxDetF(0.0f), minVolumeRatio(0.0f), maxVolumeRatio(0.0f)
{
}

bool isSoftSoftGlancingResultValid(
	const DeformableVolumeMetrics& metrics,
	const SoftContactPhaseMetrics& phase,
	const SoftSoftGlancingValidationConfig& config)
{
	const bool contactGateActive =
		metrics.completedFrames >= config.gateMinFrames;
	const PxReal deltaAngularSpeed = PxMax(
		0.0f, phase.peakSoftContactAngularSpeed - phase.preSoftAngularSpeed);
	const PxVec3 deltaAngularVelocity =
		phase.peakSoftContactAngularVelocity - phase.preSoftAngularVelocity;
	const bool contactPhasePassed = !contactGateActive ||
		(phase.initialized && phase.contactTelemetryEnabled &&
		 phase.hasPreSoftContactSample &&
		 phase.firstSoftContactFrame != PX_MAX_U32 &&
		 phase.lastSoftContactFrame >= phase.firstSoftContactFrame &&
		 phase.softContactFrames > 0 && phase.generatedGroundContacts == 0 &&
		 phase.generatedSoftContacts > 0 &&
		 phase.peakSoftContactFrame >= phase.firstSoftContactFrame &&
		 deltaAngularVelocity.isFinite() &&
		 deltaAngularVelocity.magnitude() >= config.minDeltaSpeed &&
		 deltaAngularSpeed >= config.minDeltaSpeed);
	const bool volumeHealthPassed = metrics.invertedElementSamples == 0 &&
		PxIsFinite(metrics.minDetF) && metrics.minDetF > config.minDetF &&
		PxIsFinite(metrics.maxDetF) && metrics.maxDetF < config.maxDetF &&
		PxIsFinite(metrics.minBodyVolumeRatio) &&
		metrics.minBodyVolumeRatio > config.minVolumeRatio &&
		PxIsFinite(metrics.maxBodyVolumeRatio) &&
		metrics.maxBodyVolumeRatio < config.maxVolumeRatio;

	return metrics.sceneStatics == 0 && metrics.sceneDynamics == 0 &&
		metrics.sceneDeformableVolumes == 2 &&
		metrics.surfaceTriangles == 576 && contactPhasePassed &&
		volumeHealthPassed;
}

SoftSoftTorqueValidationConfig::SoftSoftTorqueValidationConfig()
	: gateMinFrames(0), minRetentionSamples(0), minLeverArm(0.0f),
	  minAngularMomentum(0.0f), minAngularSpeed(0.0f), minDetF(0.0f),
	  maxDetF(0.0f), minVolumeRatio(0.0f), maxVolumeRatio(0.0f)
{
}

bool isSoftSoftTorqueResultValid(
	const DeformableVolumeMetrics& metrics,
	const SoftSoftTorqueMetrics& torque,
	const SoftSoftTorqueValidationConfig& config)
{
	const bool torqueGateActive =
		metrics.completedFrames >= config.gateMinFrames;
	const bool rotationPassed = !torqueGateActive ||
		(torque.softContactFrames > 0 &&
		 torque.firstContactFrame != PX_MAX_U32 &&
		 torque.firstRotationFrame != PX_MAX_U32 &&
		 torque.firstRotationFrame >= torque.firstContactFrame &&
		 torque.firstContactCentroidLeverArm >= config.minLeverArm &&
		 torque.maxAngularMomentum >= config.minAngularMomentum &&
		 torque.maxAngularSpeed >= config.minAngularSpeed &&
		 torque.retainedRotationSamples >= config.minRetentionSamples);
	const bool volumeHealthPassed =
		metrics.nonFiniteParticleSamples == 0 &&
		metrics.invertedElementSamples == 0 &&
		PxIsFinite(metrics.minDetF) && metrics.minDetF > config.minDetF &&
		PxIsFinite(metrics.maxDetF) && metrics.maxDetF < config.maxDetF &&
		PxIsFinite(metrics.minBodyVolumeRatio) &&
		metrics.minBodyVolumeRatio > config.minVolumeRatio &&
		PxIsFinite(metrics.maxBodyVolumeRatio) &&
		metrics.maxBodyVolumeRatio < config.maxVolumeRatio;
	const bool secondaryLifecyclePassed =
		metrics.sceneSecondVolumeActorCreated == 1 &&
		metrics.sceneSecondVolumeHostBuffersInitialized == 1 &&
		metrics.sceneSecondVolumeActorAdded == 1 &&
		metrics.sceneSecondVolumeActorRemoved == 1 &&
		metrics.sceneSecondVolumeActorReleased == 1 &&
		metrics.sceneSecondVolumeBoundsFinite == 1;

	return metrics.sceneStatics == 0 && metrics.sceneDynamics == 0 &&
		metrics.sceneDeformableVolumes == 2 && secondaryLifecyclePassed &&
		torque.initialized && torque.isolatedConfiguration == 1 &&
		torque.targetDistinctCollisionSimulation == 1 &&
		torque.driverDistinctCollisionSimulation == 1 &&
		torque.supportExpansionInstrumentationAvailable == 0 &&
		torque.generatedGroundContacts == 0 &&
		torque.generatedRigidContacts == 0 &&
		torque.generatedSelfContacts == 0 && metrics.groundContactFrames == 0 &&
		metrics.rigidContactFrames == 0 && rotationPassed &&
		volumeHealthPassed;
}

bool SoftSoftTorqueMonitor::initialize(
	PxDeformableVolume& driver, PxDeformableVolume& target,
	bool isolatedConfiguration,
	bool supportExpansionInstrumentationAvailable)
{
	releaseSources();
	mMetrics = SoftSoftTorqueMetrics();
	const PxTetrahedronMesh* const driverSimulationMesh =
		driver.getSimulationMesh();
	const PxTetrahedronMesh* const driverCollisionMesh =
		driver.getCollisionMesh();
	const PxTetrahedronMesh* const targetSimulationMesh =
		target.getSimulationMesh();
	const PxTetrahedronMesh* const targetCollisionMesh =
		target.getCollisionMesh();
	if(!driverSimulationMesh || !driverCollisionMesh ||
		!targetSimulationMesh || !targetCollisionMesh)
		return false;

	mDriver = &driver;
	mTarget = &target;
	mMetrics.targetSimulationVertices =
		targetSimulationMesh->getNbVertices();
	mMetrics.targetCollisionVertices = targetCollisionMesh->getNbVertices();
	mMetrics.driverSimulationVertices =
		driverSimulationMesh->getNbVertices();
	mMetrics.driverCollisionVertices = driverCollisionMesh->getNbVertices();
	mMetrics.targetDistinctCollisionSimulation =
		targetCollisionMesh != targetSimulationMesh ? 1u : 0u;
	mMetrics.driverDistinctCollisionSimulation =
		driverCollisionMesh != driverSimulationMesh ? 1u : 0u;
	mMetrics.isolatedConfiguration = isolatedConfiguration ? 1u : 0u;
	mMetrics.supportExpansionInstrumentationAvailable =
		supportExpansionInstrumentationAvailable ? 1u : 0u;
	mMetrics.initialized = mMetrics.targetSimulationVertices > 0 &&
		mMetrics.targetCollisionVertices > 0 &&
		mMetrics.driverSimulationVertices > 0 &&
		mMetrics.driverCollisionVertices > 0;
	if(!mMetrics.initialized)
	{
		releaseSources();
		return false;
	}
	return true;
}

static bool measureUnweightedVolumeCentroid(
	PxDeformableVolume& volume, PxVec3& centroid)
{
	const PxTetrahedronMesh* const simulationMesh =
		volume.getSimulationMesh();
	const PxVec4* const positions = volume.getSimPositionInvMassBufferH();
	if(!simulationMesh || !positions || simulationMesh->getNbVertices() == 0)
		return false;
	centroid = PxVec3(0.0f);
	for(PxU32 vertexIndex = 0;
		vertexIndex < simulationMesh->getNbVertices(); ++vertexIndex)
	{
		const PxVec3 position = positions[vertexIndex].getXYZ();
		if(!position.isFinite())
			return false;
		centroid += position;
	}
	centroid *= 1.0f / PxReal(simulationMesh->getNbVertices());
	return centroid.isFinite();
}

bool SoftSoftTorqueMonitor::sample(
	PxScene& scene, PxU32 completedFrame, PxReal minAngularMomentum,
	PxReal minAngularSpeed, SoftSoftTorqueFrameSample& sample)
{
	sample = SoftSoftTorqueFrameSample();
	if(!mMetrics.initialized || !mDriver || !mTarget ||
		!PxIsFinite(minAngularMomentum) || minAngularMomentum < 0.0f ||
		!PxIsFinite(minAngularSpeed) || minAngularSpeed < 0.0f)
		return false;

	PxSimulationStatistics sceneStatistics;
	scene.getSimulationStatistics(sceneStatistics);
	sample.generatedGroundContacts =
		sceneStatistics.avbdCpuSoftBodyCollisionGeneratedGroundContacts;
	sample.generatedRigidContacts =
		sceneStatistics.avbdCpuSoftBodyCollisionGeneratedRigidContacts;
	sample.generatedSoftContacts =
		sceneStatistics.avbdCpuSoftBodyCollisionGeneratedSoftContacts;
	sample.generatedSelfContacts =
		sceneStatistics.avbdCpuSoftBodyCollisionGeneratedSelfContacts;
	mMetrics.generatedGroundContacts += sample.generatedGroundContacts;
	mMetrics.generatedRigidContacts += sample.generatedRigidContacts;
	mMetrics.generatedSoftContacts += sample.generatedSoftContacts;
	mMetrics.generatedSelfContacts += sample.generatedSelfContacts;
	if(sample.generatedSoftContacts)
		mMetrics.softContactFrames++;

	PxVec3 targetCentroid(0.0f);
	PxVec3 angularMomentum(0.0f);
	PxVec3 angularVelocity(0.0f);
	if(!measureVolumeAngularState(
			*mTarget, targetCentroid, angularMomentum, angularVelocity))
		return false;
	PxVec3 driverCentroid(0.0f);
	if(!measureUnweightedVolumeCentroid(*mDriver, driverCentroid))
		return false;
	const PxVec3 centroidDelta = driverCentroid - targetCentroid;
	const PxReal transverseLeverArm = PxSqrt(
		centroidDelta.y * centroidDelta.y +
		centroidDelta.z * centroidDelta.z);
	const PxReal angularMomentumMagnitude = angularMomentum.magnitude();
	const PxReal angularSpeed = angularVelocity.magnitude();
	if(!PxIsFinite(transverseLeverArm) ||
		!PxIsFinite(angularMomentumMagnitude) || !PxIsFinite(angularSpeed))
		return false;
	mMetrics.maxCentroidLeverArm = PxMax(
		mMetrics.maxCentroidLeverArm, transverseLeverArm);
	mMetrics.maxAngularMomentum = PxMax(
		mMetrics.maxAngularMomentum, angularMomentumMagnitude);
	mMetrics.maxAngularSpeed = PxMax(mMetrics.maxAngularSpeed, angularSpeed);
	mMetrics.finalAngularMomentum = angularMomentumMagnitude;
	mMetrics.finalAngularSpeed = angularSpeed;
	if(sample.generatedSoftContacts &&
		mMetrics.firstContactFrame == PX_MAX_U32)
	{
		mMetrics.firstContactFrame = completedFrame;
		mMetrics.firstContactCentroidLeverArm = transverseLeverArm;
	}
	if(mMetrics.firstContactFrame != PX_MAX_U32 &&
		mMetrics.firstRotationFrame == PX_MAX_U32 &&
		angularMomentumMagnitude >= minAngularMomentum &&
		angularSpeed >= minAngularSpeed)
		mMetrics.firstRotationFrame = completedFrame;
	if(mMetrics.firstRotationFrame != PX_MAX_U32 &&
		angularSpeed >= minAngularSpeed * 0.25f)
		mMetrics.retainedRotationSamples++;

	const PxBounds3 driverBounds = mDriver->getWorldBounds();
	const PxBounds3 targetBounds = mTarget->getWorldBounds();
	sample.driverBoundsFinite = driverBounds.isValid() &&
		driverBounds.minimum.isFinite() && driverBounds.maximum.isFinite();
	sample.targetBoundsFinite = targetBounds.isValid() &&
		targetBounds.minimum.isFinite() && targetBounds.maximum.isFinite();
	return true;
}

GroundEmbeddedTetProbeMetrics::GroundEmbeddedTetProbeMetrics()
	: simulationVertices(0), collisionVertices(0),
	  simulationTetrahedra(0), collisionTetrahedra(0),
	  distinctCollisionSimulation(0), strictInteriorEmbedding(0),
	  selfCollisionDisabled(0), speculativeCcdDisabled(0),
	  contactTelemetryEnabled(0), hasPreGroundSample(0),
	  groundContactWindowClosed(0),
	  firstGroundContactFrame(PX_MAX_U32),
	  lastGroundContactFrame(PX_MAX_U32),
	  peakGroundRollFrame(PX_MAX_U32), groundContactWindowFrames(0),
	  generatedGroundContacts(0), generatedRigidContacts(0),
	  generatedSoftContacts(0), generatedSelfContacts(0),
	  launchSpeed(0.0f), initialMass(0.0f), initialRmsRadius(0.0f),
	  rollAxis(0.0f), preGroundAngularMomentum(0.0f),
	  preGroundAngularVelocity(0.0f),
	  peakDeltaAngularMomentum(0.0f), peakDeltaAngularVelocity(0.0f),
	  peakExpectedRollAngularMomentum(0.0f),
	  peakExpectedRollAngularSpeed(0.0f),
	  peakNormalizedRollMomentum(0.0f), peakNormalizedRollOmega(0.0f),
	  initialized(false)
{
}

bool isGroundEmbeddedTetResultValid(
	const DeformableVolumeMetrics& metrics,
	const GroundEmbeddedTetProbeMetrics& probeMetrics,
	PxReal minNormalizedRoll)
{
	const bool volumeHealthPassed =
		metrics.nonFiniteParticleSamples == 0 &&
		metrics.invertedElementSamples == 0 &&
		PxIsFinite(metrics.minDetF) && metrics.minDetF > 0.05f &&
		PxIsFinite(metrics.maxDetF) && metrics.maxDetF < 20.0f &&
		PxIsFinite(metrics.minBodyVolumeRatio) &&
		metrics.minBodyVolumeRatio > 0.05f &&
		PxIsFinite(metrics.maxBodyVolumeRatio) &&
		metrics.maxBodyVolumeRatio < 20.0f;
	const bool rotationEvidence =
		probeMetrics.contactTelemetryEnabled == 1 &&
		probeMetrics.hasPreGroundSample == 1 &&
		probeMetrics.firstGroundContactFrame != PX_MAX_U32 &&
		probeMetrics.lastGroundContactFrame >=
			probeMetrics.firstGroundContactFrame &&
		probeMetrics.peakGroundRollFrame >=
			probeMetrics.firstGroundContactFrame &&
		probeMetrics.peakGroundRollFrame <=
			probeMetrics.lastGroundContactFrame &&
		probeMetrics.groundContactWindowFrames > 0 &&
		probeMetrics.generatedGroundContacts > 0 &&
		probeMetrics.generatedRigidContacts == 0 &&
		probeMetrics.generatedSoftContacts == 0 &&
		probeMetrics.generatedSelfContacts == 0 &&
		probeMetrics.initialMass > 0.0f &&
		probeMetrics.initialRmsRadius > 0.0f &&
		probeMetrics.launchSpeed > 0.0f &&
		probeMetrics.preGroundAngularMomentum.isFinite() &&
		probeMetrics.preGroundAngularVelocity.isFinite() &&
		probeMetrics.peakDeltaAngularMomentum.isFinite() &&
		probeMetrics.peakDeltaAngularVelocity.isFinite() &&
		PxIsFinite(probeMetrics.peakNormalizedRollMomentum) &&
		PxIsFinite(probeMetrics.peakNormalizedRollOmega) &&
		probeMetrics.peakNormalizedRollMomentum > minNormalizedRoll &&
		probeMetrics.peakNormalizedRollOmega > minNormalizedRoll;

	return metrics.sceneStatics == 1 && metrics.sceneDynamics == 0 &&
		metrics.sceneDeformableVolumes == 1 && metrics.particles == 4 &&
		metrics.softBodies == 1 && metrics.tetElements == 1 &&
		metrics.surfaceTriangles == 4 && probeMetrics.initialized &&
		probeMetrics.distinctCollisionSimulation == 1 &&
		probeMetrics.strictInteriorEmbedding == 1 &&
		probeMetrics.selfCollisionDisabled == 1 &&
		probeMetrics.speculativeCcdDisabled == 1 &&
		metrics.groundContactFrames > 0 && metrics.maxGroundContacts > 0 &&
		metrics.rigidContactFrames == 0 && metrics.softContactFrames == 0 &&
		metrics.maxCentroidDrop > 0.2f && rotationEvidence &&
		volumeHealthPassed;
}

GroundEmbeddedTetProbeFrameSample::GroundEmbeddedTetProbeFrameSample()
	: generatedGroundContacts(0), generatedRigidContacts(0),
	  generatedSoftContacts(0), generatedSelfContacts(0)
{
}

bool sampleGroundEmbeddedTetProbe(
	PxScene& scene, PxDeformableVolume& volume, PxU32 completedFrame,
	GroundEmbeddedTetProbeMetrics& metrics,
	GroundEmbeddedTetProbeFrameSample& sample)
{
	sample = GroundEmbeddedTetProbeFrameSample();
	if(!metrics.initialized || !metrics.contactTelemetryEnabled ||
		!metrics.rollAxis.isFinite() ||
		metrics.rollAxis.magnitudeSquared() < 0.999f ||
		metrics.launchSpeed <= 0.0f || !PxIsFinite(metrics.launchSpeed) ||
		metrics.initialMass <= 0.0f || !PxIsFinite(metrics.initialMass) ||
		metrics.initialRmsRadius <= 0.0f ||
		!PxIsFinite(metrics.initialRmsRadius))
		return false;

	PxVec3 centroid(0.0f);
	PxVec3 angularMomentum(0.0f);
	PxVec3 angularVelocity(0.0f);
	if(!measureVolumeAngularState(
			volume, centroid, angularMomentum, angularVelocity))
		return false;

	PxSimulationStatistics sceneStatistics;
	scene.getSimulationStatistics(sceneStatistics);
	sample.generatedGroundContacts =
		sceneStatistics.avbdCpuSoftBodyCollisionGeneratedGroundContacts;
	sample.generatedRigidContacts =
		sceneStatistics.avbdCpuSoftBodyCollisionGeneratedRigidContacts;
	sample.generatedSoftContacts =
		sceneStatistics.avbdCpuSoftBodyCollisionGeneratedSoftContacts;
	sample.generatedSelfContacts =
		sceneStatistics.avbdCpuSoftBodyCollisionGeneratedSelfContacts;
	metrics.generatedGroundContacts += sample.generatedGroundContacts;
	metrics.generatedRigidContacts += sample.generatedRigidContacts;
	metrics.generatedSoftContacts += sample.generatedSoftContacts;
	metrics.generatedSelfContacts += sample.generatedSelfContacts;

	if(sample.generatedGroundContacts == 0)
	{
		if(metrics.firstGroundContactFrame == PX_MAX_U32)
		{
			metrics.hasPreGroundSample = 1;
			metrics.preGroundAngularMomentum = angularMomentum;
			metrics.preGroundAngularVelocity = angularVelocity;
		}
		else
			metrics.groundContactWindowClosed = 1;
		return true;
	}

	if(metrics.firstGroundContactFrame == PX_MAX_U32)
	{
		if(!metrics.hasPreGroundSample)
			return false;
		metrics.firstGroundContactFrame = completedFrame;
	}
	if(metrics.groundContactWindowClosed)
		return true;

	metrics.groundContactWindowFrames++;
	metrics.lastGroundContactFrame = completedFrame;
	const PxVec3 deltaAngularMomentum =
		angularMomentum - metrics.preGroundAngularMomentum;
	const PxVec3 deltaAngularVelocity =
		angularVelocity - metrics.preGroundAngularVelocity;
	const PxReal expectedMomentum =
		deltaAngularMomentum.dot(metrics.rollAxis);
	const PxReal expectedOmega = deltaAngularVelocity.dot(metrics.rollAxis);
	if(!deltaAngularMomentum.isFinite() ||
		!deltaAngularVelocity.isFinite() || !PxIsFinite(expectedMomentum) ||
		!PxIsFinite(expectedOmega))
		return false;
	if(expectedOmega > metrics.peakExpectedRollAngularSpeed)
	{
		metrics.peakExpectedRollAngularMomentum = expectedMomentum;
		metrics.peakExpectedRollAngularSpeed = expectedOmega;
		metrics.peakDeltaAngularMomentum = deltaAngularMomentum;
		metrics.peakDeltaAngularVelocity = deltaAngularVelocity;
		metrics.peakGroundRollFrame = completedFrame;
		metrics.peakNormalizedRollOmega = expectedOmega *
			metrics.initialRmsRadius / metrics.launchSpeed;
		metrics.peakNormalizedRollMomentum = expectedMomentum /
			(metrics.initialMass * metrics.initialRmsRadius *
				metrics.launchSpeed);
	}
	return PxIsFinite(metrics.peakNormalizedRollOmega) &&
		PxIsFinite(metrics.peakNormalizedRollMomentum);
}

ReverseFeatureMetrics::ReverseFeatureMetrics()
	: faceResponseObserved(0), vertexSdfExcluded(0),
	  negativeControlPassed(0), nonFiniteSamples(0),
	  positiveDisplacement(0.0f), positiveDrop(0.0f),
	  negativeDrop(0.0f), faceSeparation(PX_MAX_F32),
	  minimumVertexSeparation(PX_MAX_F32)
{
}

bool updateReverseFeatureMetrics(
	ReverseFeatureMetrics& metrics, const PxVec3& positiveCentroid,
	const PxVec3& negativeCentroid,
	const PxVec3& positiveInitialCentroid,
	const PxVec3& negativeInitialCentroid, PxReal faceSeparation,
	PxReal minimumVertexSeparation)
{
	if(!positiveCentroid.isFinite() || !negativeCentroid.isFinite() ||
		!positiveInitialCentroid.isFinite() ||
		!negativeInitialCentroid.isFinite() ||
		!PxIsFinite(faceSeparation) || !PxIsFinite(minimumVertexSeparation))
	{
		metrics.nonFiniteSamples++;
		return false;
	}
	metrics.positiveDisplacement =
		(positiveCentroid - positiveInitialCentroid).magnitude();
	metrics.positiveDrop = positiveInitialCentroid.y - positiveCentroid.y;
	metrics.negativeDrop = negativeInitialCentroid.y - negativeCentroid.y;
	metrics.faceSeparation = faceSeparation;
	metrics.minimumVertexSeparation = minimumVertexSeparation;
	metrics.vertexSdfExcluded = minimumVertexSeparation > 0.10f ? 1u : 0u;
	metrics.negativeControlPassed = metrics.negativeDrop > 0.02f ? 1u : 0u;
	metrics.faceResponseObserved =
		metrics.positiveDisplacement > 1.0e-3f &&
		faceSeparation > 0.02f &&
		metrics.positiveDrop + 0.01f < metrics.negativeDrop ? 1u : 0u;
	return true;
}

bool isReverseFeatureResponseValid(const ReverseFeatureMetrics& metrics)
{
	return metrics.faceResponseObserved == 1 &&
		metrics.vertexSdfExcluded == 1 &&
		metrics.negativeControlPassed == 1 &&
		metrics.nonFiniteSamples == 0 &&
		PxIsFinite(metrics.positiveDisplacement) &&
		metrics.positiveDisplacement > 1.0e-3f &&
		PxIsFinite(metrics.positiveDrop) &&
		PxIsFinite(metrics.negativeDrop) && metrics.negativeDrop > 0.02f &&
		metrics.positiveDrop + 0.01f < metrics.negativeDrop &&
		PxIsFinite(metrics.faceSeparation) &&
		metrics.faceSeparation > 0.02f &&
		PxIsFinite(metrics.minimumVertexSeparation) &&
		metrics.minimumVertexSeparation > 0.10f;
}

ReverseSweptMetrics::ReverseSweptMetrics()
	: responseObserved(0), negativeControlPassed(0),
	  twoSidedResponseObserved(0), vertexSweepExcluded(0),
	  nonFiniteSamples(0), positiveDisplacement(0.0f),
	  negativeDisplacement(0.0f), positiveDrop(0.0f),
	  negativeDrop(0.0f), positiveRigidDrop(0.0f),
	  negativeRigidDrop(0.0f), faceSeparation(PX_MAX_F32),
	  minimumVertexSweepSeparation(PX_MAX_F32)
{
}

DeformingReverseSweptMetrics::DeformingReverseSweptMetrics()
	: geometricSweepIsolated(0), endpointMinSeparation(PX_MAX_F32),
	  midSweepMinSeparation(PX_MAX_F32), responseDelta(0.0f)
{
}

RotationalSweepMetrics::RotationalSweepMetrics()
	: sweepIsolated(0), nonFiniteSamples(0),
	  endpointMinSeparation(PX_MAX_F32),
	  midSweepMinSeparation(PX_MAX_F32), positiveAngularTravel(0.0f),
	  negativeAngularTravel(0.0f)
{
}

TriangleSurfaceSweptValidationConfig::
	TriangleSurfaceSweptValidationConfig()
	: reverse(false), heightField(false), rotational(false)
{
}

static bool isPrescribedRotationalSweepValid(
	const RotationalSweepMetrics& metrics,
	PxReal minimumEndpointSeparation)
{
	return metrics.sweepIsolated == 1 && metrics.nonFiniteSamples == 0 &&
		PxIsFinite(metrics.endpointMinSeparation) &&
		metrics.endpointMinSeparation > minimumEndpointSeparation &&
		PxIsFinite(metrics.midSweepMinSeparation) &&
		PxIsFinite(metrics.positiveAngularTravel) &&
		PxIsFinite(metrics.negativeAngularTravel) &&
		PxAbs(metrics.positiveAngularTravel - 2.0f * PxPi / 3.0f) <
			0.002f &&
		PxAbs(metrics.negativeAngularTravel - 2.0f * PxPi / 3.0f) <
			0.002f;
}

bool isTriangleSurfaceSweptResponseValid(
	const ReverseSweptMetrics& metrics,
	const RotationalSweepMetrics& rotationalMetrics,
	const TriangleSurfaceSweptValidationConfig& config)
{
	const bool commonPassed =
		metrics.responseObserved == 1 &&
		metrics.negativeControlPassed == 1 &&
		metrics.twoSidedResponseObserved == 1 &&
		metrics.vertexSweepExcluded == 1 &&
		metrics.nonFiniteSamples == 0 &&
		PxIsFinite(metrics.positiveDisplacement) &&
		PxIsFinite(metrics.negativeDisplacement) &&
		PxIsFinite(metrics.positiveDrop) &&
		PxIsFinite(metrics.negativeDrop) &&
		(!config.reverse ||
			(PxIsFinite(metrics.minimumVertexSweepSeparation) &&
			 metrics.minimumVertexSweepSeparation >
				(config.rotational ? 0.10f : 0.05f))) &&
		(!config.rotational ||
			isPrescribedRotationalSweepValid(rotationalMetrics, 0.10f));
	if(!commonPassed)
		return false;

	const PxReal minimumPositiveDisplacement =
		config.rotational ? 2.0e-3f : (config.reverse ? 5.0e-3f : 0.02f);
	return metrics.positiveDisplacement > minimumPositiveDisplacement &&
		metrics.negativeDisplacement < 1.0e-2f;
}

ReverseSweptValidationConfig::ReverseSweptValidationConfig()
	: staticTarget(false), kinematicTarget(false), dynamicTarget(false),
	  deforming(false), capsule(false), convex(false), rotational(false)
{
}

bool normalizeDynamicReverseSweptResponse(ReverseSweptMetrics& metrics)
{
	if(metrics.responseObserved != 0 ||
		metrics.negativeDisplacement >= 5.0e-3f ||
		metrics.faceSeparation <= -0.15f ||
		metrics.positiveRigidDrop + 0.05f >= metrics.negativeRigidDrop)
		return false;
	metrics.responseObserved = 1;
	return true;
}

bool isReverseSweptResponseValid(
	const ReverseSweptMetrics& metrics,
	const DeformingReverseSweptMetrics& deformingMetrics,
	const RotationalSweepMetrics& rotationalMetrics,
	const ReverseSweptValidationConfig& config)
{
	const PxReal minimumVertexSeparation = config.rotational
		? 0.10f : ((config.capsule || config.convex) ? 0.05f : 0.10f);
	const bool commonPassed =
		metrics.responseObserved == 1 &&
		metrics.negativeControlPassed == 1 &&
		metrics.twoSidedResponseObserved == 1 &&
		metrics.vertexSweepExcluded == 1 &&
		metrics.nonFiniteSamples == 0 &&
		PxIsFinite(metrics.positiveDisplacement) &&
		PxIsFinite(metrics.negativeDisplacement) &&
		PxIsFinite(metrics.positiveDrop) &&
		PxIsFinite(metrics.negativeDrop) &&
		PxIsFinite(metrics.positiveRigidDrop) &&
		PxIsFinite(metrics.negativeRigidDrop) &&
		PxIsFinite(metrics.faceSeparation) &&
		metrics.faceSeparation > -0.15f &&
		PxIsFinite(metrics.minimumVertexSweepSeparation) &&
		metrics.minimumVertexSweepSeparation > minimumVertexSeparation;
	if(!commonPassed)
		return false;

	if(config.deforming &&
		!(deformingMetrics.geometricSweepIsolated == 1 &&
		  PxIsFinite(deformingMetrics.endpointMinSeparation) &&
		  PxIsFinite(deformingMetrics.midSweepMinSeparation) &&
		  PxIsFinite(deformingMetrics.responseDelta) &&
		  deformingMetrics.responseDelta > 0.01f))
		return false;

	if(config.rotational &&
		!(rotationalMetrics.sweepIsolated == 1 &&
		  rotationalMetrics.nonFiniteSamples == 0 &&
		  PxIsFinite(rotationalMetrics.endpointMinSeparation) &&
		  PxIsFinite(rotationalMetrics.midSweepMinSeparation) &&
		  PxIsFinite(rotationalMetrics.positiveAngularTravel) &&
		  PxIsFinite(rotationalMetrics.negativeAngularTravel) &&
		  (!config.kinematicTarget ||
			(PxAbs(rotationalMetrics.positiveAngularTravel -
				2.0f * PxPi / 3.0f) < 0.002f &&
			 PxAbs(rotationalMetrics.negativeAngularTravel -
				2.0f * PxPi / 3.0f) < 0.002f))))
		return false;

	if(config.staticTarget)
	{
		const PxReal minimumNegativeDrop = config.deforming ? 0.15f : 0.8f;
		const PxReal responseMargin = config.deforming ? 0.01f : 0.03f;
		return metrics.negativeDrop > minimumNegativeDrop &&
			metrics.positiveDrop + responseMargin < metrics.negativeDrop;
	}

	const PxReal minimumPositiveDisplacement =
		config.rotational ? 0.02f : (config.dynamicTarget ? 0.01f : 0.02f);
	const bool responsePassed =
		metrics.positiveDisplacement > minimumPositiveDisplacement ||
		(config.dynamicTarget &&
		 metrics.positiveRigidDrop + 0.05f < metrics.negativeRigidDrop);
	if(!responsePassed || metrics.negativeDisplacement >= 5.0e-3f)
		return false;

	return !config.dynamicTarget ||
		(metrics.negativeRigidDrop > (config.rotational ? 0.8f : 1.5f) &&
		 metrics.positiveRigidDrop + 0.05f < metrics.negativeRigidDrop);
}

KinematicFiniteSweptMetrics::KinematicFiniteSweptMetrics()
	: targetIssued(0), responseObserved(0), negativeControlPassed(0),
	  positiveDisplacement(0.0f), negativeDisplacement(0.0f),
	  positiveMinSeparation(PX_MAX_F32)
{
}

bool isKinematicFiniteSweptResponseValid(
	const KinematicFiniteSweptMetrics& metrics,
	const RotationalSweepMetrics& rotationalMetrics,
	bool rotational, bool convex)
{
	const bool commonPassed = metrics.targetIssued == 1 &&
		metrics.responseObserved == 1 && metrics.negativeControlPassed == 1 &&
		PxIsFinite(metrics.positiveDisplacement) &&
		metrics.positiveDisplacement > 0.02f &&
		PxIsFinite(metrics.negativeDisplacement) &&
		metrics.negativeDisplacement < 5.0e-3f &&
		PxIsFinite(metrics.positiveMinSeparation) &&
		metrics.positiveMinSeparation > -0.10f &&
		metrics.positiveMinSeparation < PX_MAX_F32;
	if(!commonPassed || !rotational)
		return commonPassed;

	return rotationalMetrics.sweepIsolated == 1 &&
		rotationalMetrics.nonFiniteSamples == 0 &&
		PxIsFinite(rotationalMetrics.endpointMinSeparation) &&
		rotationalMetrics.endpointMinSeparation > 0.05f &&
		PxIsFinite(rotationalMetrics.midSweepMinSeparation) &&
		rotationalMetrics.midSweepMinSeparation <
			(convex ? 1.0e-5f : -0.05f);
}

DynamicFiniteSweptMetrics::DynamicFiniteSweptMetrics()
	: launched(0), responseObserved(0), negativeControlPassed(0),
	  twoSidedResponseObserved(0), positiveSoftDisplacement(0.0f),
	  negativeSoftDisplacement(0.0f), positiveRigidDrop(0.0f),
	  negativeRigidDrop(0.0f), positiveMinSeparation(PX_MAX_F32)
{
}

bool normalizeDynamicFiniteSweptResponse(
	DynamicFiniteSweptMetrics& metrics)
{
	if(metrics.responseObserved != 0 ||
		metrics.negativeSoftDisplacement >= 5.0e-3f ||
		metrics.positiveMinSeparation <= -0.15f ||
		metrics.positiveRigidDrop + 0.05f >= metrics.negativeRigidDrop)
		return false;
	metrics.responseObserved = 1;
	return true;
}

bool isDynamicFiniteSweptResponseValid(
	const DynamicFiniteSweptMetrics& metrics,
	const RotationalSweepMetrics& rotationalMetrics,
	bool rotational, bool convex)
{
	const bool rigidResponse =
		metrics.positiveRigidDrop + 0.05f < metrics.negativeRigidDrop;
	const bool softResponse =
		PxIsFinite(metrics.positiveSoftDisplacement) &&
		metrics.positiveSoftDisplacement > 0.02f;
	const bool responsePassed =
		rotational ? softResponse : (softResponse || rigidResponse);
	const bool commonPassed = metrics.launched == 1 &&
		metrics.responseObserved == 1 && metrics.negativeControlPassed == 1 &&
		metrics.twoSidedResponseObserved == 1 && responsePassed &&
		PxIsFinite(metrics.negativeSoftDisplacement) &&
		metrics.negativeSoftDisplacement < 5.0e-3f &&
		PxIsFinite(metrics.positiveRigidDrop) &&
		PxIsFinite(metrics.negativeRigidDrop) &&
		metrics.negativeRigidDrop > (rotational ? 0.8f : 1.5f) &&
		rigidResponse && PxIsFinite(metrics.positiveMinSeparation) &&
		metrics.positiveMinSeparation > -0.15f &&
		metrics.positiveMinSeparation < PX_MAX_F32;
	if(!commonPassed || !rotational)
		return commonPassed;

	return rotationalMetrics.sweepIsolated == 1 &&
		rotationalMetrics.nonFiniteSamples == 0 &&
		PxIsFinite(rotationalMetrics.endpointMinSeparation) &&
		rotationalMetrics.endpointMinSeparation > 0.05f &&
		PxIsFinite(rotationalMetrics.midSweepMinSeparation) &&
		rotationalMetrics.midSweepMinSeparation <
			(convex ? 1.0e-5f : -0.05f) &&
		PxIsFinite(rotationalMetrics.positiveAngularTravel) &&
		PxIsFinite(rotationalMetrics.negativeAngularTravel) &&
		rotationalMetrics.negativeAngularTravel > 0.8f &&
		rotationalMetrics.positiveAngularTravel + 0.05f <
			rotationalMetrics.negativeAngularTravel;
}

static bool isSweptSecondaryVolumeLifecycleValid(
	const DeformableVolumeMetrics& metrics)
{
	return metrics.sceneSecondVolumeActorCreated == 1 &&
		metrics.sceneSecondVolumeHostBuffersInitialized == 1 &&
		metrics.sceneSecondVolumeActorAdded == 1 &&
		metrics.sceneActorRemoved == 1 &&
		metrics.sceneSecondVolumeActorRemoved == 1 &&
		metrics.sceneSecondVolumeActorReleased == 1 &&
		metrics.sceneSecondVolumeBoundsFinite == 1;
}

static bool isDynamicTargetPairLifecycleValid(
	const DeformableVolumeMetrics& metrics)
{
	return metrics.sceneStatics == 0 && metrics.sceneDynamics == 2 &&
		metrics.sceneDynamicActorAdded == 1 &&
		metrics.sceneSecondDynamicActorAdded == 1 &&
		metrics.sceneDynamicActorRemoved == 1 &&
		metrics.sceneSecondDynamicActorRemoved == 1 &&
		metrics.sceneDynamicActorReleased == 1 &&
		metrics.sceneSecondDynamicActorReleased == 1;
}

bool isReverseFeatureSceneResultValid(
	const DeformableVolumeMetrics& metrics,
	const ReverseFeatureMetrics& reverseFeature)
{
	return metrics.sceneStatics == 1 && metrics.sceneDynamics == 0 &&
		isSweptSecondaryVolumeLifecycleValid(metrics) &&
		isReverseFeatureResponseValid(reverseFeature);
}

bool isRigidTriangleSteadyContactResultValid(
	const DeformableVolumeMetrics& metrics, PxU32 profiledFrames,
	PxU64 faceTests, PxU64 edgeTests, PxU64 vertexTests)
{
	return metrics.sceneStatics == 1 && metrics.sceneDynamics == 0 &&
		metrics.sceneVolumeTargetBound == 1 &&
		metrics.sceneVolumeTargetMutated == 1 &&
		metrics.sceneActorRemoved == 1 && metrics.sceneActorReleased == 1 &&
		metrics.sceneBoundsFinite == 1 && profiledFrames > 0 &&
		faceTests > 0 && edgeTests > 0 && vertexTests > 0;
}

bool isTriangleSurfaceSweptSceneResultValid(
	const DeformableVolumeMetrics& metrics,
	const ReverseSweptMetrics& sweptMetrics,
	const RotationalSweepMetrics& rotationalMetrics,
	const TriangleSurfaceSweptValidationConfig& config)
{
	return isDynamicTargetPairLifecycleValid(metrics) &&
		isSweptSecondaryVolumeLifecycleValid(metrics) &&
		metrics.speculativeCcdFlagApplied == 1 &&
		metrics.speculativeCcdPreventedTunneling == 1 &&
		metrics.speculativeCcdNegativeControlTunneled == 1 &&
		isTriangleSurfaceSweptResponseValid(
			sweptMetrics, rotationalMetrics, config);
}

bool isReverseSweptSceneResultValid(
	const DeformableVolumeMetrics& metrics,
	const ReverseSweptMetrics& sweptMetrics,
	const DeformingReverseSweptMetrics& deformingMetrics,
	const RotationalSweepMetrics& rotationalMetrics,
	const ReverseSweptValidationConfig& config,
	bool staticTarget)
{
	const bool rigidLifecyclePassed = staticTarget
		? metrics.sceneStatics == 1 && metrics.sceneDynamics == 0
		: isDynamicTargetPairLifecycleValid(metrics);
	return rigidLifecyclePassed &&
		isSweptSecondaryVolumeLifecycleValid(metrics) &&
		metrics.speculativeCcdFlagApplied == 1 &&
		metrics.speculativeCcdPreventedTunneling == 1 &&
		metrics.speculativeCcdNegativeControlTunneled == 1 &&
		isReverseSweptResponseValid(
			sweptMetrics, deformingMetrics, rotationalMetrics, config);
}

bool isKinematicFiniteSweptSceneResultValid(
	const DeformableVolumeMetrics& metrics,
	const KinematicFiniteSweptMetrics& sweptMetrics,
	const RotationalSweepMetrics& rotationalMetrics,
	bool rotational, bool convex)
{
	return isDynamicTargetPairLifecycleValid(metrics) &&
		isSweptSecondaryVolumeLifecycleValid(metrics) &&
		metrics.speculativeCcdFlagApplied == 1 &&
		isKinematicFiniteSweptResponseValid(
			sweptMetrics, rotationalMetrics, rotational, convex);
}

bool isDynamicFiniteSweptSceneResultValid(
	const DeformableVolumeMetrics& metrics,
	const DynamicFiniteSweptMetrics& sweptMetrics,
	const RotationalSweepMetrics& rotationalMetrics,
	bool rotational, bool convex)
{
	return isDynamicTargetPairLifecycleValid(metrics) &&
		isSweptSecondaryVolumeLifecycleValid(metrics) &&
		metrics.speculativeCcdFlagApplied == 1 &&
		isDynamicFiniteSweptResponseValid(
			sweptMetrics, rotationalMetrics, rotational, convex);
}

bool isSpeculativeCcdSceneResultValid(
	const DeformableVolumeMetrics& metrics, bool plane, bool finiteSmooth)
{
	const bool secondaryLifecyclePassed =
		metrics.sceneSecondVolumeActorCreated == 1 &&
		metrics.sceneSecondVolumeHostBuffersInitialized == 1 &&
		metrics.sceneSecondVolumeActorAdded == 1 &&
		metrics.sceneSecondVolumeActorRemoved == 1 &&
		metrics.sceneSecondVolumeActorReleased == 1 &&
		metrics.sceneSecondVolumeBoundsFinite == 1;
	const bool positiveResponsePassed = finiteSmooth
		? PxIsFinite(metrics.speculativeCcdPositiveMinSeparation) &&
			metrics.speculativeCcdPositiveMinSeparation >= -0.05f &&
			metrics.speculativeCcdPositiveMinSeparation < PX_MAX_F32
		: metrics.speculativeCcdPositiveMinY >= (plane ? 0.49f : 0.50f);

	return metrics.sceneStatics == 1 && metrics.sceneDynamics == 0 &&
		secondaryLifecyclePassed && metrics.speculativeCcdFlagApplied == 1 &&
		metrics.speculativeCcdPreventedTunneling == 1 &&
		(plane || metrics.speculativeCcdNegativeControlTunneled == 1) &&
		PxIsFinite(metrics.speculativeCcdPositiveMinY) &&
		positiveResponsePassed &&
		PxIsFinite(metrics.speculativeCcdNegativeMaxY) &&
		(plane || metrics.speculativeCcdNegativeMaxY <= 0.44f);
}

PxReal getCapsuleSignedSeparation(
	const PxVec3& point,
	const PxTransform& capsulePose,
	PxReal capsuleRadius,
	PxReal capsuleHalfHeight)
{
	const PxVec3 localPoint = capsulePose.transformInv(point);
	const PxReal axisCoordinate = PxClamp(
		localPoint.x, -capsuleHalfHeight, capsuleHalfHeight);
	return (
		localPoint - PxVec3(axisCoordinate, 0.0f, 0.0f)).
			magnitude() - capsuleRadius;
}

PxReal getCapsuleSignedSeparation(
	const PxVec3& point,
	const PxVec3& capsuleCenter,
	PxReal capsuleRadius,
	PxReal capsuleHalfHeight)
{
	return getCapsuleSignedSeparation(
		point, PxTransform(capsuleCenter),
		capsuleRadius, capsuleHalfHeight);
}

SweptGeometrySampler::SweptGeometrySampler(
	const PxArray<PxVec3>& reverseInitialPositions,
	const PxArray<PxVec3>& deformingFreeEndPositions,
	const PxArray<PxVec3>& rotationalInitialPositions,
	PxConvexMesh* convexMesh,
	PxTriangleMesh* triangleMesh,
	PxHeightField* heightField)
	: mReverseInitialPositions(reverseInitialPositions),
	  mDeformingFreeEndPositions(deformingFreeEndPositions),
	  mRotationalInitialPositions(rotationalInitialPositions),
	  mConvexMesh(convexMesh),
	  mTriangleMesh(triangleMesh),
	  mHeightField(heightField)
{
}

static PxVec3 closestPointOnTriangleForGate(
	const PxVec3& point,
	const PxVec3& a,
	const PxVec3& b,
	const PxVec3& c)
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

bool SweptGeometrySampler::measureSphereReverseSweptSeparations(
	PxDeformableVolume* volume,
	const PxVec3& sphereCenterCurrent,
	const PxVec3& sphereCenterStart,
	const PxVec3& sphereCenterEnd,
	PxReal sphereRadius,
	PxReal capsuleHalfHeight,
	bool convexCase,
	PxReal& faceSeparation,
	PxReal& minimumVertexSweepSeparation) const
{
	PxReal currentVertexSeparation = PX_MAX_F32;
	if(!measureSmoothReverseSeparations(
			volume, sphereCenterCurrent, sphereRadius,
			capsuleHalfHeight,
			convexCase, false, false,
			faceSeparation, currentVertexSeparation) ||
		!sphereCenterCurrent.isFinite() ||
		!sphereCenterStart.isFinite() ||
		!sphereCenterEnd.isFinite())
		return false;

	const PxTetrahedronMesh* mesh = volume->getCollisionMesh();
	const PxU32 vertexCount = mesh ? mesh->getNbVertices() : 0;
	if(vertexCount == 0 ||
		mReverseInitialPositions.size() !=
			vertexCount)
		return false;
	minimumVertexSweepSeparation = PX_MAX_F32;
	for(PxU32 vertexIndex = 0;
		vertexIndex < vertexCount; ++vertexIndex)
	{
		const PxVec3 initialPosition =
			mReverseInitialPositions[vertexIndex];
		if(!initialPosition.isFinite())
			return false;
		if(convexCase)
		{
			if(!mConvexMesh)
				return false;
			const PxConvexMeshGeometry geometry(
				mConvexMesh);
			for(PxU32 sampleIndex = 0;
				sampleIndex <= 64; ++sampleIndex)
			{
				const PxReal alpha =
					PxReal(sampleIndex) / 64.0f;
				const PxVec3 center =
					sphereCenterStart +
					(sphereCenterEnd - sphereCenterStart) *
						alpha;
				const PxReal squaredDistance =
					PxGeometryQuery::pointDistance(
						initialPosition, geometry,
						PxTransform(center));
				if(!PxIsFinite(squaredDistance) ||
					squaredDistance < 0.0f)
					return false;
				minimumVertexSweepSeparation = PxMin(
					minimumVertexSweepSeparation,
					PxSqrt(squaredDistance));
			}
		}
		else
		{
			const PxVec3 centerPath =
				sphereCenterEnd - sphereCenterStart;
			const PxReal centerPathLengthSq =
				centerPath.magnitudeSquared();
			const PxReal centerPathWeight =
				centerPathLengthSq > 1.0e-20f
					? PxClamp(
						(initialPosition - sphereCenterStart).
							dot(centerPath) /
							centerPathLengthSq,
						0.0f, 1.0f)
					: 0.0f;
			const PxVec3 closestCenter =
				sphereCenterStart +
				centerPath * centerPathWeight;
			const PxVec3 closestMedialPoint =
				closestCenter + PxVec3(
					PxClamp(
						initialPosition.x - closestCenter.x,
						-capsuleHalfHeight,
						capsuleHalfHeight),
					0.0f, 0.0f);
			minimumVertexSweepSeparation = PxMin(
				minimumVertexSweepSeparation,
				(initialPosition - closestMedialPoint).
					magnitude() - sphereRadius);
		}
	}
	return PxIsFinite(minimumVertexSweepSeparation) &&
		minimumVertexSweepSeparation < PX_MAX_F32;
}

bool SweptGeometrySampler::measureDeformingReverseSweptProof(
	PxDeformableVolume* volume,
	const PxTransform& rigidPose,
	PxReal radius,
	PxReal capsuleHalfHeight,
	bool convexCase,
	PxReal& endpointMinSeparation,
	PxReal& midSweepMinSeparation,
	PxReal& minimumVertexSweepSeparation) const
{
	if(!volume || !rigidPose.isValid() ||
		!PxIsFinite(radius) || radius <= 0.0f ||
		!PxIsFinite(capsuleHalfHeight) ||
		capsuleHalfHeight < 0.0f ||
		(convexCase && !mConvexMesh))
		return false;
	const PxTetrahedronMesh* mesh = volume->getCollisionMesh();
	const PxU32 vertexCount = mesh ? mesh->getNbVertices() : 0;
	const PxU32 tetrahedronCount =
		mesh ? mesh->getNbTetrahedrons() : 0;
	if(!mesh || vertexCount == 0 || tetrahedronCount == 0 ||
		mReverseInitialPositions.size() !=
			vertexCount ||
		mDeformingFreeEndPositions.size() !=
			vertexCount)
		return false;
	const bool has16BitIndices =
		mesh->getTetrahedronMeshFlags() &
			PxTetrahedronMeshFlag::e16_BIT_INDICES;
	const PxU16* indices16 = has16BitIndices
		? static_cast<const PxU16*>(mesh->getTetrahedrons())
		: NULL;
	const PxU32* indices32 = has16BitIndices
		? NULL
		: static_cast<const PxU32*>(mesh->getTetrahedrons());
	if((has16BitIndices && !indices16) ||
		(!has16BitIndices && !indices32))
		return false;
	const PxU32 faceEndpoints[4][3] =
	{
		{0, 2, 1},
		{0, 1, 3},
		{0, 3, 2},
		{1, 2, 3}
	};
	struct IndexedBoundaryFace
	{
		PxU32 vertices[3];
		PxU32 sortedVertices[3];
		PxU32 ownerCount;
	};
	PxArray<IndexedBoundaryFace> indexedFaces;
	for(PxU32 tetrahedronIndex = 0;
		tetrahedronIndex < tetrahedronCount;
		++tetrahedronIndex)
	{
		PxU32 tetrahedron[4];
		for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
		{
			const PxU32 flatIndex =
				tetrahedronIndex * 4 + endpoint;
			tetrahedron[endpoint] = has16BitIndices
				? PxU32(indices16[flatIndex])
				: indices32[flatIndex];
			if(tetrahedron[endpoint] >= vertexCount)
				return false;
		}
		for(PxU32 faceIndex = 0;
			faceIndex < 4; ++faceIndex)
		{
			IndexedBoundaryFace face;
			for(PxU32 endpoint = 0; endpoint < 3; ++endpoint)
			{
				face.vertices[endpoint] =
					tetrahedron[
						faceEndpoints[faceIndex][endpoint]];
				face.sortedVertices[endpoint] =
					face.vertices[endpoint];
			}
			for(PxU32 i = 0; i < 2; ++i)
			{
				for(PxU32 j = i + 1; j < 3; ++j)
				{
					if(face.sortedVertices[j] <
						face.sortedVertices[i])
					{
						const PxU32 temporary =
							face.sortedVertices[i];
						face.sortedVertices[i] =
							face.sortedVertices[j];
						face.sortedVertices[j] =
							temporary;
					}
				}
			}
			face.ownerCount = 1;
			bool matched = false;
			for(PxU32 existingIndex = 0;
				existingIndex < indexedFaces.size();
				++existingIndex)
			{
				IndexedBoundaryFace& existing =
					indexedFaces[existingIndex];
				if(existing.sortedVertices[0] ==
						face.sortedVertices[0] &&
					existing.sortedVertices[1] ==
						face.sortedVertices[1] &&
					existing.sortedVertices[2] ==
						face.sortedVertices[2])
				{
					existing.ownerCount++;
					matched = true;
					break;
				}
			}
			if(!matched)
				indexedFaces.pushBack(face);
		}
	}
	const PxVec3* convexVertices = convexCase
		? mConvexMesh->getVertices()
		: NULL;
	const PxU32 convexVertexCount = convexCase
		? mConvexMesh->getNbVertices()
		: 0;
	const PxConvexMeshGeometry convexGeometry(
		convexCase ? mConvexMesh : NULL);
	const PxVec3 capsuleAxis = rigidPose.q.getBasisVector0();

	auto getFaceSeparation = [&](
		const PxVec3& a,
		const PxVec3& b,
		const PxVec3& c)
	{
		PxReal separation = PX_MAX_F32;
		if(convexCase)
		{
			for(PxU32 rigidVertex = 0;
				rigidVertex < convexVertexCount; ++rigidVertex)
			{
				const PxVec3 point =
					rigidPose.transform(
						convexVertices[rigidVertex]);
				const PxVec3 closest =
					closestPointOnTriangleForGate(
						point, a, b, c);
				separation = PxMin(
					separation,
					(point - closest).magnitude());
			}
		}
		else
		{
			const PxU32 sampleCount =
				capsuleHalfHeight > 0.0f ? 128u : 0u;
			for(PxU32 sample = 0;
				sample <= sampleCount; ++sample)
			{
				const PxReal axisCoordinate =
					sampleCount > 0
						? -capsuleHalfHeight +
							2.0f * capsuleHalfHeight *
								(PxReal(sample) /
								 PxReal(sampleCount))
						: 0.0f;
				const PxVec3 medialPoint =
					rigidPose.p +
						capsuleAxis * axisCoordinate;
				const PxVec3 closest =
					closestPointOnTriangleForGate(
						medialPoint, a, b, c);
				separation = PxMin(
					separation,
					(medialPoint - closest).magnitude() -
						radius);
			}
		}
		return separation;
	};

	endpointMinSeparation = PX_MAX_F32;
	midSweepMinSeparation = PX_MAX_F32;
	minimumVertexSweepSeparation = PX_MAX_F32;
	PxArray<PxVec3> samplePositions;
	samplePositions.resize(vertexCount);
	for(PxU32 sample = 0; sample <= 128; ++sample)
	{
		const PxReal alpha =
			PxReal(sample) / 128.0f;
		for(PxU32 vertexIndex = 0;
			vertexIndex < vertexCount; ++vertexIndex)
		{
			const PxVec3 start =
				mReverseInitialPositions[
					vertexIndex];
			const PxVec3 end =
				mDeformingFreeEndPositions[
					vertexIndex];
			if(!start.isFinite() || !end.isFinite())
				return false;
			const PxVec3 point =
				start + (end - start) * alpha;
			samplePositions[vertexIndex] = point;
			PxReal vertexSeparation = PX_MAX_F32;
			if(convexCase)
			{
				const PxReal squaredDistance =
					PxGeometryQuery::pointDistance(
						point, convexGeometry, rigidPose);
				if(!PxIsFinite(squaredDistance) ||
					squaredDistance < 0.0f)
					return false;
				vertexSeparation = PxSqrt(squaredDistance);
			}
			else
				vertexSeparation = getCapsuleSignedSeparation(
					point, rigidPose, radius,
					capsuleHalfHeight);
			minimumVertexSweepSeparation = PxMin(
				minimumVertexSweepSeparation,
				vertexSeparation);
		}

		PxReal sampleFaceSeparation = PX_MAX_F32;
		for(PxU32 faceIndex = 0;
			faceIndex < indexedFaces.size(); ++faceIndex)
		{
			const IndexedBoundaryFace& face =
				indexedFaces[faceIndex];
			if(face.ownerCount != 1)
				continue;
			const PxVec3& a =
				samplePositions[face.vertices[0]];
			const PxVec3& b =
				samplePositions[face.vertices[1]];
			const PxVec3& c =
				samplePositions[face.vertices[2]];
			const PxReal candidateFaceSeparation =
				getFaceSeparation(a, b, c);
			sampleFaceSeparation = PxMin(
				sampleFaceSeparation,
				candidateFaceSeparation);
		}
		if(sample == 0 || sample == 128)
		{
			endpointMinSeparation = PxMin(
				endpointMinSeparation,
				sampleFaceSeparation);
		}
		else
			midSweepMinSeparation = PxMin(
				midSweepMinSeparation,
				sampleFaceSeparation);
	}
	return PxIsFinite(endpointMinSeparation) &&
		PxIsFinite(midSweepMinSeparation) &&
		PxIsFinite(minimumVertexSweepSeparation);
}

bool SweptGeometrySampler::measureCapsuleFaceSeparation(
	PxDeformableVolume* volume,
	const PxTransform& capsulePose,
	PxReal capsuleRadius,
	PxReal capsuleHalfHeight,
	bool useInitialPositions,
	PxReal& faceSeparation) const
{
	if(!volume || !capsulePose.isValid() ||
		!PxIsFinite(capsuleRadius) || capsuleRadius <= 0.0f ||
		!PxIsFinite(capsuleHalfHeight) || capsuleHalfHeight < 0.0f)
		return false;
	const PxTetrahedronMesh* mesh = volume->getCollisionMesh();
	const PxVec4* positions = volume->getPositionInvMassBufferH();
	const PxU32 vertexCount = mesh ? mesh->getNbVertices() : 0;
	const PxU32 tetrahedronCount =
		mesh ? mesh->getNbTetrahedrons() : 0;
	if(!mesh || !positions || vertexCount == 0 ||
		tetrahedronCount == 0 ||
		(useInitialPositions &&
		 mReverseInitialPositions.size() !=
			vertexCount))
		return false;

	const bool has16BitIndices =
		mesh->getTetrahedronMeshFlags() &
			PxTetrahedronMeshFlag::e16_BIT_INDICES;
	const PxU16* indices16 = has16BitIndices
		? static_cast<const PxU16*>(mesh->getTetrahedrons())
		: NULL;
	const PxU32* indices32 = has16BitIndices
		? NULL
		: static_cast<const PxU32*>(mesh->getTetrahedrons());
	if((has16BitIndices && !indices16) ||
		(!has16BitIndices && !indices32))
		return false;
	const PxU32 faceEndpoints[4][3] =
	{
		{0, 2, 1},
		{0, 1, 3},
		{0, 3, 2},
		{1, 2, 3}
	};
	const PxVec3 capsuleAxis = capsulePose.q.getBasisVector0();
	faceSeparation = PX_MAX_F32;
	for(PxU32 tetrahedronIndex = 0;
		tetrahedronIndex < tetrahedronCount;
		++tetrahedronIndex)
	{
		PxU32 tetrahedron[4];
		for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
		{
			const PxU32 flatIndex =
				tetrahedronIndex * 4 + endpoint;
			tetrahedron[endpoint] = has16BitIndices
				? PxU32(indices16[flatIndex])
				: indices32[flatIndex];
			if(tetrahedron[endpoint] >= vertexCount)
				return false;
		}
		for(PxU32 faceIndex = 0; faceIndex < 4; ++faceIndex)
		{
			PxVec3 face[3];
			for(PxU32 endpoint = 0; endpoint < 3; ++endpoint)
			{
				const PxU32 vertexIndex =
					tetrahedron[faceEndpoints[faceIndex][endpoint]];
				face[endpoint] = useInitialPositions
					? mReverseInitialPositions[
						vertexIndex]
					: positions[vertexIndex].getXYZ();
				if(!face[endpoint].isFinite())
					return false;
			}
			for(PxU32 sample = 0; sample <= 128; ++sample)
			{
				const PxReal axisCoordinate =
					-capsuleHalfHeight +
					2.0f * capsuleHalfHeight *
						(PxReal(sample) / 128.0f);
				const PxVec3 medialPoint =
					capsulePose.p + capsuleAxis * axisCoordinate;
				const PxVec3 closest =
					closestPointOnTriangleForGate(
						medialPoint, face[0], face[1], face[2]);
				if(!closest.isFinite())
					return false;
				faceSeparation = PxMin(
					faceSeparation,
					(medialPoint - closest).magnitude() -
						capsuleRadius);
			}
		}
	}
	return PxIsFinite(faceSeparation) &&
		faceSeparation < PX_MAX_F32;
}

bool SweptGeometrySampler::measureConvexFaceSeparation(
	PxDeformableVolume* volume,
	const PxTransform& convexPose,
	bool useInitialPositions,
	PxReal& faceSeparation) const
{
	if(!volume || !convexPose.isValid() ||
		!mConvexMesh)
		return false;
	const PxTetrahedronMesh* mesh = volume->getCollisionMesh();
	const PxVec4* positions = volume->getPositionInvMassBufferH();
	const PxU32 vertexCount = mesh ? mesh->getNbVertices() : 0;
	const PxU32 tetrahedronCount =
		mesh ? mesh->getNbTetrahedrons() : 0;
	const PxVec3* convexVertices =
		mConvexMesh->getVertices();
	const PxU32 convexVertexCount =
		mConvexMesh->getNbVertices();
	if(!mesh || !positions || vertexCount == 0 ||
		tetrahedronCount == 0 || !convexVertices ||
		convexVertexCount == 0 ||
		(useInitialPositions &&
		 mReverseInitialPositions.size() !=
			vertexCount))
		return false;

	const bool has16BitIndices =
		mesh->getTetrahedronMeshFlags() &
			PxTetrahedronMeshFlag::e16_BIT_INDICES;
	const PxU16* indices16 = has16BitIndices
		? static_cast<const PxU16*>(mesh->getTetrahedrons())
		: NULL;
	const PxU32* indices32 = has16BitIndices
		? NULL
		: static_cast<const PxU32*>(mesh->getTetrahedrons());
	if((has16BitIndices && !indices16) ||
		(!has16BitIndices && !indices32))
		return false;
	const PxU32 faceEndpoints[4][3] =
	{
		{0, 2, 1},
		{0, 1, 3},
		{0, 3, 2},
		{1, 2, 3}
	};

	faceSeparation = PX_MAX_F32;
	for(PxU32 convexVertexIndex = 0;
		convexVertexIndex < convexVertexCount;
		++convexVertexIndex)
	{
		const PxVec3 point =
			convexPose.transform(convexVertices[convexVertexIndex]);
		if(!point.isFinite())
			return false;
		PxReal pointDistance = PX_MAX_F32;
		bool pointInside = false;
		for(PxU32 tetrahedronIndex = 0;
			tetrahedronIndex < tetrahedronCount;
			++tetrahedronIndex)
		{
			PxVec3 tetrahedron[4];
			for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
			{
				const PxU32 flatIndex =
					tetrahedronIndex * 4 + endpoint;
				const PxU32 vertexIndex = has16BitIndices
					? PxU32(indices16[flatIndex])
					: indices32[flatIndex];
				if(vertexIndex >= vertexCount)
					return false;
				tetrahedron[endpoint] = useInitialPositions
					? mReverseInitialPositions[
						vertexIndex]
					: positions[vertexIndex].getXYZ();
				if(!tetrahedron[endpoint].isFinite())
					return false;
			}

			const PxVec3 edge0 =
				tetrahedron[1] - tetrahedron[0];
			const PxVec3 edge1 =
				tetrahedron[2] - tetrahedron[0];
			const PxVec3 edge2 =
				tetrahedron[3] - tetrahedron[0];
			const PxVec3 relative =
				point - tetrahedron[0];
			const PxReal determinant =
				edge0.dot(edge1.cross(edge2));
			if(PxAbs(determinant) > 1.0e-20f)
			{
				const PxReal inverseDeterminant =
					1.0f / determinant;
				const PxReal b1 =
					relative.dot(edge1.cross(edge2)) *
						inverseDeterminant;
				const PxReal b2 =
					edge0.dot(relative.cross(edge2)) *
						inverseDeterminant;
				const PxReal b3 =
					edge0.dot(edge1.cross(relative)) *
						inverseDeterminant;
				const PxReal b0 = 1.0f - b1 - b2 - b3;
				const PxReal tolerance = 1.0e-5f;
				pointInside = pointInside ||
					(b0 >= -tolerance &&
					 b1 >= -tolerance &&
					 b2 >= -tolerance &&
					 b3 >= -tolerance);
			}

			for(PxU32 faceIndex = 0;
				faceIndex < 4; ++faceIndex)
			{
				const PxVec3 closest =
					closestPointOnTriangleForGate(
						point,
						tetrahedron[
							faceEndpoints[faceIndex][0]],
						tetrahedron[
							faceEndpoints[faceIndex][1]],
						tetrahedron[
							faceEndpoints[faceIndex][2]]);
				if(!closest.isFinite())
					return false;
				pointDistance = PxMin(
					pointDistance,
					(point - closest).magnitude());
			}
		}
		if(!PxIsFinite(pointDistance) ||
			pointDistance >= PX_MAX_F32)
			return false;
		faceSeparation = PxMin(
			faceSeparation,
			pointInside ? -pointDistance : pointDistance);
	}
	return PxIsFinite(faceSeparation) &&
		faceSeparation < PX_MAX_F32;
}

bool SweptGeometrySampler::measureRotationalConvexPointSweepSeparations(
	const PxTransform& startPose,
	const PxTransform& endPose,
	PxReal& endpointMinSeparation,
	PxReal& midSweepMinSeparation) const
{
	if(!startPose.isValid() || !endPose.isValid() ||
		!mConvexMesh ||
		mRotationalInitialPositions.empty())
		return false;
	const PxConvexMeshGeometry geometry(
		mConvexMesh);
	endpointMinSeparation = PX_MAX_F32;
	midSweepMinSeparation = PX_MAX_F32;
	for(PxU32 vertexIndex = 0;
		vertexIndex <
			mRotationalInitialPositions.size();
		++vertexIndex)
	{
		const PxVec3 point =
			mRotationalInitialPositions[
				vertexIndex];
		if(!point.isFinite())
			return false;
		const PxReal startDistanceSq =
			PxGeometryQuery::pointDistance(
				point, geometry, startPose);
		const PxReal endDistanceSq =
			PxGeometryQuery::pointDistance(
				point, geometry, endPose);
		if(!PxIsFinite(startDistanceSq) ||
			!PxIsFinite(endDistanceSq) ||
			startDistanceSq < 0.0f ||
			endDistanceSq < 0.0f)
			return false;
		endpointMinSeparation = PxMin(
			endpointMinSeparation,
			PxSqrt(PxMin(startDistanceSq, endDistanceSq)));
		for(PxU32 sample = 1; sample < 64; ++sample)
		{
			const PxReal time =
				PxReal(sample) / 64.0f;
			const PxTransform samplePose(
				startPose.p +
					(endPose.p - startPose.p) * time,
				PxSlerp(
					time, startPose.q, endPose.q).
						getNormalized());
			const PxReal sampleDistanceSq =
				PxGeometryQuery::pointDistance(
					point, geometry, samplePose);
			if(!PxIsFinite(sampleDistanceSq) ||
				sampleDistanceSq < 0.0f)
				return false;
			midSweepMinSeparation = PxMin(
				midSweepMinSeparation,
				PxSqrt(sampleDistanceSq));
		}
	}
	return PxIsFinite(endpointMinSeparation) &&
		PxIsFinite(midSweepMinSeparation);
}

bool SweptGeometrySampler::measureRotationalCapsuleReverseSweptSeparations(
	PxDeformableVolume* volume,
	const PxTransform& currentPose,
	const PxTransform& startPose,
	const PxTransform& endPose,
	PxReal capsuleRadius,
	PxReal capsuleHalfHeight,
	PxReal& faceSeparation,
	PxReal& minimumVertexSweepSeparation,
	PxReal& endpointMinSeparation,
	PxReal& midSweepMinSeparation) const
{
	if(!volume || !currentPose.isValid() ||
		!startPose.isValid() || !endPose.isValid() ||
		!measureCapsuleFaceSeparation(
			volume, currentPose, capsuleRadius,
			capsuleHalfHeight, false, faceSeparation))
		return false;
	PxReal startFaceSeparation = PX_MAX_F32;
	PxReal endFaceSeparation = PX_MAX_F32;
	if(!measureCapsuleFaceSeparation(
			volume, startPose, capsuleRadius,
			capsuleHalfHeight, true, startFaceSeparation) ||
		!measureCapsuleFaceSeparation(
			volume, endPose, capsuleRadius,
			capsuleHalfHeight, true, endFaceSeparation))
		return false;

	const PxTetrahedronMesh* mesh = volume->getCollisionMesh();
	const PxU32 vertexCount = mesh ? mesh->getNbVertices() : 0;
	if(vertexCount == 0 ||
		mReverseInitialPositions.size() !=
			vertexCount)
		return false;
	endpointMinSeparation =
		PxMin(startFaceSeparation, endFaceSeparation);
	midSweepMinSeparation = PX_MAX_F32;
	minimumVertexSweepSeparation = PX_MAX_F32;
	for(PxU32 sample = 0; sample <= 64; ++sample)
	{
		const PxReal time = PxReal(sample) / 64.0f;
		const PxTransform samplePose(
			startPose.p + (endPose.p - startPose.p) * time,
			PxSlerp(
				time, startPose.q, endPose.q).
					getNormalized());
		if(sample > 0 && sample < 64)
		{
			PxReal sampleFaceSeparation = PX_MAX_F32;
			if(!measureCapsuleFaceSeparation(
					volume, samplePose, capsuleRadius,
					capsuleHalfHeight, true,
					sampleFaceSeparation))
				return false;
			midSweepMinSeparation = PxMin(
				midSweepMinSeparation, sampleFaceSeparation);
		}
		for(PxU32 vertexIndex = 0;
			vertexIndex < vertexCount;
			++vertexIndex)
		{
			const PxVec3 point =
				mReverseInitialPositions[
					vertexIndex];
			if(!point.isFinite())
				return false;
			minimumVertexSweepSeparation = PxMin(
				minimumVertexSweepSeparation,
				getCapsuleSignedSeparation(
					point, samplePose,
					capsuleRadius, capsuleHalfHeight));
		}
	}
	return PxIsFinite(faceSeparation) &&
		PxIsFinite(minimumVertexSweepSeparation) &&
		PxIsFinite(endpointMinSeparation) &&
		PxIsFinite(midSweepMinSeparation);
}

bool SweptGeometrySampler::measureRotationalConvexReverseSweptSeparations(
	PxDeformableVolume* volume,
	const PxTransform& currentPose,
	const PxTransform& startPose,
	const PxTransform& endPose,
	PxReal& faceSeparation,
	PxReal& minimumVertexSweepSeparation,
	PxReal& endpointMinSeparation,
	PxReal& midSweepMinSeparation) const
{
	if(!volume || !currentPose.isValid() ||
		!startPose.isValid() || !endPose.isValid() ||
		!mConvexMesh ||
		!measureConvexFaceSeparation(
			volume, currentPose, false, faceSeparation))
		return false;
	PxReal startFaceSeparation = PX_MAX_F32;
	PxReal endFaceSeparation = PX_MAX_F32;
	if(!measureConvexFaceSeparation(
			volume, startPose, true, startFaceSeparation) ||
		!measureConvexFaceSeparation(
			volume, endPose, true, endFaceSeparation))
		return false;

	const PxTetrahedronMesh* mesh = volume->getCollisionMesh();
	const PxU32 vertexCount = mesh ? mesh->getNbVertices() : 0;
	if(vertexCount == 0 ||
		mReverseInitialPositions.size() !=
			vertexCount)
		return false;
	const PxConvexMeshGeometry geometry(
		mConvexMesh);
	endpointMinSeparation =
		PxMin(startFaceSeparation, endFaceSeparation);
	midSweepMinSeparation = PX_MAX_F32;
	minimumVertexSweepSeparation = PX_MAX_F32;
	for(PxU32 sample = 0; sample <= 64; ++sample)
	{
		const PxReal time = PxReal(sample) / 64.0f;
		const PxTransform samplePose(
			startPose.p + (endPose.p - startPose.p) * time,
			PxSlerp(
				time, startPose.q, endPose.q).
					getNormalized());
		if(sample > 0 && sample < 64)
		{
			PxReal sampleFaceSeparation = PX_MAX_F32;
			if(!measureConvexFaceSeparation(
					volume, samplePose, true,
					sampleFaceSeparation))
				return false;
			midSweepMinSeparation = PxMin(
				midSweepMinSeparation, sampleFaceSeparation);
		}
		for(PxU32 vertexIndex = 0;
			vertexIndex < vertexCount;
			++vertexIndex)
		{
			const PxVec3 point =
				mReverseInitialPositions[
					vertexIndex];
			if(!point.isFinite())
				return false;
			const PxReal squaredDistance =
				PxGeometryQuery::pointDistance(
					point, geometry, samplePose);
			if(!PxIsFinite(squaredDistance) ||
				squaredDistance < 0.0f)
				return false;
			minimumVertexSweepSeparation = PxMin(
				minimumVertexSweepSeparation,
				PxSqrt(squaredDistance));
		}
	}
	return PxIsFinite(faceSeparation) &&
		PxIsFinite(minimumVertexSweepSeparation) &&
		PxIsFinite(endpointMinSeparation) &&
		PxIsFinite(midSweepMinSeparation);
}

bool SweptGeometrySampler::measureRotationalTriangleSurfaceSweepSeparations(
	const PxTransform& startPose,
	const PxTransform& endPose,
	bool heightField, bool reverseCase,
	PxReal& endpointMinSeparation,
	PxReal& midSweepMinSeparation,
	PxReal& minimumVertexSweepSeparation) const
{
	if(!startPose.isValid() || !endPose.isValid() ||
		mReverseInitialPositions.size() < 4)
		return false;
	const PxVec3 bladeLocal[4] =
	{
		heightField
			? PxVec3(0.0f, 0.0f, 0.0f)
			: PxVec3(-1.0f, 0.0f, -0.1f),
		heightField
			? PxVec3(0.0f, 0.0f, 0.2f)
			: PxVec3(-1.0f, 0.0f, 0.1f),
		heightField
			? PxVec3(2.0f, 0.0f, 0.2f)
			: PxVec3(1.0f, 0.0f, 0.1f),
		heightField
			? PxVec3(2.0f, 0.0f, 0.0f)
			: PxVec3(1.0f, 0.0f, -0.1f)
	};
	auto getBladeWorld = [&](
		PxReal time, PxVec3 world[4])
	{
		const PxVec3 center =
			startPose.p + (endPose.p - startPose.p) * time;
		const PxQuat rotation = PxSlerp(
			time, startPose.q, endPose.q).getNormalized();
		for(PxU32 vertex = 0; vertex < 4; ++vertex)
			world[vertex] =
				center + rotation.rotate(bladeLocal[vertex]);
	};
	auto getPointBladeDistance = [&](
		const PxVec3& point, PxReal time)
	{
		PxVec3 world[4];
		getBladeWorld(time, world);
		const PxVec3 closest0 =
			closestPointOnTriangleForGate(
				point, world[0], world[1], world[2]);
		const PxVec3 closest1 =
			closestPointOnTriangleForGate(
				point, world[0], world[2], world[3]);
		return PxMin(
			(point - closest0).magnitude(),
			(point - closest1).magnitude());
	};
	const PxU32 boundaryFaces[4][3] =
	{
		{0, 2, 1},
		{0, 1, 3},
		{0, 3, 2},
		{1, 2, 3}
	};
	endpointMinSeparation = PX_MAX_F32;
	midSweepMinSeparation = PX_MAX_F32;
	minimumVertexSweepSeparation = PX_MAX_F32;
	for(PxU32 endpoint = 0; endpoint < 2; ++endpoint)
	{
		const PxReal time = PxReal(endpoint);
		for(PxU32 softVertex = 0;
			softVertex <
				mReverseInitialPositions.size();
			++softVertex)
		{
			const PxVec3 point =
				mReverseInitialPositions[
					softVertex];
			if(!point.isFinite())
				return false;
			endpointMinSeparation = PxMin(
				endpointMinSeparation,
				getPointBladeDistance(point, time));
		}
		if(reverseCase)
		{
			PxVec3 world[4];
			getBladeWorld(time, world);
			for(PxU32 rigidVertex = 0;
				rigidVertex < 4; ++rigidVertex)
			{
				for(PxU32 face = 0; face < 4; ++face)
				{
					const PxVec3& a =
						mReverseInitialPositions[
							boundaryFaces[face][0]];
					const PxVec3& b =
						mReverseInitialPositions[
							boundaryFaces[face][1]];
					const PxVec3& c =
						mReverseInitialPositions[
							boundaryFaces[face][2]];
					const PxVec3 closest =
						closestPointOnTriangleForGate(
							world[rigidVertex], a, b, c);
					endpointMinSeparation = PxMin(
						endpointMinSeparation,
						(world[rigidVertex] -
							closest).magnitude());
				}
			}
		}
	}

	const PxVec3& bottom0 =
		mReverseInitialPositions[0];
	const PxVec3& bottom1 =
		mReverseInitialPositions[1];
	const PxVec3& bottom2 =
		mReverseInitialPositions[3];
	for(PxU32 sample = 0; sample <= 256; ++sample)
	{
		const PxReal time = PxReal(sample) / 256.0f;
		for(PxU32 softVertex = 0;
			softVertex <
				mReverseInitialPositions.size();
			++softVertex)
		{
			const PxReal distance = getPointBladeDistance(
				mReverseInitialPositions[
					softVertex],
				time);
			if(!PxIsFinite(distance))
				return false;
			if(reverseCase)
				minimumVertexSweepSeparation = PxMin(
					minimumVertexSweepSeparation, distance);
			else
				midSweepMinSeparation = PxMin(
					midSweepMinSeparation, distance);
		}
		if(reverseCase)
		{
			PxVec3 world[4];
			getBladeWorld(time, world);
			for(PxU32 rigidVertex = 0;
				rigidVertex < 4; ++rigidVertex)
			{
				const PxVec3 projected(
					world[rigidVertex].x,
					bottom0.y,
					world[rigidVertex].z);
				const PxVec3 closest =
					closestPointOnTriangleForGate(
						projected,
						bottom0, bottom1, bottom2);
				if((projected - closest).magnitudeSquared() <=
					1.0e-8f)
					midSweepMinSeparation = PxMin(
						midSweepMinSeparation,
						bottom0.y -
							world[rigidVertex].y);
			}
		}
	}
	if(!reverseCase)
		minimumVertexSweepSeparation = PX_MAX_F32;
	return PxIsFinite(endpointMinSeparation) &&
		PxIsFinite(midSweepMinSeparation) &&
		(!reverseCase ||
		 PxIsFinite(minimumVertexSweepSeparation));
}

bool SweptGeometrySampler::measureSmoothReverseSeparations(
	PxDeformableVolume* volume,
	const PxVec3& rigidCenter,
	PxReal rigidRadius,
	PxReal capsuleHalfHeight,
	bool convexCase,
	bool triangleMeshCase,
	bool heightFieldCase,
	PxReal& faceSeparation,
	PxReal& minimumVertexSeparation) const
{
	if(!volume || !volume->getCollisionMesh() ||
		!rigidCenter.isFinite() ||
		!PxIsFinite(rigidRadius) || rigidRadius <= 0.0f ||
		!PxIsFinite(capsuleHalfHeight) ||
		capsuleHalfHeight < 0.0f ||
		(convexCase && !mConvexMesh) ||
		(triangleMeshCase && !mTriangleMesh) ||
		(heightFieldCase && !mHeightField))
		return false;
	const PxTetrahedronMesh* mesh = volume->getCollisionMesh();
	const PxVec4* positions =
		volume->getPositionInvMassBufferH();
	const PxU32 vertexCount = mesh->getNbVertices();
	const PxU32 tetrahedronCount = mesh->getNbTetrahedrons();
	if(!positions || vertexCount == 0 || tetrahedronCount == 0)
		return false;

	minimumVertexSeparation = PX_MAX_F32;
	for(PxU32 vertexIndex = 0;
		vertexIndex < vertexCount; ++vertexIndex)
	{
		const PxVec3 position = positions[vertexIndex].getXYZ();
		if(!position.isFinite())
			return false;
		minimumVertexSeparation = PxMin(
			minimumVertexSeparation,
			[&]()
			{
				if(triangleMeshCase || heightFieldCase)
				{
					const PxTriangleMeshGeometry triangleGeometry(
						mTriangleMesh);
					const PxHeightFieldGeometry heightGeometry(
						mHeightField,
						PxMeshGeometryFlags(),
						0.1f, 0.3f, 0.3f);
					const PxU32 heightFieldTriangles[] =
						{0, 1, 2, 3, 6, 7, 8, 9};
					const PxU32 triangleCount =
						triangleMeshCase
							? mTriangleMesh->
								getNbTriangles()
							: 8u;
					PxReal minimumDistance = PX_MAX_F32;
					for(PxU32 triangleIndex = 0;
						triangleIndex < triangleCount;
						++triangleIndex)
					{
						PxTriangle triangle;
						if(triangleMeshCase)
							PxMeshQuery::getTriangle(
								triangleGeometry,
								PxTransform(rigidCenter),
								triangleIndex, triangle);
						else
							PxMeshQuery::getTriangle(
								heightGeometry,
								PxTransform(rigidCenter),
								heightFieldTriangles[
									triangleIndex],
								triangle);
						const PxVec3 closest =
							closestPointOnTriangleForGate(
								position,
								triangle.verts[0],
								triangle.verts[1],
								triangle.verts[2]);
						minimumDistance = PxMin(
							minimumDistance,
							(position - closest).magnitude());
					}
					return minimumDistance;
				}
				if(convexCase)
				{
					const PxReal squaredDistance =
						PxGeometryQuery::pointDistance(
							position,
							PxConvexMeshGeometry(
								mConvexMesh),
							PxTransform(rigidCenter));
					return squaredDistance >= 0.0f
						? PxSqrt(squaredDistance)
						: -PX_MAX_F32;
				}
				PxVec3 radial = position - rigidCenter;
				radial.x -= PxClamp(
					radial.x,
					-capsuleHalfHeight,
					capsuleHalfHeight);
				return radial.magnitude() - rigidRadius;
			}());
	}

	const PxU32 faceEndpoints[4][3] =
	{
		{0, 2, 1},
		{0, 1, 3},
		{0, 3, 2},
		{1, 2, 3}
	};
	const bool has16BitIndices =
		mesh->getTetrahedronMeshFlags() &
			PxTetrahedronMeshFlag::e16_BIT_INDICES;
	const PxU16* indices16 = has16BitIndices
		? static_cast<const PxU16*>(mesh->getTetrahedrons())
		: NULL;
	const PxU32* indices32 = has16BitIndices
		? NULL
		: static_cast<const PxU32*>(mesh->getTetrahedrons());
	if((has16BitIndices && !indices16) ||
		(!has16BitIndices && !indices32))
		return false;

	faceSeparation = PX_MAX_F32;
	for(PxU32 tetrahedronIndex = 0;
		tetrahedronIndex < tetrahedronCount;
		++tetrahedronIndex)
	{
		PxU32 tetrahedron[4];
		for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
		{
			const PxU32 flatIndex =
				tetrahedronIndex * 4 + endpoint;
			tetrahedron[endpoint] = has16BitIndices
				? PxU32(indices16[flatIndex])
				: indices32[flatIndex];
			if(tetrahedron[endpoint] >= vertexCount)
				return false;
		}
		for(PxU32 faceIndex = 0; faceIndex < 4; ++faceIndex)
		{
			const PxVec3 a =
				positions[tetrahedron[
					faceEndpoints[faceIndex][0]]].getXYZ();
			const PxVec3 b =
				positions[tetrahedron[
					faceEndpoints[faceIndex][1]]].getXYZ();
			const PxVec3 c =
				positions[tetrahedron[
					faceEndpoints[faceIndex][2]]].getXYZ();
			const PxVec3 axisSamples[3] =
			{
				triangleMeshCase
					? rigidCenter +
						PxVec3(0.0f, 0.3f, 0.0f)
					: heightFieldCase
					? rigidCenter +
						PxVec3(0.3f, 0.3f, 0.3f)
					: convexCase
					? rigidCenter +
						PxVec3(0.0f, 0.3f, 0.0f)
					: rigidCenter -
						PxVec3(
							capsuleHalfHeight,
							0.0f, 0.0f),
				rigidCenter,
				rigidCenter +
					PxVec3(capsuleHalfHeight, 0.0f, 0.0f)
			};
			for(PxU32 sampleIndex = 0;
				sampleIndex <
					((convexCase || triangleMeshCase ||
					  heightFieldCase) ? 1u : 3u);
				++sampleIndex)
			{
				const PxVec3 closest =
					closestPointOnTriangleForGate(
						axisSamples[sampleIndex], a, b, c);
				if(!closest.isFinite())
					return false;
				faceSeparation = PxMin(
					faceSeparation,
					(closest - axisSamples[sampleIndex]).
						magnitude() -
						((convexCase || triangleMeshCase ||
						  heightFieldCase)
							? 0.0f : rigidRadius));
			}
		}
	}
	return PxIsFinite(faceSeparation) &&
		faceSeparation < PX_MAX_F32 &&
		PxIsFinite(minimumVertexSeparation) &&
		minimumVertexSeparation < PX_MAX_F32;
}

static bool doesTrianglePenetrateBox(
	const PxVec3& localP0,
	const PxVec3& localP1,
	const PxVec3& localP2,
	const PxVec3& halfExtents)
{
	// SAT treats touching as overlap. Shrink one extra epsilon so a face at the
	// allowed boundary does not spuriously fail the inclusive test.
	const PxReal tolerance = MAX_SURFACE_PENETRATION + 1.0e-6f;
	const PxVec3 shrunkenHalfExtents(
		PxMax(halfExtents.x - tolerance, 1.0e-6f),
		PxMax(halfExtents.y - tolerance, 1.0e-6f),
		PxMax(halfExtents.z - tolerance, 1.0e-6f));
	return Gu::intersectTriangleBox_ReferenceCode(
		PxVec3(0.0f), shrunkenHalfExtents,
		localP0, localP1, localP2) != 0;
}

bool measureCollisionSurfaceBoxGap(
	PxDeformableVolume& volume,
	const PxArray<PxU32>& surfaceTriangles,
	const PxTransform& boxPose,
	const PxVec3& boxHalfExtents,
	PxReal& outMinSignedSdf,
	bool& outTrianglePenetrated)
{
	const PxTetrahedronMesh* const collisionMesh = volume.getCollisionMesh();
	const PxVec4* const positions = volume.getPositionInvMassBufferH();
	if(!collisionMesh || !positions || surfaceTriangles.empty() ||
		surfaceTriangles.size() % 3 != 0 || !boxPose.isValid() ||
		!boxHalfExtents.isFinite() || boxHalfExtents.x <= 0.0f ||
		boxHalfExtents.y <= 0.0f || boxHalfExtents.z <= 0.0f)
		return false;

	const PxQuat inverseBoxRotation = boxPose.q.getConjugate();
	outMinSignedSdf = PX_MAX_F32;
	outTrianglePenetrated = false;
	for(PxU32 surfaceIndex = 0;
		surfaceIndex < surfaceTriangles.size(); surfaceIndex += 3)
	{
		PxVec3 localPositions[3];
		for(PxU32 vertex = 0; vertex < 3; ++vertex)
		{
			const PxU32 vertexIndex = surfaceTriangles[surfaceIndex + vertex];
			if(vertexIndex >= collisionMesh->getNbVertices())
				return false;
			const PxVec3 worldPosition = positions[vertexIndex].getXYZ();
			if(!worldPosition.isFinite())
				return false;
			const PxVec3 localPosition = inverseBoxRotation.rotate(
				worldPosition - boxPose.p);
			const PxVec3 distanceToBox(
				PxAbs(localPosition.x) - boxHalfExtents.x,
				PxAbs(localPosition.y) - boxHalfExtents.y,
				PxAbs(localPosition.z) - boxHalfExtents.z);
			const PxVec3 outsideDistance(
				PxMax(distanceToBox.x, 0.0f),
				PxMax(distanceToBox.y, 0.0f),
				PxMax(distanceToBox.z, 0.0f));
			const PxReal signedSdf = outsideDistance.magnitude() + PxMin(
				PxMax(distanceToBox.x,
					PxMax(distanceToBox.y, distanceToBox.z)), 0.0f);
			if(!localPosition.isFinite() || !PxIsFinite(signedSdf))
				return false;
			outMinSignedSdf = PxMin(outMinSignedSdf, signedSdf);
			localPositions[vertex] = localPosition;
		}
		outTrianglePenetrated = outTrianglePenetrated ||
			doesTrianglePenetrateBox(
				localPositions[0], localPositions[1], localPositions[2],
				boxHalfExtents);
	}
	return PxIsFinite(outMinSignedSdf);
}

bool measureCollisionSurfaceGroundGap(
	PxDeformableVolume& volume,
	const PxArray<PxU32>& surfaceTriangles,
	PxReal groundHeight,
	PxReal& outMinSignedSdf,
	bool& outTrianglePenetrated)
{
	const PxTetrahedronMesh* const collisionMesh = volume.getCollisionMesh();
	const PxVec4* const positions = volume.getPositionInvMassBufferH();
	if(!collisionMesh || !positions || surfaceTriangles.empty() ||
		surfaceTriangles.size() % 3 != 0 || !PxIsFinite(groundHeight))
		return false;

	outMinSignedSdf = PX_MAX_F32;
	outTrianglePenetrated = false;
	for(PxU32 surfaceIndex = 0;
		surfaceIndex < surfaceTriangles.size(); surfaceIndex += 3)
	{
		bool trianglePenetrated = false;
		for(PxU32 vertex = 0; vertex < 3; ++vertex)
		{
			const PxU32 vertexIndex = surfaceTriangles[surfaceIndex + vertex];
			if(vertexIndex >= collisionMesh->getNbVertices())
				return false;
			const PxVec3 worldPosition = positions[vertexIndex].getXYZ();
			if(!worldPosition.isFinite())
				return false;
			const PxReal signedSdf = worldPosition.y - groundHeight;
			if(!PxIsFinite(signedSdf))
				return false;
			outMinSignedSdf = PxMin(outMinSignedSdf, signedSdf);
			trianglePenetrated = trianglePenetrated ||
				signedSdf < -MAX_SURFACE_PENETRATION;
		}
		outTrianglePenetrated = outTrianglePenetrated ||
			trianglePenetrated;
	}
	return PxIsFinite(outMinSignedSdf);
}

bool isSurfaceGapSeparated(const SurfaceGapMetrics& gap)
{
	return PxIsFinite(gap.minSignedSdf) && !gap.penetrated &&
		gap.minSignedSdf >= -MAX_SURFACE_PENETRATION;
}

void initializeSurfaceGapMetrics(
	SurfaceGapMetrics& gap,
	PxReal initialSignedSdf,
	bool initiallyPenetrated)
{
	gap.initialSignedSdf = initialSignedSdf;
	gap.minSignedSdf = initialSignedSdf;
	gap.finalSignedSdf = initialSignedSdf;
	gap.penetrationFrames = 0;
	gap.penetrated = initiallyPenetrated;
}

void updateSurfaceGapMetrics(
	SurfaceGapMetrics& gap,
	PxReal signedSdf,
	bool penetrated)
{
	gap.minSignedSdf = PxMin(gap.minSignedSdf, signedSdf);
	gap.finalSignedSdf = signedSdf;
	gap.penetrated = gap.penetrated || penetrated;
	if(penetrated)
		gap.penetrationFrames++;
}

} // namespace SnippetDeformableVolumeAVBDValidation
