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

#ifndef SNIPPET_DEFORMABLE_VOLUME_AVBD_VALIDATION_H
#define SNIPPET_DEFORMABLE_VOLUME_AVBD_VALIDATION_H

#include "PxPhysicsAPI.h"

namespace SnippetDeformableVolumeAVBDValidation
{

static const physx::PxReal MAX_SURFACE_PENETRATION = 1.0e-3f;

struct DeformableVolumeMetrics
{
	physx::PxU32 initialized;
	physx::PxU32 completedFrames;
	physx::PxU32 fetchFailures;
	physx::PxU32 nonFiniteParticleSamples;
	physx::PxU32 invertedElementSamples;
	physx::PxU32 firstInversionFrame;
	physx::PxU32 firstInversionBody;
	physx::PxU32 firstInversionElement;
	physx::PxU32 invertedBodiesMask;
	physx::PxU32 particles;
	physx::PxU32 softBodies;
	physx::PxU32 tetElements;
	physx::PxU32 surfaceTriangles;
	physx::PxU32 rigidBoxes;
	physx::PxU32 sceneStatics;
	physx::PxU32 sceneDynamics;
	physx::PxU32 sceneDeformableVolumes;
	physx::PxU32 groundContactFrames;
	physx::PxU32 rigidContactFrames;
	physx::PxU32 softContactFrames;
	physx::PxU32 maxGroundContacts;
	physx::PxU32 maxRigidContacts;
	physx::PxU32 maxSoftContacts;
	physx::PxU32 invalidContactSourceSamples;
	physx::PxU32 finalInsideParticles;
	physx::PxU32 cleanupComplete;
	physx::PxU32 sceneActorCreated;
	physx::PxU32 sceneShapeAttached;
	physx::PxU32 sceneSimulationMeshAttached;
	physx::PxU32 sceneHostBuffersInitialized;
	physx::PxU32 sceneActorAdded;
	physx::PxU32 sceneActorRemoved;
	physx::PxU32 sceneActorReleased;
	physx::PxU32 sceneBoundsFinite;
	physx::PxU32 sceneStaticShapeDetached;
	physx::PxU32 sceneStaticShapeReattached;
	physx::PxU32 sceneStaticActorRemoved;
	physx::PxU32 sceneStaticActorReadded;
	physx::PxU32 sceneDynamicActorAdded;
	physx::PxU32 sceneDynamicActorRemoved;
	physx::PxU32 sceneDynamicActorReleased;
	physx::PxU32 sceneDynamicInitiallySleeping;
	physx::PxU32 sceneDynamicWokeBySoft;
	physx::PxU32 sceneDynamicFirstWakeFrame;
	physx::PxU32 sceneDynamicShapeDetached;
	physx::PxU32 sceneDynamicShapeReattached;
	physx::PxU32 sceneDynamicActorReadded;
	physx::PxU32 sceneDynamicReaddedSleeping;
	physx::PxU32 sceneDynamicRewokeBySoft;
	physx::PxU32 sceneDynamicSecondWakeFrame;
	physx::PxU32 sceneSecondDynamicActorAdded;
	physx::PxU32 sceneSecondDynamicActorRemoved;
	physx::PxU32 sceneSecondDynamicActorReleased;
	physx::PxU32 sceneSecondDynamicInitiallySleeping;
	physx::PxU32 sceneSecondDynamicWokeBySoft;
	physx::PxU32 sceneSecondDynamicFirstWakeFrame;
	physx::PxU32 sceneSecondVolumeActorCreated;
	physx::PxU32 sceneSecondVolumeHostBuffersInitialized;
	physx::PxU32 sceneSecondVolumeActorAdded;
	physx::PxU32 sceneSecondVolumeActorRemoved;
	physx::PxU32 sceneSecondVolumeActorReleased;
	physx::PxU32 sceneSecondVolumeBoundsFinite;
	physx::PxU32 sceneSoftInitiallyAwake;
	physx::PxU32 sceneSoftFirstSlept;
	physx::PxU32 sceneSoftFirstSleepFrame;
	physx::PxU32 sceneSoftSleepWakeCounterZero;
	physx::PxU32 sceneSoftSleepVelocitiesZero;
	physx::PxU32 sceneSoftStableWhileSleeping;
	physx::PxU32 sceneSoftCounterWakeIssued;
	physx::PxU32 sceneSoftWokeByCounter;
	physx::PxU32 sceneSoftCounterWakeFrame;
	physx::PxU32 sceneSoftSecondSlept;
	physx::PxU32 sceneSoftSecondSleepFrame;
	physx::PxU32 sceneSoftVelocityWakeIssued;
	physx::PxU32 sceneSoftWokeByVelocity;
	physx::PxU32 sceneSoftVelocityWakeFrame;
	physx::PxU32 sceneSoftMovedAfterVelocityWake;
	physx::PxU32 sceneSoftVelocityStopIssued;
	physx::PxU32 sceneSoftFinalSlept;
	physx::PxU32 sceneSoftFinalSleepFrame;
	physx::PxU32 sceneSoftRigidWakeActorAdded;
	physx::PxU32 sceneSoftWokeByRigid;
	physx::PxU32 sceneSoftRigidWakeFrame;
	physx::PxU32 sceneSoftMovedAfterRigidWake;
	physx::PxU32 sceneMixedFirstSlept;
	physx::PxU32 sceneMixedFirstSleepFrame;
	physx::PxU32 sceneMixedFirstStable;
	physx::PxU32 sceneMixedSecondStayedAwake;
	physx::PxU32 sceneMixedSecondMoved;
	physx::PxU32 sceneSoftChurnRemoveCount;
	physx::PxU32 sceneSoftChurnReaddCount;
	physx::PxU32 sceneSoftChurnCycles;
	physx::PxU32 sceneSoftChurnPostCompactMoveCount;
	physx::PxU32 sceneSoftChurnStable;
	physx::PxU32 sceneBufferMutationIssued;
	physx::PxU32 sceneBufferMutationWoke;
	physx::PxU32 sceneBufferMutationApplied;
	physx::PxU32 sceneBufferDriveIssued;
	physx::PxU32 sceneBufferPinHeld;
	physx::PxU32 sceneBufferDynamicMoved;
	physx::PxU32 sceneBufferInvMassRestored;
	physx::PxU32 sceneBufferRestoredMoved;
	physx::PxU32 sceneBufferResetIssued;
	physx::PxU32 sceneWorldPinCreated;
	physx::PxU32 sceneWorldPinHeld;
	physx::PxU32 sceneWorldPinActorReadded;
	physx::PxU32 sceneWorldPinReleased;
	physx::PxU32 sceneWorldPinMovedAfterRelease;
	physx::PxU32 sceneRigidAttachmentActorAdded;
	physx::PxU32 sceneRigidAttachmentInitiallySleeping;
	physx::PxU32 sceneRigidAttachmentCreated;
	physx::PxU32 sceneRigidAttachmentRigidWoke;
	physx::PxU32 sceneRigidAttachmentRigidMoved;
	physx::PxU32 sceneRigidAttachmentHeldAcrossReadd;
	physx::PxU32 sceneRigidAttachmentReleased;
	physx::PxU32 sceneRigidAttachmentSeparatedAfterRelease;
	physx::PxU32 sceneArticulationCreated;
	physx::PxU32 sceneArticulationAdded;
	physx::PxU32 sceneArticulationInitiallySleeping;
	physx::PxU32 sceneArticulationWoke;
	physx::PxU32 sceneArticulationJointSubspaceHeld;
	physx::PxU32 sceneArticulationRootStable;
	physx::PxU32 sceneElementFilterCreated;
	physx::PxU32 sceneElementFilterActorReadded;
	physx::PxU32 sceneElementFilterSuppressedContact;
	physx::PxU32 sceneElementFilterReleased;
	physx::PxU32 sceneElementFilterContactRestored;
	physx::PxU32 scenePartialFilterUnfilteredContactHeld;
	physx::PxU32 scenePartialFilterExactOwnership;
	physx::PxU32 sceneKinematicActorAdded;
	physx::PxU32 sceneKinematicTargetIssued;
	physx::PxU32 sceneKinematicTargetReached;
	physx::PxU32 sceneKinematicSoftWoke;
	physx::PxU32 sceneKinematicSoftMoved;
	physx::PxU32 sceneKinematicContactObserved;
	physx::PxU32 sceneVolumeTargetBound;
	physx::PxU32 sceneVolumeTargetMutated;
	physx::PxU32 sceneVolumeTargetWoke;
	physx::PxU32 sceneVolumeTargetReached;
	physx::PxU32 sceneVolumePartialInactiveIgnored;
	physx::PxU32 sceneVolumePartialActivated;
	physx::PxU32 sceneVolumePartialActivatedReached;
	physx::PxU32 sceneSecondSceneCreated;
	physx::PxU32 sceneSecondSceneSolverMatched;
	physx::PxU32 scenePrimarySceneReleased;
	physx::PxU32 sceneSecondSceneReleased;
	physx::PxU32 sceneMultiPrimaryStable;
	physx::PxU32 sceneMultiPrimaryDetachedStable;
	physx::PxU32 sceneMultiSecondaryUpdatedBeforeRelease;
	physx::PxU32 sceneMultiSecondaryUpdatedAfterRelease;
	physx::PxU32 sceneSoftSoftBothSlept;
	physx::PxU32 sceneSoftSoftDriveIssued;
	physx::PxU32 sceneSoftSoftDriverWoke;
	physx::PxU32 sceneSoftSoftTargetWoke;
	physx::PxU32 sceneSoftSoftTargetWakeFrame;
	physx::PxU32 sceneSoftSoftTargetMoved;
	physx::PxU32 sceneSoftSoftResetIssued;
	physx::PxU32 sceneSoftSoftBothFinalSlept;
	physx::PxU32 motionMaxVelocityBounded;
	physx::PxU32 motionSettlingApplied;
	physx::PxU32 motionSettlingSlept;
	physx::PxU32 motionControlStayedAwake;
	physx::PxU32 depenetrationLimitApplied;
	physx::PxU32 depenetrationFirstStepBounded;
	physx::PxU32 depenetrationControlSeparated;
	physx::PxU32 depenetrationGradualRecovery;
	physx::PxU32 speculativeCcdFlagApplied;
	physx::PxU32 speculativeCcdPreventedTunneling;
	physx::PxU32 speculativeCcdNegativeControlTunneled;
	physx::PxU32 movingSphereTargetIssued;
	physx::PxU32 movingSphereCcdResponseObserved;
	physx::PxU32 movingSphereNegativeControlHeld;
	physx::PxU32 dynamicSphereSweepLaunched;
	physx::PxU32 dynamicSphereSweepResponseObserved;
	physx::PxU32 dynamicSphereSweepNegativeControlTunneled;
	physx::PxU32 dynamicSphereSweepTwoSidedResponseObserved;
	physx::PxReal minDetF;
	physx::PxReal maxDetF;
	physx::PxReal minBodyVolumeRatio;
	physx::PxReal maxBodyVolumeRatio;
	physx::PxReal minY;
	physx::PxReal maxY;
	physx::PxReal maxParticleSpeed;
	physx::PxReal finalMinY;
	physx::PxReal finalMaxY;
	physx::PxReal finalMaxParticleSpeed;
	physx::PxReal maxCentroidDrop;
	physx::PxReal sceneDynamicMinY;
	physx::PxReal sceneDynamicFinalY;
	physx::PxReal sceneDynamicMaxDrop;
	physx::PxReal sceneDynamicPreContactMaxDrop;
	physx::PxReal sceneDynamicMaxDownSpeed;
	physx::PxReal sceneSecondDynamicMinY;
	physx::PxReal sceneSecondDynamicFinalY;
	physx::PxReal sceneSecondDynamicMaxDrop;
	physx::PxReal sceneSecondDynamicPreContactMaxDrop;
	physx::PxReal sceneSecondDynamicMaxDownSpeed;
	physx::PxReal sceneSecondVolumeMaxCentroidDrop;
	physx::PxReal sceneSecondVolumeFinalCentroidY;
	physx::PxReal sceneWorldPinMaxDrift;
	physx::PxReal sceneWorldPinReleasedMaxDisplacement;
	physx::PxReal sceneRigidAttachmentMaxDrift;
	physx::PxReal sceneRigidAttachmentMaxRigidDisplacement;
	physx::PxReal sceneRigidAttachmentMaxRigidSpeed;
	physx::PxReal sceneRigidAttachmentReleasedSeparation;
	physx::PxReal sceneArticulationRootMaxDisplacement;
	physx::PxReal sceneArticulationChildMaxForbiddenDisplacement;
	physx::PxReal sceneArticulationChildMaxAngularDisplacement;
	physx::PxReal sceneElementFilterMinY;
	physx::PxReal sceneElementFilterFinalMinY;
	physx::PxReal scenePartialFilterUnfilteredMinY;
	physx::PxReal sceneKinematicMaxPoseError;
	physx::PxReal sceneKinematicSoftDisplacement;
	physx::PxReal sceneKinematicFinalY;
	physx::PxReal sceneVolumeTargetFinalMaxError;
	physx::PxReal sceneVolumeTargetMaxDisplacement;
	physx::PxReal sceneVolumePartialInactiveDecoyDistance;
	physx::PxReal minDynamicSurfaceSeparation;
	physx::PxReal finalDynamicSurfaceSeparation;
	physx::PxReal motionMaxVelocityFirstStepDisplacement;
	physx::PxReal motionMaxVelocityFirstStepSpeed;
	physx::PxReal motionSettlingFinalSpeed;
	physx::PxReal motionControlFinalSpeed;
	physx::PxReal depenetrationLimitedFirstStepRise;
	physx::PxReal depenetrationControlFirstStepRise;
	physx::PxReal depenetrationLimitedFinalRise;
	physx::PxReal depenetrationLimitedMaxSpeed;
	physx::PxReal speculativeCcdPositiveMinY;
	physx::PxReal speculativeCcdPositiveMinSeparation;
	physx::PxReal speculativeCcdNegativeMaxY;
	physx::PxReal movingSpherePositiveDisplacement;
	physx::PxReal movingSphereNegativeDisplacement;
	physx::PxReal movingSpherePositiveMinSeparation;
	physx::PxReal dynamicSphereSweepPositiveSoftDisplacement;
	physx::PxReal dynamicSphereSweepNegativeSoftDisplacement;
	physx::PxReal dynamicSphereSweepPositiveRigidDrop;
	physx::PxReal dynamicSphereSweepNegativeRigidDrop;
	physx::PxReal dynamicSphereSweepPositiveMinSeparation;
	bool solverReadbackMatched;

	DeformableVolumeMetrics();
};

struct SceneCommonValidationConfig
{
	physx::PxU32 expectedFrames;
	physx::PxU32 expectedSceneVolumeCount;
	physx::PxU32 expectedSceneDynamicCount;
	bool centroidDropOptional;

	SceneCommonValidationConfig();
};

bool isSceneCommonResultValid(
	const DeformableVolumeMetrics& metrics,
	const SceneCommonValidationConfig& config,
	physx::PxU32 fatalErrorCount);

enum ComponentValidationCase
{
	eCOMPONENT_DENSE_NO_CONTACT,
	eCOMPONENT_MANY_SMALL_NO_CONTACT,
	eCOMPONENT_GROUND,
	eCOMPONENT_STATIC_BOX,
	eCOMPONENT_SOFT_SOFT,
	eCOMPONENT_CONE_GROUND,
	eCOMPONENT_CURRENT_ALL
};

struct ComponentValidationConfig
{
	physx::PxU32 expectedFrames;
	physx::PxU32 fatalErrorCount;
	ComponentValidationCase caseType;

	ComponentValidationConfig();
};

bool isComponentResultValid(
	const DeformableVolumeMetrics& metrics,
	const ComponentValidationConfig& config);

bool isSoftSleepWakeResultValid(const DeformableVolumeMetrics& metrics);
bool isSoftRigidWakeResultValid(const DeformableVolumeMetrics& metrics);
bool isBufferMutationResultValid(const DeformableVolumeMetrics& metrics);
bool isWorldPinResultValid(const DeformableVolumeMetrics& metrics);
bool isMixedSleepIslandResultValid(
	const DeformableVolumeMetrics& metrics);
bool isSoftChurnResultValid(const DeformableVolumeMetrics& metrics);
bool isVolumeKinematicTargetResultValid(
	const DeformableVolumeMetrics& metrics,
	bool fullTarget);
bool isMultiSceneIsolationResultValid(
	const DeformableVolumeMetrics& metrics);
bool isSoftSoftWakeResultValid(const DeformableVolumeMetrics& metrics);
bool isSoftPairAttachmentResultValid(
	const DeformableVolumeMetrics& metrics);
bool isKinematicRigidResultValid(
	const DeformableVolumeMetrics& metrics);

struct DynamicSceneValidationConfig
{
	physx::PxReal initialY;
	bool smooth;
	bool capsule;
	bool convex;
	bool twoActors;
	bool multiSoftIsland;
	bool churn;

	DynamicSceneValidationConfig();
};

bool isDynamicSceneResultValid(
	const DeformableVolumeMetrics& metrics,
	const DynamicSceneValidationConfig& config);

bool isNoStaticSceneResultValid(
	const DeformableVolumeMetrics& metrics);
bool isGroundSceneResultValid(
	const DeformableVolumeMetrics& metrics);
bool isStaticBoxSceneResultValid(
	const DeformableVolumeMetrics& metrics,
	bool churn);

enum TaskGraphValidationCase
{
	eTASK_GRAPH_PURE_SOFT,
	eTASK_GRAPH_WORLD_PLANE,
	eTASK_GRAPH_RIGID_SDF,
	eTASK_GRAPH_WRITE_BACK,
	eTASK_GRAPH_PIPELINE
};

struct TaskGraphValidationMetrics
{
	physx::PxU32 profiledFrames;
	physx::PxU64 submittedSolveTasks;
	physx::PxU64 completedSolveTasks;
	physx::PxU64 serialSolveTasks;
	physx::PxU64 pureSoftEligibleIslands;
	physx::PxU64 pureSoftEligibleParticles;
	physx::PxU64 submittedPredictionTasks;
	physx::PxU64 completedPredictionTasks;
	physx::PxU32 peakActivePredictionTasks;
	physx::PxU64 serialPredictionStages;
	physx::PxU64 submittedWriteBackTasks;
	physx::PxU64 completedWriteBackTasks;
	physx::PxU32 peakActiveWriteBackTasks;
	physx::PxU64 serialWriteBackStages;
	physx::PxU64 submittedCausalLayerTasks;
	physx::PxU64 completedCausalLayerTasks;
	physx::PxU32 peakActiveCausalLayerTasks;
	physx::PxU64 causalLayerFanIns;
	physx::PxU64 serialCausalLayerFallbacks;
	physx::PxU32 maxCausalLayerOccupancy;
	physx::PxU64 submittedWorldPlaneContactTasks;
	physx::PxU64 completedWorldPlaneContactTasks;
	physx::PxU32 peakActiveWorldPlaneContactTasks;
	physx::PxU64 worldPlaneContactFanIns;
	physx::PxU64 serialWorldPlaneContactFallbacks;
	physx::PxU64 workspaceGrowthEvents;
	physx::PxU64 contactWorkspaceGrowthEvents;
	physx::PxU64 contactSweepScratchGrowthEvents;
	physx::PxU64 contactOutputGrowthEvents;

	TaskGraphValidationMetrics();
};

struct TaskGraphValidationConfig
{
	TaskGraphValidationCase caseType;
	physx::PxU32 entryCount;
	physx::PxU32 dispatcherThreads;
	bool parallelExecution;
	bool sequentialExecution;
	bool requireSteadyStateNoGrowth;

	TaskGraphValidationConfig();
};

bool isTaskGraphResultValid(
	const DeformableVolumeMetrics& metrics,
	const TaskGraphValidationMetrics& taskGraph,
	const TaskGraphValidationConfig& config);

struct VolumeSkinningMetrics
{
	physx::PxU32 initialized;
	physx::PxU32 finiteFrames;
	physx::PxU32 evaluatedFrames;
	physx::PxU32 vertices;
	physx::PxU32 triangles;
	physx::PxReal maxDisplacement;

	VolumeSkinningMetrics();
};

bool isVolumeSkinningResultValid(
	const DeformableVolumeMetrics& metrics,
	const VolumeSkinningMetrics& skinning,
	physx::PxU32 expectedFrames);
bool isMotionControlsResultValid(
	const DeformableVolumeMetrics& metrics,
	physx::PxReal dt);
bool isMaxDepenetrationVelocityResultValid(
	const DeformableVolumeMetrics& metrics,
	physx::PxReal dt);

struct AttachmentValidationConfig
{
	bool rigid;
	bool staticTarget;
	bool kinematic;
	bool articulation;

	AttachmentValidationConfig();
};

bool isAttachmentResultValid(
	const DeformableVolumeMetrics& metrics,
	const AttachmentValidationConfig& config);

struct ElementFilterValidationConfig
{
	bool partial;
	physx::PxReal surfaceTolerance;
	physx::PxReal minSuppressedDepth;
	physx::PxReal contactOffsetLimit;

	ElementFilterValidationConfig();
};

bool finalizeAndValidateElementFilterResult(
	DeformableVolumeMetrics& metrics,
	const ElementFilterValidationConfig& config);

struct SurfaceGapMetrics
{
	physx::PxReal initialSignedSdf;
	physx::PxReal minSignedSdf;
	physx::PxReal finalSignedSdf;
	physx::PxU32 penetrationFrames;
	bool penetrated;

	SurfaceGapMetrics();
};

static const physx::PxU32 MAX_VOLUME_HEALTH_BODIES = 5;

struct VolumeBodyHealthMetrics
{
	physx::PxReal minDetF;
	physx::PxReal maxDetF;
	physx::PxReal minVolumeRatio;
	physx::PxReal maxVolumeRatio;
	physx::PxReal finalMinDetF;
	physx::PxReal finalMaxDetF;
	physx::PxReal finalVolumeRatio;
	physx::PxU32 minDetFFrame;
	physx::PxU32 maxDetFFrame;
	physx::PxU32 minVolumeRatioFrame;

	VolumeBodyHealthMetrics();
};

struct VolumeHealthSample
{
	physx::PxU32 nonFiniteParticleSamples;
	physx::PxU32 invertedElementSamples;
	physx::PxU32 invertedBodiesMask;
	physx::PxU32 firstInversionBody;
	physx::PxU32 firstInversionElement;
	physx::PxReal minY;
	physx::PxReal maxY;
	physx::PxReal maxParticleSpeed;
	physx::PxReal minDetF;
	physx::PxReal maxDetF;
	physx::PxReal minBodyVolumeRatio;
	physx::PxReal maxBodyVolumeRatio;

	VolumeHealthSample();
};

class VolumeHealthMonitor
{
  public:
	VolumeHealthMonitor();

	bool initialize(
		physx::PxDeformableVolume* const* volumes,
		physx::PxU32 volumeCount);
	bool sample(
		physx::PxU32 completedFrame,
		VolumeHealthSample& sample);
	void reset();

	bool empty() const;
	physx::PxU32 getBodyCount() const;
	physx::PxDeformableVolume* getBody(physx::PxU32 bodyIndex) const;
	const VolumeBodyHealthMetrics& getBodyMetrics(
		physx::PxU32 bodyIndex) const;

  private:
	struct TetRestState
	{
		physx::PxMat33 dmInv;
		physx::PxReal restVolume;

		TetRestState();
	};

	struct VolumeRestState
	{
		physx::PxDeformableVolume* volume;
		physx::PxArray<TetRestState> tets;
		physx::PxReal totalRestVolume;

		VolumeRestState();
	};

	physx::PxArray<VolumeRestState> mRestStates;
	VolumeBodyHealthMetrics mBodyMetrics[MAX_VOLUME_HEALTH_BODIES];
};

bool isVolumeHealthWithinLimits(
	physx::PxU32 nonFiniteParticleSamples,
	physx::PxU32 invertedElementSamples,
	physx::PxReal minDetF, physx::PxReal maxDetF,
	physx::PxReal minBodyVolumeRatio,
	physx::PxReal maxBodyVolumeRatio,
	physx::PxReal requiredMinDetF,
	physx::PxReal requiredMaxDetF,
	physx::PxReal requiredMinVolumeRatio,
	physx::PxReal requiredMaxVolumeRatio);

struct OgcSandwichMetrics
{
	physx::PxU64 generatedRigidContacts;
	physx::PxU64 nativeIslandSteps;
	physx::PxReal initialUpperBottomOffset;
	physx::PxReal initialLowerTopOffset;
	physx::PxReal maxUpperCompression;
	physx::PxReal maxLowerCompression;
	physx::PxReal upperJawMinDetF;
	physx::PxReal lowerJawMinDetF;
	physx::PxReal minUpperJawSignedSdf;
	physx::PxReal minLowerJawSignedSdf;
	physx::PxReal finalUpperJawSignedSdf;
	physx::PxReal finalLowerJawSignedSdf;
	physx::PxU32 upperJawTrianglePenetrationFrames;
	physx::PxU32 lowerJawTrianglePenetrationFrames;
	bool upperJawTrianglePenetrated;
	bool lowerJawTrianglePenetrated;
	physx::PxVec3 initialBoxPosition;
	physx::PxReal maxBoxLateralOffset;
	physx::PxReal maxBoxLateralSpeed;
	physx::PxReal maxBoxNormalOffset;
	physx::PxReal maxBoxNormalSpeed;
	bool collisionTelemetryEnabled;
	bool initialized;

	OgcSandwichMetrics();
};

struct OgcSandwichFrameSample
{
	physx::PxU64 generatedRigidContacts;
	physx::PxVec3 boxPosition;
	physx::PxVec3 boxVelocity;

	OgcSandwichFrameSample();
};

bool isOgcSandwichSurfaceSeparated(const OgcSandwichMetrics& metrics);
bool isOgcSandwichCompressed(
	const OgcSandwichMetrics& metrics,
	physx::PxReal minJawCompression);
bool isOgcSandwichContained(
	const OgcSandwichMetrics& metrics,
	physx::PxReal maxLateralOffset,
	physx::PxReal maxLateralSpeed,
	physx::PxReal maxNormalOffset,
	physx::PxReal maxNormalSpeed);

class OgcSandwichMonitor
{
  public:
	OgcSandwichMonitor();

	bool initialize(
		physx::PxDeformableVolume& upperJaw,
		physx::PxDeformableVolume& lowerJaw,
		physx::PxRigidDynamic& box,
		const physx::PxVec3& boxHalfExtents);
	bool sample(
		physx::PxScene& scene,
		const VolumeHealthMonitor& healthMonitor,
		OgcSandwichFrameSample& sample);
	void releaseSources();

	const OgcSandwichMetrics& getMetrics() const;

  private:
	bool measureJawContactPatchOffset(
		physx::PxDeformableVolume& jaw,
		bool lowerFace,
		physx::PxReal& outOffset) const;

	OgcSandwichMetrics mMetrics;
	physx::PxDeformableVolume* mUpperJaw;
	physx::PxDeformableVolume* mLowerJaw;
	physx::PxRigidDynamic* mBox;
	physx::PxArray<physx::PxU32> mUpperJawSurfaceTriangles;
	physx::PxArray<physx::PxU32> mLowerJawSurfaceTriangles;
	physx::PxVec3 mBoxHalfExtents;
};

struct VisualInteractionMetrics
{
	physx::PxU64 generatedRigidContacts;
	physx::PxU64 generatedSoftContacts;
	physx::PxU64 nativeIslandSteps;
	physx::PxReal initialUpperJawSignedSdf;
	physx::PxReal initialLowerJawSignedSdf;
	physx::PxReal minUpperJawSignedSdf;
	physx::PxReal minLowerJawSignedSdf;
	physx::PxReal finalUpperJawSignedSdf;
	physx::PxReal finalLowerJawSignedSdf;
	physx::PxU32 upperJawTrianglePenetrationFrames;
	physx::PxU32 lowerJawTrianglePenetrationFrames;
	physx::PxU32 upperJawTrianglePenetrationFirstFrame;
	physx::PxU32 lowerJawTrianglePenetrationFirstFrame;
	bool upperJawTrianglePenetrated;
	bool lowerJawTrianglePenetrated;
	SurfaceGapMetrics upperJawPedestalGap;
	SurfaceGapMetrics lowerJawPedestalGap;
	SurfaceGapMetrics upperJawGroundGap;
	SurfaceGapMetrics lowerJawGroundGap;
	bool collisionTelemetryEnabled;
	bool initialized;

	VisualInteractionMetrics();
};

struct VisualInteractionFrameSample
{
	physx::PxU64 generatedRigidContacts;
	physx::PxU64 generatedSoftContacts;
	physx::PxVec3 boxPosition;
	physx::PxVec3 boxLinearVelocity;

	VisualInteractionFrameSample();
};

bool isVisualDynamicSurfaceSeparated(
	const VisualInteractionMetrics& metrics);
bool isVisualStaticSurfaceSeparated(
	const VisualInteractionMetrics& metrics);

class VisualInteractionMonitor
{
  public:
	VisualInteractionMonitor();

	bool initialize(
		physx::PxDeformableVolume& upperJaw,
		physx::PxDeformableVolume& lowerJaw,
		physx::PxRigidDynamic& dynamicBox,
		physx::PxRigidStatic& pedestal,
		const physx::PxVec3& dynamicBoxHalfExtents,
		const physx::PxVec3& pedestalHalfExtents,
		physx::PxReal groundHeight);
	bool sample(
		physx::PxScene& scene,
		physx::PxU32 completedFrame,
		VisualInteractionFrameSample& sample);
	void releaseSources();

	const VisualInteractionMetrics& getMetrics() const;

  private:
	VisualInteractionMetrics mMetrics;
	physx::PxDeformableVolume* mUpperJaw;
	physx::PxDeformableVolume* mLowerJaw;
	physx::PxRigidDynamic* mDynamicBox;
	physx::PxRigidStatic* mPedestal;
	physx::PxArray<physx::PxU32> mUpperJawSurfaceTriangles;
	physx::PxArray<physx::PxU32> mLowerJawSurfaceTriangles;
	physx::PxVec3 mDynamicBoxHalfExtents;
	physx::PxVec3 mPedestalHalfExtents;
	physx::PxReal mGroundHeight;
};

static const physx::PxU32 ROTATION_CHECKPOINT_CAPACITY = 8;

struct RotationSamplingConfig
{
	physx::PxU32 earlyEndFrame;
	physx::PxU32 lateBeginFrame;
	physx::PxU32 windowBeginFrame;
	physx::PxU32 windowEndFrame;
	physx::PxU32 checkpointInterval;
	physx::PxU32 checkpointCount;
	physx::PxReal groundEnterHeight;
	physx::PxReal groundExitHeight;

	RotationSamplingConfig();
};

struct RotationMetrics
{
	physx::PxReal maxOrientationChange;
	physx::PxReal maxAngularSpeed;
	physx::PxReal finalAngularSpeed;
	physx::PxReal earlyMaxAngularSpeed;
	physx::PxReal lateMaxAngularSpeed;
	physx::PxReal windowMinAngularSpeed;
	physx::PxReal windowMaxAngularSpeed;
	physx::PxF64 windowAngularSpeedSum;
	physx::PxU32 windowSampleCount;
	physx::PxReal finalLinearSpeed;
	physx::PxVec3 finalLinearVelocity;
	physx::PxVec3 finalAngularVelocity;
	physx::PxVec3 finalCentroid;
	physx::PxVec3 finalLowestCollisionOffset;
	physx::PxReal finalRigidRollSlipSpeed;
	physx::PxReal finalMinCollisionY;
	physx::PxReal windowMinLinearSpeed;
	physx::PxReal windowMaxLinearSpeed;
	physx::PxF64 windowLinearSpeedSum;
	physx::PxReal checkpointAngularSpeeds[ROTATION_CHECKPOINT_CAPACITY];
	physx::PxReal checkpointLinearSpeeds[ROTATION_CHECKPOINT_CAPACITY];
	physx::PxReal checkpointCentroidY[ROTATION_CHECKPOINT_CAPACITY];
	physx::PxU32 maxAngularSpeedFrame;
	physx::PxU32 groundContactEpisodes;
	physx::PxU32 firstGroundContactFrame;
	physx::PxU32 secondGroundContactFrame;
	physx::PxReal minSecondGroundEpisodeAngularSpeed;
	bool groundContactActive;
	bool initialized;

	RotationMetrics();
};

bool isRotationResponseObserved(
	const RotationMetrics& metrics,
	physx::PxReal minOrientationChange,
	physx::PxReal minAngularSpeed);
bool isRotationLongRunBounded(
	const RotationMetrics& metrics,
	physx::PxU32 completedFrames,
	physx::PxU32 minFrames,
	physx::PxReal maxLateSpeedFloor,
	physx::PxReal maxLateSpeedRatio);
bool isRollingKinematicsValid(
	const RotationMetrics& metrics,
	physx::PxReal minOrientationChange,
	physx::PxReal minAngularSpeed,
	physx::PxReal maxRigidSlipSpeed);

class RotationMonitor
{
  public:
	RotationMonitor();

	bool initialize(
		physx::PxDeformableVolume& volume,
		const RotationSamplingConfig& config);
	bool sample(physx::PxU32 completedFrame);
	void releaseSource();

	const RotationMetrics& getMetrics() const;

  private:
	RotationMetrics mMetrics;
	RotationSamplingConfig mConfig;
	physx::PxDeformableVolume* mVolume;
	physx::PxU32 mAxisVertex0;
	physx::PxU32 mAxisVertex1;
	physx::PxVec3 mInitialAxis;
};

struct SoftContactPhaseMetrics
{
	physx::PxU32 firstSoftContactFrame;
	physx::PxU32 lastSoftContactFrame;
	physx::PxU32 peakSoftContactFrame;
	physx::PxU32 softContactFrames;
	physx::PxU64 generatedGroundContacts;
	physx::PxU64 generatedSoftContacts;
	physx::PxReal preSoftAngularMomentum;
	physx::PxReal preSoftAngularSpeed;
	physx::PxReal peakSoftContactAngularMomentum;
	physx::PxReal peakSoftContactAngularSpeed;
	physx::PxReal lastSoftContactAngularMomentum;
	physx::PxReal lastSoftContactAngularSpeed;
	physx::PxReal finalPostSoftAngularMomentum;
	physx::PxReal finalPostSoftAngularSpeed;
	physx::PxReal lastNoSoftAngularMomentum;
	physx::PxReal lastNoSoftAngularSpeed;
	physx::PxVec3 preSoftAngularVelocity;
	physx::PxVec3 peakSoftContactAngularVelocity;
	physx::PxVec3 lastNoSoftAngularVelocity;
	bool hasPreSoftContactSample;
	bool contactTelemetryEnabled;
	bool initialized;

	SoftContactPhaseMetrics();
};

struct SoftContactPhaseFrameSample
{
	physx::PxU64 generatedGroundContacts;
	physx::PxU64 generatedSoftContacts;

	SoftContactPhaseFrameSample();
};

class SoftContactPhaseMonitor
{
  public:
	SoftContactPhaseMonitor();

	bool initialize(
		physx::PxDeformableVolume& target,
		bool contactTelemetryEnabled);
	bool sample(
		physx::PxScene& scene,
		physx::PxU32 completedFrame,
		SoftContactPhaseFrameSample& sample);
	void releaseSource();

	const SoftContactPhaseMetrics& getMetrics() const;

  private:
	SoftContactPhaseMetrics mMetrics;
	physx::PxDeformableVolume* mTarget;
};

struct SoftSoftTorqueMetrics
{
	physx::PxU32 targetSimulationVertices;
	physx::PxU32 targetCollisionVertices;
	physx::PxU32 driverSimulationVertices;
	physx::PxU32 driverCollisionVertices;
	physx::PxU32 targetDistinctCollisionSimulation;
	physx::PxU32 driverDistinctCollisionSimulation;
	physx::PxU32 isolatedConfiguration;
	physx::PxU32 supportExpansionInstrumentationAvailable;
	physx::PxU32 softContactFrames;
	physx::PxU32 firstContactFrame;
	physx::PxU32 firstRotationFrame;
	physx::PxU32 retainedRotationSamples;
	physx::PxU64 generatedSoftContacts;
	physx::PxU64 generatedGroundContacts;
	physx::PxU64 generatedRigidContacts;
	physx::PxU64 generatedSelfContacts;
	physx::PxReal firstContactCentroidLeverArm;
	physx::PxReal maxCentroidLeverArm;
	physx::PxReal maxAngularMomentum;
	physx::PxReal finalAngularMomentum;
	physx::PxReal maxAngularSpeed;
	physx::PxReal finalAngularSpeed;
	bool initialized;

	SoftSoftTorqueMetrics();
};

struct SoftSoftTorqueFrameSample
{
	physx::PxU64 generatedGroundContacts;
	physx::PxU64 generatedRigidContacts;
	physx::PxU64 generatedSoftContacts;
	physx::PxU64 generatedSelfContacts;
	bool driverBoundsFinite;
	bool targetBoundsFinite;

	SoftSoftTorqueFrameSample();
};

class SoftSoftTorqueMonitor
{
  public:
	SoftSoftTorqueMonitor();

	bool initialize(
		physx::PxDeformableVolume& driver,
		physx::PxDeformableVolume& target,
		bool isolatedConfiguration,
		bool supportExpansionInstrumentationAvailable);
	bool sample(
		physx::PxScene& scene,
		physx::PxU32 completedFrame,
		physx::PxReal minAngularMomentum,
		physx::PxReal minAngularSpeed,
		SoftSoftTorqueFrameSample& sample);
	void releaseSources();

	const SoftSoftTorqueMetrics& getMetrics() const;

  private:
	SoftSoftTorqueMetrics mMetrics;
	physx::PxDeformableVolume* mDriver;
	physx::PxDeformableVolume* mTarget;
};

struct OgcSandwichValidationConfig
{
	physx::PxU32 gateMinFrames;
	physx::PxU64 minNativeIslandSteps;
	physx::PxReal minJawCompression;
	physx::PxReal maxLateralOffset;
	physx::PxReal maxLateralSpeed;
	physx::PxReal maxNormalOffset;
	physx::PxReal maxNormalSpeed;
	physx::PxReal minDetF;
	physx::PxReal maxDetF;

	OgcSandwichValidationConfig();
};

bool isOgcSandwichResultValid(
	const DeformableVolumeMetrics& metrics,
	const OgcSandwichMetrics& sandwich,
	const OgcSandwichValidationConfig& config);

struct VisualShowcaseValidationConfig
{
	physx::PxU32 rotationGateMinFrames;
	physx::PxU32 interactionGateMinFrames;
	physx::PxU32 minSurfaceTriangles;
	physx::PxReal primaryMinOrientationChange;
	physx::PxReal primaryMinAngularSpeed;
	physx::PxReal sphereMinOrientationChange;
	physx::PxReal sphereMinAngularSpeed;
	physx::PxReal minDetF;
	physx::PxReal maxDetF;
	physx::PxReal minVolumeRatio;
	physx::PxReal maxVolumeRatio;
	bool sphereLongRunBounded;

	VisualShowcaseValidationConfig();
};

bool isVisualShowcaseResultValid(
	const DeformableVolumeMetrics& metrics,
	const RotationMetrics& primaryRotation,
	const RotationMetrics& sphereRotation,
	const VisualInteractionMetrics& interaction,
	const VisualShowcaseValidationConfig& config);

struct SphereLongRollValidationConfig
{
	physx::PxReal minDetF;
	physx::PxReal maxDetF;
	physx::PxReal minVolumeRatio;
	physx::PxReal maxVolumeRatio;
	bool rollingKinematicsValid;
	bool longRunRegressionPassed;

	SphereLongRollValidationConfig();
};

bool isSphereLongRollResultValid(
	const DeformableVolumeMetrics& metrics,
	const VolumeBodyHealthMetrics& sphereHealth,
	const SphereLongRollValidationConfig& config);

struct SoftSoftGlancingValidationConfig
{
	physx::PxU32 gateMinFrames;
	physx::PxReal minDeltaSpeed;
	physx::PxReal minDetF;
	physx::PxReal maxDetF;
	physx::PxReal minVolumeRatio;
	physx::PxReal maxVolumeRatio;

	SoftSoftGlancingValidationConfig();
};

bool isSoftSoftGlancingResultValid(
	const DeformableVolumeMetrics& metrics,
	const SoftContactPhaseMetrics& phase,
	const SoftSoftGlancingValidationConfig& config);

struct SoftSoftTorqueValidationConfig
{
	physx::PxU32 gateMinFrames;
	physx::PxU32 minRetentionSamples;
	physx::PxReal minLeverArm;
	physx::PxReal minAngularMomentum;
	physx::PxReal minAngularSpeed;
	physx::PxReal minDetF;
	physx::PxReal maxDetF;
	physx::PxReal minVolumeRatio;
	physx::PxReal maxVolumeRatio;

	SoftSoftTorqueValidationConfig();
};

bool isSoftSoftTorqueResultValid(
	const DeformableVolumeMetrics& metrics,
	const SoftSoftTorqueMetrics& torque,
	const SoftSoftTorqueValidationConfig& config);

struct GroundEmbeddedTetProbeMetrics
{
	physx::PxU32 simulationVertices;
	physx::PxU32 collisionVertices;
	physx::PxU32 simulationTetrahedra;
	physx::PxU32 collisionTetrahedra;
	physx::PxU32 distinctCollisionSimulation;
	physx::PxU32 strictInteriorEmbedding;
	physx::PxU32 selfCollisionDisabled;
	physx::PxU32 speculativeCcdDisabled;
	physx::PxU32 contactTelemetryEnabled;
	physx::PxU32 hasPreGroundSample;
	physx::PxU32 groundContactWindowClosed;
	physx::PxU32 firstGroundContactFrame;
	physx::PxU32 lastGroundContactFrame;
	physx::PxU32 peakGroundRollFrame;
	physx::PxU32 groundContactWindowFrames;
	physx::PxU64 generatedGroundContacts;
	physx::PxU64 generatedRigidContacts;
	physx::PxU64 generatedSoftContacts;
	physx::PxU64 generatedSelfContacts;
	physx::PxReal launchSpeed;
	physx::PxReal initialMass;
	physx::PxReal initialRmsRadius;
	physx::PxVec3 rollAxis;
	physx::PxVec3 preGroundAngularMomentum;
	physx::PxVec3 preGroundAngularVelocity;
	physx::PxVec3 peakDeltaAngularMomentum;
	physx::PxVec3 peakDeltaAngularVelocity;
	physx::PxReal peakExpectedRollAngularMomentum;
	physx::PxReal peakExpectedRollAngularSpeed;
	physx::PxReal peakNormalizedRollMomentum;
	physx::PxReal peakNormalizedRollOmega;
	bool initialized;

	GroundEmbeddedTetProbeMetrics();
};

struct GroundEmbeddedTetProbeFrameSample
{
	physx::PxU64 generatedGroundContacts;
	physx::PxU64 generatedRigidContacts;
	physx::PxU64 generatedSoftContacts;
	physx::PxU64 generatedSelfContacts;

	GroundEmbeddedTetProbeFrameSample();
};

bool isGroundEmbeddedTetResultValid(
	const DeformableVolumeMetrics& metrics,
	const GroundEmbeddedTetProbeMetrics& probeMetrics,
	physx::PxReal minNormalizedRoll);

bool sampleGroundEmbeddedTetProbe(
	physx::PxScene& scene,
	physx::PxDeformableVolume& volume,
	physx::PxU32 completedFrame,
	GroundEmbeddedTetProbeMetrics& metrics,
	GroundEmbeddedTetProbeFrameSample& sample);

struct ReverseFeatureMetrics
{
	physx::PxU32 faceResponseObserved;
	physx::PxU32 vertexSdfExcluded;
	physx::PxU32 negativeControlPassed;
	physx::PxU32 nonFiniteSamples;
	physx::PxReal positiveDisplacement;
	physx::PxReal positiveDrop;
	physx::PxReal negativeDrop;
	physx::PxReal faceSeparation;
	physx::PxReal minimumVertexSeparation;

	ReverseFeatureMetrics();
};

bool updateReverseFeatureMetrics(
	ReverseFeatureMetrics& metrics,
	const physx::PxVec3& positiveCentroid,
	const physx::PxVec3& negativeCentroid,
	const physx::PxVec3& positiveInitialCentroid,
	const physx::PxVec3& negativeInitialCentroid,
	physx::PxReal faceSeparation,
	physx::PxReal minimumVertexSeparation);

bool isReverseFeatureResponseValid(const ReverseFeatureMetrics& metrics);
bool isReverseFeatureSceneResultValid(
	const DeformableVolumeMetrics& metrics,
	const ReverseFeatureMetrics& reverseFeature);
bool isRigidTriangleSteadyContactResultValid(
	const DeformableVolumeMetrics& metrics,
	physx::PxU32 profiledFrames,
	physx::PxU64 faceTests,
	physx::PxU64 edgeTests,
	physx::PxU64 vertexTests);

struct ReverseSweptMetrics
{
	physx::PxU32 responseObserved;
	physx::PxU32 negativeControlPassed;
	physx::PxU32 twoSidedResponseObserved;
	physx::PxU32 vertexSweepExcluded;
	physx::PxU32 nonFiniteSamples;
	physx::PxReal positiveDisplacement;
	physx::PxReal negativeDisplacement;
	physx::PxReal positiveDrop;
	physx::PxReal negativeDrop;
	physx::PxReal positiveRigidDrop;
	physx::PxReal negativeRigidDrop;
	physx::PxReal faceSeparation;
	physx::PxReal minimumVertexSweepSeparation;

	ReverseSweptMetrics();
};

struct DeformingReverseSweptMetrics
{
	physx::PxU32 geometricSweepIsolated;
	physx::PxReal endpointMinSeparation;
	physx::PxReal midSweepMinSeparation;
	physx::PxReal responseDelta;

	DeformingReverseSweptMetrics();
};

struct RotationalSweepMetrics
{
	physx::PxU32 sweepIsolated;
	physx::PxU32 nonFiniteSamples;
	physx::PxReal endpointMinSeparation;
	physx::PxReal midSweepMinSeparation;
	physx::PxReal positiveAngularTravel;
	physx::PxReal negativeAngularTravel;

	RotationalSweepMetrics();
};

struct TriangleSurfaceSweptValidationConfig
{
	bool reverse;
	bool heightField;
	bool rotational;

	TriangleSurfaceSweptValidationConfig();
};

bool isTriangleSurfaceSweptResponseValid(
	const ReverseSweptMetrics& metrics,
	const RotationalSweepMetrics& rotationalMetrics,
	const TriangleSurfaceSweptValidationConfig& config);
bool isTriangleSurfaceSweptSceneResultValid(
	const DeformableVolumeMetrics& metrics,
	const ReverseSweptMetrics& sweptMetrics,
	const RotationalSweepMetrics& rotationalMetrics,
	const TriangleSurfaceSweptValidationConfig& config);

struct ReverseSweptValidationConfig
{
	bool staticTarget;
	bool kinematicTarget;
	bool dynamicTarget;
	bool deforming;
	bool capsule;
	bool convex;
	bool rotational;

	ReverseSweptValidationConfig();
};

bool normalizeDynamicReverseSweptResponse(ReverseSweptMetrics& metrics);
bool isReverseSweptResponseValid(
	const ReverseSweptMetrics& metrics,
	const DeformingReverseSweptMetrics& deformingMetrics,
	const RotationalSweepMetrics& rotationalMetrics,
	const ReverseSweptValidationConfig& config);
bool isReverseSweptSceneResultValid(
	const DeformableVolumeMetrics& metrics,
	const ReverseSweptMetrics& sweptMetrics,
	const DeformingReverseSweptMetrics& deformingMetrics,
	const RotationalSweepMetrics& rotationalMetrics,
	const ReverseSweptValidationConfig& config,
	bool staticTarget);

struct KinematicFiniteSweptMetrics
{
	physx::PxU32 targetIssued;
	physx::PxU32 responseObserved;
	physx::PxU32 negativeControlPassed;
	physx::PxReal positiveDisplacement;
	physx::PxReal negativeDisplacement;
	physx::PxReal positiveMinSeparation;

	KinematicFiniteSweptMetrics();
};

bool isKinematicFiniteSweptResponseValid(
	const KinematicFiniteSweptMetrics& metrics,
	const RotationalSweepMetrics& rotationalMetrics,
	bool rotational, bool convex);
bool isKinematicFiniteSweptSceneResultValid(
	const DeformableVolumeMetrics& metrics,
	const KinematicFiniteSweptMetrics& sweptMetrics,
	const RotationalSweepMetrics& rotationalMetrics,
	bool rotational, bool convex);

struct DynamicFiniteSweptMetrics
{
	physx::PxU32 launched;
	physx::PxU32 responseObserved;
	physx::PxU32 negativeControlPassed;
	physx::PxU32 twoSidedResponseObserved;
	physx::PxReal positiveSoftDisplacement;
	physx::PxReal negativeSoftDisplacement;
	physx::PxReal positiveRigidDrop;
	physx::PxReal negativeRigidDrop;
	physx::PxReal positiveMinSeparation;

	DynamicFiniteSweptMetrics();
};

bool normalizeDynamicFiniteSweptResponse(
	DynamicFiniteSweptMetrics& metrics);
bool isDynamicFiniteSweptResponseValid(
	const DynamicFiniteSweptMetrics& metrics,
	const RotationalSweepMetrics& rotationalMetrics,
	bool rotational, bool convex);
bool isDynamicFiniteSweptSceneResultValid(
	const DeformableVolumeMetrics& metrics,
	const DynamicFiniteSweptMetrics& sweptMetrics,
	const RotationalSweepMetrics& rotationalMetrics,
	bool rotational, bool convex);

bool isSpeculativeCcdSceneResultValid(
	const DeformableVolumeMetrics& metrics,
	bool plane, bool finiteSmooth);

physx::PxReal getCapsuleSignedSeparation(
	const physx::PxVec3& point,
	const physx::PxTransform& capsulePose,
	physx::PxReal capsuleRadius,
	physx::PxReal capsuleHalfHeight);

physx::PxReal getCapsuleSignedSeparation(
	const physx::PxVec3& point,
	const physx::PxVec3& capsuleCenter,
	physx::PxReal capsuleRadius,
	physx::PxReal capsuleHalfHeight);

class SweptGeometrySampler
{
  public:
	SweptGeometrySampler(
		const physx::PxArray<physx::PxVec3>& reverseInitialPositions,
		const physx::PxArray<physx::PxVec3>& deformingFreeEndPositions,
		const physx::PxArray<physx::PxVec3>& rotationalInitialPositions,
		physx::PxConvexMesh* convexMesh,
		physx::PxTriangleMesh* triangleMesh,
		physx::PxHeightField* heightField);

	bool measureSphereReverseSweptSeparations(
		physx::PxDeformableVolume* volume,
		const physx::PxVec3& rigidCenterCurrent,
		const physx::PxVec3& rigidCenterStart,
		const physx::PxVec3& rigidCenterEnd,
		physx::PxReal rigidRadius,
		physx::PxReal capsuleHalfHeight,
		bool convex,
		physx::PxReal& faceSeparation,
		physx::PxReal& minimumVertexSweepSeparation) const;

	bool measureDeformingReverseSweptProof(
		physx::PxDeformableVolume* volume,
		const physx::PxTransform& rigidPose,
		physx::PxReal radius,
		physx::PxReal capsuleHalfHeight,
		bool convex,
		physx::PxReal& endpointMinSeparation,
		physx::PxReal& midSweepMinSeparation,
		physx::PxReal& minimumVertexSweepSeparation) const;

	bool measureSmoothReverseSeparations(
		physx::PxDeformableVolume* volume,
		const physx::PxVec3& rigidCenter,
		physx::PxReal rigidRadius,
		physx::PxReal capsuleHalfHeight,
		bool convex, bool triangleMesh, bool heightField,
		physx::PxReal& faceSeparation,
		physx::PxReal& minimumVertexSeparation) const;

	bool measureRotationalConvexPointSweepSeparations(
		const physx::PxTransform& startPose,
		const physx::PxTransform& endPose,
		physx::PxReal& endpointMinSeparation,
		physx::PxReal& midSweepMinSeparation) const;

	bool measureRotationalCapsuleReverseSweptSeparations(
		physx::PxDeformableVolume* volume,
		const physx::PxTransform& currentPose,
		const physx::PxTransform& startPose,
		const physx::PxTransform& endPose,
		physx::PxReal capsuleRadius,
		physx::PxReal capsuleHalfHeight,
		physx::PxReal& faceSeparation,
		physx::PxReal& minimumVertexSweepSeparation,
		physx::PxReal& endpointMinSeparation,
		physx::PxReal& midSweepMinSeparation) const;

	bool measureRotationalConvexReverseSweptSeparations(
		physx::PxDeformableVolume* volume,
		const physx::PxTransform& currentPose,
		const physx::PxTransform& startPose,
		const physx::PxTransform& endPose,
		physx::PxReal& faceSeparation,
		physx::PxReal& minimumVertexSweepSeparation,
		physx::PxReal& endpointMinSeparation,
		physx::PxReal& midSweepMinSeparation) const;

	bool measureRotationalTriangleSurfaceSweepSeparations(
		const physx::PxTransform& startPose,
		const physx::PxTransform& endPose,
		bool heightField, bool reverse,
		physx::PxReal& endpointMinSeparation,
		physx::PxReal& midSweepMinSeparation,
		physx::PxReal& minimumVertexSweepSeparation) const;

  private:
	bool measureCapsuleFaceSeparation(
		physx::PxDeformableVolume* volume,
		const physx::PxTransform& capsulePose,
		physx::PxReal capsuleRadius,
		physx::PxReal capsuleHalfHeight,
		bool useInitialPositions,
		physx::PxReal& faceSeparation) const;

	bool measureConvexFaceSeparation(
		physx::PxDeformableVolume* volume,
		const physx::PxTransform& convexPose,
		bool useInitialPositions,
		physx::PxReal& faceSeparation) const;

	const physx::PxArray<physx::PxVec3>& mReverseInitialPositions;
	const physx::PxArray<physx::PxVec3>& mDeformingFreeEndPositions;
	const physx::PxArray<physx::PxVec3>& mRotationalInitialPositions;
	physx::PxConvexMesh* mConvexMesh;
	physx::PxTriangleMesh* mTriangleMesh;
	physx::PxHeightField* mHeightField;
};

bool measureCollisionSurfaceBoxGap(
	physx::PxDeformableVolume& volume,
	const physx::PxArray<physx::PxU32>& surfaceTriangles,
	const physx::PxTransform& boxPose,
	const physx::PxVec3& boxHalfExtents,
	physx::PxReal& outMinSignedSdf,
	bool& outTrianglePenetrated);

bool measureCollisionSurfaceGroundGap(
	physx::PxDeformableVolume& volume,
	const physx::PxArray<physx::PxU32>& surfaceTriangles,
	physx::PxReal groundHeight,
	physx::PxReal& outMinSignedSdf,
	bool& outTrianglePenetrated);

bool isSurfaceGapSeparated(const SurfaceGapMetrics& gap);

bool measureVolumeAngularState(
	physx::PxDeformableVolume& volume,
	physx::PxVec3& centroid,
	physx::PxVec3& angularMomentum,
	physx::PxVec3& angularVelocity);

void initializeSurfaceGapMetrics(
	SurfaceGapMetrics& gap,
	physx::PxReal initialSignedSdf,
	bool initiallyPenetrated);

void updateSurfaceGapMetrics(
	SurfaceGapMetrics& gap,
	physx::PxReal signedSdf,
	bool penetrated);

} // namespace SnippetDeformableVolumeAVBDValidation

#endif
