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

// Headless fixture validation and structured result orchestration.

#include "SnippetDeformableVolumeAVBDHeadless.h"
#include "SnippetDeformableVolumeAVBDFixtures.h"

#include <cfloat>

using namespace physx;
using namespace SnippetDeformableVolumeAVBDFixtures;
using namespace SnippetDeformableVolumeAVBDReport;
using namespace SnippetDeformableVolumeAVBDValidation;

namespace SnippetDeformableVolumeAVBDHeadless
{

static const PxU32 SCENE_CPU_VISUAL_ROTATION_GATE_MIN_FRAMES = 80;
static const PxReal SCENE_CPU_VISUAL_MIN_ORIENTATION_CHANGE = 0.25f;
static const PxReal SCENE_CPU_VISUAL_MIN_ANGULAR_SPEED = 0.5f;
static const PxReal SCENE_CPU_VISUAL_SPHERE_MIN_ORIENTATION_CHANGE = 0.1f;
static const PxReal SCENE_CPU_VISUAL_SPHERE_MIN_ANGULAR_SPEED = 0.2f;
static const PxReal SCENE_CPU_VISUAL_MIN_DET_F = 0.35f;
static const PxReal SCENE_CPU_VISUAL_MAX_DET_F = 2.5f;
static const PxReal SCENE_CPU_VISUAL_MIN_VOLUME_RATIO = 0.9f;
static const PxReal SCENE_CPU_VISUAL_MAX_VOLUME_RATIO = 1.1f;
static const PxReal SCENE_CPU_VISUAL_SHOWCASE_MIN_DET_F = 0.03f;
static const PxReal SCENE_CPU_VISUAL_SHOWCASE_MIN_VOLUME_RATIO = 0.89f;
static const PxReal SCENE_CPU_ELEMENT_FILTER_MIN_SUPPRESSED_DEPTH = 0.02f;
static const PxReal SCENE_CPU_ELEMENT_FILTER_SURFACE_TOLERANCE = 0.005f;
static const PxReal SCENE_CPU_ELEMENT_FILTER_CONTACT_OFFSET_LIMIT = 0.06f;
static const PxU32 SCENE_CPU_VISUAL_INTERACTION_GATE_MIN_FRAMES = 120u;
static const PxU32 SCENE_CPU_VISUAL_MIN_SURFACE_TRIANGLES = 1152;
static const PxU32 SCENE_CPU_OGC_SANDWICH_GATE_MIN_FRAMES = 90u;
static const PxU32 SCENE_CPU_OGC_SANDWICH_MIN_NATIVE_ISLAND_STEPS = 18u;
static const PxReal SCENE_CPU_OGC_SANDWICH_MIN_JAW_COMPRESSION = 0.005f;
static const PxReal SCENE_CPU_OGC_SANDWICH_MAX_LATERAL_OFFSET = 0.35f;
static const PxReal SCENE_CPU_OGC_SANDWICH_MAX_LATERAL_SPEED = 1.0f;
static const PxReal SCENE_CPU_OGC_SANDWICH_MAX_NORMAL_OFFSET = 0.35f;
static const PxReal SCENE_CPU_OGC_SANDWICH_MAX_NORMAL_SPEED = 1.0f;
static const PxU32 SCENE_CPU_SOFT_SOFT_TORQUE_GATE_MIN_FRAMES = 120u;
static const PxU32 SCENE_CPU_SOFT_SOFT_TORQUE_MIN_RETENTION_SAMPLES = 16u;
static const PxReal SCENE_CPU_SOFT_SOFT_TORQUE_MIN_LEVER_ARM = 0.15f;
static const PxU32 SCENE_CPU_SPHERE_SOFT_SOFT_GLANCING_GATE_MIN_FRAMES = 120u;
static const PxReal SCENE_CPU_SPHERE_SOFT_SOFT_GLANCING_MIN_DELTA_SPEED = 1.0e-3f;
static const PxReal SCENE_CPU_GROUND_EMBEDDED_TET_PROBE_MIN_NORMALIZED_ROLL =
	128.0f * FLT_EPSILON;

static TriangleSurfaceSweptValidationConfig
getTriangleSurfaceSweptValidationConfig(const std::string& caseName)
{
	TriangleSurfaceSweptValidationConfig config;
	config.reverse =
		isSceneCpuVolumeTriangleSurfaceReverseSweptCcdCase(caseName);
	config.heightField =
		isSceneCpuVolumeHeightFieldSweptCcdCase(caseName);
	config.rotational =
		isSceneCpuVolumeRotationalTriangleSurfaceSweptCcdCase(caseName);
	return config;
}

static ReverseSweptValidationConfig getReverseSweptValidationConfig(
	const std::string& caseName)
{
	ReverseSweptValidationConfig config;
	config.deforming =
		isSceneCpuVolumeDeformingReverseSweptCcdCase(caseName);
	config.capsule =
		isSceneCpuVolumeCapsuleReverseSweptCcdCase(caseName);
	config.convex =
		isSceneCpuVolumeConvexReverseSweptCcdCase(caseName);
	config.rotational =
		isSceneCpuVolumeRotationalCapsuleReverseSweptCcdCase(caseName) ||
		isSceneCpuVolumeRotationalConvexReverseSweptCcdCase(caseName);
	config.staticTarget = config.deforming ||
		caseName.find("scene-volume-static-") != std::string::npos;
	config.kinematicTarget =
		caseName.find("kinematic") != std::string::npos;
	config.dynamicTarget =
		!config.staticTarget && !config.kinematicTarget;
	return config;
}

HeadlessResultContext::HeadlessResultContext()
	: caseName(NULL), solverName(NULL), frames(0), dt(0.0f),
	  dispatcherThreads(0), parallelExecution(false),
	  sequentialExecution(false), metrics(NULL), performance(NULL),
	  ogcSandwichMetrics(NULL), visualRotationMetrics(NULL),
	  visualPrimaryCubeRotationMetrics(NULL),
	  visualSphereRotationMetrics(NULL), visualInteractionMetrics(NULL),
	  softContactPhaseMetrics(NULL), softSoftTorqueMetrics(NULL),
	  groundEmbeddedTetMetrics(NULL), volumeHealthMonitor(NULL),
	  volumeSkinningMetrics(NULL), reverseFeatureMetrics(NULL),
	  reverseSweptMetrics(NULL), deformingReverseSweptMetrics(NULL),
	  rotationalSweepMetrics(NULL), dynamicInitialY(0.0f),
	  secondDynamicInitialY(0.0f), visualSphereLongRunBounded(false),
	  sphereRollingKinematicsValid(false),
	  sphereLongRollRegressionPassed(false),
	  sphereRollWindowBeginFrame(0), sphereRollWindowEndFrame(0),
	  sphereRollCheckpointCount(0), sphereRollCheckpointInterval(0),
	  ogcPressureDriveFrames(0), softSoftTorqueMinAngularMomentum(0.0f),
	  softSoftTorqueMinAngularSpeed(0.0f), fatalErrors(0),
	  warningErrors(0)
{
}
static bool validateSceneCpuTaskGraphResult(
	const HeadlessResultContext& context,
	TaskGraphValidationCase caseType, PxU32 entryCount = 0)
{
	const DeformableVolumePerformanceMetrics& gPerformance =
		*context.performance;
	const DeformableVolumeMetrics& gMetrics = *context.metrics;
	TaskGraphValidationMetrics taskGraph;
	taskGraph.profiledFrames = gPerformance.profiledFrames;
	taskGraph.submittedSolveTasks =
		gPerformance.taskGraphSubmittedSolveTasks;
	taskGraph.completedSolveTasks =
		gPerformance.taskGraphCompletedSolveTasks;
	taskGraph.serialSolveTasks = gPerformance.taskGraphSerialSolveTasks;
	taskGraph.pureSoftEligibleIslands =
		gPerformance.taskGraphPureSoftEligibleIslands;
	taskGraph.pureSoftEligibleParticles =
		gPerformance.taskGraphPureSoftEligibleParticles;
	taskGraph.submittedPredictionTasks =
		gPerformance.taskGraphSubmittedPredictionTasks;
	taskGraph.completedPredictionTasks =
		gPerformance.taskGraphCompletedPredictionTasks;
	taskGraph.peakActivePredictionTasks =
		gPerformance.taskGraphPeakActivePredictionTasks;
	taskGraph.serialPredictionStages =
		gPerformance.taskGraphSerialPredictionStages;
	taskGraph.submittedWriteBackTasks =
		gPerformance.taskGraphSubmittedWriteBackTasks;
	taskGraph.completedWriteBackTasks =
		gPerformance.taskGraphCompletedWriteBackTasks;
	taskGraph.peakActiveWriteBackTasks =
		gPerformance.taskGraphPeakActiveWriteBackTasks;
	taskGraph.serialWriteBackStages =
		gPerformance.taskGraphSerialWriteBackStages;
	taskGraph.submittedCausalLayerTasks =
		gPerformance.taskGraphSubmittedCausalLayerTasks;
	taskGraph.completedCausalLayerTasks =
		gPerformance.taskGraphCompletedCausalLayerTasks;
	taskGraph.peakActiveCausalLayerTasks =
		gPerformance.taskGraphPeakActiveCausalLayerTasks;
	taskGraph.causalLayerFanIns =
		gPerformance.taskGraphCausalLayerFanIns;
	taskGraph.serialCausalLayerFallbacks =
		gPerformance.taskGraphSerialCausalLayerFallbacks;
	taskGraph.maxCausalLayerOccupancy =
		gPerformance.taskGraphMaxCausalLayerOccupancy;
	taskGraph.submittedWorldPlaneContactTasks =
		gPerformance.taskGraphSubmittedWorldPlaneContactTasks;
	taskGraph.completedWorldPlaneContactTasks =
		gPerformance.taskGraphCompletedWorldPlaneContactTasks;
	taskGraph.peakActiveWorldPlaneContactTasks =
		gPerformance.taskGraphPeakActiveWorldPlaneContactTasks;
	taskGraph.worldPlaneContactFanIns =
		gPerformance.taskGraphWorldPlaneContactFanIns;
	taskGraph.serialWorldPlaneContactFallbacks =
		gPerformance.taskGraphSerialWorldPlaneContactFallbacks;
	taskGraph.workspaceGrowthEvents =
		gPerformance.solverStages.workspaceGrowthEvents;
	taskGraph.contactWorkspaceGrowthEvents =
		gPerformance.solverStages.contactWorkspaceGrowthEvents;
	taskGraph.contactSweepScratchGrowthEvents =
		gPerformance.solverStages.contactSweepScratchGrowthEvents;
	taskGraph.contactOutputGrowthEvents =
		gPerformance.solverStages.contactOutputGrowthEvents;

	TaskGraphValidationConfig config;
	config.caseType = caseType;
	config.entryCount = entryCount;
	config.dispatcherThreads = context.dispatcherThreads;
	config.parallelExecution =
		context.parallelExecution;
	config.sequentialExecution =
		context.sequentialExecution;
	config.requireSteadyStateNoGrowth =
		context.frames > gPerformance.profiledFrames;
	return isTaskGraphResultValid(gMetrics, taskGraph, config);
}

bool validateHeadlessResult(HeadlessResultContext& context)
{
	const std::string& caseName = *context.caseName;
	DeformableVolumeMetrics& gMetrics = *context.metrics;
	const DeformableVolumePerformanceMetrics& gPerformance =
		*context.performance;
	const RotationMetrics& gSceneCpuVisualPrimaryCubeRotationMetrics =
		*context.visualPrimaryCubeRotationMetrics;
	const RotationMetrics& gSceneCpuVisualSphereRotationMetrics =
		*context.visualSphereRotationMetrics;
	const VisualInteractionMetrics& gSceneCpuVisualInteractionMetrics =
		*context.visualInteractionMetrics;
	const SoftContactPhaseMetrics& gSceneCpuSoftContactPhaseMetrics =
		*context.softContactPhaseMetrics;
	const SoftSoftTorqueMetrics& gSceneCpuSoftSoftTorqueMetrics =
		*context.softSoftTorqueMetrics;
	const GroundEmbeddedTetProbeMetrics&
		gSceneCpuGroundEmbeddedTetProbeMetrics =
			*context.groundEmbeddedTetMetrics;
	const VolumeHealthMonitor& gSceneCpuVolumeHealthMonitor =
		*context.volumeHealthMonitor;
	const VolumeSkinningMetrics& gVolumeSkinningMetrics =
		*context.volumeSkinningMetrics;
	const ReverseFeatureMetrics& gSphereReverseFeatureMetrics =
		*context.reverseFeatureMetrics;
	ReverseSweptMetrics& gSphereReverseSweptMetrics =
		*context.reverseSweptMetrics;
	const DeformingReverseSweptMetrics&
		gDeformingVolumeReverseSweptMetrics =
			*context.deformingReverseSweptMetrics;
	const RotationalSweepMetrics& gCapsuleRotationalSweepMetrics =
		*context.rotationalSweepMetrics;
	const PxReal gSceneCpuDynamicInitialY = context.dynamicInitialY;
	if(isSceneCpuVolumeCase(caseName))
	{
		const bool visualShowcaseCase =
			caseName == "scene-volume-visual-showcase";
		const bool ogcSandwichCase =
			caseName == "scene-volume-ogc-sandwich";
		const bool sphereLongRollCase =
			caseName == "scene-volume-sphere-long-roll";
		const bool sphereSoftSoftGlancingCase =
			caseName == "scene-volume-sphere-soft-soft-glancing";
		const bool softSoftTorqueCase =
			caseName == "scene-volume-soft-soft-torque";
		const bool groundEmbeddedTetProbeCase =
			caseName == "scene-volume-ground-embedded-tet-probe";
		const bool dynamicChurnCase =
			caseName == "scene-volume-dynamic-churn";
		const bool multiDynamicBoxCase =
			caseName == "scene-volume-multi-dynamic-box";
		const bool multiSoftIslandCase =
			caseName == "scene-volume-multi-soft-islands";
		const bool taskGraphPureSoftCase =
			isSceneCpuVolumeTaskGraphPureSoftCase(caseName);
		const bool taskGraphWorldPlaneCase =
			caseName == "scene-volume-taskgraph-world-plane";
		const bool taskGraphRigidBoxSdfCase =
			caseName == "scene-volume-taskgraph-rigid-box-sdf";
		const bool taskGraphRigidSphereSdfCase =
			caseName == "scene-volume-taskgraph-rigid-sphere-sdf";
		const bool taskGraphRigidCapsuleSdfCase =
			caseName == "scene-volume-taskgraph-rigid-capsule-sdf";
		const bool taskGraphRigidConvexSdfCase =
			caseName == "scene-volume-taskgraph-rigid-convex-sdf";
		const bool taskGraphRigidTriangleSurfaceCase =
			caseName == "scene-volume-taskgraph-rigid-triangle-surface";
		const bool taskGraphRigidTriangleSurfaceLargeCase =
			isSceneCpuVolumeTaskGraphRigidTriangleSurfaceLargeCase(caseName);
		const bool taskGraphRigidTriangleSurfaceFeatureOverlapCase =
			isSceneCpuVolumeTaskGraphRigidTriangleSurfaceFeatureOverlapCase(
				caseName);
		const bool taskGraphRigidTriangleSurfaceThresholdCase =
			isSceneCpuVolumeTaskGraphRigidTriangleSurfaceThresholdCase(caseName);
		const bool taskGraphWriteBackFourWayCase =
			isSceneCpuVolumeTaskGraphWriteBackFourWayCase(caseName);
		const bool taskGraphWriteBackCase =
			isSceneCpuVolumeTaskGraphWriteBackCase(caseName);
		const bool taskGraphPipelineCase =
			isSceneCpuVolumeTaskGraphPipelineCase(caseName);
		const bool mixedSleepIslandCase =
			caseName == "scene-volume-mixed-sleep-islands";
		const bool softChurnCase =
			caseName == "scene-volume-soft-churn";
		const bool bufferMutationCase =
			caseName == "scene-volume-buffer-mutation";
		const bool worldPinCase =
			caseName == "scene-volume-world-pin" ||
			caseName == "scene-volume-world-element-attachment";
		const bool rigidAttachmentCase =
			caseName == "scene-volume-rigid-attachment" ||
			caseName == "scene-volume-rigid-element-attachment";
		const bool staticAttachmentCase =
			caseName == "scene-volume-static-attachment" ||
			caseName == "scene-volume-static-element-attachment";
		const bool kinematicAttachmentCase =
			caseName == "scene-volume-kinematic-attachment" ||
			caseName ==
				"scene-volume-kinematic-element-attachment";
		const bool articulationAttachmentCase =
			caseName == "scene-volume-articulation-attachment" ||
			caseName ==
				"scene-volume-articulation-element-attachment";
		const bool attachmentCase =
			rigidAttachmentCase || staticAttachmentCase ||
			kinematicAttachmentCase ||
			articulationAttachmentCase;
		const bool partialElementFilterCase =
			caseName ==
				"scene-volume-partial-element-filter";
		const bool elementFilterCase =
			caseName == "scene-volume-element-filter" ||
			partialElementFilterCase;
		const bool kinematicBoxCase =
			isSceneCpuVolumeKinematicRigidCase(caseName);
		const bool multiSceneIsolationCase =
			caseName == "scene-volume-multi-scene-isolation";
		const bool softSoftWakeCase =
			caseName == "scene-volume-soft-soft-wake";
		const bool softPairAttachmentCase =
			caseName == "scene-volume-volume-attachment";
		const bool fullKinematicTargetCase =
			caseName ==
				"scene-volume-full-kinematic-target";
		const bool partialKinematicTargetCase =
			caseName ==
				"scene-volume-partial-kinematic-target";
		const bool volumeKinematicTargetCase =
			fullKinematicTargetCase ||
			partialKinematicTargetCase;
		const bool motionControlsCase =
			caseName == "scene-volume-motion-controls";
		const bool maxDepenetrationVelocityCase =
			caseName ==
				"scene-volume-max-depenetration-velocity";
		const bool triangleSurfaceSweptCcdCase =
			isSceneCpuVolumeTriangleSurfaceSweptCcdCase(
				caseName);
		const bool rigidTriangleSteadyContactCase =
			isSceneCpuVolumeRigidTriangleSteadyContactCase(caseName);
		const bool sphereReverseSweptCcdCase =
			isSceneCpuVolumeSphereReverseSweptCcdCase(caseName);
		const bool deformingReverseSweptCcdCase =
			isSceneCpuVolumeDeformingReverseSweptCcdCase(
				caseName);
		const bool staticSphereReverseSweptCcdCase =
			deformingReverseSweptCcdCase ||
			caseName ==
				"scene-volume-static-sphere-reverse-swept-ccd" ||
			caseName ==
				"scene-volume-static-capsule-reverse-swept-ccd" ||
			caseName ==
				"scene-volume-static-convex-reverse-swept-ccd";
		const bool kinematicSphereReverseSweptCcdCase =
			caseName ==
				"scene-volume-kinematic-sphere-reverse-swept-ccd" ||
			caseName ==
				"scene-volume-kinematic-capsule-reverse-swept-ccd" ||
			caseName ==
				"scene-volume-rotating-kinematic-capsule-reverse-swept-ccd" ||
			caseName ==
				"scene-volume-rotating-kinematic-convex-reverse-swept-ccd" ||
			caseName ==
				"scene-volume-kinematic-convex-reverse-swept-ccd";
		const bool dynamicSphereReverseSweptCcdCase =
			caseName ==
				"scene-volume-dynamic-sphere-reverse-swept-ccd" ||
			caseName ==
				"scene-volume-dynamic-capsule-reverse-swept-ccd" ||
			caseName ==
				"scene-volume-dynamic-rotating-capsule-reverse-swept-ccd" ||
			caseName ==
				"scene-volume-dynamic-rotating-convex-reverse-swept-ccd" ||
			caseName ==
				"scene-volume-dynamic-convex-reverse-swept-ccd";
		const bool speculativeCcdCase =
			isSceneCpuVolumeSpeculativeCcdCase(caseName) ||
			sphereReverseSweptCcdCase;
		const bool planeSpeculativeCcdCase =
			caseName == "scene-volume-plane-speculative-ccd";
		const bool sphereSpeculativeCcdCase =
			caseName == "scene-volume-sphere-speculative-ccd";
		const bool capsuleSpeculativeCcdCase =
			caseName == "scene-volume-capsule-speculative-ccd";
		const bool convexSpeculativeCcdCase =
			caseName == "scene-volume-convex-speculative-ccd";
		const bool finiteSmoothSpeculativeCcdCase =
			sphereSpeculativeCcdCase || capsuleSpeculativeCcdCase ||
			convexSpeculativeCcdCase;
		const bool rotatingKinematicCapsuleSpeculativeCcdCase =
			caseName ==
				"scene-volume-rotating-kinematic-capsule-speculative-ccd";
		const bool rotatingKinematicConvexSpeculativeCcdCase =
			caseName ==
				"scene-volume-rotating-kinematic-convex-speculative-ccd";
		const bool dynamicRotatingCapsuleSpeculativeCcdCase =
			caseName ==
				"scene-volume-dynamic-rotating-capsule-relative-swept-ccd";
		const bool dynamicRotatingConvexSpeculativeCcdCase =
			caseName ==
				"scene-volume-dynamic-rotating-convex-relative-swept-ccd";
		const bool movingKinematicFiniteSpeculativeCcdCase =
			caseName ==
				"scene-volume-moving-kinematic-sphere-speculative-ccd" ||
			caseName ==
				"scene-volume-moving-kinematic-capsule-speculative-ccd" ||
			rotatingKinematicCapsuleSpeculativeCcdCase ||
			rotatingKinematicConvexSpeculativeCcdCase ||
			caseName ==
				"scene-volume-moving-kinematic-convex-speculative-ccd";
		const bool dynamicFiniteRelativeSweptCcdCase =
			caseName ==
				"scene-volume-dynamic-sphere-relative-swept-ccd" ||
			caseName ==
				"scene-volume-dynamic-capsule-relative-swept-ccd" ||
			dynamicRotatingCapsuleSpeculativeCcdCase ||
			dynamicRotatingConvexSpeculativeCcdCase ||
			caseName ==
				"scene-volume-dynamic-convex-relative-swept-ccd";
		const bool sphereReverseFeatureCase =
			caseName == "scene-volume-sphere-reverse-feature";
		const bool capsuleReverseFeatureCase =
			caseName == "scene-volume-capsule-reverse-feature";
		const bool convexReverseFeatureCase =
			caseName == "scene-volume-convex-reverse-feature";
		const bool triangleMeshReverseFeatureCase =
			caseName ==
				"scene-volume-triangle-mesh-reverse-feature";
		const bool heightFieldReverseFeatureCase =
			caseName ==
				"scene-volume-heightfield-reverse-feature";
		const bool smoothReverseFeatureCase =
			sphereReverseFeatureCase ||
			capsuleReverseFeatureCase ||
			convexReverseFeatureCase ||
			triangleMeshReverseFeatureCase ||
			heightFieldReverseFeatureCase;
		const bool skinningCase =
			caseName == "scene-volume-skinning";
		const bool twoSoftVolumeCase =
			sphereLongRollCase || sphereSoftSoftGlancingCase ||
			softSoftTorqueCase || ogcSandwichCase ||
			multiSoftIslandCase || mixedSleepIslandCase ||
			softChurnCase || multiSceneIsolationCase ||
			softSoftWakeCase || softPairAttachmentCase ||
			motionControlsCase ||
			maxDepenetrationVelocityCase ||
			speculativeCcdCase || smoothReverseFeatureCase ||
			taskGraphWriteBackCase;
		const PxU32 expectedSceneVolumeCount =
			visualShowcaseCase ? 5u :
			taskGraphWriteBackFourWayCase ? 4u :
			(twoSoftVolumeCase ? 2u : 1u);
		const bool softSleepWakeCase =
			caseName == "scene-volume-sleep-wake";
		const bool softRigidWakeCase =
			caseName == "scene-volume-rigid-wake";
		const bool twoDynamicActorsCase =
			multiDynamicBoxCase || multiSoftIslandCase ||
			movingKinematicFiniteSpeculativeCcdCase ||
			dynamicFiniteRelativeSweptCcdCase ||
			kinematicSphereReverseSweptCcdCase ||
			dynamicSphereReverseSweptCcdCase ||
			triangleSurfaceSweptCcdCase;
		const bool dynamicBoxCase =
			caseName == "scene-volume-dynamic-box" ||
			caseName == "scene-volume-true-boundary-dynamic-box" ||
			caseName == "scene-volume-dynamic-sphere" ||
			caseName == "scene-volume-dynamic-capsule" ||
			caseName == "scene-volume-dynamic-convex" ||
			dynamicChurnCase || twoDynamicActorsCase;
		const bool dynamicSphereCase =
			caseName == "scene-volume-dynamic-sphere" ||
			caseName == "scene-volume-dynamic-capsule";
		const bool dynamicCapsuleCase =
			caseName == "scene-volume-dynamic-capsule";
		const bool dynamicConvexCase =
			caseName == "scene-volume-dynamic-convex";
		const bool dynamicSmoothCase =
			dynamicSphereCase || dynamicConvexCase;
		SceneCommonValidationConfig commonConfig;
		commonConfig.expectedFrames = context.frames;
		commonConfig.expectedSceneVolumeCount = expectedSceneVolumeCount;
		commonConfig.expectedSceneDynamicCount =
			twoDynamicActorsCase ? 2u :
			(visualShowcaseCase || ogcSandwichCase || dynamicBoxCase ||
			 softRigidWakeCase || kinematicBoxCase || rigidAttachmentCase ||
			 kinematicAttachmentCase ? 1u : 0u);
		commonConfig.centroidDropOptional =
			softSleepWakeCase || softRigidWakeCase ||
			mixedSleepIslandCase || softChurnCase || bufferMutationCase ||
			worldPinCase || attachmentCase || elementFilterCase ||
			kinematicBoxCase || multiSceneIsolationCase ||
			softSoftWakeCase || sphereSoftSoftGlancingCase ||
			softSoftTorqueCase || softPairAttachmentCase ||
			volumeKinematicTargetCase || motionControlsCase ||
			ogcSandwichCase || maxDepenetrationVelocityCase ||
			taskGraphPureSoftCase ||
			taskGraphRigidTriangleSurfaceFeatureOverlapCase ||
			taskGraphWriteBackCase ||
			speculativeCcdCase || smoothReverseFeatureCase;
		const bool commonPassed = isSceneCommonResultValid(
			gMetrics, commonConfig, context.fatalErrors);
		if(!commonPassed)
			return false;
		if(ogcSandwichCase)
		{
			OgcSandwichValidationConfig config;
			config.gateMinFrames = SCENE_CPU_OGC_SANDWICH_GATE_MIN_FRAMES;
			config.minNativeIslandSteps =
				SCENE_CPU_OGC_SANDWICH_MIN_NATIVE_ISLAND_STEPS;
			config.minJawCompression =
				SCENE_CPU_OGC_SANDWICH_MIN_JAW_COMPRESSION;
			config.maxLateralOffset =
				SCENE_CPU_OGC_SANDWICH_MAX_LATERAL_OFFSET;
			config.maxLateralSpeed =
				SCENE_CPU_OGC_SANDWICH_MAX_LATERAL_SPEED;
			config.maxNormalOffset =
				SCENE_CPU_OGC_SANDWICH_MAX_NORMAL_OFFSET;
			config.maxNormalSpeed =
				SCENE_CPU_OGC_SANDWICH_MAX_NORMAL_SPEED;
			config.minDetF = 0.05f;
			config.maxDetF = SCENE_CPU_VISUAL_MAX_DET_F;
			return isOgcSandwichResultValid(
				gMetrics, *context.ogcSandwichMetrics, config);
		}
		if(visualShowcaseCase)
		{
			VisualShowcaseValidationConfig config;
			config.rotationGateMinFrames =
				SCENE_CPU_VISUAL_ROTATION_GATE_MIN_FRAMES;
			config.interactionGateMinFrames =
				SCENE_CPU_VISUAL_INTERACTION_GATE_MIN_FRAMES;
			config.minSurfaceTriangles =
				SCENE_CPU_VISUAL_MIN_SURFACE_TRIANGLES;
			config.primaryMinOrientationChange =
				SCENE_CPU_VISUAL_MIN_ORIENTATION_CHANGE;
			config.primaryMinAngularSpeed =
				SCENE_CPU_VISUAL_MIN_ANGULAR_SPEED;
			config.sphereMinOrientationChange =
				SCENE_CPU_VISUAL_SPHERE_MIN_ORIENTATION_CHANGE;
			config.sphereMinAngularSpeed =
				SCENE_CPU_VISUAL_SPHERE_MIN_ANGULAR_SPEED;
			config.minDetF = SCENE_CPU_VISUAL_SHOWCASE_MIN_DET_F;
			config.maxDetF = SCENE_CPU_VISUAL_MAX_DET_F;
			config.minVolumeRatio =
				SCENE_CPU_VISUAL_SHOWCASE_MIN_VOLUME_RATIO;
			config.maxVolumeRatio = SCENE_CPU_VISUAL_MAX_VOLUME_RATIO;
			config.sphereLongRunBounded =
				context.visualSphereLongRunBounded;
			return isVisualShowcaseResultValid(
				gMetrics, gSceneCpuVisualPrimaryCubeRotationMetrics,
				gSceneCpuVisualSphereRotationMetrics,
				gSceneCpuVisualInteractionMetrics, config);
		}
		if(sphereLongRollCase)
		{
			SphereLongRollValidationConfig config;
			config.minDetF = SCENE_CPU_VISUAL_MIN_DET_F;
			config.maxDetF = SCENE_CPU_VISUAL_MAX_DET_F;
			config.minVolumeRatio = SCENE_CPU_VISUAL_MIN_VOLUME_RATIO;
			config.maxVolumeRatio = SCENE_CPU_VISUAL_MAX_VOLUME_RATIO;
			config.rollingKinematicsValid =
				context.sphereRollingKinematicsValid;
			config.longRunRegressionPassed =
				context.sphereLongRollRegressionPassed;
			return isSphereLongRollResultValid(
				gMetrics,
				gSceneCpuVolumeHealthMonitor.getBodyMetrics(1), config);
		}
		if(sphereSoftSoftGlancingCase)
		{
			SoftSoftGlancingValidationConfig config;
			config.gateMinFrames =
				SCENE_CPU_SPHERE_SOFT_SOFT_GLANCING_GATE_MIN_FRAMES;
			config.minDeltaSpeed =
				SCENE_CPU_SPHERE_SOFT_SOFT_GLANCING_MIN_DELTA_SPEED;
			config.minDetF = SCENE_CPU_VISUAL_MIN_DET_F;
			config.maxDetF = SCENE_CPU_VISUAL_MAX_DET_F;
			config.minVolumeRatio = SCENE_CPU_VISUAL_MIN_VOLUME_RATIO;
			config.maxVolumeRatio = SCENE_CPU_VISUAL_MAX_VOLUME_RATIO;
			return isSoftSoftGlancingResultValid(
				gMetrics, gSceneCpuSoftContactPhaseMetrics, config);
		}
		if(softSoftTorqueCase)
		{
			SoftSoftTorqueValidationConfig config;
			config.gateMinFrames =
				SCENE_CPU_SOFT_SOFT_TORQUE_GATE_MIN_FRAMES;
			config.minRetentionSamples =
				SCENE_CPU_SOFT_SOFT_TORQUE_MIN_RETENTION_SAMPLES;
			config.minLeverArm =
				SCENE_CPU_SOFT_SOFT_TORQUE_MIN_LEVER_ARM;
			config.minAngularMomentum =
				context.softSoftTorqueMinAngularMomentum;
			config.minAngularSpeed =
				context.softSoftTorqueMinAngularSpeed;
			config.minDetF = SCENE_CPU_VISUAL_MIN_DET_F;
			config.maxDetF = SCENE_CPU_VISUAL_MAX_DET_F;
			config.minVolumeRatio = SCENE_CPU_VISUAL_MIN_VOLUME_RATIO;
			config.maxVolumeRatio = SCENE_CPU_VISUAL_MAX_VOLUME_RATIO;
			return isSoftSoftTorqueResultValid(
				gMetrics, gSceneCpuSoftSoftTorqueMetrics, config);
		}
		if(taskGraphPureSoftCase)
			return validateSceneCpuTaskGraphResult(context,
				SnippetDeformableVolumeAVBDValidation::eTASK_GRAPH_PURE_SOFT);
		if(taskGraphWorldPlaneCase)
			return validateSceneCpuTaskGraphResult(context,
				SnippetDeformableVolumeAVBDValidation::eTASK_GRAPH_WORLD_PLANE);
		if(taskGraphRigidBoxSdfCase || taskGraphRigidSphereSdfCase ||
			taskGraphRigidCapsuleSdfCase || taskGraphRigidConvexSdfCase ||
			taskGraphRigidTriangleSurfaceCase ||
			taskGraphRigidTriangleSurfaceLargeCase ||
			taskGraphRigidTriangleSurfaceFeatureOverlapCase ||
			taskGraphRigidTriangleSurfaceThresholdCase)
			return validateSceneCpuTaskGraphResult(context,
				SnippetDeformableVolumeAVBDValidation::eTASK_GRAPH_RIGID_SDF);
		if(taskGraphWriteBackCase)
			return validateSceneCpuTaskGraphResult(context,
				taskGraphPipelineCase
					? SnippetDeformableVolumeAVBDValidation::
						eTASK_GRAPH_PIPELINE
					: SnippetDeformableVolumeAVBDValidation::
						eTASK_GRAPH_WRITE_BACK,
				taskGraphWriteBackFourWayCase ? 4u : 2u);
		if(skinningCase)
			return isVolumeSkinningResultValid(
				gMetrics, gVolumeSkinningMetrics, context.frames);
		if(motionControlsCase)
			return isMotionControlsResultValid(gMetrics, context.dt);
		if(maxDepenetrationVelocityCase)
			return isMaxDepenetrationVelocityResultValid(
				gMetrics, context.dt);
		if(smoothReverseFeatureCase)
			return isReverseFeatureSceneResultValid(
				gMetrics, gSphereReverseFeatureMetrics);
		if(rigidTriangleSteadyContactCase)
			return isRigidTriangleSteadyContactResultValid(
				gMetrics, gPerformance.profiledFrames,
				gPerformance.collision.rigidTriangleSurfaceFaceTests,
				gPerformance.collision.rigidTriangleSurfaceEdgeTests,
				gPerformance.collision.rigidTriangleSurfaceVertexTests);
		if(triangleSurfaceSweptCcdCase)
		{
			const TriangleSurfaceSweptValidationConfig sweptConfig =
				getTriangleSurfaceSweptValidationConfig(caseName);
			return isTriangleSurfaceSweptSceneResultValid(
				gMetrics, gSphereReverseSweptMetrics,
				gCapsuleRotationalSweepMetrics, sweptConfig);
		}
		if(sphereReverseSweptCcdCase)
		{
			const ReverseSweptValidationConfig sweptConfig =
				getReverseSweptValidationConfig(caseName);
			// This fixture makes the deformable volume 100x heavier than
			// the rigid body. Normalize its one-shot public verdict here,
			// outside the measured step loop: a valid dynamic impact may be
			// expressed mostly by arresting the rigid body.
			if(sweptConfig.dynamicTarget &&
				normalizeDynamicReverseSweptResponse(
					gSphereReverseSweptMetrics))
			{
				gMetrics.speculativeCcdPreventedTunneling = 1;
			}
			return isReverseSweptSceneResultValid(
				gMetrics, gSphereReverseSweptMetrics,
				gDeformingVolumeReverseSweptMetrics,
				gCapsuleRotationalSweepMetrics, sweptConfig,
				staticSphereReverseSweptCcdCase);
		}
		if(dynamicFiniteRelativeSweptCcdCase)
		{
			DynamicFiniteSweptMetrics sweptMetrics =
				context.dynamicFiniteSweptMetrics;
			if(normalizeDynamicFiniteSweptResponse(sweptMetrics))
			{
				gMetrics.dynamicSphereSweepResponseObserved =
					sweptMetrics.responseObserved;
				gMetrics.speculativeCcdPreventedTunneling = 1;
			}
			const bool rotational =
				dynamicRotatingCapsuleSpeculativeCcdCase ||
				dynamicRotatingConvexSpeculativeCcdCase;
			return isDynamicFiniteSweptSceneResultValid(
				gMetrics, sweptMetrics, gCapsuleRotationalSweepMetrics,
				rotational, dynamicRotatingConvexSpeculativeCcdCase);
		}
		if(movingKinematicFiniteSpeculativeCcdCase)
		{
			const KinematicFiniteSweptMetrics sweptMetrics =
				context.kinematicFiniteSweptMetrics;
			const bool rotational =
				rotatingKinematicCapsuleSpeculativeCcdCase ||
				rotatingKinematicConvexSpeculativeCcdCase;
			return isKinematicFiniteSweptSceneResultValid(
				gMetrics, sweptMetrics, gCapsuleRotationalSweepMetrics,
				rotational, rotatingKinematicConvexSpeculativeCcdCase);
		}
		if(speculativeCcdCase)
			return isSpeculativeCcdSceneResultValid(
				gMetrics, planeSpeculativeCcdCase,
				finiteSmoothSpeculativeCcdCase);
		if(volumeKinematicTargetCase)
			return isVolumeKinematicTargetResultValid(
				gMetrics, fullKinematicTargetCase);
		if(multiSceneIsolationCase)
			return isMultiSceneIsolationResultValid(gMetrics);
		if(softSoftWakeCase)
			return isSoftSoftWakeResultValid(gMetrics);
		if(softPairAttachmentCase)
			return isSoftPairAttachmentResultValid(gMetrics);
		if(kinematicBoxCase)
			return isKinematicRigidResultValid(gMetrics);
		if(dynamicBoxCase)
		{
			DynamicSceneValidationConfig dynamicConfig;
			dynamicConfig.initialY = gSceneCpuDynamicInitialY;
			dynamicConfig.smooth = dynamicSmoothCase;
			dynamicConfig.capsule = dynamicCapsuleCase;
			dynamicConfig.convex = dynamicConvexCase;
			dynamicConfig.twoActors = twoDynamicActorsCase;
			dynamicConfig.multiSoftIsland = multiSoftIslandCase;
			dynamicConfig.churn = dynamicChurnCase;
			return isDynamicSceneResultValid(gMetrics, dynamicConfig);
		}
		if(softSleepWakeCase)
			return isSoftSleepWakeResultValid(gMetrics);
		if(softRigidWakeCase)
			return isSoftRigidWakeResultValid(gMetrics);
		if(bufferMutationCase)
			return isBufferMutationResultValid(gMetrics);
		if(worldPinCase)
			return isWorldPinResultValid(gMetrics);
		if(attachmentCase)
		{
			AttachmentValidationConfig attachmentConfig;
			attachmentConfig.rigid = rigidAttachmentCase;
			attachmentConfig.staticTarget = staticAttachmentCase;
			attachmentConfig.kinematic = kinematicAttachmentCase;
			attachmentConfig.articulation = articulationAttachmentCase;
			return isAttachmentResultValid(gMetrics, attachmentConfig);
		}
		if(elementFilterCase)
		{
			ElementFilterValidationConfig filterConfig;
			filterConfig.partial = partialElementFilterCase;
			filterConfig.surfaceTolerance =
				SCENE_CPU_ELEMENT_FILTER_SURFACE_TOLERANCE;
			filterConfig.minSuppressedDepth =
				SCENE_CPU_ELEMENT_FILTER_MIN_SUPPRESSED_DEPTH;
			filterConfig.contactOffsetLimit =
				SCENE_CPU_ELEMENT_FILTER_CONTACT_OFFSET_LIMIT;
			return finalizeAndValidateElementFilterResult(
				gMetrics, filterConfig);
		}
		if(mixedSleepIslandCase)
			return isMixedSleepIslandResultValid(gMetrics);
		if(softChurnCase)
			return isSoftChurnResultValid(gMetrics);
		if(caseName == "scene-volume-lifecycle" ||
			caseName == "scene-volume-corotational")
			return isNoStaticSceneResultValid(gMetrics);
		if(caseName == "scene-volume-ground")
			return isGroundSceneResultValid(gMetrics);
		if(groundEmbeddedTetProbeCase)
			return isGroundEmbeddedTetResultValid(
				gMetrics, gSceneCpuGroundEmbeddedTetProbeMetrics,
				SCENE_CPU_GROUND_EMBEDDED_TET_PROBE_MIN_NORMALIZED_ROLL);
		return isStaticBoxSceneResultValid(
			gMetrics, caseName == "scene-volume-static-churn");
	}

	ComponentValidationConfig componentConfig;
	componentConfig.expectedFrames = context.frames;
	componentConfig.fatalErrorCount = context.fatalErrors;
	if(isComponentDenseNoContactCase(caseName))
		componentConfig.caseType =
			SnippetDeformableVolumeAVBDValidation::
				eCOMPONENT_DENSE_NO_CONTACT;
	else if(isComponentManySmallNoContactCase(caseName))
		componentConfig.caseType =
			SnippetDeformableVolumeAVBDValidation::
				eCOMPONENT_MANY_SMALL_NO_CONTACT;
	else if(caseName == "volume-ground")
		componentConfig.caseType =
			SnippetDeformableVolumeAVBDValidation::eCOMPONENT_GROUND;
	else if(caseName == "volume-static-box")
		componentConfig.caseType =
			SnippetDeformableVolumeAVBDValidation::eCOMPONENT_STATIC_BOX;
	else if(caseName == "soft-soft")
		componentConfig.caseType =
			SnippetDeformableVolumeAVBDValidation::eCOMPONENT_SOFT_SOFT;
	else if(caseName == "cone-ground")
		componentConfig.caseType =
			SnippetDeformableVolumeAVBDValidation::eCOMPONENT_CONE_GROUND;
	return isComponentResultValid(gMetrics, componentConfig);
}

void printHeadlessResult(
	const HeadlessResultContext& context, bool passed)
{
	const std::string& caseName = *context.caseName;
	const DeformableVolumeMetrics& gMetrics = *context.metrics;
	const DeformableVolumePerformanceMetrics& gPerformance =
		*context.performance;
	const RotationMetrics& gSceneCpuVisualRotationMetrics =
		*context.visualRotationMetrics;
	const RotationMetrics& gSceneCpuVisualPrimaryCubeRotationMetrics =
		*context.visualPrimaryCubeRotationMetrics;
	const RotationMetrics& gSceneCpuVisualSphereRotationMetrics =
		*context.visualSphereRotationMetrics;
	const VisualInteractionMetrics& gSceneCpuVisualInteractionMetrics =
		*context.visualInteractionMetrics;
	const SoftContactPhaseMetrics& gSceneCpuSoftContactPhaseMetrics =
		*context.softContactPhaseMetrics;
	const SoftSoftTorqueMetrics& gSceneCpuSoftSoftTorqueMetrics =
		*context.softSoftTorqueMetrics;
	const GroundEmbeddedTetProbeMetrics&
		gSceneCpuGroundEmbeddedTetProbeMetrics =
			*context.groundEmbeddedTetMetrics;
	const VolumeHealthMonitor& gSceneCpuVolumeHealthMonitor =
		*context.volumeHealthMonitor;
	const VolumeSkinningMetrics& gVolumeSkinningMetrics =
		*context.volumeSkinningMetrics;
	const ReverseFeatureMetrics& gSphereReverseFeatureMetrics =
		*context.reverseFeatureMetrics;
	const ReverseSweptMetrics& gSphereReverseSweptMetrics =
		*context.reverseSweptMetrics;
	const DeformingReverseSweptMetrics&
		gDeformingVolumeReverseSweptMetrics =
			*context.deformingReverseSweptMetrics;
	const RotationalSweepMetrics& gCapsuleRotationalSweepMetrics =
		*context.rotationalSweepMetrics;
	const PxReal gSceneCpuDynamicInitialY = context.dynamicInitialY;
	const PxReal gSceneCpuSecondDynamicInitialY =
		context.secondDynamicInitialY;
	if(caseName == "scene-volume-sphere-long-roll")
	{
		VolumeBodyHealthMetrics bodyMetrics[2];
		for(PxU32 bodyIndex = 0; bodyIndex < 2u; ++bodyIndex)
			bodyMetrics[bodyIndex] =
				gSceneCpuVolumeHealthMonitor.getBodyMetrics(bodyIndex);
		SphereLongRollReportConfig config;
		config.frames = context.frames;
		config.completedFrames = gMetrics.completedFrames;
		config.windowBeginFrame =
			context.sphereRollWindowBeginFrame;
		config.windowEndFrame = context.sphereRollWindowEndFrame;
		config.checkpointCount = context.sphereRollCheckpointCount;
		config.checkpointInterval =
			context.sphereRollCheckpointInterval;
		config.longRunBounded = context.visualSphereLongRunBounded;
		config.regressionBounded =
			context.sphereLongRollRegressionPassed;
		config.rollingKinematicsValid =
			context.sphereRollingKinematicsValid;
		config.passed = passed;
		printSphereLongRollReport(
			gSceneCpuVisualSphereRotationMetrics, bodyMetrics, 2u,
			gSceneCpuSoftContactPhaseMetrics, config);
	}
	if(caseName ==
		"scene-volume-sphere-soft-soft-glancing")
	{
		SoftSoftGlancingReportConfig config;
		config.frames = context.frames;
		config.passed = passed;
		printSoftSoftGlancingReport(
			gSceneCpuVisualSphereRotationMetrics,
			gSceneCpuSoftContactPhaseMetrics, config);
	}
	if(caseName == "scene-volume-soft-soft-torque")
	{
		SoftSoftTorqueReportConfig config;
		config.frames = context.frames;
		config.minRetentionSamples =
			SCENE_CPU_SOFT_SOFT_TORQUE_MIN_RETENTION_SAMPLES;
		config.passed = passed;
		printSoftSoftTorqueReport(gSceneCpuSoftSoftTorqueMetrics, config);
	}
	if(caseName ==
		"scene-volume-ground-embedded-tet-probe")
	{
		GroundEmbeddedTetReportConfig config;
		config.frames = context.frames;
		config.groundContactFrames = gMetrics.groundContactFrames;
		config.maxGroundContacts = gMetrics.maxGroundContacts;
		config.groundPositionAlRows =
			gPerformance.solverStages.groundTetPatchGroundPositionAlRows;
		config.fourSupportRows =
			gPerformance.solverStages.groundTetPatchFourSupportRows;
		config.singleTetRows =
			gPerformance.solverStages.groundTetPatchSingleTetRows;
		config.activeRows =
			gPerformance.solverStages.groundTetPatchActiveRows;
		config.velocityTangentOwnerRows =
			gPerformance.solverStages.worldStaticVelocityTangentOwnerRows;
		config.velocityTangentAppliedRows =
			gPerformance.solverStages.worldStaticVelocityTangentAppliedRows;
		config.nonFiniteParticleSamples = gMetrics.nonFiniteParticleSamples;
		config.invertedElementSamples = gMetrics.invertedElementSamples;
		config.minDetF = gMetrics.minDetF;
		config.maxDetF = gMetrics.maxDetF;
		config.minBodyVolumeRatio = gMetrics.minBodyVolumeRatio;
		config.maxBodyVolumeRatio = gMetrics.maxBodyVolumeRatio;
		config.requiredMinDetF = 0.05f;
		config.requiredMaxDetF = 20.0f;
		config.requiredMinVolumeRatio = 0.05f;
		config.requiredMaxVolumeRatio = 20.0f;
		config.passed = passed;
		printGroundEmbeddedTetReport(
			gSceneCpuGroundEmbeddedTetProbeMetrics, config);
	}
	if(caseName == "scene-volume-ogc-sandwich")
	{
		const OgcSandwichMetrics& metrics =
			*context.ogcSandwichMetrics;
		OgcSandwichReportConfig config;
		config.pressureDriveFrames =
			context.ogcPressureDriveFrames;
		config.minJawCompression =
			SCENE_CPU_OGC_SANDWICH_MIN_JAW_COMPRESSION;
		config.maxLateralOffset =
			SCENE_CPU_OGC_SANDWICH_MAX_LATERAL_OFFSET;
		config.maxLateralSpeed =
			SCENE_CPU_OGC_SANDWICH_MAX_LATERAL_SPEED;
		config.maxNormalOffset =
			SCENE_CPU_OGC_SANDWICH_MAX_NORMAL_OFFSET;
		config.maxNormalSpeed =
			SCENE_CPU_OGC_SANDWICH_MAX_NORMAL_SPEED;
		config.passed = passed;
		printOgcSandwichReport(metrics, config);
	}
	if(caseName == "scene-volume-visual-showcase")
	{
		VisualRotationReportConfig rotationConfig;
		rotationConfig.minPrimaryOrientationChange =
			SCENE_CPU_VISUAL_MIN_ORIENTATION_CHANGE;
		rotationConfig.minPrimaryAngularSpeed =
			SCENE_CPU_VISUAL_MIN_ANGULAR_SPEED;
		rotationConfig.minSphereOrientationChange =
			SCENE_CPU_VISUAL_SPHERE_MIN_ORIENTATION_CHANGE;
		rotationConfig.minSphereAngularSpeed =
			SCENE_CPU_VISUAL_SPHERE_MIN_ANGULAR_SPEED;
		rotationConfig.sphereLongRunBounded =
			context.visualSphereLongRunBounded;
		rotationConfig.sphereRollingKinematicsValid =
			context.sphereRollingKinematicsValid;
		printVisualRotationReport(
			gSceneCpuVisualPrimaryCubeRotationMetrics,
			gSceneCpuVisualRotationMetrics,
			gSceneCpuVisualSphereRotationMetrics, rotationConfig);
		VisualInteractionReportConfig interactionConfig;
		interactionConfig.completedFrames = gMetrics.completedFrames;
		interactionConfig.gateMinFrames =
			SCENE_CPU_VISUAL_INTERACTION_GATE_MIN_FRAMES;
		interactionConfig.dynamicInitiallySleeping =
			gMetrics.sceneDynamicInitiallySleeping;
		interactionConfig.dynamicActorAdded = gMetrics.sceneDynamicActorAdded;
		interactionConfig.dynamicActorRemoved =
			gMetrics.sceneDynamicActorRemoved;
		interactionConfig.dynamicActorReleased =
			gMetrics.sceneDynamicActorReleased;
		interactionConfig.rigidContactFrames = gMetrics.rigidContactFrames;
		interactionConfig.softContactFrames = gMetrics.softContactFrames;
		interactionConfig.dynamicMaxDrop = gMetrics.sceneDynamicMaxDrop;
		interactionConfig.dynamicMaxDownSpeed =
			gMetrics.sceneDynamicMaxDownSpeed;
		printVisualInteractionReport(
			gSceneCpuVisualInteractionMetrics, interactionConfig);
		static const char* bodyNames[5] =
			{"primaryCube", "sphere", "cone", "tiltedCube", "followerCube"};
		VolumeBodyHealthMetrics bodyHealthMetrics[5];
		for(PxU32 bodyIndex = 0; bodyIndex < 5; ++bodyIndex)
			bodyHealthMetrics[bodyIndex] =
				gSceneCpuVolumeHealthMonitor.getBodyMetrics(bodyIndex);
		VolumeHealthReportConfig healthConfig;
		healthConfig.nonFiniteParticleSamples =
			gMetrics.nonFiniteParticleSamples;
		healthConfig.invertedElementSamples =
			gMetrics.invertedElementSamples;
		healthConfig.minDetF = gMetrics.minDetF;
		healthConfig.maxDetF = gMetrics.maxDetF;
		healthConfig.minBodyVolumeRatio = gMetrics.minBodyVolumeRatio;
		healthConfig.maxBodyVolumeRatio = gMetrics.maxBodyVolumeRatio;
		healthConfig.requiredMinDetF =
			SCENE_CPU_VISUAL_SHOWCASE_MIN_DET_F;
		healthConfig.requiredMaxDetF = SCENE_CPU_VISUAL_MAX_DET_F;
		healthConfig.requiredMinVolumeRatio =
			SCENE_CPU_VISUAL_SHOWCASE_MIN_VOLUME_RATIO;
		healthConfig.requiredMaxVolumeRatio =
			SCENE_CPU_VISUAL_MAX_VOLUME_RATIO;
		printVolumeHealthReport(
			bodyHealthMetrics, bodyNames, 5, healthConfig);
	}
	if(caseName == "scene-volume-skinning")
		printVolumeSkinningReport(gVolumeSkinningMetrics, passed);
	if(isSceneCpuVolumeRigidTriangleSteadyContactCase(
			caseName))
	{
		RigidTriangleSteadyContactReportConfig config;
		config.frames = context.frames;
		config.profileFrames = gPerformance.profiledFrames;
		config.faceTests =
			gPerformance.collision.rigidTriangleSurfaceFaceTests;
		config.edgeTests =
			gPerformance.collision.rigidTriangleSurfaceEdgeTests;
		config.vertexTests =
			gPerformance.collision.rigidTriangleSurfaceVertexTests;
		config.passed = passed;
		printRigidTriangleSteadyContactReport(config);
	}
	if(isSceneCpuVolumeTriangleSurfaceSweptCcdCase(
			caseName))
	{
		const bool reverseCase =
			isSceneCpuVolumeTriangleSurfaceReverseSweptCcdCase(
				caseName);
		const bool heightField =
			isSceneCpuVolumeHeightFieldSweptCcdCase(
				caseName);
		const bool rotationalCase =
			isSceneCpuVolumeRotationalTriangleSurfaceSweptCcdCase(
				caseName);
		printTriangleSurfaceSweptReport(
			reverseCase
				? "AVBD_TRIANGLE_SURFACE_REVERSE_SWEPT"
				: "AVBD_TRIANGLE_SURFACE_FORWARD_SWEPT",
			context.frames,
			"kinematic",
			heightField ? "heightfield" : "triangle-mesh",
			gSphereReverseSweptMetrics, passed);
		if(rotationalCase)
		{
			printTriangleSurfaceRotationalSweptReport(
				context.frames,
				heightField ? "heightfield" : "triangle-mesh",
				reverseCase ? "reverse" : "forward",
				gSphereReverseSweptMetrics,
				gCapsuleRotationalSweepMetrics, passed);
		}
	}
	if(isSceneCpuVolumeSphereReverseSweptCcdCase(
			caseName))
	{
		const bool deformingSoftTarget =
			isSceneCpuVolumeDeformingReverseSweptCcdCase(
				caseName);
		const bool capsuleTarget =
			isSceneCpuVolumeCapsuleReverseSweptCcdCase(
				caseName);
		const bool convexTarget =
			isSceneCpuVolumeConvexReverseSweptCcdCase(
				caseName);
		const char* target =
			deformingSoftTarget ||
			caseName ==
					"scene-volume-static-sphere-reverse-swept-ccd" ||
				caseName ==
					"scene-volume-static-capsule-reverse-swept-ccd" ||
				caseName ==
					"scene-volume-static-convex-reverse-swept-ccd"
				? "static"
				: (caseName ==
						"scene-volume-kinematic-sphere-reverse-swept-ccd" ||
				   caseName ==
						"scene-volume-kinematic-capsule-reverse-swept-ccd" ||
				   caseName ==
						"scene-volume-rotating-kinematic-capsule-reverse-swept-ccd" ||
				   caseName ==
						"scene-volume-rotating-kinematic-convex-reverse-swept-ccd" ||
				   caseName ==
						"scene-volume-kinematic-convex-reverse-swept-ccd")
					? "kinematic"
					: "dynamic";
		printReverseSweptReport(
			convexTarget
				? "AVBD_CONVEX_REVERSE_SWEPT"
				: capsuleTarget
				? "AVBD_CAPSULE_REVERSE_SWEPT"
				: "AVBD_SPHERE_REVERSE_SWEPT",
			context.frames, target,
			gSphereReverseSweptMetrics, passed);
		if(deformingSoftTarget)
		{
			printDeformingReverseSweptReport(
				context.frames,
				convexTarget ? "convex" :
					(capsuleTarget ? "capsule" : "sphere"),
				gSphereReverseSweptMetrics,
				gDeformingVolumeReverseSweptMetrics, passed);
		}
		const bool rotationalCapsuleTarget =
			isSceneCpuVolumeRotationalCapsuleReverseSweptCcdCase(
				caseName);
		const bool rotationalConvexTarget =
			isSceneCpuVolumeRotationalConvexReverseSweptCcdCase(
				caseName);
		if(rotationalCapsuleTarget || rotationalConvexTarget)
		{
			printRotationalReverseSweptReport(
				rotationalConvexTarget
					? "AVBD_CONVEX_ROTATIONAL_REVERSE_SWEPT"
					: "AVBD_CAPSULE_ROTATIONAL_REVERSE_SWEPT",
				context.frames, target,
				gSphereReverseSweptMetrics,
				gCapsuleRotationalSweepMetrics, passed);
		}
	}
	if(caseName ==
			"scene-volume-rotating-kinematic-capsule-speculative-ccd" ||
		caseName ==
			"scene-volume-rotating-kinematic-convex-speculative-ccd")
	{
		const KinematicFiniteSweptMetrics sweptMetrics =
			context.kinematicFiniteSweptMetrics;
		printKinematicRotationalSweptReport(
			caseName ==
				"scene-volume-rotating-kinematic-convex-speculative-ccd"
				? "AVBD_CONVEX_ROTATIONAL_SWEPT"
				: "AVBD_CAPSULE_ROTATIONAL_SWEPT",
			context.frames, sweptMetrics,
			gCapsuleRotationalSweepMetrics, passed);
	}
	if(caseName ==
			"scene-volume-dynamic-rotating-capsule-relative-swept-ccd" ||
		caseName ==
			"scene-volume-dynamic-rotating-convex-relative-swept-ccd")
	{
		const DynamicFiniteSweptMetrics sweptMetrics =
			context.dynamicFiniteSweptMetrics;
		printDynamicRotationalSweptReport(
			caseName ==
				"scene-volume-dynamic-rotating-convex-relative-swept-ccd"
				? "AVBD_CONVEX_DYNAMIC_ROTATIONAL_SWEPT"
				: "AVBD_CAPSULE_DYNAMIC_ROTATIONAL_SWEPT",
			context.frames, sweptMetrics,
			gCapsuleRotationalSweepMetrics, passed);
	}
	const bool capsuleReverseFeatureOutput =
		caseName ==
			"scene-volume-capsule-reverse-feature";
	const bool convexReverseFeatureOutput =
		caseName ==
			"scene-volume-convex-reverse-feature";
	const bool triangleMeshReverseFeatureOutput =
		caseName ==
			"scene-volume-triangle-mesh-reverse-feature";
	const bool heightFieldReverseFeatureOutput =
		caseName ==
			"scene-volume-heightfield-reverse-feature";
	if(caseName ==
			"scene-volume-sphere-reverse-feature" ||
		capsuleReverseFeatureOutput ||
		convexReverseFeatureOutput ||
		triangleMeshReverseFeatureOutput ||
		heightFieldReverseFeatureOutput)
	{
		const char* reverseFeatureTag =
			triangleMeshReverseFeatureOutput
				? "AVBD_TRIANGLE_MESH_REVERSE_FEATURE"
				: heightFieldReverseFeatureOutput
				? "AVBD_HEIGHTFIELD_REVERSE_FEATURE"
				: convexReverseFeatureOutput
				? "AVBD_CONVEX_REVERSE_FEATURE"
				: capsuleReverseFeatureOutput
				? "AVBD_CAPSULE_REVERSE_FEATURE"
				: "AVBD_SPHERE_REVERSE_FEATURE";
		printReverseFeatureReport(
			reverseFeatureTag, context.frames,
			gSphereReverseFeatureMetrics, passed);
	}
	const HeadlessGateClassification classification =
		classifyHeadlessGate(caseName);
	HeadlessGateReportConfig config;
	config.caseName = caseName.c_str();
	config.solverName =
		context.solverName;
	config.validation = classification.validation;
	config.sceneIntegrated = classification.sceneIntegrated;
	config.passed = passed;
	config.dynamicInitialY = gSceneCpuDynamicInitialY;
	config.secondDynamicInitialY = gSceneCpuSecondDynamicInitialY;
	config.fatalErrors = context.fatalErrors;
	config.warningErrors = context.warningErrors;
	printHeadlessGateReport(gMetrics, config);
}

} // namespace SnippetDeformableVolumeAVBDHeadless
