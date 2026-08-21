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

#ifndef SNIPPET_DEFORMABLE_VOLUME_AVBD_REPORT_H
#define SNIPPET_DEFORMABLE_VOLUME_AVBD_REPORT_H

#include "SnippetDeformableVolumeAVBDValidation.h"
#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"

namespace SnippetDeformableVolumeAVBDReport
{

struct DeformableVolumePerformanceMetrics
{
	physx::PxU32 warmupFrames;
	physx::PxU32 profiledFrames;
	physx::PxU32 softWorkers;
	physx::PxArray<physx::PxReal> stepSamplesMs;
	physx::PxF64 initialContactMs;
	physx::PxF64 solverMs;
	physx::PxF64 sceneMs;
	physx::PxF64 metricsMs;
	physx::PxReal avgStepMs;
	physx::PxReal p50StepMs;
	physx::PxReal p95StepMs;
	physx::PxReal maxStepMs;
	physx::PxU64 topologySoftBodies;
	physx::PxU64 topologySoftParticles;
	physx::PxU64 topologyTriElements;
	physx::PxU64 topologyTetElements;
	physx::PxU64 topologyBendElements;
	physx::PxU64 topologySurfaceTriangles;
	physx::PxU64 topologySurfaceVertices;
	physx::PxU64 topologySurfaceEdges;
	physx::PxU64 topologyRigidBoxes;
	// Scene CPU AVBD has two mutually exclusive solve authorities per step.
	// Keep both in schema-2 so task-graph validation can prove that an
	// experimental component continuation never consumes a native rigid/soft
	// island merely because the scene also contains dynamic rigid actors.
	physx::PxU64 componentFallbackSteps;
	physx::PxU64 nativeIslandSteps;
	// These two counters retain step granularity which the aggregate authority
	// counters above intentionally do not have.  They make it possible for an
	// opt-in task-graph audit to prove that no causal-layer task was submitted
	// during a native-island step, even when one profile window contains both
	// ownership modes after an attachment is released.
	physx::PxU64 nativeIslandComponentFallbackOverlapFrames;
	physx::PxU64 nativeIslandCausalLayerTaskOverlapFrames;
	// Raw topology for one shared rigid triangle mesh asset. This deliberately
	// does not multiply by shape instances: it is the immutable hierarchy size
	// used by the rigid-triangle query/cache contract.
	physx::PxU64 topologyRigidTriangleMeshTriangles;
	// P6 Scene-published CPU ISA dispatch snapshot.  The Snippet never probes
	// CPUID itself: these values are the actual low-level context selection.
	physx::PxU32 cpuIsaRequested;
	physx::PxU32 cpuIsaSelected;
	physx::PxU32 cpuIsaCompiledBackendMask;
	physx::PxU32 cpuIsaCapabilityMask;
	physx::PxU32 cpuIsaForceModeRejected;
	physx::PxU32 cpuIsaKernelSelfTestPassed;
	physx::PxU32 cpuIsaFmaUsed;
	physx::PxReal cpuIsaKernelSelfTestValue;
	physx::PxU32 taskGraphRequestedDispatcherWorkers;
	physx::PxU32 taskGraphPeakActiveSolveTasks;
	physx::PxU64 taskGraphSubmittedSolveTasks;
	physx::PxU64 taskGraphCompletedSolveTasks;
	physx::PxU64 taskGraphBarrierTasks;
	physx::PxU64 taskGraphSerialSolveTasks;
	physx::PxU64 taskGraphSubmittedPredictionTasks;
	physx::PxU64 taskGraphCompletedPredictionTasks;
	physx::PxU32 taskGraphPeakActivePredictionTasks;
	physx::PxU64 taskGraphSerialPredictionStages;
	physx::PxU64 taskGraphSubmittedWriteBackTasks;
	physx::PxU64 taskGraphCompletedWriteBackTasks;
	physx::PxU32 taskGraphPeakActiveWriteBackTasks;
	physx::PxU64 taskGraphSerialWriteBackStages;
	physx::PxU64 taskGraphSubmittedCausalLayerTasks;
	physx::PxU64 taskGraphCompletedCausalLayerTasks;
	physx::PxU32 taskGraphPeakActiveCausalLayerTasks;
	physx::PxU64 taskGraphCausalLayerFanIns;
	physx::PxU64 taskGraphSerialCausalLayerFallbacks;
	physx::PxU32 taskGraphMaxCausalLayerOccupancy;
	physx::PxU64 taskGraphTotalCausalLayerOccupancy;
	physx::PxU64 taskGraphSubmittedWorldPlaneContactTasks;
	physx::PxU64 taskGraphCompletedWorldPlaneContactTasks;
	physx::PxU32 taskGraphPeakActiveWorldPlaneContactTasks;
	physx::PxU64 taskGraphWorldPlaneContactFanIns;
	physx::PxU64 taskGraphSerialWorldPlaneContactFallbacks;
	physx::PxU64 taskGraphSubmittedRigidBoxSdfContactTasks;
	physx::PxU64 taskGraphCompletedRigidBoxSdfContactTasks;
	physx::PxU32 taskGraphPeakActiveRigidBoxSdfContactTasks;
	physx::PxU64 taskGraphRigidBoxSdfContactFanIns;
	physx::PxU64 taskGraphSerialRigidBoxSdfContactFallbacks;
	physx::PxU64 taskGraphSubmittedRigidSphereSdfContactTasks;
	physx::PxU64 taskGraphCompletedRigidSphereSdfContactTasks;
	physx::PxU32 taskGraphPeakActiveRigidSphereSdfContactTasks;
	physx::PxU64 taskGraphRigidSphereSdfContactFanIns;
	physx::PxU64 taskGraphSerialRigidSphereSdfContactFallbacks;
	physx::PxU64 taskGraphSubmittedRigidCapsuleSdfContactTasks;
	physx::PxU64 taskGraphCompletedRigidCapsuleSdfContactTasks;
	physx::PxU32 taskGraphPeakActiveRigidCapsuleSdfContactTasks;
	physx::PxU64 taskGraphRigidCapsuleSdfContactFanIns;
	physx::PxU64 taskGraphSerialRigidCapsuleSdfContactFallbacks;
	physx::PxU64 taskGraphSubmittedRigidConvexSdfContactTasks;
	physx::PxU64 taskGraphCompletedRigidConvexSdfContactTasks;
	physx::PxU32 taskGraphPeakActiveRigidConvexSdfContactTasks;
	physx::PxU64 taskGraphRigidConvexSdfContactFanIns;
	physx::PxU64 taskGraphSerialRigidConvexSdfContactFallbacks;
	physx::PxU64 taskGraphSubmittedRigidTriangleSurfaceContactTasks;
	physx::PxU64 taskGraphCompletedRigidTriangleSurfaceContactTasks;
	physx::PxU32 taskGraphPeakActiveRigidTriangleSurfaceContactTasks;
	physx::PxU64 taskGraphRigidTriangleSurfaceContactFanIns;
	physx::PxU64 taskGraphSerialRigidTriangleSurfaceContactFallbacks;
	physx::PxU64 taskGraphPureSoftEligibleIslands;
	physx::PxU64 taskGraphPureSoftEligibleParticles;
	// Scene statistics expose this rigid-query count only in aggregate.  Keep
	// it separate from the component-path shape-specific counters below so a
	// Scene AVBD measurement never mislabels triangle work as box work.
	physx::PxU64 collisionRigidParticleTests;
	physx::Dy::AvbdSoftBodyStepStats solverStages;
	physx::Dy::AvbdSoftCollisionStats collision;

	DeformableVolumePerformanceMetrics()
		: warmupFrames(0), profiledFrames(0), softWorkers(1),
		  initialContactMs(0.0),
		  solverMs(0.0), sceneMs(0.0), metricsMs(0.0),
		  avgStepMs(0.0f), p50StepMs(0.0f), p95StepMs(0.0f),
		  maxStepMs(0.0f), topologySoftBodies(0),
		  topologySoftParticles(0), topologyTriElements(0),
	  topologyTetElements(0), topologyBendElements(0),
	  topologySurfaceTriangles(0), topologySurfaceVertices(0),
	  topologySurfaceEdges(0), topologyRigidBoxes(0),
	  componentFallbackSteps(0), nativeIslandSteps(0),
	  nativeIslandComponentFallbackOverlapFrames(0),
	  nativeIslandCausalLayerTaskOverlapFrames(0),
	  topologyRigidTriangleMeshTriangles(0),
	  cpuIsaRequested(0), cpuIsaSelected(0),
	  cpuIsaCompiledBackendMask(0), cpuIsaCapabilityMask(0),
	  cpuIsaForceModeRejected(0), cpuIsaKernelSelfTestPassed(0),
	  cpuIsaFmaUsed(0), cpuIsaKernelSelfTestValue(0.0f),
		  taskGraphRequestedDispatcherWorkers(0),
		  taskGraphPeakActiveSolveTasks(0),
		  taskGraphSubmittedSolveTasks(0),
		  taskGraphCompletedSolveTasks(0), taskGraphBarrierTasks(0),
		  taskGraphSerialSolveTasks(0),
		  taskGraphSubmittedPredictionTasks(0),
		  taskGraphCompletedPredictionTasks(0),
		  taskGraphPeakActivePredictionTasks(0),
		  taskGraphSerialPredictionStages(0),
		  taskGraphSubmittedWriteBackTasks(0),
		  taskGraphCompletedWriteBackTasks(0),
		  taskGraphPeakActiveWriteBackTasks(0),
		  taskGraphSerialWriteBackStages(0),
		  taskGraphSubmittedCausalLayerTasks(0),
		  taskGraphCompletedCausalLayerTasks(0),
		  taskGraphPeakActiveCausalLayerTasks(0),
		  taskGraphCausalLayerFanIns(0),
		  taskGraphSerialCausalLayerFallbacks(0),
		  taskGraphMaxCausalLayerOccupancy(0),
		  taskGraphTotalCausalLayerOccupancy(0),
		  taskGraphSubmittedWorldPlaneContactTasks(0),
		  taskGraphCompletedWorldPlaneContactTasks(0),
		  taskGraphPeakActiveWorldPlaneContactTasks(0),
		  taskGraphWorldPlaneContactFanIns(0),
		  taskGraphSerialWorldPlaneContactFallbacks(0),
		  taskGraphSubmittedRigidBoxSdfContactTasks(0),
		  taskGraphCompletedRigidBoxSdfContactTasks(0),
		  taskGraphPeakActiveRigidBoxSdfContactTasks(0),
		  taskGraphRigidBoxSdfContactFanIns(0),
		  taskGraphSerialRigidBoxSdfContactFallbacks(0),
		  taskGraphSubmittedRigidSphereSdfContactTasks(0),
		  taskGraphCompletedRigidSphereSdfContactTasks(0),
		  taskGraphPeakActiveRigidSphereSdfContactTasks(0),
		  taskGraphRigidSphereSdfContactFanIns(0),
		  taskGraphSerialRigidSphereSdfContactFallbacks(0),
		  taskGraphSubmittedRigidCapsuleSdfContactTasks(0),
		  taskGraphCompletedRigidCapsuleSdfContactTasks(0),
		  taskGraphPeakActiveRigidCapsuleSdfContactTasks(0),
		  taskGraphRigidCapsuleSdfContactFanIns(0),
		  taskGraphSerialRigidCapsuleSdfContactFallbacks(0),
		  taskGraphSubmittedRigidConvexSdfContactTasks(0),
		  taskGraphCompletedRigidConvexSdfContactTasks(0),
		  taskGraphPeakActiveRigidConvexSdfContactTasks(0),
		  taskGraphRigidConvexSdfContactFanIns(0),
		  taskGraphSerialRigidConvexSdfContactFallbacks(0),
		  taskGraphSubmittedRigidTriangleSurfaceContactTasks(0),
		  taskGraphCompletedRigidTriangleSurfaceContactTasks(0),
		  taskGraphPeakActiveRigidTriangleSurfaceContactTasks(0),
		  taskGraphRigidTriangleSurfaceContactFanIns(0),
		  taskGraphSerialRigidTriangleSurfaceContactFallbacks(0),
		  taskGraphPureSoftEligibleIslands(0),
		  taskGraphPureSoftEligibleParticles(0),
		  collisionRigidParticleTests(0)
	{
	}
};

struct PerformanceReportConfig
{
	const char* caseName;
	const char* executionName;
	physx::PxU32 dispatcherThreads;
};

void finalizePerformanceMetrics(DeformableVolumePerformanceMetrics& metrics);

void printPerformanceResult(
	const DeformableVolumePerformanceMetrics& metrics,
	const PerformanceReportConfig& config);

struct HeadlessGateReportConfig
{
	const char* caseName;
	const char* solverName;
	const char* validation;
	bool sceneIntegrated;
	bool passed;
	physx::PxReal dynamicInitialY;
	physx::PxReal secondDynamicInitialY;
	physx::PxU32 fatalErrors;
	physx::PxU32 warningErrors;
};

void printHeadlessGateReport(
	const SnippetDeformableVolumeAVBDValidation::DeformableVolumeMetrics&
		metrics,
	const HeadlessGateReportConfig& config);

struct OgcSandwichReportConfig
{
	physx::PxU32 pressureDriveFrames;
	physx::PxReal minJawCompression;
	physx::PxReal maxLateralOffset;
	physx::PxReal maxLateralSpeed;
	physx::PxReal maxNormalOffset;
	physx::PxReal maxNormalSpeed;
	bool passed;
};

void printOgcSandwichReport(
	const SnippetDeformableVolumeAVBDValidation::OgcSandwichMetrics& metrics,
	const OgcSandwichReportConfig& config);

struct VisualInteractionReportConfig
{
	physx::PxU32 completedFrames;
	physx::PxU32 gateMinFrames;
	physx::PxU32 dynamicInitiallySleeping;
	physx::PxU32 dynamicActorAdded;
	physx::PxU32 dynamicActorRemoved;
	physx::PxU32 dynamicActorReleased;
	physx::PxU32 rigidContactFrames;
	physx::PxU32 softContactFrames;
	physx::PxReal dynamicMaxDrop;
	physx::PxReal dynamicMaxDownSpeed;
};

void printVisualInteractionReport(
	const SnippetDeformableVolumeAVBDValidation::VisualInteractionMetrics&
		metrics,
	const VisualInteractionReportConfig& config);

struct VisualRotationReportConfig
{
	physx::PxReal minPrimaryOrientationChange;
	physx::PxReal minPrimaryAngularSpeed;
	physx::PxReal minSphereOrientationChange;
	physx::PxReal minSphereAngularSpeed;
	bool sphereLongRunBounded;
	bool sphereRollingKinematicsValid;
};

void printVisualRotationReport(
	const SnippetDeformableVolumeAVBDValidation::RotationMetrics&
		primaryCube,
	const SnippetDeformableVolumeAVBDValidation::RotationMetrics& upperJaw,
	const SnippetDeformableVolumeAVBDValidation::RotationMetrics& sphere,
	const VisualRotationReportConfig& config);

struct VolumeHealthReportConfig
{
	physx::PxU32 nonFiniteParticleSamples;
	physx::PxU32 invertedElementSamples;
	physx::PxReal minDetF;
	physx::PxReal maxDetF;
	physx::PxReal minBodyVolumeRatio;
	physx::PxReal maxBodyVolumeRatio;
	physx::PxReal requiredMinDetF;
	physx::PxReal requiredMaxDetF;
	physx::PxReal requiredMinVolumeRatio;
	physx::PxReal requiredMaxVolumeRatio;
};

void printVolumeHealthReport(
	const SnippetDeformableVolumeAVBDValidation::VolumeBodyHealthMetrics*
		bodyMetrics,
	const char* const* bodyNames, physx::PxU32 bodyCount,
	const VolumeHealthReportConfig& config);

struct SphereLongRollReportConfig
{
	physx::PxU32 frames;
	physx::PxU32 completedFrames;
	physx::PxU32 windowBeginFrame;
	physx::PxU32 windowEndFrame;
	physx::PxU32 checkpointCount;
	physx::PxU32 checkpointInterval;
	bool longRunBounded;
	bool regressionBounded;
	bool rollingKinematicsValid;
	bool passed;
};

void printSphereLongRollReport(
	const SnippetDeformableVolumeAVBDValidation::RotationMetrics& metrics,
	const SnippetDeformableVolumeAVBDValidation::VolumeBodyHealthMetrics*
		bodyMetrics,
	physx::PxU32 bodyCount,
	const SnippetDeformableVolumeAVBDValidation::SoftContactPhaseMetrics&
		phaseMetrics,
	const SphereLongRollReportConfig& config);

struct SoftSoftGlancingReportConfig
{
	physx::PxU32 frames;
	bool passed;
};

void printSoftSoftGlancingReport(
	const SnippetDeformableVolumeAVBDValidation::RotationMetrics& metrics,
	const SnippetDeformableVolumeAVBDValidation::SoftContactPhaseMetrics&
		phaseMetrics,
	const SoftSoftGlancingReportConfig& config);

struct SoftSoftTorqueReportConfig
{
	physx::PxU32 frames;
	physx::PxU32 minRetentionSamples;
	bool passed;
};

void printSoftSoftTorqueReport(
	const SnippetDeformableVolumeAVBDValidation::SoftSoftTorqueMetrics& metrics,
	const SoftSoftTorqueReportConfig& config);

struct GroundEmbeddedTetReportConfig
{
	physx::PxU32 frames;
	physx::PxU32 groundContactFrames;
	physx::PxU32 maxGroundContacts;
	physx::PxU64 groundPositionAlRows;
	physx::PxU64 fourSupportRows;
	physx::PxU64 singleTetRows;
	physx::PxU64 activeRows;
	physx::PxU64 velocityTangentOwnerRows;
	physx::PxU64 velocityTangentAppliedRows;
	physx::PxU32 nonFiniteParticleSamples;
	physx::PxU32 invertedElementSamples;
	physx::PxReal minDetF;
	physx::PxReal maxDetF;
	physx::PxReal minBodyVolumeRatio;
	physx::PxReal maxBodyVolumeRatio;
	physx::PxReal requiredMinDetF;
	physx::PxReal requiredMaxDetF;
	physx::PxReal requiredMinVolumeRatio;
	physx::PxReal requiredMaxVolumeRatio;
	bool passed;
};

void printGroundEmbeddedTetReport(
	const SnippetDeformableVolumeAVBDValidation::GroundEmbeddedTetProbeMetrics&
		metrics,
	const GroundEmbeddedTetReportConfig& config);

void printVolumeSkinningReport(
	const SnippetDeformableVolumeAVBDValidation::VolumeSkinningMetrics& metrics,
	bool passed);

struct RigidTriangleSteadyContactReportConfig
{
	physx::PxU32 frames;
	physx::PxU32 profileFrames;
	physx::PxU64 faceTests;
	physx::PxU64 edgeTests;
	physx::PxU64 vertexTests;
	bool passed;
};

void printRigidTriangleSteadyContactReport(
	const RigidTriangleSteadyContactReportConfig& config);

void printReverseFeatureReport(
	const char* marker, physx::PxU32 frames,
	const SnippetDeformableVolumeAVBDValidation::ReverseFeatureMetrics&
		metrics,
	bool passed);

void printTriangleSurfaceSweptReport(
	const char* marker, physx::PxU32 frames, const char* target,
	const char* geometry,
	const SnippetDeformableVolumeAVBDValidation::ReverseSweptMetrics&
		metrics,
	bool passed);

void printTriangleSurfaceRotationalSweptReport(
	physx::PxU32 frames, const char* geometry, const char* owner,
	const SnippetDeformableVolumeAVBDValidation::ReverseSweptMetrics&
		metrics,
	const SnippetDeformableVolumeAVBDValidation::RotationalSweepMetrics&
		rotational,
	bool passed);

void printReverseSweptReport(
	const char* marker, physx::PxU32 frames, const char* target,
	const SnippetDeformableVolumeAVBDValidation::ReverseSweptMetrics&
		metrics,
	bool passed);

void printDeformingReverseSweptReport(
	physx::PxU32 frames, const char* geometry,
	const SnippetDeformableVolumeAVBDValidation::ReverseSweptMetrics&
		metrics,
	const SnippetDeformableVolumeAVBDValidation::DeformingReverseSweptMetrics&
		deforming,
	bool passed);

void printRotationalReverseSweptReport(
	const char* marker, physx::PxU32 frames, const char* target,
	const SnippetDeformableVolumeAVBDValidation::ReverseSweptMetrics&
		metrics,
	const SnippetDeformableVolumeAVBDValidation::RotationalSweepMetrics&
		rotational,
	bool passed);

void printKinematicRotationalSweptReport(
	const char* marker, physx::PxU32 frames,
	const SnippetDeformableVolumeAVBDValidation::KinematicFiniteSweptMetrics&
		metrics,
	const SnippetDeformableVolumeAVBDValidation::RotationalSweepMetrics&
		rotational,
	bool passed);

void printDynamicRotationalSweptReport(
	const char* marker, physx::PxU32 frames,
	const SnippetDeformableVolumeAVBDValidation::DynamicFiniteSweptMetrics&
		metrics,
	const SnippetDeformableVolumeAVBDValidation::RotationalSweepMetrics&
		rotational,
	bool passed);

} // namespace SnippetDeformableVolumeAVBDReport

#endif
