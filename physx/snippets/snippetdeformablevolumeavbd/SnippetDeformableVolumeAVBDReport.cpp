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

#include "SnippetDeformableVolumeAVBDReport.h"

#include <cstdio>
#include "PxAvbdCpuIsa.h"
#include "foundation/PxSort.h"
#include "foundation/PxThread.h"

using namespace physx;
using namespace physx::Dy;
using namespace SnippetDeformableVolumeAVBDValidation;

namespace SnippetDeformableVolumeAVBDReport
{

void finalizePerformanceMetrics(DeformableVolumePerformanceMetrics& metrics)
{
	if(metrics.stepSamplesMs.empty())
		return;
	PxF64 sumStepMs = 0.0;
	for(PxU32 i = 0; i < metrics.stepSamplesMs.size(); ++i)
		sumStepMs += metrics.stepSamplesMs[i];
	PxSort(
		metrics.stepSamplesMs.begin(),
		metrics.stepSamplesMs.size());
	metrics.avgStepMs = PxReal(
		sumStepMs / PxF64(metrics.stepSamplesMs.size()));
	const PxU32 last = metrics.stepSamplesMs.size() - 1;
	metrics.p50StepMs =
		metrics.stepSamplesMs[PxU32(PxCeil(0.5f * PxReal(last)))];
	metrics.p95StepMs =
		metrics.stepSamplesMs[PxU32(PxCeil(0.95f * PxReal(last)))];
	metrics.maxStepMs = metrics.stepSamplesMs[last];
}

static const char* getAvbdCpuIsaModeName(PxU32 mode)
{
	return mode == 0u ? "auto" :
		(mode == 1u ? "sse2" : (mode == 2u ? "avx2fma" : "invalid"));
}

static const char* getAvbdCpuIsaCompiledBackendNames(PxU32 mask)
{
	return (mask & 2u) ? "sse2,avx2fma" : "sse2";
}

void printPerformanceResult(
	const DeformableVolumePerformanceMetrics& metrics,
	const PerformanceReportConfig& config)
{
#if PX_DEBUG
	const char* buildProfile = "debug";
#elif PX_CHECKED
	const char* buildProfile = "checked";
#elif PX_PROFILE
	const char* buildProfile = "profile";
#else
	const char* buildProfile = "release";
#endif
	const PxU32 physicalCores = PxThread::getNbPhysicalCores();
	const bool taskGraphObserved =
		metrics.taskGraphSubmittedSolveTasks > 0 ||
		metrics.taskGraphSerialSolveTasks > 0;
	const PxU32 peakActiveSoftWorkers = PxMax(
		metrics.taskGraphPeakActiveSolveTasks,
		metrics.taskGraphPeakActiveCausalLayerTasks);
	const bool softParallel = peakActiveSoftWorkers > 1;
	const char* softExecution = softParallel ? "parallel" : "serial";
	const PxU32 softWorkers = softParallel
		? peakActiveSoftWorkers : 1u;
	const PxU32 actualSoftWorkers = taskGraphObserved
		? PxMax(peakActiveSoftWorkers, 1u) : 1u;
	const char* softScheduler = taskGraphObserved
		? "sceneTaskgraph" : "componentSerial";
	const PxF64 divisor = metrics.profiledFrames ?
		PxF64(metrics.profiledFrames) : 1.0;
	const AvbdSoftBodyStepStats& stages = metrics.solverStages;
	const PxF64 solverStageMs =
		stages.predictionMs + stages.contactIndexMs +
		stages.bodyPrecomputeMs + stages.bodySolveMs +
		stages.particleSolveMs + stages.projectionMs + stages.dualMs +
		stages.redetectMs + stages.velocityMs + stages.frictionMs;
	const PxF64 closureMs =
		metrics.initialContactMs + metrics.solverMs +
		metrics.sceneMs + metrics.metricsMs;
	// A component-serial fixture has no PxScene statistics instance.  Query the
	// same once-selected process dispatch directly so every AVBD execution path
	// reports actual rather than Scene-only/default ISA telemetry.
	PxAvbdCpuIsaTelemetry cpuIsa;
	PxGetAvbdCpuIsaTelemetry(cpuIsa);
	const char* requestedIsa = getAvbdCpuIsaModeName(cpuIsa.requestedIsa);
	const char* selectedIsa = getAvbdCpuIsaModeName(cpuIsa.selectedIsa);
	const char* compiledIsaBackends =
		getAvbdCpuIsaCompiledBackendNames(cpuIsa.compiledBackendMask);

	std::printf(
		"[AVBD_PERF] schema=2 snippet=SnippetDeformableVolumeAVBD "
		"case=%s buildProfile=%s requestedIsa=%s selectedIsa=%s "
		"compiledIsaBackends=%s fmaSupported=%u fmaUsed=%u "
		"forceIsaRejected=%u isaKernelSelfTest=%s isaProbeValue=%.9g sceneExecution=%s "
		"dispatcherThreads=%u physicalCores=%u "
		"softScheduler=%s softExecution=%s softWorkers=%u "
		"actualSoftWorkers=%u componentFallbackSteps=%llu "
		"nativeIslandSteps=%llu "
		"nativeIslandComponentFallbackOverlapFrames=%llu "
		"nativeIslandCausalLayerTaskOverlapFrames=%llu "
		"taskCount=%llu barrierCount=%llu "
		"taskGraphRequestedWorkers=%u taskGraphCompletedTasks=%llu "
		"taskGraphSerialTasks=%llu taskGraphPureSoftEligibleIslands=%llu "
		"taskGraphPureSoftEligibleParticles=%llu "
		"predictionTaskCount=%llu predictionCompletedTasks=%llu "
		"predictionPeakActiveTasks=%u predictionSerialStages=%llu "
		"writeBackTaskCount=%llu writeBackCompletedTasks=%llu "
		"writeBackPeakActiveTasks=%u writeBackSerialStages=%llu "
		"causalLayerTaskCount=%llu causalLayerCompletedTasks=%llu "
		"causalLayerPeakActiveTasks=%u causalLayerFanIns=%llu "
		"causalLayerSerialFallbacks=%llu causalLayerMaxOccupancy=%u "
		"causalLayerTotalOccupancy=%llu "
		"worldPlaneContactTaskCount=%llu "
		"worldPlaneContactCompletedTasks=%llu "
		"worldPlaneContactPeakActiveTasks=%u "
		"worldPlaneContactFanIns=%llu "
		"worldPlaneContactSerialFallbacks=%llu "
		"rigidBoxSdfContactTaskCount=%llu "
		"rigidBoxSdfContactCompletedTasks=%llu "
		"rigidBoxSdfContactPeakActiveTasks=%u "
		"rigidBoxSdfContactFanIns=%llu "
		"rigidBoxSdfContactSerialFallbacks=%llu "
		"rigidSphereSdfContactTaskCount=%llu "
		"rigidSphereSdfContactCompletedTasks=%llu "
		"rigidSphereSdfContactPeakActiveTasks=%u "
		"rigidSphereSdfContactFanIns=%llu "
		"rigidSphereSdfContactSerialFallbacks=%llu "
		"rigidCapsuleSdfContactTaskCount=%llu "
		"rigidCapsuleSdfContactCompletedTasks=%llu "
		"rigidCapsuleSdfContactPeakActiveTasks=%u "
		"rigidCapsuleSdfContactFanIns=%llu "
		"rigidCapsuleSdfContactSerialFallbacks=%llu "
		"rigidConvexSdfContactTaskCount=%llu "
		"rigidConvexSdfContactCompletedTasks=%llu "
		"rigidConvexSdfContactPeakActiveTasks=%u "
		"rigidConvexSdfContactFanIns=%llu "
		"rigidConvexSdfContactSerialFallbacks=%llu "
		"rigidTriangleSurfaceContactTaskCount=%llu "
		"rigidTriangleSurfaceContactCompletedTasks=%llu "
		"rigidTriangleSurfaceContactPeakActiveTasks=%u "
		"rigidTriangleSurfaceContactFanIns=%llu "
		"rigidTriangleSurfaceContactSerialFallbacks=%llu "
		"p4ColorPlanCount=%u p4DynamicAccessGroupCount=%u "
		"p4ColoredSerialSweeps=%llu p4ColoredSerialFallbackSweeps=%llu "
		"p8CensusParticleSolves=%llu p8CensusTriEvaluations=%llu "
		"p8CensusCorotationalTetEvaluations=%llu "
		"p8CensusNeoHookeanTetEvaluations=%llu "
		"p8CensusBendingEvaluations=%llu p8CensusContactEvaluations=%llu "
		"p8CensusTetPacket8FullPackets=%llu "
		"p8CensusTetPacket8TailLanes=%llu "
		"p8TetIrBodies=%llu p8TetIrPackets=%llu "
		"p8TetIrActiveLanes=%llu p8TetIrTailLanes=%llu "
		"p8TetIrActiveTailLanes=%llu "
		"p8TetIrInvalidBodies=%llu "
		"topologySoftBodies=%llu topologySoftParticles=%llu "
		"topologyTriElements=%llu topologyTetElements=%llu "
		"topologyBendElements=%llu topologySurfaceTriangles=%llu "
		"topologySurfaceVertices=%llu topologySurfaceEdges=%llu "
		"topologyRigidBoxes=%llu topologyRigidTriangleMeshTriangles=%llu "
		"warmupFrames=%u profileFrames=%u "
		"avgStepMs=%.9g p50StepMs=%.9g p95StepMs=%.9g maxStepMs=%.9g "
		"initialContactMs=%.9g solverMs=%.9g sceneMs=%.9g metricsMs=%.9g "
		"predictionMs=%.9g contactIndexMs=%.9g bodyPrecomputeMs=%.9g "
		"bodySolveMs=%.9g particleSolveMs=%.9g projectionMs=%.9g "
		"dualMs=%.9g redetectMs=%.9g velocityMs=%.9g frictionMs=%.9g "
		"solverUnattributedMs=%.9g closureMs=%.9g "
		"requestedOuterIterations=%llu requestedInnerIterations=%llu "
		"executedOuterIterations=%llu executedInnerIterations=%llu "
		"particleSweeps=%llu "
		"convergenceAuthority=localSolveResidualConsecutive "
		"convergenceTolerance=0.0001 convergenceSweeps=2 "
		"trustRegionLimitedParticleSteps=%llu "
		"positiveJLimitedParticleSteps=%llu "
		"positiveJRejectedParticleSteps=%llu "
		"nonFiniteRejectedParticleSteps=%llu "
		"tetLinearizationCacheFallbackParticleSteps=%llu "
		"legacyAppliedConvergedOuterIterations=%llu "
		"residualConvergedOuterIterations=%llu "
		"unsafeAppliedConvergenceCandidates=%llu "
		"budgetExhaustedOuterIterations=%llu "
		"shadowResidual1e5ConvergedOuterIterations=%llu "
		"shadowResidual1e5SavedInnerIterations=%llu "
		"shadowResidual1e4ConvergedOuterIterations=%llu "
		"shadowResidual1e4SavedInnerIterations=%llu "
		"workspaceGrowthEvents=%llu "
		"workspaceGrowthBytes=%llu contactWorkspaceGrowthEvents=%llu "
		"contactWorkspaceGrowthBytes=%llu contactOutputGrowthEvents=%llu "
		"contactOutputGrowthBytes=%llu finalMaxDisplacement=%.9g "
		"finalMaxLocalSolveDisplacement=%.9g "
		"finalMaxAppliedDisplacement=%.9g "
		"detectionCalls=%llu bodyPairs=%llu overlappingBodyPairs=%llu "
		"particleSurfaceCandidates=%llu insideTriangleTests=%llu "
		"closestTriangleTests=%llu selfTriangleTests=%llu "
		"rigidParticleBoxTests=%llu rigidParticleTests=%llu "
		"rigidTriangleFaceCandidates=%llu rigidTriangleFaceTests=%llu "
		"rigidTriangleEdgeCandidates=%llu rigidTriangleEdgeTests=%llu "
		"rigidTriangleVertexCandidates=%llu rigidTriangleVertexTests=%llu "
		"generatedGroundContacts=%llu "
		"generatedRigidContacts=%llu generatedSoftContacts=%llu "
		"generatedSelfContacts=%llu rigidParticleSphereTests=%llu "
		"rigidParticleCapsuleTests=%llu rigidParticleConvexTests=%llu "
		"rigidParticleTriangleSurfaceTests=%llu\n",
		config.caseName, buildProfile,
		requestedIsa, selectedIsa, compiledIsaBackends,
		(cpuIsa.capabilityMask & PxAvbdCpuIsaCapabilityFlag::eFMA) ? 1u : 0u,
		cpuIsa.fmaUsed,
		cpuIsa.forceModeRejected,
		cpuIsa.kernelSelfTestPassed ? "pass" : "fail",
		double(cpuIsa.kernelSelfTestValue),
		config.executionName,
		config.dispatcherThreads, physicalCores,
		softScheduler, softExecution, softWorkers, actualSoftWorkers,
		static_cast<unsigned long long>(metrics.componentFallbackSteps),
		static_cast<unsigned long long>(metrics.nativeIslandSteps),
		static_cast<unsigned long long>(
			metrics.nativeIslandComponentFallbackOverlapFrames),
		static_cast<unsigned long long>(
			metrics.nativeIslandCausalLayerTaskOverlapFrames),
		static_cast<unsigned long long>(
			metrics.taskGraphSubmittedSolveTasks),
		static_cast<unsigned long long>(metrics.taskGraphBarrierTasks),
		metrics.taskGraphRequestedDispatcherWorkers,
		static_cast<unsigned long long>(
			metrics.taskGraphCompletedSolveTasks),
		static_cast<unsigned long long>(
			metrics.taskGraphSerialSolveTasks),
		static_cast<unsigned long long>(
			metrics.taskGraphPureSoftEligibleIslands),
		static_cast<unsigned long long>(
			metrics.taskGraphPureSoftEligibleParticles),
		static_cast<unsigned long long>(
			metrics.taskGraphSubmittedPredictionTasks),
		static_cast<unsigned long long>(
			metrics.taskGraphCompletedPredictionTasks),
		metrics.taskGraphPeakActivePredictionTasks,
		static_cast<unsigned long long>(
			metrics.taskGraphSerialPredictionStages),
		static_cast<unsigned long long>(
			metrics.taskGraphSubmittedWriteBackTasks),
		static_cast<unsigned long long>(
			metrics.taskGraphCompletedWriteBackTasks),
		metrics.taskGraphPeakActiveWriteBackTasks,
		static_cast<unsigned long long>(
			metrics.taskGraphSerialWriteBackStages),
		static_cast<unsigned long long>(
			metrics.taskGraphSubmittedCausalLayerTasks),
		static_cast<unsigned long long>(
			metrics.taskGraphCompletedCausalLayerTasks),
		metrics.taskGraphPeakActiveCausalLayerTasks,
		static_cast<unsigned long long>(
			metrics.taskGraphCausalLayerFanIns),
		static_cast<unsigned long long>(
			metrics.taskGraphSerialCausalLayerFallbacks),
		metrics.taskGraphMaxCausalLayerOccupancy,
		static_cast<unsigned long long>(
			metrics.taskGraphTotalCausalLayerOccupancy),
		static_cast<unsigned long long>(
			metrics.taskGraphSubmittedWorldPlaneContactTasks),
		static_cast<unsigned long long>(
			metrics.taskGraphCompletedWorldPlaneContactTasks),
		metrics.taskGraphPeakActiveWorldPlaneContactTasks,
		static_cast<unsigned long long>(
			metrics.taskGraphWorldPlaneContactFanIns),
		static_cast<unsigned long long>(
			metrics.taskGraphSerialWorldPlaneContactFallbacks),
		static_cast<unsigned long long>(
			metrics.taskGraphSubmittedRigidBoxSdfContactTasks),
		static_cast<unsigned long long>(
			metrics.taskGraphCompletedRigidBoxSdfContactTasks),
		metrics.taskGraphPeakActiveRigidBoxSdfContactTasks,
		static_cast<unsigned long long>(
			metrics.taskGraphRigidBoxSdfContactFanIns),
		static_cast<unsigned long long>(
			metrics.taskGraphSerialRigidBoxSdfContactFallbacks),
		static_cast<unsigned long long>(
			metrics.taskGraphSubmittedRigidSphereSdfContactTasks),
		static_cast<unsigned long long>(
			metrics.taskGraphCompletedRigidSphereSdfContactTasks),
		metrics.taskGraphPeakActiveRigidSphereSdfContactTasks,
		static_cast<unsigned long long>(
			metrics.taskGraphRigidSphereSdfContactFanIns),
		static_cast<unsigned long long>(
			metrics.taskGraphSerialRigidSphereSdfContactFallbacks),
		static_cast<unsigned long long>(
			metrics.taskGraphSubmittedRigidCapsuleSdfContactTasks),
		static_cast<unsigned long long>(
			metrics.taskGraphCompletedRigidCapsuleSdfContactTasks),
		metrics.taskGraphPeakActiveRigidCapsuleSdfContactTasks,
		static_cast<unsigned long long>(
			metrics.taskGraphRigidCapsuleSdfContactFanIns),
		static_cast<unsigned long long>(
			metrics.taskGraphSerialRigidCapsuleSdfContactFallbacks),
		static_cast<unsigned long long>(
			metrics.taskGraphSubmittedRigidConvexSdfContactTasks),
		static_cast<unsigned long long>(
			metrics.taskGraphCompletedRigidConvexSdfContactTasks),
		metrics.taskGraphPeakActiveRigidConvexSdfContactTasks,
		static_cast<unsigned long long>(
			metrics.taskGraphRigidConvexSdfContactFanIns),
		static_cast<unsigned long long>(
			metrics.taskGraphSerialRigidConvexSdfContactFallbacks),
		static_cast<unsigned long long>(
			metrics.taskGraphSubmittedRigidTriangleSurfaceContactTasks),
		static_cast<unsigned long long>(
			metrics.taskGraphCompletedRigidTriangleSurfaceContactTasks),
		metrics.taskGraphPeakActiveRigidTriangleSurfaceContactTasks,
		static_cast<unsigned long long>(
			metrics.taskGraphRigidTriangleSurfaceContactFanIns),
		static_cast<unsigned long long>(
			metrics.taskGraphSerialRigidTriangleSurfaceContactFallbacks),
		stages.particlePrimalColorCount,
		stages.particlePrimalDynamicAccessGroupCount,
		static_cast<unsigned long long>(
			stages.particlePrimalColoredSerialSweeps),
		static_cast<unsigned long long>(
			stages.particlePrimalColoredSerialFallbackSweeps),
		static_cast<unsigned long long>(
			stages.particlePrimalCensusDynamicParticleSolves),
		static_cast<unsigned long long>(
			stages.particlePrimalCensusTriangleEvaluations),
		static_cast<unsigned long long>(
			stages.particlePrimalCensusCorotationalTetEvaluations),
		static_cast<unsigned long long>(
			stages.particlePrimalCensusNeoHookeanTetEvaluations),
		static_cast<unsigned long long>(
			stages.particlePrimalCensusBendingEvaluations),
		static_cast<unsigned long long>(
			stages.particlePrimalCensusContactEvaluations),
		static_cast<unsigned long long>(
			stages.particlePrimalCensusTetPacket8FullPackets),
		static_cast<unsigned long long>(
			stages.particlePrimalCensusTetPacket8TailLanes),
		static_cast<unsigned long long>(
			stages.particlePrimalTetPacketIrBodies),
		static_cast<unsigned long long>(
			stages.particlePrimalTetPacketIrPackets),
		static_cast<unsigned long long>(
			stages.particlePrimalTetPacketIrActiveLanes),
		static_cast<unsigned long long>(
			stages.particlePrimalTetPacketIrTailLanes),
		static_cast<unsigned long long>(
			stages.particlePrimalTetPacketIrActiveTailLanes),
		static_cast<unsigned long long>(
			stages.particlePrimalTetPacketIrInvalidBodies),
		static_cast<unsigned long long>(metrics.topologySoftBodies),
		static_cast<unsigned long long>(metrics.topologySoftParticles),
		static_cast<unsigned long long>(metrics.topologyTriElements),
		static_cast<unsigned long long>(metrics.topologyTetElements),
		static_cast<unsigned long long>(metrics.topologyBendElements),
		static_cast<unsigned long long>(metrics.topologySurfaceTriangles),
		static_cast<unsigned long long>(metrics.topologySurfaceVertices),
		static_cast<unsigned long long>(metrics.topologySurfaceEdges),
		static_cast<unsigned long long>(metrics.topologyRigidBoxes),
		static_cast<unsigned long long>(
			metrics.topologyRigidTriangleMeshTriangles),
		metrics.warmupFrames,
		metrics.profiledFrames, double(metrics.avgStepMs),
		double(metrics.p50StepMs),
		double(metrics.p95StepMs), double(metrics.maxStepMs),
		double(metrics.initialContactMs / divisor),
		double(metrics.solverMs / divisor),
		double(metrics.sceneMs / divisor),
		double(metrics.metricsMs / divisor),
		double(stages.predictionMs / divisor),
		double(stages.contactIndexMs / divisor),
		double(stages.bodyPrecomputeMs / divisor),
		double(stages.bodySolveMs / divisor),
		double(stages.particleSolveMs / divisor),
		double(stages.projectionMs / divisor),
		double(stages.dualMs / divisor),
		double(stages.redetectMs / divisor),
		double(stages.velocityMs / divisor),
		double(stages.frictionMs / divisor),
		double((metrics.solverMs - solverStageMs) / divisor),
		double(closureMs / divisor),
		static_cast<unsigned long long>(stages.requestedOuterIterations),
		static_cast<unsigned long long>(stages.requestedInnerIterations),
		static_cast<unsigned long long>(stages.executedOuterIterations),
		static_cast<unsigned long long>(stages.executedInnerIterations),
		static_cast<unsigned long long>(stages.particleSweeps),
		static_cast<unsigned long long>(
			stages.trustRegionLimitedParticleSteps),
		static_cast<unsigned long long>(
			stages.positiveJLimitedParticleSteps),
		static_cast<unsigned long long>(
			stages.positiveJRejectedParticleSteps),
		static_cast<unsigned long long>(
			stages.nonFiniteRejectedParticleSteps),
		static_cast<unsigned long long>(
			stages.tetLinearizationCacheFallbackParticleSteps),
		static_cast<unsigned long long>(
			stages.legacyAppliedConvergedOuterIterations),
		static_cast<unsigned long long>(
			stages.residualConvergedOuterIterations),
		static_cast<unsigned long long>(
			stages.unsafeAppliedConvergenceCandidates),
		static_cast<unsigned long long>(
			stages.budgetExhaustedOuterIterations),
		static_cast<unsigned long long>(
			stages.shadowResidual1e5ConvergedOuterIterations),
		static_cast<unsigned long long>(
			stages.shadowResidual1e5SavedInnerIterations),
		static_cast<unsigned long long>(
			stages.shadowResidual1e4ConvergedOuterIterations),
		static_cast<unsigned long long>(
			stages.shadowResidual1e4SavedInnerIterations),
		static_cast<unsigned long long>(stages.workspaceGrowthEvents),
		static_cast<unsigned long long>(stages.workspaceGrowthBytes),
		static_cast<unsigned long long>(
			stages.contactWorkspaceGrowthEvents),
		static_cast<unsigned long long>(
			stages.contactWorkspaceGrowthBytes),
		static_cast<unsigned long long>(
			stages.contactOutputGrowthEvents),
		static_cast<unsigned long long>(
			stages.contactOutputGrowthBytes),
		double(stages.finalMaxDisplacement),
		double(stages.finalMaxLocalSolveDisplacement),
		double(stages.finalMaxAppliedDisplacement),
		static_cast<unsigned long long>(
			metrics.collision.detectionCalls),
		static_cast<unsigned long long>(metrics.collision.bodyPairs),
		static_cast<unsigned long long>(
			metrics.collision.overlappingBodyPairs),
		static_cast<unsigned long long>(
			metrics.collision.particleSurfaceCandidates),
		static_cast<unsigned long long>(
			metrics.collision.insideTriangleTests),
		static_cast<unsigned long long>(
			metrics.collision.closestTriangleTests),
		static_cast<unsigned long long>(
			metrics.collision.selfTriangleTests),
		static_cast<unsigned long long>(
			metrics.collision.rigidParticleBoxTests),
		static_cast<unsigned long long>(
			metrics.collisionRigidParticleTests),
		static_cast<unsigned long long>(
			metrics.collision.rigidTriangleSurfaceFaceCandidates),
		static_cast<unsigned long long>(
			metrics.collision.rigidTriangleSurfaceFaceTests),
		static_cast<unsigned long long>(
			metrics.collision.rigidTriangleSurfaceEdgeCandidates),
		static_cast<unsigned long long>(
			metrics.collision.rigidTriangleSurfaceEdgeTests),
		static_cast<unsigned long long>(
			metrics.collision.rigidTriangleSurfaceVertexCandidates),
		static_cast<unsigned long long>(
			metrics.collision.rigidTriangleSurfaceVertexTests),
		static_cast<unsigned long long>(
			metrics.collision.generatedGroundContacts),
		static_cast<unsigned long long>(
			metrics.collision.generatedRigidContacts),
		static_cast<unsigned long long>(
			metrics.collision.generatedSoftContacts),
		static_cast<unsigned long long>(
			metrics.collision.generatedSelfContacts),
		static_cast<unsigned long long>(
			metrics.collision.rigidParticleSphereTests),
		static_cast<unsigned long long>(
			metrics.collision.rigidParticleCapsuleTests),
		static_cast<unsigned long long>(
			metrics.collision.rigidParticleConvexTests),
		static_cast<unsigned long long>(
			metrics.collision.rigidParticleTriangleSurfaceTests));
	std::printf(
		"[AVBD_WORLD_STATIC_TANGENT_OWNER] rows=%llu appliedRows=%llu\n",
		static_cast<unsigned long long>(
			stages.worldStaticVelocityTangentOwnerRows),
		static_cast<unsigned long long>(
			stages.worldStaticVelocityTangentAppliedRows));
}

void printHeadlessGateReport(
	const DeformableVolumeMetrics& metrics,
	const HeadlessGateReportConfig& config)
{
	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetDeformableVolumeAVBD "
		"case=%s solver=%s validation=%s "
		"sceneSoftIntegration=%u status=%s initialized=%u "
		"frames=%u fetchFailures=%u particles=%u softBodies=%u "
		"tetElements=%u surfaceTriangles=%u rigidBoxes=%u "
		"sceneStatics=%u sceneDynamics=%u sceneDeformableVolumes=%u "
		"sceneActorCreated=%u sceneShapeAttached=%u "
		"sceneSimulationMeshAttached=%u "
		"sceneHostBuffersInitialized=%u sceneActorAdded=%u "
		"sceneActorRemoved=%u sceneActorReleased=%u "
		"sceneBoundsFinite=%u "
		"sceneSecondVolumeActorCreated=%u "
		"sceneSecondVolumeHostBuffersInitialized=%u "
		"sceneSecondVolumeActorAdded=%u "
		"sceneSecondVolumeActorRemoved=%u "
		"sceneSecondVolumeActorReleased=%u "
		"sceneSecondVolumeBoundsFinite=%u "
		"sceneSoftInitiallyAwake=%u "
		"sceneSoftFirstSlept=%u "
		"sceneSoftFirstSleepFrame=%u "
		"sceneSoftSleepWakeCounterZero=%u "
		"sceneSoftSleepVelocitiesZero=%u "
		"sceneSoftStableWhileSleeping=%u "
		"sceneSoftCounterWakeIssued=%u "
		"sceneSoftWokeByCounter=%u "
		"sceneSoftCounterWakeFrame=%u "
		"sceneSoftSecondSlept=%u "
		"sceneSoftSecondSleepFrame=%u "
		"sceneSoftVelocityWakeIssued=%u "
		"sceneSoftWokeByVelocity=%u "
		"sceneSoftVelocityWakeFrame=%u "
		"sceneSoftMovedAfterVelocityWake=%u "
		"sceneSoftVelocityStopIssued=%u "
		"sceneSoftFinalSlept=%u "
		"sceneSoftFinalSleepFrame=%u "
		"sceneSoftRigidWakeActorAdded=%u "
		"sceneSoftWokeByRigid=%u "
		"sceneSoftRigidWakeFrame=%u "
		"sceneSoftMovedAfterRigidWake=%u "
		"sceneMixedFirstSlept=%u "
		"sceneMixedFirstSleepFrame=%u "
		"sceneMixedFirstStable=%u "
		"sceneMixedSecondStayedAwake=%u "
		"sceneMixedSecondMoved=%u "
		"sceneSoftChurnRemoveCount=%u "
		"sceneSoftChurnReaddCount=%u "
		"sceneSoftChurnCycles=%u "
		"sceneSoftChurnPostCompactMoveCount=%u "
		"sceneSoftChurnStable=%u "
		"sceneBufferMutationIssued=%u "
		"sceneBufferMutationWoke=%u "
		"sceneBufferMutationApplied=%u "
		"sceneBufferDriveIssued=%u "
		"sceneBufferPinHeld=%u "
		"sceneBufferDynamicMoved=%u "
		"sceneBufferInvMassRestored=%u "
		"sceneBufferRestoredMoved=%u "
		"sceneBufferResetIssued=%u "
		"sceneWorldPinCreated=%u sceneWorldPinHeld=%u "
		"sceneWorldPinActorReadded=%u "
		"sceneWorldPinReleased=%u "
		"sceneWorldPinMovedAfterRelease=%u "
		"sceneRigidAttachmentActorAdded=%u "
		"sceneRigidAttachmentInitiallySleeping=%u "
		"sceneRigidAttachmentCreated=%u "
		"sceneRigidAttachmentRigidWoke=%u "
		"sceneRigidAttachmentRigidMoved=%u "
		"sceneRigidAttachmentHeldAcrossReadd=%u "
		"sceneRigidAttachmentReleased=%u "
		"sceneRigidAttachmentSeparatedAfterRelease=%u "
		"sceneArticulationCreated=%u "
		"sceneArticulationAdded=%u "
		"sceneArticulationInitiallySleeping=%u "
		"sceneArticulationWoke=%u "
		"sceneArticulationJointSubspaceHeld=%u "
		"sceneArticulationRootStable=%u "
		"sceneElementFilterCreated=%u "
		"sceneElementFilterActorReadded=%u "
		"sceneElementFilterSuppressedContact=%u "
		"sceneElementFilterReleased=%u "
		"sceneElementFilterContactRestored=%u "
		"scenePartialFilterUnfilteredContactHeld=%u "
		"scenePartialFilterExactOwnership=%u "
		"sceneKinematicActorAdded=%u "
		"sceneKinematicTargetIssued=%u "
		"sceneKinematicTargetReached=%u "
		"sceneKinematicSoftWoke=%u "
		"sceneKinematicSoftMoved=%u "
		"sceneKinematicContactObserved=%u "
		"sceneVolumeTargetBound=%u "
		"sceneVolumeTargetMutated=%u "
		"sceneVolumeTargetWoke=%u "
		"sceneVolumeTargetReached=%u "
		"sceneVolumePartialInactiveIgnored=%u "
		"sceneVolumePartialActivated=%u "
		"sceneVolumePartialActivatedReached=%u "
		"sceneSecondSceneCreated=%u "
		"sceneSecondSceneSolverMatched=%u "
		"scenePrimarySceneReleased=%u "
		"sceneSecondSceneReleased=%u "
		"sceneMultiPrimaryStable=%u "
		"sceneMultiPrimaryDetachedStable=%u "
		"sceneMultiSecondaryUpdatedBeforeRelease=%u "
		"sceneMultiSecondaryUpdatedAfterRelease=%u "
		"sceneSoftSoftBothSlept=%u "
		"sceneSoftSoftDriveIssued=%u "
		"sceneSoftSoftDriverWoke=%u "
		"sceneSoftSoftTargetWoke=%u "
		"sceneSoftSoftTargetWakeFrame=%u "
		"sceneSoftSoftTargetMoved=%u "
		"sceneSoftSoftResetIssued=%u "
		"sceneSoftSoftBothFinalSlept=%u "
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
		"sceneStaticShapeDetached=%u "
		"sceneStaticShapeReattached=%u "
		"sceneStaticActorRemoved=%u sceneStaticActorReadded=%u "
		"sceneDynamicActorAdded=%u "
		"sceneDynamicActorRemoved=%u "
		"sceneDynamicActorReleased=%u "
		"sceneDynamicInitiallySleeping=%u "
		"sceneDynamicWokeBySoft=%u "
		"sceneDynamicFirstWakeFrame=%u "
		"sceneDynamicShapeDetached=%u "
		"sceneDynamicShapeReattached=%u "
		"sceneDynamicActorReadded=%u "
		"sceneDynamicReaddedSleeping=%u "
		"sceneDynamicRewokeBySoft=%u "
		"sceneDynamicSecondWakeFrame=%u "
		"sceneSecondDynamicActorAdded=%u "
		"sceneSecondDynamicActorRemoved=%u "
		"sceneSecondDynamicActorReleased=%u "
		"sceneSecondDynamicInitiallySleeping=%u "
		"sceneSecondDynamicWokeBySoft=%u "
		"sceneSecondDynamicFirstWakeFrame=%u "
		"groundContactFrames=%u rigidContactFrames=%u "
		"softContactFrames=%u maxGroundContacts=%u "
		"maxRigidContacts=%u maxSoftContacts=%u "
		"invalidContactSourceSamples=%u finalInsideParticles=%u "
		"nonFiniteParticleSamples=%u "
		"invertedElementSamples=%u firstInversionFrame=%u "
		"firstInversionBody=%u firstInversionElement=%u "
		"invertedBodiesMask=%u minDetF=%.9g maxDetF=%.9g "
		"minBodyVolumeRatio=%.9g maxBodyVolumeRatio=%.9g "
		"minY=%.9g maxY=%.9g finalMinY=%.9g finalMaxY=%.9g "
		"maxParticleSpeed=%.9g finalMaxParticleSpeed=%.9g "
		"maxCentroidDrop=%.9g "
		"sceneSecondVolumeMaxCentroidDrop=%.9g "
		"sceneSecondVolumeFinalCentroidY=%.9g "
		"sceneWorldPinMaxDrift=%.9g "
		"sceneWorldPinReleasedMaxDisplacement=%.9g "
		"sceneRigidAttachmentMaxDrift=%.9g "
		"sceneRigidAttachmentMaxRigidDisplacement=%.9g "
		"sceneRigidAttachmentMaxRigidSpeed=%.9g "
		"sceneRigidAttachmentReleasedSeparation=%.9g "
		"sceneArticulationRootMaxDisplacement=%.9g "
		"sceneArticulationChildMaxForbiddenDisplacement=%.9g "
		"sceneArticulationChildMaxAngularDisplacement=%.9g "
		"sceneElementFilterMinY=%.9g "
		"sceneElementFilterFinalMinY=%.9g "
		"scenePartialFilterUnfilteredMinY=%.9g "
		"sceneKinematicMaxPoseError=%.9g "
		"sceneKinematicSoftDisplacement=%.9g "
		"sceneKinematicFinalY=%.9g "
		"sceneVolumeTargetFinalMaxError=%.9g "
		"sceneVolumeTargetMaxDisplacement=%.9g "
		"sceneVolumePartialInactiveDecoyDistance=%.9g "
		"sceneDynamicInitialY=%.9g "
		"sceneDynamicMinY=%.9g sceneDynamicFinalY=%.9g "
		"sceneDynamicMaxDrop=%.9g "
		"sceneDynamicPreContactMaxDrop=%.9g "
		"sceneDynamicMaxDownSpeed=%.9g "
		"sceneSecondDynamicInitialY=%.9g "
		"sceneSecondDynamicMinY=%.9g "
		"sceneSecondDynamicFinalY=%.9g "
		"sceneSecondDynamicMaxDrop=%.9g "
		"sceneSecondDynamicPreContactMaxDrop=%.9g "
		"sceneSecondDynamicMaxDownSpeed=%.9g "
		"minDynamicSurfaceSeparation=%.9g "
		"finalDynamicSurfaceSeparation=%.9g "
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
		"solverReadbackMatched=%u "
		"fatalErrors=%u warningErrors=%u cleanupComplete=%u\n",
		config.caseName,
		config.solverName,
		config.validation,
		config.sceneIntegrated ? 1u : 0u,
		config.passed ? "PASS" : "FAIL", metrics.initialized,
		metrics.completedFrames, metrics.fetchFailures,
		metrics.particles, metrics.softBodies, metrics.tetElements,
		metrics.surfaceTriangles, metrics.rigidBoxes,
		metrics.sceneStatics, metrics.sceneDynamics,
		metrics.sceneDeformableVolumes,
		metrics.sceneActorCreated, metrics.sceneShapeAttached,
		metrics.sceneSimulationMeshAttached,
		metrics.sceneHostBuffersInitialized, metrics.sceneActorAdded,
		metrics.sceneActorRemoved, metrics.sceneActorReleased,
		metrics.sceneBoundsFinite,
		metrics.sceneSecondVolumeActorCreated,
		metrics.sceneSecondVolumeHostBuffersInitialized,
		metrics.sceneSecondVolumeActorAdded,
		metrics.sceneSecondVolumeActorRemoved,
		metrics.sceneSecondVolumeActorReleased,
		metrics.sceneSecondVolumeBoundsFinite,
		metrics.sceneSoftInitiallyAwake,
		metrics.sceneSoftFirstSlept,
		metrics.sceneSoftFirstSleepFrame,
		metrics.sceneSoftSleepWakeCounterZero,
		metrics.sceneSoftSleepVelocitiesZero,
		metrics.sceneSoftStableWhileSleeping,
		metrics.sceneSoftCounterWakeIssued,
		metrics.sceneSoftWokeByCounter,
		metrics.sceneSoftCounterWakeFrame,
		metrics.sceneSoftSecondSlept,
		metrics.sceneSoftSecondSleepFrame,
		metrics.sceneSoftVelocityWakeIssued,
		metrics.sceneSoftWokeByVelocity,
		metrics.sceneSoftVelocityWakeFrame,
		metrics.sceneSoftMovedAfterVelocityWake,
		metrics.sceneSoftVelocityStopIssued,
		metrics.sceneSoftFinalSlept,
		metrics.sceneSoftFinalSleepFrame,
		metrics.sceneSoftRigidWakeActorAdded,
		metrics.sceneSoftWokeByRigid,
		metrics.sceneSoftRigidWakeFrame,
		metrics.sceneSoftMovedAfterRigidWake,
		metrics.sceneMixedFirstSlept,
		metrics.sceneMixedFirstSleepFrame,
		metrics.sceneMixedFirstStable,
		metrics.sceneMixedSecondStayedAwake,
		metrics.sceneMixedSecondMoved,
		metrics.sceneSoftChurnRemoveCount,
		metrics.sceneSoftChurnReaddCount,
		metrics.sceneSoftChurnCycles,
		metrics.sceneSoftChurnPostCompactMoveCount,
		metrics.sceneSoftChurnStable,
		metrics.sceneBufferMutationIssued,
		metrics.sceneBufferMutationWoke,
		metrics.sceneBufferMutationApplied,
		metrics.sceneBufferDriveIssued,
		metrics.sceneBufferPinHeld,
		metrics.sceneBufferDynamicMoved,
		metrics.sceneBufferInvMassRestored,
		metrics.sceneBufferRestoredMoved,
		metrics.sceneBufferResetIssued,
		metrics.sceneWorldPinCreated,
		metrics.sceneWorldPinHeld,
		metrics.sceneWorldPinActorReadded,
		metrics.sceneWorldPinReleased,
		metrics.sceneWorldPinMovedAfterRelease,
		metrics.sceneRigidAttachmentActorAdded,
		metrics.sceneRigidAttachmentInitiallySleeping,
		metrics.sceneRigidAttachmentCreated,
		metrics.sceneRigidAttachmentRigidWoke,
		metrics.sceneRigidAttachmentRigidMoved,
		metrics.sceneRigidAttachmentHeldAcrossReadd,
		metrics.sceneRigidAttachmentReleased,
		metrics.sceneRigidAttachmentSeparatedAfterRelease,
		metrics.sceneArticulationCreated,
		metrics.sceneArticulationAdded,
		metrics.sceneArticulationInitiallySleeping,
		metrics.sceneArticulationWoke,
		metrics.sceneArticulationJointSubspaceHeld,
		metrics.sceneArticulationRootStable,
		metrics.sceneElementFilterCreated,
		metrics.sceneElementFilterActorReadded,
		metrics.sceneElementFilterSuppressedContact,
		metrics.sceneElementFilterReleased,
		metrics.sceneElementFilterContactRestored,
		metrics.scenePartialFilterUnfilteredContactHeld,
		metrics.scenePartialFilterExactOwnership,
		metrics.sceneKinematicActorAdded,
		metrics.sceneKinematicTargetIssued,
		metrics.sceneKinematicTargetReached,
		metrics.sceneKinematicSoftWoke,
		metrics.sceneKinematicSoftMoved,
		metrics.sceneKinematicContactObserved,
		metrics.sceneVolumeTargetBound,
		metrics.sceneVolumeTargetMutated,
		metrics.sceneVolumeTargetWoke,
		metrics.sceneVolumeTargetReached,
		metrics.sceneVolumePartialInactiveIgnored,
		metrics.sceneVolumePartialActivated,
		metrics.sceneVolumePartialActivatedReached,
		metrics.sceneSecondSceneCreated,
		metrics.sceneSecondSceneSolverMatched,
		metrics.scenePrimarySceneReleased,
		metrics.sceneSecondSceneReleased,
		metrics.sceneMultiPrimaryStable,
		metrics.sceneMultiPrimaryDetachedStable,
		metrics.sceneMultiSecondaryUpdatedBeforeRelease,
		metrics.sceneMultiSecondaryUpdatedAfterRelease,
		metrics.sceneSoftSoftBothSlept,
		metrics.sceneSoftSoftDriveIssued,
		metrics.sceneSoftSoftDriverWoke,
		metrics.sceneSoftSoftTargetWoke,
		metrics.sceneSoftSoftTargetWakeFrame,
		metrics.sceneSoftSoftTargetMoved,
		metrics.sceneSoftSoftResetIssued,
		metrics.sceneSoftSoftBothFinalSlept,
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
		metrics.sceneStaticShapeDetached,
		metrics.sceneStaticShapeReattached,
		metrics.sceneStaticActorRemoved,
		metrics.sceneStaticActorReadded,
		metrics.sceneDynamicActorAdded,
		metrics.sceneDynamicActorRemoved,
		metrics.sceneDynamicActorReleased,
		metrics.sceneDynamicInitiallySleeping,
		metrics.sceneDynamicWokeBySoft,
		metrics.sceneDynamicFirstWakeFrame,
		metrics.sceneDynamicShapeDetached,
		metrics.sceneDynamicShapeReattached,
		metrics.sceneDynamicActorReadded,
		metrics.sceneDynamicReaddedSleeping,
		metrics.sceneDynamicRewokeBySoft,
		metrics.sceneDynamicSecondWakeFrame,
		metrics.sceneSecondDynamicActorAdded,
		metrics.sceneSecondDynamicActorRemoved,
		metrics.sceneSecondDynamicActorReleased,
		metrics.sceneSecondDynamicInitiallySleeping,
		metrics.sceneSecondDynamicWokeBySoft,
		metrics.sceneSecondDynamicFirstWakeFrame,
		metrics.groundContactFrames,
		metrics.rigidContactFrames, metrics.softContactFrames,
		metrics.maxGroundContacts, metrics.maxRigidContacts,
		metrics.maxSoftContacts, metrics.invalidContactSourceSamples,
		metrics.finalInsideParticles,
		metrics.nonFiniteParticleSamples, metrics.invertedElementSamples,
		metrics.firstInversionFrame, metrics.firstInversionBody,
		metrics.firstInversionElement, metrics.invertedBodiesMask,
		double(metrics.minDetF), double(metrics.maxDetF),
		double(metrics.minBodyVolumeRatio),
		double(metrics.maxBodyVolumeRatio), double(metrics.minY),
		double(metrics.maxY), double(metrics.finalMinY),
		double(metrics.finalMaxY), double(metrics.maxParticleSpeed),
		double(metrics.finalMaxParticleSpeed),
		double(metrics.maxCentroidDrop),
		double(metrics.sceneSecondVolumeMaxCentroidDrop),
		double(metrics.sceneSecondVolumeFinalCentroidY),
		double(metrics.sceneWorldPinMaxDrift),
		double(metrics.sceneWorldPinReleasedMaxDisplacement),
		double(metrics.sceneRigidAttachmentMaxDrift),
		double(metrics.sceneRigidAttachmentMaxRigidDisplacement),
		double(metrics.sceneRigidAttachmentMaxRigidSpeed),
		double(metrics.sceneRigidAttachmentReleasedSeparation),
		double(metrics.sceneArticulationRootMaxDisplacement),
		double(
			metrics.sceneArticulationChildMaxForbiddenDisplacement),
		double(
			metrics.sceneArticulationChildMaxAngularDisplacement),
		double(metrics.sceneElementFilterMinY),
		double(metrics.sceneElementFilterFinalMinY),
		double(metrics.scenePartialFilterUnfilteredMinY),
		double(metrics.sceneKinematicMaxPoseError),
		double(metrics.sceneKinematicSoftDisplacement),
		double(metrics.sceneKinematicFinalY),
		double(metrics.sceneVolumeTargetFinalMaxError),
		double(metrics.sceneVolumeTargetMaxDisplacement),
		double(
			metrics.
				sceneVolumePartialInactiveDecoyDistance),
		double(config.dynamicInitialY),
		double(metrics.sceneDynamicMinY),
		double(metrics.sceneDynamicFinalY),
		double(metrics.sceneDynamicMaxDrop),
		double(metrics.sceneDynamicPreContactMaxDrop),
		double(metrics.sceneDynamicMaxDownSpeed),
		double(config.secondDynamicInitialY),
		double(metrics.sceneSecondDynamicMinY),
		double(metrics.sceneSecondDynamicFinalY),
		double(metrics.sceneSecondDynamicMaxDrop),
		double(metrics.sceneSecondDynamicPreContactMaxDrop),
		double(metrics.sceneSecondDynamicMaxDownSpeed),
		double(metrics.minDynamicSurfaceSeparation),
		double(metrics.finalDynamicSurfaceSeparation),
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
		metrics.solverReadbackMatched ? 1u : 0u,
		config.fatalErrors, config.warningErrors,
		metrics.cleanupComplete);
}


void printOgcSandwichReport(
	const OgcSandwichMetrics& metrics,
	const OgcSandwichReportConfig& config)
{
	const bool surfaceSeparated = isOgcSandwichSurfaceSeparated(metrics);
	const bool jawsCompressed =
		isOgcSandwichCompressed(metrics, config.minJawCompression);
	const bool noEscape = isOgcSandwichContained(
		metrics, config.maxLateralOffset, config.maxLateralSpeed,
		config.maxNormalOffset, config.maxNormalSpeed);
	std::printf(
		"[AVBD_OGC_SANDWICH] pressureDriveFrames=%u "
		"nativeIslandSteps=%llu generatedRigidContacts=%llu "
		"minUpperVertexSdf=%.9g minLowerVertexSdf=%.9g "
		"upperTrianglePenetrationFrames=%u "
		"lowerTrianglePenetrationFrames=%u "
		"maxUpperCompression=%.9g maxLowerCompression=%.9g "
		"upperMinDetF=%.9g lowerMinDetF=%.9g "
		"maxBoxLateralOffset=%.9g maxBoxLateralSpeed=%.9g "
		"maxBoxNormalOffset=%.9g maxBoxNormalSpeed=%.9g "
		"surfaceSeparated=%u jawsCompressed=%u noEscape=%u result=%s\n",
		config.pressureDriveFrames,
		static_cast<unsigned long long>(metrics.nativeIslandSteps),
		static_cast<unsigned long long>(metrics.generatedRigidContacts),
		double(metrics.minUpperJawSignedSdf),
		double(metrics.minLowerJawSignedSdf),
		metrics.upperJawTrianglePenetrationFrames,
		metrics.lowerJawTrianglePenetrationFrames,
		double(metrics.maxUpperCompression),
		double(metrics.maxLowerCompression),
		double(metrics.upperJawMinDetF), double(metrics.lowerJawMinDetF),
		double(metrics.maxBoxLateralOffset),
		double(metrics.maxBoxLateralSpeed),
		double(metrics.maxBoxNormalOffset),
		double(metrics.maxBoxNormalSpeed),
		surfaceSeparated ? 1u : 0u, jawsCompressed ? 1u : 0u,
		noEscape ? 1u : 0u, config.passed ? "PASS" : "FAIL");
}

void printVisualInteractionReport(
	const VisualInteractionMetrics& metrics,
	const VisualInteractionReportConfig& config)
{
	const bool dynamicFalling = config.dynamicInitiallySleeping == 0 &&
		config.dynamicMaxDrop > 0.25f &&
		config.dynamicMaxDownSpeed > 0.25f;
	const bool dynamicSurfaceSeparated =
		isVisualDynamicSurfaceSeparated(metrics);
	const bool staticSurfaceSeparated =
		isVisualStaticSurfaceSeparated(metrics);
	const bool allSurfacesSeparated =
		dynamicSurfaceSeparated && staticSurfaceSeparated;
	const bool mixedInteractionsObserved = dynamicSurfaceSeparated &&
		metrics.initialized && metrics.collisionTelemetryEnabled &&
		metrics.nativeIslandSteps > 0 && config.dynamicActorAdded == 1 &&
		dynamicFalling && config.dynamicActorRemoved == 1 &&
		config.dynamicActorReleased == 1;
	const bool gateActive = config.completedFrames >= config.gateMinFrames;

	std::printf(
		"[AVBD_VISUAL_INTERACTIONS] dynamicRigid=falling-box "
		"gateActive=%u nativeOgcObserved=%u initiallyAwake=%u falling=%u "
		"dynamicSurfaceSeparated=%u staticSurfaceSeparated=%u "
		"surfaceSeparated=%u nativeIslandSteps=%llu "
		"generatedRigidContacts=%llu generatedSoftContacts=%llu "
		"rigidContactFrames=%u softContactFrames=%u result=%s\n",
		gateActive ? 1u : 0u, mixedInteractionsObserved ? 1u : 0u,
		config.dynamicInitiallySleeping == 0 ? 1u : 0u,
		dynamicFalling ? 1u : 0u, dynamicSurfaceSeparated ? 1u : 0u,
		staticSurfaceSeparated ? 1u : 0u,
		allSurfacesSeparated ? 1u : 0u,
		static_cast<unsigned long long>(metrics.nativeIslandSteps),
		static_cast<unsigned long long>(metrics.generatedRigidContacts),
		static_cast<unsigned long long>(metrics.generatedSoftContacts),
		config.rigidContactFrames, config.softContactFrames,
		(allSurfacesSeparated && (!gateActive || mixedInteractionsObserved)) ?
			"PASS" : "FAIL");
	std::printf(
		"[AVBD_VISUAL_RIGID_GAP] "
		"initialUpperVertexSignedSdf=%.9g initialLowerVertexSignedSdf=%.9g "
		"minUpperVertexSignedSdf=%.9g minLowerVertexSignedSdf=%.9g "
		"finalUpperVertexSignedSdf=%.9g finalLowerVertexSignedSdf=%.9g "
		"maxUpperVertexPenetration=%.9g maxLowerVertexPenetration=%.9g "
		"trianglePenetrationFramesUpper=%u "
		"trianglePenetrationFramesLower=%u "
		"trianglePenetrationFirstFrameUpper=%u "
		"trianglePenetrationFirstFrameLower=%u\n",
		double(metrics.initialUpperJawSignedSdf),
		double(metrics.initialLowerJawSignedSdf),
		double(metrics.minUpperJawSignedSdf),
		double(metrics.minLowerJawSignedSdf),
		double(metrics.finalUpperJawSignedSdf),
		double(metrics.finalLowerJawSignedSdf),
		double(PxMax(-metrics.minUpperJawSignedSdf, 0.0f)),
		double(PxMax(-metrics.minLowerJawSignedSdf, 0.0f)),
		metrics.upperJawTrianglePenetrationFrames,
		metrics.lowerJawTrianglePenetrationFrames,
		metrics.upperJawTrianglePenetrationFirstFrame,
		metrics.lowerJawTrianglePenetrationFirstFrame);
	std::printf(
		"[AVBD_VISUAL_STATIC_GAP] "
		"initialUpperPedestalVertexSignedSdf=%.9g "
		"initialLowerPedestalVertexSignedSdf=%.9g "
		"minUpperPedestalVertexSignedSdf=%.9g "
		"minLowerPedestalVertexSignedSdf=%.9g "
		"finalUpperPedestalVertexSignedSdf=%.9g "
		"finalLowerPedestalVertexSignedSdf=%.9g "
		"maxUpperPedestalPenetration=%.9g "
		"maxLowerPedestalPenetration=%.9g "
		"pedestalTrianglePenetratedUpper=%u "
		"pedestalTrianglePenetratedLower=%u "
		"pedestalTrianglePenetrationFramesUpper=%u "
		"pedestalTrianglePenetrationFramesLower=%u "
		"initialUpperGroundVertexSignedSdf=%.9g "
		"initialLowerGroundVertexSignedSdf=%.9g "
		"minUpperGroundVertexSignedSdf=%.9g "
		"minLowerGroundVertexSignedSdf=%.9g "
		"finalUpperGroundVertexSignedSdf=%.9g "
		"finalLowerGroundVertexSignedSdf=%.9g "
		"maxUpperGroundPenetration=%.9g "
		"maxLowerGroundPenetration=%.9g "
		"groundPlanePenetratedUpper=%u groundPlanePenetratedLower=%u "
		"groundPlanePenetrationFramesUpper=%u "
		"groundPlanePenetrationFramesLower=%u "
		"staticSurfaceSeparated=%u\n",
		double(metrics.upperJawPedestalGap.initialSignedSdf),
		double(metrics.lowerJawPedestalGap.initialSignedSdf),
		double(metrics.upperJawPedestalGap.minSignedSdf),
		double(metrics.lowerJawPedestalGap.minSignedSdf),
		double(metrics.upperJawPedestalGap.finalSignedSdf),
		double(metrics.lowerJawPedestalGap.finalSignedSdf),
		double(PxMax(-metrics.upperJawPedestalGap.minSignedSdf, 0.0f)),
		double(PxMax(-metrics.lowerJawPedestalGap.minSignedSdf, 0.0f)),
		metrics.upperJawPedestalGap.penetrated ? 1u : 0u,
		metrics.lowerJawPedestalGap.penetrated ? 1u : 0u,
		metrics.upperJawPedestalGap.penetrationFrames,
		metrics.lowerJawPedestalGap.penetrationFrames,
		double(metrics.upperJawGroundGap.initialSignedSdf),
		double(metrics.lowerJawGroundGap.initialSignedSdf),
		double(metrics.upperJawGroundGap.minSignedSdf),
		double(metrics.lowerJawGroundGap.minSignedSdf),
		double(metrics.upperJawGroundGap.finalSignedSdf),
		double(metrics.lowerJawGroundGap.finalSignedSdf),
		double(PxMax(-metrics.upperJawGroundGap.minSignedSdf, 0.0f)),
		double(PxMax(-metrics.lowerJawGroundGap.minSignedSdf, 0.0f)),
		metrics.upperJawGroundGap.penetrated ? 1u : 0u,
		metrics.lowerJawGroundGap.penetrated ? 1u : 0u,
		metrics.upperJawGroundGap.penetrationFrames,
		metrics.lowerJawGroundGap.penetrationFrames,
		staticSurfaceSeparated ? 1u : 0u);
}

void printVisualRotationReport(
	const RotationMetrics& primaryCube, const RotationMetrics& upperJaw,
	const RotationMetrics& sphere,
	const VisualRotationReportConfig& config)
{
	const bool primaryCubeRotationPassed = isRotationResponseObserved(
		primaryCube, config.minPrimaryOrientationChange,
		config.minPrimaryAngularSpeed);
	const bool sphereRollPassed = isRotationResponseObserved(
		sphere, config.minSphereOrientationChange,
		config.minSphereAngularSpeed);

	std::printf(
		"[AVBD_VISUAL_ROTATION] body=primaryCube "
		"maxOrientationChange=%.9g "
		"maxAngularSpeed=%.9g finalAngularSpeed=%.9g "
		"maxAngularSpeedFrame=%u result=%s\n",
		double(primaryCube.maxOrientationChange),
		double(primaryCube.maxAngularSpeed),
		double(primaryCube.finalAngularSpeed),
		primaryCube.maxAngularSpeedFrame,
		primaryCubeRotationPassed ? "PASS" : "FAIL");
	std::printf(
		"[AVBD_VISUAL_PRIMARY_GROUND_EPISODES] episodes=%u "
		"firstFrame=%u secondFrame=%u active=%u "
		"finalMinY=%.9g minSecondEpisodeAngularSpeed=%.9g\n",
		primaryCube.groundContactEpisodes,
		primaryCube.firstGroundContactFrame,
		primaryCube.secondGroundContactFrame,
		primaryCube.groundContactActive ? 1u : 0u,
		double(primaryCube.finalMinCollisionY),
		double(primaryCube.minSecondGroundEpisodeAngularSpeed));
	std::printf(
		"[AVBD_VISUAL_UPPER_JAW_ROTATION] "
		"maxOrientationChange=%.9g maxAngularSpeed=%.9g "
		"finalAngularSpeed=%.9g maxAngularSpeedFrame=%u\n",
		double(upperJaw.maxOrientationChange),
		double(upperJaw.maxAngularSpeed),
		double(upperJaw.finalAngularSpeed), upperJaw.maxAngularSpeedFrame);
	std::printf(
		"[AVBD_VISUAL_SPHERE_ROLL] maxOrientationChange=%.9g "
		"maxAngularSpeed=%.9g finalAngularSpeed=%.9g "
		"earlyMaxAngularSpeed=%.9g lateMaxAngularSpeed=%.9g "
		"maxAngularSpeedFrame=%u longRunBounded=%u result=%s\n",
		double(sphere.maxOrientationChange),
		double(sphere.maxAngularSpeed), double(sphere.finalAngularSpeed),
		double(sphere.earlyMaxAngularSpeed),
		double(sphere.lateMaxAngularSpeed), sphere.maxAngularSpeedFrame,
		config.sphereLongRunBounded ? 1u : 0u,
		sphereRollPassed && config.sphereLongRunBounded ? "PASS" : "FAIL");
	std::printf(
		"[AVBD_VISUAL_SPHERE_KINEMATICS] "
		"linearVelocity=(%.9g,%.9g,%.9g) "
		"angularVelocity=(%.9g,%.9g,%.9g) "
		"lowestOffset=(%.9g,%.9g,%.9g) rigidRollSlipSpeed=%.9g "
		"groundEpisodes=%u firstGroundFrame=%u secondGroundFrame=%u "
		"groundActive=%u rollingKinematicsValid=%u\n",
		double(sphere.finalLinearVelocity.x),
		double(sphere.finalLinearVelocity.y),
		double(sphere.finalLinearVelocity.z),
		double(sphere.finalAngularVelocity.x),
		double(sphere.finalAngularVelocity.y),
		double(sphere.finalAngularVelocity.z),
		double(sphere.finalLowestCollisionOffset.x),
		double(sphere.finalLowestCollisionOffset.y),
		double(sphere.finalLowestCollisionOffset.z),
		double(sphere.finalRigidRollSlipSpeed),
		sphere.groundContactEpisodes, sphere.firstGroundContactFrame,
		sphere.secondGroundContactFrame,
		sphere.groundContactActive ? 1u : 0u,
		config.sphereRollingKinematicsValid ? 1u : 0u);
}

void printVolumeHealthReport(
	const VolumeBodyHealthMetrics* bodyMetrics,
	const char* const* bodyNames, PxU32 bodyCount,
	const VolumeHealthReportConfig& config)
{
	const bool volumeHealthPassed = isVolumeHealthWithinLimits(
		config.nonFiniteParticleSamples, config.invertedElementSamples,
		config.minDetF, config.maxDetF, config.minBodyVolumeRatio,
		config.maxBodyVolumeRatio, config.requiredMinDetF,
		config.requiredMaxDetF, config.requiredMinVolumeRatio,
		config.requiredMaxVolumeRatio);

	std::printf(
		"[AVBD_VISUAL_VOLUME_HEALTH] nonFiniteSamples=%u "
		"invertedSamples=%u minDetF=%.9g maxDetF=%.9g "
		"minBodyVolumeRatio=%.9g maxBodyVolumeRatio=%.9g "
		"result=%s\n",
		config.nonFiniteParticleSamples, config.invertedElementSamples,
		double(config.minDetF), double(config.maxDetF),
		double(config.minBodyVolumeRatio),
		double(config.maxBodyVolumeRatio),
		volumeHealthPassed ? "PASS" : "FAIL");
	for(PxU32 bodyIndex = 0; bodyIndex < bodyCount; ++bodyIndex)
	{
		const VolumeBodyHealthMetrics& metrics = bodyMetrics[bodyIndex];
		std::printf(
			"[AVBD_VISUAL_VOLUME_BODY] body=%s index=%u "
			"minDetF=%.9g minDetFFrame=%u maxDetF=%.9g "
			"maxDetFFrame=%u minVolumeRatio=%.9g "
			"minVolumeRatioFrame=%u finalMinDetF=%.9g "
			"finalMaxDetF=%.9g finalVolumeRatio=%.9g\n",
			bodyNames[bodyIndex], bodyIndex, double(metrics.minDetF),
			metrics.minDetFFrame, double(metrics.maxDetF),
			metrics.maxDetFFrame, double(metrics.minVolumeRatio),
			metrics.minVolumeRatioFrame, double(metrics.finalMinDetF),
			double(metrics.finalMaxDetF), double(metrics.finalVolumeRatio));
	}
}

static void printSphereKinematics(
	const char* marker, const RotationMetrics& metrics,
	bool rollingKinematicsValid)
{
	std::printf(
		"[%s] linearVelocity=(%.9g,%.9g,%.9g) "
		"angularVelocity=(%.9g,%.9g,%.9g) "
		"lowestOffset=(%.9g,%.9g,%.9g) rigidRollSlipSpeed=%.9g "
		"groundEpisodes=%u firstGroundFrame=%u secondGroundFrame=%u "
		"groundActive=%u rollingKinematicsValid=%u\n",
		marker,
		double(metrics.finalLinearVelocity.x),
		double(metrics.finalLinearVelocity.y),
		double(metrics.finalLinearVelocity.z),
		double(metrics.finalAngularVelocity.x),
		double(metrics.finalAngularVelocity.y),
		double(metrics.finalAngularVelocity.z),
		double(metrics.finalLowestCollisionOffset.x),
		double(metrics.finalLowestCollisionOffset.y),
		double(metrics.finalLowestCollisionOffset.z),
		double(metrics.finalRigidRollSlipSpeed),
		metrics.groundContactEpisodes, metrics.firstGroundContactFrame,
		metrics.secondGroundContactFrame,
		metrics.groundContactActive ? 1u : 0u,
		rollingKinematicsValid ? 1u : 0u);
}

void printSphereLongRollReport(
	const RotationMetrics& metrics,
	const VolumeBodyHealthMetrics* bodyMetrics, PxU32 bodyCount,
	const SoftContactPhaseMetrics& phaseMetrics,
	const SphereLongRollReportConfig& config)
{
	const PxReal windowMinimum = metrics.windowSampleCount > 0
		? metrics.windowMinAngularSpeed : 0.0f;
	const PxF64 windowMean = metrics.windowSampleCount > 0
		? metrics.windowAngularSpeedSum / PxF64(metrics.windowSampleCount)
		: 0.0;
	const PxReal windowLinearMinimum = metrics.windowSampleCount > 0
		? metrics.windowMinLinearSpeed : 0.0f;
	const PxF64 windowLinearMean = metrics.windowSampleCount > 0
		? metrics.windowLinearSpeedSum / PxF64(metrics.windowSampleCount)
		: 0.0;
	std::printf(
		"[AVBD_SPHERE_LONG_ROLL] frames=%u "
		"maxOrientationChange=%.9g maxAngularSpeed=%.9g "
		"maxAngularSpeedFrame=%u finalAngularSpeed=%.9g "
		"windowBegin=%u windowEnd=%u windowSamples=%u "
		"windowMinAngularSpeed=%.9g windowMeanAngularSpeed=%.9g "
		"windowMaxAngularSpeed=%.9g finalLinearSpeed=%.9g "
		"windowMinLinearSpeed=%.9g windowMeanLinearSpeed=%.9g "
		"windowMaxLinearSpeed=%.9g longRunBounded=%u "
		"regressionBounded=%u result=%s\n",
		config.frames,
		double(metrics.maxOrientationChange),
		double(metrics.maxAngularSpeed),
		metrics.maxAngularSpeedFrame,
		double(metrics.finalAngularSpeed),
		config.windowBeginFrame, config.windowEndFrame,
		metrics.windowSampleCount,
		double(windowMinimum), double(windowMean),
		double(metrics.windowMaxAngularSpeed),
		double(metrics.finalLinearSpeed),
		double(windowLinearMinimum), double(windowLinearMean),
		double(metrics.windowMaxLinearSpeed),
		config.longRunBounded ? 1u : 0u,
		config.regressionBounded ? 1u : 0u,
		config.passed ? "PASS" : "FAIL");
	printSphereKinematics(
		"AVBD_SPHERE_LONG_ROLL_KINEMATICS", metrics,
		config.rollingKinematicsValid);
	for(PxU32 bodyIndex = 0; bodyIndex < bodyCount; ++bodyIndex)
	{
		const VolumeBodyHealthMetrics& health = bodyMetrics[bodyIndex];
		std::printf(
			"[AVBD_SPHERE_LONG_ROLL_BODY_HEALTH] body=%s index=%u "
			"minDetF=%.9g maxDetF=%.9g minVolumeRatio=%.9g "
			"maxVolumeRatio=%.9g finalMinDetF=%.9g "
			"finalMaxDetF=%.9g finalVolumeRatio=%.9g\n",
			bodyIndex == 0u ? "impactor" : "sphere", bodyIndex,
			double(health.minDetF), double(health.maxDetF),
			double(health.minVolumeRatio),
			double(health.maxVolumeRatio),
			double(health.finalMinDetF),
			double(health.finalMaxDetF),
			double(health.finalVolumeRatio));
	}
	for(PxU32 checkpointIndex = 0;
		checkpointIndex < config.checkpointCount; ++checkpointIndex)
	{
		const PxU32 checkpointFrame =
			(checkpointIndex + 1) * config.checkpointInterval;
		if(checkpointFrame > config.completedFrames)
			break;
		std::printf(
			"[AVBD_SPHERE_LONG_ROLL_CHECKPOINT] frame=%u "
			"angularSpeed=%.9g linearSpeed=%.9g centroidY=%.9g\n",
			checkpointFrame,
			double(metrics.checkpointAngularSpeeds[checkpointIndex]),
			double(metrics.checkpointLinearSpeeds[checkpointIndex]),
			double(metrics.checkpointCentroidY[checkpointIndex]));
	}
	const PxReal deltaAngularMomentum = PxMax(0.0f,
		phaseMetrics.peakSoftContactAngularMomentum -
			phaseMetrics.preSoftAngularMomentum);
	const PxReal deltaAngularSpeed = PxMax(0.0f,
		phaseMetrics.peakSoftContactAngularSpeed -
			phaseMetrics.preSoftAngularSpeed);
	std::printf(
		"[AVBD_SPHERE_LONG_ROLL_CONTACT_PHASE] initialized=%u "
		"contactTelemetry=%s preSoftContactSample=%u "
		"firstSoftContactFrame=%u lastSoftContactFrame=%u "
		"peakSoftContactFrame=%u softContactFrames=%u "
		"generatedGroundContacts=%llu generatedSoftContacts=%llu "
		"preSoftAngularMomentum=%.9g preSoftAngularSpeed=%.9g "
		"peakSoftContactAngularMomentum=%.9g "
		"peakSoftContactAngularSpeed=%.9g "
		"lastSoftContactAngularMomentum=%.9g "
		"lastSoftContactAngularSpeed=%.9g "
		"finalPostSoftAngularMomentum=%.9g "
		"finalPostSoftAngularSpeed=%.9g "
		"deltaAngularMomentum=%.9g deltaAngularSpeed=%.9g\n",
		phaseMetrics.initialized ? 1u : 0u,
		phaseMetrics.contactTelemetryEnabled ? "enabled" : "disabled",
		phaseMetrics.hasPreSoftContactSample ? 1u : 0u,
		phaseMetrics.firstSoftContactFrame,
		phaseMetrics.lastSoftContactFrame,
		phaseMetrics.peakSoftContactFrame,
		phaseMetrics.softContactFrames,
		static_cast<unsigned long long>(
			phaseMetrics.generatedGroundContacts),
		static_cast<unsigned long long>(
			phaseMetrics.generatedSoftContacts),
		double(phaseMetrics.preSoftAngularMomentum),
		double(phaseMetrics.preSoftAngularSpeed),
		double(phaseMetrics.peakSoftContactAngularMomentum),
		double(phaseMetrics.peakSoftContactAngularSpeed),
		double(phaseMetrics.lastSoftContactAngularMomentum),
		double(phaseMetrics.lastSoftContactAngularSpeed),
		double(phaseMetrics.finalPostSoftAngularMomentum),
		double(phaseMetrics.finalPostSoftAngularSpeed),
		double(deltaAngularMomentum), double(deltaAngularSpeed));
}

void printSoftSoftGlancingReport(
	const RotationMetrics& metrics,
	const SoftContactPhaseMetrics& phaseMetrics,
	const SoftSoftGlancingReportConfig& config)
{
	const PxReal deltaAngularMomentum = PxMax(0.0f,
		phaseMetrics.peakSoftContactAngularMomentum -
			phaseMetrics.preSoftAngularMomentum);
	const PxReal deltaAngularSpeed = PxMax(0.0f,
		phaseMetrics.peakSoftContactAngularSpeed -
			phaseMetrics.preSoftAngularSpeed);
	const PxVec3 deltaAngularVelocity =
		phaseMetrics.peakSoftContactAngularVelocity -
			phaseMetrics.preSoftAngularVelocity;
	std::printf(
		"[AVBD_SPHERE_SOFT_SOFT_GLANCING] frames=%u "
		"contactTelemetry=%s preSoftContactSample=%u "
		"firstSoftContactFrame=%u "
		"lastSoftContactFrame=%u peakSoftContactFrame=%u "
		"softContactFrames=%u generatedGroundContacts=%llu "
		"generatedSoftContacts=%llu preSoftAngularMomentum=%.9g "
		"preSoftAngularSpeed=%.9g preSoftAngularVelocityX=%.9g "
		"preSoftAngularVelocityY=%.9g preSoftAngularVelocityZ=%.9g "
		"peakSoftContactAngularMomentum=%.9g "
		"peakSoftContactAngularSpeed=%.9g "
		"peakSoftContactAngularVelocityX=%.9g "
		"peakSoftContactAngularVelocityY=%.9g "
		"peakSoftContactAngularVelocityZ=%.9g "
		"deltaAngularMomentum=%.9g deltaAngularSpeed=%.9g "
		"deltaAngularVelocityX=%.9g deltaAngularVelocityY=%.9g "
		"deltaAngularVelocityZ=%.9g maxOrientationChange=%.9g "
		"maxAngularSpeed=%.9g maxAngularSpeedFrame=%u result=%s\n",
		config.frames,
		phaseMetrics.contactTelemetryEnabled ? "enabled" : "disabled",
		phaseMetrics.hasPreSoftContactSample ? 1u : 0u,
		phaseMetrics.firstSoftContactFrame,
		phaseMetrics.lastSoftContactFrame,
		phaseMetrics.peakSoftContactFrame,
		phaseMetrics.softContactFrames,
		static_cast<unsigned long long>(
			phaseMetrics.generatedGroundContacts),
		static_cast<unsigned long long>(
			phaseMetrics.generatedSoftContacts),
		double(phaseMetrics.preSoftAngularMomentum),
		double(phaseMetrics.preSoftAngularSpeed),
		double(phaseMetrics.preSoftAngularVelocity.x),
		double(phaseMetrics.preSoftAngularVelocity.y),
		double(phaseMetrics.preSoftAngularVelocity.z),
		double(phaseMetrics.peakSoftContactAngularMomentum),
		double(phaseMetrics.peakSoftContactAngularSpeed),
		double(phaseMetrics.peakSoftContactAngularVelocity.x),
		double(phaseMetrics.peakSoftContactAngularVelocity.y),
		double(phaseMetrics.peakSoftContactAngularVelocity.z),
		double(deltaAngularMomentum), double(deltaAngularSpeed),
		double(deltaAngularVelocity.x), double(deltaAngularVelocity.y),
		double(deltaAngularVelocity.z),
		double(metrics.maxOrientationChange),
		double(metrics.maxAngularSpeed), metrics.maxAngularSpeedFrame,
		config.passed ? "PASS" : "FAIL");
}

void printSoftSoftTorqueReport(
	const SoftSoftTorqueMetrics& metrics,
	const SoftSoftTorqueReportConfig& config)
{
	const bool retentionPassed =
		metrics.firstRotationFrame != PX_MAX_U32 &&
		metrics.retainedRotationSamples >= config.minRetentionSamples;
	std::printf(
		"[AVBD_SOFT_SOFT_TORQUE] frames=%u isolated=%u "
		"targetDistinctCollisionSimulation=%u "
		"driverDistinctCollisionSimulation=%u "
		"targetSimulationVertices=%u targetCollisionVertices=%u "
		"driverSimulationVertices=%u driverCollisionVertices=%u "
		"supportEvidence=embeddedCollisionSimulationTopology "
		"supportExpansionInstrumentation=%s "
		"softContactFrames=%u generatedSoftContacts=%llu "
		"generatedGroundContacts=%llu generatedRigidContacts=%llu "
		"generatedSelfContacts=%llu firstContactFrame=%u "
		"firstRotationFrame=%u firstContactCentroidLeverArm=%.9g "
		"maxCentroidLeverArm=%.9g targetMaxAngularMomentum=%.9g "
		"targetFinalAngularMomentum=%.9g targetMaxAngularSpeed=%.9g "
		"targetFinalAngularSpeed=%.9g retainedRotationSamples=%u "
		"retentionPassed=%u result=%s\n",
		config.frames, metrics.isolatedConfiguration,
		metrics.targetDistinctCollisionSimulation,
		metrics.driverDistinctCollisionSimulation,
		metrics.targetSimulationVertices,
		metrics.targetCollisionVertices,
		metrics.driverSimulationVertices,
		metrics.driverCollisionVertices,
		metrics.supportExpansionInstrumentationAvailable
			? "available" : "unavailable",
		metrics.softContactFrames,
		static_cast<unsigned long long>(metrics.generatedSoftContacts),
		static_cast<unsigned long long>(metrics.generatedGroundContacts),
		static_cast<unsigned long long>(metrics.generatedRigidContacts),
		static_cast<unsigned long long>(metrics.generatedSelfContacts),
		metrics.firstContactFrame, metrics.firstRotationFrame,
		double(metrics.firstContactCentroidLeverArm),
		double(metrics.maxCentroidLeverArm),
		double(metrics.maxAngularMomentum),
		double(metrics.finalAngularMomentum),
		double(metrics.maxAngularSpeed),
		double(metrics.finalAngularSpeed),
		metrics.retainedRotationSamples, retentionPassed ? 1u : 0u,
		config.passed ? "PASS" : "FAIL");
}

void printGroundEmbeddedTetReport(
	const GroundEmbeddedTetProbeMetrics& metrics,
	const GroundEmbeddedTetReportConfig& config)
{
	const PxReal preGroundAngularMomentum =
		metrics.preGroundAngularMomentum.magnitude();
	const PxReal preGroundAngularSpeed =
		metrics.preGroundAngularVelocity.magnitude();
	const bool volumeHealthPassed = isVolumeHealthWithinLimits(
		config.nonFiniteParticleSamples, config.invertedElementSamples,
		config.minDetF, config.maxDetF, config.minBodyVolumeRatio,
		config.maxBodyVolumeRatio, config.requiredMinDetF,
		config.requiredMaxDetF, config.requiredMinVolumeRatio,
		config.requiredMaxVolumeRatio);
	std::printf(
		"[AVBD_GROUND_EMBEDDED_TET_PROBE] frames=%u "
		"simVertices=%u simTetrahedra=%u "
		"collisionVertices=%u collisionTetrahedra=%u "
		"distinctCollisionSimulation=%u strictInteriorEmbedding=%u "
		"selfCollisionDisabled=%u speculativeCcdDisabled=%u "
		"contactTelemetryEnabled=%u preGroundSample=%u "
		"firstGroundContactFrame=%u lastGroundContactFrame=%u "
		"peakGroundRollFrame=%u groundContactWindowFrames=%u "
		"generatedGroundContacts=%llu generatedRigidContacts=%llu "
		"generatedSoftContacts=%llu generatedSelfContacts=%llu "
		"launchSpeed=%.9g initialMass=%.9g initialRmsRadius=%.9g "
		"preGroundAngularMomentum=%.9g preGroundAngularSpeed=%.9g "
		"peakDeltaAngularMomentumX=%.9g "
		"peakDeltaAngularMomentumY=%.9g "
		"peakDeltaAngularMomentumZ=%.9g "
		"peakDeltaAngularVelocityX=%.9g "
		"peakDeltaAngularVelocityY=%.9g "
		"peakDeltaAngularVelocityZ=%.9g "
		"peakExpectedRollAngularMomentum=%.9g "
		"peakExpectedRollAngularSpeed=%.9g "
		"peakNormalizedRollMomentum=%.9g "
		"peakNormalizedRollOmega=%.9g "
		"groundContactFrames=%u maxGroundContacts=%u "
		"groundTetPatchGroundRows=%llu "
		"groundTetPatchFourSupportRows=%llu "
		"groundTetPatchSingleTetRows=%llu "
		"groundTetPatchActiveRows=%llu "
		"velocityTangentOwnerRows=%llu "
		"velocityTangentAppliedRows=%llu "
		"minDetF=%.9g maxDetF=%.9g "
		"minBodyVolumeRatio=%.9g maxBodyVolumeRatio=%.9g "
		"health=%s result=%s\n",
		config.frames, metrics.simulationVertices,
		metrics.simulationTetrahedra,
		metrics.collisionVertices,
		metrics.collisionTetrahedra,
		metrics.distinctCollisionSimulation,
		metrics.strictInteriorEmbedding,
		metrics.selfCollisionDisabled,
		metrics.speculativeCcdDisabled,
		metrics.contactTelemetryEnabled,
		metrics.hasPreGroundSample,
		metrics.firstGroundContactFrame,
		metrics.lastGroundContactFrame,
		metrics.peakGroundRollFrame,
		metrics.groundContactWindowFrames,
		static_cast<unsigned long long>(metrics.generatedGroundContacts),
		static_cast<unsigned long long>(metrics.generatedRigidContacts),
		static_cast<unsigned long long>(metrics.generatedSoftContacts),
		static_cast<unsigned long long>(metrics.generatedSelfContacts),
		double(metrics.launchSpeed), double(metrics.initialMass),
		double(metrics.initialRmsRadius),
		double(preGroundAngularMomentum), double(preGroundAngularSpeed),
		double(metrics.peakDeltaAngularMomentum.x),
		double(metrics.peakDeltaAngularMomentum.y),
		double(metrics.peakDeltaAngularMomentum.z),
		double(metrics.peakDeltaAngularVelocity.x),
		double(metrics.peakDeltaAngularVelocity.y),
		double(metrics.peakDeltaAngularVelocity.z),
		double(metrics.peakExpectedRollAngularMomentum),
		double(metrics.peakExpectedRollAngularSpeed),
		double(metrics.peakNormalizedRollMomentum),
		double(metrics.peakNormalizedRollOmega),
		config.groundContactFrames, config.maxGroundContacts,
		static_cast<unsigned long long>(config.groundPositionAlRows),
		static_cast<unsigned long long>(config.fourSupportRows),
		static_cast<unsigned long long>(config.singleTetRows),
		static_cast<unsigned long long>(config.activeRows),
		static_cast<unsigned long long>(config.velocityTangentOwnerRows),
		static_cast<unsigned long long>(config.velocityTangentAppliedRows),
		double(config.minDetF), double(config.maxDetF),
		double(config.minBodyVolumeRatio),
		double(config.maxBodyVolumeRatio),
		volumeHealthPassed ? "PASS" : "FAIL",
		config.passed ? "PASS" : "FAIL");
}

void printVolumeSkinningReport(
	const VolumeSkinningMetrics& metrics, bool passed)
{
	std::printf(
		"[AVBD_CPU_SKINNING] schema=1 snippet=SnippetDeformableVolumeAVBD "
		"kind=volume vertices=%u triangles=%u "
		"evaluatedFrames=%u finiteFrames=%u "
		"maxDisplacement=%.9g status=%s\n",
		metrics.vertices, metrics.triangles,
		metrics.evaluatedFrames, metrics.finiteFrames,
		double(metrics.maxDisplacement), passed ? "PASS" : "FAIL");
}

void printRigidTriangleSteadyContactReport(
	const RigidTriangleSteadyContactReportConfig& config)
{
	std::printf(
		"[AVBD_RIGID_TRIANGLE_STEADY_CONTACT] frames=%u "
		"profileFrames=%u faceTests=%llu edgeTests=%llu "
		"vertexTests=%llu result=%s\n",
		config.frames, config.profileFrames,
		static_cast<unsigned long long>(config.faceTests),
		static_cast<unsigned long long>(config.edgeTests),
		static_cast<unsigned long long>(config.vertexTests),
		config.passed ? "PASS" : "FAIL");
}

void printReverseFeatureReport(
	const char* marker, PxU32 frames,
	const ReverseFeatureMetrics& metrics, bool passed)
{
	std::printf(
		"[%s] frames=%u "
		"faceResponseObserved=%u vertexSdfExcluded=%u "
		"negativeControlPassed=%u nonFiniteSamples=%u "
		"positiveDisplacement=%.9g positiveDrop=%.9g "
		"negativeDrop=%.9g faceSeparation=%.9g "
		"minimumVertexSeparation=%.9g result=%s\n",
		marker, frames, metrics.faceResponseObserved,
		metrics.vertexSdfExcluded, metrics.negativeControlPassed,
		metrics.nonFiniteSamples, double(metrics.positiveDisplacement),
		double(metrics.positiveDrop), double(metrics.negativeDrop),
		double(metrics.faceSeparation),
		double(metrics.minimumVertexSeparation),
		passed ? "PASS" : "FAIL");
}

void printTriangleSurfaceSweptReport(
	const char* marker, PxU32 frames, const char* target,
	const char* geometry, const ReverseSweptMetrics& metrics, bool passed)
{
	std::printf(
		"[%s] frames=%u target=%s geometry=%s "
		"responseObserved=%u negativeControlPassed=%u "
		"vertexSweepExcluded=%u nonFiniteSamples=%u "
		"positiveDisplacement=%.9g negativeDisplacement=%.9g "
		"positiveDrop=%.9g negativeDrop=%.9g "
		"minimumVertexSweepSeparation=%.9g result=%s\n",
		marker, frames, target, geometry, metrics.responseObserved,
		metrics.negativeControlPassed, metrics.vertexSweepExcluded,
		metrics.nonFiniteSamples, double(metrics.positiveDisplacement),
		double(metrics.negativeDisplacement), double(metrics.positiveDrop),
		double(metrics.negativeDrop),
		double(metrics.minimumVertexSweepSeparation),
		passed ? "PASS" : "FAIL");
}

void printTriangleSurfaceRotationalSweptReport(
	PxU32 frames, const char* geometry, const char* owner,
	const ReverseSweptMetrics& metrics,
	const RotationalSweepMetrics& rotational, bool passed)
{
	std::printf(
		"[AVBD_TRIANGLE_SURFACE_ROTATIONAL_SWEPT] "
		"frames=%u target=kinematic geometry=%s owner=%s "
		"responseObserved=%u negativeControlPassed=%u "
		"vertexSweepExcluded=%u endpointMinSeparation=%.9g "
		"midSweepMinSeparation=%.9g "
		"minimumVertexSweepSeparation=%.9g "
		"positiveDisplacement=%.9g negativeDisplacement=%.9g "
		"positiveAngularTravel=%.9g negativeAngularTravel=%.9g "
		"result=%s\n",
		frames, geometry, owner, metrics.responseObserved,
		metrics.negativeControlPassed, metrics.vertexSweepExcluded,
		double(rotational.endpointMinSeparation),
		double(rotational.midSweepMinSeparation),
		double(metrics.minimumVertexSweepSeparation),
		double(metrics.positiveDisplacement),
		double(metrics.negativeDisplacement),
		double(rotational.positiveAngularTravel),
		double(rotational.negativeAngularTravel),
		passed ? "PASS" : "FAIL");
}

void printReverseSweptReport(
	const char* marker, PxU32 frames, const char* target,
	const ReverseSweptMetrics& metrics, bool passed)
{
	std::printf(
		"[%s] frames=%u target=%s "
		"responseObserved=%u negativeControlPassed=%u "
		"twoSidedResponseObserved=%u vertexSweepExcluded=%u "
		"nonFiniteSamples=%u positiveDisplacement=%.9g "
		"negativeDisplacement=%.9g positiveDrop=%.9g "
		"negativeDrop=%.9g positiveRigidDrop=%.9g "
		"negativeRigidDrop=%.9g faceSeparation=%.9g "
		"minimumVertexSweepSeparation=%.9g result=%s\n",
		marker, frames, target, metrics.responseObserved,
		metrics.negativeControlPassed, metrics.twoSidedResponseObserved,
		metrics.vertexSweepExcluded, metrics.nonFiniteSamples,
		double(metrics.positiveDisplacement),
		double(metrics.negativeDisplacement), double(metrics.positiveDrop),
		double(metrics.negativeDrop), double(metrics.positiveRigidDrop),
		double(metrics.negativeRigidDrop), double(metrics.faceSeparation),
		double(metrics.minimumVertexSweepSeparation),
		passed ? "PASS" : "FAIL");
}

void printDeformingReverseSweptReport(
	PxU32 frames, const char* geometry,
	const ReverseSweptMetrics& metrics,
	const DeformingReverseSweptMetrics& deforming, bool passed)
{
	std::printf(
		"[AVBD_DEFORMING_VOLUME_REVERSE_SWEPT] "
		"frames=%u geometry=%s target=static owner=reverse "
		"responseObserved=%u negativeControlPassed=%u "
		"geometricSweepIsolated=%u vertexSweepExcluded=%u "
		"nonFiniteSamples=%u endpointMinSeparation=%.9g "
		"midSweepMinSeparation=%.9g "
		"minimumVertexSweepSeparation=%.9g responseDelta=%.9g "
		"positiveDrop=%.9g negativeDrop=%.9g result=%s\n",
		frames, geometry, metrics.responseObserved,
		metrics.negativeControlPassed, deforming.geometricSweepIsolated,
		metrics.vertexSweepExcluded, metrics.nonFiniteSamples,
		double(deforming.endpointMinSeparation),
		double(deforming.midSweepMinSeparation),
		double(metrics.minimumVertexSweepSeparation),
		double(deforming.responseDelta), double(metrics.positiveDrop),
		double(metrics.negativeDrop), passed ? "PASS" : "FAIL");
}

void printRotationalReverseSweptReport(
	const char* marker, PxU32 frames, const char* target,
	const ReverseSweptMetrics& metrics,
	const RotationalSweepMetrics& rotational, bool passed)
{
	std::printf(
		"[%s] frames=%u target=%s owner=reverse "
		"responseObserved=%u negativeControlPassed=%u "
		"twoSidedResponseObserved=%u vertexSweepExcluded=%u "
		"endpointMinSeparation=%.9g midSweepMinSeparation=%.9g "
		"positiveDisplacement=%.9g negativeDisplacement=%.9g "
		"positiveAngularTravel=%.9g negativeAngularTravel=%.9g "
		"result=%s\n",
		marker, frames, target, metrics.responseObserved,
		metrics.negativeControlPassed, metrics.twoSidedResponseObserved,
		metrics.vertexSweepExcluded,
		double(rotational.endpointMinSeparation),
		double(rotational.midSweepMinSeparation),
		double(metrics.positiveDisplacement),
		double(metrics.negativeDisplacement),
		double(rotational.positiveAngularTravel),
		double(rotational.negativeAngularTravel),
		passed ? "PASS" : "FAIL");
}

void printKinematicRotationalSweptReport(
	const char* marker, PxU32 frames,
	const KinematicFiniteSweptMetrics& metrics,
	const RotationalSweepMetrics& rotational, bool passed)
{
	std::printf(
		"[%s] frames=%u target=kinematic owner=forward "
		"responseObserved=%u negativeControlPassed=%u "
		"endpointMinSeparation=%.9g midSweepMinSeparation=%.9g "
		"positiveDisplacement=%.9g negativeDisplacement=%.9g "
		"result=%s\n",
		marker, frames, metrics.responseObserved,
		metrics.negativeControlPassed,
		double(rotational.endpointMinSeparation),
		double(rotational.midSweepMinSeparation),
		double(metrics.positiveDisplacement),
		double(metrics.negativeDisplacement),
		passed ? "PASS" : "FAIL");
}

void printDynamicRotationalSweptReport(
	const char* marker, PxU32 frames,
	const DynamicFiniteSweptMetrics& metrics,
	const RotationalSweepMetrics& rotational, bool passed)
{
	std::printf(
		"[%s] frames=%u target=dynamic owner=forward "
		"responseObserved=%u negativeControlPassed=%u "
		"twoSidedResponseObserved=%u endpointMinSeparation=%.9g "
		"midSweepMinSeparation=%.9g positiveDisplacement=%.9g "
		"negativeDisplacement=%.9g positiveAngularTravel=%.9g "
		"negativeAngularTravel=%.9g result=%s\n",
		marker, frames, metrics.responseObserved,
		metrics.negativeControlPassed,
		metrics.twoSidedResponseObserved,
		double(rotational.endpointMinSeparation),
		double(rotational.midSweepMinSeparation),
		double(metrics.positiveSoftDisplacement),
		double(metrics.negativeSoftDisplacement),
		double(rotational.positiveAngularTravel),
		double(rotational.negativeAngularTravel),
		passed ? "PASS" : "FAIL");
}

} // namespace SnippetDeformableVolumeAVBDReport
