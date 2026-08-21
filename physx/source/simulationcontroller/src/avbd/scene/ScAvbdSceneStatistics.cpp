// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

		void AvbdCpuSoftScene::writeAvbdCpuSoftBodyStatistics(
			PxSimulationStatistics& stats) const
		{
			stats.avbdCpuSoftBodyComponentFallbackSteps =
				mLastComponentFallbackSteps;
			stats.avbdCpuSoftBodyNativeIslandSteps =
				mLastNativeIslandSteps;
			stats.avbdCpuTaskGraphRequestedDispatcherWorkers =
				mStandaloneTaskGraphTelemetry.requestedDispatcherWorkers.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedSolveTasks =
				mStandaloneTaskGraphTelemetry.submittedSolveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedSolveTasks =
				mStandaloneTaskGraphTelemetry.completedSolveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActiveSolveTasks =
				mStandaloneTaskGraphTelemetry.peakActiveSolveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphBarrierTasks =
				mStandaloneTaskGraphTelemetry.causalLayerFanIns.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialSolveTasks =
				mStandaloneTaskGraphTelemetry.serialSolveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPureSoftEligibleIslands =
				mStandaloneTaskGraphTelemetry.pureSoftEligibleIslands.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPureSoftEligibleParticles =
				mStandaloneTaskGraphTelemetry.pureSoftEligibleParticles.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedPredictionTasks =
				mStandaloneTaskGraphTelemetry.prediction.submittedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedPredictionTasks =
				mStandaloneTaskGraphTelemetry.prediction.completedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActivePredictionTasks =
				mStandaloneTaskGraphTelemetry.prediction.peakActiveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialPredictionStages =
				mStandaloneTaskGraphTelemetry.prediction.serialFallbacks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedWriteBackTasks =
				mStandaloneTaskGraphTelemetry.writeBack.submittedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedWriteBackTasks =
				mStandaloneTaskGraphTelemetry.writeBack.completedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActiveWriteBackTasks =
				mStandaloneTaskGraphTelemetry.writeBack.peakActiveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialWriteBackStages =
				mStandaloneTaskGraphTelemetry.writeBack.serialFallbacks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedCausalLayerTasks =
				mStandaloneTaskGraphTelemetry.submittedCausalLayerTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedCausalLayerTasks =
				mStandaloneTaskGraphTelemetry.completedCausalLayerTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActiveCausalLayerTasks =
				mStandaloneTaskGraphTelemetry.peakActiveCausalLayerTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCausalLayerFanIns =
				mStandaloneTaskGraphTelemetry.causalLayerFanIns.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialCausalLayerFallbacks =
				mStandaloneTaskGraphTelemetry.serialCausalLayerFallbacks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphMaxCausalLayerOccupancy =
				mStandaloneTaskGraphTelemetry.maxCausalLayerOccupancy.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphTotalCausalLayerOccupancy =
				mStandaloneTaskGraphTelemetry.totalCausalLayerOccupancy.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedWorldPlaneContactTasks =
				mStandaloneTaskGraphTelemetry.worldPlaneContact.submittedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedWorldPlaneContactTasks =
				mStandaloneTaskGraphTelemetry.worldPlaneContact.completedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActiveWorldPlaneContactTasks =
				mStandaloneTaskGraphTelemetry.worldPlaneContact.peakActiveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphWorldPlaneContactFanIns =
				mStandaloneTaskGraphTelemetry.worldPlaneContact.fanIns.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialWorldPlaneContactFallbacks =
				mStandaloneTaskGraphTelemetry.worldPlaneContact.serialFallbacks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedRigidBoxSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidBoxSdfContact.submittedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedRigidBoxSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidBoxSdfContact.completedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActiveRigidBoxSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidBoxSdfContact.peakActiveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphRigidBoxSdfContactFanIns =
				mStandaloneTaskGraphTelemetry.rigidBoxSdfContact.fanIns.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialRigidBoxSdfContactFallbacks =
				mStandaloneTaskGraphTelemetry.rigidBoxSdfContact.serialFallbacks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedRigidSphereSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidSphereSdfContact.submittedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedRigidSphereSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidSphereSdfContact.completedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActiveRigidSphereSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidSphereSdfContact.peakActiveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphRigidSphereSdfContactFanIns =
				mStandaloneTaskGraphTelemetry.rigidSphereSdfContact.fanIns.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialRigidSphereSdfContactFallbacks =
				mStandaloneTaskGraphTelemetry.rigidSphereSdfContact.serialFallbacks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedRigidCapsuleSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidCapsuleSdfContact.submittedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedRigidCapsuleSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidCapsuleSdfContact.completedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActiveRigidCapsuleSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidCapsuleSdfContact.peakActiveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphRigidCapsuleSdfContactFanIns =
				mStandaloneTaskGraphTelemetry.rigidCapsuleSdfContact.fanIns.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialRigidCapsuleSdfContactFallbacks =
				mStandaloneTaskGraphTelemetry.rigidCapsuleSdfContact.serialFallbacks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedRigidConvexSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidConvexSdfContact.submittedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedRigidConvexSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidConvexSdfContact.completedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActiveRigidConvexSdfContactTasks =
				mStandaloneTaskGraphTelemetry.rigidConvexSdfContact.peakActiveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphRigidConvexSdfContactFanIns =
				mStandaloneTaskGraphTelemetry.rigidConvexSdfContact.fanIns.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialRigidConvexSdfContactFallbacks =
				mStandaloneTaskGraphTelemetry.rigidConvexSdfContact.serialFallbacks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedRigidTriangleSurfaceContactTasks =
				mStandaloneTaskGraphTelemetry.rigidTriangleSurfaceContact.submittedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedRigidTriangleSurfaceContactTasks =
				mStandaloneTaskGraphTelemetry.rigidTriangleSurfaceContact.completedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActiveRigidTriangleSurfaceContactTasks =
				mStandaloneTaskGraphTelemetry.rigidTriangleSurfaceContact.peakActiveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphRigidTriangleSurfaceContactFanIns =
				mStandaloneTaskGraphTelemetry.rigidTriangleSurfaceContact.fanIns.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialRigidTriangleSurfaceContactFallbacks =
				mStandaloneTaskGraphTelemetry.rigidTriangleSurfaceContact.serialFallbacks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSubmittedSelfBvhContactTasks =
				mStandaloneTaskGraphTelemetry.selfBvhContact.submittedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphCompletedSelfBvhContactTasks =
				mStandaloneTaskGraphTelemetry.selfBvhContact.completedTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphPeakActiveSelfBvhContactTasks =
				mStandaloneTaskGraphTelemetry.selfBvhContact.peakActiveTasks.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSelfBvhContactFanIns =
				mStandaloneTaskGraphTelemetry.selfBvhContact.fanIns.load(
					std::memory_order_relaxed);
			stats.avbdCpuTaskGraphSerialSelfBvhContactFallbacks =
				mStandaloneTaskGraphTelemetry.selfBvhContact.serialFallbacks.load(
					std::memory_order_relaxed);
			stats.avbdCpuSoftBodyWorkspaceGrowthEvents =
				mLastStepStats.workspaceGrowthEvents;
			stats.avbdCpuSoftBodyWorkspaceGrowthBytes =
				mLastStepStats.workspaceGrowthBytes;
			stats.avbdCpuSoftBodyContactWorkspaceGrowthEvents =
				mLastStepStats.contactWorkspaceGrowthEvents;
			stats.avbdCpuSoftBodyContactWorkspaceGrowthBytes =
				mLastStepStats.contactWorkspaceGrowthBytes;
			stats.avbdCpuSoftBodyContactSweepScratchGrowthEvents =
				mLastStepStats.contactSweepScratchGrowthEvents;
			stats.avbdCpuSoftBodyContactSweepScratchGrowthBytes =
				mLastStepStats.contactSweepScratchGrowthBytes;
			stats.avbdCpuSoftBodyContactOutputGrowthEvents =
				mLastStepStats.contactOutputGrowthEvents;
			stats.avbdCpuSoftBodyContactOutputGrowthBytes =
				mLastStepStats.contactOutputGrowthBytes;
			stats.avbdCpuSoftBodyPeakContactOutputCount =
				mLastStepStats.peakContactOutputCount;
			stats.avbdCpuSoftBodyPeakContactOutputCapacity =
				mLastStepStats.peakContactOutputCapacity;
			stats.avbdCpuSoftBodyPeakContactIncidenceCount =
				mLastStepStats.peakContactIncidenceCount;
			stats.avbdCpuSoftBodyPeakContactIncidenceCapacity =
				mLastStepStats.peakContactIncidenceCapacity;
			stats.avbdCpuSoftBodyPeakStateTransferContactCount =
				mLastStepStats.peakStateTransferContactCount;
			stats.avbdCpuSoftBodyPeakStateTransferContactCapacity =
				mLastStepStats.peakStateTransferContactCapacity;
			stats.avbdCpuSoftBodyPeakStateTransferUsedCapacity =
				mLastStepStats.peakStateTransferUsedCapacity;
			stats.avbdCpuSoftBodyParticlePrimalColorCount =
				mLastStepStats.particlePrimalColorCount;
			stats.avbdCpuSoftBodyParticlePrimalDynamicAccessGroupCount =
				mLastStepStats.particlePrimalDynamicAccessGroupCount;
			stats.avbdCpuSoftBodyParticlePrimalColoredSerialSweeps =
				mLastStepStats.particlePrimalColoredSerialSweeps;
			stats.avbdCpuSoftBodyParticlePrimalColoredSerialFallbackSweeps =
				mLastStepStats.particlePrimalColoredSerialFallbackSweeps;
			stats.avbdCpuSoftBodyGroundTetPatchGroundPositionAlRows =
				mLastStepStats.groundTetPatchGroundPositionAlRows;
			stats.avbdCpuSoftBodyGroundTetPatchFourSupportRows =
				mLastStepStats.groundTetPatchFourSupportRows;
			stats.avbdCpuSoftBodyGroundTetPatchSingleTetRows =
				mLastStepStats.groundTetPatchSingleTetRows;
			stats.avbdCpuSoftBodyGroundTetPatchActiveRows =
				mLastStepStats.groundTetPatchActiveRows;
			stats.avbdCpuSoftBodyWorldStaticVelocityTangentOwnerRows =
				mLastStepStats.worldStaticVelocityTangentOwnerRows;
			stats.avbdCpuSoftBodyWorldStaticVelocityTangentAppliedRows =
				mLastStepStats.worldStaticVelocityTangentAppliedRows;
			stats.avbdCpuSoftBodyParticlePrimalCensusDynamicParticleSolves =
				mLastStepStats.particlePrimalCensusDynamicParticleSolves;
			stats.avbdCpuSoftBodyParticlePrimalCensusTriangleEvaluations =
				mLastStepStats.particlePrimalCensusTriangleEvaluations;
			stats.avbdCpuSoftBodyParticlePrimalCensusCorotationalTetEvaluations =
				mLastStepStats.
					particlePrimalCensusCorotationalTetEvaluations;
			stats.avbdCpuSoftBodyParticlePrimalCensusNeoHookeanTetEvaluations =
				mLastStepStats.particlePrimalCensusNeoHookeanTetEvaluations;
			stats.avbdCpuSoftBodyParticlePrimalCensusBendingEvaluations =
				mLastStepStats.particlePrimalCensusBendingEvaluations;
			stats.avbdCpuSoftBodyParticlePrimalCensusContactEvaluations =
				mLastStepStats.particlePrimalCensusContactEvaluations;
			stats.avbdCpuSoftBodyParticlePrimalCensusTetPacket8FullPackets =
				mLastStepStats.particlePrimalCensusTetPacket8FullPackets;
			stats.avbdCpuSoftBodyParticlePrimalCensusTetPacket8TailLanes =
				mLastStepStats.particlePrimalCensusTetPacket8TailLanes;
			stats.avbdCpuSoftBodyParticlePrimalTetPacketIrBodies =
				mLastStepStats.particlePrimalTetPacketIrBodies;
			stats.avbdCpuSoftBodyParticlePrimalTetPacketIrPackets =
				mLastStepStats.particlePrimalTetPacketIrPackets;
			stats.avbdCpuSoftBodyParticlePrimalTetPacketIrActiveLanes =
				mLastStepStats.particlePrimalTetPacketIrActiveLanes;
			stats.avbdCpuSoftBodyParticlePrimalTetPacketIrTailLanes =
				mLastStepStats.particlePrimalTetPacketIrTailLanes;
			stats.avbdCpuSoftBodyParticlePrimalTetPacketIrActiveTailLanes =
				mLastStepStats.particlePrimalTetPacketIrActiveTailLanes;
			stats.avbdCpuSoftBodyParticlePrimalTetPacketIrInvalidBodies =
				mLastStepStats.particlePrimalTetPacketIrInvalidBodies;
			stats.avbdCpuSoftBodyCollisionDetectionCalls =
				mLastCollisionStats.detectionCalls;
			stats.avbdCpuSoftBodyCollisionBodyPairs =
				mLastCollisionStats.bodyPairs;
			stats.avbdCpuSoftBodyCollisionOverlappingBodyPairs =
				mLastCollisionStats.overlappingBodyPairs;
			stats.avbdCpuSoftBodyCollisionParticleSurfaceCandidates =
				mLastCollisionStats.particleSurfaceCandidates;
			stats.avbdCpuSoftBodyCollisionInsideTriangleTests =
				mLastCollisionStats.insideTriangleTests;
			stats.avbdCpuSoftBodyCollisionClosestTriangleTests =
				mLastCollisionStats.closestTriangleTests;
			stats.avbdCpuSoftBodyCollisionSelfTriangleTests =
				mLastCollisionStats.selfTriangleTests;
			stats.avbdCpuSoftBodyCollisionSelfTriangleBoundsBuilt =
				mLastCollisionStats.selfTriangleBoundsBuilt;
			stats.avbdCpuSoftBodyCollisionSelfVertexSweepEntriesBuilt =
				mLastCollisionStats.selfVertexSweepEntriesBuilt;
			stats.avbdCpuSoftBodyCollisionSelfEdgeBoundsBuilt =
				mLastCollisionStats.selfEdgeBoundsBuilt;
			stats.avbdCpuSoftBodyCollisionSurfaceBvhRefitNodes =
				mLastCollisionStats.surfaceTriangleBvhRefitNodes;
			stats.avbdCpuSoftBodyCollisionSurfaceBvhCandidates =
				mLastCollisionStats.surfaceTriangleBvhCandidateTriangles;
			stats.avbdCpuSoftBodyCollisionSurfaceEdgeBvhRefitNodes =
				mLastCollisionStats.surfaceEdgeBvhRefitNodes;
			stats.avbdCpuSoftBodyCollisionSurfaceEdgeBvhCandidates =
				mLastCollisionStats.surfaceEdgeBvhCandidateEdges;
			stats.avbdCpuSoftBodyCollisionRigidParticleTests =
				mLastCollisionStats.rigidParticleBoxTests +
				mLastCollisionStats.rigidParticleSphereTests +
				mLastCollisionStats.rigidParticleCapsuleTests +
				mLastCollisionStats.rigidParticleConvexTests +
				mLastCollisionStats.rigidParticleTriangleSurfaceTests;
			stats.avbdCpuSoftBodyCollisionRigidTriangleFaceCandidates =
				mLastCollisionStats.rigidTriangleSurfaceFaceCandidates;
			stats.avbdCpuSoftBodyCollisionRigidTriangleFaceTests =
				mLastCollisionStats.rigidTriangleSurfaceFaceTests;
			stats.avbdCpuSoftBodyCollisionRigidTriangleEdgeCandidates =
				mLastCollisionStats.rigidTriangleSurfaceEdgeCandidates;
			stats.avbdCpuSoftBodyCollisionRigidTriangleEdgeTests =
				mLastCollisionStats.rigidTriangleSurfaceEdgeTests;
			stats.avbdCpuSoftBodyCollisionRigidTriangleVertexCandidates =
				mLastCollisionStats.rigidTriangleSurfaceVertexCandidates;
			stats.avbdCpuSoftBodyCollisionRigidTriangleVertexTests =
				mLastCollisionStats.rigidTriangleSurfaceVertexTests;
			stats.avbdCpuSoftBodyCollisionGeneratedGroundContacts =
				mLastCollisionStats.generatedGroundContacts;
			stats.avbdCpuSoftBodyCollisionGeneratedRigidContacts =
				mLastCollisionStats.generatedRigidContacts;
			stats.avbdCpuSoftBodyCollisionGeneratedSoftContacts =
				mLastCollisionStats.generatedSoftContacts;
			stats.avbdCpuSoftBodyCollisionGeneratedSelfContacts =
				mLastCollisionStats.generatedSelfContacts;
		}

} // namespace Sc
} // namespace physx
