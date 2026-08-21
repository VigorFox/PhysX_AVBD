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

#ifndef PX_SIMULATION_STATISTICS_H
#define PX_SIMULATION_STATISTICS_H

#include "foundation/PxAssert.h"
#include "PxPhysXConfig.h"
#include "foundation/PxSimpleTypes.h"
#include "geometry/PxGeometry.h"

#if !PX_DOXYGEN
namespace physx
{
#endif

/**
\brief Structure used to retrieve actual sizes/counts for the configuration parameters provided in PxGpuDynamicsMemoryConfig.

\note All the values in this structure are reported as the maximum over the lifetime of a PxScene.

\see PxScene::getSimulationStatistics(), PxSimulationStatistics, PxSceneDesc::PxGpuDynamicsMemoryConfig
*/
struct PxGpuDynamicsMemoryConfigStatistics
{
	PxU64 	tempBufferCapacity; 		//!< actual size needed (bytes) for PxGpuDynamicsMemoryConfig::tempBufferCapacity.
	PxU32	rigidContactCount;			//!< actual number of rigid contacts needed - see PxGpuDynamicsMemoryConfig::maxRigidContactCount.
	PxU32	rigidPatchCount;			//!< actual number of rigid contact patches needed - see PxGpuDynamicsMemoryConfig::maxRigidPatchCount.
	PxU32	foundLostPairs;				//!< actual number of lost/found pairs needed - see PxGpuDynamicsMemoryConfig::foundLostPairsCapacity.
	PxU32	foundLostAggregatePairs;	//!< actual number of lost/found aggregate pairs needed - see PxGpuDynamicsMemoryConfig::foundLostAggregatePairsCapacity.
	PxU32	totalAggregatePairs;		//!< actual number of aggregate pairs needed - see PxGpuDynamicsMemoryConfig::totalAggregatePairsCapacity.
	PxU32	deformableSurfaceContacts;	//!< actual number of  deformable surface contacts needed - see PxGpuDynamicsMemoryConfig::maxDeformableSurfaceContacts.
	PxU32	deformableVolumeContacts;	//!< actual number of deformable volume contact needed - see PxGpuDynamicsMemoryConfig::maxDeformableVolumeContacts.
	PxU32	softbodyContacts;			//!< deprecated, use deformableVolumeContacts.
	PxU32	particleContacts;			//!< actual number of particle contacts needed - see PxGpuDynamicsMemoryConfig::maxParticleContacts.
	PxU32	collisionStackSize;			//!< actual size (bytes) needed for the collision stack - see PxGpuDynamicsMemoryConfig::collisionStackSize.

	PxGpuDynamicsMemoryConfigStatistics() :
		tempBufferCapacity			(0),
		rigidContactCount			(0),
		rigidPatchCount				(0),
		foundLostPairs				(0),
		foundLostAggregatePairs		(0),
		totalAggregatePairs			(0),
		deformableSurfaceContacts	(0),
		deformableVolumeContacts	(0),
		softbodyContacts			(0), // deprecated
		particleContacts			(0),
		collisionStackSize			(0)
	{ }
};

/**
\brief Class used to retrieve statistics for a simulation step.

\see PxScene::getSimulationStatistics()
*/
class PxSimulationStatistics
{
public:

	/**
	\brief Different types of rigid body collision pair statistics.
	\see getRbPairStats
	*/
	enum RbPairStatsType
	{
		/**
		\brief Shape pairs processed as discrete contact pairs for the current simulation step.
		*/
		eDISCRETE_CONTACT_PAIRS,

		/**
		\brief Shape pairs processed as swept integration pairs for the current simulation step.

		\note Counts the pairs for which special CCD (continuous collision detection) work was actually done and NOT the number of pairs which were configured for CCD. 
		Furthermore, there can be multiple CCD passes and all processed pairs of all passes are summed up, hence the number can be larger than the amount of pairs which have been configured for CCD.

		\see PxPairFlag::eDETECT_CCD_CONTACT,
		*/
		eCCD_PAIRS,

		/**
		\brief Shape pairs processed with user contact modification enabled for the current simulation step.

		\see PxContactModifyCallback
		*/
		eMODIFIED_CONTACT_PAIRS,

		/**
		\brief Trigger shape pairs processed for the current simulation step.

		\see PxShapeFlag::eTRIGGER_SHAPE
		*/
		eTRIGGER_PAIRS
	};


//objects:
	/**
	\brief Number of active PxConstraint objects (joints etc.) for the current simulation step.
	*/
	PxU32   nbActiveConstraints;

	/**
	\brief Number of active dynamic bodies for the current simulation step.

	\note Does not include active kinematic bodies
	*/
	PxU32   nbActiveDynamicBodies;

	/**
	\brief Number of active kinematic bodies for the current simulation step.
	
	\note Kinematic deactivation occurs at the end of the frame after the last call to PxRigidDynamic::setKinematicTarget() was called so kinematics that are
	deactivated in a given frame will be included by this counter.
	*/
	PxU32   nbActiveKinematicBodies;

	/**
	\brief Number of static bodies for the current simulation step.
	*/
	PxU32	nbStaticBodies;

	/**
	\brief Number of dynamic bodies for the current simulation step.

	\note Includes inactive bodies and articulation links
	\note Does not include kinematic bodies
	*/
	PxU32   nbDynamicBodies;

	/**
	\brief Number of kinematic bodies for the current simulation step.

	\note Includes inactive bodies
	*/
	PxU32   nbKinematicBodies;

	/**
	\brief Number of shapes of each geometry type.
	*/

	PxU32	nbShapes[PxGeometryType::eGEOMETRY_COUNT];

	/**
	\brief Number of aggregates in the scene.
	*/
	PxU32	nbAggregates;
	
	/**
	\brief Number of articulations in the scene.
	*/
	PxU32	nbArticulations;

//solver:
	/**
	\brief The number of 1D axis constraints(joints+contact) present in the current simulation step.
	*/
	PxU32	nbAxisSolverConstraints;

	/**
	\brief The size (in bytes) of the compressed contact stream in the current simulation step
	*/
	PxU32   compressedContactSize;

	/**
	\brief The total required size (in bytes) of the contact constraints in the current simulation step
	*/
	PxU32   requiredContactConstraintMemory;

	/**
	\brief The peak amount of memory (in bytes) that was allocated for constraints (this includes joints) in the current simulation step
	*/
	PxU32   peakConstraintMemory;

	/**
	\brief Number of AVBD CPU component-fallback soft-body solves in the current simulation step.

	A value of zero does not imply that a scene contains no AVBD deformables: a
	native rigid/soft island may own the step instead.  The associated workspace
	growth fields therefore describe the component fallback path only.
	*/
	PxU32	avbdCpuSoftBodyComponentFallbackSteps;

	/**
	\brief Number of native rigid/soft AVBD island solves selected in the current simulation step.
	*/
	PxU32	avbdCpuSoftBodyNativeIslandSteps;

	/**
	\brief AVBD CPU ISA dispatch snapshot for this Scene.

	`Requested` and `Selected` use 0=auto, 1=sse2, 2=avx2fma and 3=invalid.
	`CompiledBackendMask` uses bit 0=SSE2 and bit 1=AVX2+FMA. Capability bits
	are SSE2, AVX, OSXSAVE, XMM/YMM state, AVX2 and FMA in that order. These
	are immutable process/context configuration diagnostics, not a timing API.
	*/
	PxU32	avbdCpuIsaRequested;
	PxU32	avbdCpuIsaSelected;
	PxU32	avbdCpuIsaCompiledBackendMask;
	PxU32	avbdCpuIsaCapabilityMask;
	PxU32	avbdCpuIsaForceModeRejected;
	PxU32	avbdCpuIsaKernelSelfTestPassed;
	PxU32	avbdCpuIsaFmaUsed;
	PxReal	avbdCpuIsaKernelSelfTestValue;

	/**
	\brief AVBD CPU component-fallback persistent-workspace capacity growth in the current simulation step.

	The fields include both the initial OGC contact preparation and all solver
	redetection stages.  They are intended for performance telemetry; they are
	not a memory-accounting API.
	*/
	PxU64	avbdCpuSoftBodyWorkspaceGrowthEvents;
	PxU64	avbdCpuSoftBodyWorkspaceGrowthBytes;
	PxU64	avbdCpuSoftBodyContactWorkspaceGrowthEvents;
	PxU64	avbdCpuSoftBodyContactWorkspaceGrowthBytes;
	PxU64	avbdCpuSoftBodyContactSweepScratchGrowthEvents;
	PxU64	avbdCpuSoftBodyContactSweepScratchGrowthBytes;
	PxU64	avbdCpuSoftBodyContactOutputGrowthEvents;
	PxU64	avbdCpuSoftBodyContactOutputGrowthBytes;

	/**
	\brief Per completed component-fallback step contact capacity watermarks.

	These values identify the contact-output, contact-incidence and contact-state
	transfer capacities required by the observed step. They are diagnostics for
	persistent-workspace policy, not a public allocation guarantee.
	*/
	PxU32	avbdCpuSoftBodyPeakContactOutputCount;
	PxU32	avbdCpuSoftBodyPeakContactOutputCapacity;
	PxU32	avbdCpuSoftBodyPeakContactIncidenceCount;
	PxU32	avbdCpuSoftBodyPeakContactIncidenceCapacity;
	PxU32	avbdCpuSoftBodyPeakStateTransferContactCount;
	PxU32	avbdCpuSoftBodyPeakStateTransferContactCapacity;
	PxU32	avbdCpuSoftBodyPeakStateTransferUsedCapacity;

	/**
	\brief AVBD CPU component-fallback P4 particle-primal color-plan telemetry.

	`ColorCount` and `DynamicAccessGroupCount` are the largest complete plan
	published in the step. The sweep counters distinguish the explicit
	validation-only colored serial schedule from its required serial fallback;
	they do not report worker parallelism.
	*/
	PxU32	avbdCpuSoftBodyParticlePrimalColorCount;
	PxU32	avbdCpuSoftBodyParticlePrimalDynamicAccessGroupCount;
	PxU64	avbdCpuSoftBodyParticlePrimalColoredSerialSweeps;
	PxU64	avbdCpuSoftBodyParticlePrimalColoredSerialFallbackSweeps;

	/**
	\brief Read-only qualification counters for the default-off AVBD CPU
	single-tet ground-patch experiment.

	They count prepared PositionAL ground rows observed at redetection epochs.
	They do not imply that a coupled patch solve was applied.
	*/
	PxU64	avbdCpuSoftBodyGroundTetPatchGroundPositionAlRows;
	PxU64	avbdCpuSoftBodyGroundTetPatchFourSupportRows;
	PxU64	avbdCpuSoftBodyGroundTetPatchSingleTetRows;
	PxU64	avbdCpuSoftBodyGroundTetPatchActiveRows;

	/**
	\brief Default-off AVBD CPU world-static velocity-tangent-owner telemetry.

	`Rows` counts final velocity rows that used the separate tangent owner;
	`AppliedRows` counts finite local disk projections actually written.
	*/
	PxU64	avbdCpuSoftBodyWorldStaticVelocityTangentOwnerRows;
	PxU64	avbdCpuSoftBodyWorldStaticVelocityTangentAppliedRows;
	/**
	\brief AVBD CPU P8.1 particle-primal material-work census.

	The process-start opt-in census counts work that reached the scalar
	particle-primal authority. `TetPacket8*` measures consecutive tet-incidence
	capacity only; it does not report executed SIMD packets or permit a solver
	ordering change.
	*/
	PxU64	avbdCpuSoftBodyParticlePrimalCensusDynamicParticleSolves;
	PxU64	avbdCpuSoftBodyParticlePrimalCensusTriangleEvaluations;
	PxU64	avbdCpuSoftBodyParticlePrimalCensusCorotationalTetEvaluations;
	PxU64	avbdCpuSoftBodyParticlePrimalCensusNeoHookeanTetEvaluations;
	PxU64	avbdCpuSoftBodyParticlePrimalCensusBendingEvaluations;
	PxU64	avbdCpuSoftBodyParticlePrimalCensusContactEvaluations;
	PxU64	avbdCpuSoftBodyParticlePrimalCensusTetPacket8FullPackets;
	PxU64	avbdCpuSoftBodyParticlePrimalCensusTetPacket8TailLanes;

	/**
	\brief AVBD CPU P8.2 canonical corotational-tet packet-IR telemetry.

	These fields describe immutable topology metadata built under the P8.2
	opt-in. They are explicitly not a count of executed SIMD packets.
	*/
	PxU64	avbdCpuSoftBodyParticlePrimalTetPacketIrBodies;
	PxU64	avbdCpuSoftBodyParticlePrimalTetPacketIrPackets;
	PxU64	avbdCpuSoftBodyParticlePrimalTetPacketIrActiveLanes;
	PxU64	avbdCpuSoftBodyParticlePrimalTetPacketIrTailLanes;
	PxU64	avbdCpuSoftBodyParticlePrimalTetPacketIrActiveTailLanes;
	PxU64	avbdCpuSoftBodyParticlePrimalTetPacketIrInvalidBodies;

	/**
	\brief AVBD CPU component-fallback OGC work counters for the current simulation step.

	The counters include the initial detection and every outer redetection. They
	are performance diagnostics only: they do not describe public contact-pair
	statistics or relax OGC coverage.
	*/
	PxU64	avbdCpuSoftBodyCollisionDetectionCalls;
	PxU64	avbdCpuSoftBodyCollisionBodyPairs;
	PxU64	avbdCpuSoftBodyCollisionOverlappingBodyPairs;
	PxU64	avbdCpuSoftBodyCollisionParticleSurfaceCandidates;
	PxU64	avbdCpuSoftBodyCollisionInsideTriangleTests;
	PxU64	avbdCpuSoftBodyCollisionClosestTriangleTests;
	PxU64	avbdCpuSoftBodyCollisionSelfTriangleTests;
	PxU64	avbdCpuSoftBodyCollisionSelfTriangleBoundsBuilt;
	PxU64	avbdCpuSoftBodyCollisionSelfVertexSweepEntriesBuilt;
	PxU64	avbdCpuSoftBodyCollisionSelfEdgeBoundsBuilt;
	PxU64	avbdCpuSoftBodyCollisionSurfaceBvhRefitNodes;
	PxU64	avbdCpuSoftBodyCollisionSurfaceBvhCandidates;
	PxU64	avbdCpuSoftBodyCollisionSurfaceEdgeBvhRefitNodes;
	PxU64	avbdCpuSoftBodyCollisionSurfaceEdgeBvhCandidates;
	PxU64	avbdCpuSoftBodyCollisionRigidParticleTests;
	PxU64	avbdCpuSoftBodyCollisionRigidTriangleFaceCandidates;
	PxU64	avbdCpuSoftBodyCollisionRigidTriangleFaceTests;
	PxU64	avbdCpuSoftBodyCollisionRigidTriangleEdgeCandidates;
	PxU64	avbdCpuSoftBodyCollisionRigidTriangleEdgeTests;
	PxU64	avbdCpuSoftBodyCollisionRigidTriangleVertexCandidates;
	PxU64	avbdCpuSoftBodyCollisionRigidTriangleVertexTests;
	PxU64	avbdCpuSoftBodyCollisionGeneratedGroundContacts;
	PxU64	avbdCpuSoftBodyCollisionGeneratedRigidContacts;
	PxU64	avbdCpuSoftBodyCollisionGeneratedSoftContacts;
	PxU64	avbdCpuSoftBodyCollisionGeneratedSelfContacts;

	/**
	\brief AVBD CPU Scene-taskgraph telemetry for the current simulation step.

	The counters describe PxTaskManager/PxCpuDispatcher work only.  They do not
	include an AVBD-private thread pool (none is permitted by the CPU AVBD
	taskgraph contract). `peakActiveSolveTasks` is observed concurrency, whereas
	`requestedDispatcherWorkers` is the dispatcher capacity requested by the
	scene. `pureSoftEligible*` identifies large soft work by particle count, not
	by the unrelated rigid-body count; it is a P2/P4 scheduling denominator.
	*/
	PxU32	avbdCpuTaskGraphRequestedDispatcherWorkers;
	PxU32	avbdCpuTaskGraphSubmittedSolveTasks;
	PxU32	avbdCpuTaskGraphCompletedSolveTasks;
	PxU32	avbdCpuTaskGraphPeakActiveSolveTasks;
	PxU32	avbdCpuTaskGraphBarrierTasks;
	PxU32	avbdCpuTaskGraphSerialSolveTasks;
	PxU32	avbdCpuTaskGraphSubmittedPredictionTasks;
	PxU32	avbdCpuTaskGraphCompletedPredictionTasks;
	PxU32	avbdCpuTaskGraphPeakActivePredictionTasks;
	PxU32	avbdCpuTaskGraphSerialPredictionStages;
	PxU32	avbdCpuTaskGraphSubmittedWriteBackTasks;
	PxU32	avbdCpuTaskGraphCompletedWriteBackTasks;
	PxU32	avbdCpuTaskGraphPeakActiveWriteBackTasks;
	PxU32	avbdCpuTaskGraphSerialWriteBackStages;
	/**
	\brief AVBD P4.5 causal-layer particle-primal task/fan-in telemetry.

	These are actual Scene dispatcher child tasks and parent reductions. They
	do not reinterpret the P4 causal-plan/sweep counters as concurrency.
	*/
	PxU32	avbdCpuTaskGraphSubmittedCausalLayerTasks;
	PxU32	avbdCpuTaskGraphCompletedCausalLayerTasks;
	PxU32	avbdCpuTaskGraphPeakActiveCausalLayerTasks;
	PxU32	avbdCpuTaskGraphCausalLayerFanIns;
	PxU32	avbdCpuTaskGraphSerialCausalLayerFallbacks;
	PxU32	avbdCpuTaskGraphMaxCausalLayerOccupancy;
	PxU64	avbdCpuTaskGraphTotalCausalLayerOccupancy;
	/** AVBD P5.3b world-plane candidate task/fan-in telemetry. */
	PxU32	avbdCpuTaskGraphSubmittedWorldPlaneContactTasks;
	PxU32	avbdCpuTaskGraphCompletedWorldPlaneContactTasks;
	PxU32	avbdCpuTaskGraphPeakActiveWorldPlaneContactTasks;
	PxU32	avbdCpuTaskGraphWorldPlaneContactFanIns;
	PxU32	avbdCpuTaskGraphSerialWorldPlaneContactFallbacks;
	/** AVBD P5.4b static rigid-box discrete-SDF candidate task/fan-in telemetry. */
	PxU32	avbdCpuTaskGraphSubmittedRigidBoxSdfContactTasks;
	PxU32	avbdCpuTaskGraphCompletedRigidBoxSdfContactTasks;
	PxU32	avbdCpuTaskGraphPeakActiveRigidBoxSdfContactTasks;
	PxU32	avbdCpuTaskGraphRigidBoxSdfContactFanIns;
	PxU32	avbdCpuTaskGraphSerialRigidBoxSdfContactFallbacks;
	/** AVBD P5.5b static rigid-sphere discrete-SDF candidate task/fan-in telemetry. */
	PxU32	avbdCpuTaskGraphSubmittedRigidSphereSdfContactTasks;
	PxU32	avbdCpuTaskGraphCompletedRigidSphereSdfContactTasks;
	PxU32	avbdCpuTaskGraphPeakActiveRigidSphereSdfContactTasks;
	PxU32	avbdCpuTaskGraphRigidSphereSdfContactFanIns;
	PxU32	avbdCpuTaskGraphSerialRigidSphereSdfContactFallbacks;
	/** AVBD P5.6b static rigid-capsule discrete-SDF candidate task/fan-in telemetry. */
	PxU32	avbdCpuTaskGraphSubmittedRigidCapsuleSdfContactTasks;
	PxU32	avbdCpuTaskGraphCompletedRigidCapsuleSdfContactTasks;
	PxU32	avbdCpuTaskGraphPeakActiveRigidCapsuleSdfContactTasks;
	PxU32	avbdCpuTaskGraphRigidCapsuleSdfContactFanIns;
	PxU32	avbdCpuTaskGraphSerialRigidCapsuleSdfContactFallbacks;
	/** AVBD P5.7b static rigid-convex discrete-SDF candidate task/fan-in telemetry. */
	PxU32	avbdCpuTaskGraphSubmittedRigidConvexSdfContactTasks;
	PxU32	avbdCpuTaskGraphCompletedRigidConvexSdfContactTasks;
	PxU32	avbdCpuTaskGraphPeakActiveRigidConvexSdfContactTasks;
	PxU32	avbdCpuTaskGraphRigidConvexSdfContactFanIns;
	PxU32	avbdCpuTaskGraphSerialRigidConvexSdfContactFallbacks;
	/** AVBD P5.8b static rigid-triangle current-pose candidate task/fan-in telemetry. */
	PxU32	avbdCpuTaskGraphSubmittedRigidTriangleSurfaceContactTasks;
	PxU32	avbdCpuTaskGraphCompletedRigidTriangleSurfaceContactTasks;
	PxU32	avbdCpuTaskGraphPeakActiveRigidTriangleSurfaceContactTasks;
	PxU32	avbdCpuTaskGraphRigidTriangleSurfaceContactFanIns;
	PxU32	avbdCpuTaskGraphSerialRigidTriangleSurfaceContactFallbacks;
	/** AVBD P5.10b self-BVH VF/EE task/fan-in telemetry. */
	PxU32	avbdCpuTaskGraphSubmittedSelfBvhContactTasks;
	PxU32	avbdCpuTaskGraphCompletedSelfBvhContactTasks;
	PxU32	avbdCpuTaskGraphPeakActiveSelfBvhContactTasks;
	PxU32	avbdCpuTaskGraphSelfBvhContactFanIns;
	PxU32	avbdCpuTaskGraphSerialSelfBvhContactFallbacks;
	PxU32	avbdCpuTaskGraphPureSoftEligibleIslands;
	PxU32	avbdCpuTaskGraphPureSoftEligibleParticles;

//broadphase:
	/**
	\brief Get number of broadphase volumes added for the current simulation step.

	\return Number of broadphase volumes added.
	*/
	PX_FORCE_INLINE	PxU32 getNbBroadPhaseAdds() const
	{
		return nbBroadPhaseAdds;
	}

	/**
	\brief Get number of broadphase volumes removed for the current simulation step.

	\return Number of broadphase volumes removed.
	*/
	PX_FORCE_INLINE	PxU32 getNbBroadPhaseRemoves() const
	{
		return nbBroadPhaseRemoves;
	}

//collisions:
	/**
	\brief Get number of shape collision pairs of a certain type processed for the current simulation step.

	There is an entry for each geometry pair type.

	\note entry[i][j] = entry[j][i], hence, if you want the sum of all pair
	      types, you need to discard the symmetric entries

	\param[in] pairType The type of pair for which to get information
	\param[in] g0 The geometry type of one pair object
	\param[in] g1 The geometry type of the other pair object
	\return Number of processed pairs of the specified geometry types.
	*/
	PxU32 getRbPairStats(RbPairStatsType pairType, PxGeometryType::Enum g0, PxGeometryType::Enum g1) const
	{
		PX_ASSERT_WITH_MESSAGE(	(pairType >= eDISCRETE_CONTACT_PAIRS) &&
								(pairType <= eTRIGGER_PAIRS),
								"Invalid pairType in PxSimulationStatistics::getRbPairStats");

		if (g0 >= PxGeometryType::eGEOMETRY_COUNT || g1 >= PxGeometryType::eGEOMETRY_COUNT)
		{
			PX_ASSERT(false);
			return 0;
		}

		PxU32 nbPairs = 0;
		switch(pairType)
		{
			case eDISCRETE_CONTACT_PAIRS:
				nbPairs = nbDiscreteContactPairs[g0][g1];
				break;
			case eCCD_PAIRS:
				nbPairs = nbCCDPairs[g0][g1];
				break;
			case eMODIFIED_CONTACT_PAIRS:
				nbPairs = nbModifiedContactPairs[g0][g1];
				break;
			case eTRIGGER_PAIRS:
				nbPairs = nbTriggerPairs[g0][g1];
				break;
		}
		return nbPairs;
	}

	/**
	\brief Total number of (non CCD) pairs reaching narrow phase
	*/
	PxU32	nbDiscreteContactPairsTotal;

	/**
	\brief Total number of (non CCD) pairs for which contacts are successfully cached (<=nbDiscreteContactPairsTotal)
	\note This includes pairs for which no contacts are generated, it still counts as a cache hit.
	*/
	PxU32	nbDiscreteContactPairsWithCacheHits;

	/**
	\brief Total number of (non CCD) pairs for which at least 1 contact was generated (<=nbDiscreteContactPairsTotal)
	*/
	PxU32	nbDiscreteContactPairsWithContacts;

	/**
	\brief Number of new pairs found by BP this frame
	*/
	PxU32	nbNewPairs;

	/**
	\brief Number of lost pairs from BP this frame
	*/
	PxU32	nbLostPairs;

	/**
	\brief Number of new touches found by NP this frame
	*/
	PxU32	nbNewTouches;

	/**
	\brief Number of lost touches from NP this frame
	*/
	PxU32	nbLostTouches;

	/**
	\brief Number of partitions used by the solver this frame
	*/
	PxU32	nbPartitions;

	/**
	\brief GPU device memory in bytes allocated for particle state accessible through API
	*/
	PxU64	gpuMemParticles;

	/**
	\brief GPU device memory in bytes allocated for deformable surface state accessible through API
	*/
	PxU64	gpuMemDeformableSurfaces;

	/**
	\brief GPU device memory in bytes allocated for deformable volume state accessible through API
	*/
	PxU64	gpuMemDeformableVolumes;

	/**
	\brief GPU device memory in bytes allocated for internal heap allocation
	*/
	PxU64	gpuMemHeap;

	/**
	\brief GPU device heap memory used for broad phase in bytes
	*/
	PxU64	gpuMemHeapBroadPhase;

	/**
	\brief GPU device heap memory used for narrow phase in bytes
	*/
	PxU64	gpuMemHeapNarrowPhase;

	/**
	\brief GPU device heap memory used for solver in bytes
	*/
	PxU64	gpuMemHeapSolver;

	/**
	\brief GPU device heap memory used for articulations in bytes
	*/
	PxU64	gpuMemHeapArticulation;

	/**
	\brief GPU device heap memory used for simulation pipeline in bytes
	*/
	PxU64	gpuMemHeapSimulation;

	/**
	\brief GPU device heap memory used for articulations in the simulation pipeline in bytes
	*/
	PxU64	gpuMemHeapSimulationArticulation;

	/**
	\brief GPU device heap memory used for particles in the simulation pipeline in bytes
	*/
	PxU64	gpuMemHeapSimulationParticles;

	/**
	\brief GPU device heap memory used for deformable surfaces in the simulation pipeline in bytes
	*/
	PxU64	gpuMemHeapSimulationDeformableSurface;

	/**
	\brief GPU device heap memory used for deformable volumes in the simulation pipeline in bytes
	*/
	PxU64	gpuMemHeapSimulationDeformableVolume;

	/**
	\brief GPU device heap memory used for shared buffers in the particles pipeline in bytes
	*/
	PxU64	gpuMemHeapParticles;

	/**
	\brief GPU device heap memory used for shared buffers in the deformable surface pipeline in bytes
	*/
	PxU64	gpuMemHeapDeformableSurfaces;

	/**
	\brief GPU device heap memory used for shared buffers in the deformable volume pipeline in bytes
	*/
	PxU64	gpuMemHeapDeformableVolumes;

	/**
	\brief GPU device heap memory not covered by other stats in bytes
	*/
	PxU64	gpuMemHeapOther;

	/**
	\brief Structure containing statistics about actual count/sizes used for the configuration parameters in PxGpuDynamicsMemoryConfig
	*/
	PxGpuDynamicsMemoryConfigStatistics gpuDynamicsMemoryConfigStatistics;


	PxSimulationStatistics() :
		nbActiveConstraints						(0),
		nbActiveDynamicBodies					(0),
		nbActiveKinematicBodies					(0),
		nbStaticBodies							(0),
		nbDynamicBodies							(0),
		nbKinematicBodies						(0),
		nbAggregates							(0),
		nbArticulations							(0),
		nbAxisSolverConstraints					(0),
		compressedContactSize					(0),
		requiredContactConstraintMemory			(0),
		peakConstraintMemory					(0),
		avbdCpuSoftBodyComponentFallbackSteps	(0),
		avbdCpuSoftBodyNativeIslandSteps		(0),
		avbdCpuIsaRequested					(0),
		avbdCpuIsaSelected					(0),
		avbdCpuIsaCompiledBackendMask			(0),
		avbdCpuIsaCapabilityMask				(0),
		avbdCpuIsaForceModeRejected			(0),
		avbdCpuIsaKernelSelfTestPassed			(0),
		avbdCpuIsaFmaUsed						(0),
		avbdCpuIsaKernelSelfTestValue			(0.0f),
		avbdCpuSoftBodyWorkspaceGrowthEvents	(0),
		avbdCpuSoftBodyWorkspaceGrowthBytes		(0),
		avbdCpuSoftBodyContactWorkspaceGrowthEvents(0),
		avbdCpuSoftBodyContactWorkspaceGrowthBytes(0),
		avbdCpuSoftBodyContactSweepScratchGrowthEvents(0),
		avbdCpuSoftBodyContactSweepScratchGrowthBytes(0),
		avbdCpuSoftBodyContactOutputGrowthEvents(0),
		avbdCpuSoftBodyContactOutputGrowthBytes(0),
		avbdCpuSoftBodyPeakContactOutputCount(0),
		avbdCpuSoftBodyPeakContactOutputCapacity(0),
		avbdCpuSoftBodyPeakContactIncidenceCount(0),
		avbdCpuSoftBodyPeakContactIncidenceCapacity(0),
		avbdCpuSoftBodyPeakStateTransferContactCount(0),
		avbdCpuSoftBodyPeakStateTransferContactCapacity(0),
		avbdCpuSoftBodyPeakStateTransferUsedCapacity(0),
		avbdCpuSoftBodyParticlePrimalColorCount(0),
		avbdCpuSoftBodyParticlePrimalDynamicAccessGroupCount(0),
		avbdCpuSoftBodyParticlePrimalColoredSerialSweeps(0),
		avbdCpuSoftBodyParticlePrimalColoredSerialFallbackSweeps(0),
		avbdCpuSoftBodyGroundTetPatchGroundPositionAlRows(0),
		avbdCpuSoftBodyGroundTetPatchFourSupportRows(0),
		avbdCpuSoftBodyGroundTetPatchSingleTetRows(0),
		avbdCpuSoftBodyGroundTetPatchActiveRows(0),
		avbdCpuSoftBodyWorldStaticVelocityTangentOwnerRows(0),
		avbdCpuSoftBodyWorldStaticVelocityTangentAppliedRows(0),
		avbdCpuSoftBodyParticlePrimalCensusDynamicParticleSolves(0),
		avbdCpuSoftBodyParticlePrimalCensusTriangleEvaluations(0),
		avbdCpuSoftBodyParticlePrimalCensusCorotationalTetEvaluations(0),
		avbdCpuSoftBodyParticlePrimalCensusNeoHookeanTetEvaluations(0),
		avbdCpuSoftBodyParticlePrimalCensusBendingEvaluations(0),
		avbdCpuSoftBodyParticlePrimalCensusContactEvaluations(0),
		avbdCpuSoftBodyParticlePrimalCensusTetPacket8FullPackets(0),
		avbdCpuSoftBodyParticlePrimalCensusTetPacket8TailLanes(0),
		avbdCpuSoftBodyParticlePrimalTetPacketIrBodies(0),
		avbdCpuSoftBodyParticlePrimalTetPacketIrPackets(0),
		avbdCpuSoftBodyParticlePrimalTetPacketIrActiveLanes(0),
		avbdCpuSoftBodyParticlePrimalTetPacketIrTailLanes(0),
		avbdCpuSoftBodyParticlePrimalTetPacketIrActiveTailLanes(0),
		avbdCpuSoftBodyParticlePrimalTetPacketIrInvalidBodies(0),
		avbdCpuSoftBodyCollisionDetectionCalls(0),
		avbdCpuSoftBodyCollisionBodyPairs(0),
		avbdCpuSoftBodyCollisionOverlappingBodyPairs(0),
		avbdCpuSoftBodyCollisionParticleSurfaceCandidates(0),
		avbdCpuSoftBodyCollisionInsideTriangleTests(0),
		avbdCpuSoftBodyCollisionClosestTriangleTests(0),
		avbdCpuSoftBodyCollisionSelfTriangleTests(0),
		avbdCpuSoftBodyCollisionSelfTriangleBoundsBuilt(0),
		avbdCpuSoftBodyCollisionSelfVertexSweepEntriesBuilt(0),
		avbdCpuSoftBodyCollisionSelfEdgeBoundsBuilt(0),
		avbdCpuSoftBodyCollisionSurfaceBvhRefitNodes(0),
		avbdCpuSoftBodyCollisionSurfaceBvhCandidates(0),
		avbdCpuSoftBodyCollisionSurfaceEdgeBvhRefitNodes(0),
		avbdCpuSoftBodyCollisionSurfaceEdgeBvhCandidates(0),
		avbdCpuSoftBodyCollisionRigidParticleTests(0),
		avbdCpuSoftBodyCollisionRigidTriangleFaceCandidates(0),
		avbdCpuSoftBodyCollisionRigidTriangleFaceTests(0),
		avbdCpuSoftBodyCollisionRigidTriangleEdgeCandidates(0),
		avbdCpuSoftBodyCollisionRigidTriangleEdgeTests(0),
		avbdCpuSoftBodyCollisionRigidTriangleVertexCandidates(0),
		avbdCpuSoftBodyCollisionRigidTriangleVertexTests(0),
		avbdCpuSoftBodyCollisionGeneratedGroundContacts(0),
		avbdCpuSoftBodyCollisionGeneratedRigidContacts(0),
		avbdCpuSoftBodyCollisionGeneratedSoftContacts(0),
		avbdCpuSoftBodyCollisionGeneratedSelfContacts(0),
		avbdCpuTaskGraphRequestedDispatcherWorkers(0),
		avbdCpuTaskGraphSubmittedSolveTasks(0),
		avbdCpuTaskGraphCompletedSolveTasks(0),
		avbdCpuTaskGraphPeakActiveSolveTasks(0),
		avbdCpuTaskGraphBarrierTasks(0),
		avbdCpuTaskGraphSerialSolveTasks(0),
		avbdCpuTaskGraphSubmittedPredictionTasks(0),
		avbdCpuTaskGraphCompletedPredictionTasks(0),
		avbdCpuTaskGraphPeakActivePredictionTasks(0),
		avbdCpuTaskGraphSerialPredictionStages(0),
		avbdCpuTaskGraphSubmittedWriteBackTasks(0),
		avbdCpuTaskGraphCompletedWriteBackTasks(0),
		avbdCpuTaskGraphPeakActiveWriteBackTasks(0),
		avbdCpuTaskGraphSerialWriteBackStages(0),
		avbdCpuTaskGraphSubmittedCausalLayerTasks(0),
		avbdCpuTaskGraphCompletedCausalLayerTasks(0),
		avbdCpuTaskGraphPeakActiveCausalLayerTasks(0),
		avbdCpuTaskGraphCausalLayerFanIns(0),
		avbdCpuTaskGraphSerialCausalLayerFallbacks(0),
		avbdCpuTaskGraphMaxCausalLayerOccupancy(0),
		avbdCpuTaskGraphTotalCausalLayerOccupancy(0),
		avbdCpuTaskGraphSubmittedWorldPlaneContactTasks(0),
		avbdCpuTaskGraphCompletedWorldPlaneContactTasks(0),
		avbdCpuTaskGraphPeakActiveWorldPlaneContactTasks(0),
		avbdCpuTaskGraphWorldPlaneContactFanIns(0),
		avbdCpuTaskGraphSerialWorldPlaneContactFallbacks(0),
		avbdCpuTaskGraphSubmittedRigidBoxSdfContactTasks(0),
		avbdCpuTaskGraphCompletedRigidBoxSdfContactTasks(0),
		avbdCpuTaskGraphPeakActiveRigidBoxSdfContactTasks(0),
		avbdCpuTaskGraphRigidBoxSdfContactFanIns(0),
		avbdCpuTaskGraphSerialRigidBoxSdfContactFallbacks(0),
		avbdCpuTaskGraphSubmittedRigidSphereSdfContactTasks(0),
		avbdCpuTaskGraphCompletedRigidSphereSdfContactTasks(0),
		avbdCpuTaskGraphPeakActiveRigidSphereSdfContactTasks(0),
		avbdCpuTaskGraphRigidSphereSdfContactFanIns(0),
		avbdCpuTaskGraphSerialRigidSphereSdfContactFallbacks(0),
		avbdCpuTaskGraphSubmittedRigidCapsuleSdfContactTasks(0),
		avbdCpuTaskGraphCompletedRigidCapsuleSdfContactTasks(0),
		avbdCpuTaskGraphPeakActiveRigidCapsuleSdfContactTasks(0),
		avbdCpuTaskGraphRigidCapsuleSdfContactFanIns(0),
		avbdCpuTaskGraphSerialRigidCapsuleSdfContactFallbacks(0),
		avbdCpuTaskGraphSubmittedRigidConvexSdfContactTasks(0),
		avbdCpuTaskGraphCompletedRigidConvexSdfContactTasks(0),
		avbdCpuTaskGraphPeakActiveRigidConvexSdfContactTasks(0),
		avbdCpuTaskGraphRigidConvexSdfContactFanIns(0),
		avbdCpuTaskGraphSerialRigidConvexSdfContactFallbacks(0),
		avbdCpuTaskGraphSubmittedRigidTriangleSurfaceContactTasks(0),
		avbdCpuTaskGraphCompletedRigidTriangleSurfaceContactTasks(0),
		avbdCpuTaskGraphPeakActiveRigidTriangleSurfaceContactTasks(0),
		avbdCpuTaskGraphRigidTriangleSurfaceContactFanIns(0),
		avbdCpuTaskGraphSerialRigidTriangleSurfaceContactFallbacks(0),
		avbdCpuTaskGraphSubmittedSelfBvhContactTasks(0),
		avbdCpuTaskGraphCompletedSelfBvhContactTasks(0),
		avbdCpuTaskGraphPeakActiveSelfBvhContactTasks(0),
		avbdCpuTaskGraphSelfBvhContactFanIns(0),
		avbdCpuTaskGraphSerialSelfBvhContactFallbacks(0),
		avbdCpuTaskGraphPureSoftEligibleIslands(0),
		avbdCpuTaskGraphPureSoftEligibleParticles(0),
		nbDiscreteContactPairsTotal				(0),
		nbDiscreteContactPairsWithCacheHits		(0),
		nbDiscreteContactPairsWithContacts		(0),
		nbNewPairs								(0),
		nbLostPairs								(0),
		nbNewTouches							(0),
		nbLostTouches							(0),
		nbPartitions							(0),
		gpuMemParticles							(0),
		gpuMemDeformableSurfaces				(0),
		gpuMemDeformableVolumes					(0),
		gpuMemHeap								(0),
		gpuMemHeapBroadPhase					(0),
		gpuMemHeapNarrowPhase					(0),
		gpuMemHeapSolver						(0),
		gpuMemHeapArticulation					(0),
		gpuMemHeapSimulation					(0),
		gpuMemHeapSimulationArticulation		(0),
		gpuMemHeapSimulationParticles			(0),
		gpuMemHeapSimulationDeformableSurface	(0),
		gpuMemHeapSimulationDeformableVolume	(0),
		gpuMemHeapParticles						(0),
		gpuMemHeapDeformableSurfaces			(0), 
		gpuMemHeapDeformableVolumes				(0),
		gpuMemHeapOther							(0)
	{
		nbBroadPhaseAdds = 0;
		nbBroadPhaseRemoves = 0;

		for(PxU32 i=0; i < PxGeometryType::eGEOMETRY_COUNT; i++)
		{
			for(PxU32 j=0; j < PxGeometryType::eGEOMETRY_COUNT; j++)
			{
				nbDiscreteContactPairs[i][j] = 0;
				nbModifiedContactPairs[i][j] = 0;
				nbCCDPairs[i][j] = 0;
				nbTriggerPairs[i][j] = 0;
			}
		}

		for(PxU32 i=0; i < PxGeometryType::eGEOMETRY_COUNT; i++)
		{
			nbShapes[i] = 0;
		}
	}


	//
	// We advise to not access these members directly. Use the provided accessor methods instead.
	//
//broadphase:
	PxU32	nbBroadPhaseAdds;
	PxU32	nbBroadPhaseRemoves;

//collisions:
	PxU32   nbDiscreteContactPairs[PxGeometryType::eGEOMETRY_COUNT][PxGeometryType::eGEOMETRY_COUNT];
	PxU32   nbCCDPairs[PxGeometryType::eGEOMETRY_COUNT][PxGeometryType::eGEOMETRY_COUNT];
	PxU32   nbModifiedContactPairs[PxGeometryType::eGEOMETRY_COUNT][PxGeometryType::eGEOMETRY_COUNT];
	PxU32   nbTriggerPairs[PxGeometryType::eGEOMETRY_COUNT][PxGeometryType::eGEOMETRY_COUNT];
};

#if !PX_DOXYGEN
} // namespace physx
#endif

#endif
