// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef DY_AVBD_SOFT_BODY_STEP_H
#define DY_AVBD_SOFT_BODY_STEP_H

#include "avbd/contact/DyAvbdContact.h"
#include "avbd/contact/DyAvbdDetectionPlan.h"
#include "avbd/solver/soft/DyAvbdSoftBodyPrimal.h"
#include "avbd/solver/soft/DyAvbdSoftBodyPrimalPolicy.h"
#include "avbd/solver/soft/DyAvbdSoftBodyRuntime.h"
#include "avbd/solver/soft/DyAvbdSoftBodyScheduling.h"
#include "avbd/solver/soft/DyAvbdSoftBodyWorkspace.h"
#include "foundation/PxArray.h"
#include "foundation/PxTime.h"

#include <cstring>

namespace physx
{
namespace Dy
{

struct AvbdOGCParams;

// Scene/component bridge used to rebuild the current-pose contact set between
// nonlinear outer iterations. The callback owns contact discovery only; the
// step owner retains scheduling and solver-state ownership.
typedef void (*AvbdContactRedetectFn)(
	AvbdSoftParticle* particles, PxU32 numParticles,
	AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	PxArray<AvbdSoftContact>& contacts, void* userData);

// Per-step measurements consumed by validation and performance tooling. Keep
// these counters out of creation/runtime data so disabling diagnostics cannot
// affect the solver's data ownership or control flow.
struct AvbdSoftBodyStepStats
{
	PxF64 predictionMs;
	PxF64 contactIndexMs;
	PxF64 bodyPrecomputeMs;
	PxF64 bodySolveMs;
	PxF64 particleSolveMs;
	PxF64 projectionMs;
	PxF64 dualMs;
	PxF64 redetectMs;
	PxF64 velocityMs;
	PxF64 frictionMs;
	PxU64 requestedOuterIterations;
	PxU64 requestedInnerIterations;
	PxU64 executedOuterIterations;
	PxU64 executedInnerIterations;
	PxU64 particleSweeps;
	PxU64 groundTetPatchGroundPositionAlRows;
	PxU64 groundTetPatchFourSupportRows;
	PxU64 groundTetPatchSingleTetRows;
	PxU64 groundTetPatchActiveRows;
	PxU64 worldStaticVelocityTangentOwnerRows;
	PxU64 worldStaticVelocityTangentAppliedRows;
	PxU64 workspaceGrowthEvents;
	PxU64 workspaceGrowthBytes;
	PxU64 contactWorkspaceGrowthEvents;
	PxU64 contactWorkspaceGrowthBytes;
	PxU64 contactSweepScratchGrowthEvents;
	PxU64 contactSweepScratchGrowthBytes;
	PxU64 contactOutputGrowthEvents;
	PxU64 contactOutputGrowthBytes;
	PxU32 peakContactOutputCount;
	PxU32 peakContactOutputCapacity;
	PxU32 peakContactIncidenceCount;
	PxU32 peakContactIncidenceCapacity;
	PxU32 peakStateTransferContactCount;
	PxU32 peakStateTransferContactCapacity;
	PxU32 peakStateTransferUsedCapacity;
	PxU32 particlePrimalColorCount;
	PxU32 particlePrimalDynamicAccessGroupCount;
	PxU64 particlePrimalColoredSerialSweeps;
	PxU64 particlePrimalColoredSerialFallbackSweeps;
	PxU64 trustRegionLimitedParticleSteps;
	PxU64 positiveJLimitedParticleSteps;
	PxU64 positiveJRejectedParticleSteps;
	PxU64 nonFiniteRejectedParticleSteps;
	PxU64 tetLinearizationCacheFallbackParticleSteps;
	PxU64 legacyAppliedConvergedOuterIterations;
	PxU64 residualConvergedOuterIterations;
	PxU64 unsafeAppliedConvergenceCandidates;
	PxU64 budgetExhaustedOuterIterations;
	PxU64 shadowResidual1e5ConvergedOuterIterations;
	PxU64 shadowResidual1e5SavedInnerIterations;
	PxU64 shadowResidual1e4ConvergedOuterIterations;
	PxU64 shadowResidual1e4SavedInnerIterations;
	PxU64 particlePrimalCensusDynamicParticleSolves;
	PxU64 particlePrimalCensusTriangleEvaluations;
	PxU64 particlePrimalCensusCorotationalTetEvaluations;
	PxU64 particlePrimalCensusNeoHookeanTetEvaluations;
	PxU64 particlePrimalCensusBendingEvaluations;
	PxU64 particlePrimalCensusContactEvaluations;
	PxU64 particlePrimalCensusTetPacket8FullPackets;
	PxU64 particlePrimalCensusTetPacket8TailLanes;
	PxU64 particlePrimalTetPacketIrBodies;
	PxU64 particlePrimalTetPacketIrPackets;
	PxU64 particlePrimalTetPacketIrActiveLanes;
	PxU64 particlePrimalTetPacketIrTailLanes;
	PxU64 particlePrimalTetPacketIrActiveTailLanes;
	PxU64 particlePrimalTetPacketIrInvalidBodies;
	PxReal finalMaxLocalSolveDisplacement;
	PxReal finalMaxAppliedDisplacement;
	PxReal finalMaxDisplacement;

	AvbdSoftBodyStepStats()
		: predictionMs(0.0), contactIndexMs(0.0), bodyPrecomputeMs(0.0),
		  bodySolveMs(0.0), particleSolveMs(0.0), projectionMs(0.0),
		  dualMs(0.0), redetectMs(0.0), velocityMs(0.0), frictionMs(0.0),
		  requestedOuterIterations(0), requestedInnerIterations(0),
		  executedOuterIterations(0), executedInnerIterations(0),
		  particleSweeps(0), groundTetPatchGroundPositionAlRows(0),
		  groundTetPatchFourSupportRows(0),
		  groundTetPatchSingleTetRows(0), groundTetPatchActiveRows(0),
		  worldStaticVelocityTangentOwnerRows(0),
		  worldStaticVelocityTangentAppliedRows(0),
		  workspaceGrowthEvents(0), workspaceGrowthBytes(0),
		  contactWorkspaceGrowthEvents(0), contactWorkspaceGrowthBytes(0),
		  contactSweepScratchGrowthEvents(0),
		  contactSweepScratchGrowthBytes(0), contactOutputGrowthEvents(0),
		  contactOutputGrowthBytes(0), peakContactOutputCount(0),
		  peakContactOutputCapacity(0), peakContactIncidenceCount(0),
		  peakContactIncidenceCapacity(0), peakStateTransferContactCount(0),
		  peakStateTransferContactCapacity(0), peakStateTransferUsedCapacity(0),
		  particlePrimalColorCount(0),
		  particlePrimalDynamicAccessGroupCount(0),
		  particlePrimalColoredSerialSweeps(0),
		  particlePrimalColoredSerialFallbackSweeps(0),
		  trustRegionLimitedParticleSteps(0),
		  positiveJLimitedParticleSteps(0), positiveJRejectedParticleSteps(0),
		  nonFiniteRejectedParticleSteps(0),
		  tetLinearizationCacheFallbackParticleSteps(0),
		  legacyAppliedConvergedOuterIterations(0),
		  residualConvergedOuterIterations(0),
		  unsafeAppliedConvergenceCandidates(0),
		  budgetExhaustedOuterIterations(0),
		  shadowResidual1e5ConvergedOuterIterations(0),
		  shadowResidual1e5SavedInnerIterations(0),
		  shadowResidual1e4ConvergedOuterIterations(0),
		  shadowResidual1e4SavedInnerIterations(0),
		  particlePrimalCensusDynamicParticleSolves(0),
		  particlePrimalCensusTriangleEvaluations(0),
		  particlePrimalCensusCorotationalTetEvaluations(0),
		  particlePrimalCensusNeoHookeanTetEvaluations(0),
		  particlePrimalCensusBendingEvaluations(0),
		  particlePrimalCensusContactEvaluations(0),
		  particlePrimalCensusTetPacket8FullPackets(0),
		  particlePrimalCensusTetPacket8TailLanes(0),
		  particlePrimalTetPacketIrBodies(0),
		  particlePrimalTetPacketIrPackets(0),
		  particlePrimalTetPacketIrActiveLanes(0),
		  particlePrimalTetPacketIrTailLanes(0),
		  particlePrimalTetPacketIrActiveTailLanes(0),
		  particlePrimalTetPacketIrInvalidBodies(0),
		  finalMaxLocalSolveDisplacement(0.0f),
		  finalMaxAppliedDisplacement(0.0f), finalMaxDisplacement(0.0f)
	{
	}

	PX_FORCE_INLINE void reset()
	{
		std::memset(this, 0, sizeof(*this));
	}
};

enum class AvbdSoftBodyStepAdvanceResult : PxU8
{
	eREDETECTION_READY,
	eCAUSAL_LAYER_READY,
	eCOMPLETE,
	eINVALID
};

// Parent-owned resumable continuation for prediction, current-pose
// redetection, causal-layer fan-in and scalar completion.
struct AvbdSoftBodyStepState
{
	AvbdSoftBodyStepState();

	bool beginAfterPrediction(
		AvbdSoftParticle* inputParticles, PxU32 inputNumParticles,
		AvbdSoftBody* inputSoftBodies, PxU32 inputNumSoftBodies,
		AvbdSoftContact* inputContacts, PxU32 inputNumContacts,
		PxReal inputDt, PxU32 inputOuterIterations,
		PxU32 inputInnerIterations, PxU32 inputRequestedInnerBudget,
		PxReal inputAvbdBeta, AvbdContactRedetectFn inputRedetectFn,
		PxArray<AvbdSoftContact>* inputContactsArray,
		void* inputRedetectUserData, PxReal inputChebyshevRho,
		AvbdSoftBodyStepStats* inputStepStats,
		AvbdSoftBodyWorkspace& inputWorkspace,
		const AvbdSelfCollisionAdjacency* inputSelfCollisionAdjacencies,
		PxU32 inputNumSelfCollisionAdjacencies,
		const PxU8* inputSelfCollisionEnabled,
		const AvbdOGCParams* inputOgcParams,
		AvbdParticlePrimalSchedule inputParticlePrimalSchedule,
		bool inputDeferRedetectionToParent = false,
		bool inputPublishIndependentBodySweeps = false);

	AvbdSoftBodyStepAdvanceResult advance();
	bool completePendingRedetection();

	bool getPublishedCausalLayer(
		PxU32& layerIndex, PxU32& packedBegin, PxU32& packedEnd,
		const AvbdParticlePrimalSolveContext*& solveContext,
		const AvbdSoftBody*& bodies, PxU32& bodyCount,
		const PxU32*& particleBodyIndices,
		const PxU32*& packedParticleIndices) const;

	bool completePublishedCausalLayer(
		const AvbdParticlePrimalRangeObservation* observations,
		PxU32 observationCount);

	bool getPublishedIndependentBodySweep(
		const AvbdParticlePrimalSolveContext*& solveContext,
		const AvbdSoftBody*& bodies, PxU32& bodyCount) const;

	bool completePublishedIndependentBodySweep(
		const AvbdParticlePrimalRangeObservation* observations,
		PxU32 observationCount);

	void runToCompletionSerial();

	PX_FORCE_INLINE bool isComplete() const
	{
		return phase == Phase::eCOMPLETE;
	}

private:
	enum class Phase : PxU8
	{
		eIDLE,
		eOUTER_PREPARE,
		eINNER_BEGIN,
		eCAUSAL_LAYER,
		eDUAL,
		eREDETECTION,
		eCOMPLETE,
		eINVALID
	};

	void prepareOuterIteration();
	bool beginInnerSweep();
	void finishParticlePrimalSweep();
	void updateDualAndRedetect();
	void finishInitialRedetection();
	void rebuildParticleContactIndex();
	void finalizeStep();

	AvbdSoftParticle* particles;
	PxU32 numParticles;
	AvbdSoftBody* softBodies;
	PxU32 numSoftBodies;
	AvbdSoftContact* contacts;
	PxU32 numContacts;
	PxReal dt;
	PxReal invDt;
	PxReal invDtSq;
	PxU32 outerIterations;
	PxU32 requestedInnerIterationBudget;
	PxU32 remainingInnerIterationBudget;
	PxU32 outerIt;
	PxU32 currentInnerIterations;
	PxU32 innerIt;
	PxReal avbdBeta;
	AvbdContactRedetectFn redetectFn;
	PxArray<AvbdSoftContact>* contactsArray;
	void* redetectUserData;
	PxReal chebyshevRho;
	bool useChebyshev;
	PxReal chebyOmega;
	PxReal adaptiveRho;
	PxReal prevMaxDxSq;
	PxU32 shadowResidual1e5ConsecutiveSweeps;
	PxU32 shadowResidual1e4ConsecutiveSweeps;
	bool shadowResidual1e5Recorded;
	bool shadowResidual1e4Recorded;
	bool legacyAppliedConvergenceRecorded;
	PxU32 residualConsecutiveSweeps;
	AvbdSoftBodyStepStats* stepStats;
	AvbdSoftBodyWorkspace* workspace;
	const AvbdSelfCollisionAdjacency* selfCollisionAdjacencies;
	PxU32 numSelfCollisionAdjacencies;
	const PxU8* selfCollisionEnabled;
	const AvbdOGCParams* ogcParams;
	bool deferRedetectionToParent;
	bool pendingInitialRedetection;
	bool reuseComponentOgcSafetyEpoch;
	bool componentOgcSafetyEpochActive;
	bool componentOgcSafetyEpochLimited;
	bool publishIndependentBodySweeps;
	bool independentBodySweepPublished;
	AvbdParticlePrimalSchedule particlePrimalSchedule;
	bool validateParticlePrimalAccessPlan;
	AvbdParticlePrimalSolveContext particlePrimalSolveContext;
	AvbdParticlePrimalRangeObservation particlePrimalObservation;
	AvbdParticlePrimalCausalLayerState causalLayerState;
	PxTime stageTimer;
	Phase phase;
};

#if !defined(PX_PHYSX_STATIC_LIB) && PX_WINDOWS_FAMILY && \
	defined(DY_AVBD_SOFT_BODY_COMPONENT_EXPORTS)
	#define DY_AVBD_SOFT_BODY_STEP_API __declspec(dllexport)
#elif PX_UNIX_FAMILY
	#define DY_AVBD_SOFT_BODY_STEP_API PX_UNIX_EXPORT
#else
	#define DY_AVBD_SOFT_BODY_STEP_API
#endif

// Canonical scalar component step. Scene and task-graph code consume only
// this contract; the implementation has one LowLevelDynamics owner.
DY_AVBD_SOFT_BODY_STEP_API void avbdStepSoftBodies(
	AvbdSoftParticle* particles, PxU32 numParticles,
	AvbdSoftBody* softBodies, PxU32 numSoftBodies,
	AvbdSoftContact* contacts, PxU32 numContacts,
	PxReal dt, const PxVec3& gravity,
	PxU32 outerIterations = 1, PxU32 innerIterations = 10,
	PxReal avbdBeta = 1000.0f,
	AvbdContactRedetectFn redetectFn = NULL,
	PxArray<AvbdSoftContact>* contactsArray = NULL,
	void* redetectUserData = NULL,
	PxReal chebyshevRho = 0.92f,
	AvbdSoftBodyStepStats* stepStats = NULL,
	AvbdSoftBodyWorkspace* persistentWorkspace = NULL,
	PxU32 totalInnerIterationBudget = 0,
	const AvbdSelfCollisionAdjacency* selfCollisionAdjacencies = NULL,
	PxU32 numSelfCollisionAdjacencies = 0,
	const PxU8* selfCollisionEnabled = NULL,
	const AvbdOGCParams* ogcParams = NULL,
	AvbdSoftBodyStepExecutionMode executionMode =
		AvbdSoftBodyStepExecutionMode::eFULL,
	AvbdParticlePrimalSchedule inputParticlePrimalSchedule =
		AvbdParticlePrimalSchedule::eDEFAULT);

#undef DY_AVBD_SOFT_BODY_STEP_API

} // namespace Dy
} // namespace physx

#endif // DY_AVBD_SOFT_BODY_STEP_H
