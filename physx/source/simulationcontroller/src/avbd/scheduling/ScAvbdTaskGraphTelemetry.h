/*
 * AVBD standalone task graph telemetry ownership.
 *
 * This state is a Scene/task-boundary diagnostic and is intentionally kept
 * out of the solver context and contact IR.
 */
#pragma once

#include "foundation/PxSimpleTypes.h"
#include <atomic>

namespace physx {
namespace Sc {

struct AvbdStandaloneTaskGraphTelemetry
{
	// Source-local OGC counters deliberately live at the Scene task
	// boundary. Dy::AvbdDynamicsContext exposes compatibility hooks for
	// these sources, but those hooks are intentionally no-ops so they do
	// not add atomics to the solver context's hot path.
	struct TaskStageTelemetry
	{
		std::atomic<PxU32>	submittedTasks;
		std::atomic<PxU32>	completedTasks;
		std::atomic<PxU32>	activeTasks;
		std::atomic<PxU32>	peakActiveTasks;
		std::atomic<PxU32>	fanIns;
		std::atomic<PxU32>	serialFallbacks;

		TaskStageTelemetry()
		{
			reset();
		}

		static void atomicMax(
			std::atomic<PxU32>& target, PxU32 value)
		{
			PxU32 observed = target.load(std::memory_order_relaxed);
			while(observed < value && !target.compare_exchange_weak(
				observed, value, std::memory_order_relaxed,
				std::memory_order_relaxed))
			{
			}
		}

		void reset()
		{
			submittedTasks.store(0, std::memory_order_relaxed);
			completedTasks.store(0, std::memory_order_relaxed);
			activeTasks.store(0, std::memory_order_relaxed);
			peakActiveTasks.store(0, std::memory_order_relaxed);
			fanIns.store(0, std::memory_order_relaxed);
			serialFallbacks.store(0, std::memory_order_relaxed);
		}

		void recordTasksSubmitted(PxU32 taskCount)
		{
			submittedTasks.fetch_add(
				taskCount, std::memory_order_relaxed);
		}

		void beginTask()
		{
			const PxU32 active = activeTasks.fetch_add(
				1, std::memory_order_relaxed) + 1;
			atomicMax(peakActiveTasks, active);
		}

		void endTask()
		{
			activeTasks.fetch_sub(1, std::memory_order_relaxed);
			completedTasks.fetch_add(1, std::memory_order_relaxed);
		}

		void recordFanIn()
		{
			fanIns.fetch_add(1, std::memory_order_relaxed);
		}

		void recordSerialFallback()
		{
			serialFallbacks.fetch_add(1, std::memory_order_relaxed);
		}
	};

	std::atomic<PxU32>	requestedDispatcherWorkers;
	std::atomic<PxU32>	submittedSolveTasks;
	std::atomic<PxU32>	completedSolveTasks;
	std::atomic<PxU32>	activeSolveTasks;
	std::atomic<PxU32>	peakActiveSolveTasks;
	std::atomic<PxU32>	serialSolveTasks;
	std::atomic<PxU32>	pureSoftEligibleIslands;
	std::atomic<PxU32>	pureSoftEligibleParticles;
	std::atomic<PxU32>	submittedCausalLayerTasks;
	std::atomic<PxU32>	completedCausalLayerTasks;
	std::atomic<PxU32>	activeCausalLayerTasks;
	std::atomic<PxU32>	peakActiveCausalLayerTasks;
	std::atomic<PxU32>	causalLayerFanIns;
	std::atomic<PxU32>	serialCausalLayerFallbacks;
	std::atomic<PxU32>	maxCausalLayerOccupancy;
	std::atomic<PxU64>	totalCausalLayerOccupancy;
	TaskStageTelemetry	prediction;
	TaskStageTelemetry	writeBack;
	TaskStageTelemetry	worldPlaneContact;
	TaskStageTelemetry	rigidBoxSdfContact;
	TaskStageTelemetry	rigidSphereSdfContact;
	TaskStageTelemetry	rigidCapsuleSdfContact;
	TaskStageTelemetry	rigidConvexSdfContact;
	TaskStageTelemetry	rigidTriangleSurfaceContact;
	TaskStageTelemetry	selfBvhContact;

	AvbdStandaloneTaskGraphTelemetry()
	{
		reset();
	}

	static void atomicMax(std::atomic<PxU32>& target, PxU32 value)
	{
		PxU32 observed = target.load(std::memory_order_relaxed);
		while(observed < value && !target.compare_exchange_weak(
			observed, value, std::memory_order_relaxed,
			std::memory_order_relaxed))
		{
		}
	}

	void reset(PxU32 dispatcherWorkers = 0)
	{
		requestedDispatcherWorkers.store(
			dispatcherWorkers, std::memory_order_relaxed);
		submittedSolveTasks.store(0, std::memory_order_relaxed);
		completedSolveTasks.store(0, std::memory_order_relaxed);
		activeSolveTasks.store(0, std::memory_order_relaxed);
		peakActiveSolveTasks.store(0, std::memory_order_relaxed);
		serialSolveTasks.store(0, std::memory_order_relaxed);
		pureSoftEligibleIslands.store(0, std::memory_order_relaxed);
		pureSoftEligibleParticles.store(0, std::memory_order_relaxed);
		submittedCausalLayerTasks.store(0, std::memory_order_relaxed);
		completedCausalLayerTasks.store(0, std::memory_order_relaxed);
		activeCausalLayerTasks.store(0, std::memory_order_relaxed);
		peakActiveCausalLayerTasks.store(0, std::memory_order_relaxed);
		causalLayerFanIns.store(0, std::memory_order_relaxed);
		serialCausalLayerFallbacks.store(0, std::memory_order_relaxed);
		maxCausalLayerOccupancy.store(0, std::memory_order_relaxed);
		totalCausalLayerOccupancy.store(0, std::memory_order_relaxed);
		prediction.reset();
		writeBack.reset();
		worldPlaneContact.reset();
		rigidBoxSdfContact.reset();
		rigidSphereSdfContact.reset();
		rigidCapsuleSdfContact.reset();
		rigidConvexSdfContact.reset();
		rigidTriangleSurfaceContact.reset();
		selfBvhContact.reset();
	}

	void recordSolveSubmission(
		PxU32 dispatcherWorkers, PxU32 particleCount)
	{
		requestedDispatcherWorkers.store(
			dispatcherWorkers, std::memory_order_relaxed);
		submittedSolveTasks.fetch_add(1, std::memory_order_relaxed);
		pureSoftEligibleIslands.fetch_add(1, std::memory_order_relaxed);
		pureSoftEligibleParticles.fetch_add(
			particleCount, std::memory_order_relaxed);
		// The Scene top-level task is the only solve owner.  Counting it at
		// submission also covers the interval while it is queued; child
		// primal tasks have their own active counter below.
		const PxU32 active = activeSolveTasks.fetch_add(
			1, std::memory_order_relaxed) + 1;
		atomicMax(peakActiveSolveTasks, active);
	}

	void recordSerialSolve(PxU32 dispatcherWorkers, PxU32 particleCount)
	{
		requestedDispatcherWorkers.store(
			dispatcherWorkers, std::memory_order_relaxed);
		serialSolveTasks.fetch_add(1, std::memory_order_relaxed);
		pureSoftEligibleIslands.fetch_add(1, std::memory_order_relaxed);
		pureSoftEligibleParticles.fetch_add(
			particleCount, std::memory_order_relaxed);
	}

	void beginSolveTask()
	{
		const PxU32 active = activeSolveTasks.fetch_add(
			1, std::memory_order_relaxed) + 1;
		atomicMax(peakActiveSolveTasks, active);
	}

	void endSolveTask()
	{
		PxU32 active = activeSolveTasks.load(
			std::memory_order_relaxed);
		while(active && !activeSolveTasks.compare_exchange_weak(
			active, active - 1, std::memory_order_relaxed,
			std::memory_order_relaxed))
		{
		}
		PxU32 completed = completedSolveTasks.load(
			std::memory_order_relaxed);
		const PxU32 submitted = submittedSolveTasks.load(
			std::memory_order_relaxed);
		while(completed < submitted && !completedSolveTasks.
			compare_exchange_weak(completed, completed + 1,
				std::memory_order_relaxed,
				std::memory_order_relaxed))
		{
		}
	}

	void recordCausalLayerTasksSubmitted(
		PxU32 taskCount, PxU32 occupancy)
	{
		submittedCausalLayerTasks.fetch_add(
			taskCount, std::memory_order_relaxed);
		atomicMax(maxCausalLayerOccupancy, occupancy);
		totalCausalLayerOccupancy.fetch_add(
			occupancy, std::memory_order_relaxed);
	}

	void beginCausalLayerTask()
	{
		const PxU32 active = activeCausalLayerTasks.fetch_add(
			1, std::memory_order_relaxed) + 1;
		atomicMax(peakActiveCausalLayerTasks, active);
	}

	void endCausalLayerTask()
	{
		activeCausalLayerTasks.fetch_sub(1, std::memory_order_relaxed);
		completedCausalLayerTasks.fetch_add(1, std::memory_order_relaxed);
	}

	void recordCausalLayerFanIn()
	{
		causalLayerFanIns.fetch_add(1, std::memory_order_relaxed);
	}

	void recordPredictionTasksSubmitted(PxU32 taskCount)
	{
		prediction.recordTasksSubmitted(taskCount);
	}

	void beginPredictionTask()
	{
		prediction.beginTask();
	}

	void endPredictionTask()
	{
		prediction.endTask();
	}

	void recordSerialPredictionStage()
	{
		prediction.recordSerialFallback();
	}

	void recordWriteBackTasksSubmitted(PxU32 taskCount)
	{
		writeBack.recordTasksSubmitted(taskCount);
	}

	void beginWriteBackTask()
	{
		writeBack.beginTask();
	}

	void endWriteBackTask()
	{
		writeBack.endTask();
	}

	void recordSerialWriteBackStage()
	{
		writeBack.recordSerialFallback();
	}

	void recordSerialCausalLayerFallback()
	{
		serialCausalLayerFallbacks.fetch_add(
			1, std::memory_order_relaxed);
	}

	void recordWorldPlaneContactTasksSubmitted(PxU32 taskCount)
	{
		worldPlaneContact.recordTasksSubmitted(taskCount);
	}

	void beginWorldPlaneContactTask()
	{
		worldPlaneContact.beginTask();
	}

	void endWorldPlaneContactTask()
	{
		worldPlaneContact.endTask();
	}

	void recordWorldPlaneContactFanIn()
	{
		worldPlaneContact.recordFanIn();
	}

	void recordSerialWorldPlaneContactFallback()
	{
		worldPlaneContact.recordSerialFallback();
	}

	void recordRigidBoxSdfContactTasksSubmitted(PxU32 taskCount)
	{
		rigidBoxSdfContact.recordTasksSubmitted(taskCount);
	}

	void beginRigidBoxSdfContactTask()
	{
		rigidBoxSdfContact.beginTask();
	}

	void endRigidBoxSdfContactTask()
	{
		rigidBoxSdfContact.endTask();
	}

	void recordRigidBoxSdfContactFanIn()
	{
		rigidBoxSdfContact.recordFanIn();
	}

	void recordSerialRigidBoxSdfContactFallback()
	{
		rigidBoxSdfContact.recordSerialFallback();
	}

	void recordRigidSphereSdfContactTasksSubmitted(PxU32 taskCount)
	{
		rigidSphereSdfContact.recordTasksSubmitted(taskCount);
	}

	void beginRigidSphereSdfContactTask()
	{
		rigidSphereSdfContact.beginTask();
	}

	void endRigidSphereSdfContactTask()
	{
		rigidSphereSdfContact.endTask();
	}

	void recordRigidSphereSdfContactFanIn()
	{
		rigidSphereSdfContact.recordFanIn();
	}

	void recordSerialRigidSphereSdfContactFallback()
	{
		rigidSphereSdfContact.recordSerialFallback();
	}

	void recordRigidCapsuleSdfContactTasksSubmitted(PxU32 taskCount)
	{
		rigidCapsuleSdfContact.recordTasksSubmitted(taskCount);
	}

	void beginRigidCapsuleSdfContactTask()
	{
		rigidCapsuleSdfContact.beginTask();
	}

	void endRigidCapsuleSdfContactTask()
	{
		rigidCapsuleSdfContact.endTask();
	}

	void recordRigidCapsuleSdfContactFanIn()
	{
		rigidCapsuleSdfContact.recordFanIn();
	}

	void recordSerialRigidCapsuleSdfContactFallback()
	{
		rigidCapsuleSdfContact.recordSerialFallback();
	}

	void recordRigidConvexSdfContactTasksSubmitted(PxU32 taskCount)
	{
		rigidConvexSdfContact.recordTasksSubmitted(taskCount);
	}

	void beginRigidConvexSdfContactTask()
	{
		rigidConvexSdfContact.beginTask();
	}

	void endRigidConvexSdfContactTask()
	{
		rigidConvexSdfContact.endTask();
	}

	void recordRigidConvexSdfContactFanIn()
	{
		rigidConvexSdfContact.recordFanIn();
	}

	void recordSerialRigidConvexSdfContactFallback()
	{
		rigidConvexSdfContact.recordSerialFallback();
	}

	void recordRigidTriangleSurfaceContactTasksSubmitted(PxU32 taskCount)
	{
		rigidTriangleSurfaceContact.recordTasksSubmitted(taskCount);
	}

	void beginRigidTriangleSurfaceContactTask()
	{
		rigidTriangleSurfaceContact.beginTask();
	}

	void endRigidTriangleSurfaceContactTask()
	{
		rigidTriangleSurfaceContact.endTask();
	}

	void recordRigidTriangleSurfaceContactFanIn()
	{
		rigidTriangleSurfaceContact.recordFanIn();
	}

	void recordSerialRigidTriangleSurfaceContactFallback()
	{
		rigidTriangleSurfaceContact.recordSerialFallback();
	}

	void recordSelfBvhContactTasksSubmitted(PxU32 taskCount)
	{
		selfBvhContact.recordTasksSubmitted(taskCount);
	}

	void beginSelfBvhContactTask()
	{
		selfBvhContact.beginTask();
	}

	void endSelfBvhContactTask()
	{
		selfBvhContact.endTask();
	}

	void recordSelfBvhContactFanIn()
	{
		selfBvhContact.recordFanIn();
	}

	void recordSerialSelfBvhContactFallback()
	{
		selfBvhContact.recordSerialFallback();
	}
};

} // namespace Sc
} // namespace physx
