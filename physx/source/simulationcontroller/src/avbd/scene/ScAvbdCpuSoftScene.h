// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SC_AVBD_CPU_SOFT_SCENE_H
#define SC_AVBD_CPU_SOFT_SCENE_H

#include "ScScene.h"
#include "ScConstraintCore.h"
#include "ScArticulationJointCore.h"
#include "ScArticulationTendonCore.h"
#include "ScArticulationMimicJointCore.h"
#include "ScArticulationSim.h"
#include "ScArticulationTendonSim.h"
#include "ScArticulationMimicJointSim.h"
#include "ScDeformableSurfaceCore.h"
#include "ScDeformableVolumeCore.h"
#include "ScBodyCore.h"
#include "ScBodySim.h"
#include "ScStaticCore.h"
#include "ScShapeCore.h"
#include "ScSimStats.h"
#include "ScSimulationController.h"
#include "ScSqBoundsManager.h"
#include "ScArticulationCore.h"
#include "ScShapeInteraction.h"
#include "avbd/selection/ScAvbdIslandSelectionStorage.h"
#include "avbd/contact/ScAvbdCollisionProxyWorkspace.h"
#include "avbd/scheduling/ScAvbdTaskGraphTelemetry.h"
#include "avbd/contact/ScAvbdContactDetectionView.h"
#include "DyIslandManager.h"
#include "avbd/pipeline/DyAvbdDynamics.h"
#include "avbd/backend/gpu/DyAvbdGpuWaveBackend.h"
#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"
#include "DyDeformableSurface.h"
#include "DyDeformableVolume.h"
#include "foundation/PxHashMap.h"
#include "foundation/PxTime.h"
#include "geometry/PxHeightField.h"
#include "geometry/PxHeightFieldGeometry.h"
#include "geometry/PxMeshQuery.h"
#include "geometry/PxTriangle.h"
#include "geometry/PxTriangleMesh.h"
#include "GuTetrahedronMesh.h"
#include "PxsDeformableSurfaceMaterialCore.h"
#include "PxsMaterialCore.h"
#include "PxsMemoryManager.h"
#include "avbd/scene/ScAvbdSceneEntries.h"
#include "avbd/scene/ScAvbdSceneStateTypes.h"
#include "avbd/scheduling/ScAvbdSchedulingPolicy.h"
#include "avbd/selection/ScAvbdIslandSelectionPlan.h"
#include "avbd/selection/ScAvbdOgcPairPlan.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace physx
{
namespace Sc
{

class AvbdCpuSoftScene :
	public PxUserAllocated,
	public Dy::AvbdSoftIslandProvider
{
	typedef AvbdIslandSelectionStorage IslandSelectionStorage;

	class WriteBackTask;
	class PredictionTask;
	class CausalLayerTask;
	class CausalLayerFinishTask;
	class WorldPlaneContactTask;
	class WorldPlaneContactFinishTask;
	class RigidBoxSdfContactTask;
	class RigidBoxSdfContactFinishTask;
	class RigidSphereSdfContactTask;
	class RigidSphereSdfContactFinishTask;
	class RigidCapsuleSdfContactTask;
	class RigidCapsuleSdfContactFinishTask;
	class RigidConvexSdfContactTask;
	class RigidConvexSdfContactFinishTask;
	class RigidTriangleSurfaceContactTask;
	class RigidTriangleSurfaceContactFinishTask;
	class SelfBvhContactTask;
	class SelfBvhContactFinishTask;
	class StaticWorldSelfOgcContactTask;
	class StaticWorldSelfOgcContactFinishTask;

public:
	AvbdCpuSoftScene(
		const PxsDeformableVolumeMaterialManager& deformableMaterialManager,
		const PxsDeformableSurfaceMaterialManager& surfaceMaterialManager,
		const PxsMaterialManager& rigidMaterialManager,
		PxU64 contextId,
		IG::SimpleIslandManager& islandManager);

	~AvbdCpuSoftScene();

	bool add(
		DeformableVolumeCore& core,
		PxTetrahedronMesh& simulationMesh,
		PxTetrahedronMesh& collisionMesh,
		PxDeformableVolumeAuxData& auxData,
		const PxsDeformableVolumeMaterialManager& materialManager);

	bool addSurface(
		DeformableSurfaceCore& core,
		PxTriangleMesh& triangleMesh);

	void addStaticShape(StaticCore& core, ShapeCore& shape);

	void removeStaticShape(
		StaticCore& core, const ShapeCore& shape);

	void removeStatic(StaticCore& core);

	void addDynamicShape(BodyCore& core, ShapeCore& shape);

	void removeDynamicShape(
		BodyCore& core, const ShapeCore& shape);

	void removeDynamic(BodyCore& core);

	void remove(DeformableVolumeCore& core);

	void removeSurface(DeformableSurfaceCore& core);

	bool buildLocalElementPoint(
		ActorCore& core,
		bool surfaceElement,
		PxU32 elementIndex,
		const PxVec4& barycentric,
		Dy::AvbdSoftPoint& point);

	PxU32 addWorldPin(
		ActorCore& core,
		PxU32 localVertex,
		const PxVec3& worldTarget);

	PxU32 addWorldPin(
		ActorCore& core,
		const Dy::AvbdSoftPoint& localPoint,
		const PxVec3& worldTarget);

	PxU32 addWorldElementPin(
		ActorCore& core,
		bool surfaceElement,
		PxU32 elementIndex,
		const PxVec4& barycentric,
		const PxVec3& worldTarget);

	bool updateWorldPin(
		ActorCore& core,
		PxU32 handle,
		const PxVec3& worldTarget);

	void removeWorldPin(
		ActorCore& core,
		PxU32 handle);

	bool computePrescribedAttachmentWorldTarget(
		RigidCore& prescribedCore,
		const PxVec3& actorLocalTarget,
		PxVec3& worldTarget) const;

	PxU32 addKinematicAttachment(
		ActorCore& softCore,
		BodyCore& kinematicCore,
		PxU32 localVertex,
		const PxVec3& actorLocalTarget);

	PxU32 addPrescribedAttachment(
		ActorCore& softCore,
		RigidCore& prescribedCore,
		const Dy::AvbdSoftPoint& localPoint,
		const PxVec3& actorLocalTarget);

	PxU32 addKinematicElementAttachment(
		ActorCore& softCore,
		BodyCore& kinematicCore,
		bool surfaceElement,
		PxU32 elementIndex,
		const PxVec4& barycentric,
		const PxVec3& actorLocalTarget);

	PxU32 addStaticAttachment(
		ActorCore& softCore,
		StaticCore& staticCore,
		PxU32 localVertex,
		const PxVec3& actorLocalTarget);

	PxU32 addStaticElementAttachment(
		ActorCore& softCore,
		StaticCore& staticCore,
		bool surfaceElement,
		PxU32 elementIndex,
		const PxVec4& barycentric,
		const PxVec3& actorLocalTarget);

	bool updatePrescribedAttachment(
		ActorCore& softCore,
		PxU32 handle,
		const PxVec3& actorLocalTarget);

	void removePrescribedAttachment(
		ActorCore& softCore,
		PxU32 handle);

	PxU32 addRigidAttachment(
		ActorCore& softCore,
		BodyCore& rigidCore,
		PxU32 localVertex,
		const PxVec3& actorLocalTarget);

	PxU32 addRigidAttachment(
		ActorCore& softCore,
		BodyCore& rigidCore,
		const Dy::AvbdSoftPoint& localPoint,
		const PxVec3& actorLocalTarget);

	PxU32 addRigidElementAttachment(
		ActorCore& softCore,
		BodyCore& rigidCore,
		bool surfaceElement,
		PxU32 elementIndex,
		const PxVec4& barycentric,
		const PxVec3& actorLocalTarget);

	bool updateRigidAttachment(
		ActorCore& softCore,
		PxU32 handle,
		const PxVec3& actorLocalTarget);

	void removeRigidAttachment(
		ActorCore& softCore,
		PxU32 handle);

	PxU32 addArticulationAttachment(
		ActorCore& softCore,
		BodyCore& linkCore,
		PxU32 localVertex,
		const PxVec3& actorLocalTarget);

	PxU32 addArticulationAttachment(
		ActorCore& softCore,
		BodyCore& linkCore,
		const Dy::AvbdSoftPoint& localPoint,
		const PxVec3& actorLocalTarget);

	PxU32 addArticulationElementAttachment(
		ActorCore& softCore,
		BodyCore& linkCore,
		bool surfaceElement,
		PxU32 elementIndex,
		const PxVec4& barycentric,
		const PxVec3& actorLocalTarget);

	bool updateArticulationAttachment(
		ActorCore& softCore,
		PxU32 handle,
		const PxVec3& actorLocalTarget);

	void removeArticulationAttachment(
		ActorCore& softCore,
		PxU32 handle);

	PxU32 addSoftPairAttachment(
		ActorCore& softCore0,
		const Dy::AvbdSoftPoint& localPoint0,
		ActorCore& softCore1,
		const Dy::AvbdSoftPoint& localPoint1);

	PxU32 addSoftPairAttachment(
		ActorCore& softCore0,
		bool element0,
		PxU32 index0,
		const PxVec4& barycentric0,
		ActorCore& softCore1,
		bool element1,
		PxU32 index1,
		const PxVec4& barycentric1);

	void removeSoftPairAttachment(
		ActorCore& softCore,
		PxU32 handle);

	PxU32 addRigidActorFilter(
		ActorCore& softCore,
		ActorCore& rigidCore,
		const PxU32* elementIndices = NULL,
		PxU32 elementCount = 0,
		bool filterAllElements = true);

	PxU32 addVolumeRigidActorFilter(
		DeformableVolumeCore& softCore,
		ActorCore& rigidCore,
		const PxU32* collisionElementIndices,
		PxU32 collisionElementCount,
		bool filterAllElements);

	void removeRigidActorFilter(
		ActorCore& softCore,
		PxU32 handle);

	PxU32 addCompiledDeformablePairFilter(
		ActorCore& core0,
		ActorCore& core1,
		const PxU32* elementIndices0,
		const PxU32* elementIndices1,
		PxU32 pairCount);

	bool expandVolumeCollisionElement(
		const Entry& entry,
		PxU32 collisionElement,
		PxArray<PxU32>& simulationElements) const;

	PxU32 addSurfaceSurfaceFilter(
		DeformableSurfaceCore& core0,
		DeformableSurfaceCore& core1,
		const PxU32* elementIndices0,
		const PxU32* elementIndices1,
		PxU32 pairCount);

	PxU32 addVolumeSurfaceFilter(
		DeformableVolumeCore& volumeCore,
		DeformableSurfaceCore& surfaceCore,
		const PxU32* volumeCollisionElements,
		const PxU32* surfaceElements,
		PxU32 pairCount);

	PxU32 addVolumeVolumeFilter(
		DeformableVolumeCore& core0,
		DeformableVolumeCore& core1,
		const PxU32* collisionElements0,
		const PxU32* collisionElements1,
		PxU32 pairCount);

	void removeDeformablePairFilter(
		ActorCore& core,
		PxU32 handle);

	void removeEntry(ActorCore& core);

	void prepareIslandGeneration(
		PxReal dt, const PxVec3& gravity, bool sleepingEnabled);

	virtual bool prepareSoftIslandSelections(
		Dy::AvbdSolverBody* solverBodies,
		PxsRigidBody* const* rigidBodies,
		Dy::FeatherstoneArticulation* const* articulationForBody,
		const PxU32* linkIndexForBody,
		const PxU32* islandBodyStarts,
		const PxU32* islandBodyCounts,
		const PxU32* activeIslandIds,
		PxU32 islandCount,
		PxReal dt,
		const PxVec3& gravity,
		PxArray<Dy::AvbdSoftIslandSelection>& selections) PX_OVERRIDE;

		// P5 collision-leaf tasks consume the direct simulation topology, while
		// the ordinary component lifecycle expands a cooked collision mesh back
		// to simulation-space Jacobians synchronously.  Keep that narrower P5
		// capability separate from P2/P3/P4/P6 scheduling: a distinct collision
		// mesh must not demote prediction, body-local primal work, or write-back
		// to the scalar reference path.
		bool hasDirectSimulationCollisionDomain() const;

		bool shouldScheduleStandaloneTaskGraph(
			PxU32 dispatcherWorkers) const;

		// Resolve the component particle schedule at the Scene boundary, where
		// worker availability and the public determinism promise are known.  The
		// default production route is relaxed coloring on a useful parallel
		// workload; an explicit process policy remains available for diagnostics
		// and forced fallback.  Enhanced determinism is a contract, not a tuning
		// hint, so it always retains the ordered scalar authority.
		Dy::AvbdParticlePrimalSchedule getParticlePrimalSchedule() const;

		// A dense persistent soft/soft manifold has many mutually dependent
		// vertices and therefore produces many short color layers. Publishing
		// every one as a dispatcher task turns the layer barriers into the
		// dominant cost (especially with two workers). Keep the identical relaxed
		// color plan, but execute its ordered layers inline in the component
		// continuation once the previous epoch proves that this is a contact-dense
		// manifold. This follows the same batching principle used by CPU XPBD
		// solvers: parallelize useful independent batches, not synchronization
		// points between tiny constraint colors.
		bool shouldInlineDenseSoftPairColoredPrimal() const;

		PxU32 getStandaloneTaskGraphParticleCount() const;

		void setStandaloneTaskGraphExecutionPolicy(
			PxU32 workerCount, bool enhancedDeterminism);

		bool canUseIndependentBodySweepTaskFanIn() const;

		// P3 prediction setup: all mutable Scene/OGC state up to the exact
		// low-level prediction boundary is complete when this returns. The
		// resulting particle write set is disjoint by whole Entry.
		bool prepareStandaloneComponentSolve(
			PxReal dt,
			const PxVec3& gravity,
			const PxsDeformableVolumeMaterialManager& materialManager,
			const PxsMaterialManager& rigidMaterialManager,
			bool sleepingEnabled);

		void predictStandaloneComponentRange(
			PxU32 entryBegin, PxU32 entryEnd,
			PxReal dt, const PxVec3& gravity);

		void predictStandaloneComponent(
			PxReal dt, const PxVec3& gravity);

		PxU32 getStandalonePredictionTaskCount(
			PxU32 dispatcherWorkers) const;

		void submitStandalonePredictionTasks(
			PxU32 taskCount, PxReal dt, const PxVec3& gravity,
			PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext);

		// The state machine publishes one causal layer at a time. P4.5.3a
		// deliberately submits one whole-layer child as the taskgraph reference;
		// P4.5.3b may partition the same stable packed interval, but only under
		// its explicit validation switch.
		bool ensureCausalLayerTaskPool(
			PxU32 requiredChildTasks, Scene& owner,
			Dy::AvbdDynamicsContext& taskGraphContext);

		CausalLayerTask* acquireCausalLayerTask();

		CausalLayerFinishTask* acquireCausalLayerFinishTask();

		void recycleCausalLayerTask(PxU32 index);

		void recycleCausalLayerFinishTask(PxU32 index);

		bool hasCausalLayerTaskSlots(PxU32 taskCount);

		PxU32 getCausalLayerTaskCount(
			PxU32 dispatcherWorkers, PxU32 layerOccupancy) const;

		static PX_FORCE_INLINE PxU64 getIndependentBodySweepWorkEstimate(
			const Dy::AvbdSoftBody& body)
		{
			const Dy::AvbdSoftBodyCompiledData& compiled = body.compiled;
			const PxU64 work = PxU64(compiled.particleCount) +
				PxU64(compiled.triElements.size()) * 3u +
				PxU64(compiled.tetElements.size()) * 4u +
				PxU64(compiled.bendElements.size()) * 4u;
			return work ? work : 1u;
		}

		static PX_FORCE_INLINE PxU64 getIndependentBodySweepTarget(
			PxU64 totalWork, PxU32 boundaryIndex, PxU32 taskCount)
		{
			// Compute floor(totalWork * boundaryIndex / taskCount) without
			// overflowing the totalWork product.
			const PxU64 quotient = totalWork / taskCount;
			const PxU64 remainder = totalWork % taskCount;
			return quotient * boundaryIndex +
				(remainder * boundaryIndex) / taskCount;
		}

		static PX_FORCE_INLINE PxU64 getIndependentBodySweepDistance(
			PxU64 prefixWork, PxU64 targetWork)
		{
			return prefixWork > targetWork ? prefixWork - targetWork :
				targetWork - prefixWork;
		}

		bool submitStandaloneCausalLayerTask(
			PxU32 dispatcherWorkers, Scene& owner,
			PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext);

		bool canUseWorldPlaneContactTaskTransaction() const;

		PxU32 getWorldPlaneContactTaskCount(PxU32 dispatcherWorkers) const;

		bool beginWorldPlaneContactTaskTransaction();

		void completeWorldPlaneContactTaskTransaction();

		bool ensureWorldPlaneContactTaskPool(PxU32 requiredChildTasks,
			Scene& owner);

		WorldPlaneContactTask* acquireWorldPlaneContactTask();

		WorldPlaneContactFinishTask* acquireWorldPlaneContactFinishTask();

		void recycleWorldPlaneContactTask(PxU32 index);

		void recycleWorldPlaneContactFinishTask(PxU32 index);

		bool submitStandaloneWorldPlaneContactTask(
			PxU32 dispatcherWorkers, Scene& owner,
			PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext);

		Dy::AvbdSoftBodyStepAdvanceResult
			advanceStandaloneComponentStateWithSceneRedetection(
				bool allowWorldPlaneTask = false,
				bool* worldPlaneContactTaskReady = NULL,
				bool allowRigidBoxSdfTask = false,
				bool* rigidBoxSdfContactTaskReady = NULL,
				bool allowRigidSphereSdfTask = false,
				bool* rigidSphereSdfContactTaskReady = NULL);

		bool runStandaloneComponentStateWithSceneRedetection();

		bool completeStandaloneWorldPlaneContactTask(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady,
			bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady);

		bool finishStandaloneWorldPlaneContactSerialFallback(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady,
			bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady);

		bool canUseRigidBoxSdfContactTaskTransaction() const;

		PxU32 getRigidBoxSdfContactTaskCount(PxU32 dispatcherWorkers) const;

		bool beginRigidBoxSdfContactTaskTransaction();

		void completeRigidBoxSdfContactTaskTransaction();

		bool ensureRigidBoxSdfContactTaskPool(PxU32 requiredChildTasks, Scene& owner);

		RigidBoxSdfContactTask* acquireRigidBoxSdfContactTask();

		RigidBoxSdfContactFinishTask* acquireRigidBoxSdfContactFinishTask();

		void recycleRigidBoxSdfContactTask(PxU32 index);

		void recycleRigidBoxSdfContactFinishTask(PxU32 index);

		bool submitStandaloneRigidBoxSdfContactTask(
			PxU32 dispatcherWorkers, Scene& owner,
			PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext);

		bool completeStandaloneRigidBoxSdfContactTask(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady);

		bool finishStandaloneRigidBoxSdfContactSerialFallback(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady);

		bool canUseRigidSphereSdfContactTaskTransaction() const;

		PxU32 getRigidSphereSdfContactTaskCount(PxU32 dispatcherWorkers) const;

		bool beginRigidSphereSdfContactTaskTransaction();

		void completeRigidSphereSdfContactTaskTransaction();

		bool ensureRigidSphereSdfContactTaskPool(PxU32 requiredChildTasks, Scene& owner);

		RigidSphereSdfContactTask* acquireRigidSphereSdfContactTask();

		RigidSphereSdfContactFinishTask* acquireRigidSphereSdfContactFinishTask();

		void recycleRigidSphereSdfContactTask(PxU32 index);

		void recycleRigidSphereSdfContactFinishTask(PxU32 index);

		bool submitStandaloneRigidSphereSdfContactTask(
			PxU32 dispatcherWorkers, Scene& owner,
			PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext);

		bool completeStandaloneRigidSphereSdfContactTask(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady);

		bool finishStandaloneRigidSphereSdfContactSerialFallback(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady);

		bool canUseRigidCapsuleSdfContactTaskTransaction() const;

		PxU32 getRigidCapsuleSdfContactTaskCount(PxU32 dispatcherWorkers) const;

		bool beginRigidCapsuleSdfContactTaskTransaction();

		void completeRigidCapsuleSdfContactTaskTransaction();

		bool ensureRigidCapsuleSdfContactTaskPool(PxU32 requiredChildTasks, Scene& owner);

		RigidCapsuleSdfContactTask* acquireRigidCapsuleSdfContactTask();

		RigidCapsuleSdfContactFinishTask* acquireRigidCapsuleSdfContactFinishTask();

		void recycleRigidCapsuleSdfContactTask(PxU32 index);

		void recycleRigidCapsuleSdfContactFinishTask(PxU32 index);

		bool submitStandaloneRigidCapsuleSdfContactTask(
			PxU32 dispatcherWorkers, Scene& owner,
			PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext);

		bool completeStandaloneRigidCapsuleSdfContactTask(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady);

		bool finishStandaloneRigidCapsuleSdfContactSerialFallback(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady);

		bool canUseRigidConvexSdfContactTaskTransaction() const;

		PxU32 getRigidConvexSdfContactTaskCount(PxU32 dispatcherWorkers) const;

		bool beginRigidConvexSdfContactTaskTransaction();

		void completeRigidConvexSdfContactTaskTransaction();

		bool ensureRigidConvexSdfContactTaskPool(PxU32 requiredChildTasks, Scene& owner);

		RigidConvexSdfContactTask* acquireRigidConvexSdfContactTask();

		RigidConvexSdfContactFinishTask* acquireRigidConvexSdfContactFinishTask();

		void recycleRigidConvexSdfContactTask(PxU32 index);

		void recycleRigidConvexSdfContactFinishTask(PxU32 index);

		bool submitStandaloneRigidConvexSdfContactTask(
			PxU32 dispatcherWorkers, Scene& owner,
			PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext);

		bool completeStandaloneRigidConvexSdfContactTask(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady);

		bool finishStandaloneRigidConvexSdfContactSerialFallback(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady);

		bool canUseRigidTriangleSurfaceContactTaskTransaction() const;

		PxU32 getRigidTriangleSurfaceContactTaskCount(
			PxU32 dispatcherWorkers) const;

		bool beginRigidTriangleSurfaceContactTaskTransaction();

		void completeRigidTriangleSurfaceContactTaskTransaction();

		bool ensureRigidTriangleSurfaceContactTaskPool(
			PxU32 requiredChildTasks, Scene& owner,
			Dy::AvbdDynamicsContext& taskGraphContext);

		RigidTriangleSurfaceContactTask* acquireRigidTriangleSurfaceContactTask();

		RigidTriangleSurfaceContactFinishTask* acquireRigidTriangleSurfaceContactFinishTask();

		void recycleRigidTriangleSurfaceContactTask(PxU32 index);

		void recycleRigidTriangleSurfaceContactFinishTask(PxU32 index);

		bool submitStandaloneRigidTriangleSurfaceContactTask(
			PxU32 dispatcherWorkers, Scene& owner,
			PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext);

		bool completeStandaloneRigidTriangleSurfaceContactTask(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady);

		bool finishStandaloneRigidTriangleSurfaceContactSerialFallback(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady);

		bool canUseSelfBvhContactTaskTransaction() const;

		PxU32 getSelfBvhContactTaskCount(PxU32 dispatcherWorkers) const;

		bool beginSelfBvhContactTaskTransaction();

		void completeSelfBvhContactTaskTransaction();

		bool ensureSelfBvhContactTaskPool(PxU32 requiredChildTasks, Scene& owner);

		SelfBvhContactTask* acquireSelfBvhContactTask();

		SelfBvhContactFinishTask* acquireSelfBvhContactFinishTask();

		void recycleSelfBvhContactTask(PxU32 index);

		void recycleSelfBvhContactFinishTask(PxU32 index);

		bool submitStandaloneSelfBvhContactTask(
			PxU32 dispatcherWorkers, Scene& owner,
			PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext);

		bool completeStandaloneSelfBvhContactTask(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady);

		bool finishStandaloneSelfBvhContactSerialFallback(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady);

		bool canUseStaticWorldSelfOgcContactTaskTransaction() const;

		// Keep the ownership decision stable for every outer epoch of the
		// resumable solve.  The aggregate is a Scene-owned redetection bridge in
		// exactly the same sense as the legacy P5 leaves; testing only the raw
		// environment bridge here would run its first epoch in parallel and then
		// silently fall back to the synchronous callback.
		bool usesStandaloneSceneRedetectionBridge() const;

		PxU32 getStaticWorldSelfOgcContactTaskCount(
			PxU32 dispatcherWorkers) const;

		bool beginStaticWorldSelfOgcContactTaskTransaction();

		void completeStaticWorldSelfOgcContactTaskTransaction();

		bool ensureStaticWorldSelfOgcContactTaskPool(
			PxU32 requiredChildTasks, Scene& owner);

		StaticWorldSelfOgcContactTask* acquireStaticWorldSelfOgcContactTask();

		StaticWorldSelfOgcContactFinishTask* acquireStaticWorldSelfOgcContactFinishTask();

		void recycleStaticWorldSelfOgcContactTask(PxU32 index);

		void recycleStaticWorldSelfOgcContactFinishTask(PxU32 index);

		bool submitStandaloneStaticWorldSelfOgcContactTask(
			PxU32 dispatcherWorkers, Scene& owner,
			PxBaseTask* continuation,
			Dy::AvbdDynamicsContext& taskGraphContext);

		bool completeStandaloneStaticWorldSelfOgcContactTask(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady);

		bool finishStandaloneStaticWorldSelfOgcContactSerialFallback(
			PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
			bool& nextLayerReady, bool& nextWorldPlaneContactTaskReady,
			bool& nextRigidBoxSdfContactTaskReady,
			bool& nextRigidSphereSdfContactTaskReady);

	bool completeStandaloneCausalLayerTask(
		PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext,
		bool& nextLayerReady,
		bool& nextWorldPlaneContactTaskReady,
		bool& nextRigidBoxSdfContactTaskReady,
		bool& nextRigidSphereSdfContactTaskReady);

	bool finishStandaloneCausalLayerSerialFallback(
		PxReal dt, Dy::AvbdDynamicsContext& taskGraphContext);

	void finishStandaloneComponentSolve(PxReal dt);

	bool resumeStandaloneComponentSolve(
		PxReal dt, const PxVec3& gravity,
		bool* causalLayerTaskReady = NULL,
		bool* worldPlaneContactTaskReady = NULL,
		bool* rigidBoxSdfContactTaskReady = NULL,
		bool* rigidSphereSdfContactTaskReady = NULL);

	bool stepStandaloneComponentSolveOnly(
		PxReal dt,
		const PxVec3& gravity,
		const PxsDeformableVolumeMaterialManager& materialManager,
		const PxsMaterialManager& rigidMaterialManager,
		bool sleepingEnabled);

	PxU32 getStandaloneWriteBackTaskCount(PxU32 dispatcherWorkers) const;

	void submitStandaloneWriteBackTasks(
		PxU32 taskCount, PxBaseTask* continuation,
		Dy::AvbdDynamicsContext& taskGraphContext);

	void writeBackStandaloneComponentRange(
		PxU32 entryBegin, PxU32 entryEnd);

	void writeBackStandaloneComponent();

	void finishStandaloneComponentStep(PxReal dt, bool sleepingEnabled);

	void step(
		PxReal dt,
		const PxVec3& gravity,
		const PxsDeformableVolumeMaterialManager& materialManager,
		const PxsMaterialManager& rigidMaterialManager,
		bool sleepingEnabled);

private:
	PxU32 estimateInitialComponentContactCapacity() const;

	void reserveLifecycleContactCapacity();

	void reserveLifecycleCollisionScratch();

	void prepareComponentFallback(
		const PxsDeformableVolumeMaterialManager& materialManager,
		const PxsMaterialManager& rigidMaterialManager);

	void resumeComponentFallback(PxReal dt, const PxVec3& gravity);

	void stepComponentFallback(
		PxReal dt,
		const PxVec3& gravity,
		const PxsDeformableVolumeMaterialManager& materialManager,
		const PxsMaterialManager& rigidMaterialManager);

public:
	void recordStandaloneTaskGraphSubmission(
		PxU32 dispatcherWorkers, PxU32 particleCount)
	{
		mStandaloneTaskGraphTelemetry.recordSolveSubmission(
			dispatcherWorkers, particleCount);
	}

	void recordStandaloneTaskGraphSerialSolve(
		PxU32 dispatcherWorkers, PxU32 particleCount)
	{
		mStandaloneTaskGraphTelemetry.recordSerialSolve(
			dispatcherWorkers, particleCount);
	}

	void recordStandaloneSerialPredictionStage()
	{
		mStandaloneTaskGraphTelemetry.recordSerialPredictionStage();
	}

	void recordStandaloneSerialWriteBackStage()
	{
		mStandaloneTaskGraphTelemetry.recordSerialWriteBackStage();
	}

	void finishStandaloneTaskGraphNoOp();

	void writeAvbdCpuSoftBodyStatistics(PxSimulationStatistics& stats) const;

private:
const PxsDeformableSurfaceMaterialCore*
		getSurfaceMaterial(
			const DeformableSurfaceCore& core) const;

bool rebuildSurfaceRestState(Entry& entry);

Entry* findEntry(ActorCore& core);

PX_FORCE_INLINE PxU32 getParticleStart(
		const Entry& entry) const
	{
		PX_ASSERT(entry.bodyIndex < mBodies.size());
		return mBodies[entry.bodyIndex].compiled.particleStart;
	}

PX_FORCE_INLINE PxU32 getParticleCount(
		const Entry& entry) const
	{
		PX_ASSERT(entry.bodyIndex < mBodies.size());
		return mBodies[entry.bodyIndex].compiled.particleCount;
	}

bool isVolumeKinematicTargetActive(
		const Dy::DeformableVolumeCore& core,
		const PxVec4& target) const;

bool appendVolumeKinematicTargetPins(
		const Entry& entry,
		const PxArray<Dy::AvbdKinematicPin>& previousPins,
		PxArray<Dy::AvbdKinematicPin>& pins) const;

bool rebuildEntryPins(Entry& entry);

void refreshVolumeKinematicTargets();

void refreshPrescribedAttachmentTargets();

void removeWorldPinsForCore(ActorCore& core);

void removeRigidAttachmentsForSoft(ActorCore& core);

void removeArticulationAttachmentsForSoft(
		ActorCore& core);

void removeSoftPairAttachmentsForSoft(
		ActorCore& core);

void removePrescribedAttachmentsForSoft(
		ActorCore& core);

void removePrescribedAttachmentsForRigid(
		RigidCore& core);

void removeRigidAttachmentsForRigid(BodyCore& core);

void removeArticulationAttachmentsForLink(
		BodyCore& core);

void sleepEntry(Entry& entry);

void wakeEntry(Entry& entry, PxReal wakeCounter);

void finalizeDeformableMotionControls(PxReal dt);

bool hasUnforcedRestBendingResidual(const Entry& entry) const;

void updateSleepStates(
		PxReal dt, bool sleepingEnabled);

		void clearIslandSelectionStorages();

		void invalidateNativeIslandSelectionCaches();

		IslandSelectionStorage* acquireIslandSelectionStorage(
			IG::IslandId nativeIslandId);

		static bool rebaseParticleIndex(
			PxU32& index, PxU32 globalStart,
			PxU32 particleCount, PxU32 localStart);

		static bool copyAndRebaseSoftBody(
			const Dy::AvbdSoftBody& source,
			PxU32 globalStart, PxU32 particleCount,
			PxU32 localStart,
			Dy::AvbdSoftBody& destination);

		static bool rebaseSoftBodyParticleRangeInPlace(
			Dy::AvbdSoftBody& body,
			PxU32 oldStart, PxU32 particleCount,
			PxU32 newStart, bool compileObjectives = true);

		bool getCanonicalIslandParticleRange(
			const IslandSelectionStorage& storage,
			PxU32& particleStart,
			PxU32& particleCount) const;

		Dy::AvbdSoftParticle* getIslandSelectionParticles(
			IslandSelectionStorage& storage);

		PxU32 getIslandSelectionParticleCount(
			const IslandSelectionStorage& storage) const;

		bool findRigidBodyIndexInIsland(
			BodyCore& rigidCore,
			PxsRigidBody* const* rigidBodies,
			const Dy::AvbdSolverBody* solverBodies,
			PxU32 bodyStart,
			PxU32 bodyCount,
			PxU32& localBodyIndex) const;

		bool findArticulationBodyIndexInIsland(
			BodyCore& linkCore,
			Dy::FeatherstoneArticulation* const*
				articulationForBody,
			const PxU32* linkIndexForBody,
			const Dy::AvbdSolverBody* solverBodies,
			PxU32 bodyStart,
			PxU32 bodyCount,
			PxU32& localBodyIndex) const;

		RigidAttachmentEntry* findRigidAttachment(
			ActorCore& softCore,
			PxU32 handle);

		ArticulationAttachmentEntry*
			findArticulationAttachment(
				ActorCore& softCore,
				PxU32 handle);

		SoftPairAttachmentEntry* findSoftPairAttachment(
			ActorCore& softCore,
			PxU32 handle);

		PrescribedAttachmentEntry* findPrescribedAttachment(
			ActorCore& softCore,
			PxU32 handle);

		bool buildIslandSelectionStorage(
			IslandSelectionStorage& storage,
			Dy::AvbdSolverBody* solverBodies,
			PxsRigidBody* const* rigidBodies,
			Dy::FeatherstoneArticulation* const*
				articulationForBody,
			const PxU32* linkIndexForBody,
			PxU32 bodyStart, PxU32 bodyCount,
			PxReal dt, const PxVec3& gravity);

		void copyIslandSelectionResults(
			IslandSelectionStorage& storage);

		PxU32 findNativeIslandEdge(
			const ActorCore* softCore,
			const BodyCore* rigidCore) const;

		void ensureNativeIslandEdge(
			Entry& softEntry, BodyCore& rigidCore);

		PxU32 findNativeSoftSoftIslandEdge(
			const ActorCore* softCore0,
			const ActorCore* softCore1) const;

		void ensureNativeSoftSoftIslandEdge(
			Entry& softEntry0, Entry& softEntry1);

		void removeNativeIslandEdgesForRigid(BodyCore& core);

		void removeNativeIslandEdgesForSoft(
			ActorCore& core);

		void clearNativeIslandEdges();

		bool computeSoftBounds(
			const Entry& entry, PxBounds3& bounds) const;

		// Native-island topology must use the same public collision boundary as
		// OGC.  A cooked volume's simulation tetrahedra can lie strictly inside
		// its collision mesh; culling only against their AABB can therefore see a
		// current-pose OGC row too late to connect the dynamic rigid island for
		// that step.
		bool computeCollisionDomainSoftBounds(
			const Entry& entry, PxBounds3& bounds) const;

		bool expandSoftBoundsForPrediction(
			const Entry& entry, PxReal dt, const PxVec3& gravity,
			PxBounds3& bounds) const;

		// This is deliberately an endpoint-only DCD broad phase.  It is used
		// for sources that have opted out of speculative CCD: the resulting
		// contact pass evaluates the predicted endpoint only, never the segment
		// between the old and predicted poses.  Keep this separate from
		// expandSoftBoundsForPrediction(), which includes the old bounds for the
		// swept/CCD admission path.
		bool computePredictedCollisionDomainSoftBounds(
			const Entry& entry, PxReal dt, const PxVec3& gravity,
			PxBounds3& bounds) const;

		static PxBounds3 computeBoxBounds(
			const Dy::AvbdRigidBox& box);

		static PxBounds3 computeSphereBounds(
			const Dy::AvbdRigidSphere& sphere);

		static PxBounds3 computeCapsuleBounds(
			const Dy::AvbdRigidCapsule& capsule);

		static PxBounds3 computeConvexBounds(
			const Dy::AvbdRigidConvex& convex);

		// Build a conservative AABB for one rigid shape at the same discrete
		// endpoint used by non-CCD soft admission. The sphere is an endpoint
		// orientation envelope, not a source-to-end sweep: only the predicted
		// body center is included. Keeping this helper common prevents smooth
		// primitives from silently falling back to source-pose topology while
		// boxes use endpoint ownership.
		static bool computeDynamicEndpointEnvelopeBounds(
			const DynamicShapeEntry& entry,
			const PxVec3& shapeCenter, PxReal shapeRadius,
			PxReal dt, const PxVec3& gravity, PxBounds3& bounds);

		static PxBounds3 computeTriangleSurfaceBounds(
			const Dy::AvbdRigidTriangleSurface& surface);

		static void getRigidMaterialValues(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			PxMaterialTableIndex tableIndex,
			PxReal& friction, PxU8& combineMode);

		static bool appendTriangleSurfaceTriangle(
			const PxTriangle& sourceTriangle,
			const PxU32 sourceVertexIndices[3],
			PxU32 sourceTriangleIndex,
			PxReal friction, PxU8 frictionCombineMode,
			PxHashMap<PxU32, PxU32>& vertexMap,
			PxHashMap<PxU64, PxU32>& edgeMap,
			Dy::AvbdRigidTriangleSurface& surface);

		static bool finalizeTriangleSurfaceTopology(
			Dy::AvbdRigidTriangleSurface& surface,
			bool suppressBoundaryEdges);

		static PxBounds3 getRigidTriangleSurfaceTriangleBounds(
			const Dy::AvbdRigidTriangleSurface& surface,
			PxU32 triangleIndex);

		static PxU32 buildRigidTriangleSurfaceBvhNode(
			Dy::AvbdRigidTriangleSurface& surface,
			PxU32 firstPrimitive, PxU32 primitiveCount);

		static void buildRigidTriangleSurfaceBvh(
			Dy::AvbdRigidTriangleSurface& surface);

		static bool compileTriangleMeshTopology(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			const PxTriangleMeshGeometry& geometry,
			Dy::AvbdRigidTriangleSurface& surface);

		static bool compileHeightFieldTopology(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			const PxHeightFieldGeometry& geometry,
			Dy::AvbdRigidTriangleSurface& surface);

		static bool sameTriangleSurfaceVec3(
			const PxVec3& lhs, const PxVec3& rhs);

		static bool sameTriangleSurfaceQuat(
			const PxQuat& lhs, const PxQuat& rhs);

		static bool getTriangleMeshMaterialValues(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			const PxTriangleMesh& mesh,
			PxU32 sourceTriangleIndex, PxReal& friction,
			PxU8& frictionCombineMode);

		static bool getHeightFieldMaterialValues(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			const PxHeightField& heightField,
			PxU32 sourceTriangleIndex, PxReal& friction,
			PxU8& frictionCombineMode);

		static bool refreshTriangleMeshSurfaceMaterials(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			const PxTriangleMeshGeometry& geometry,
			Dy::AvbdRigidTriangleSurface& surface);

		static bool refreshHeightFieldSurfaceMaterials(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			const PxHeightFieldGeometry& geometry,
			Dy::AvbdRigidTriangleSurface& surface);

		static bool triangleMeshTopologyMatches(
			const PxTriangleMeshGeometry& geometry,
			const Dy::AvbdRigidTriangleSurface& surface);

		static bool heightFieldTopologyMatches(
			const PxHeightFieldGeometry& geometry,
			const Dy::AvbdRigidTriangleSurface& surface);

		static void setTriangleMeshTopologyIdentity(
			const PxTriangleMeshGeometry& geometry,
			Dy::AvbdRigidTriangleSurface& surface);

		static void setHeightFieldTopologyIdentity(
			const PxHeightFieldGeometry& geometry,
			Dy::AvbdRigidTriangleSurface& surface);

		static bool refreshRigidTriangleSurfaceTopology(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			Dy::AvbdRigidTriangleSurface& surface);

		static bool compileConvexTopology(
			const PxConvexMeshGeometry& geometry,
			Dy::AvbdRigidConvex& convex);

		bool compileDynamicConvex(
			const DynamicShapeEntry& entry,
			Dy::AvbdRigidConvex& convex) const;

		bool compileDynamicTriangleSurface(
			const DynamicShapeEntry& entry,
			Dy::AvbdRigidTriangleSurface& surface) const;

		bool compileDynamicBox(
			const DynamicShapeEntry& entry,
			Dy::AvbdRigidBox& box) const;

		bool compileDynamicSphere(
			const DynamicShapeEntry& entry,
			Dy::AvbdRigidSphere& sphere) const;

		bool compileDynamicCapsule(
			const DynamicShapeEntry& entry,
			Dy::AvbdRigidCapsule& capsule) const;

		static const PxsDeformableVolumeMaterialCore* getMaterial(
			const DeformableVolumeCore& core,
			const PxsDeformableVolumeMaterialManager& materialManager);

		static PxReal getStaticFriction(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager);

		static PxU8 getStaticFrictionCombineMode(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager);

		Dy::AvbdRigidTriangleSurface& getRigidTriangleSurface(
			PxU64 primitiveKey);

		void compileWorldStatics(
			const PxsMaterialManager& materialManager);

		void compileDynamicBoxesForIsland(
			PxsRigidBody* const* rigidBodies,
			const Dy::AvbdSolverBody* solverBodies,
			PxU32 bodyStart,
			PxU32 bodyCount,
			PxArray<Dy::AvbdRigidBox>& boxes);

		void compileDynamicSpheresForIsland(
			PxsRigidBody* const* rigidBodies,
			Dy::AvbdSolverBody* solverBodies,
			PxU32 bodyStart,
			PxU32 bodyCount,
			PxReal dt,
			const PxVec3& gravity,
			PxArray<Dy::AvbdRigidSphere>& spheres);

		void compileDynamicCapsulesForIsland(
			PxsRigidBody* const* rigidBodies,
			Dy::AvbdSolverBody* solverBodies,
			PxU32 bodyStart,
			PxU32 bodyCount,
			PxReal dt,
			const PxVec3& gravity,
			PxArray<Dy::AvbdRigidCapsule>& capsules);

		void compileDynamicConvexesForIsland(
			PxsRigidBody* const* rigidBodies,
			Dy::AvbdSolverBody* solverBodies,
			PxU32 bodyStart,
			PxU32 bodyCount,
			PxReal dt,
			const PxVec3& gravity,
			PxArray<Dy::AvbdRigidConvex>& convexes);

		static bool readTetrahedronIndices(
			const PxTetrahedronMesh& mesh, PxArray<PxU32>& indices);

		static bool readTriangleIndices(
			const PxTriangleMesh& mesh, PxArray<PxU32>& indices);

		static void reportInvalidCollisionEmbedding(const char* reason);

		static bool validateVolumeCollisionEmbedding(
			PxTetrahedronMesh& simulationMesh,
			PxTetrahedronMesh& collisionMesh,
			PxDeformableVolumeAuxData& publicAuxData);

		static PxVec3 evaluateWeightedParticlePosition(
			const Dy::AvbdWeightedContactPoint& point,
			const Dy::AvbdSoftParticle* particles, PxU32 particleCount,
			PxU32 source);

	bool rebuildCollisionDetectionScene();

	bool refreshCollisionDetectionScene(
		const Dy::AvbdSoftParticle* sourceParticles,
		PxU32 sourceParticleCount);

	bool expandCollisionDetectionPoint(
		const PxU32* proxyIndices, const PxReal* proxyWeights,
		PxU32 proxyCount,
		const PxArray<Dy::AvbdWeightedContactPoint>& vertexMappings,
		Dy::AvbdWeightedContactPoint& output) const;

	bool rebuildSubsetCollisionDetectionScene(
		const Dy::AvbdSoftParticle* sourceParticles,
		PxU32 sourceParticleCount,
		const Dy::AvbdSoftBody* sourceBodies, PxU32 sourceBodyCount,
		ActorCore* const* softCores);

	PxU32 findCollisionBodyForParticle(
		PxU32 particleIndex,
		const PxArray<Dy::AvbdSoftBody>& collisionBodies) const;

	PxU32 resolveCollisionElementForFeature(
		const Dy::AvbdSoftContactGeometry& geometry,
		const Dy::AvbdSoftBodyCompiledData& compiled,
		PxU32 collisionFeatureParticle) const;

	bool expandCollisionDetectionContacts(
		PxArray<Dy::AvbdSoftContact>& contacts,
		PxU32 simulationParticleCount,
		const PxArray<Dy::AvbdWeightedContactPoint>& vertexMappings,
		const PxArray<Dy::AvbdSoftBody>& collisionBodies,
		Dy::AvbdOgcGeometryEpochSidecar* ogcGeometrySidecar) const;

	void refreshSelfCollisionEnabled();

	ActorCore* findRigidCoreForPrimitive(PxU64 primitiveKey) const;

	ActorCore* findSoftCoreForContactBody(
		const Dy::AvbdSoftBody* bodies,
		PxU32 numBodies,
		ActorCore* const* softCores,
		PxU32 particleIndex) const;

	ActorCore* findSoftCoreForContactBodyIndex(
		const Dy::AvbdSoftBody* bodies,
		PxU32 numBodies,
		ActorCore* const* softCores,
		PxU32 bodyIndex) const;

	const Dy::AvbdSoftBody* findSoftBodyForContactParticle(
		const Dy::AvbdSoftBody* bodies,
		PxU32 numBodies,
		PxU32 particleIndex) const;

	bool isRigidActorContactFiltered(
		const Dy::AvbdSoftBody& body,
		ActorCore& softCore,
		ActorCore& rigidCore,
		PxU32 particleIndex) const;

	bool isDeformablePairContactFiltered(
		const Dy::AvbdSoftBody& queryBody,
		ActorCore& queryCore,
		ActorCore& targetCore,
		PxU32 queryParticleIndex,
		PxU32 targetSourceElementIndex) const;

	bool removeRigidActorFilteredContacts(
		const Dy::AvbdSoftBody* bodies,
		PxU32 numBodies,
		ActorCore* const* softCores,
		PxArray<Dy::AvbdSoftContact>& contacts,
		Dy::AvbdOgcGeometryEpochSidecar* geometrySidecar) const;

	bool removeDeformablePairFilteredContacts(
		const Dy::AvbdSoftBody* bodies,
		PxU32 numBodies,
		ActorCore* const* softCores,
		PxArray<Dy::AvbdSoftContact>& contacts,
		Dy::AvbdOgcGeometryEpochSidecar* geometrySidecar) const;

	void detectContacts(
		Dy::AvbdSoftParticle* particles,
		PxU32 numParticles,
		Dy::AvbdSoftBody* bodies,
		PxU32 numBodies,
		PxArray<Dy::AvbdSoftContact>& contacts,
		const Dy::AvbdRigidBox* rigidBoxes = NULL,
		PxU32 numRigidBoxes = 0,
		const Dy::AvbdSelfCollisionAdjacency*
			selfCollisionAdjacencies = NULL,
		PxU32 numSelfCollisionAdjacencies = 0,
		const PxU8* selfCollisionEnabled = NULL,
		ActorCore* const* softCores = NULL,
		const Dy::AvbdRigidSphere* rigidSpheres = NULL,
		PxU32 numRigidSpheres = 0,
		const Dy::AvbdRigidCapsule* rigidCapsules = NULL,
		PxU32 numRigidCapsules = 0,
		const Dy::AvbdRigidConvex* rigidConvexes = NULL,
		PxU32 numRigidConvexes = 0,
		const Dy::AvbdRigidTriangleSurface*
			rigidTriangleSurfaces = NULL,
		PxU32 numRigidTriangleSurfaces = 0,
		Dy::AvbdOgcGeometryEpochSidecar* ogcGeometrySidecar = NULL);

	// Parent-owned component contact publication. The contact stream, sparse
	// geometry sidecar and pair plan are invalidated/published as one epoch.
	void beginComponentContactRedetection();
	void completeComponentContactRedetection();
	bool publishComponentOgcGeometryEpoch();

	static void redetectContacts(
		Dy::AvbdSoftParticle* particles,
		PxU32 numParticles,
		Dy::AvbdSoftBody* bodies,
		PxU32 numBodies,
		PxArray<Dy::AvbdSoftContact>& contacts,
		void* userData);

	void refreshSurfaceFlattening(Entry& entry);

	void applyDeformablePreintegrationControls(Entry& entry);

	void syncHostInputs(
		Entry& entry,
		const PxsDeformableVolumeMaterialManager& materialManager);

	void writeBack(Entry& entry);

		PxArray<Entry>					mEntries;
		PxArray<PredictionTask*>			mPredictionTasks;
		PxArray<WriteBackTask*>			mWriteBackTasks;
		// P4.5.3 keeps a bounded Scene-owned task pool. It is grown only by a
		// parent before a layer is submitted and recycled only after dispatcher
		// release; children never allocate, resize or inspect Scene state.
		PxMutex							mCausalLayerTaskPoolMutex;
		PxArray<CausalLayerTask*>			mCausalLayerTasks;
		PxArray<CausalLayerFinishTask*>	mCausalLayerFinishTasks;
		PxArray<PxU32>					mFreeCausalLayerTaskIndices;
		PxArray<PxU32>					mFreeCausalLayerFinishTaskIndices;
		PxArray<Dy::AvbdParticlePrimalRangeObservation>
									mCausalLayerRangeObservations;
		// P5.3b uses a separate, bounded pool because collision leaves own
		// private contact streams rather than primal range observations.
		PxMutex							mWorldPlaneContactTaskPoolMutex;
		PxArray<WorldPlaneContactTask*>	mWorldPlaneContactTasks;
		PxArray<WorldPlaneContactFinishTask*>
									mWorldPlaneContactFinishTasks;
		PxArray<PxU32>					mFreeWorldPlaneContactTaskIndices;
		PxArray<PxU32>					mFreeWorldPlaneContactFinishTaskIndices;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mWorldPlaneContactTaskOutputs;
		// P5.4b owns a distinct bounded pool so a rigid-box SDF epoch cannot
		// borrow/relabel world-plane task telemetry or private output storage.
		PxMutex							mRigidBoxSdfContactTaskPoolMutex;
		PxArray<RigidBoxSdfContactTask*>	mRigidBoxSdfContactTasks;
		PxArray<RigidBoxSdfContactFinishTask*>
									mRigidBoxSdfContactFinishTasks;
		PxArray<PxU32>					mFreeRigidBoxSdfContactTaskIndices;
		PxArray<PxU32>					mFreeRigidBoxSdfContactFinishTaskIndices;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidBoxSdfContactTaskOutputs;
		// P5.12b keeps the swept family separate until parent fan-in so the
		// canonical current-then-swept contact order is mechanically visible.
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidBoxSweptSdfContactTaskOutputs;
		// P5.5b does not borrow the box pool: its eligibility and the parent
		// sphere swept/feature suffix are independently observable.
		PxMutex							mRigidSphereSdfContactTaskPoolMutex;
		PxArray<RigidSphereSdfContactTask*>	mRigidSphereSdfContactTasks;
		PxArray<RigidSphereSdfContactFinishTask*>
									mRigidSphereSdfContactFinishTasks;
		PxArray<PxU32>					mFreeRigidSphereSdfContactTaskIndices;
		PxArray<PxU32>					mFreeRigidSphereSdfContactFinishTaskIndices;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidSphereSdfContactTaskOutputs;
		// P5.13b retains swept ranges independently until the parent completes
		// the canonical all-current then all-swept family merge.
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidSphereSweptSdfContactTaskOutputs;
		// P5.14b retains swept capsule ranges independently until the parent
		// completes the canonical all-current then all-swept family merge.
		// The only shared object with spheres is the continuation slot, which is
		// mutually exclusive by primitive eligibility.
		PxMutex							mRigidCapsuleSdfContactTaskPoolMutex;
		PxArray<RigidCapsuleSdfContactTask*>	mRigidCapsuleSdfContactTasks;
		PxArray<RigidCapsuleSdfContactFinishTask*>
									mRigidCapsuleSdfContactFinishTasks;
		PxArray<PxU32>					mFreeRigidCapsuleSdfContactTaskIndices;
		PxArray<PxU32>					mFreeRigidCapsuleSdfContactFinishTaskIndices;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidCapsuleSdfContactTaskOutputs;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidCapsuleSweptSdfContactTaskOutputs;
		PxMutex							mRigidConvexSdfContactTaskPoolMutex;
		PxArray<RigidConvexSdfContactTask*>	mRigidConvexSdfContactTasks;
		PxArray<RigidConvexSdfContactFinishTask*>
									mRigidConvexSdfContactFinishTasks;
		PxArray<PxU32>					mFreeRigidConvexSdfContactTaskIndices;
		PxArray<PxU32>					mFreeRigidConvexSdfContactFinishTaskIndices;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidConvexSdfContactTaskOutputs;
		// P5.15b retains swept convex ranges independently until the parent
		// completes the canonical all-current then all-swept family merge.
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidConvexSweptSdfContactTaskOutputs;
		// P5.16b retains current and swept triangle-SDF range outputs until the
		// parent completes the canonical all-current then all-swept merge.
		PxMutex							mRigidTriangleSurfaceContactTaskPoolMutex;
		PxArray<RigidTriangleSurfaceContactTask*>
									mRigidTriangleSurfaceContactTasks;
		PxArray<RigidTriangleSurfaceContactFinishTask*>
									mRigidTriangleSurfaceContactFinishTasks;
		PxArray<PxU32>					mFreeRigidTriangleSurfaceContactTaskIndices;
		PxArray<PxU32>					mFreeRigidTriangleSurfaceContactFinishTaskIndices;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidTriangleSurfaceContactTaskOutputs;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidTriangleSurfaceSweptSdfContactTaskOutputs;
		// P5.17d feature rows are partitioned by canonical plan index and
		// stable-merged only by the parent after both SDF output families.
		Dy::AvbdRigidTriangleSurfaceFeaturePlan
									mRigidTriangleSurfaceFeaturePlan;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidTriangleSurfaceFeatureContactTaskOutputs;
		// P5.27 default-off candidate: one complete private output per canonical
		// feature-plan row, independently of child scheduling order.
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mRigidTriangleSurfaceFeatureContactPlanOutputs;
		bool						mRigidTriangleSurfaceFeatureRowPrivateOutputTaskPlan;
		bool						mRigidTriangleSurfaceFeatureRoundRobinTaskPlan;
		PxArray<Dy::AvbdSoftCollisionStats>
									mRigidTriangleSurfaceContactTaskStats;
		// P5.10b keeps self-BVH leaves separate from pair leaves. Task output
		// order encodes the required VF phase followed by the EE phase.
		PxMutex							mSelfBvhContactTaskPoolMutex;
		PxArray<SelfBvhContactTask*>	mSelfBvhContactTasks;
		PxArray<SelfBvhContactFinishTask*>
									mSelfBvhContactFinishTasks;
		PxArray<PxU32>					mFreeSelfBvhContactTaskIndices;
		PxArray<PxU32>					mFreeSelfBvhContactFinishTaskIndices;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mSelfBvhContactTaskOutputs;
		PxArray<Dy::AvbdSoftCollisionStats>
									mSelfBvhContactTaskStats;
		Dy::AvbdSoftContactWorkspace		mSelfBvhSerialRangeWorkspace;
		// P5 static-world+self aggregate: each child owns a disjoint range, but
		// its source streams remain physically separate until the parent merges
		// world, box-current, box-swept, box-features, self-VF, then self-EE.
		PxMutex							mStaticWorldSelfOgcContactFinishTaskPoolMutex;
		PxArray<StaticWorldSelfOgcContactTask*>
									mStaticWorldSelfOgcContactTasks;
		PxArray<StaticWorldSelfOgcContactFinishTask*>
									mStaticWorldSelfOgcContactFinishTasks;
		PxArray<PxU32>					mFreeStaticWorldSelfOgcContactTaskIndices;
		PxArray<PxU32>					mFreeStaticWorldSelfOgcContactFinishTaskIndices;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mStaticWorldSelfOgcWorldTaskOutputs;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mStaticWorldSelfOgcBoxTaskOutputs;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mStaticWorldSelfOgcBoxSweptTaskOutputs;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mStaticWorldSelfOgcSelfVertexTaskOutputs;
		PxArray<PxArray<Dy::AvbdSoftContact> >
									mStaticWorldSelfOgcSelfEdgeTaskOutputs;
		PxArray<Dy::AvbdSoftCollisionStats>
									mStaticWorldSelfOgcTaskStats;
		PxArray<StaticShapeEntry>		mStaticShapes;
		PxArray<DynamicShapeEntry>		mDynamicShapes;
		PxArray<WorldPinEntry>			mWorldPins;
		PxArray<RigidAttachmentEntry>	mRigidAttachments;
		PxArray<ArticulationAttachmentEntry>
										mArticulationAttachments;
		PxArray<SoftPairAttachmentEntry>
										mSoftPairAttachments;
		PxArray<PrescribedAttachmentEntry>
										mPrescribedAttachments;
		PxArray<RigidActorFilterEntry>	mRigidActorFilters;
		PxArray<DeformablePairFilterEntry>
										mDeformablePairFilters;
		PxArray<NativeIslandEdgeEntry>	mNativeIslandEdges;
		PxArray<NativeSoftSoftIslandEdgeEntry>
										mNativeSoftSoftIslandEdges;
		AvbdIslandSelectionStoragePool	mIslandSelectionStorages;
		PxArray<Dy::AvbdSoftParticle>	mParticles;
		PxArray<Dy::AvbdSoftBody>		mBodies;
		// Public CPU AVBD detects against the cooked collision-domain mesh.
		// These proxy particles are geometry only; prepared contacts are expanded
		// through mCollisionProxy.collisionVertexMappings before either solver sees them.
		AvbdCollisionProxyWorkspace	mCollisionProxy;
		PxArray<Dy::AvbdSelfCollisionAdjacency>
										mSelfCollisionAdjacencies;
		PxArray<PxU8>					mSelfCollisionEnabled;
		PxArray<Dy::AvbdWorldPlane>		mWorldPlanes;
		PxArray<Dy::AvbdRigidBox>		mRigidBoxes;
		PxArray<Dy::AvbdRigidSphere>		mRigidSpheres;
		PxArray<Dy::AvbdRigidCapsule>	mRigidCapsules;
		PxArray<Dy::AvbdRigidConvex>		mRigidConvexes;
		PxArray<Dy::AvbdRigidTriangleSurface>
										mRigidTriangleSurfaces;
		PxArray<Dy::AvbdSoftContact>		mContacts;
		Dy::AvbdOGCParams				mContactParams;
		Dy::AvbdSoftBodyWorkspace		mWorkspace;
		Dy::AvbdSoftBodyStepState		mStandaloneComponentStepState;
		Dy::AvbdSoftBodyStepStats		mLastStepStats;
		Dy::AvbdSoftBodyStepStats		mStandaloneStepStats;
		Dy::AvbdSoftCollisionStats		mLastCollisionStats;
		ComponentFallbackPlan			mComponentFallbackPlan;
		PxU64							mContextId;
		const PxsDeformableVolumeMaterialManager&
										mDeformableMaterialManager;
		const PxsDeformableSurfaceMaterialManager&
										mSurfaceMaterialManager;
		const PxsMaterialManager&		mRigidMaterialManager;
		IG::SimpleIslandManager&		mIslandManager;
		PxU64							mNextPrimitiveKey;
		PxU32							mRigidTriangleSurfaceCompileStamp;
		PxU32							mNextWorldPinHandle;
		PxU32							mNextRigidAttachmentHandle;
		PxU32							mNextArticulationAttachmentHandle;
		PxU32							mNextSoftPairAttachmentHandle;
		PxU32							mNextPrescribedAttachmentHandle;
		PxU32							mNextRigidActorFilterHandle;
		PxU32							mNextDeformablePairFilterHandle;
		bool							mDynamicsOwnsStep;
		PxU32							mDynamicsSelectedEntryCount;
		PxU32							mLastComponentFallbackSteps;
		PxU32							mLastNativeIslandSteps;
		bool							mComponentFallbackPlanPrepared;
		bool							mStandaloneComponentSolvePrepared;
		bool							mStandaloneComponentPostSolvePending;
		PxU32						mStandaloneTaskGraphDispatcherWorkers;
		bool						mStandaloneTaskGraphEnhancedDeterminism;
		Dy::AvbdParticlePrimalSchedule	mStandaloneParticlePrimalSchedule;
		StandaloneTaskGraphTelemetry	mStandaloneTaskGraphTelemetry;
		bool							mP3ForceSplitPrediction;
		bool							mCollisionStatsEnabled;
		bool							mWorldPlaneContactTransactionPending;
		bool							mRigidBoxSdfContactTransactionPending;
		bool							mRigidSphereSdfContactTransactionPending;
		bool							mRigidCapsuleSdfContactTransactionPending;
		bool							mRigidConvexSdfContactTransactionPending;
			bool							mRigidTriangleSurfaceContactTransactionPending;
		bool							mSelfBvhContactTransactionPending;
		PxU32						mSelfBvhContactBodyIndex;
		bool							mStaticWorldSelfOgcContactTransactionPending;
		bool							mWorkspacePreflightPending;

};

} // namespace Sc
} // namespace physx

#include "avbd/scheduling/ScAvbdContactTasks.h"

#endif // SC_AVBD_CPU_SOFT_SCENE_H
