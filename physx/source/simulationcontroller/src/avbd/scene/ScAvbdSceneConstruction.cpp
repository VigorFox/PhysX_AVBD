// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

		AvbdCpuSoftScene::AvbdCpuSoftScene(
			const PxsDeformableVolumeMaterialManager&
				deformableMaterialManager,
			const PxsDeformableSurfaceMaterialManager&
				surfaceMaterialManager,
			const PxsMaterialManager& rigidMaterialManager,
			PxU64 contextId,
			IG::SimpleIslandManager& islandManager)
			: mContextId(contextId),
			  mDeformableMaterialManager(deformableMaterialManager),
			  mSurfaceMaterialManager(surfaceMaterialManager),
			  mRigidMaterialManager(rigidMaterialManager),
			  mIslandManager(islandManager),
			  mNextPrimitiveKey(1),
			  mRigidTriangleSurfaceCompileStamp(0),
			  mNextWorldPinHandle(1),
			  mNextRigidAttachmentHandle(1),
			  mNextArticulationAttachmentHandle(1),
			  mNextSoftPairAttachmentHandle(1),
			  mNextPrescribedAttachmentHandle(1),
			  mNextRigidActorFilterHandle(1),
			  mNextDeformablePairFilterHandle(1),
			  mDynamicsOwnsStep(false),
			  mDynamicsSelectedEntryCount(0),
			  mLastComponentFallbackSteps(0),
			  mLastNativeIslandSteps(0),
			  mComponentFallbackPlanPrepared(false),
			  mStandaloneComponentSolvePrepared(false),
			  mStandaloneComponentPostSolvePending(false),
			  mStandaloneTaskGraphDispatcherWorkers(0),
			  mStandaloneTaskGraphEnhancedDeterminism(false),
			  mStandaloneParticlePrimalSchedule(
				  Dy::AvbdParticlePrimalSchedule::eSERIAL_LINEAR),
			  mP3ForceSplitPrediction(false),
			  mCollisionStatsEnabled(false),
			  mWorldPlaneContactTransactionPending(false),
			  mRigidBoxSdfContactTransactionPending(false),
			  mRigidSphereSdfContactTransactionPending(false),
			  mRigidCapsuleSdfContactTransactionPending(false),
			  mRigidConvexSdfContactTransactionPending(false),
			  mRigidTriangleSurfaceContactTransactionPending(false),
			  mRigidTriangleSurfaceFeatureRowPrivateOutputTaskPlan(false),
			  mRigidTriangleSurfaceFeatureRoundRobinTaskPlan(false),
			  mSelfBvhContactTransactionPending(false),
			  mSelfBvhContactBodyIndex(PX_MAX_U32),
			  mStaticWorldSelfOgcContactTransactionPending(false),
			  mWorkspacePreflightPending(true)
		{
			const char* const collisionTelemetry =
				std::getenv("PHYSX_AVBD_COLLISION_TELEMETRY");
			mCollisionStatsEnabled = collisionTelemetry &&
				collisionTelemetry[0] == '1' &&
				collisionTelemetry[1] == '\0';
			const char* const forceSplitPrediction =
				std::getenv("PHYSX_AVBD_P3_FORCE_SPLIT_PREDICTION");
			mP3ForceSplitPrediction = forceSplitPrediction &&
				forceSplitPrediction[0] == '1' &&
				forceSplitPrediction[1] == '\0';
		}

		AvbdCpuSoftScene::~AvbdCpuSoftScene()
		{
			for(PxU32 i = 0; i < mPredictionTasks.size(); i++)
				PX_DELETE(mPredictionTasks[i]);
			mPredictionTasks.clear();
			for(PxU32 i = 0; i < mWriteBackTasks.size(); i++)
				PX_DELETE(mWriteBackTasks[i]);
			mWriteBackTasks.clear();
			for(PxU32 i = 0; i < mCausalLayerTasks.size(); i++)
				PX_DELETE(mCausalLayerTasks[i]);
			mCausalLayerTasks.clear();
			for(PxU32 i = 0; i < mCausalLayerFinishTasks.size(); i++)
				PX_DELETE(mCausalLayerFinishTasks[i]);
			mCausalLayerFinishTasks.clear();
			for(PxU32 i = 0; i < mWorldPlaneContactTasks.size(); i++)
				PX_DELETE(mWorldPlaneContactTasks[i]);
			mWorldPlaneContactTasks.clear();
			for(PxU32 i = 0; i < mWorldPlaneContactFinishTasks.size(); i++)
				PX_DELETE(mWorldPlaneContactFinishTasks[i]);
			mWorldPlaneContactFinishTasks.clear();
			for(PxU32 i = 0; i < mRigidBoxSdfContactTasks.size(); i++)
				PX_DELETE(mRigidBoxSdfContactTasks[i]);
			mRigidBoxSdfContactTasks.clear();
			for(PxU32 i = 0; i < mRigidBoxSdfContactFinishTasks.size(); i++)
				PX_DELETE(mRigidBoxSdfContactFinishTasks[i]);
			mRigidBoxSdfContactFinishTasks.clear();
			for(PxU32 i = 0; i < mRigidSphereSdfContactTasks.size(); i++)
				PX_DELETE(mRigidSphereSdfContactTasks[i]);
			mRigidSphereSdfContactTasks.clear();
			for(PxU32 i = 0; i < mRigidSphereSdfContactFinishTasks.size(); i++)
				PX_DELETE(mRigidSphereSdfContactFinishTasks[i]);
			mRigidSphereSdfContactFinishTasks.clear();
			for(PxU32 i = 0; i < mRigidCapsuleSdfContactTasks.size(); i++)
				PX_DELETE(mRigidCapsuleSdfContactTasks[i]);
			mRigidCapsuleSdfContactTasks.clear();
			for(PxU32 i = 0; i < mRigidCapsuleSdfContactFinishTasks.size(); i++)
				PX_DELETE(mRigidCapsuleSdfContactFinishTasks[i]);
			mRigidCapsuleSdfContactFinishTasks.clear();
			for(PxU32 i = 0; i < mRigidConvexSdfContactTasks.size(); i++)
				PX_DELETE(mRigidConvexSdfContactTasks[i]);
			mRigidConvexSdfContactTasks.clear();
			for(PxU32 i = 0; i < mRigidConvexSdfContactFinishTasks.size(); i++)
				PX_DELETE(mRigidConvexSdfContactFinishTasks[i]);
			mRigidConvexSdfContactFinishTasks.clear();
			for(PxU32 i = 0; i < mRigidTriangleSurfaceContactTasks.size(); i++)
				PX_DELETE(mRigidTriangleSurfaceContactTasks[i]);
			mRigidTriangleSurfaceContactTasks.clear();
			for(PxU32 i = 0; i < mRigidTriangleSurfaceContactFinishTasks.size(); i++)
				PX_DELETE(mRigidTriangleSurfaceContactFinishTasks[i]);
			mRigidTriangleSurfaceContactFinishTasks.clear();
			for(PxU32 i = 0; i < mSelfBvhContactTasks.size(); i++)
				PX_DELETE(mSelfBvhContactTasks[i]);
			mSelfBvhContactTasks.clear();
			for(PxU32 i = 0; i < mSelfBvhContactFinishTasks.size(); i++)
				PX_DELETE(mSelfBvhContactFinishTasks[i]);
			mSelfBvhContactFinishTasks.clear();
			for(PxU32 i = 0;
				i < mStaticWorldSelfOgcContactTasks.size(); i++)
				PX_DELETE(mStaticWorldSelfOgcContactTasks[i]);
			mStaticWorldSelfOgcContactTasks.clear();
			for(PxU32 i = 0;
				i < mStaticWorldSelfOgcContactFinishTasks.size(); i++)
				PX_DELETE(mStaticWorldSelfOgcContactFinishTasks[i]);
			mStaticWorldSelfOgcContactFinishTasks.clear();
			clearNativeIslandEdges();
			clearIslandSelectionStorages();
			for(PxU32 i = 0; i < mEntries.size(); i++)
			{
				Entry& entry = mEntries[i];
				mIslandManager.removeNode(entry.islandNode);
				entry.destroyIslandObject();
			}
		}

} // namespace Sc
} // namespace physx
