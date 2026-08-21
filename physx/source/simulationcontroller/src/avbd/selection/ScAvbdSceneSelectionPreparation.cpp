// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/scene/ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{
	namespace
	{
		void configureTerminalOgcGeometryProvider(
			Dy::AvbdSoftIslandExecutionPlan& plan,
			AvbdIslandSelectionStorage& storage,
			const PxArray<Dy::AvbdWorldPlane>& worldPlanes,
			const PxArray<Dy::AvbdRigidTriangleSurface>& triangleSurfaces,
			PxReal contactRadius)
		{
			Dy::AvbdOgcCurrentPoseGeometryProvider& provider =
				plan.terminalGeometryProvider;
			provider.collisionBodies = storage.terminalCollisionBodies.begin();
			provider.numCollisionBodies = storage.terminalCollisionBodies.size();
			provider.collisionVertexMappings =
				storage.terminalCollisionVertexMappings.begin();
			provider.numCollisionVertexMappings =
				storage.terminalCollisionVertexMappings.size();
			provider.worldPlanes = worldPlanes.begin();
			provider.numWorldPlanes = worldPlanes.size();
			provider.rigidBoxes = storage.rigidBoxes.begin();
			provider.numRigidBoxes = storage.rigidBoxes.size();
			provider.rigidSpheres = storage.rigidSpheres.begin();
			provider.numRigidSpheres = storage.rigidSpheres.size();
			provider.rigidCapsules = storage.rigidCapsules.begin();
			provider.numRigidCapsules = storage.rigidCapsules.size();
			provider.rigidConvexes = storage.rigidConvexes.begin();
			provider.numRigidConvexes = storage.rigidConvexes.size();
			provider.rigidTriangleSurfaces = triangleSurfaces.begin();
			provider.numRigidTriangleSurfaces = triangleSurfaces.size();
			provider.contactRadius = contactRadius;
			provider.includeSoftTargets = true;
		}
	}

		bool AvbdCpuSoftScene::prepareSoftIslandSelections(
			Dy::AvbdSolverBody* solverBodies,
			PxsRigidBody* const* rigidBodies,
			Dy::FeatherstoneArticulation* const*
				articulationForBody,
			const PxU32* linkIndexForBody,
			const PxU32* islandBodyStarts,
			const PxU32* islandBodyCounts,
			const PxU32* activeIslandIds,
			PxU32 islandCount,
			PxReal dt,
			const PxVec3& gravity,
			PxArray<Dy::AvbdSoftIslandSelection>& selections)
		{
			selections.clear();
			mDynamicsOwnsStep = false;
			mDynamicsSelectedEntryCount = 0;
			if(dt <= 0.0f || mBodies.empty() || !solverBodies ||
				!rigidBodies ||
				(!mArticulationAttachments.empty() &&
				 (!articulationForBody || !linkIndexForBody)) ||
				!islandBodyStarts ||
				!islandBodyCounts || !activeIslandIds ||
				islandCount == 0 || mEntries.empty())
				return false;

			const IG::IslandSim& islandSim =
				mIslandManager.getAccurateIslandSim();
			for(PxU32 i = 0; i < mEntries.size(); i++)
			{
				Entry& entry = mEntries[i];
				if(!entry.sleeping)
					continue;
				const PxNodeIndex node = entry.islandNode;
				if(!node.isValid() ||
					node.index() >= islandSim.getNbNodes())
					return false;
				const IG::IslandId entryIslandId =
					islandSim.getIslandIds()[node.index()];
				for(PxU32 islandIndex = 0;
					islandIndex < islandCount; islandIndex++)
				{
					if(activeIslandIds[islandIndex] ==
							entryIslandId &&
						islandBodyCounts[islandIndex] > 0)
					{
						wakeEntry(
							entry,
							ScInternalWakeCounterResetValue);
						break;
					}
				}
			}

			PxU32 awakeEntryCount = 0;
			for(PxU32 i = 0; i < mEntries.size(); i++)
			{
				if(mEntries[i].sleeping)
					continue;
				syncHostInputs(
					mEntries[i], mDeformableMaterialManager);
				awakeEntryCount++;
			}
			if(awakeEntryCount == 0)
				return false;

			// AVBD consumes the articulation response strictly as a
			// generalized inverse-mass operator for its position owner.
			// Refresh it once per attached articulation at prep time; do not
			// enter Featherstone's velocity-impulse solve.
			for(PxU32 attachmentIndex = 0;
				attachmentIndex <
					mArticulationAttachments.size();
				attachmentIndex++)
			{
				BodySim* linkSim =
					mArticulationAttachments[attachmentIndex].
						linkCore->getSim();
				Dy::FeatherstoneArticulation* articulation =
					linkSim ? linkSim->getArticulation() : NULL;
				if(!articulation)
					return false;
				bool alreadyPrepared = false;
				for(PxU32 priorIndex = 0;
					priorIndex < attachmentIndex; priorIndex++)
				{
					BodySim* priorLinkSim =
						mArticulationAttachments[priorIndex].
							linkCore->getSim();
					if(priorLinkSim &&
						priorLinkSim->getArticulation() ==
							articulation)
					{
						alreadyPrepared = true;
						break;
					}
				}
				if(!alreadyPrepared)
					articulation->
						prepareAvbdGeneralizedPositionResponse();
			}

			compileWorldStatics(mRigidMaterialManager);
			for(PxU32 i = 0; i < mIslandSelectionStorages.size(); i++)
			{
				mIslandSelectionStorages[i]->touched = false;
				mIslandSelectionStorages[i]->entryIndices.clear();
				mIslandSelectionStorages[i]->selectedIsland = PX_MAX_U32;
			}

			for(PxU32 i = 0; i < mEntries.size(); i++)
			{
				if(mEntries[i].sleeping)
					continue;
				const PxNodeIndex node = mEntries[i].islandNode;
				if(!node.isValid() ||
					node.index() >= islandSim.getNbNodes())
					return false;
				const IG::IslandId entryIslandId =
					islandSim.getIslandIds()[node.index()];
				if(entryIslandId == IG_INVALID_ISLAND)
					return false;

				IslandSelectionStorage* storage =
					acquireIslandSelectionStorage(entryIslandId);
				if(!storage)
					return false;
				storage->entryIndices.pushBack(i);
			}

			auto discardNativeSelections = [&]()
			{
				selections.clear();
				invalidateNativeIslandSelectionCaches();
			};

			PxU32 selectedEntryCount = 0;
			IslandSelectionStorage* singleNativeStorage = NULL;
			// A later all-awake promotion is safe only when every provisional
			// selection uses the same active rigid island.  Several disconnected
			// soft components may legitimately contact one dynamic rigid in the
			// same frame; they can be rebuilt as one native selection.  Distinct
			// rigid islands remain an ownership boundary and must fall back.
			PxU32 nativeSelectionIsland = PX_MAX_U32;
			bool nativeSelectionsShareIsland = true;
			bool duplicateNativeSelectionIsland = false;
			for(PxU32 storageIndex = 0;
				storageIndex < mIslandSelectionStorages.size();
				storageIndex++)
			{
				IslandSelectionStorage& storage =
					*mIslandSelectionStorages[storageIndex];
				if(!storage.touched || storage.entryIndices.empty())
					continue;

				for(PxU32 islandIndex = 0;
					islandIndex < islandCount; islandIndex++)
				{
					if(activeIslandIds[islandIndex] ==
						storage.nativeIslandId)
					{
						storage.selectedIsland = islandIndex;
						break;
					}
				}
				// A soft-rigid edge created during this frame's predictive
				// topology pass is visible to the speculative graph before
				// the accurate graph has necessarily merged its islands.
				// When the soft side still resolves to an empty native island,
				// bridge selection to the one unambiguous active rigid island
				// named by that same native edge. The edge remains authoritative
				// topology and the normal accurate-island merge owns later
				// frames; no out-of-island rigid index is ever fabricated.
				if(storage.selectedIsland == PX_MAX_U32 ||
					islandBodyCounts[storage.selectedIsland] == 0)
				{
					PxU32 bridgeIsland = PX_MAX_U32;
					bool bridgeAmbiguous = false;
					for(PxU32 entryOrder = 0;
						entryOrder < storage.entryIndices.size();
						entryOrder++)
					{
						const ActorCore* softCore =
							mEntries[
								storage.entryIndices[entryOrder]].
								getActorCore();
						for(PxU32 edgeIndex = 0;
							edgeIndex < mNativeIslandEdges.size();
							edgeIndex++)
						{
							const NativeIslandEdgeEntry& edge =
								mNativeIslandEdges[edgeIndex];
							if(!edge.touched ||
								edge.softCore != softCore)
								continue;
							BodySim* rigidSim =
								edge.rigidCore->getSim();
							if(!rigidSim)
								continue;
							const PxNodeIndex rigidNode =
								rigidSim->getNodeIndex();
							if(!rigidNode.isValid() ||
								rigidNode.index() >=
									islandSim.getNbNodes())
								continue;
							const IG::IslandId rigidIslandId =
								islandSim.getIslandIds()[
									rigidNode.index()];
							for(PxU32 islandIndex = 0;
								islandIndex < islandCount;
								islandIndex++)
							{
								if(activeIslandIds[islandIndex] !=
										rigidIslandId ||
									islandBodyCounts[islandIndex] == 0)
									continue;
								if(bridgeIsland == PX_MAX_U32)
									bridgeIsland = islandIndex;
								else if(bridgeIsland != islandIndex)
									bridgeAmbiguous = true;
								break;
							}
						}
					}
					if(!bridgeAmbiguous &&
						bridgeIsland != PX_MAX_U32)
						storage.selectedIsland = bridgeIsland;
				}
				if(storage.selectedIsland != PX_MAX_U32 &&
					islandBodyCounts[storage.selectedIsland] > 0)
				{
					for(PxU32 entryIndex = 0;
						entryIndex < storage.entryIndices.size();
						entryIndex++)
					{
						Entry& entry =
							mEntries[
								storage.entryIndices[entryIndex]];
						if(entry.sleeping)
							wakeEntry(
								entry,
								ScInternalWakeCounterResetValue);
					}
				}
				if(storage.selectedIsland == PX_MAX_U32 ||
					!buildIslandSelectionStorage(
						storage, solverBodies, rigidBodies,
						articulationForBody, linkIndexForBody,
						islandBodyStarts[storage.selectedIsland],
						islandBodyCounts[storage.selectedIsland],
						dt, gravity))
				{
					// This soft-only/native island has no unified rigid or
					// generalized target. Leave it for the component fallback
					// without discarding independent complete selections.
					storage.touched = false;
					storage.selectedIsland = PX_MAX_U32;
					continue;
				}

				PxU32 innerIterations = 1;
				for(PxU32 entryIndex = 0;
					entryIndex < storage.entryIndices.size();
					entryIndex++)
				{
					const Entry& entry =
						mEntries[storage.entryIndices[entryIndex]];
					innerIterations = PxMax<PxU32>(
						innerIterations,
						entry.getSolverIterationCounts() & 0xff);
				}

				Dy::AvbdSoftIslandSelection selection;
				selection.particles =
					getIslandSelectionParticles(storage);
				selection.numParticles =
					getIslandSelectionParticleCount(storage);
				selection.bodies = storage.bodies.begin();
				selection.numBodies = storage.bodies.size();
				selection.contacts = storage.contacts.begin();
				selection.numContacts = storage.contacts.size();
				selection.islandIndex = storage.selectedIsland;
				selection.iterationOverride = innerIterations;
				selection.executionPlan.particleBodyIndices =
					storage.particleBodyIndices.begin();
				selection.executionPlan.numParticleBodyIndices =
					storage.particleBodyIndices.size();
				selection.executionPlan.contactStarts =
					storage.contactStarts.begin();
				selection.executionPlan.numContactStarts =
					storage.contactStarts.size();
				selection.executionPlan.contactRefs =
					storage.contactRefs.begin();
				selection.executionPlan.numContactRefs =
					storage.contactRefs.size();
				selection.executionPlan.triangleCoreSafetyStarts =
					storage.triangleCoreSafetyStarts.begin();
				selection.executionPlan.numTriangleCoreSafetyStarts =
					storage.triangleCoreSafetyStarts.size();
				selection.executionPlan.triangleCoreSafetyRefs =
					storage.triangleCoreSafetyRefs.begin();
				selection.executionPlan.numTriangleCoreSafetyRefs =
					storage.triangleCoreSafetyRefs.size();
				selection.executionPlan.rigidTargetContactStarts =
					storage.rigidTargetContactStarts.begin();
				selection.executionPlan.numRigidTargetContactStarts =
					storage.rigidTargetContactStarts.size();
				selection.executionPlan.rigidTargetContactRefs =
					storage.rigidTargetContactRefs.begin();
				selection.executionPlan.numRigidTargetContactRefs =
					storage.rigidTargetContactRefs.size();
				configureTerminalOgcGeometryProvider(
					selection.executionPlan, storage, mWorldPlanes,
					mRigidTriangleSurfaces, mContactParams.contactRadius);
				selection.executionPlan.ogcPairStates =
					storage.ogcPairStates.begin();
				selection.executionPlan.numOgcPairStates =
					storage.ogcPairStates.size();
				selection.executionPlan.ogcPairIndices =
					storage.ogcPairIndices.begin();
				selection.executionPlan.numOgcPairIndices =
					storage.ogcPairIndices.size();
				selection.executionPlan.ogcPairContactStarts =
					storage.ogcPairContactStarts.begin();
				selection.executionPlan.numOgcPairContactStarts =
					storage.ogcPairContactStarts.size();
				selection.executionPlan.ogcPairContactRefs =
					storage.ogcPairContactRefs.begin();
				selection.executionPlan.numOgcPairContactRefs =
					storage.ogcPairContactRefs.size();
				selection.executionPlan.ogcTriangleCoreCertificates =
					storage.ogcGeometrySidecar.triangleCoreCertificates.begin();
				selection.executionPlan.numOgcTriangleCoreCertificates =
					storage.ogcGeometrySidecar.triangleCoreCertificates.size();
				selection.executionPlan.ogcContactTriangleCoreIndices =
					storage.ogcGeometrySidecar.contactTriangleCoreIndices.begin();
				selection.executionPlan.numOgcContactTriangleCoreIndices =
					storage.ogcGeometrySidecar.contactTriangleCoreIndices.size();
				selection.executionPlan.ogcGeometryEpoch =
					storage.ogcGeometrySidecar.geometryEpoch;
				selection.executionPlan.postAlWorkspace =
					&storage.postAlWorkspace;
				selection.executionPlan.ogcAdmissionWorkspace =
					&storage.postAlWorkspace.poseWriteAdmission.scratch;
				if(!selection.isComplete() ||
					!selection.executionPlan.isComplete(
						selection.numParticles))
				{
					discardNativeSelections();
					return false;
				}
				// `buildIslandSelectionStorage()` has already published the exact
				// same dt/gravity prediction into this particle buffer for swept
				// contact selection.  Only expose that lifecycle token after the
				// base support program has passed its provider-boundary check.
				selection.executionPlan.softPredictionPrepared = true;
				for(PxU32 priorIndex = 0;
					priorIndex < selections.size(); ++priorIndex)
				{
					if(selections[priorIndex].islandIndex ==
						storage.selectedIsland)
					{
						duplicateNativeSelectionIsland = true;
						break;
					}
				}
				selections.pushBack(selection);
				if(singleNativeStorage)
				{
					if(nativeSelectionIsland != storage.selectedIsland)
						nativeSelectionsShareIsland = false;
				}
				else
				{
					singleNativeStorage = &storage;
					nativeSelectionIsland = storage.selectedIsland;
				}
				selectedEntryCount += storage.entryIndices.size();
			}

			// A component fallback has no selected-entry mask today.  Letting it
			// run after a partial native selection advances selected particles and
			// contact state once through the component route and then overwrites
			// only part of that state with the native copyback.  That is not a
			// valid ownership split.  Likewise, the dynamics provider requires
			// every published selection to name a unique rigid island.  Until the
			// indexed fallback view owns a disjoint subset, both cases must become
			// one all-awake native selection or fall back completely.
			const bool needsSingleOwnerPromotion =
				selectedEntryCount != awakeEntryCount ||
				duplicateNativeSelectionIsland;
			if(needsSingleOwnerPromotion)
			{
				// A single dynamic-rigid island can safely own the entire awake
				// soft scene when every dynamic shape which has a predictive edge to
				// those actors belongs to that same rigid island. Unrelated dynamic
				// actors elsewhere in the Scene are not part of this ownership
				// transaction and must not invalidate it. Attachments are deliberately
				// excluded: their generalized owners require the original island
				// membership to be preserved exactly.
				if(!singleNativeStorage || !nativeSelectionsShareIsland ||
					!mWorldPins.empty() ||
					!mRigidAttachments.empty() ||
					!mArticulationAttachments.empty() ||
					!mSoftPairAttachments.empty() ||
					!mPrescribedAttachments.empty() ||
					singleNativeStorage->selectedIsland >= islandCount)
				{
					discardNativeSelections();
					return false;
				}

				IslandSelectionStorage& storage = *singleNativeStorage;
				bool hasRigidBodyTarget = false;
				for(PxU32 contactIndex = 0;
					contactIndex < storage.contacts.size(); ++contactIndex)
				{
					const Dy::AvbdSoftContactGeometry& geometry =
						storage.contacts[contactIndex].geometry;
					if(geometry.hasRigidBodyTarget())
					{
						if(geometry.targetIndex >=
							islandBodyCounts[storage.selectedIsland])
						{
							discardNativeSelections();
							return false;
						}
						hasRigidBodyTarget = true;
					}
				}
				const PxU32 selectedBodyStart =
					islandBodyStarts[storage.selectedIsland];
				const PxU32 selectedBodyCount =
					islandBodyCounts[storage.selectedIsland];
				if(!hasRigidBodyTarget || selectedBodyCount == 0)
				{
					discardNativeSelections();
					return false;
				}

				// A promoted selection may only see dynamic shapes whose two-sided
				// rigid response lives in its selected island.  Static and
				// kinematic shapes remain ordinary world/prescribed targets.
				for(PxU32 shapeIndex = 0;
					shapeIndex < mDynamicShapes.size(); ++shapeIndex)
				{
					const DynamicShapeEntry& dynamicEntry =
						mDynamicShapes[shapeIndex];
					if(!dynamicEntry.core || !dynamicEntry.shape)
					{
						discardNativeSelections();
						return false;
					}
					const ShapeCore& shape = *dynamicEntry.shape;
					if(!(shape.getFlags() &
							PxShapeFlag::eSIMULATION_SHAPE))
						continue;
					BodySim* const bodySim =
						dynamicEntry.core->getSim();
					if(!bodySim)
					{
						discardNativeSelections();
						return false;
					}
					if(bodySim->isArticulationLink())
					{
						discardNativeSelections();
						return false;
					}
					if(bodySim->isKinematic())
						continue;
					bool touchesPromotedSoft = false;
					for(PxU32 edgeIndex = 0;
						edgeIndex < mNativeIslandEdges.size(); ++edgeIndex)
					{
						const NativeIslandEdgeEntry& edge =
							mNativeIslandEdges[edgeIndex];
						if(!edge.touched ||
							edge.rigidCore != dynamicEntry.core)
							continue;
						for(PxU32 entryIndex = 0;
							entryIndex < mEntries.size(); ++entryIndex)
						{
							const Entry& promotedEntry = mEntries[entryIndex];
							if(!promotedEntry.sleeping &&
								edge.softCore == promotedEntry.getActorCore())
							{
								touchesPromotedSoft = true;
								break;
							}
						}
						if(touchesPromotedSoft)
							break;
					}
					if(!touchesPromotedSoft)
						continue;
					const PxGeometryType::Enum geometryType =
						shape.getGeometryType();
					if(geometryType != PxGeometryType::eBOX &&
						geometryType != PxGeometryType::eSPHERE &&
						geometryType != PxGeometryType::eCAPSULE &&
						geometryType != PxGeometryType::eCONVEXMESH)
					{
						discardNativeSelections();
						return false;
					}
					PxU32 localRigidBodyIndex = PX_MAX_U32;
					if(!findRigidBodyIndexInIsland(
							*dynamicEntry.core, rigidBodies, solverBodies,
							selectedBodyStart, selectedBodyCount,
							localRigidBodyIndex))
					{
						discardNativeSelections();
						return false;
					}
				}

				for(PxU32 storageIndex = 0;
					storageIndex < mIslandSelectionStorages.size();
					++storageIndex)
				{
					IslandSelectionStorage& candidate =
						*mIslandSelectionStorages[storageIndex];
					if(&candidate != &storage)
						candidate.touched = false;
				}

				// Existing selections point into storage-owned arrays. The old
				// selection is no longer a valid view once the promoted storage is
				// rebuilt, so remove it before changing those arrays.
				selections.clear();
				storage.entryIndices.clear();
				for(PxU32 entryIndex = 0;
					entryIndex < mEntries.size(); ++entryIndex)
					if(!mEntries[entryIndex].sleeping)
						storage.entryIndices.pushBack(entryIndex);
				if(storage.entryIndices.size() != awakeEntryCount ||
					!buildIslandSelectionStorage(
						storage, solverBodies, rigidBodies,
						articulationForBody, linkIndexForBody,
						selectedBodyStart, selectedBodyCount, dt, gravity))
				{
					discardNativeSelections();
					return false;
				}

				hasRigidBodyTarget = false;
				for(PxU32 contactIndex = 0;
					contactIndex < storage.contacts.size(); ++contactIndex)
				{
					const Dy::AvbdSoftContactGeometry& geometry =
						storage.contacts[contactIndex].geometry;
					if(geometry.hasRigidBodyTarget())
					{
						if(geometry.targetIndex >= selectedBodyCount)
						{
							discardNativeSelections();
							return false;
						}
						hasRigidBodyTarget = true;
					}
				}
				if(!hasRigidBodyTarget)
				{
					discardNativeSelections();
					return false;
				}

				PxU32 innerIterations = 1;
				for(PxU32 entryIndex = 0;
					entryIndex < storage.entryIndices.size(); ++entryIndex)
				{
					const Entry& entry =
						mEntries[storage.entryIndices[entryIndex]];
					innerIterations = PxMax<PxU32>(
						innerIterations,
						entry.getSolverIterationCounts() & 0xff);
				}

				Dy::AvbdSoftIslandSelection promotedSelection;
				promotedSelection.particles =
					getIslandSelectionParticles(storage);
				promotedSelection.numParticles =
					getIslandSelectionParticleCount(storage);
				promotedSelection.bodies = storage.bodies.begin();
				promotedSelection.numBodies = storage.bodies.size();
				promotedSelection.contacts = storage.contacts.begin();
				promotedSelection.numContacts = storage.contacts.size();
				promotedSelection.islandIndex = storage.selectedIsland;
				promotedSelection.iterationOverride = innerIterations;
				promotedSelection.executionPlan.particleBodyIndices =
					storage.particleBodyIndices.begin();
				promotedSelection.executionPlan.numParticleBodyIndices =
					storage.particleBodyIndices.size();
				promotedSelection.executionPlan.contactStarts =
					storage.contactStarts.begin();
				promotedSelection.executionPlan.numContactStarts =
					storage.contactStarts.size();
				promotedSelection.executionPlan.contactRefs =
					storage.contactRefs.begin();
				promotedSelection.executionPlan.numContactRefs =
					storage.contactRefs.size();
				promotedSelection.executionPlan.triangleCoreSafetyStarts =
					storage.triangleCoreSafetyStarts.begin();
				promotedSelection.executionPlan.numTriangleCoreSafetyStarts =
					storage.triangleCoreSafetyStarts.size();
				promotedSelection.executionPlan.triangleCoreSafetyRefs =
					storage.triangleCoreSafetyRefs.begin();
				promotedSelection.executionPlan.numTriangleCoreSafetyRefs =
					storage.triangleCoreSafetyRefs.size();
				promotedSelection.executionPlan.rigidTargetContactStarts =
					storage.rigidTargetContactStarts.begin();
				promotedSelection.executionPlan.numRigidTargetContactStarts =
					storage.rigidTargetContactStarts.size();
				promotedSelection.executionPlan.rigidTargetContactRefs =
					storage.rigidTargetContactRefs.begin();
				promotedSelection.executionPlan.numRigidTargetContactRefs =
					storage.rigidTargetContactRefs.size();
				configureTerminalOgcGeometryProvider(
					promotedSelection.executionPlan, storage, mWorldPlanes,
					mRigidTriangleSurfaces, mContactParams.contactRadius);
				promotedSelection.executionPlan.ogcPairStates =
					storage.ogcPairStates.begin();
				promotedSelection.executionPlan.numOgcPairStates =
					storage.ogcPairStates.size();
				promotedSelection.executionPlan.ogcPairIndices =
					storage.ogcPairIndices.begin();
				promotedSelection.executionPlan.numOgcPairIndices =
					storage.ogcPairIndices.size();
				promotedSelection.executionPlan.ogcPairContactStarts =
					storage.ogcPairContactStarts.begin();
				promotedSelection.executionPlan.numOgcPairContactStarts =
					storage.ogcPairContactStarts.size();
				promotedSelection.executionPlan.ogcPairContactRefs =
					storage.ogcPairContactRefs.begin();
				promotedSelection.executionPlan.numOgcPairContactRefs =
					storage.ogcPairContactRefs.size();
				promotedSelection.executionPlan.ogcTriangleCoreCertificates =
					storage.ogcGeometrySidecar.triangleCoreCertificates.begin();
				promotedSelection.executionPlan.numOgcTriangleCoreCertificates =
					storage.ogcGeometrySidecar.triangleCoreCertificates.size();
				promotedSelection.executionPlan.ogcContactTriangleCoreIndices =
					storage.ogcGeometrySidecar.contactTriangleCoreIndices.begin();
				promotedSelection.executionPlan.numOgcContactTriangleCoreIndices =
					storage.ogcGeometrySidecar.contactTriangleCoreIndices.size();
				promotedSelection.executionPlan.ogcGeometryEpoch =
					storage.ogcGeometrySidecar.geometryEpoch;
				promotedSelection.executionPlan.postAlWorkspace =
					&storage.postAlWorkspace;
				promotedSelection.executionPlan.ogcAdmissionWorkspace =
					&storage.postAlWorkspace.poseWriteAdmission.scratch;
				if(!promotedSelection.isComplete() ||
					!promotedSelection.executionPlan.isComplete(
						promotedSelection.numParticles))
				{
					discardNativeSelections();
					return false;
				}
				promotedSelection.executionPlan.softPredictionPrepared = true;
				selections.clear();
				selections.pushBack(promotedSelection);
				selectedEntryCount = storage.entryIndices.size();
			}

			// Publish native ownership only for a complete, island-disjoint cover.
			// This is the final provider-boundary invariant: no later component
			// stage has a selected-entry mask, and LowLevelDynamics requires one
			// owner per active rigid island. Any routing regression therefore fails
			// closed before either solver can mutate canonical state.
			if(!selections.empty())
			{
				if(selectedEntryCount != awakeEntryCount)
				{
					discardNativeSelections();
					return false;
				}
				for(PxU32 selectionIndex = 0;
					selectionIndex < selections.size(); ++selectionIndex)
				{
					if(!selections[selectionIndex].isComplete())
					{
						discardNativeSelections();
						return false;
					}
					for(PxU32 priorIndex = 0;
						priorIndex < selectionIndex; ++priorIndex)
					{
						if(selections[priorIndex].islandIndex ==
							selections[selectionIndex].islandIndex)
						{
							discardNativeSelections();
							return false;
						}
					}
				}
			}

			mDynamicsOwnsStep = !selections.empty();
			mDynamicsSelectedEntryCount =
				mDynamicsOwnsStep ? selectedEntryCount : 0;
			return mDynamicsOwnsStep;
		}

} // namespace Sc
} // namespace physx
