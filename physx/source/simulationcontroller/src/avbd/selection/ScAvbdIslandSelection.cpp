// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/scene/ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

void AvbdCpuSoftScene::copyIslandSelectionResults(
			IslandSelectionStorage& storage) {
			if(!storage.usesCanonicalParticleRange)
			{
				PX_ASSERT(
					storage.particles.size() ==
					storage.globalParticleIndices.size());
				const PxU32 particleCount = PxMin(
					storage.particles.size(),
					storage.globalParticleIndices.size());
				for(PxU32 i = 0; i < particleCount; i++)
				{
					const PxU32 globalIndex =
						storage.globalParticleIndices[i];
					if(globalIndex < mParticles.size())
						mParticles[globalIndex] = storage.particles[i];
				}
			}
			else
			{
				// The native solver was bound directly to this canonical range;
				// only the separately rebound AL runtime state below requires a
				// transfer back to the Scene owner.
				PX_ASSERT(storage.particles.empty());
				PX_ASSERT(storage.globalParticleIndices.empty());
			}
			PX_ASSERT(
				storage.bodies.size() == storage.entryIndices.size());
			const PxU32 bodyCount = PxMin(
				storage.bodies.size(), storage.entryIndices.size());
			for(PxU32 i = 0; i < bodyCount; i++)
			{
				const Entry& entry =
					mEntries[storage.entryIndices[i]];
				if(entry.bodyIndex < mBodies.size())
				{
					Dy::AvbdSoftBodyRuntimeState& destination =
						mBodies[entry.bodyIndex].runtime;
					const Dy::AvbdSoftBodyRuntimeState& source =
						storage.bodies[i].runtime;
					PX_ASSERT(
						destination.pins.size() ==
							source.pins.size());
					const PxU32 pinCount = PxMin(
						destination.pins.size(),
						source.pins.size());
					for(PxU32 pinIndex = 0;
						pinIndex < pinCount; pinIndex++)
					{
						const Dy::AvbdKinematicPin& sourcePin =
							source.pins[pinIndex];
						destination.pins[pinIndex].alLambda =
							sourcePin.alLambda;
						destination.pins[pinIndex].k =
							sourcePin.k;
						destination.pins[pinIndex].kMax =
							sourcePin.kMax;
						if(sourcePin.targetKind ==
							Dy::AvbdSoftPinTargetKind::
								ePRESCRIBED_RIGID)
						{
							PrescribedAttachmentEntry*
								destinationAttachment =
									findPrescribedAttachment(
										*entry.getActorCore(),
										sourcePin.sourceHandle);
							if(destinationAttachment)
							{
								destinationAttachment->alLambda =
									sourcePin.alLambda;
								destinationAttachment->k =
									sourcePin.k;
								destinationAttachment->kMax =
									sourcePin.kMax;
							}
						}
					}
					for(PxU32 attachmentIndex = 0;
						attachmentIndex <
							source.attachments.size();
						attachmentIndex++)
					{
						const Dy::AvbdSoftAttachment&
							sourceAttachment =
								source.attachments[
									attachmentIndex];
						switch(sourceAttachment.targetKind)
						{
						case Dy::AvbdSoftAttachmentTargetKind::
							eDYNAMIC_RIGID:
						{
							const PxU32 handle =
								sourceAttachment.sourceHandle;
							RigidAttachmentEntry*
								destinationAttachment =
									findRigidAttachment(
										*entry.getActorCore(),
										handle);
							if(!destinationAttachment)
								continue;
							destinationAttachment->alLambda =
								sourceAttachment.alLambda;
							destinationAttachment->k =
								sourceAttachment.k;
							destinationAttachment->kMax =
								sourceAttachment.kMax;
							break;
						}
						case Dy::AvbdSoftAttachmentTargetKind::
							eARTICULATION_LINK:
						{
							const PxU32 handle =
								sourceAttachment.sourceHandle;
							ArticulationAttachmentEntry*
								destinationAttachment =
									findArticulationAttachment(
										*entry.getActorCore(),
										handle);
							if(!destinationAttachment)
								continue;
							destinationAttachment->alLambda =
								sourceAttachment.alLambda;
							destinationAttachment->k =
								sourceAttachment.k;
							destinationAttachment->kMax =
								sourceAttachment.kMax;
							break;
						}
						case Dy::AvbdSoftAttachmentTargetKind::
							eDYNAMIC_SOFT:
						{
							const PxU32 handle =
								sourceAttachment.sourceHandle;
							SoftPairAttachmentEntry*
								destinationAttachment =
									findSoftPairAttachment(
										*entry.getActorCore(),
										handle);
							if(!destinationAttachment)
								continue;
							destinationAttachment->alLambda =
								sourceAttachment.alLambda;
							destinationAttachment->k =
								sourceAttachment.k;
							destinationAttachment->kMax =
								sourceAttachment.kMax;
							break;
						}
						case Dy::AvbdSoftAttachmentTargetKind::
							eUNSUPPORTED:
						default:
							PX_ASSERT(false);
							break;
						}
					}
					PX_ASSERT(destination.attachments.empty());
					destination.compileObjectiveProgram(
						getParticleStart(entry),
						getParticleCount(entry));
				}
			}
		}
// Native island topology edge ownership is kept outside ScScene's facade body.

PxU32 AvbdCpuSoftScene::findNativeIslandEdge(
			const ActorCore* softCore,
			const BodyCore* rigidCore) const {
			for(PxU32 i = 0;
				i < mNativeIslandEdges.size(); i++)
			{
				const NativeIslandEdgeEntry& entry =
					mNativeIslandEdges[i];
				if(entry.softCore == softCore &&
					entry.rigidCore == rigidCore)
					return i;
			}
			return PX_MAX_U32;
		}

void AvbdCpuSoftScene::ensureNativeIslandEdge(
			Entry& softEntry, BodyCore& rigidCore) {
			const PxU32 existingIndex =
				findNativeIslandEdge(
					softEntry.getActorCore(), &rigidCore);
			if(existingIndex != PX_MAX_U32)
			{
				mNativeIslandEdges[existingIndex].touched = true;
				return;
			}

			BodySim* bodySim = rigidCore.getSim();
			if(!bodySim || !bodySim->getNodeIndex().isValid())
				return;
			const IG::EdgeIndex edgeIndex =
				mIslandManager.addContactManager(
					NULL, softEntry.islandNode,
					bodySim->getNodeIndex(), NULL,
					IG::Edge::eSOFT_BODY_CONTACT);
			mIslandManager.setEdgeConnected(
				edgeIndex, IG::Edge::eSOFT_BODY_CONTACT);
			mNativeIslandEdges.pushBack(
				NativeIslandEdgeEntry(
					*softEntry.getActorCore(),
					rigidCore, edgeIndex));
		}

PxU32 AvbdCpuSoftScene::findNativeSoftSoftIslandEdge(
			const ActorCore* softCore0,
			const ActorCore* softCore1) const {
			for(PxU32 i = 0;
				i < mNativeSoftSoftIslandEdges.size(); i++)
			{
				const NativeSoftSoftIslandEdgeEntry& entry =
					mNativeSoftSoftIslandEdges[i];
				if((entry.softCore0 == softCore0 &&
						entry.softCore1 == softCore1) ||
					(entry.softCore0 == softCore1 &&
						entry.softCore1 == softCore0))
					return i;
			}
			return PX_MAX_U32;
		}

void AvbdCpuSoftScene::ensureNativeSoftSoftIslandEdge(
			Entry& softEntry0, Entry& softEntry1) {
			const PxU32 existingIndex =
				findNativeSoftSoftIslandEdge(
					softEntry0.getActorCore(),
					softEntry1.getActorCore());
			if(existingIndex != PX_MAX_U32)
			{
				mNativeSoftSoftIslandEdges[existingIndex].touched = true;
				return;
			}
			if(!softEntry0.islandNode.isValid() ||
				!softEntry1.islandNode.isValid())
				return;

			const IG::EdgeIndex edgeIndex =
				mIslandManager.addContactManager(
					NULL, softEntry0.islandNode,
					softEntry1.islandNode, NULL,
					IG::Edge::eSOFT_BODY_CONTACT);
			mIslandManager.setEdgeConnected(
				edgeIndex, IG::Edge::eSOFT_BODY_CONTACT);
			mNativeSoftSoftIslandEdges.pushBack(
				NativeSoftSoftIslandEdgeEntry(
					*softEntry0.getActorCore(),
					*softEntry1.getActorCore(), edgeIndex));
		}

void AvbdCpuSoftScene::removeNativeIslandEdgesForRigid(BodyCore& core) {
			for(PxU32 i = mNativeIslandEdges.size();
				i > 0; i--)
			{
				if(mNativeIslandEdges[i - 1].rigidCore == &core)
				{
					mIslandManager.removeConnection(
						mNativeIslandEdges[i - 1].edgeIndex);
					mNativeIslandEdges.replaceWithLast(i - 1);
				}
			}
		}

void AvbdCpuSoftScene::removeNativeIslandEdgesForSoft(
			ActorCore& core) {
			for(PxU32 i = mNativeIslandEdges.size();
				i > 0; i--)
			{
				if(mNativeIslandEdges[i - 1].softCore == &core)
				{
					mIslandManager.removeConnection(
						mNativeIslandEdges[i - 1].edgeIndex);
					mNativeIslandEdges.replaceWithLast(i - 1);
				}
			}
			for(PxU32 i = mNativeSoftSoftIslandEdges.size();
				i > 0; i--)
			{
				const NativeSoftSoftIslandEdgeEntry& edge =
					mNativeSoftSoftIslandEdges[i - 1];
				if(edge.softCore0 == &core ||
					edge.softCore1 == &core)
				{
					mIslandManager.removeConnection(edge.edgeIndex);
					mNativeSoftSoftIslandEdges.replaceWithLast(i - 1);
				}
			}
		}

void AvbdCpuSoftScene::clearNativeIslandEdges() {
			for(PxU32 i = 0;
				i < mNativeIslandEdges.size(); i++)
				mIslandManager.removeConnection(
					mNativeIslandEdges[i].edgeIndex);
			mNativeIslandEdges.clear();
			for(PxU32 i = 0;
				i < mNativeSoftSoftIslandEdges.size(); i++)
				mIslandManager.removeConnection(
					mNativeSoftSoftIslandEdges[i].edgeIndex);
			mNativeSoftSoftIslandEdges.clear();
		}
// Native island selection compilation and contact ownership implementation.

bool AvbdCpuSoftScene::buildIslandSelectionStorage(
			IslandSelectionStorage& storage,
			Dy::AvbdSolverBody* solverBodies,
			PxsRigidBody* const* rigidBodies,
			Dy::FeatherstoneArticulation* const*
				articulationForBody,
			const PxU32* linkIndexForBody,
			PxU32 bodyStart, PxU32 bodyCount,
			PxReal dt, const PxVec3& gravity) {
			bool membershipMatches =
				storage.softCores.size() == storage.entryIndices.size();
			for(PxU32 i = 0;
				membershipMatches && i < storage.entryIndices.size(); i++)
			{
				membershipMatches =
					storage.softCores[i] ==
					mEntries[storage.entryIndices[i]].
						getActorCore();
			}
			if(!membershipMatches)
			{
				storage.contacts.clear();
				storage.ogcGeometrySidecar.clear();
			}
			storage.softCores.clear();
			storage.usesCanonicalParticleRange = false;
			storage.canonicalParticleStart = PX_MAX_U32;
			storage.canonicalParticleCount = 0;
			storage.globalParticleIndices.clear();
			storage.particles.clear();
			storage.bodies.clear();
			storage.particleBodyIndices.clear();
			storage.contactStarts.clear();
			storage.contactCounts.clear();
			storage.contactRefs.clear();
			storage.triangleCoreSafetyStarts.clear();
			storage.triangleCoreSafetyCounts.clear();
			storage.triangleCoreSafetyRefs.clear();
			storage.rigidTargetContactStarts.clear();
			storage.rigidTargetContactCounts.clear();
			storage.rigidTargetContactRefs.clear();
			storage.selfCollisionAdjacencies.clear();
			storage.selfCollisionEnabled.clear();
			storage.rigidBoxes.clear();
			storage.selectedDynamicBoxes.clear();
			storage.terminalCollisionBodies.clear();
			storage.terminalCollisionVertexMappings.clear();
			storage.ogcPairStates.clear();
			storage.ogcGeometrySidecar.clear();
			storage.ogcPairIndices.clear();
			storage.rigidSpheres.clear();
			storage.selectedDynamicSpheres.clear();
			storage.rigidCapsules.clear();
			storage.selectedDynamicCapsules.clear();
			storage.rigidConvexes.clear();
			storage.selectedDynamicConvexes.clear();
			storage.probeContacts.clear();

			PxU32 canonicalParticleStart = PX_MAX_U32;
			PxU32 canonicalParticleCount = 0;
			if(getCanonicalIslandParticleRange(
				storage, canonicalParticleStart,
				canonicalParticleCount))
			{
				storage.usesCanonicalParticleRange = true;
				storage.canonicalParticleStart =
					canonicalParticleStart;
				storage.canonicalParticleCount =
					canonicalParticleCount;
			}

			for(PxU32 entryOrder = 0;
				entryOrder < storage.entryIndices.size(); entryOrder++)
			{
				const Entry& entry =
					mEntries[storage.entryIndices[entryOrder]];
				storage.softCores.pushBack(entry.getActorCore());
				const PxU32 particleStart =
					getParticleStart(entry);
				const PxU32 particleCount =
					getParticleCount(entry);
				const PxU32 localStart =
					storage.usesCanonicalParticleRange
						? particleStart -
							storage.canonicalParticleStart
						: storage.particles.size();
				if(!storage.usesCanonicalParticleRange)
				{
					for(PxU32 i = 0; i < particleCount; i++)
					{
						const PxU32 globalIndex = particleStart + i;
						if(globalIndex >= mParticles.size())
							return false;
						storage.globalParticleIndices.pushBack(globalIndex);
						storage.particles.pushBack(
							mParticles[globalIndex]);
					}
				}
				if(entry.bodyIndex >= mBodies.size())
					return false;
				if(entry.bodyIndex >=
					mSelfCollisionAdjacencies.size())
					return false;
				Dy::AvbdSoftBody localBody;
				if(!copyAndRebaseSoftBody(
					mBodies[entry.bodyIndex],
					particleStart, particleCount,
					localStart, localBody))
					return false;
				storage.bodies.pushBack(localBody);
				storage.selfCollisionAdjacencies.pushBack(
					mSelfCollisionAdjacencies[entry.bodyIndex]);
				storage.selfCollisionEnabled.pushBack(
					(entry.getBodyCore().bodyFlags &
						PxDeformableBodyFlag::
							eDISABLE_SELF_COLLISION)
					? 0u : 1u);
			}
			// Contact selection happens before the unified AVBD solve's
			// prediction stage. Publish the same current-frame soft prediction
			// now so the non-CCD endpoint-DCD path can select endpoint OGC
			// objectives; a validated native execution plan transfers ownership of
			// this pass to the solver instead of recomputing it before iteration.
			Dy::AvbdSoftParticle* const selectionParticles =
				getIslandSelectionParticles(storage);
			const PxU32 selectionParticleCount =
				getIslandSelectionParticleCount(storage);
			for(PxU32 particleIndex = 0;
				particleIndex < selectionParticleCount; particleIndex++)
				selectionParticles[particleIndex].computePrediction(
					dt, gravity);

			bool hasRigidAttachment = false;
			bool hasArticulationAttachment = false;
			bool hasSoftPairAttachment = false;
			for(PxU32 entryOrder = 0;
				entryOrder < storage.entryIndices.size(); entryOrder++)
			{
				const Entry& entry =
					mEntries[storage.entryIndices[entryOrder]];
				Dy::AvbdSoftBody& localBody =
					storage.bodies[entryOrder];
				for(PxU32 attachmentIndex = 0;
					attachmentIndex < mRigidAttachments.size();
					attachmentIndex++)
				{
					const RigidAttachmentEntry& source =
						mRigidAttachments[attachmentIndex];
					if(source.softCore != entry.getActorCore() ||
						!Dy::avbdIsSoftPointValid(
							source.localPoint, 0,
							localBody.compiled.particleCount))
						continue;
					PxU32 localRigidBodyIndex = PX_MAX_U32;
					if(!findRigidBodyIndexInIsland(
						*source.rigidCore, rigidBodies,
						solverBodies, bodyStart, bodyCount,
						localRigidBodyIndex))
						return false;

					Dy::AvbdSoftAttachment attachment;
					attachment.point = source.localPoint;
					for(PxU32 endpoint = 0;
						endpoint < attachment.point.particleCount;
						endpoint++)
					{
						attachment.point.
							particleIndices[endpoint] +=
								localBody.compiled.particleStart;
					}
					attachment.rigidBodyIdx =
						localRigidBodyIndex;
					attachment.sourceHandle = source.handle;
					attachment.targetKind =
						Dy::AvbdSoftAttachmentTargetKind::
							eDYNAMIC_RIGID;
					attachment.localOffset =
						source.rigidCore->getBody2Actor().
							getInverse().transform(
								source.actorLocalTarget);
					attachment.alLambda = source.alLambda;
					attachment.k = source.k;
					attachment.kMax = source.kMax;
					localBody.runtime.attachments.pushBack(
						attachment);
					hasRigidAttachment = true;
				}
				for(PxU32 attachmentIndex = 0;
					attachmentIndex <
						mArticulationAttachments.size();
					attachmentIndex++)
				{
					const ArticulationAttachmentEntry& source =
						mArticulationAttachments[
							attachmentIndex];
					if(source.softCore != entry.getActorCore() ||
						!Dy::avbdIsSoftPointValid(
							source.localPoint, 0,
							localBody.compiled.particleCount))
						continue;
					PxU32 localLinkBodyIndex = PX_MAX_U32;
					if(!findArticulationBodyIndexInIsland(
						*source.linkCore, articulationForBody,
						linkIndexForBody, solverBodies,
						bodyStart, bodyCount,
						localLinkBodyIndex))
						return false;

					Dy::AvbdSoftAttachment attachment;
					attachment.point = source.localPoint;
					for(PxU32 endpoint = 0;
						endpoint <
							attachment.point.particleCount;
						endpoint++)
						attachment.point.
							particleIndices[endpoint] +=
							localBody.compiled.particleStart;
					attachment.rigidBodyIdx =
						localLinkBodyIndex;
					attachment.sourceHandle = source.handle;
					attachment.targetKind =
						Dy::AvbdSoftAttachmentTargetKind::
							eARTICULATION_LINK;
					attachment.localOffset =
						source.linkCore->getBody2Actor().
							getInverse().transform(
								source.actorLocalTarget);
					attachment.alLambda = source.alLambda;
					attachment.k = source.k;
					attachment.kMax = source.kMax;
					localBody.runtime.attachments.pushBack(
						attachment);
					hasArticulationAttachment = true;
				}
				for(PxU32 attachmentIndex = 0;
					attachmentIndex < mSoftPairAttachments.size();
					attachmentIndex++)
				{
					const SoftPairAttachmentEntry& source =
						mSoftPairAttachments[attachmentIndex];
					if(source.softCore[0] != entry.getActorCore())
						continue;
					PxU32 targetEntryOrder = PX_MAX_U32;
					for(PxU32 candidate = 0;
						candidate < storage.softCores.size();
						candidate++)
					{
						if(storage.softCores[candidate] ==
							source.softCore[1])
						{
							targetEntryOrder = candidate;
							break;
						}
					}
					if(targetEntryOrder == PX_MAX_U32 ||
						targetEntryOrder >= storage.bodies.size())
						return false;
					const Dy::AvbdSoftBody& targetBody =
						storage.bodies[targetEntryOrder];
					if(!Dy::avbdIsSoftPointValid(
							source.localPoint[0], 0,
							localBody.compiled.particleCount) ||
						!Dy::avbdIsSoftPointValid(
							source.localPoint[1], 0,
							targetBody.compiled.particleCount))
						return false;

					Dy::AvbdSoftAttachment attachment;
					attachment.point = source.localPoint[0];
					for(PxU32 endpoint = 0;
						endpoint < attachment.point.particleCount;
						endpoint++)
						attachment.point.
							particleIndices[endpoint] +=
							localBody.compiled.particleStart;
					attachment.targetPoint = source.localPoint[1];
					for(PxU32 endpoint = 0;
						endpoint <
							attachment.targetPoint.particleCount;
						endpoint++)
						attachment.targetPoint.
							particleIndices[endpoint] +=
							targetBody.compiled.particleStart;
					attachment.rigidBodyIdx = PX_MAX_U32;
					attachment.sourceHandle = source.handle;
					attachment.targetKind =
						Dy::AvbdSoftAttachmentTargetKind::
							eDYNAMIC_SOFT;
					attachment.alLambda = source.alLambda;
					attachment.k = source.k;
					attachment.kMax = source.kMax;
					localBody.runtime.attachments.pushBack(
						attachment);
					hasSoftPairAttachment = true;
				}
				localBody.runtime.compileObjectiveProgram(
					localBody.compiled.particleStart,
					localBody.compiled.particleCount);
				if(!localBody.runtime.isObjectiveProgramCurrent(
					localBody.compiled.particleStart,
					localBody.compiled.particleCount))
					return false;
			}

			compileDynamicBoxesForIsland(
				rigidBodies, solverBodies, bodyStart, bodyCount,
				storage.selectedDynamicBoxes);
			compileDynamicSpheresForIsland(
				rigidBodies, solverBodies, bodyStart, bodyCount,
				dt, gravity, storage.selectedDynamicSpheres);
			compileDynamicCapsulesForIsland(
				rigidBodies, solverBodies, bodyStart, bodyCount,
				dt, gravity, storage.selectedDynamicCapsules);
			compileDynamicConvexesForIsland(
				rigidBodies, solverBodies, bodyStart, bodyCount,
				dt, gravity,
				storage.selectedDynamicConvexes);
			if(!storage.selectedDynamicBoxes.empty())
			{
				Dy::avbdDetectSoftRigidSDF(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicBoxes.begin(),
					storage.selectedDynamicBoxes.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					NULL, 0,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidSweptSDF(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicBoxes.begin(),
					storage.selectedDynamicBoxes.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidOGCFeatures(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicBoxes.begin(),
					storage.selectedDynamicBoxes.size(),
					storage.bodies.begin(),
					storage.bodies.size(),
					storage.probeContacts,
					mContactParams.contactRadius);
			}
			if(!storage.selectedDynamicSpheres.empty())
			{
				Dy::avbdDetectSoftRigidSphereSDF(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicSpheres.begin(),
					storage.selectedDynamicSpheres.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidSphereSweptSDF(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicSpheres.begin(),
					storage.selectedDynamicSpheres.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidSphereSweptOGCFeatures(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicSpheres.begin(),
					storage.selectedDynamicSpheres.size(),
					storage.bodies.begin(),
					storage.bodies.size(),
					storage.probeContacts,
					mContactParams.contactRadius);
				Dy::avbdDetectSoftRigidSphereOGCFeatures(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicSpheres.begin(),
					storage.selectedDynamicSpheres.size(),
					storage.bodies.begin(),
					storage.bodies.size(),
					storage.probeContacts,
					mContactParams.contactRadius);
			}
			if(!storage.selectedDynamicCapsules.empty())
			{
				Dy::avbdDetectSoftRigidCapsuleSDF(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicCapsules.begin(),
					storage.selectedDynamicCapsules.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidCapsuleSweptSDF(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicCapsules.begin(),
					storage.selectedDynamicCapsules.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidCapsuleSweptOGCFeatures(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicCapsules.begin(),
					storage.selectedDynamicCapsules.size(),
					storage.bodies.begin(),
					storage.bodies.size(),
					storage.probeContacts,
					mContactParams.contactRadius);
				Dy::avbdDetectSoftRigidCapsuleOGCFeatures(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicCapsules.begin(),
					storage.selectedDynamicCapsules.size(),
					storage.bodies.begin(),
					storage.bodies.size(),
					storage.probeContacts,
					mContactParams.contactRadius);
			}
			if(!storage.selectedDynamicConvexes.empty())
			{
				Dy::avbdDetectSoftRigidConvexSDF(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicConvexes.begin(),
					storage.selectedDynamicConvexes.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidConvexSweptSDF(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicConvexes.begin(),
					storage.selectedDynamicConvexes.size(),
					storage.probeContacts,
					mContactParams.contactRadius,
					storage.bodies.begin(),
					storage.bodies.size());
				Dy::avbdDetectSoftRigidConvexSweptOGCFeatures(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicConvexes.begin(),
					storage.selectedDynamicConvexes.size(),
					storage.bodies.begin(),
					storage.bodies.size(),
					storage.probeContacts,
					mContactParams.contactRadius);
				Dy::avbdDetectSoftRigidConvexOGCFeatures(
					selectionParticles, selectionParticleCount,
					storage.selectedDynamicConvexes.begin(),
					storage.selectedDynamicConvexes.size(),
					storage.bodies.begin(),
					storage.bodies.size(),
					storage.probeContacts,
					mContactParams.contactRadius);
			}
			// probeContacts is only a fast-path hint. A distinct public collision
			// mesh may overlap even when the simulation tetrahedra used by this
			// legacy probe do not; the authoritative proxy redetection below owns
			// the final island-selection decision.

			for(PxU32 i = 0; i < mRigidBoxes.size(); i++)
				storage.rigidBoxes.pushBack(mRigidBoxes[i]);
			for(PxU32 i = 0;
				i < storage.selectedDynamicBoxes.size(); i++)
				storage.rigidBoxes.pushBack(
					storage.selectedDynamicBoxes[i]);
			for(PxU32 i = 0; i < mRigidSpheres.size(); i++)
				storage.rigidSpheres.pushBack(mRigidSpheres[i]);
			for(PxU32 i = 0;
				i < storage.selectedDynamicSpheres.size(); i++)
				storage.rigidSpheres.pushBack(
					storage.selectedDynamicSpheres[i]);
			for(PxU32 i = 0; i < mRigidCapsules.size(); i++)
				storage.rigidCapsules.pushBack(mRigidCapsules[i]);
			for(PxU32 i = 0;
				i < storage.selectedDynamicCapsules.size(); i++)
				storage.rigidCapsules.pushBack(
					storage.selectedDynamicCapsules[i]);
			for(PxU32 i = 0; i < mRigidConvexes.size(); i++)
				storage.rigidConvexes.pushBack(
					mRigidConvexes[i]);
			for(PxU32 i = 0;
				i < storage.selectedDynamicConvexes.size(); i++)
				storage.rigidConvexes.pushBack(
					storage.selectedDynamicConvexes[i]);
			// A non-CCD dynamic-box selection must discover contacts at the same
			// discrete endpoint which the native solver will warmstart to.  Looking
			// only at the source pose leaves an endpoint-first overlap with no
			// prepared row, while appending a second source/endpoint contact stream
			// double-owns the same OGC feature.  Keep the scope deliberately narrow
			// until endpoint variants exist for every dynamic shape type.
			bool useEndpointOnlyBoxDcd =
				!storage.selectedDynamicBoxes.empty() &&
				storage.selectedDynamicSpheres.empty() &&
				storage.selectedDynamicCapsules.empty() &&
				storage.selectedDynamicConvexes.empty();
			for(PxU32 bodyIndex = 0;
				useEndpointOnlyBoxDcd &&
				bodyIndex < storage.bodies.size(); ++bodyIndex)
			{
				useEndpointOnlyBoxDcd =
					!storage.bodies[bodyIndex].compiled.
						speculativeCCDEnabled;
			}

			PxArray<Dy::AvbdSoftParticle> endpointParticles;
			PxArray<Dy::AvbdRigidBox> endpointRigidBoxes;
			Dy::AvbdSoftParticle* contactParticles = selectionParticles;
			const Dy::AvbdRigidBox* contactRigidBoxes =
				storage.rigidBoxes.begin();
			if(useEndpointOnlyBoxDcd)
			{
				endpointParticles.resize(selectionParticleCount);
				for(PxU32 particleIndex = 0;
					particleIndex < selectionParticleCount; ++particleIndex)
				{
					endpointParticles[particleIndex] =
						selectionParticles[particleIndex];
					Dy::AvbdSoftParticle& endpointParticle =
						endpointParticles[particleIndex];
					if(!endpointParticle.predictedPosition.isFinite())
						return false;
					// The temporary collision domain is one pose, not a segment.
					endpointParticle.position =
						endpointParticle.predictedPosition;
					endpointParticle.initialPosition =
						endpointParticle.predictedPosition;
					endpointParticle.outerPosition =
						endpointParticle.predictedPosition;
				}

				endpointRigidBoxes = storage.rigidBoxes;
				for(PxU32 boxIndex = 0;
					boxIndex < endpointRigidBoxes.size(); ++boxIndex)
				{
					Dy::AvbdRigidBox& endpointBox =
						endpointRigidBoxes[boxIndex];
					if(endpointBox.targetKind !=
						Dy::AvbdSoftContactTargetKind::eRIGID_BODY)
						continue;
					if(endpointBox.targetIndex >= bodyCount)
						return false;
					Dy::AvbdSolverBody& endpointBody =
						solverBodies[bodyStart + endpointBox.targetIndex];
					endpointBody.computePrediction(dt, gravity);
					const PxTransform endpointBodyToWorld(
						endpointBody.predictedPosition,
						endpointBody.predictedRotation);
					const PxTransform endpointShapeToWorld =
						endpointBodyToWorld *
						endpointBox.shapeToRigidBody;
					if(!endpointShapeToWorld.isValid())
						return false;
					endpointBox.center = endpointShapeToWorld.p;
					endpointBox.rotation = endpointShapeToWorld.q;
					// The selected source bodies are all non-CCD.  Pinning previous
					// to this same endpoint makes that invariant explicit even if a
					// future detector grows another swept branch.
					endpointBox.previousCenter = endpointBox.center;
					endpointBox.previousRotation = endpointBox.rotation;
				}
				contactParticles = endpointParticles.begin();
				contactRigidBoxes = endpointRigidBoxes.begin();
			}

			detectContacts(
				contactParticles, selectionParticleCount,
				storage.bodies.begin(), storage.bodies.size(),
				storage.contacts, contactRigidBoxes,
				storage.rigidBoxes.size(),
				storage.selfCollisionAdjacencies.begin(),
				storage.selfCollisionAdjacencies.size(),
				storage.selfCollisionEnabled.begin(),
				storage.softCores.begin(),
				storage.rigidSpheres.begin(),
				storage.rigidSpheres.size(),
				storage.rigidCapsules.begin(),
				storage.rigidCapsules.size(),
				storage.rigidConvexes.begin(),
				storage.rigidConvexes.size(), NULL, 0,
				&storage.ogcGeometrySidecar);
			// Preserve a task-local cooked collision proxy for the terminal OGC
			// epoch.  detectContacts may have used Scene-global subset scratch to
			// build the authoritative contact stream; copying just the immutable
			// topology/mapping here lets the native solver refit it from its final
			// pose without a Scene callback or a second owner of storage.contacts.
			if(!rebuildSubsetCollisionDetectionScene(
				selectionParticles, selectionParticleCount,
				storage.bodies.begin(), storage.bodies.size(),
				storage.softCores.begin()))
				return false;
			if(mCollisionProxy.subsetBodies.size() != storage.bodies.size() ||
				mCollisionProxy.subsetParticles.size() !=
					mCollisionProxy.subsetVertexMappings.size())
				return false;
			storage.terminalCollisionBodies = mCollisionProxy.subsetBodies;
			storage.terminalCollisionVertexMappings =
				mCollisionProxy.subsetVertexMappings;
			// Compile one mutable OGC epoch record per body/shape pair.  The
			// contact stream remains AL-owned; this map is the shared pair scheduler.
			if(!compileAvbdIslandOgcPairPlan(storage))
				return false;
			bool hasRigidTargetContact = false;
			for(PxU32 i = 0; i < storage.contacts.size(); i++)
				if(storage.contacts[i].geometry.hasRigidBodyTarget())
				{
					hasRigidTargetContact = true;
					break;
				}
			// Some speculative finite-shape rows are materialized only after the
			// mixed solver owns both endpoints.  The component fallback has no
			// selected-entry mask, so every predictive dynamic storage must remain
			// under the same frame owner while any awake speculative-CCD source is
			// active.  Without that explicit public policy, current-pose OGC sources
			// still require an actual rigid-target row.
			const bool hasSelectedDynamicTarget =
				!storage.selectedDynamicBoxes.empty() ||
				!storage.selectedDynamicSpheres.empty() ||
				!storage.selectedDynamicCapsules.empty() ||
				!storage.selectedDynamicConvexes.empty();
			bool requiresSpeculativeCcdOwnershipCover = false;
			for(PxU32 entryIndex = 0;
				hasSelectedDynamicTarget && entryIndex < mEntries.size();
				++entryIndex)
			{
				const Entry& entry = mEntries[entryIndex];
				if(!entry.sleeping &&
					entry.getBodyCore().bodyFlags.isSet(
					PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD))
				{
					requiresSpeculativeCcdOwnershipCover = true;
					break;
				}
			}
			const bool isNativeEligible = hasRigidTargetContact ||
				(hasSelectedDynamicTarget &&
				 requiresSpeculativeCcdOwnershipCover) ||
				hasRigidAttachment ||
				hasArticulationAttachment ||
				hasSoftPairAttachment;
			if(!isNativeEligible)
				return false;
			return compileAvbdIslandSelectionExecutionPlan(
				storage, selectionParticleCount, bodyCount);
		}

} // namespace Sc
} // namespace physx
