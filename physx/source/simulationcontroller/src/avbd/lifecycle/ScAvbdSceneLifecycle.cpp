// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/scene/ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

		const PxsDeformableSurfaceMaterialCore*
			AvbdCpuSoftScene::getSurfaceMaterial(
				const DeformableSurfaceCore& core) const
		{
			const PxArray<PxU16>& handles =
				core.getCore().materialHandles;
			if(handles.empty() ||
				handles[0] == MATERIAL_INVALID_HANDLE ||
				handles[0] >= mSurfaceMaterialManager.getMaxSize())
				return NULL;
			const PxsDeformableSurfaceMaterialCore* material =
				mSurfaceMaterialManager.getMaterial(handles[0]);
			return material->mMaterialIndex == handles[0]
				? material : NULL;
		}

		bool AvbdCpuSoftScene::rebuildSurfaceRestState(Entry& entry)
		{
			PX_ASSERT(entry.kind == eSURFACE && entry.surfaceCore);
			Dy::DeformableSurfaceCore& core =
				entry.surfaceCore->getCore();
			const PxU32 particleStart = getParticleStart(entry);
			const PxU32 numVertices = getParticleCount(entry);
			const PxU32 numTriangles =
				entry.triangleMesh->getNbTriangles();
			PxArray<PxVec3> restVertices;
			restVertices.resize(numVertices);
			for(PxU32 i = 0; i < numVertices; i++)
			{
				restVertices[i] = core.restPosition[i].getXYZ();
				if(!restVertices[i].isFinite())
					return false;
			}
			PxArray<PxU32> triangles;
			triangles.resize(3 * numTriangles);
			const bool has16BitIndices =
				entry.triangleMesh->getTriangleMeshFlags() &
				PxTriangleMeshFlag::e16_BIT_INDICES;
			if(has16BitIndices)
			{
				const PxU16* source =
					static_cast<const PxU16*>(
						entry.triangleMesh->getTriangles());
				for(PxU32 i = 0; i < triangles.size(); i++)
					triangles[i] = source[i];
			}
			else
			{
				const PxU32* source =
					static_cast<const PxU32*>(
						entry.triangleMesh->getTriangles());
				for(PxU32 i = 0; i < triangles.size(); i++)
					triangles[i] = source[i];
			}

			const PxsDeformableSurfaceMaterialCore* material =
				getSurfaceMaterial(*entry.surfaceCore);
			PxArray<Dy::AvbdSoftParticle> rebuiltParticles;
			PxArray<Dy::AvbdSoftBody> rebuiltBodies;
			Dy::avbdCreateSoftBody(
				restVertices.begin(), numVertices,
				NULL, 0,
				triangles.begin(), triangles.size(),
				material ? material->youngs : 1.0e5f,
				material ? material->poissons : 0.3f,
				1.0f,
				entry.surfaceCore->getLinearDamping() +
					(material ? material->elasticityDamping : 0.0f),
				material ? material->bendingStiffness : 0.0f,
				material
					? PxMax(material->thickness, 1.0e-4f)
					: 0.01f,
				rebuiltParticles, rebuiltBodies,
				(entry.surfaceCore->getSurfaceFlags() &
					PxDeformableSurfaceFlag::eENABLE_FLATTENING)
					? true : false,
				entry.surfaceCore->
					getSelfCollisionFilterDistance(),
				material ? material->dynamicFriction : 0.5f);
			if(rebuiltBodies.size() != 1 ||
				rebuiltParticles.size() != numVertices)
				return false;
			if(!rebaseSoftBodyParticleRangeInPlace(
				rebuiltBodies[0], 0, numVertices,
				particleStart))
				return false;
			mBodies[entry.bodyIndex] = rebuiltBodies[0];
			PX_ASSERT(
				entry.bodyIndex < mSelfCollisionAdjacencies.size());
			Dy::avbdBuildSelfCollisionAdjacency(
				mBodies[entry.bodyIndex],
				mSelfCollisionAdjacencies[entry.bodyIndex]);
			if(!rebuildEntryPins(entry))
				return false;
			for(PxU32 i = 0; i < numVertices; i++)
			{
				Dy::AvbdSoftParticle& particle =
					mParticles[particleStart + i];
				particle.initialPosition = restVertices[i];
				particle.elasticK = 0.0f;
			}
			return true;
		}

		Entry* AvbdCpuSoftScene::findEntry(ActorCore& core)
		{
			for(PxU32 i = 0; i < mEntries.size(); i++)
			{
				Entry& entry = mEntries[i];
				if(entry.getActorCore() == &core)
					return &entry;
			}
			return NULL;
		}

		bool AvbdCpuSoftScene::isVolumeKinematicTargetActive(
			const Dy::DeformableVolumeCore& core,
			const PxVec4& target) const
		{
			if(!core.kinematicTarget ||
				!target.getXYZ().isFinite())
				return false;
			if(core.bodyFlags & PxDeformableBodyFlag::eKINEMATIC)
				return true;
			return
				(core.volumeFlags &
					PxDeformableVolumeFlag::ePARTIALLY_KINEMATIC) &&
				target.w == 0.0f;
		}

		bool AvbdCpuSoftScene::appendVolumeKinematicTargetPins(
			const Entry& entry,
			const PxArray<Dy::AvbdKinematicPin>& previousPins,
			PxArray<Dy::AvbdKinematicPin>& pins) const
		{
			if(entry.kind != eVOLUME || !entry.volumeCore)
				return true;
			const Dy::DeformableVolumeCore& core =
				entry.volumeCore->getCore();
			if(!core.kinematicTarget ||
				!(core.bodyFlags &
						PxDeformableBodyFlag::eKINEMATIC) &&
				!(core.volumeFlags &
						PxDeformableVolumeFlag::
							ePARTIALLY_KINEMATIC))
				return true;
			const PxU32 particleStart = getParticleStart(entry);
			const PxU32 particleCount = getParticleCount(entry);
			for(PxU32 localIndex = 0;
				localIndex < particleCount; localIndex++)
			{
				const PxVec4& target =
					core.kinematicTarget[localIndex];
				if(!isVolumeKinematicTargetActive(core, target))
					continue;
				Dy::AvbdKinematicPin pin;
				pin.point.setVertex(particleStart + localIndex);
				pin.sourceHandle = localIndex;
				pin.targetKind =
					Dy::AvbdSoftPinTargetKind::
						eDEFORMABLE_KINEMATIC;
				pin.worldTarget = target.getXYZ();
				pin.previousWorldTarget = pin.worldTarget;
				pin.k = 1.0e8f;
				pin.kMax = 1.0e10f;
				for(PxU32 previousIndex = 0;
					previousIndex < previousPins.size();
					previousIndex++)
				{
					const Dy::AvbdKinematicPin& previous =
						previousPins[previousIndex];
					if(previous.targetKind != pin.targetKind ||
						previous.sourceHandle != localIndex)
						continue;
					pin.previousWorldTarget =
						previous.worldTarget;
					pin.alLambda = previous.alLambda;
					pin.k = previous.k;
					pin.kMax = previous.kMax;
					break;
				}
				pins.pushBack(pin);
			}
			return true;
		}

		bool AvbdCpuSoftScene::rebuildEntryPins(Entry& entry)
		{
			if(entry.bodyIndex >= mBodies.size())
				return false;
			Dy::AvbdSoftBody& body = mBodies[entry.bodyIndex];
			const PxU32 particleStart =
				body.compiled.particleStart;
			const PxU32 particleCount =
				body.compiled.particleCount;
			const PxArray<Dy::AvbdKinematicPin> previousPins =
				body.runtime.pins;
			body.runtime.pins.clear();
			if(!appendVolumeKinematicTargetPins(
				entry, previousPins, body.runtime.pins))
				return false;
			for(PxU32 i = 0; i < mWorldPins.size(); i++)
			{
				const WorldPinEntry& source = mWorldPins[i];
				if(source.softCore != entry.getActorCore())
					continue;
				if(!Dy::avbdIsSoftPointValid(
					source.localPoint, 0, particleCount))
					return false;
				Dy::AvbdKinematicPin pin;
				pin.point = source.localPoint;
				for(PxU32 endpoint = 0;
					endpoint < pin.point.particleCount; endpoint++)
					pin.point.particleIndices[endpoint] +=
						particleStart;
				pin.sourceHandle = source.handle;
				pin.targetKind =
					Dy::AvbdSoftPinTargetKind::eWORLD_FIXED;
				pin.worldTarget = source.worldTarget;
				pin.previousWorldTarget =
					source.worldTarget;
				// Public vertex-to-world attachments are fixed positional
				// objectives. Keep their compliance below the public gate
				// tolerance while retaining the position-level AL owner.
				pin.k = 1.0e8f;
				pin.kMax = 1.0e10f;
				body.runtime.pins.pushBack(pin);
			}
			for(PxU32 i = 0;
				i < mPrescribedAttachments.size(); i++)
			{
				PrescribedAttachmentEntry& source =
					mPrescribedAttachments[i];
				if(source.softCore != entry.getActorCore() ||
					!source.active)
					continue;
				if(!Dy::avbdIsSoftPointValid(
					source.localPoint, 0, particleCount))
					return false;
				PxVec3 worldTarget;
				if(!computePrescribedAttachmentWorldTarget(
					*source.prescribedCore,
					source.actorLocalTarget, worldTarget))
					return false;
				source.worldTarget = worldTarget;
				Dy::AvbdKinematicPin pin;
				pin.point = source.localPoint;
				for(PxU32 endpoint = 0;
					endpoint < pin.point.particleCount; endpoint++)
					pin.point.particleIndices[endpoint] +=
						particleStart;
				pin.sourceHandle = source.handle;
				pin.targetKind =
					Dy::AvbdSoftPinTargetKind::
						ePRESCRIBED_RIGID;
				pin.worldTarget = source.worldTarget;
				pin.previousWorldTarget =
					source.previousWorldTarget;
				pin.alLambda = source.alLambda;
				pin.k = source.k;
				pin.kMax = source.kMax;
				body.runtime.pins.pushBack(pin);
			}
			body.runtime.compileObjectiveProgram(
				particleStart, particleCount);
			return body.runtime.isObjectiveProgramCurrent(
				particleStart, particleCount);
		}

		void AvbdCpuSoftScene::refreshVolumeKinematicTargets()
		{
			for(PxU32 entryIndex = 0;
				entryIndex < mEntries.size(); entryIndex++)
			{
				Entry& entry = mEntries[entryIndex];
				if(entry.kind != eVOLUME || !entry.volumeCore ||
					entry.bodyIndex >= mBodies.size())
					continue;
				const Dy::DeformableVolumeCore& core =
					entry.volumeCore->getCore();
				Dy::AvbdSoftBodyRuntimeState& runtime =
					mBodies[entry.bodyIndex].runtime;
				const PxU32 particleCount =
					getParticleCount(entry);
				PxU32 expectedCount = 0;
				if(core.kinematicTarget)
				{
					for(PxU32 localIndex = 0;
						localIndex < particleCount; localIndex++)
					{
						if(isVolumeKinematicTargetActive(
							core,
							core.kinematicTarget[localIndex]))
							expectedCount++;
					}
				}

				PxU32 existingCount = 0;
				bool needsRebuild = false;
				for(PxU32 pinIndex = 0;
					pinIndex < runtime.pins.size(); pinIndex++)
				{
					const Dy::AvbdKinematicPin& pin =
						runtime.pins[pinIndex];
					if(pin.targetKind !=
						Dy::AvbdSoftPinTargetKind::
							eDEFORMABLE_KINEMATIC)
						continue;
					existingCount++;
					if(pin.sourceHandle >= particleCount ||
						!core.kinematicTarget ||
						!isVolumeKinematicTargetActive(
							core,
							core.kinematicTarget[
								pin.sourceHandle]))
						needsRebuild = true;
				}
				needsRebuild =
					needsRebuild || existingCount != expectedCount;
				if(needsRebuild)
				{
					const bool rebuilt = rebuildEntryPins(entry);
					PX_ASSERT(rebuilt);
					if(rebuilt && expectedCount > 0)
						wakeEntry(
							entry,
							ScInternalWakeCounterResetValue);
					continue;
				}

				bool targetMoved = false;
				for(PxU32 pinIndex = 0;
					pinIndex < runtime.pins.size(); pinIndex++)
				{
					Dy::AvbdKinematicPin& pin =
						runtime.pins[pinIndex];
					if(pin.targetKind !=
						Dy::AvbdSoftPinTargetKind::
							eDEFORMABLE_KINEMATIC)
						continue;
					PX_ASSERT(core.kinematicTarget &&
						pin.sourceHandle < particleCount);
					if(!core.kinematicTarget ||
						pin.sourceHandle >= particleCount)
						continue;
					const PxVec3 previousTarget =
						pin.worldTarget;
					const PxVec3 worldTarget =
						core.kinematicTarget[
							pin.sourceHandle].getXYZ();
					pin.previousWorldTarget = previousTarget;
					pin.worldTarget = worldTarget;
					targetMoved = targetMoved ||
						(worldTarget - previousTarget).
							magnitudeSquared() > 1.0e-12f;
				}
				if(targetMoved)
					wakeEntry(
						entry,
						ScInternalWakeCounterResetValue);
			}
		}

		void AvbdCpuSoftScene::refreshPrescribedAttachmentTargets()
		{
			for(PxU32 i = 0;
				i < mPrescribedAttachments.size(); i++)
			{
				PrescribedAttachmentEntry& attachment =
					mPrescribedAttachments[i];
				Entry* entry =
					findEntry(*attachment.softCore);
				if(!entry)
					continue;
				PxVec3 worldTarget;
				const bool active =
					computePrescribedAttachmentWorldTarget(
						*attachment.prescribedCore,
						attachment.actorLocalTarget,
						worldTarget);
				if(active != attachment.active)
				{
					attachment.active = active;
					if(active)
					{
						attachment.worldTarget =
							worldTarget;
						attachment.previousWorldTarget =
							worldTarget;
					}
					const bool rebuilt =
						rebuildEntryPins(*entry);
					PX_ASSERT(rebuilt);
					PX_UNUSED(rebuilt);
					if(active)
						wakeEntry(
							*entry,
							ScInternalWakeCounterResetValue);
					continue;
				}
				if(!active)
					continue;

				Dy::AvbdSoftBodyRuntimeState& runtime =
					mBodies[entry->bodyIndex].runtime;
				Dy::AvbdKinematicPin* pin = NULL;
				for(PxU32 pinIndex = 0;
					pinIndex < runtime.pins.size(); pinIndex++)
				{
					Dy::AvbdKinematicPin& candidate =
						runtime.pins[pinIndex];
					if(candidate.targetKind ==
							Dy::AvbdSoftPinTargetKind::
								ePRESCRIBED_RIGID &&
						candidate.sourceHandle ==
							attachment.handle)
					{
						pin = &candidate;
						break;
					}
				}
				PX_ASSERT(pin);
				if(!pin)
					continue;
				const PxVec3 previousTarget =
					attachment.worldTarget;
				attachment.previousWorldTarget =
					previousTarget;
				attachment.worldTarget = worldTarget;
				pin->previousWorldTarget = previousTarget;
				pin->worldTarget = worldTarget;
				if((worldTarget - previousTarget).
					magnitudeSquared() > 1.0e-12f)
					wakeEntry(
						*entry,
						ScInternalWakeCounterResetValue);
			}
		}

		void AvbdCpuSoftScene::removeWorldPinsForCore(ActorCore& core)
		{
			for(PxU32 i = mWorldPins.size(); i > 0; i--)
			{
				const WorldPinEntry& pin = mWorldPins[i - 1];
				if(pin.softCore == &core)
					mWorldPins.replaceWithLast(i - 1);
			}
		}

		void AvbdCpuSoftScene::removeRigidAttachmentsForSoft(ActorCore& core)
		{
			bool removed = false;
			for(PxU32 i = mRigidAttachments.size(); i > 0; i--)
			{
				if(mRigidAttachments[i - 1].softCore == &core)
				{
					mRigidAttachments.replaceWithLast(i - 1);
					removed = true;
				}
			}
			if(removed)
			{
				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
			}
		}

		void AvbdCpuSoftScene::removeArticulationAttachmentsForSoft(
			ActorCore& core)
		{
			bool removed = false;
			for(PxU32 i = mArticulationAttachments.size();
				i > 0; i--)
			{
				if(mArticulationAttachments[i - 1].softCore ==
					&core)
				{
					mArticulationAttachments.replaceWithLast(
						i - 1);
					removed = true;
				}
			}
			if(removed)
			{
				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
			}
		}

		void AvbdCpuSoftScene::removeSoftPairAttachmentsForSoft(
			ActorCore& core)
		{
			bool removed = false;
			for(PxU32 i = mSoftPairAttachments.size(); i > 0; i--)
			{
				const SoftPairAttachmentEntry& attachment =
					mSoftPairAttachments[i - 1];
				if(attachment.softCore[0] != &core &&
					attachment.softCore[1] != &core)
					continue;
				ActorCore* otherCore =
					attachment.softCore[0] == &core
						? attachment.softCore[1]
						: attachment.softCore[0];
				mSoftPairAttachments.replaceWithLast(i - 1);
				Entry* otherEntry =
					otherCore ? findEntry(*otherCore) : NULL;
				if(otherEntry)
					wakeEntry(
						*otherEntry,
						ScInternalWakeCounterResetValue);
				removed = true;
			}
			if(removed)
			{
				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
			}
		}

		void AvbdCpuSoftScene::removePrescribedAttachmentsForSoft(
			ActorCore& core)
		{
			for(PxU32 i = mPrescribedAttachments.size();
				i > 0; i--)
			{
				if(mPrescribedAttachments[i - 1].softCore ==
					&core)
					mPrescribedAttachments.replaceWithLast(
						i - 1);
			}
		}

		void AvbdCpuSoftScene::removePrescribedAttachmentsForRigid(
			RigidCore& core)
		{
			for(PxU32 i = mPrescribedAttachments.size();
				i > 0; i--)
			{
				if(mPrescribedAttachments[i - 1].
					prescribedCore != &core)
					continue;
				ActorCore* softCore =
					mPrescribedAttachments[i - 1].softCore;
				mPrescribedAttachments.replaceWithLast(i - 1);
				Entry* entry =
					softCore ? findEntry(*softCore) : NULL;
				if(entry)
				{
					const bool rebuilt =
						rebuildEntryPins(*entry);
					PX_ASSERT(rebuilt);
					PX_UNUSED(rebuilt);
					wakeEntry(
						*entry,
						ScInternalWakeCounterResetValue);
				}
			}
		}

		void AvbdCpuSoftScene::removeRigidAttachmentsForRigid(BodyCore& core)
		{
			bool removed = false;
			for(PxU32 i = mRigidAttachments.size(); i > 0; i--)
			{
				if(mRigidAttachments[i - 1].rigidCore == &core)
				{
					ActorCore* softCore =
						mRigidAttachments[i - 1].softCore;
					mRigidAttachments.replaceWithLast(i - 1);
					Entry* entry =
						softCore ? findEntry(*softCore) : NULL;
					if(entry)
						wakeEntry(
							*entry,
							ScInternalWakeCounterResetValue);
					removed = true;
				}
			}
			if(removed)
			{
				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
			}
		}

		void AvbdCpuSoftScene::removeArticulationAttachmentsForLink(
			BodyCore& core)
		{
			bool removed = false;
			for(PxU32 i = mArticulationAttachments.size();
				i > 0; i--)
			{
				if(mArticulationAttachments[i - 1].linkCore !=
					&core)
					continue;
				ActorCore* softCore =
					mArticulationAttachments[i - 1].softCore;
				mArticulationAttachments.replaceWithLast(i - 1);
				Entry* entry =
					softCore ? findEntry(*softCore) : NULL;
				if(entry)
					wakeEntry(
						*entry,
						ScInternalWakeCounterResetValue);
				removed = true;
			}
			if(removed)
			{
				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
			}
		}

		void AvbdCpuSoftScene::sleepEntry(Entry& entry)
		{
			Dy::DeformableBodyCore& core =
				entry.getBodyCore();
			PxVec4* velocities = entry.getVelocity();
			PX_ASSERT(velocities);
			const PxU32 particleStart = getParticleStart(entry);
			const PxU32 particleCount = getParticleCount(entry);
			for(PxU32 i = 0; i < particleCount; i++)
			{
				Dy::AvbdSoftParticle& particle =
					mParticles[particleStart + i];
				particle.velocity = PxVec3(0.0f);
				particle.prevVelocity = PxVec3(0.0f);
				particle.predictedPosition = particle.position;
				particle.outerPosition = particle.position;
				particle.invMass = 0.0f;
				particle.mass = 0.0f;
				particle.gravityScale = 0.0f;
				const PxReal velocityW =
					velocities[i].w;
				velocities[i] =
					PxVec4(PxVec3(0.0f), velocityW);
			}
			core.wakeCounter = 0.0f;
			core.cpuAvbdSleeping = true;
			core.cpuAvbdWakeRequested = false;
			entry.sleeping = true;
			mIslandManager.deactivateNode(entry.islandNode);
		}

		void AvbdCpuSoftScene::wakeEntry(Entry& entry, PxReal wakeCounter)
		{
			Dy::DeformableBodyCore& core =
				entry.getBodyCore();
			PxVec4* positions = entry.getPositionInvMass();
			PxVec4* velocities = entry.getVelocity();
			PX_ASSERT(positions && velocities);
			const PxReal gravityScale =
				(entry.getActorFlags() &
					PxActorFlag::eDISABLE_GRAVITY)
				? 0.0f : 1.0f;
			const PxU32 particleStart = getParticleStart(entry);
			const PxU32 particleCount = getParticleCount(entry);
			for(PxU32 i = 0; i < particleCount; i++)
			{
				Dy::AvbdSoftParticle& particle =
					mParticles[particleStart + i];
				const PxReal invMass = PxMax(
					positions[i].w, 0.0f);
				particle.invMass = invMass;
				particle.mass =
					invMass > 0.0f ? 1.0f / invMass : 0.0f;
				particle.velocity =
					velocities[i].getXYZ();
				particle.prevVelocity = particle.velocity;
				particle.gravityScale = gravityScale;
			}
			core.wakeCounter = PxMax(wakeCounter, 0.0f);
			core.cpuAvbdSleeping = false;
			core.cpuAvbdWakeRequested = false;
			entry.sleeping = false;
			mIslandManager.activateNode(entry.islandNode);
		}

		void AvbdCpuSoftScene::finalizeDeformableMotionControls(PxReal dt)
		{
			for(PxU32 entryIndex = 0;
				entryIndex < mEntries.size(); entryIndex++)
			{
				Entry& entry = mEntries[entryIndex];
				if(entry.sleeping)
					continue;

				const PxU32 particleStart =
					getParticleStart(entry);
				const PxU32 particleCount =
					getParticleCount(entry);
				PxReal maxSpeedSquared = 0.0f;
				for(PxU32 i = 0; i < particleCount; i++)
				{
					const Dy::AvbdSoftParticle& particle =
						mParticles[particleStart + i];
					if(particle.invMass <= 0.0f)
						continue;
					maxSpeedSquared = PxMax(
						maxSpeedSquared,
						particle.velocity.magnitudeSquared());
				}

				const Dy::DeformableBodyCore& core =
					entry.getBodyCore();
				const PxReal settlingThreshold =
					PxMax(core.settlingThreshold, 0.0f);
				if(maxSpeedSquared >
					settlingThreshold * settlingThreshold)
					continue;
				const PxReal settlingScale = PxMax(
					1.0f -
						PxMax(core.settlingDamping, 0.0f) * dt,
					0.0f);
				if(settlingScale >= 1.0f)
					continue;
				for(PxU32 i = 0; i < particleCount; i++)
				{
					Dy::AvbdSoftParticle& particle =
						mParticles[particleStart + i];
					if(particle.invMass <= 0.0f)
						continue;
					particle.velocity *= settlingScale;
					particle.prevVelocity = particle.velocity;
				}
			}
		}

		bool AvbdCpuSoftScene::hasUnforcedRestBendingResidual(const Entry& entry) const
		{
			if(entry.bodyIndex >= mBodies.size() || !mContacts.empty() ||
				!(entry.getActorFlags() & PxActorFlag::eDISABLE_GRAVITY))
				return false;
			const Dy::AvbdSoftBody& body = mBodies[entry.bodyIndex];
			if(body.material.bendingStiffness <= 0.0f ||
				body.compiled.bendElements.empty() ||
				!body.runtime.compiledObjectives.empty() ||
				body.compiled.selfCollisionRestPositions.size() !=
					body.compiled.particleCount)
				return false;

			// A fixed vertex away from its authored rest position is an external
			// boundary condition; its equilibrium may legitimately retain bend.
			// Only the unforced/rest-boundary case has rest dihedrals as a valid
			// cold sleep certificate.
			for(PxU32 localIndex = 0;
				localIndex < body.compiled.particleCount; localIndex++)
			{
				const Dy::AvbdSoftParticle& particle = mParticles[
					body.compiled.particleStart + localIndex];
				if(particle.invMass <= 0.0f &&
					(particle.position -
					 body.compiled.selfCollisionRestPositions[localIndex]).
						magnitudeSquared() > 1.0e-8f)
					return false;
			}

			const PxReal bendingSleepAngleTolerance = 1.0e-2f;
			for(PxU32 bendingIndex = 0;
				bendingIndex < body.compiled.bendElements.size(); bendingIndex++)
			{
				const Dy::AvbdBendingElement& bending =
					body.compiled.bendElements[bendingIndex];
				const PxReal angle =
					Dy::AvbdSoftBodyCompiledData::computeDihedralAngle(
						mParticles[bending.opp0].position,
						mParticles[bending.opp1].position,
						mParticles[bending.edgeStart].position,
						mParticles[bending.edgeEnd].position);
				const PxReal angleDifference = angle - bending.restAngle;
				const PxReal wrappedError = PxAtan2(
					PxSin(angleDifference), PxCos(angleDifference));
				if(!PxIsFinite(wrappedError) ||
					PxAbs(wrappedError) > bendingSleepAngleTolerance)
					return true;
			}
			return false;
		}

		void AvbdCpuSoftScene::updateSleepStates(
			PxReal dt, bool sleepingEnabled)
		{
			// Low velocity alone is not a stationarity certificate for AVBD.
			// Strong damping can remove velocity while the final particle block
			// still requests a visible elastic correction (most notably for a
			// slowly flattening cloth hinge).  The scalar component solve already
			// publishes its final pre-limiter H^-1 f displacement, so consume that
			// cold step-level certificate before freezing particle inverse masses.
			// It is intentionally conservative for multi-body components: one
			// unresolved body keeps the component awake rather than freezing a
			// coupled peer prematurely.
			const PxReal componentSleepResidualThreshold = 1.0e-4f;
			const bool componentResidualPending =
				mLastComponentFallbackSteps != 0 &&
				(!PxIsFinite(
					mLastStepStats.finalMaxLocalSolveDisplacement) ||
				 mLastStepStats.finalMaxLocalSolveDisplacement >
					componentSleepResidualThreshold);
			for(PxU32 entryIndex = 0;
				entryIndex < mEntries.size(); entryIndex++)
			{
				Entry& entry = mEntries[entryIndex];
				if(!sleepingEnabled)
				{
					if(entry.sleeping)
						wakeEntry(
							entry,
							ScInternalWakeCounterResetValue);
					continue;
				}
				if(entry.sleeping)
					continue;

				PxReal maxSpeedSquared = 0.0f;
				const PxU32 particleStart =
					getParticleStart(entry);
				const PxU32 particleCount =
					getParticleCount(entry);
				for(PxU32 i = 0; i < particleCount; i++)
				{
					const Dy::AvbdSoftParticle& particle =
						mParticles[particleStart + i];
					if(particle.invMass <= 0.0f)
						continue;
					maxSpeedSquared = PxMax(
						maxSpeedSquared,
						particle.velocity.magnitudeSquared());
				}
				Dy::DeformableBodyCore& core =
					entry.getBodyCore();
				bool kinematicTargetResidualPending = false;
				if(entry.bodyIndex < mBodies.size())
				{
					const Dy::AvbdSoftBodyRuntimeState& runtime =
						mBodies[entry.bodyIndex].runtime;
					for(PxU32 pinIndex = 0;
						pinIndex < runtime.pins.size(); pinIndex++)
					{
						const Dy::AvbdKinematicPin& pin =
							runtime.pins[pinIndex];
						if(pin.targetKind !=
							Dy::AvbdSoftPinTargetKind::
								eDEFORMABLE_KINEMATIC)
							continue;
						const PxReal
							kinematicTargetResidualSquared =
								(Dy::avbdGetSoftPointPosition(
									pin.point,
									mParticles.begin()) -
								 pin.worldTarget).
									magnitudeSquared();
						if(kinematicTargetResidualSquared >
							1.0e-8f)
						{
							kinematicTargetResidualPending = true;
							break;
						}
					}
				}
				if(kinematicTargetResidualPending ||
					componentResidualPending)
				{
					core.wakeCounter = PxMax(
						core.wakeCounter,
						ScInternalWakeCounterResetValue);
					continue;
				}
				const PxReal sleepThreshold =
					PxMax(core.sleepThreshold, 0.0f);
				if(maxSpeedSquared >
					sleepThreshold * sleepThreshold)
				{
					core.wakeCounter = PxMax(
						core.wakeCounter,
						ScInternalWakeCounterResetValue);
					continue;
				}
				if(hasUnforcedRestBendingResidual(entry))
				{
					core.wakeCounter = PxMax(
						core.wakeCounter,
						ScInternalWakeCounterResetValue);
					continue;
				}

				core.wakeCounter = PxMax(
					core.wakeCounter - dt, 0.0f);
				if(core.wakeCounter == 0.0f)
					sleepEntry(entry);
			}
		}

} // namespace Sc
} // namespace physx
