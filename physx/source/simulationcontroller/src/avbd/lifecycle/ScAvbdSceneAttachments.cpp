// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/scene/ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

		bool AvbdCpuSoftScene::add(
			DeformableVolumeCore& core,
			PxTetrahedronMesh& simulationMesh,
			PxTetrahedronMesh& collisionMesh,
			PxDeformableVolumeAuxData& auxData,
			const PxsDeformableVolumeMaterialManager& materialManager)
		{
			const PxU32 numVertices = simulationMesh.getNbVertices();
			const PxU32 numTets = simulationMesh.getNbTetrahedrons();
			Dy::DeformableVolumeCore& dyCore = core.getCore();
			if(numVertices == 0 || numTets == 0 ||
				!dyCore.simPositionInvMass || !dyCore.simVelocity ||
				!dyCore.positionInvMass || !dyCore.restPosition)
				return false;
			if(!validateVolumeCollisionEmbedding(
				simulationMesh, collisionMesh, auxData))
				return false;

			for(PxU32 i = 0; i < mEntries.size(); i++)
				if(mEntries[i].volumeCore == &core)
					return false;

			for(PxU32 i = 0; i < numVertices; i++)
			{
				const PxVec3 position =
					dyCore.simPositionInvMass[i].getXYZ();
				if(!position.isFinite() ||
					!PxIsFinite(dyCore.simPositionInvMass[i].w) ||
					!dyCore.simVelocity[i].getXYZ().isFinite())
					return false;
			}
			core.initializeCpuAvbdSimulationRestPositions(
				dyCore.simPositionInvMass, numVertices);
			const PxArray<PxVec3>& restVertices =
				core.getCpuAvbdSimulationRestPositions();
			if(restVertices.size() != numVertices)
				return false;
			for(PxU32 i = 0; i < numVertices; i++)
				if(!restVertices[i].isFinite())
					return false;

			PxArray<PxU32> tetrahedra;
			tetrahedra.resize(4 * numTets);
			const bool has16BitIndices =
				simulationMesh.getTetrahedronMeshFlags() &
				PxTetrahedronMeshFlag::e16_BIT_INDICES;
			if(has16BitIndices)
			{
				const PxU16* source =
					static_cast<const PxU16*>(
						simulationMesh.getTetrahedrons());
				for(PxU32 i = 0; i < tetrahedra.size(); i++)
					tetrahedra[i] = source[i];
			}
			else
			{
				const PxU32* source =
					static_cast<const PxU32*>(
						simulationMesh.getTetrahedrons());
				for(PxU32 i = 0; i < tetrahedra.size(); i++)
					tetrahedra[i] = source[i];
			}
			for(PxU32 i = 0; i < tetrahedra.size(); i++)
				if(tetrahedra[i] >= numVertices)
					return false;

			const PxsDeformableVolumeMaterialCore* material =
				getMaterial(core, materialManager);
			const PxReal youngs = material ? material->youngs : 1.0e5f;
			const PxReal poissons =
				material ? material->poissons : 0.3f;
			const PxReal materialDamping =
				material ? material->elasticityDamping : 0.0f;
			const bool coRotationalVolumeModel =
				!material || material->materialModel ==
					PxDeformableVolumeMaterialModel::eCO_ROTATIONAL;
			const PxReal gravityScale =
				(core.getActorFlags() & PxActorFlag::eDISABLE_GRAVITY)
				? 0.0f : 1.0f;
			const PxU32 bodyIndex = mBodies.size();
			const PxU32 particleStart = Dy::avbdCreateSoftBody(
				restVertices.begin(), numVertices,
				tetrahedra.begin(), tetrahedra.size(),
				NULL, 0,
				youngs, poissons,
				1.0f, core.getLinearDamping() + materialDamping,
				0.0f, 0.01f,
				mParticles, mBodies, false,
				core.getSelfCollisionFilterDistance(),
				material ? material->dynamicFriction : 0.5f,
				coRotationalVolumeModel);

			for(PxU32 i = 0; i < numVertices; i++)
			{
				Dy::AvbdSoftParticle& particle =
					mParticles[particleStart + i];
				const PxVec4& positionInvMass =
					dyCore.simPositionInvMass[i];
				particle.position = positionInvMass.getXYZ();
				particle.initialPosition = particle.position;
				particle.predictedPosition = particle.position;
				particle.outerPosition = particle.position;
				particle.velocity = dyCore.simVelocity[i].getXYZ();
				particle.prevVelocity = particle.velocity;
				particle.invMass =
					PxMax(positionInvMass.w, 0.0f);
				particle.mass = particle.invMass > 0.0f
					? 1.0f / particle.invMass : 0.0f;
				particle.damping =
					core.getLinearDamping() + materialDamping;
				particle.gravityScale = gravityScale;
			}

			void* islandObjectMemory = PX_ALLOC(
				sizeof(Dy::DeformableVolume),
				"AVBD CPU deformable island object");
			Dy::DeformableVolume* islandObject =
				islandObjectMemory
				? PX_PLACEMENT_NEW(
					islandObjectMemory,
					Dy::DeformableVolume)(NULL, dyCore)
				: NULL;
			if(!islandObject)
			{
				mBodies.replaceWithLast(bodyIndex);
				mParticles.resize(particleStart);
				return false;
			}
			const PxNodeIndex islandNode =
				mIslandManager.addNode(
					false, false,
					IG::Node::eDEFORMABLE_VOLUME_TYPE,
					islandObject);
			PxReal maxSpeedSquared = 0.0f;
			for(PxU32 i = 0; i < numVertices; i++)
			{
				if(dyCore.simPositionInvMass[i].w > 0.0f)
					maxSpeedSquared = PxMax(
						maxSpeedSquared,
						dyCore.simVelocity[i].getXYZ().
							magnitudeSquared());
			}
			const PxReal sleepThreshold =
				PxMax(dyCore.sleepThreshold, 0.0f);
			const bool startsSleeping =
				dyCore.wakeCounter == 0.0f &&
				maxSpeedSquared <=
					sleepThreshold * sleepThreshold;
			if(startsSleeping)
				mIslandManager.deactivateNode(islandNode);
			else
				mIslandManager.activateNode(islandNode);
			mEntries.pushBack(Entry(
				core, simulationMesh, collisionMesh, auxData,
				bodyIndex, *islandObject, islandNode,
				startsSleeping));
			// Reserve topology-bounded solver and collision-query scratch while
			// the actor is added, before its first simulation step. Contact
			// output capacity remains subject to the separate source-aware
			// policy, because its density is collision-state dependent.
			mWorkspace.reserve(mParticles.size(), 0);
			reserveLifecycleContactCapacity();
			reserveLifecycleCollisionScratch();
			PX_ASSERT(mSelfCollisionAdjacencies.size() == bodyIndex);
			mSelfCollisionAdjacencies.resize(bodyIndex + 1);
			Dy::avbdBuildSelfCollisionAdjacency(
				mBodies[bodyIndex],
				mSelfCollisionAdjacencies[bodyIndex]);
			if(startsSleeping)
				sleepEntry(mEntries.back());
			else
			{
				dyCore.cpuAvbdSleeping = false;
				dyCore.cpuAvbdWakeRequested = false;
			}
			dyCore.dirty = false;
			dyCore.dirtyFlags = PxDeformableVolumeDataFlags(0);
			if(!rebuildCollisionDetectionScene())
			{
				reportInvalidCollisionEmbedding(
					"CPU AVBD failed to build the cooked collision-domain detection scene.");
				removeEntry(core);
				return false;
			}
			return true;
		}

		bool AvbdCpuSoftScene::addSurface(
			DeformableSurfaceCore& core,
			PxTriangleMesh& triangleMesh)
		{
			const PxU32 numVertices = triangleMesh.getNbVertices();
			const PxU32 numTriangles = triangleMesh.getNbTriangles();
			Dy::DeformableSurfaceCore& dyCore = core.getCore();
			if(numVertices == 0 || numTriangles == 0 ||
				!dyCore.positionInvMass || !dyCore.velocity ||
				!dyCore.restPosition)
				return false;

			for(PxU32 i = 0; i < mEntries.size(); i++)
				if(mEntries[i].surfaceCore == &core)
					return false;

			PxArray<PxVec3> restVertices;
			restVertices.resize(numVertices);
			for(PxU32 i = 0; i < numVertices; i++)
			{
				const PxVec4& restPosition = dyCore.restPosition[i];
				const PxVec4& positionInvMass =
					dyCore.positionInvMass[i];
				if(!restPosition.getXYZ().isFinite() ||
					!positionInvMass.getXYZ().isFinite() ||
					!PxIsFinite(positionInvMass.w) ||
					!dyCore.velocity[i].getXYZ().isFinite())
					return false;
				restVertices[i] = restPosition.getXYZ();
			}

			PxArray<PxU32> triangles;
			triangles.resize(3 * numTriangles);
			const bool has16BitIndices =
				triangleMesh.getTriangleMeshFlags() &
				PxTriangleMeshFlag::e16_BIT_INDICES;
			if(has16BitIndices)
			{
				const PxU16* source =
					static_cast<const PxU16*>(
						triangleMesh.getTriangles());
				for(PxU32 i = 0; i < triangles.size(); i++)
					triangles[i] = source[i];
			}
			else
			{
				const PxU32* source =
					static_cast<const PxU32*>(
						triangleMesh.getTriangles());
				for(PxU32 i = 0; i < triangles.size(); i++)
					triangles[i] = source[i];
			}
			for(PxU32 i = 0; i < triangles.size(); i++)
				if(triangles[i] >= numVertices)
					return false;

			const PxsDeformableSurfaceMaterialCore* material =
				getSurfaceMaterial(core);
			const PxReal youngs =
				material ? material->youngs : 1.0e5f;
			const PxReal poissons =
				material ? material->poissons : 0.3f;
			const PxReal materialDamping =
				material ? material->elasticityDamping : 0.0f;
			const PxReal bendingStiffness =
				material ? material->bendingStiffness : 0.0f;
			const PxReal thickness =
				material ? PxMax(material->thickness, 1.0e-4f)
						 : 0.01f;
			const PxReal gravityScale =
				(core.getActorFlags() & PxActorFlag::eDISABLE_GRAVITY)
				? 0.0f : 1.0f;
			const PxU32 bodyIndex = mBodies.size();
			const PxU32 particleStart = Dy::avbdCreateSoftBody(
				restVertices.begin(), numVertices,
				NULL, 0,
				triangles.begin(), triangles.size(),
				youngs, poissons,
				1.0f, core.getLinearDamping() + materialDamping,
				bendingStiffness, thickness,
				mParticles, mBodies,
				(core.getSurfaceFlags() &
					PxDeformableSurfaceFlag::eENABLE_FLATTENING)
					? true : false,
				core.getSelfCollisionFilterDistance(),
				material ? material->dynamicFriction : 0.5f);

			for(PxU32 i = 0; i < numVertices; i++)
			{
				Dy::AvbdSoftParticle& particle =
					mParticles[particleStart + i];
				const PxVec4& positionInvMass =
					dyCore.positionInvMass[i];
				particle.position = positionInvMass.getXYZ();
				particle.initialPosition = restVertices[i];
				particle.predictedPosition = particle.position;
				particle.outerPosition = particle.position;
				particle.velocity = dyCore.velocity[i].getXYZ();
				particle.prevVelocity = particle.velocity;
				particle.invMass =
					PxMax(positionInvMass.w, 0.0f);
				particle.mass = particle.invMass > 0.0f
					? 1.0f / particle.invMass : 0.0f;
				particle.damping =
					core.getLinearDamping() + materialDamping;
				particle.gravityScale = gravityScale;
			}

			void* islandObjectMemory = PX_ALLOC(
				sizeof(Dy::DeformableSurface),
				"AVBD CPU deformable surface island object");
			Dy::DeformableSurface* islandObject =
				islandObjectMemory
					? PX_PLACEMENT_NEW(
						islandObjectMemory,
						Dy::DeformableSurface)(NULL, dyCore)
					: NULL;
			if(!islandObject)
			{
				mBodies.replaceWithLast(bodyIndex);
				mParticles.resize(particleStart);
				return false;
			}
			const PxNodeIndex islandNode =
				mIslandManager.addNode(
					false, false,
					IG::Node::eDEFORMABLE_SURFACE_TYPE,
					islandObject);
			PxReal maxSpeedSquared = 0.0f;
			for(PxU32 i = 0; i < numVertices; i++)
			{
				if(dyCore.positionInvMass[i].w > 0.0f)
					maxSpeedSquared = PxMax(
						maxSpeedSquared,
						dyCore.velocity[i].getXYZ().
							magnitudeSquared());
			}
			const PxReal sleepThreshold =
				PxMax(dyCore.sleepThreshold, 0.0f);
			const bool startsSleeping =
				dyCore.wakeCounter == 0.0f &&
				maxSpeedSquared <=
					sleepThreshold * sleepThreshold;
			if(startsSleeping)
				mIslandManager.deactivateNode(islandNode);
			else
				mIslandManager.activateNode(islandNode);
			mEntries.pushBack(Entry(
				core, triangleMesh, bodyIndex,
				*islandObject, islandNode, startsSleeping));
			// Keep particle/body-sized scratch out of the simulation hot path.
			// Known peer/rigid sources may reserve a budgeted contact capacity
			// here; unbounded contact density remains separately measured.
			mWorkspace.reserve(mParticles.size(), 0);
			reserveLifecycleContactCapacity();
			reserveLifecycleCollisionScratch();
			PX_ASSERT(mSelfCollisionAdjacencies.size() == bodyIndex);
			mSelfCollisionAdjacencies.resize(bodyIndex + 1);
			Dy::avbdBuildSelfCollisionAdjacency(
				mBodies[bodyIndex],
				mSelfCollisionAdjacencies[bodyIndex]);
			if(startsSleeping)
				sleepEntry(mEntries.back());
			else
			{
				dyCore.cpuAvbdSleeping = false;
				dyCore.cpuAvbdWakeRequested = false;
			}
			dyCore.dirty = false;
			dyCore.dirtyFlags = PxDeformableSurfaceDataFlags(0);
			if(!rebuildCollisionDetectionScene())
			{
				removeEntry(core);
				return false;
			}
			return true;
		}

		void AvbdCpuSoftScene::addStaticShape(StaticCore& core, ShapeCore& shape)
		{
			for(PxU32 i = 0; i < mStaticShapes.size(); i++)
			{
				if(mStaticShapes[i].core == &core &&
					mStaticShapes[i].shape == &shape)
					return;
			}
			mStaticShapes.pushBack(StaticShapeEntry(
				core, shape, mNextPrimitiveKey++));
		}

		void AvbdCpuSoftScene::removeStaticShape(
			StaticCore& core, const ShapeCore& shape)
		{
			for(PxU32 i = mStaticShapes.size(); i > 0; i--)
			{
				const StaticShapeEntry& entry = mStaticShapes[i - 1];
				if(entry.core == &core && entry.shape == &shape)
				{
					mStaticShapes.replaceWithLast(i - 1);
					return;
				}
			}
		}

		void AvbdCpuSoftScene::removeStatic(StaticCore& core)
		{
			for(PxU32 i = mStaticShapes.size(); i > 0; i--)
			{
				if(mStaticShapes[i - 1].core == &core)
					mStaticShapes.replaceWithLast(i - 1);
			}
			removePrescribedAttachmentsForRigid(core);
		}

		void AvbdCpuSoftScene::addDynamicShape(BodyCore& core, ShapeCore& shape)
		{
			for(PxU32 i = 0; i < mDynamicShapes.size(); i++)
			{
				if(mDynamicShapes[i].core == &core &&
					mDynamicShapes[i].shape == &shape)
					return;
			}
			mDynamicShapes.pushBack(DynamicShapeEntry(
				core, shape, mNextPrimitiveKey++));
		}

		void AvbdCpuSoftScene::removeDynamicShape(
			BodyCore& core, const ShapeCore& shape)
		{
			for(PxU32 i = mDynamicShapes.size(); i > 0; i--)
			{
				const DynamicShapeEntry& entry =
					mDynamicShapes[i - 1];
				if(entry.core == &core && entry.shape == &shape)
				{
					mDynamicShapes.replaceWithLast(i - 1);
					bool hasRemainingShape = false;
					for(PxU32 j = 0; j < mDynamicShapes.size(); j++)
					{
						if(mDynamicShapes[j].core == &core)
						{
							hasRemainingShape = true;
							break;
						}
					}
					if(!hasRemainingShape)
						removeNativeIslandEdgesForRigid(core);
					return;
				}
			}
		}

		void AvbdCpuSoftScene::removeDynamic(BodyCore& core)
		{
			for(PxU32 i = mDynamicShapes.size(); i > 0; i--)
			{
				if(mDynamicShapes[i - 1].core == &core)
					mDynamicShapes.replaceWithLast(i - 1);
			}
			removePrescribedAttachmentsForRigid(core);
			removeRigidAttachmentsForRigid(core);
			removeArticulationAttachmentsForLink(core);
			removeNativeIslandEdgesForRigid(core);
		}

		void AvbdCpuSoftScene::remove(DeformableVolumeCore& core)
		{
			removeEntry(core);
		}

		void AvbdCpuSoftScene::removeSurface(DeformableSurfaceCore& core)
		{
			removeEntry(core);
		}

		bool AvbdCpuSoftScene::buildLocalElementPoint(
			ActorCore& core,
			bool surfaceElement,
			PxU32 elementIndex,
			const PxVec4& barycentric,
			Dy::AvbdSoftPoint& point)
		{
			const Entry* entry = findEntry(core);
			if(!entry || entry->bodyIndex >= mBodies.size() ||
				!barycentric.isFinite())
				return false;
			const Dy::AvbdSoftBody& body =
				mBodies[entry->bodyIndex];
			const PxU32 endpointCount = surfaceElement ? 3u : 4u;
			const PxU32* topology = surfaceElement
				? body.compiled.triangles.begin()
				: body.compiled.tetrahedra.begin();
			const PxU32 topologyCount = surfaceElement
				? body.compiled.triangles.size()
				: body.compiled.tetrahedra.size();
			if(elementIndex >= topologyCount / endpointCount)
				return false;

			const PxReal weights[4] = {
				barycentric.x, barycentric.y,
				barycentric.z, barycentric.w};
			PxReal weightSum = 0.0f;
			for(PxU32 endpoint = 0;
				endpoint < endpointCount; endpoint++)
			{
				if(weights[endpoint] < 0.0f ||
					weights[endpoint] > 1.0f)
					return false;
				weightSum += weights[endpoint];
			}
			if(PxAbs(weightSum - 1.0f) > 1.0e-4f)
				return false;

			point.particleCount = endpointCount;
			for(PxU32 endpoint = 0; endpoint < endpointCount; endpoint++)
			{
				point.particleIndices[endpoint] =
					topology[elementIndex * endpointCount + endpoint];
				point.weights[endpoint] = weights[endpoint];
			}
			for(PxU32 endpoint = endpointCount; endpoint < 4; endpoint++)
			{
				point.particleIndices[endpoint] = PX_MAX_U32;
				point.weights[endpoint] = 0.0f;
			}
			return Dy::avbdIsSoftPointValid(
				point, 0, body.compiled.particleCount);
		}

		PxU32 AvbdCpuSoftScene::addWorldPin(
			ActorCore& core,
			PxU32 localVertex,
			const PxVec3& worldTarget)
		{
			Dy::AvbdSoftPoint point;
			point.setVertex(localVertex);
			return addWorldPin(core, point, worldTarget);
		}

		PxU32 AvbdCpuSoftScene::addWorldPin(
			ActorCore& core,
			const Dy::AvbdSoftPoint& localPoint,
			const PxVec3& worldTarget)
		{
			Entry* entry = findEntry(core);
			if(!entry || !Dy::avbdIsSoftPointValid(
					localPoint, 0, getParticleCount(*entry)) ||
				!worldTarget.isFinite())
				return PX_MAX_U32;

			PxU32 handle = mNextWorldPinHandle++;
			if(handle == PX_MAX_U32)
				handle = mNextWorldPinHandle++;
			if(handle == 0 || handle == PX_MAX_U32)
				return PX_MAX_U32;

			mWorldPins.pushBack(WorldPinEntry(
				core, localPoint, worldTarget, handle));
			if(!rebuildEntryPins(*entry))
			{
				mWorldPins.popBack();
				return PX_MAX_U32;
			}
			wakeEntry(*entry, ScInternalWakeCounterResetValue);
			return handle;
		}

		PxU32 AvbdCpuSoftScene::addWorldElementPin(
			ActorCore& core,
			bool surfaceElement,
			PxU32 elementIndex,
			const PxVec4& barycentric,
			const PxVec3& worldTarget)
		{
			Dy::AvbdSoftPoint point;
			if(!buildLocalElementPoint(
				core, surfaceElement, elementIndex,
				barycentric, point))
				return PX_MAX_U32;
			return addWorldPin(core, point, worldTarget);
		}

		bool AvbdCpuSoftScene::updateWorldPin(
			ActorCore& core,
			PxU32 handle,
			const PxVec3& worldTarget)
		{
			if(!worldTarget.isFinite())
				return false;
			for(PxU32 i = 0; i < mWorldPins.size(); i++)
			{
				WorldPinEntry& pin = mWorldPins[i];
				if(pin.softCore != &core || pin.handle != handle)
					continue;
				Entry* entry = findEntry(core);
				if(!entry)
					return false;
				const PxVec3 oldTarget = pin.worldTarget;
				pin.worldTarget = worldTarget;
				if(!rebuildEntryPins(*entry))
				{
					pin.worldTarget = oldTarget;
					const bool restored = rebuildEntryPins(*entry);
					PX_ASSERT(restored);
					PX_UNUSED(restored);
					return false;
				}
				wakeEntry(*entry, ScInternalWakeCounterResetValue);
				return true;
			}
			return false;
		}

		void AvbdCpuSoftScene::removeWorldPin(
			ActorCore& core,
			PxU32 handle)
		{
			for(PxU32 i = 0; i < mWorldPins.size(); i++)
			{
				if(mWorldPins[i].softCore != &core ||
					mWorldPins[i].handle != handle)
					continue;
				mWorldPins.replaceWithLast(i);
				Entry* entry = findEntry(core);
				if(entry)
				{
					const bool rebuilt = rebuildEntryPins(*entry);
					PX_ASSERT(rebuilt);
					PX_UNUSED(rebuilt);
					wakeEntry(
						*entry, ScInternalWakeCounterResetValue);
				}
				return;
			}
		}

		bool AvbdCpuSoftScene::computePrescribedAttachmentWorldTarget(
			RigidCore& prescribedCore,
			const PxVec3& actorLocalTarget,
			PxVec3& worldTarget) const
		{
			if(!actorLocalTarget.isFinite())
				return false;
			const PxActorType::Enum actorType =
				prescribedCore.getActorCoreType();
			if(actorType == PxActorType::eRIGID_STATIC)
			{
				const StaticCore& staticCore =
					static_cast<const StaticCore&>(
						prescribedCore);
				if(!staticCore.getSim())
					return false;
				const PxTransform& actorToWorld =
					staticCore.getActor2World();
				worldTarget =
					actorToWorld.transform(actorLocalTarget);
				return actorToWorld.isValid() &&
					worldTarget.isFinite();
			}
			if(actorType != PxActorType::eRIGID_DYNAMIC)
				return false;

			BodyCore& kinematicCore =
				static_cast<BodyCore&>(prescribedCore);
			BodySim* bodySim = kinematicCore.getSim();
			if(!bodySim || !bodySim->isKinematic() ||
				bodySim->isArticulationLink())
				return false;
			const PxsBodyCore& bodyCore = kinematicCore.getCore();
			PxTransform bodyToWorld = bodyCore.body2World;
			PxTransform commandedBodyToWorld;
			if(kinematicCore.getKinematicTarget(
				commandedBodyToWorld))
				bodyToWorld = commandedBodyToWorld;
			const PxVec3 bodyLocalTarget =
				bodyCore.getBody2Actor().getInverse().
					transform(actorLocalTarget);
			worldTarget = bodyToWorld.transform(bodyLocalTarget);
			return bodyToWorld.isValid() && worldTarget.isFinite();
		}

		PxU32 AvbdCpuSoftScene::addKinematicAttachment(
			ActorCore& softCore,
			BodyCore& kinematicCore,
			PxU32 localVertex,
			const PxVec3& actorLocalTarget)
		{
			Dy::AvbdSoftPoint point;
			point.setVertex(localVertex);
			return addPrescribedAttachment(
				softCore, kinematicCore, point,
				actorLocalTarget);
		}

		PxU32 AvbdCpuSoftScene::addPrescribedAttachment(
			ActorCore& softCore,
			RigidCore& prescribedCore,
			const Dy::AvbdSoftPoint& localPoint,
			const PxVec3& actorLocalTarget)
		{
			Entry* entry = findEntry(softCore);
			PxVec3 worldTarget;
			if(!entry || !Dy::avbdIsSoftPointValid(
					localPoint, 0, getParticleCount(*entry)) ||
				!computePrescribedAttachmentWorldTarget(
					prescribedCore, actorLocalTarget,
					worldTarget))
				return PX_MAX_U32;

			PxU32 handle =
				mNextPrescribedAttachmentHandle++;
			if(handle == PX_MAX_U32)
				handle =
					mNextPrescribedAttachmentHandle++;
			if(handle == 0 || handle == PX_MAX_U32)
				return PX_MAX_U32;

			mPrescribedAttachments.pushBack(
				PrescribedAttachmentEntry(
					softCore, prescribedCore, localPoint,
					actorLocalTarget, worldTarget, handle));
			if(!rebuildEntryPins(*entry))
			{
				mPrescribedAttachments.popBack();
				const bool restored = rebuildEntryPins(*entry);
				PX_ASSERT(restored);
				PX_UNUSED(restored);
				return PX_MAX_U32;
			}
			wakeEntry(*entry, ScInternalWakeCounterResetValue);
			return handle;
		}

		PxU32 AvbdCpuSoftScene::addKinematicElementAttachment(
			ActorCore& softCore,
			BodyCore& kinematicCore,
			bool surfaceElement,
			PxU32 elementIndex,
			const PxVec4& barycentric,
			const PxVec3& actorLocalTarget)
		{
			Dy::AvbdSoftPoint point;
			if(!buildLocalElementPoint(
				softCore, surfaceElement, elementIndex,
				barycentric, point))
				return PX_MAX_U32;
			return addPrescribedAttachment(
				softCore, kinematicCore, point,
				actorLocalTarget);
		}

		PxU32 AvbdCpuSoftScene::addStaticAttachment(
			ActorCore& softCore,
			StaticCore& staticCore,
			PxU32 localVertex,
			const PxVec3& actorLocalTarget)
		{
			Dy::AvbdSoftPoint point;
			point.setVertex(localVertex);
			return addPrescribedAttachment(
				softCore, staticCore, point,
				actorLocalTarget);
		}

		PxU32 AvbdCpuSoftScene::addStaticElementAttachment(
			ActorCore& softCore,
			StaticCore& staticCore,
			bool surfaceElement,
			PxU32 elementIndex,
			const PxVec4& barycentric,
			const PxVec3& actorLocalTarget)
		{
			Dy::AvbdSoftPoint point;
			if(!buildLocalElementPoint(
				softCore, surfaceElement, elementIndex,
				barycentric, point))
				return PX_MAX_U32;
			return addPrescribedAttachment(
				softCore, staticCore, point,
				actorLocalTarget);
		}

		bool AvbdCpuSoftScene::updatePrescribedAttachment(
			ActorCore& softCore,
			PxU32 handle,
			const PxVec3& actorLocalTarget)
		{
			for(PxU32 i = 0;
				i < mPrescribedAttachments.size(); i++)
			{
				PrescribedAttachmentEntry& attachment =
					mPrescribedAttachments[i];
				if(attachment.softCore != &softCore ||
					attachment.handle != handle)
					continue;
				Entry* entry = findEntry(softCore);
				PxVec3 worldTarget;
				if(!entry ||
					!computePrescribedAttachmentWorldTarget(
						*attachment.prescribedCore,
						actorLocalTarget, worldTarget))
					return false;
				const PxVec3 oldActorLocalTarget =
					attachment.actorLocalTarget;
				const PxVec3 oldWorldTarget =
					attachment.worldTarget;
				const PxVec3 oldPreviousWorldTarget =
					attachment.previousWorldTarget;
				const PxVec3 oldLambda =
					attachment.alLambda;
				attachment.actorLocalTarget =
					actorLocalTarget;
				attachment.previousWorldTarget =
					worldTarget;
				attachment.worldTarget = worldTarget;
				attachment.alLambda = PxVec3(0.0f);
				if(!rebuildEntryPins(*entry))
				{
					attachment.actorLocalTarget =
						oldActorLocalTarget;
					attachment.worldTarget = oldWorldTarget;
					attachment.previousWorldTarget =
						oldPreviousWorldTarget;
					attachment.alLambda = oldLambda;
					const bool restored =
						rebuildEntryPins(*entry);
					PX_ASSERT(restored);
					PX_UNUSED(restored);
					return false;
				}
				wakeEntry(
					*entry, ScInternalWakeCounterResetValue);
				return true;
			}
			return false;
		}

		void AvbdCpuSoftScene::removePrescribedAttachment(
			ActorCore& softCore,
			PxU32 handle)
		{
			for(PxU32 i = 0;
				i < mPrescribedAttachments.size(); i++)
			{
				const PrescribedAttachmentEntry& attachment =
					mPrescribedAttachments[i];
				if(attachment.softCore != &softCore ||
					attachment.handle != handle)
					continue;
				mPrescribedAttachments.replaceWithLast(i);
				Entry* entry = findEntry(softCore);
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
				return;
			}
		}

		PxU32 AvbdCpuSoftScene::addRigidAttachment(
			ActorCore& softCore,
			BodyCore& rigidCore,
			PxU32 localVertex,
			const PxVec3& actorLocalTarget)
		{
			Dy::AvbdSoftPoint point;
			point.setVertex(localVertex);
			return addRigidAttachment(
				softCore, rigidCore, point, actorLocalTarget);
		}

		PxU32 AvbdCpuSoftScene::addRigidAttachment(
			ActorCore& softCore,
			BodyCore& rigidCore,
			const Dy::AvbdSoftPoint& localPoint,
			const PxVec3& actorLocalTarget)
		{
			Entry* entry = findEntry(softCore);
			BodySim* bodySim = rigidCore.getSim();
			if(!entry || !Dy::avbdIsSoftPointValid(
					localPoint, 0, getParticleCount(*entry)) ||
				!actorLocalTarget.isFinite() || !bodySim ||
				bodySim->isKinematic() ||
				bodySim->isArticulationLink())
				return PX_MAX_U32;

			PxU32 handle = mNextRigidAttachmentHandle++;
			if(handle == PX_MAX_U32)
				handle = mNextRigidAttachmentHandle++;
			if(handle == 0 || handle == PX_MAX_U32)
				return PX_MAX_U32;

			mRigidAttachments.pushBack(
				RigidAttachmentEntry(
					softCore, rigidCore, localPoint,
					actorLocalTarget, handle));
			clearIslandSelectionStorages();
			mDynamicsOwnsStep = false;
			ensureNativeIslandEdge(*entry, rigidCore);
			wakeEntry(*entry, ScInternalWakeCounterResetValue);
			rigidCore.wakeUp(ScInternalWakeCounterResetValue);
			return handle;
		}

		PxU32 AvbdCpuSoftScene::addRigidElementAttachment(
			ActorCore& softCore,
			BodyCore& rigidCore,
			bool surfaceElement,
			PxU32 elementIndex,
			const PxVec4& barycentric,
			const PxVec3& actorLocalTarget)
		{
			Dy::AvbdSoftPoint point;
			if(!buildLocalElementPoint(
				softCore, surfaceElement, elementIndex,
				barycentric, point))
				return PX_MAX_U32;
			return addRigidAttachment(
				softCore, rigidCore, point, actorLocalTarget);
		}

		bool AvbdCpuSoftScene::updateRigidAttachment(
			ActorCore& softCore,
			PxU32 handle,
			const PxVec3& actorLocalTarget)
		{
			if(!actorLocalTarget.isFinite())
				return false;
			for(PxU32 i = 0; i < mRigidAttachments.size(); i++)
			{
				RigidAttachmentEntry& attachment =
					mRigidAttachments[i];
				if(attachment.softCore != &softCore ||
					attachment.handle != handle)
					continue;
				Entry* entry = findEntry(softCore);
				BodySim* bodySim =
					attachment.rigidCore->getSim();
				if(!entry || !bodySim ||
					bodySim->isKinematic() ||
					bodySim->isArticulationLink())
					return false;
				attachment.actorLocalTarget = actorLocalTarget;
				attachment.alLambda = PxVec3(0.0f);
				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
				ensureNativeIslandEdge(
					*entry, *attachment.rigidCore);
				wakeEntry(
					*entry, ScInternalWakeCounterResetValue);
				attachment.rigidCore->wakeUp(
					ScInternalWakeCounterResetValue);
				return true;
			}
			return false;
		}

		void AvbdCpuSoftScene::removeRigidAttachment(
			ActorCore& softCore,
			PxU32 handle)
		{
			for(PxU32 i = 0; i < mRigidAttachments.size(); i++)
			{
				const RigidAttachmentEntry& attachment =
					mRigidAttachments[i];
				if(attachment.softCore != &softCore ||
					attachment.handle != handle)
					continue;
				BodyCore* rigidCore = attachment.rigidCore;
				mRigidAttachments.replaceWithLast(i);
				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
				Entry* entry = findEntry(softCore);
				if(entry)
					wakeEntry(
						*entry, ScInternalWakeCounterResetValue);
				if(rigidCore && rigidCore->getSim())
					rigidCore->wakeUp(
						ScInternalWakeCounterResetValue);
				return;
			}
		}

		PxU32 AvbdCpuSoftScene::addArticulationAttachment(
			ActorCore& softCore,
			BodyCore& linkCore,
			PxU32 localVertex,
			const PxVec3& actorLocalTarget)
		{
			Dy::AvbdSoftPoint point;
			point.setVertex(localVertex);
			return addArticulationAttachment(
				softCore, linkCore, point, actorLocalTarget);
		}

		PxU32 AvbdCpuSoftScene::addArticulationAttachment(
			ActorCore& softCore,
			BodyCore& linkCore,
			const Dy::AvbdSoftPoint& localPoint,
			const PxVec3& actorLocalTarget)
		{
			Entry* entry = findEntry(softCore);
			BodySim* bodySim = linkCore.getSim();
			if(!entry || !Dy::avbdIsSoftPointValid(
					localPoint, 0, getParticleCount(*entry)) ||
				!actorLocalTarget.isFinite() || !bodySim ||
				!bodySim->isArticulationLink() ||
				!bodySim->getArticulation())
				return PX_MAX_U32;

			PxU32 handle =
				mNextArticulationAttachmentHandle++;
			if(handle == PX_MAX_U32)
				handle =
					mNextArticulationAttachmentHandle++;
			if(handle == 0 || handle == PX_MAX_U32)
				return PX_MAX_U32;

			mArticulationAttachments.pushBack(
				ArticulationAttachmentEntry(
					softCore, linkCore, localPoint,
					actorLocalTarget, handle));
			clearIslandSelectionStorages();
			mDynamicsOwnsStep = false;
			ensureNativeIslandEdge(*entry, linkCore);
			wakeEntry(*entry, ScInternalWakeCounterResetValue);
			linkCore.wakeUp(ScInternalWakeCounterResetValue);
			return handle;
		}

		PxU32 AvbdCpuSoftScene::addArticulationElementAttachment(
			ActorCore& softCore,
			BodyCore& linkCore,
			bool surfaceElement,
			PxU32 elementIndex,
			const PxVec4& barycentric,
			const PxVec3& actorLocalTarget)
		{
			Dy::AvbdSoftPoint point;
			if(!buildLocalElementPoint(
				softCore, surfaceElement, elementIndex,
				barycentric, point))
				return PX_MAX_U32;
			return addArticulationAttachment(
				softCore, linkCore, point, actorLocalTarget);
		}

		bool AvbdCpuSoftScene::updateArticulationAttachment(
			ActorCore& softCore,
			PxU32 handle,
			const PxVec3& actorLocalTarget)
		{
			if(!actorLocalTarget.isFinite())
				return false;
			for(PxU32 i = 0;
				i < mArticulationAttachments.size(); i++)
			{
				ArticulationAttachmentEntry& attachment =
					mArticulationAttachments[i];
				if(attachment.softCore != &softCore ||
					attachment.handle != handle)
					continue;
				Entry* entry = findEntry(softCore);
				BodySim* bodySim =
					attachment.linkCore->getSim();
				if(!entry || !bodySim ||
					!bodySim->isArticulationLink() ||
					!bodySim->getArticulation())
					return false;
				attachment.actorLocalTarget = actorLocalTarget;
				attachment.alLambda = PxVec3(0.0f);
				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
				ensureNativeIslandEdge(
					*entry, *attachment.linkCore);
				wakeEntry(
					*entry, ScInternalWakeCounterResetValue);
				attachment.linkCore->wakeUp(
					ScInternalWakeCounterResetValue);
				return true;
			}
			return false;
		}

		void AvbdCpuSoftScene::removeArticulationAttachment(
			ActorCore& softCore,
			PxU32 handle)
		{
			for(PxU32 i = 0;
				i < mArticulationAttachments.size(); i++)
			{
				const ArticulationAttachmentEntry& attachment =
					mArticulationAttachments[i];
				if(attachment.softCore != &softCore ||
					attachment.handle != handle)
					continue;
				BodyCore* linkCore = attachment.linkCore;
				mArticulationAttachments.replaceWithLast(i);
				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
				Entry* entry = findEntry(softCore);
				if(entry)
					wakeEntry(
						*entry, ScInternalWakeCounterResetValue);
				if(linkCore && linkCore->getSim())
					linkCore->wakeUp(
						ScInternalWakeCounterResetValue);
				return;
			}
		}

		PxU32 AvbdCpuSoftScene::addSoftPairAttachment(
			ActorCore& softCore0,
			const Dy::AvbdSoftPoint& localPoint0,
			ActorCore& softCore1,
			const Dy::AvbdSoftPoint& localPoint1)
		{
			Entry* entry0 = findEntry(softCore0);
			Entry* entry1 = findEntry(softCore1);
			if(&softCore0 == &softCore1 || !entry0 || !entry1 ||
				!Dy::avbdIsSoftPointValid(
					localPoint0, 0, getParticleCount(*entry0)) ||
				!Dy::avbdIsSoftPointValid(
					localPoint1, 0, getParticleCount(*entry1)))
				return PX_MAX_U32;

			PxU32 handle = mNextSoftPairAttachmentHandle++;
			if(handle == PX_MAX_U32)
				handle = mNextSoftPairAttachmentHandle++;
			if(handle == 0 || handle == PX_MAX_U32)
				return PX_MAX_U32;

			mSoftPairAttachments.pushBack(
				SoftPairAttachmentEntry(
					softCore0, localPoint0,
					softCore1, localPoint1, handle));
			clearIslandSelectionStorages();
			mDynamicsOwnsStep = false;
			ensureNativeSoftSoftIslandEdge(*entry0, *entry1);
			wakeEntry(*entry0, ScInternalWakeCounterResetValue);
			wakeEntry(*entry1, ScInternalWakeCounterResetValue);
			return handle;
		}

		PxU32 AvbdCpuSoftScene::addSoftPairAttachment(
			ActorCore& softCore0,
			bool element0,
			PxU32 index0,
			const PxVec4& barycentric0,
			ActorCore& softCore1,
			bool element1,
			PxU32 index1,
			const PxVec4& barycentric1)
		{
			Entry* entry0 = findEntry(softCore0);
			Entry* entry1 = findEntry(softCore1);
			if(!entry0 || !entry1)
				return PX_MAX_U32;

			Dy::AvbdSoftPoint point0;
			Dy::AvbdSoftPoint point1;
			if(element0)
			{
				if(!buildLocalElementPoint(
					softCore0, entry0->kind == eSURFACE,
					index0, barycentric0, point0))
					return PX_MAX_U32;
			}
			else
				point0.setVertex(index0);
			if(element1)
			{
				if(!buildLocalElementPoint(
					softCore1, entry1->kind == eSURFACE,
					index1, barycentric1, point1))
					return PX_MAX_U32;
			}
			else
				point1.setVertex(index1);
			return addSoftPairAttachment(
				softCore0, point0, softCore1, point1);
		}

		void AvbdCpuSoftScene::removeSoftPairAttachment(
			ActorCore& softCore,
			PxU32 handle)
		{
			for(PxU32 i = 0; i < mSoftPairAttachments.size(); i++)
			{
				const SoftPairAttachmentEntry& attachment =
					mSoftPairAttachments[i];
				if(attachment.softCore[0] != &softCore ||
					attachment.handle != handle)
					continue;
				ActorCore* softCore0 = attachment.softCore[0];
				ActorCore* softCore1 = attachment.softCore[1];
				mSoftPairAttachments.replaceWithLast(i);
				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
				Entry* entry0 =
					softCore0 ? findEntry(*softCore0) : NULL;
				Entry* entry1 =
					softCore1 ? findEntry(*softCore1) : NULL;
				if(entry0)
					wakeEntry(
						*entry0, ScInternalWakeCounterResetValue);
				if(entry1)
					wakeEntry(
						*entry1, ScInternalWakeCounterResetValue);
				return;
			}
		}

		PxU32 AvbdCpuSoftScene::addRigidActorFilter(
			ActorCore& softCore,
			ActorCore& rigidCore,
			const PxU32* elementIndices,
			PxU32 elementCount,
			bool filterAllElements)
		{
			if(!findEntry(softCore) ||
				(!filterAllElements &&
					(!elementIndices || elementCount == 0)))
				return PX_MAX_U32;
			PxU32 handle = mNextRigidActorFilterHandle++;
			if(handle == PX_MAX_U32)
				handle = mNextRigidActorFilterHandle++;
			if(handle == 0 || handle == PX_MAX_U32)
				return PX_MAX_U32;
			mRigidActorFilters.pushBack(
				RigidActorFilterEntry(
					softCore, rigidCore,
					elementIndices, elementCount,
					handle, filterAllElements));
			return handle;
		}

		PxU32 AvbdCpuSoftScene::addVolumeRigidActorFilter(
			DeformableVolumeCore& softCore,
			ActorCore& rigidCore,
			const PxU32* collisionElementIndices,
			PxU32 collisionElementCount,
			bool filterAllElements)
		{
			if(filterAllElements)
				return addRigidActorFilter(
					softCore, rigidCore);
			Entry* entry = findEntry(softCore);
			if(!entry || entry->kind != eVOLUME ||
				!collisionElementIndices ||
				collisionElementCount == 0 ||
				!entry->collisionMesh ||
				!entry->simulationMesh ||
				!entry->auxData)
				return PX_MAX_U32;
			const PxU32 publicElementCount =
				entry->collisionMesh->getNbTetrahedrons();
			for(PxU32 selectedIndex = 0;
				selectedIndex < collisionElementCount;
				++selectedIndex)
			{
				if(collisionElementIndices[selectedIndex] >= publicElementCount)
					return PX_MAX_U32;
			}
			return addRigidActorFilter(
				softCore, rigidCore,
				collisionElementIndices,
				collisionElementCount, false);
		}

		void AvbdCpuSoftScene::removeRigidActorFilter(
			ActorCore& softCore,
			PxU32 handle)
		{
			for(PxU32 i = 0; i < mRigidActorFilters.size(); ++i)
			{
				const RigidActorFilterEntry& filter =
					mRigidActorFilters[i];
				if(filter.softCore == &softCore &&
					filter.handle == handle)
				{
					mRigidActorFilters.replaceWithLast(i);
					return;
				}
			}
		}

		PxU32 AvbdCpuSoftScene::addCompiledDeformablePairFilter(
			ActorCore& core0,
			ActorCore& core1,
			const PxU32* elementIndices0,
			const PxU32* elementIndices1,
			PxU32 pairCount)
		{
			if(&core0 == &core1 || !elementIndices0 ||
				!elementIndices1 || pairCount == 0)
				return PX_MAX_U32;
			PxU32 handle = mNextDeformablePairFilterHandle++;
			if(handle == PX_MAX_U32)
				handle = mNextDeformablePairFilterHandle++;
			if(handle == 0 || handle == PX_MAX_U32)
				return PX_MAX_U32;
			mDeformablePairFilters.pushBack(
				DeformablePairFilterEntry(
					core0, core1, elementIndices0,
					elementIndices1, pairCount, handle));
			return handle;
		}

		bool AvbdCpuSoftScene::expandVolumeCollisionElement(
			const Entry& entry,
			PxU32 collisionElement,
			PxArray<PxU32>& simulationElements) const
		{
			simulationElements.clear();
			if(collisionElement == eELEMENT_FILTER_ALL)
			{
				simulationElements.pushBack(eELEMENT_FILTER_ALL);
				return true;
			}
			if(entry.kind != eVOLUME ||
				!entry.collisionMesh ||
				!entry.simulationMesh ||
				!entry.auxData)
				return false;
			const PxU32 collisionElementCount =
				entry.collisionMesh->getNbTetrahedrons();
			if(collisionElement >= collisionElementCount)
				return false;
			simulationElements.pushBack(collisionElement);
			return true;
		}

		PxU32 AvbdCpuSoftScene::addSurfaceSurfaceFilter(
			DeformableSurfaceCore& core0,
			DeformableSurfaceCore& core1,
			const PxU32* elementIndices0,
			const PxU32* elementIndices1,
			PxU32 pairCount)
		{
			if(&core0 == &core1 || !elementIndices0 ||
				!elementIndices1 || pairCount == 0)
				return PX_MAX_U32;
			Entry* entry0 = findEntry(core0);
			Entry* entry1 = findEntry(core1);
			if(!entry0 || !entry1 ||
				entry0->kind != eSURFACE ||
				entry1->kind != eSURFACE ||
				!entry0->triangleMesh ||
				!entry1->triangleMesh)
				return PX_MAX_U32;
			const PxU32 elementCount0 =
				entry0->triangleMesh->getNbTriangles();
			const PxU32 elementCount1 =
				entry1->triangleMesh->getNbTriangles();
			for(PxU32 i = 0; i < pairCount; ++i)
			{
				if((elementIndices0[i] != eELEMENT_FILTER_ALL &&
						elementIndices0[i] >= elementCount0) ||
					(elementIndices1[i] != eELEMENT_FILTER_ALL &&
						elementIndices1[i] >= elementCount1))
					return PX_MAX_U32;
			}
			return addCompiledDeformablePairFilter(
				core0, core1, elementIndices0,
				elementIndices1, pairCount);
		}

		PxU32 AvbdCpuSoftScene::addVolumeSurfaceFilter(
			DeformableVolumeCore& volumeCore,
			DeformableSurfaceCore& surfaceCore,
			const PxU32* volumeCollisionElements,
			const PxU32* surfaceElements,
			PxU32 pairCount)
		{
			if(!volumeCollisionElements || !surfaceElements ||
				pairCount == 0)
				return PX_MAX_U32;
			Entry* volumeEntry = findEntry(volumeCore);
			Entry* surfaceEntry = findEntry(surfaceCore);
			if(!volumeEntry || !surfaceEntry ||
				volumeEntry->kind != eVOLUME ||
				surfaceEntry->kind != eSURFACE ||
				!surfaceEntry->triangleMesh)
				return PX_MAX_U32;
			const PxU32 surfaceElementCount =
				surfaceEntry->triangleMesh->getNbTriangles();
			PxArray<PxU32> compiledVolumeElements;
			PxArray<PxU32> compiledSurfaceElements;
			PxArray<PxU32> expandedVolumeElements;
			for(PxU32 pairIndex = 0;
				pairIndex < pairCount; ++pairIndex)
			{
				const PxU32 surfaceElement =
					surfaceElements[pairIndex];
				if(surfaceElement != eELEMENT_FILTER_ALL &&
					surfaceElement >= surfaceElementCount)
					return PX_MAX_U32;
				if(!expandVolumeCollisionElement(
					*volumeEntry,
					volumeCollisionElements[pairIndex],
					expandedVolumeElements))
					return PX_MAX_U32;
				for(PxU32 i = 0;
					i < expandedVolumeElements.size(); ++i)
				{
					compiledVolumeElements.pushBack(
						expandedVolumeElements[i]);
					compiledSurfaceElements.pushBack(
						surfaceElement);
				}
			}
			return addCompiledDeformablePairFilter(
				volumeCore, surfaceCore,
				compiledVolumeElements.begin(),
				compiledSurfaceElements.begin(),
				compiledVolumeElements.size());
		}

		PxU32 AvbdCpuSoftScene::addVolumeVolumeFilter(
			DeformableVolumeCore& core0,
			DeformableVolumeCore& core1,
			const PxU32* collisionElements0,
			const PxU32* collisionElements1,
			PxU32 pairCount)
		{
			if(&core0 == &core1 || !collisionElements0 ||
				!collisionElements1 || pairCount == 0)
				return PX_MAX_U32;
			Entry* entry0 = findEntry(core0);
			Entry* entry1 = findEntry(core1);
			if(!entry0 || !entry1 ||
				entry0->kind != eVOLUME ||
				entry1->kind != eVOLUME)
				return PX_MAX_U32;
			PxArray<PxU32> compiledElements0;
			PxArray<PxU32> compiledElements1;
			PxArray<PxU32> expandedElements0;
			PxArray<PxU32> expandedElements1;
			for(PxU32 pairIndex = 0;
				pairIndex < pairCount; ++pairIndex)
			{
				if(!expandVolumeCollisionElement(
						*entry0,
						collisionElements0[pairIndex],
						expandedElements0) ||
					!expandVolumeCollisionElement(
						*entry1,
						collisionElements1[pairIndex],
						expandedElements1))
					return PX_MAX_U32;
				for(PxU32 i = 0;
					i < expandedElements0.size(); ++i)
				{
					for(PxU32 j = 0;
						j < expandedElements1.size(); ++j)
					{
						compiledElements0.pushBack(
							expandedElements0[i]);
						compiledElements1.pushBack(
							expandedElements1[j]);
					}
				}
			}
			return addCompiledDeformablePairFilter(
				core0, core1,
				compiledElements0.begin(),
				compiledElements1.begin(),
				compiledElements0.size());
		}

		void AvbdCpuSoftScene::removeDeformablePairFilter(
			ActorCore& core,
			PxU32 handle)
		{
			for(PxU32 i = 0;
				i < mDeformablePairFilters.size(); ++i)
			{
				const DeformablePairFilterEntry& filter =
					mDeformablePairFilters[i];
				if((filter.core0 == &core ||
						filter.core1 == &core) &&
					filter.handle == handle)
				{
					mDeformablePairFilters.replaceWithLast(i);
					return;
				}
			}
		}

		void AvbdCpuSoftScene::removeEntry(ActorCore& core)
		{
			for(PxU32 entryIndex = 0;
				entryIndex < mEntries.size(); entryIndex++)
			{
				Entry& entry = mEntries[entryIndex];
				if(entry.getActorCore() != &core)
					continue;

				clearIslandSelectionStorages();
				mDynamicsOwnsStep = false;
				removeNativeIslandEdgesForSoft(core);
				removePrescribedAttachmentsForSoft(core);
				removeRigidAttachmentsForSoft(core);
				removeArticulationAttachmentsForSoft(core);
				removeSoftPairAttachmentsForSoft(core);
				removeWorldPinsForCore(core);
				mIslandManager.removeNode(entry.islandNode);
				entry.destroyIslandObject();

				const PxU32 removedParticleStart =
					getParticleStart(entry);
				const PxU32 removedParticleCount =
					getParticleCount(entry);
				for(PxU32 i = 0; i < mEntries.size(); i++)
				{
					if(i == entryIndex)
						continue;
					Entry& remainingEntry = mEntries[i];
					const PxU32 remainingParticleStart =
						getParticleStart(remainingEntry);
					const PxU32 remainingParticleCount =
						getParticleCount(remainingEntry);
					if(remainingParticleStart <
						removedParticleStart + removedParticleCount)
						continue;
					const PxU32 rebasedParticleStart =
						remainingParticleStart -
						removedParticleCount;
					const bool rebased =
						rebaseSoftBodyParticleRangeInPlace(
							mBodies[remainingEntry.bodyIndex],
							remainingParticleStart,
							remainingParticleCount,
							rebasedParticleStart);
					PX_ASSERT(rebased);
					PX_UNUSED(rebased);
				}
				mParticles.removeRange(
					removedParticleStart, removedParticleCount);
				mContacts.clear();
				mWorkspace.reset();

				const PxU32 removedBodyIndex = entry.bodyIndex;
				const PxU32 lastBodyIndex = mBodies.size() - 1;
				if(removedBodyIndex != lastBodyIndex)
				{
					for(PxU32 i = 0; i < mEntries.size(); i++)
					{
						if(mEntries[i].bodyIndex == lastBodyIndex)
						{
							mEntries[i].bodyIndex = removedBodyIndex;
							break;
						}
					}
				}
				PX_ASSERT(
					mSelfCollisionAdjacencies.size() == mBodies.size());
				mSelfCollisionAdjacencies.replaceWithLast(
					removedBodyIndex);
				mSelfCollisionEnabled.clear();
				mBodies.replaceWithLast(removedBodyIndex);
				mEntries.replaceWithLast(entryIndex);
				if(mEntries.empty())
				{
					clearNativeIslandEdges();
					mParticles.clear();
					mBodies.clear();
					mSelfCollisionAdjacencies.clear();
					mSelfCollisionEnabled.clear();
					mContacts.clear();
					mWorkspace.reset();
					mWorkspacePreflightPending = true;
					mCollisionProxy.collisionParticles.clear();
					mCollisionProxy.collisionBodies.clear();
					mCollisionProxy.collisionVertexMappings.clear();
					mCollisionProxy.collisionSelfCollisionAdjacencies.clear();
				}
				else
				{
					const bool rebuilt = rebuildCollisionDetectionScene();
					PX_ASSERT(rebuilt);
					PX_UNUSED(rebuilt);
				}
				return;
			}
		}

} // namespace Sc
} // namespace physx
