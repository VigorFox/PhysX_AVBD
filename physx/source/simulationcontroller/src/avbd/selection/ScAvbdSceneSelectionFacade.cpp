// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/scene/ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

		void AvbdCpuSoftScene::clearIslandSelectionStorages()
		{
			mIslandSelectionStorages.release();
		}

		void AvbdCpuSoftScene::invalidateNativeIslandSelectionCaches()
		{
			mIslandSelectionStorages.invalidate();
			mDynamicsOwnsStep = false;
			mDynamicsSelectedEntryCount = 0;
		}

		AvbdCpuSoftScene::IslandSelectionStorage*
			AvbdCpuSoftScene::acquireIslandSelectionStorage(
			IG::IslandId nativeIslandId)
		{
			return mIslandSelectionStorages.acquire(nativeIslandId);
		}

		bool AvbdCpuSoftScene::rebaseParticleIndex(
			PxU32& index, PxU32 globalStart,
			PxU32 particleCount, PxU32 localStart)
		{
			if(index == PX_MAX_U32)
				return true;
			if(index < globalStart ||
				index - globalStart >= particleCount)
				return false;
			index = localStart + (index - globalStart);
			return true;
		}

		bool AvbdCpuSoftScene::copyAndRebaseSoftBody(
			const Dy::AvbdSoftBody& source,
			PxU32 globalStart, PxU32 particleCount,
			PxU32 localStart,
			Dy::AvbdSoftBody& destination)
		{
			if(source.compiled.particleStart != globalStart ||
				source.compiled.particleCount != particleCount ||
				!source.runtime.attachments.empty())
				return false;
			destination = source;
			// Native selection appends island-local attachments immediately after
			// this rebase and compiles the objective program once from that final
			// state.  Do not compile an intermediate program that cannot be
			// consumed before it is invalidated again.
			return rebaseSoftBodyParticleRangeInPlace(
				destination, globalStart, particleCount, localStart, false);
		}

		bool AvbdCpuSoftScene::rebaseSoftBodyParticleRangeInPlace(
			Dy::AvbdSoftBody& body,
			PxU32 oldStart, PxU32 particleCount,
			PxU32 newStart, bool compileObjectives)
		{
			if(body.compiled.particleStart != oldStart ||
				body.compiled.particleCount != particleCount)
				return false;
			body.compiled.particleStart = newStart;
			for(PxU32 i = 0;
				i < body.compiled.triElements.size(); i++)
			{
				Dy::AvbdTriElement& element =
					body.compiled.triElements[i];
				if(!rebaseParticleIndex(
					element.p0, oldStart, particleCount, newStart) ||
					!rebaseParticleIndex(
						element.p1, oldStart, particleCount, newStart) ||
					!rebaseParticleIndex(
						element.p2, oldStart, particleCount, newStart))
					return false;
			}
			for(PxU32 i = 0;
				i < body.compiled.tetElements.size(); i++)
			{
				Dy::AvbdTetElement& element =
					body.compiled.tetElements[i];
				if(!rebaseParticleIndex(
					element.p0, oldStart, particleCount, newStart) ||
					!rebaseParticleIndex(
						element.p1, oldStart, particleCount, newStart) ||
					!rebaseParticleIndex(
						element.p2, oldStart, particleCount, newStart) ||
					!rebaseParticleIndex(
						element.p3, oldStart, particleCount, newStart))
					return false;
			}
			for(PxU32 i = 0;
				i < body.compiled.bendElements.size(); i++)
			{
				Dy::AvbdBendingElement& element =
					body.compiled.bendElements[i];
				if(!rebaseParticleIndex(
					element.opp0, oldStart, particleCount, newStart) ||
					!rebaseParticleIndex(
						element.opp1, oldStart, particleCount, newStart) ||
					!rebaseParticleIndex(
						element.edgeStart, oldStart,
						particleCount, newStart) ||
					!rebaseParticleIndex(
						element.edgeEnd, oldStart,
						particleCount, newStart))
					return false;
			}
			for(PxU32 i = 0; i < body.compiled.edges.size(); i++)
			{
				Dy::AvbdEdgeInfo& edge = body.compiled.edges[i];
				if(!rebaseParticleIndex(
					edge.p0, oldStart, particleCount, newStart) ||
					!rebaseParticleIndex(
						edge.p1, oldStart, particleCount, newStart))
					return false;
			}
			for(PxU32 i = 0;
				i < body.compiled.surfaceTriangles.size(); i++)
			{
				if(!rebaseParticleIndex(
					body.compiled.surfaceTriangles[i],
					oldStart, particleCount, newStart))
					return false;
			}
			for(PxU32 i = 0;
				i < body.compiled.surfaceVertices.size(); i++)
			{
				if(!rebaseParticleIndex(
					body.compiled.surfaceVertices[i],
					oldStart, particleCount, newStart))
					return false;
			}
			for(PxU32 i = 0;
				i < body.compiled.surfaceEdges.size(); i++)
			{
				Dy::AvbdEdgeInfo& edge =
					body.compiled.surfaceEdges[i];
				if(!rebaseParticleIndex(
						edge.p0, oldStart,
						particleCount, newStart) ||
					!rebaseParticleIndex(
						edge.p1, oldStart,
						particleCount, newStart))
					return false;
			}
			for(PxU32 i = 0;
				i < body.runtime.attachments.size(); i++)
			{
				Dy::AvbdSoftPoint& point =
					body.runtime.attachments[i].point;
				for(PxU32 endpoint = 0;
					endpoint < point.particleCount; endpoint++)
				{
					if(!rebaseParticleIndex(
						point.particleIndices[endpoint],
						oldStart, particleCount, newStart))
						return false;
				}
			}
			for(PxU32 i = 0; i < body.runtime.pins.size(); i++)
			{
				Dy::AvbdSoftPoint& point =
					body.runtime.pins[i].point;
				for(PxU32 endpoint = 0;
					endpoint < point.particleCount; endpoint++)
				{
					if(!rebaseParticleIndex(
						point.particleIndices[endpoint],
						oldStart, particleCount, newStart))
						return false;
				}
			}
			if(!compileObjectives)
				return true;
			body.runtime.compileObjectiveProgram(
				newStart, particleCount);
			return body.runtime.isObjectiveProgramCurrent(
				newStart, particleCount);
		}

		bool AvbdCpuSoftScene::getCanonicalIslandParticleRange(
			const IslandSelectionStorage& storage,
			PxU32& particleStart,
			PxU32& particleCount) const
		{
			particleStart = PX_MAX_U32;
			particleCount = 0;
			for(PxU32 entryOrder = 0;
				entryOrder < storage.entryIndices.size(); entryOrder++)
			{
				const PxU32 entryIndex = storage.entryIndices[entryOrder];
				if(entryIndex >= mEntries.size())
					return false;
				const Entry& entry = mEntries[entryIndex];
				const PxU32 entryParticleStart =
					getParticleStart(entry);
				const PxU32 entryParticleCount =
					getParticleCount(entry);
				if(entryParticleStart > mParticles.size() ||
					entryParticleCount >
						mParticles.size() - entryParticleStart)
					return false;
				if(particleStart == PX_MAX_U32)
					particleStart = entryParticleStart;
				if(entryParticleStart != particleStart + particleCount ||
					entryParticleCount > PX_MAX_U32 - particleCount)
					return false;
				particleCount += entryParticleCount;
			}
			return particleStart != PX_MAX_U32 && particleCount > 0;
		}

		Dy::AvbdSoftParticle* AvbdCpuSoftScene::getIslandSelectionParticles(
			IslandSelectionStorage& storage)
		{
			if(storage.usesCanonicalParticleRange)
			{
				PX_ASSERT(
					storage.canonicalParticleStart <= mParticles.size() &&
					storage.canonicalParticleCount <=
						mParticles.size() -
						storage.canonicalParticleStart);
				return mParticles.begin() +
					storage.canonicalParticleStart;
			}
			return storage.particles.begin();
		}

		PxU32 AvbdCpuSoftScene::getIslandSelectionParticleCount(
			const IslandSelectionStorage& storage) const
		{
			return storage.usesCanonicalParticleRange
				? storage.canonicalParticleCount
				: storage.particles.size();
		}

		bool AvbdCpuSoftScene::findRigidBodyIndexInIsland(
			BodyCore& rigidCore,
			PxsRigidBody* const* rigidBodies,
			const Dy::AvbdSolverBody* solverBodies,
			PxU32 bodyStart,
			PxU32 bodyCount,
			PxU32& localBodyIndex) const
		{
			BodySim* bodySim = rigidCore.getSim();
			if(!bodySim || bodySim->isKinematic() ||
				bodySim->isArticulationLink())
				return false;
			const PxsRigidBody* lowLevelBody =
				&bodySim->getLowLevelBody();
			for(PxU32 i = 0; i < bodyCount; i++)
			{
				const PxU32 globalBodyIndex = bodyStart + i;
				if(rigidBodies[globalBodyIndex] != lowLevelBody)
					continue;
				if(solverBodies[globalBodyIndex].isStatic())
					return false;
				localBodyIndex = i;
				return true;
			}
			return false;
		}

		bool AvbdCpuSoftScene::findArticulationBodyIndexInIsland(
			BodyCore& linkCore,
			Dy::FeatherstoneArticulation* const*
				articulationForBody,
			const PxU32* linkIndexForBody,
			const Dy::AvbdSolverBody* solverBodies,
			PxU32 bodyStart,
			PxU32 bodyCount,
			PxU32& localBodyIndex) const
		{
			BodySim* bodySim = linkCore.getSim();
			if(!bodySim || !bodySim->isArticulationLink() ||
				!bodySim->getArticulation())
				return false;
			for(PxU32 i = 0; i < bodyCount; i++)
			{
				const PxU32 globalBodyIndex = bodyStart + i;
				Dy::FeatherstoneArticulation* articulation =
					articulationForBody[globalBodyIndex];
				if(!articulation ||
					articulation != bodySim->getArticulation())
					continue;
				const PxU32 linkIndex =
					linkIndexForBody[globalBodyIndex];
				const Dy::ArticulationData& data =
					articulation->getArticulationData();
				if(linkIndex >= data.getLinkCount() ||
					data.getLink(linkIndex).bodyCore !=
						&linkCore.getCore())
					continue;
				if(solverBodies[globalBodyIndex].isStatic())
					return false;
				localBodyIndex = i;
				return true;
			}
			return false;
		}

		RigidAttachmentEntry*
			AvbdCpuSoftScene::findRigidAttachment(
			ActorCore& softCore,
			PxU32 handle)
		{
			for(PxU32 i = 0; i < mRigidAttachments.size(); i++)
			{
				RigidAttachmentEntry& attachment =
					mRigidAttachments[i];
				if(attachment.softCore == &softCore &&
					attachment.handle == handle)
					return &attachment;
			}
			return NULL;
		}

		ArticulationAttachmentEntry*
			AvbdCpuSoftScene::findArticulationAttachment(
				ActorCore& softCore,
				PxU32 handle)
		{
			for(PxU32 i = 0;
				i < mArticulationAttachments.size(); i++)
			{
				ArticulationAttachmentEntry& attachment =
					mArticulationAttachments[i];
				if(attachment.softCore == &softCore &&
					attachment.handle == handle)
					return &attachment;
			}
			return NULL;
		}

		SoftPairAttachmentEntry*
			AvbdCpuSoftScene::findSoftPairAttachment(
			ActorCore& softCore,
			PxU32 handle)
		{
			for(PxU32 i = 0; i < mSoftPairAttachments.size(); i++)
			{
				SoftPairAttachmentEntry& attachment =
					mSoftPairAttachments[i];
				if(attachment.softCore[0] == &softCore &&
					attachment.handle == handle)
					return &attachment;
			}
			return NULL;
		}

		PrescribedAttachmentEntry*
			AvbdCpuSoftScene::findPrescribedAttachment(
			ActorCore& softCore,
			PxU32 handle)
		{
			for(PxU32 i = 0;
				i < mPrescribedAttachments.size(); i++)
			{
				PrescribedAttachmentEntry& attachment =
					mPrescribedAttachments[i];
				if(attachment.softCore == &softCore &&
					attachment.handle == handle)
					return &attachment;
			}
			return NULL;
		}

		void AvbdCpuSoftScene::compileDynamicBoxesForIsland(
			PxsRigidBody* const* rigidBodies,
			const Dy::AvbdSolverBody* solverBodies,
			PxU32 bodyStart,
			PxU32 bodyCount,
			PxArray<Dy::AvbdRigidBox>& boxes)
		{
			boxes.clear();
			for(PxU32 shapeIndex = 0;
				shapeIndex < mDynamicShapes.size(); shapeIndex++)
			{
				const DynamicShapeEntry& entry =
					mDynamicShapes[shapeIndex];
				BodySim* bodySim = entry.core->getSim();
				if(!bodySim || bodySim->isKinematic() ||
					bodySim->isArticulationLink())
					continue;
				const ShapeCore& shape = *entry.shape;
				if(!(shape.getFlags() &
						PxShapeFlag::eSIMULATION_SHAPE) ||
					shape.getGeometryType() !=
						PxGeometryType::eBOX)
					continue;

				const PxsRigidBody* lowLevelBody =
					&bodySim->getLowLevelBody();
				PxU32 globalBodyIndex = PX_MAX_U32;
				for(PxU32 localBodyIndex = 0;
					localBodyIndex < bodyCount; localBodyIndex++)
				{
					const PxU32 candidateIndex =
						bodyStart + localBodyIndex;
					if(rigidBodies[candidateIndex] ==
						lowLevelBody)
					{
						globalBodyIndex = candidateIndex;
						break;
					}
				}
				if(globalBodyIndex == PX_MAX_U32 ||
					solverBodies[globalBodyIndex].isStatic())
					continue;

				const PxsBodyCore& bodyCore =
					entry.core->getCore();
				const PxTransform actorToWorld =
					bodyCore.body2World *
					bodyCore.getBody2Actor().getInverse();
				const PxTransform shapeToWorld =
					actorToWorld * shape.getShape2Actor();
				if(!shapeToWorld.isValid())
					continue;

				const PxBoxGeometry& geometry =
					static_cast<const PxBoxGeometry&>(
						shape.getGeometry());
				Dy::AvbdRigidBox box;
				box.center = shapeToWorld.p;
				box.rotation = shapeToWorld.q;
				box.halfExtent = geometry.halfExtents;
				box.friction = getStaticFriction(
					shape, mRigidMaterialManager);
				box.frictionCombineMode =
					getStaticFrictionCombineMode(
						shape, mRigidMaterialManager);
				box.primitiveKey = entry.primitiveKey;
				box.targetKind =
					Dy::AvbdSoftContactTargetKind::eRIGID_BODY;
				box.targetIndex =
					globalBodyIndex - bodyStart;
				box.shapeToRigidBody =
					bodyCore.body2World.getInverse() *
					shapeToWorld;
				boxes.pushBack(box);
			}
		}

		void AvbdCpuSoftScene::compileDynamicSpheresForIsland(
			PxsRigidBody* const* rigidBodies,
			Dy::AvbdSolverBody* solverBodies,
			PxU32 bodyStart,
			PxU32 bodyCount,
			PxReal dt,
			const PxVec3& gravity,
			PxArray<Dy::AvbdRigidSphere>& spheres)
		{
			spheres.clear();
			for(PxU32 shapeIndex = 0;
				shapeIndex < mDynamicShapes.size(); shapeIndex++)
			{
				const DynamicShapeEntry& entry =
					mDynamicShapes[shapeIndex];
				BodySim* bodySim = entry.core->getSim();
				if(!bodySim || bodySim->isKinematic() ||
					bodySim->isArticulationLink())
					continue;

				Dy::AvbdRigidSphere sphere;
				if(!compileDynamicSphere(entry, sphere))
					continue;

				const PxsRigidBody* lowLevelBody =
					&bodySim->getLowLevelBody();
				PxU32 globalBodyIndex = PX_MAX_U32;
				for(PxU32 localBodyIndex = 0;
					localBodyIndex < bodyCount; localBodyIndex++)
				{
					const PxU32 candidateIndex =
						bodyStart + localBodyIndex;
					if(rigidBodies[candidateIndex] ==
						lowLevelBody)
					{
						globalBodyIndex = candidateIndex;
						break;
					}
				}
				if(globalBodyIndex == PX_MAX_U32 ||
					solverBodies[globalBodyIndex].isStatic())
					continue;

				const PxsBodyCore& bodyCore =
					entry.core->getCore();
				const PxTransform shapeToWorld(
					sphere.center, sphere.rotation);
				sphere.targetKind =
					Dy::AvbdSoftContactTargetKind::eRIGID_BODY;
				sphere.targetIndex =
					globalBodyIndex - bodyStart;
				sphere.shapeToRigidBody =
					bodyCore.body2World.getInverse() *
						shapeToWorld;
				Dy::AvbdSolverBody& solverBody =
					solverBodies[globalBodyIndex];
				solverBody.computePrediction(dt, gravity);
				const PxTransform predictedBodyToWorld(
					solverBody.predictedPosition,
					solverBody.predictedRotation);
				const PxTransform predictedShapeToWorld =
					predictedBodyToWorld * sphere.shapeToRigidBody;
				if(predictedShapeToWorld.isValid())
				{
					sphere.predictedCenter =
						predictedShapeToWorld.p;
					sphere.predictedRotation =
						predictedShapeToWorld.q;
					sphere.predictedPoseValid = true;
				}
				spheres.pushBack(sphere);
			}
		}

		void AvbdCpuSoftScene::compileDynamicCapsulesForIsland(
			PxsRigidBody* const* rigidBodies,
			Dy::AvbdSolverBody* solverBodies,
			PxU32 bodyStart,
			PxU32 bodyCount,
			PxReal dt,
			const PxVec3& gravity,
			PxArray<Dy::AvbdRigidCapsule>& capsules)
		{
			capsules.clear();
			for(PxU32 shapeIndex = 0;
				shapeIndex < mDynamicShapes.size(); shapeIndex++)
			{
				const DynamicShapeEntry& entry =
					mDynamicShapes[shapeIndex];
				BodySim* bodySim = entry.core->getSim();
				if(!bodySim || bodySim->isKinematic() ||
					bodySim->isArticulationLink())
					continue;

				Dy::AvbdRigidCapsule capsule;
				if(!compileDynamicCapsule(entry, capsule))
					continue;

				const PxsRigidBody* lowLevelBody =
					&bodySim->getLowLevelBody();
				PxU32 globalBodyIndex = PX_MAX_U32;
				for(PxU32 localBodyIndex = 0;
					localBodyIndex < bodyCount; localBodyIndex++)
				{
					const PxU32 candidateIndex =
						bodyStart + localBodyIndex;
					if(rigidBodies[candidateIndex] ==
						lowLevelBody)
					{
						globalBodyIndex = candidateIndex;
						break;
					}
				}
				if(globalBodyIndex == PX_MAX_U32 ||
					solverBodies[globalBodyIndex].isStatic())
					continue;

				const PxsBodyCore& bodyCore =
					entry.core->getCore();
				const PxTransform shapeToWorld(
					capsule.center, capsule.rotation);
				capsule.targetKind =
					Dy::AvbdSoftContactTargetKind::eRIGID_BODY;
				capsule.targetIndex =
					globalBodyIndex - bodyStart;
				capsule.shapeToRigidBody =
					bodyCore.body2World.getInverse() *
						shapeToWorld;
				Dy::AvbdSolverBody& solverBody =
					solverBodies[globalBodyIndex];
				solverBody.computePrediction(dt, gravity);
				const PxTransform predictedBodyToWorld(
					solverBody.predictedPosition,
					solverBody.predictedRotation);
				const PxTransform predictedShapeToWorld =
					predictedBodyToWorld *
						capsule.shapeToRigidBody;
				if(predictedShapeToWorld.isValid())
				{
					capsule.predictedCenter =
						predictedShapeToWorld.p;
					capsule.predictedRotation =
						predictedShapeToWorld.q;
					capsule.predictedPoseValid = true;
				}
				capsules.pushBack(capsule);
			}
		}

		void AvbdCpuSoftScene::compileDynamicConvexesForIsland(
			PxsRigidBody* const* rigidBodies,
			Dy::AvbdSolverBody* solverBodies,
			PxU32 bodyStart,
			PxU32 bodyCount,
			PxReal dt,
			const PxVec3& gravity,
			PxArray<Dy::AvbdRigidConvex>& convexes)
		{
			convexes.clear();
			for(PxU32 shapeIndex = 0;
				shapeIndex < mDynamicShapes.size(); ++shapeIndex)
			{
				const DynamicShapeEntry& entry =
					mDynamicShapes[shapeIndex];
				BodySim* bodySim = entry.core->getSim();
				if(!bodySim || bodySim->isKinematic() ||
					bodySim->isArticulationLink())
					continue;
				Dy::AvbdRigidConvex convex;
				if(!compileDynamicConvex(entry, convex))
					continue;
				const PxsRigidBody* lowLevelBody =
					&bodySim->getLowLevelBody();
				PxU32 globalBodyIndex = PX_MAX_U32;
				for(PxU32 localBodyIndex = 0;
					localBodyIndex < bodyCount;
					++localBodyIndex)
				{
					const PxU32 candidateIndex =
						bodyStart + localBodyIndex;
					if(rigidBodies[candidateIndex] ==
						lowLevelBody)
					{
						globalBodyIndex = candidateIndex;
						break;
					}
				}
				if(globalBodyIndex == PX_MAX_U32 ||
					solverBodies[globalBodyIndex].isStatic())
					continue;
				const PxsBodyCore& bodyCore =
					entry.core->getCore();
				const PxTransform shapeToWorld(
					convex.center, convex.rotation);
				convex.targetKind =
					Dy::AvbdSoftContactTargetKind::
						eRIGID_BODY;
				convex.targetIndex =
					globalBodyIndex - bodyStart;
				convex.shapeToRigidBody =
					bodyCore.body2World.getInverse() *
						shapeToWorld;
				Dy::AvbdSolverBody& solverBody =
					solverBodies[globalBodyIndex];
				solverBody.computePrediction(dt, gravity);
				const PxTransform predictedBodyToWorld(
					solverBody.predictedPosition,
					solverBody.predictedRotation);
				const PxTransform predictedShapeToWorld =
					predictedBodyToWorld *
						convex.shapeToRigidBody;
				if(predictedShapeToWorld.isValid())
				{
					convex.predictedCenter =
						predictedShapeToWorld.p;
					convex.predictedRotation =
						predictedShapeToWorld.q;
					convex.predictedPoseValid = true;
				}
				convexes.pushBack(convex);
			}
		}

		bool AvbdCpuSoftScene::readTetrahedronIndices(
			const PxTetrahedronMesh& mesh, PxArray<PxU32>& indices)
		{
			const PxU32 indexCount = mesh.getNbTetrahedrons() * 4;
			indices.resize(indexCount);
			const bool has16BitIndices =
				mesh.getTetrahedronMeshFlags() &
					PxTetrahedronMeshFlag::e16_BIT_INDICES;
			if(has16BitIndices)
			{
				const PxU16* source = static_cast<const PxU16*>(
					mesh.getTetrahedrons());
				if(!source && indexCount)
					return false;
				for(PxU32 i = 0; i < indexCount; ++i)
					indices[i] = source[i];
			}
			else
			{
				const PxU32* source = static_cast<const PxU32*>(
					mesh.getTetrahedrons());
				if(!source && indexCount)
					return false;
				for(PxU32 i = 0; i < indexCount; ++i)
					indices[i] = source[i];
			}
			for(PxU32 i = 0; i < indexCount; ++i)
				if(indices[i] >= mesh.getNbVertices())
					return false;
			return true;
		}

		bool AvbdCpuSoftScene::readTriangleIndices(
			const PxTriangleMesh& mesh, PxArray<PxU32>& indices)
		{
			const PxU32 indexCount = mesh.getNbTriangles() * 3;
			indices.resize(indexCount);
			const bool has16BitIndices =
				mesh.getTriangleMeshFlags() &
					PxTriangleMeshFlag::e16_BIT_INDICES;
			if(has16BitIndices)
			{
				const PxU16* source = static_cast<const PxU16*>(
					mesh.getTriangles());
				if(!source && indexCount)
					return false;
				for(PxU32 i = 0; i < indexCount; ++i)
					indices[i] = source[i];
			}
			else
			{
				const PxU32* source = static_cast<const PxU32*>(
					mesh.getTriangles());
				if(!source && indexCount)
					return false;
				for(PxU32 i = 0; i < indexCount; ++i)
					indices[i] = source[i];
			}
			for(PxU32 i = 0; i < indexCount; ++i)
				if(indices[i] >= mesh.getNbVertices())
					return false;
			return true;
		}

		void AvbdCpuSoftScene::reportInvalidCollisionEmbedding(const char* reason)
		{
			PxGetFoundation().error(
				PxErrorCode::eINVALID_PARAMETER, PX_FL, reason);
		}

		bool AvbdCpuSoftScene::validateVolumeCollisionEmbedding(
			PxTetrahedronMesh& simulationMesh,
			PxTetrahedronMesh& collisionMesh,
			PxDeformableVolumeAuxData& publicAuxData)
		{
			PxArray<PxU32> simulationTets;
			PxArray<PxU32> collisionTets;
			if(!readTetrahedronIndices(simulationMesh, simulationTets) ||
				!readTetrahedronIndices(collisionMesh, collisionTets))
			{
				reportInvalidCollisionEmbedding(
					"CPU AVBD deformable volume has invalid tetrahedron indices.");
				return false;
			}
			const PxU32 collisionVertexCount = collisionMesh.getNbVertices();
			if(&collisionMesh == &simulationMesh)
				return collisionVertexCount == simulationMesh.getNbVertices();

			Gu::DeformableVolumeAuxData& auxData =
				static_cast<Gu::DeformableVolumeAuxData&>(publicAuxData);
			const PxU32* remap = auxData.mVertsRemapInGridModel;
			const PxReal* barycentrics =
				auxData.mVertsBarycentricInGridModel;
			if(!remap || !barycentrics)
			{
				reportInvalidCollisionEmbedding(
					"CPU AVBD requires cooked collision-to-simulation vertex embedding for distinct meshes.");
				return false;
			}

			const PxVec3* simulationVertices = simulationMesh.getVertices();
			const PxVec3* collisionVertices = collisionMesh.getVertices();
			if(!simulationVertices || !collisionVertices)
				return false;
			PxBounds3 collisionBounds = PxBounds3::empty();
			for(PxU32 vertexIndex = 0; vertexIndex < collisionVertexCount;
				++vertexIndex)
				collisionBounds.include(collisionVertices[vertexIndex]);
			const PxReal objectScale = PxMax(
				collisionBounds.getDimensions().magnitude(), 1.0f);
			const PxReal restTolerance = 1.0e-4f * objectScale;
			for(PxU32 vertexIndex = 0; vertexIndex < collisionVertexCount;
				++vertexIndex)
			{
				const PxU32 tetIndex = remap[vertexIndex];
				if(tetIndex >= simulationMesh.getNbTetrahedrons())
				{
					reportInvalidCollisionEmbedding(
						"CPU AVBD collision embedding references an invalid simulation tetrahedron.");
					return false;
				}
				PxVec3 embeddedRest(0.0f);
				PxReal weightSum = 0.0f;
				for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
				{
					const PxReal weight = barycentrics[4 * vertexIndex + endpoint];
					if(!PxIsFinite(weight))
					{
						reportInvalidCollisionEmbedding(
							"CPU AVBD collision embedding contains a non-finite barycentric weight.");
						return false;
					}
					const PxU32 localVertex = simulationTets[4 * tetIndex + endpoint];
					embeddedRest += simulationVertices[localVertex] * weight;
					weightSum += weight;
				}
				if(!embeddedRest.isFinite() || !PxIsFinite(weightSum) ||
					PxAbs(weightSum - 1.0f) > 1.0e-3f ||
					(embeddedRest - collisionVertices[vertexIndex]).magnitude() >
						restTolerance)
				{
					reportInvalidCollisionEmbedding(
						"CPU AVBD collision embedding fails the cooked rest-position invariant.");
					return false;
				}
			}
			return true;
		}

		PxVec3 AvbdCpuSoftScene::evaluateWeightedParticlePosition(
			const Dy::AvbdWeightedContactPoint& point,
			const Dy::AvbdSoftParticle* particles, PxU32 particleCount,
			PxU32 source)
		{
			PxVec3 value(0.0f);
			for(PxU32 i = 0; i < point.count; ++i)
			{
				const PxU32 particleIndex = point.particleIndices[i];
				if(particleIndex >= particleCount)
					return PxVec3(PX_MAX_F32);
				const Dy::AvbdSoftParticle& particle = particles[particleIndex];
				const PxVec3 sample = source == 0 ? particle.position :
					source == 1 ? particle.predictedPosition :
					source == 2 ? particle.initialPosition :
					particle.outerPosition;
				value += sample * point.weights[i];
			}
			return value;
		}
}
}
