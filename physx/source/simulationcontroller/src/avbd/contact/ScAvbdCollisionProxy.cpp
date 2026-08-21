// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/scene/ScAvbdCpuSoftScene.h"
#include "avbd/ogc/DyAvbdOgcPairState.h"

namespace physx
{
namespace Sc
{

bool AvbdCpuSoftScene::refreshCollisionDetectionScene(
	const Dy::AvbdSoftParticle* sourceParticles,
	PxU32 sourceParticleCount)
{
	if(mCollisionProxy.collisionParticles.size() != mCollisionProxy.collisionVertexMappings.size() ||
		mCollisionProxy.collisionBodies.size() != mBodies.size())
		return false;
	for(PxU32 vertexIndex = 0;
		vertexIndex < mCollisionProxy.collisionParticles.size(); ++vertexIndex)
	{
		const Dy::AvbdWeightedContactPoint& mapping =
			mCollisionProxy.collisionVertexMappings[vertexIndex];
		Dy::AvbdSoftParticle& destination =
			mCollisionProxy.collisionParticles[vertexIndex];
		destination.position = evaluateWeightedParticlePosition(
			mapping, sourceParticles, sourceParticleCount, 0);
		destination.predictedPosition = evaluateWeightedParticlePosition(
			mapping, sourceParticles, sourceParticleCount, 1);
		destination.initialPosition = evaluateWeightedParticlePosition(
			mapping, sourceParticles, sourceParticleCount, 2);
		destination.outerPosition = evaluateWeightedParticlePosition(
			mapping, sourceParticles, sourceParticleCount, 3);
		destination.velocity = PxVec3(0.0f);
		destination.prevVelocity = PxVec3(0.0f);
		bool dynamic = false;
		for(PxU32 endpoint = 0; endpoint < mapping.count; ++endpoint)
		{
			const PxU32 sourceIndex = mapping.particleIndices[endpoint];
			if(sourceIndex >= sourceParticleCount)
				return false;
			destination.velocity += sourceParticles[sourceIndex].velocity *
				mapping.weights[endpoint];
			destination.prevVelocity +=
				sourceParticles[sourceIndex].prevVelocity *
				mapping.weights[endpoint];
			dynamic = dynamic || sourceParticles[sourceIndex].invMass > 0.0f;
		}
		destination.invMass = dynamic ? 1.0f : 0.0f;
		destination.mass = dynamic ? 1.0f : 0.0f;
		if(!destination.position.isFinite() ||
			!destination.predictedPosition.isFinite() ||
			!destination.initialPosition.isFinite() ||
			!destination.velocity.isFinite())
			return false;
	}
	for(PxU32 bodyIndex = 0; bodyIndex < mBodies.size(); ++bodyIndex)
	{
		mCollisionProxy.collisionBodies[bodyIndex].compiled.speculativeCCDEnabled =
			mBodies[bodyIndex].compiled.speculativeCCDEnabled;
		mCollisionProxy.collisionBodies[bodyIndex].compiled.maxDepenetrationVelocity =
			mBodies[bodyIndex].compiled.maxDepenetrationVelocity;
		mCollisionProxy.collisionBodies[bodyIndex].compiled.selfCollisionStressTolerance =
			mBodies[bodyIndex].compiled.selfCollisionStressTolerance;
	}
	// Keep the public collision buffer synchronized with the exact same
	// evaluated positions consumed by this detection epoch.
	if(sourceParticles == mParticles.begin() &&
		sourceParticleCount == mParticles.size())
	{
		for(PxU32 entryIndex = 0; entryIndex < mEntries.size();
			++entryIndex)
		{
			Entry& entry = mEntries[entryIndex];
			if(entry.kind != eVOLUME ||
				entry.bodyIndex >= mCollisionProxy.collisionBodies.size())
				continue;
			Dy::DeformableVolumeCore& core =
				entry.volumeCore->getCore();
			const Dy::AvbdSoftBodyCompiledData& collisionCompiled =
				mCollisionProxy.collisionBodies[entry.bodyIndex].compiled;
			for(PxU32 localVertex = 0;
				localVertex < collisionCompiled.particleCount;
				++localVertex)
			{
				const PxReal invMass = core.positionInvMass[localVertex].w;
				core.positionInvMass[localVertex] = PxVec4(
					mCollisionProxy.collisionParticles[
						collisionCompiled.particleStart + localVertex].position,
					invMass);
			}
		}
	}
	return true;
}

bool AvbdCpuSoftScene::rebuildSubsetCollisionDetectionScene(
	const Dy::AvbdSoftParticle* sourceParticles,
	PxU32 sourceParticleCount,
	const Dy::AvbdSoftBody* sourceBodies, PxU32 sourceBodyCount,
	ActorCore* const* softCores)
{
	if(!sourceParticles || !sourceBodies || !softCores ||
		sourceBodyCount == 0 ||
		mCollisionProxy.collisionBodies.size() != mBodies.size())
		return false;
	mCollisionProxy.subsetParticles.clear();
	mCollisionProxy.subsetBodies.clear();
	mCollisionProxy.subsetVertexMappings.clear();
	mCollisionProxy.subsetSelfCollisionAdjacencies.clear();

	for(PxU32 localBodyIndex = 0; localBodyIndex < sourceBodyCount;
		++localBodyIndex)
	{
		Entry* entry = softCores[localBodyIndex]
			? findEntry(*softCores[localBodyIndex]) : NULL;
		if(!entry || entry->bodyIndex >= mCollisionProxy.collisionBodies.size())
			return false;
		const Dy::AvbdSoftBody& globalCollisionBody =
			mCollisionProxy.collisionBodies[entry->bodyIndex];
		const PxU32 oldCollisionStart =
			globalCollisionBody.compiled.particleStart;
		const PxU32 collisionVertexCount =
			globalCollisionBody.compiled.particleCount;
		if(oldCollisionStart > mCollisionProxy.collisionParticles.size() ||
			collisionVertexCount >
				mCollisionProxy.collisionParticles.size() - oldCollisionStart)
			return false;
		const PxU32 newCollisionStart =
			mCollisionProxy.subsetParticles.size();
		for(PxU32 localVertex = 0;
			localVertex < collisionVertexCount; ++localVertex)
			mCollisionProxy.subsetParticles.pushBack(
				mCollisionProxy.collisionParticles[oldCollisionStart + localVertex]);
		mCollisionProxy.subsetBodies.pushBack(globalCollisionBody);
		if(!rebaseSoftBodyParticleRangeInPlace(
			mCollisionProxy.subsetBodies.back(), oldCollisionStart,
			collisionVertexCount, newCollisionStart))
			return false;
		Dy::AvbdSoftBodyCompiledData& subsetCompiled =
			mCollisionProxy.subsetBodies.back().compiled;
		const Dy::AvbdSoftBodyCompiledData& sourceCompiled =
			sourceBodies[localBodyIndex].compiled;
		// The global collision proxy owns immutable cooked topology, but its
		// frame-varying body controls may predate the latest public host sync.
		// A selected island must consume the same controls as its authoritative
		// simulation body, just like refreshCollisionDetectionScene() does for
		// the full-scene path.
		subsetCompiled.speculativeCCDEnabled =
			sourceCompiled.speculativeCCDEnabled;
		subsetCompiled.maxDepenetrationVelocity =
			sourceCompiled.maxDepenetrationVelocity;
		subsetCompiled.selfCollisionStressTolerance =
			sourceCompiled.selfCollisionStressTolerance;

		const PxU32 sourceParticleStart =
			sourceBodies[localBodyIndex].compiled.particleStart;
		if(entry->kind == eVOLUME)
		{
			PxArray<PxU32> simulationTets;
			if(!readTetrahedronIndices(
				*entry->simulationMesh, simulationTets))
				return false;
			Gu::DeformableVolumeAuxData& auxData =
				static_cast<Gu::DeformableVolumeAuxData&>(
					*entry->auxData);
			for(PxU32 localVertex = 0;
				localVertex < collisionVertexCount; ++localVertex)
			{
				Dy::AvbdWeightedContactPoint mapping;
				if(entry->collisionMesh == entry->simulationMesh)
					mapping.setVertex(sourceParticleStart + localVertex);
				else
				{
					const PxU32 tetIndex =
						auxData.mVertsRemapInGridModel[localVertex];
					if(tetIndex >= entry->simulationMesh->getNbTetrahedrons())
						return false;
					for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
					{
						const PxU32 localParticle =
							simulationTets[4 * tetIndex + endpoint];
						if(!mapping.appendMerged(
							sourceParticleStart + localParticle,
							auxData.mVertsBarycentricInGridModel[
								4 * localVertex + endpoint]))
							return false;
					}
					mapping.removeNearZero();
				}
				mCollisionProxy.subsetVertexMappings.pushBack(mapping);
			}
		}
		else
		{
			for(PxU32 localVertex = 0;
				localVertex < collisionVertexCount; ++localVertex)
			{
				Dy::AvbdWeightedContactPoint mapping;
				mapping.setVertex(sourceParticleStart + localVertex);
				mCollisionProxy.subsetVertexMappings.pushBack(mapping);
			}
		}
	}
	if(mCollisionProxy.subsetParticles.size() !=
		mCollisionProxy.subsetVertexMappings.size())
		return false;

	for(PxU32 vertexIndex = 0;
		vertexIndex < mCollisionProxy.subsetParticles.size(); ++vertexIndex)
	{
		const Dy::AvbdWeightedContactPoint& mapping =
			mCollisionProxy.subsetVertexMappings[vertexIndex];
		Dy::AvbdSoftParticle& destination =
			mCollisionProxy.subsetParticles[vertexIndex];
		destination.position = evaluateWeightedParticlePosition(
			mapping, sourceParticles, sourceParticleCount, 0);
		destination.predictedPosition = evaluateWeightedParticlePosition(
			mapping, sourceParticles, sourceParticleCount, 1);
		destination.initialPosition = evaluateWeightedParticlePosition(
			mapping, sourceParticles, sourceParticleCount, 2);
		destination.outerPosition = evaluateWeightedParticlePosition(
			mapping, sourceParticles, sourceParticleCount, 3);
		destination.velocity = PxVec3(0.0f);
		destination.prevVelocity = PxVec3(0.0f);
		bool dynamic = false;
		for(PxU32 endpoint = 0; endpoint < mapping.count; ++endpoint)
		{
			const PxU32 sourceIndex = mapping.particleIndices[endpoint];
			if(sourceIndex >= sourceParticleCount)
				return false;
			destination.velocity += sourceParticles[sourceIndex].velocity *
				mapping.weights[endpoint];
			destination.prevVelocity +=
				sourceParticles[sourceIndex].prevVelocity *
				mapping.weights[endpoint];
			dynamic = dynamic || sourceParticles[sourceIndex].invMass > 0.0f;
		}
		destination.invMass = dynamic ? 1.0f : 0.0f;
		destination.mass = dynamic ? 1.0f : 0.0f;
	}
	Dy::avbdBuildAllSelfCollisionAdjacencies(
		mCollisionProxy.subsetBodies.begin(),
		mCollisionProxy.subsetBodies.size(),
		mCollisionProxy.subsetSelfCollisionAdjacencies);
	return mCollisionProxy.subsetBodies.size() == sourceBodyCount;
}

bool AvbdCpuSoftScene::rebuildCollisionDetectionScene()
{
	mCollisionProxy.collisionParticles.clear();
	mCollisionProxy.collisionBodies.clear();
	mCollisionProxy.collisionVertexMappings.clear();
	mCollisionProxy.collisionSelfCollisionAdjacencies.clear();
	for(PxU32 bodyIndex = 0; bodyIndex < mBodies.size(); ++bodyIndex)
	{
		Entry* entry = NULL;
		for(PxU32 entryIndex = 0; entryIndex < mEntries.size();
			++entryIndex)
			if(mEntries[entryIndex].bodyIndex == bodyIndex)
			{
				entry = &mEntries[entryIndex];
				break;
			}
		if(!entry)
			return false;

		PxArray<PxVec3> vertices;
		PxArray<PxU32> elements;
		const PxU32 sourceParticleStart =
			mBodies[bodyIndex].compiled.particleStart;
		if(entry->kind == eVOLUME)
		{
			if(!validateVolumeCollisionEmbedding(
				*entry->simulationMesh, *entry->collisionMesh,
				*entry->auxData) ||
				!readTetrahedronIndices(*entry->collisionMesh, elements))
				return false;
			const PxU32 vertexCount = entry->collisionMesh->getNbVertices();
			vertices.resize(vertexCount);
			PxArray<PxU32> simulationTets;
			if(!readTetrahedronIndices(*entry->simulationMesh,
				simulationTets))
				return false;
			Gu::DeformableVolumeAuxData& auxData =
				static_cast<Gu::DeformableVolumeAuxData&>(
					*entry->auxData);
			for(PxU32 vertexIndex = 0; vertexIndex < vertexCount;
				++vertexIndex)
			{
				Dy::AvbdWeightedContactPoint mapping;
				if(entry->collisionMesh == entry->simulationMesh)
					mapping.setVertex(sourceParticleStart + vertexIndex);
				else
				{
					const PxU32 tetIndex =
						auxData.mVertsRemapInGridModel[vertexIndex];
					for(PxU32 endpoint = 0; endpoint < 4; ++endpoint)
					{
						const PxU32 localParticle =
							simulationTets[4 * tetIndex + endpoint];
						if(!mapping.appendMerged(
							sourceParticleStart + localParticle,
							auxData.mVertsBarycentricInGridModel[
								4 * vertexIndex + endpoint]))
							return false;
					}
					mapping.removeNearZero();
				}
				if(mapping.count == 0)
					return false;
				vertices[vertexIndex] = evaluateWeightedParticlePosition(
					mapping, mParticles.begin(), mParticles.size(), 0);
				mCollisionProxy.collisionVertexMappings.pushBack(mapping);
			}
			Dy::avbdCreateSoftBody(
				vertices.begin(), vertices.size(),
				elements.begin(), elements.size(), NULL, 0,
				1.0f, 0.3f, 1.0f, 0.0f, 0.0f, 0.01f,
				mCollisionProxy.collisionParticles, mCollisionProxy.collisionBodies, false,
				mBodies[bodyIndex].compiled.selfCollisionFilterDistance,
				mBodies[bodyIndex].material.dynamicFriction, false);
		}
		else
		{
			if(!readTriangleIndices(*entry->triangleMesh, elements))
				return false;
			const PxU32 vertexCount = entry->triangleMesh->getNbVertices();
			vertices.resize(vertexCount);
			for(PxU32 vertexIndex = 0; vertexIndex < vertexCount;
				++vertexIndex)
			{
				Dy::AvbdWeightedContactPoint mapping;
				mapping.setVertex(sourceParticleStart + vertexIndex);
				vertices[vertexIndex] =
					mParticles[sourceParticleStart + vertexIndex].position;
				mCollisionProxy.collisionVertexMappings.pushBack(mapping);
			}
			Dy::avbdCreateSoftBody(
				vertices.begin(), vertices.size(), NULL, 0,
				elements.begin(), elements.size(),
				1.0f, 0.3f, 1.0f, 0.0f, 0.0f, 0.01f,
				mCollisionProxy.collisionParticles, mCollisionProxy.collisionBodies, false,
				mBodies[bodyIndex].compiled.selfCollisionFilterDistance,
				mBodies[bodyIndex].material.dynamicFriction);
		}
		if(mCollisionProxy.collisionBodies.size() != bodyIndex + 1)
			return false;
		Dy::AvbdSoftBodyCompiledData& collisionCompiled =
			mCollisionProxy.collisionBodies[bodyIndex].compiled;
		const Dy::AvbdSoftBodyCompiledData& sourceCompiled =
			mBodies[bodyIndex].compiled;
		collisionCompiled.maxDepenetrationVelocity =
			sourceCompiled.maxDepenetrationVelocity;
		collisionCompiled.selfCollisionStressTolerance =
			sourceCompiled.selfCollisionStressTolerance;
		collisionCompiled.speculativeCCDEnabled =
			sourceCompiled.speculativeCCDEnabled;
	}
	if(mCollisionProxy.collisionParticles.size() != mCollisionProxy.collisionVertexMappings.size())
		return false;
	Dy::avbdBuildAllSelfCollisionAdjacencies(
		mCollisionProxy.collisionBodies.begin(), mCollisionProxy.collisionBodies.size(),
		mCollisionProxy.collisionSelfCollisionAdjacencies);
	return mCollisionProxy.collisionBodies.size() == mBodies.size();
}
void AvbdCpuSoftScene::detectContacts(
			Dy::AvbdSoftParticle* particles,
			PxU32 numParticles,
			Dy::AvbdSoftBody* bodies,
			PxU32 numBodies,
			PxArray<Dy::AvbdSoftContact>& contacts,
			const Dy::AvbdRigidBox* rigidBoxes,
			PxU32 numRigidBoxes,
			const Dy::AvbdSelfCollisionAdjacency*
				selfCollisionAdjacencies,
			PxU32 numSelfCollisionAdjacencies,
			const PxU8* selfCollisionEnabled,
			ActorCore* const* softCores,
			const Dy::AvbdRigidSphere* rigidSpheres,
			PxU32 numRigidSpheres,
			const Dy::AvbdRigidCapsule* rigidCapsules,
			PxU32 numRigidCapsules,
			const Dy::AvbdRigidConvex* rigidConvexes,
			PxU32 numRigidConvexes,
			const Dy::AvbdRigidTriangleSurface*
				rigidTriangleSurfaces,
			PxU32 numRigidTriangleSurfaces,
			Dy::AvbdOgcGeometryEpochSidecar* ogcGeometrySidecar){
			const bool canonicalComponentRequest =
				particles == mParticles.begin() && bodies == mBodies.begin();
			if(!ogcGeometrySidecar && canonicalComponentRequest)
				ogcGeometrySidecar = &mWorkspace.componentOgcGeometrySidecar;
			if(ogcGeometrySidecar)
				ogcGeometrySidecar->clear();
			if(canonicalComponentRequest)
			{
				mWorkspace.componentOgcPairStates.clear();
				mWorkspace.componentOgcPairIndices.clear();
			}
			if(!rigidBoxes)
			{
				rigidBoxes = mRigidBoxes.begin();
				numRigidBoxes = mRigidBoxes.size();
			}
			if(!rigidSpheres)
			{
				rigidSpheres = mRigidSpheres.begin();
				numRigidSpheres = mRigidSpheres.size();
			}
			if(!rigidCapsules)
			{
				rigidCapsules = mRigidCapsules.begin();
				numRigidCapsules = mRigidCapsules.size();
			}
			if(!rigidConvexes)
			{
				rigidConvexes = mRigidConvexes.begin();
				numRigidConvexes = mRigidConvexes.size();
			}
			if(!rigidTriangleSurfaces)
			{
				rigidTriangleSurfaces =
					mRigidTriangleSurfaces.begin();
				numRigidTriangleSurfaces =
					mRigidTriangleSurfaces.size();
			}
			if(!selfCollisionAdjacencies &&
				bodies == mBodies.begin() &&
				numBodies == mBodies.size())
			{
				PX_ASSERT(
					mSelfCollisionAdjacencies.size() == mBodies.size());
				refreshSelfCollisionEnabled();
				selfCollisionAdjacencies =
					mSelfCollisionAdjacencies.begin();
				numSelfCollisionAdjacencies =
					mSelfCollisionAdjacencies.size();
				selfCollisionEnabled = mSelfCollisionEnabled.begin();
			}
			const bool fullSceneCollisionRequest =
				particles == mParticles.begin() &&
				numParticles == mParticles.size() &&
				bodies == mBodies.begin() &&
				numBodies == mBodies.size();
			const auto publishComponentGeometryEpoch = [&]()
			{
				if(!fullSceneCollisionRequest)
					return true;
				if(!ogcGeometrySidecar ||
					ogcGeometrySidecar->contactTriangleCoreIndices.size() !=
						contacts.size())
					return false;
				return Dy::compileOgcPairProviderPlan(
					contacts.begin(), contacts.size(),
					rigidBoxes, numRigidBoxes, numBodies, 0u,
					Dy::eOGC_PAIR_PROVIDER_WORLD_STATIC |
						Dy::eOGC_PAIR_PROVIDER_DEFORMABLE,
					mWorkspace.componentOgcPairStates,
					mWorkspace.componentOgcPairIndices);
			};
			const auto finishComponentGeometryEpoch = [&]()
			{
				if(publishComponentGeometryEpoch())
					return true;
				contacts.clear();
				if(ogcGeometrySidecar)
					ogcGeometrySidecar->clear();
				mWorkspace.componentOgcPairStates.clear();
				mWorkspace.componentOgcPairIndices.clear();
				reportInvalidCollisionEmbedding(
					"CPU AVBD failed to publish the component OGC geometry epoch.");
				return false;
			};
			const auto filterDetectedContacts = [&](const Dy::AvbdSoftBody* filterBodies,
				PxU32 filterBodyCount, ActorCore* const* filterCores)
			{
				if(removeRigidActorFilteredContacts(
						filterBodies, filterBodyCount, filterCores,
						contacts, ogcGeometrySidecar) &&
					removeDeformablePairFilteredContacts(
						filterBodies, filterBodyCount, filterCores,
						contacts, ogcGeometrySidecar))
					return true;
				contacts.clear();
				if(ogcGeometrySidecar)
					ogcGeometrySidecar->clear();
				reportInvalidCollisionEmbedding(
					"CPU AVBD failed to compact the filtered OGC geometry epoch.");
				return false;
			};
			Dy::AvbdSoftContactDetectionView detectionView;
			configureAvbdContactDetectionView(
				detectionView,
				rigidBoxes, numRigidBoxes,
				rigidSpheres, numRigidSpheres,
				rigidCapsules, numRigidCapsules,
				rigidConvexes, numRigidConvexes,
				rigidTriangleSurfaces, numRigidTriangleSurfaces,
				mWorldPlanes.begin(), mWorldPlanes.size(),
				selfCollisionEnabled);
			if(fullSceneCollisionRequest)
			{
				bool exactSimulationCollisionDomain = true;
				for(PxU32 entryIndex = 0;
					entryIndex < mEntries.size(); ++entryIndex)
				{
					const Entry& entry = mEntries[entryIndex];
					if(entry.kind == eVOLUME &&
						entry.collisionMesh != entry.simulationMesh)
					{
						exactSimulationCollisionDomain = false;
						break;
					}
				}
				if(exactSimulationCollisionDomain)
				{
					// A shared mesh object is already the authoritative public
					// collision domain. Keeping this identity case direct makes
					// serial redetection and P5's range leaves consume the same
					// particles, topology and contact-state metadata. Distinct
					// cooked meshes remain on the proxy/embedding path below.
					detectionView.particles = particles;
					detectionView.numParticles = numParticles;
					detectionView.softBodies = bodies;
					detectionView.numSoftBodies = numBodies;
					detectionView.selfCollisionAdjacencies =
						selfCollisionAdjacencies;
					detectionView.numSelfCollisionAdjacencies =
						numSelfCollisionAdjacencies;
					Dy::avbdDetectAllOGCContacts(
						detectionView, contacts, mContactParams, 0.0f,
						mCollisionStatsEnabled ? &mLastCollisionStats : NULL,
						&mWorkspace.contact, ogcGeometrySidecar);
					if(!filterDetectedContacts(bodies, numBodies, softCores))
						return;
					finishComponentGeometryEpoch();
					return;
				}
				// Public CPU AVBD collision is defined on the cooked collision
				// domain.  Never reinterpret the simulation bodies as collision
				// geometry when the proxy scene is incomplete: doing so changes the
				// contact surface, feature identities and friction anchors mid-step.
				if(mCollisionProxy.collisionBodies.size() != mBodies.size())
				{
					reportInvalidCollisionEmbedding(
						"CPU AVBD collision-domain proxy scene is incomplete before contact detection.");
					contacts.clear();
					return;
				}
				if(!refreshCollisionDetectionScene(particles, numParticles))
				{
					reportInvalidCollisionEmbedding(
						"CPU AVBD collision embedding became invalid before contact detection.");
					contacts.clear();
					return;
				}
				refreshSelfCollisionEnabled();
				detectionView.particles = mCollisionProxy.collisionParticles.begin();
				detectionView.numParticles = mCollisionProxy.collisionParticles.size();
				detectionView.softBodies = mCollisionProxy.collisionBodies.begin();
				detectionView.numSoftBodies = mCollisionProxy.collisionBodies.size();
				detectionView.selfCollisionAdjacencies =
					mCollisionProxy.collisionSelfCollisionAdjacencies.begin();
				detectionView.numSelfCollisionAdjacencies =
					mCollisionProxy.collisionSelfCollisionAdjacencies.size();
				detectionView.selfCollisionEnabled =
					mSelfCollisionEnabled.begin();
				Dy::avbdDetectAllOGCContacts(
					detectionView, contacts, mContactParams, 0.0f,
					mCollisionStatsEnabled ? &mLastCollisionStats : NULL,
					&mWorkspace.contact, ogcGeometrySidecar);
				if(!filterDetectedContacts(
						mCollisionProxy.collisionBodies.begin(),
						mCollisionProxy.collisionBodies.size(), softCores))
					return;
				if(!expandCollisionDetectionContacts(
					contacts, numParticles, mCollisionProxy.collisionVertexMappings,
					mCollisionProxy.collisionBodies, ogcGeometrySidecar))
				{
					reportInvalidCollisionEmbedding(
						"CPU AVBD failed to expand a collision-domain contact into simulation particles.");
					contacts.clear();
					return;
				}
				finishComponentGeometryEpoch();
				return;
			}
			const bool useSubsetCookedCollisionDomain =
				softCores && particles && bodies && numBodies > 0;
			if(useSubsetCookedCollisionDomain)
			{
				if(!rebuildSubsetCollisionDetectionScene(
						particles, numParticles, bodies, numBodies, softCores))
				{
					// An island selected from public actors must retain the same
					// collision domain as the full Scene.  Falling through to the
					// simulation bodies would make OGC collide against the voxel FEM
					// boundary and can leave persistent friction state attached to a
					// geometrically unrelated feature.
					reportInvalidCollisionEmbedding(
						"CPU AVBD failed to build the collision-domain proxy for a public actor subset.");
					contacts.clear();
					return;
				}
				detectionView.particles = mCollisionProxy.subsetParticles.begin();
				detectionView.numParticles = mCollisionProxy.subsetParticles.size();
				detectionView.softBodies = mCollisionProxy.subsetBodies.begin();
				detectionView.numSoftBodies = mCollisionProxy.subsetBodies.size();
				detectionView.selfCollisionAdjacencies =
					mCollisionProxy.subsetSelfCollisionAdjacencies.begin();
				detectionView.numSelfCollisionAdjacencies =
					mCollisionProxy.subsetSelfCollisionAdjacencies.size();
				Dy::avbdDetectAllOGCContacts(
					detectionView, contacts, mContactParams, 0.0f,
					mCollisionStatsEnabled ? &mLastCollisionStats : NULL,
					&mWorkspace.contact, ogcGeometrySidecar);
				if(!filterDetectedContacts(
						mCollisionProxy.subsetBodies.begin(),
						mCollisionProxy.subsetBodies.size(), softCores))
					return;
				if(!expandCollisionDetectionContacts(
					contacts, numParticles,
					mCollisionProxy.subsetVertexMappings,
					mCollisionProxy.subsetBodies, ogcGeometrySidecar))
				{
					contacts.clear();
					return;
				}
				return;
			}
			// This direct-domain path is reserved for legacy low-level callers
			// that do not represent public Scene actors.  Public full-Scene and
			// subset requests have both returned above and therefore cannot
			// silently fall back from their collision proxy to simulation tets.
			detectionView.particles = particles;
			detectionView.numParticles = numParticles;
			detectionView.softBodies = bodies;
			detectionView.numSoftBodies = numBodies;
			detectionView.selfCollisionAdjacencies =
				selfCollisionAdjacencies;
			detectionView.numSelfCollisionAdjacencies =
				numSelfCollisionAdjacencies;
			Dy::avbdDetectAllOGCContacts(
				detectionView, contacts, mContactParams, 0.0f,
				mCollisionStatsEnabled ? &mLastCollisionStats : NULL,
				&mWorkspace.contact, ogcGeometrySidecar);
	if(!filterDetectedContacts(bodies, numBodies, softCores))
		return;
	if(ogcGeometrySidecar &&
		ogcGeometrySidecar->contactTriangleCoreIndices.size() !=
			contacts.size())
	{
		contacts.clear();
		ogcGeometrySidecar->clear();
	}
}

void AvbdCpuSoftScene::redetectContacts(
	Dy::AvbdSoftParticle* particles,
	PxU32 numParticles,
	Dy::AvbdSoftBody* bodies,
	PxU32 numBodies,
	PxArray<Dy::AvbdSoftContact>& contacts,
	void* userData)
{
	static_cast<AvbdCpuSoftScene*>(userData)->detectContacts(
		particles, numParticles, bodies, numBodies, contacts);
}

bool AvbdCpuSoftScene::publishComponentOgcGeometryEpoch()
{
	if(mWorkspace.componentOgcGeometrySidecar.
			contactTriangleCoreIndices.size() != mContacts.size())
		return false;
	return Dy::compileOgcPairProviderPlan(
		mContacts.begin(), mContacts.size(),
		mRigidBoxes.begin(), mRigidBoxes.size(),
		mBodies.size(), 0u,
		Dy::eOGC_PAIR_PROVIDER_WORLD_STATIC |
			Dy::eOGC_PAIR_PROVIDER_DEFORMABLE,
		mWorkspace.componentOgcPairStates,
		mWorkspace.componentOgcPairIndices);
}

void AvbdCpuSoftScene::beginComponentContactRedetection()
{
	mWorkspace.componentOgcGeometrySidecar.clear();
	mWorkspace.componentOgcPairStates.clear();
	mWorkspace.componentOgcPairIndices.clear();
	Dy::avbdBeginSoftContactRedetection(
		mContacts, mWorkspace.contact,
		mCollisionStatsEnabled ? &mLastCollisionStats : NULL);
}

void AvbdCpuSoftScene::completeComponentContactRedetection()
{
	Dy::avbdCompleteSoftContactRedetection(
		mParticles.begin(), mContacts, mWorkspace.contact);
	if(!mWorkspace.componentOgcGeometrySidecar.resizeContactMapping(
			mContacts.size()))
	{
		mContacts.clear();
		mWorkspace.componentOgcGeometrySidecar.clear();
		mWorkspace.componentOgcPairStates.clear();
		mWorkspace.componentOgcPairIndices.clear();
		reportInvalidCollisionEmbedding(
			"CPU AVBD failed to complete a task-merged OGC geometry mapping.");
		return;
	}
	if(!removeRigidActorFilteredContacts(
			mBodies.begin(), mBodies.size(), NULL, mContacts,
			&mWorkspace.componentOgcGeometrySidecar) ||
		!removeDeformablePairFilteredContacts(
			mBodies.begin(), mBodies.size(), NULL, mContacts,
			&mWorkspace.componentOgcGeometrySidecar))
	{
		mContacts.clear();
		mWorkspace.componentOgcGeometrySidecar.clear();
		mWorkspace.componentOgcPairStates.clear();
		mWorkspace.componentOgcPairIndices.clear();
		reportInvalidCollisionEmbedding(
			"CPU AVBD failed to compact a task-merged OGC geometry epoch.");
		return;
	}
	if(publishComponentOgcGeometryEpoch())
		return;
	mContacts.clear();
	mWorkspace.componentOgcGeometrySidecar.clear();
	mWorkspace.componentOgcPairStates.clear();
	mWorkspace.componentOgcPairIndices.clear();
	reportInvalidCollisionEmbedding(
		"CPU AVBD failed to publish a task-merged component OGC geometry epoch.");
}

void AvbdCpuSoftScene::refreshSelfCollisionEnabled()
{
	mSelfCollisionEnabled.resize(mBodies.size());
	for(PxU32 i = 0; i < mSelfCollisionEnabled.size(); ++i)
		mSelfCollisionEnabled[i] = 0;
	for(PxU32 i = 0; i < mEntries.size(); ++i)
	{
		const Entry& entry = mEntries[i];
		if(entry.bodyIndex < mSelfCollisionEnabled.size())
			mSelfCollisionEnabled[entry.bodyIndex] =
				(entry.getBodyCore().bodyFlags &
					PxDeformableBodyFlag::eDISABLE_SELF_COLLISION)
				? 0u : 1u;
	}
}


bool AvbdCpuSoftScene::expandCollisionDetectionPoint(
	const PxU32* proxyIndices, const PxReal* proxyWeights,
	PxU32 proxyCount,
	const PxArray<Dy::AvbdWeightedContactPoint>& vertexMappings,
	Dy::AvbdWeightedContactPoint& output) const
{
	output.clear();
	for(PxU32 proxyOrder = 0; proxyOrder < proxyCount; ++proxyOrder)
	{
		const PxU32 proxyIndex = proxyIndices[proxyOrder];
		if(proxyIndex >= vertexMappings.size() ||
			!PxIsFinite(proxyWeights[proxyOrder]))
			return false;
		const Dy::AvbdWeightedContactPoint& vertexMapping =
			vertexMappings[proxyIndex];
		for(PxU32 endpoint = 0; endpoint < vertexMapping.count; ++endpoint)
			if(!output.appendMerged(
				vertexMapping.particleIndices[endpoint],
				proxyWeights[proxyOrder] * vertexMapping.weights[endpoint]))
				return false;
	}
	output.removeNearZero();
	return output.count != 0;
}

PxU32 AvbdCpuSoftScene::findCollisionBodyForParticle(
	PxU32 particleIndex,
	const PxArray<Dy::AvbdSoftBody>& collisionBodies) const
{
	for(PxU32 bodyIndex = 0; bodyIndex < collisionBodies.size(); ++bodyIndex)
	{
		const Dy::AvbdSoftBodyCompiledData& compiled =
			collisionBodies[bodyIndex].compiled;
		if(particleIndex >= compiled.particleStart &&
			particleIndex - compiled.particleStart < compiled.particleCount)
			return bodyIndex;
	}
	return PX_MAX_U32;
}

PxU32 AvbdCpuSoftScene::resolveCollisionElementForFeature(
	const Dy::AvbdSoftContactGeometry& geometry,
	const Dy::AvbdSoftBodyCompiledData& compiled,
	PxU32 collisionFeatureParticle) const
{
	PxU32 featureVertices[3] =
	{
		collisionFeatureParticle, PX_MAX_U32, PX_MAX_U32
	};
	PxU32 featureVertexCount = 1;
	if(geometry.hasBarycentricQueryPoint())
	{
		featureVertexCount = 0;
		while(featureVertexCount < 3 &&
			geometry.queryParticleIndices[featureVertexCount] != PX_MAX_U32)
		{
			featureVertices[featureVertexCount] =
				geometry.queryParticleIndices[featureVertexCount];
			++featureVertexCount;
		}
	}
	if(featureVertexCount == 0 ||
		compiled.surfaceTriangleElementIndices.size() !=
			compiled.surfaceTriangles.size() / 3)
		return PX_MAX_U32;

	PxU32 owner = PX_MAX_U32;
	for(PxU32 triangleIndex = 0;
		triangleIndex < compiled.surfaceTriangles.size() / 3;
		++triangleIndex)
	{
		const PxU32* triangle =
			compiled.surfaceTriangles.begin() + 3 * triangleIndex;
		bool containsFeature = true;
		for(PxU32 featureIndex = 0;
			featureIndex < featureVertexCount; ++featureIndex)
		{
			const PxU32 featureVertex = featureVertices[featureIndex];
			if(triangle[0] != featureVertex && triangle[1] != featureVertex &&
				triangle[2] != featureVertex)
			{
				containsFeature = false;
				break;
			}
		}
		if(containsFeature)
			owner = PxMin(
				owner, compiled.surfaceTriangleElementIndices[triangleIndex]);
	}
	return owner;
}

bool AvbdCpuSoftScene::expandCollisionDetectionContacts(
	PxArray<Dy::AvbdSoftContact>& contacts,
	PxU32 simulationParticleCount,
	const PxArray<Dy::AvbdWeightedContactPoint>& vertexMappings,
	const PxArray<Dy::AvbdSoftBody>& collisionBodies,
	Dy::AvbdOgcGeometryEpochSidecar* ogcGeometrySidecar) const
{
	if(!ogcGeometrySidecar)
		return false;
	if(ogcGeometrySidecar->contactTriangleCoreIndices.size() !=
		contacts.size())
		return false;
	const auto failExpansion = [&]() -> bool
	{
		ogcGeometrySidecar->clear();
		return false;
	};
	for(PxU32 contactIndex = 0; contactIndex < contacts.size(); ++contactIndex)
	{
		Dy::AvbdSoftContactGeometry& geometry = contacts[contactIndex].geometry;
		const PxU32 collisionFeatureParticle = geometry.particleIdx;
		geometry.collisionFeatureParticleIdx = collisionFeatureParticle;
		geometry.queryBodyIndex = findCollisionBodyForParticle(
			collisionFeatureParticle, collisionBodies);
		if(geometry.queryBodyIndex == PX_MAX_U32)
			return failExpansion();
		if(geometry.hasBarycentricQueryPoint())
		{
			PxU32 count = 0;
			while(count < 3 && geometry.queryParticleIndices[count] != PX_MAX_U32)
				++count;
			if(!expandCollisionDetectionPoint(
				geometry.queryParticleIndices, geometry.queryWeights,
				count, vertexMappings, geometry.queryPoint))
				return failExpansion();
			// A triangle-core OGC row needs more than its centroid once the
			// nonlinear island solver starts moving both endpoints. Preserve the
			// independent embedded supports of all three proxy vertices so that
			// the pair trust region can test the whole triangle at candidate poses.
			Dy::AvbdOgcTriangleCoreCertificate* certificate =
				ogcGeometrySidecar->getTriangleCoreMutable(contactIndex);
			if(certificate)
			{
				if(count != 3)
					return failExpansion();
				for(PxU32 vertex = 0; vertex < 3; ++vertex)
				{
					const Dy::AvbdWeightedContactPoint proxyPoint =
						certificate->points[vertex];
					Dy::AvbdWeightedContactPoint& expandedPoint =
						certificate->points[vertex];
					if(!expandCollisionDetectionPoint(
						proxyPoint.particleIndices, proxyPoint.weights,
						proxyPoint.count, vertexMappings,
						expandedPoint))
						return failExpansion();
					for(PxU32 endpoint = 0;
						endpoint < expandedPoint.count;
						++endpoint)
						if(expandedPoint.particleIndices[endpoint] >=
							simulationParticleCount)
							return failExpansion();
				}
				if(!certificate->isValid())
					return failExpansion();
			}
		}
		else
		{
			const PxU32 proxyIndices[1] = {collisionFeatureParticle};
			const PxReal proxyWeights[1] = {1.0f};
			if(!expandCollisionDetectionPoint(
				proxyIndices, proxyWeights, 1, vertexMappings,
				geometry.queryPoint))
				return failExpansion();
		}
		for(PxU32 endpoint = 0; endpoint < geometry.queryPoint.count; ++endpoint)
			if(geometry.queryPoint.particleIndices[endpoint] >= simulationParticleCount)
				return failExpansion();
		geometry.particleIdx = geometry.queryPoint.particleIndices[0];

		const Dy::AvbdSoftBodyCompiledData& queryCompiled =
			collisionBodies[geometry.queryBodyIndex].compiled;
		geometry.queryCollisionElementIndex = resolveCollisionElementForFeature(
			geometry, queryCompiled, collisionFeatureParticle);
		if(geometry.queryCollisionElementIndex == PX_MAX_U32 &&
			collisionFeatureParticle >= queryCompiled.particleStart)
		{
			const PxU32 localParticle =
				collisionFeatureParticle - queryCompiled.particleStart;
			if(localParticle < queryCompiled.elementAdjacency.size())
			{
				const PxArray<Dy::AvbdParticleElementRef>& refs =
					queryCompiled.triElements.empty()
						? queryCompiled.elementAdjacency[localParticle].tetRefs
						: queryCompiled.elementAdjacency[localParticle].triRefs;
				if(!refs.empty())
				{
					const PxU32 compiledElement = refs[0].index;
					if(queryCompiled.triElements.empty() &&
						compiledElement < queryCompiled.tetElements.size())
						geometry.queryCollisionElementIndex =
							queryCompiled.tetElements[compiledElement].sourceElementIndex;
					else if(compiledElement < queryCompiled.triElements.size())
						geometry.queryCollisionElementIndex =
							queryCompiled.triElements[compiledElement].sourceElementIndex;
				}
			}
		}

		if(geometry.hasDeformableSurfaceTarget())
		{
			PxU32 count = 0;
			while(count < 3 && geometry.surfaceParticleIndices[count] != PX_MAX_U32)
				++count;
			if(!expandCollisionDetectionPoint(
				geometry.surfaceParticleIndices, geometry.surfaceWeights,
				count, vertexMappings, geometry.targetPoint))
				return failExpansion();
			for(PxU32 endpoint = 0; endpoint < geometry.targetPoint.count; ++endpoint)
				if(geometry.targetPoint.particleIndices[endpoint] >= simulationParticleCount)
					return failExpansion();
			geometry.targetCollisionElementIndex = geometry.targetSourceElementIndex;
		}
	}
	return true;
}
// Static shape to OGC proxy compilation remains in the collision-domain implementation unit.

void AvbdCpuSoftScene::compileWorldStatics(
			const PxsMaterialManager& materialManager) {
			mWorldPlanes.clear();
			mRigidBoxes.clear();
			mRigidSpheres.clear();
			mRigidCapsules.clear();
			mRigidConvexes.clear();
			if(++mRigidTriangleSurfaceCompileStamp == 0)
			{
				mRigidTriangleSurfaceCompileStamp = 1;
				for(PxU32 surfaceIndex = 0;
					surfaceIndex < mRigidTriangleSurfaces.size();
					++surfaceIndex)
					mRigidTriangleSurfaces[surfaceIndex].
						sceneCompileStamp = 0;
			}
			const PxU32 triangleSurfaceCompileStamp =
				mRigidTriangleSurfaceCompileStamp;
			PxU32 triangleSurfaceCompileOrder = 0;
			for(PxU32 i = 0; i < mStaticShapes.size(); i++)
			{
				const StaticShapeEntry& entry = mStaticShapes[i];
				const ShapeCore& shape = *entry.shape;
				if(!(shape.getFlags() &
					PxShapeFlag::eSIMULATION_SHAPE))
					continue;
				const PxTransform shapeToWorld =
					entry.core->getActor2World() *
					shape.getShape2Actor();
				if(!shapeToWorld.isValid())
					continue;
				const PxReal friction =
					getStaticFriction(shape, materialManager);
				const PxU8 frictionCombineMode =
					getStaticFrictionCombineMode(
						shape, materialManager);
				if(shape.getGeometryType() ==
					PxGeometryType::ePLANE)
				{
					Dy::AvbdWorldPlane plane;
					plane.normal =
						shapeToWorld.q.rotate(
							PxVec3(1.0f, 0.0f, 0.0f)).
							getNormalized();
					plane.offset =
						plane.normal.dot(shapeToWorld.p);
					plane.friction = friction;
					plane.frictionCombineMode =
						frictionCombineMode;
					plane.primitiveKey = entry.primitiveKey;
					mWorldPlanes.pushBack(plane);
				}
				else if(shape.getGeometryType() ==
					PxGeometryType::eBOX)
				{
					const PxBoxGeometry& geometry =
						static_cast<const PxBoxGeometry&>(
							shape.getGeometry());
					Dy::AvbdRigidBox box;
					box.center = shapeToWorld.p;
					box.rotation = shapeToWorld.q;
					box.halfExtent = geometry.halfExtents;
					box.friction = friction;
					box.frictionCombineMode =
						frictionCombineMode;
					box.primitiveKey = entry.primitiveKey;
					mRigidBoxes.pushBack(box);
				}
				else if(shape.getGeometryType() ==
					PxGeometryType::eSPHERE)
				{
					const PxSphereGeometry& geometry =
						static_cast<const PxSphereGeometry&>(
							shape.getGeometry());
					if(geometry.radius <= 0.0f ||
						!PxIsFinite(geometry.radius))
						continue;
					Dy::AvbdRigidSphere sphere;
					sphere.center = shapeToWorld.p;
					sphere.rotation = shapeToWorld.q;
					sphere.radius = geometry.radius;
					sphere.friction = friction;
					sphere.frictionCombineMode =
						frictionCombineMode;
					sphere.primitiveKey = entry.primitiveKey;
					mRigidSpheres.pushBack(sphere);
				}
				else if(shape.getGeometryType() ==
					PxGeometryType::eCAPSULE)
				{
					const PxCapsuleGeometry& geometry =
						static_cast<const PxCapsuleGeometry&>(
							shape.getGeometry());
					if(geometry.radius <= 0.0f ||
						geometry.halfHeight < 0.0f ||
						!PxIsFinite(geometry.radius) ||
						!PxIsFinite(geometry.halfHeight))
						continue;
					Dy::AvbdRigidCapsule capsule;
					capsule.center = shapeToWorld.p;
					capsule.rotation = shapeToWorld.q;
					capsule.radius = geometry.radius;
					capsule.halfHeight = geometry.halfHeight;
					capsule.friction = friction;
					capsule.frictionCombineMode =
						frictionCombineMode;
					capsule.primitiveKey = entry.primitiveKey;
					mRigidCapsules.pushBack(capsule);
				}
				else if(shape.getGeometryType() ==
					PxGeometryType::eCONVEXMESH)
				{
					const PxConvexMeshGeometry& geometry =
						static_cast<
							const PxConvexMeshGeometry&>(
								shape.getGeometry());
					Dy::AvbdRigidConvex convex;
					if(!compileConvexTopology(
							geometry, convex))
						continue;
					convex.center = shapeToWorld.p;
					convex.rotation = shapeToWorld.q;
					convex.previousCenter = shapeToWorld.p;
					convex.previousRotation = shapeToWorld.q;
					convex.friction = friction;
					convex.frictionCombineMode =
						frictionCombineMode;
					convex.primitiveKey =
						entry.primitiveKey;
					mRigidConvexes.pushBack(convex);
				}
				else if(shape.getGeometryType() ==
						PxGeometryType::eTRIANGLEMESH ||
					shape.getGeometryType() ==
						PxGeometryType::eHEIGHTFIELD)
				{
					Dy::AvbdRigidTriangleSurface& surface =
						getRigidTriangleSurface(entry.primitiveKey);
					if(!refreshRigidTriangleSurfaceTopology(
							shape, materialManager, surface))
						continue;
					surface.center = shapeToWorld.p;
					surface.rotation = shapeToWorld.q;
					surface.previousCenter = shapeToWorld.p;
					surface.previousRotation =
						shapeToWorld.q;
					surface.primitiveKey =
						entry.primitiveKey;
					surface.targetKind =
						Dy::AvbdSoftContactTargetKind::
							eWORLD_STATIC;
					surface.targetIndex = PX_MAX_U32;
					surface.shapeToRigidBody = PxTransform(PxIdentity);
					surface.sceneCompileStamp =
						triangleSurfaceCompileStamp;
					surface.sceneCompileOrder =
						triangleSurfaceCompileOrder++;
				}
			}
			for(PxU32 i = 0; i < mDynamicShapes.size(); i++)
			{
				const DynamicShapeEntry& entry = mDynamicShapes[i];
				BodySim* bodySim = entry.core->getSim();
				if(!bodySim || !bodySim->isKinematic() ||
					bodySim->isArticulationLink())
					continue;
				Dy::AvbdRigidBox box;
				if(compileDynamicBox(entry, box))
				{
					// A prescribed kinematic is a one-way moving position
					// objective.  Its explicit prep owner keeps it out of
					// both world-static warmstart and the rigid 6x6 block.
					box.targetKind =
						Dy::AvbdSoftContactTargetKind::
							eKINEMATIC_RIGID;
					mRigidBoxes.pushBack(box);
					continue;
				}
				Dy::AvbdRigidSphere sphere;
				if(compileDynamicSphere(entry, sphere))
				{
					sphere.targetKind =
						Dy::AvbdSoftContactTargetKind::
							eKINEMATIC_RIGID;
					mRigidSpheres.pushBack(sphere);
					continue;
				}
				Dy::AvbdRigidCapsule capsule;
				if(compileDynamicCapsule(entry, capsule))
				{
					capsule.targetKind =
						Dy::AvbdSoftContactTargetKind::
							eKINEMATIC_RIGID;
					mRigidCapsules.pushBack(capsule);
					continue;
				}
				Dy::AvbdRigidConvex convex;
				if(compileDynamicConvex(entry, convex))
				{
					convex.targetKind =
						Dy::AvbdSoftContactTargetKind::
							eKINEMATIC_RIGID;
					mRigidConvexes.pushBack(convex);
					continue;
				}
				Dy::AvbdRigidTriangleSurface& surface =
					getRigidTriangleSurface(entry.primitiveKey);
				if(compileDynamicTriangleSurface(entry, surface))
				{
					surface.targetKind =
						Dy::AvbdSoftContactTargetKind::
							eKINEMATIC_RIGID;
					surface.targetIndex = PX_MAX_U32;
					surface.shapeToRigidBody = PxTransform(PxIdentity);
					surface.sceneCompileStamp =
						triangleSurfaceCompileStamp;
					surface.sceneCompileOrder =
						triangleSurfaceCompileOrder++;
				}
			}
			// Retire shapes removed from the scene (or whose geometry became
			// invalid) after the complete static-then-kinematic traversal.
			// Reorder the retained cache to that traversal order so contact
			// generation keeps the exact legacy source ordering.
			for(PxU32 surfaceIndex = mRigidTriangleSurfaces.size();
				surfaceIndex > 0; --surfaceIndex)
			{
				if(mRigidTriangleSurfaces[surfaceIndex - 1].
					sceneCompileStamp != triangleSurfaceCompileStamp)
					mRigidTriangleSurfaces.replaceWithLast(
						surfaceIndex - 1);
			}
			for(PxU32 expectedOrder = 0;
				expectedOrder < mRigidTriangleSurfaces.size();
				++expectedOrder)
			{
				for(PxU32 surfaceIndex = expectedOrder + 1;
					surfaceIndex < mRigidTriangleSurfaces.size();
					++surfaceIndex)
				{
					if(mRigidTriangleSurfaces[surfaceIndex].
						sceneCompileOrder == expectedOrder)
					{
						PxSwap(mRigidTriangleSurfaces[expectedOrder],
							mRigidTriangleSurfaces[surfaceIndex]);
						break;
					}
				}
			}
		}
// Collision-domain bounds and endpoint envelope helpers live with proxy compilation.

bool AvbdCpuSoftScene::computeSoftBounds(
			const Entry& entry, PxBounds3& bounds) const {
			const PxU32 particleStart = getParticleStart(entry);
			const PxU32 particleCount = getParticleCount(entry);
			if(particleCount == 0 ||
				particleStart > mParticles.size() ||
				particleCount >
					mParticles.size() - particleStart)
				return false;

			bounds = PxBounds3::empty();
			for(PxU32 i = 0; i < particleCount; i++)
			{
				const PxVec3& position =
					mParticles[particleStart + i].position;
				if(!position.isFinite())
					return false;
				bounds.include(position);
			}
			return !bounds.isEmpty();
		}

bool AvbdCpuSoftScene::computeCollisionDomainSoftBounds(
			const Entry& entry, PxBounds3& bounds) const {
			if(entry.bodyIndex >= mBodies.size() ||
				mCollisionProxy.collisionBodies.size() != mBodies.size() ||
				entry.bodyIndex >= mCollisionProxy.collisionBodies.size() ||
				entry.collisionMesh == entry.simulationMesh)
				return computeSoftBounds(entry, bounds);

			const Dy::AvbdSoftBodyCompiledData& collisionCompiled =
				mCollisionProxy.collisionBodies[entry.bodyIndex].compiled;
			const PxU32 collisionParticleStart =
				collisionCompiled.particleStart;
			const PxU32 collisionParticleCount =
				collisionCompiled.particleCount;
			if(collisionParticleCount == 0 ||
				collisionParticleStart > mCollisionProxy.collisionVertexMappings.size() ||
				collisionParticleCount >
					mCollisionProxy.collisionVertexMappings.size() -
						collisionParticleStart)
				return false;

			bounds = PxBounds3::empty();
			for(PxU32 localParticleIndex = 0;
				localParticleIndex < collisionParticleCount;
				++localParticleIndex)
			{
				const Dy::AvbdWeightedContactPoint& mapping =
					mCollisionProxy.collisionVertexMappings[
						collisionParticleStart + localParticleIndex];
				const PxVec3 position = evaluateWeightedParticlePosition(
					mapping, mParticles.begin(), mParticles.size(), 0);
				if(!position.isFinite())
					return false;
				bounds.include(position);
			}
			return !bounds.isEmpty();
		}

bool AvbdCpuSoftScene::expandSoftBoundsForPrediction(
			const Entry& entry, PxReal dt, const PxVec3& gravity,
			PxBounds3& bounds) const {
			if(dt <= 0.0f || !PxIsFinite(dt) ||
				!gravity.isFinite() || bounds.isEmpty())
				return false;
			const PxU32 particleStart = getParticleStart(entry);
			const PxU32 particleCount = getParticleCount(entry);
			if(particleCount == 0 ||
				particleStart > mParticles.size() ||
				particleCount > mParticles.size() - particleStart)
				return false;
			const PxReal dtSq = dt * dt;
			for(PxU32 i = 0; i < particleCount; i++)
			{
				const Dy::AvbdSoftParticle& particle =
					mParticles[particleStart + i];
				if(particle.invMass <= 0.0f)
					continue;
				const PxVec3 predictedPosition =
					particle.position + particle.velocity * dt +
					gravity * (particle.gravityScale * dtSq);
				if(!predictedPosition.isFinite())
					return false;
				bounds.include(predictedPosition);
			}
			return true;
		}

bool AvbdCpuSoftScene::computePredictedCollisionDomainSoftBounds(
			const Entry& entry, PxReal dt, const PxVec3& gravity,
			PxBounds3& bounds) const {
			if(dt <= 0.0f || !PxIsFinite(dt) ||
				!gravity.isFinite() || entry.bodyIndex >= mBodies.size())
				return false;

			const PxReal dtSq = dt * dt;
			const auto predictPosition = [&] (
				const Dy::AvbdSoftParticle& particle,
				PxVec3& position)
			{
				if(!particle.position.isFinite() ||
					!particle.velocity.isFinite() ||
					!PxIsFinite(particle.gravityScale))
					return false;
				position = particle.position;
				if(particle.invMass > 0.0f)
					position += particle.velocity * dt +
						gravity * (particle.gravityScale * dtSq);
				return position.isFinite();
			};

			bounds = PxBounds3::empty();
			if(mCollisionProxy.collisionBodies.size() != mBodies.size() ||
				entry.bodyIndex >= mCollisionProxy.collisionBodies.size() ||
				entry.collisionMesh == entry.simulationMesh)
			{
				const PxU32 particleStart = getParticleStart(entry);
				const PxU32 particleCount = getParticleCount(entry);
				if(particleCount == 0 || particleStart > mParticles.size() ||
					particleCount > mParticles.size() - particleStart)
					return false;
				for(PxU32 i = 0; i < particleCount; ++i)
				{
					PxVec3 position;
					if(!predictPosition(mParticles[particleStart + i], position))
						return false;
					bounds.include(position);
				}
				return !bounds.isEmpty();
			}

			const Dy::AvbdSoftBodyCompiledData& collisionCompiled =
				mCollisionProxy.collisionBodies[entry.bodyIndex].compiled;
			const PxU32 collisionParticleStart =
				collisionCompiled.particleStart;
			const PxU32 collisionParticleCount =
				collisionCompiled.particleCount;
			if(collisionParticleCount == 0 ||
				collisionParticleStart > mCollisionProxy.collisionVertexMappings.size() ||
				collisionParticleCount >
					mCollisionProxy.collisionVertexMappings.size() - collisionParticleStart)
				return false;
			for(PxU32 localParticleIndex = 0;
				localParticleIndex < collisionParticleCount;
				++localParticleIndex)
			{
				const Dy::AvbdWeightedContactPoint& mapping =
					mCollisionProxy.collisionVertexMappings[
						collisionParticleStart + localParticleIndex];
				if(mapping.count == 0 ||
					mapping.count >
						Dy::AVBD_CONTACT_POINT_MAX_SUPPORT)
					return false;
				PxVec3 position(0.0f);
				for(PxU32 supportIndex = 0;
					supportIndex < mapping.count; ++supportIndex)
				{
					const PxU32 particleIndex =
						mapping.particleIndices[supportIndex];
					const PxReal weight = mapping.weights[supportIndex];
					if(particleIndex >= mParticles.size() ||
						!PxIsFinite(weight))
						return false;
					PxVec3 supportPosition;
					if(!predictPosition(
							mParticles[particleIndex], supportPosition))
						return false;
					position += supportPosition * weight;
				}
				if(!position.isFinite())
					return false;
				bounds.include(position);
			}
			return !bounds.isEmpty();
		}

PxBounds3 AvbdCpuSoftScene::computeBoxBounds(
			const Dy::AvbdRigidBox& box) {
			const PxMat33 basis(box.rotation);
			const PxVec3& h = box.halfExtent;
			const PxVec3 extent(
				PxAbs(basis.column0.x) * h.x +
					PxAbs(basis.column1.x) * h.y +
					PxAbs(basis.column2.x) * h.z,
				PxAbs(basis.column0.y) * h.x +
					PxAbs(basis.column1.y) * h.y +
					PxAbs(basis.column2.y) * h.z,
				PxAbs(basis.column0.z) * h.x +
					PxAbs(basis.column1.z) * h.y +
					PxAbs(basis.column2.z) * h.z);
			return PxBounds3(
				box.center - extent, box.center + extent);
		}

PxBounds3 AvbdCpuSoftScene::computeSphereBounds(
			const Dy::AvbdRigidSphere& sphere) {
			const PxVec3 extent(PxMax(sphere.radius, 0.0f));
			return PxBounds3(
				sphere.center - extent,
				sphere.center + extent);
		}

PxBounds3 AvbdCpuSoftScene::computeCapsuleBounds(
			const Dy::AvbdRigidCapsule& capsule) {
			const PxVec3 axisOffset =
				capsule.rotation.getBasisVector0() *
					PxMax(capsule.halfHeight, 0.0f);
			const PxVec3 extent(PxMax(capsule.radius, 0.0f));
			const PxVec3 endpoint0 =
				capsule.center - axisOffset;
			const PxVec3 endpoint1 =
				capsule.center + axisOffset;
			return PxBounds3(
				endpoint0.minimum(endpoint1) - extent,
				endpoint0.maximum(endpoint1) + extent);
		}

PxBounds3 AvbdCpuSoftScene::computeConvexBounds(
			const Dy::AvbdRigidConvex& convex) {
			PxBounds3 bounds = PxBounds3::empty();
			for(PxU32 vertexIndex = 0;
				vertexIndex < convex.vertices.size(); ++vertexIndex)
			{
				const PxVec3 worldVertex =
					convex.center +
					convex.rotation.rotate(
						convex.vertices[vertexIndex]);
				if(!worldVertex.isFinite())
					return PxBounds3::empty();
				bounds.include(worldVertex);
			}
			return bounds;
		}
// Rigid proxy topology, material, and dynamic-shape compilation helpers.

bool AvbdCpuSoftScene::computeDynamicEndpointEnvelopeBounds(
			const DynamicShapeEntry& entry,
			const PxVec3& shapeCenter, PxReal shapeRadius,
			PxReal dt, const PxVec3& gravity, PxBounds3& bounds) {
			BodySim* const bodySim = entry.core ? entry.core->getSim() : NULL;
			if(!bodySim || bodySim->isKinematic() ||
				bodySim->isArticulationLink() ||
				!shapeCenter.isFinite() || !PxIsFinite(shapeRadius) ||
				shapeRadius < 0.0f)
				return false;
			const PxsBodyCore& bodyCore = entry.core->getCore();
			const PxVec3 bodyCenter = bodyCore.body2World.p;
			const PxVec3 predictedBodyCenter = bodyCenter +
				bodyCore.linearVelocity * dt +
				(bodyCore.disableGravity ? PxVec3(0.0f) :
					gravity * (dt * dt));
			const PxReal endpointRadius =
				shapeRadius + (shapeCenter - bodyCenter).magnitude();
			if(!bodyCenter.isFinite() || !predictedBodyCenter.isFinite() ||
				!PxIsFinite(endpointRadius))
				return false;
			const PxVec3 endpointExtent(endpointRadius);
			bounds = PxBounds3(
				predictedBodyCenter - endpointExtent,
				predictedBodyCenter + endpointExtent);
			return true;
		}

PxBounds3 AvbdCpuSoftScene::computeTriangleSurfaceBounds(
			const Dy::AvbdRigidTriangleSurface& surface) {
			PxBounds3 bounds = PxBounds3::empty();
			for(PxU32 vertexIndex = 0;
				vertexIndex < surface.vertices.size();
				++vertexIndex)
			{
				const PxVec3 worldVertex =
					surface.center +
					surface.rotation.rotate(
						surface.vertices[vertexIndex].point);
				if(!worldVertex.isFinite())
					return PxBounds3::empty();
				bounds.include(worldVertex);
			}
			return bounds;
		}

void AvbdCpuSoftScene::getRigidMaterialValues(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			PxMaterialTableIndex tableIndex,
			PxReal& friction, PxU8& combineMode) {
			const PxU16* materialIndices =
				shape.getMaterialIndices();
			const PxU32 materialCount =
				shape.getNbMaterialIndices();
			const PxU32 resolvedTableIndex =
				tableIndex == PxMaterialTableIndex(0xffff)
					? 0u : PxU32(tableIndex);
			friction = 0.5f;
			combineMode =
				PxU8(PxCombineMode::eAVERAGE);
			if(!materialIndices ||
				resolvedTableIndex >= materialCount)
				return;
			const PxU16 materialIndex =
				materialIndices[resolvedTableIndex];
			if(materialIndex == MATERIAL_INVALID_HANDLE ||
				materialIndex >= materialManager.getMaxSize())
				return;
			const PxsMaterialCore* material =
				materialManager.getMaterial(materialIndex);
			if(material->mMaterialIndex != materialIndex)
				return;
			friction =
				PxMax(material->dynamicFriction, 0.0f);
			combineMode =
				PxU8(material->getFrictionCombineMode());
		}

bool AvbdCpuSoftScene::appendTriangleSurfaceTriangle(
			const PxTriangle& sourceTriangle,
			const PxU32 sourceVertexIndices[3],
			PxU32 sourceTriangleIndex,
			PxReal friction, PxU8 frictionCombineMode,
			PxHashMap<PxU32, PxU32>& vertexMap,
			PxHashMap<PxU64, PxU32>& edgeMap,
			Dy::AvbdRigidTriangleSurface& surface) {
			PxU32 vertices[3] =
				{PX_MAX_U32, PX_MAX_U32, PX_MAX_U32};
			for(PxU32 endpoint = 0;
				endpoint < 3; ++endpoint)
			{
				const PxHashMap<PxU32, PxU32>::Entry* entry =
					vertexMap.find(
						sourceVertexIndices[endpoint]);
				if(entry)
					vertices[endpoint] = entry->second;
				else
				{
					const PxVec3& point =
						sourceTriangle.verts[endpoint];
					if(!point.isFinite())
						return false;
					Dy::AvbdRigidTriangleSurfaceVertex vertex;
					vertex.point = point;
					vertex.friction = friction;
					vertex.frictionCombineMode =
						frictionCombineMode;
					vertex.sourceTriangleIndex =
						sourceTriangleIndex;
					vertex.outward = PxVec3(0.0f);
					vertices[endpoint] =
						surface.vertices.size();
					surface.vertices.pushBack(vertex);
					vertexMap.insert(
						sourceVertexIndices[endpoint],
						vertices[endpoint]);
				}
			}
			if(vertices[0] == vertices[1] ||
				vertices[0] == vertices[2] ||
				vertices[1] == vertices[2])
				return true;
			PxVec3 normal =
				(sourceTriangle.verts[1] -
					sourceTriangle.verts[0]).cross(
						sourceTriangle.verts[2] -
							sourceTriangle.verts[0]);
			const PxReal normalMagnitudeSq =
				normal.magnitudeSquared();
			if(normalMagnitudeSq <= 1.0e-12f ||
				!PxIsFinite(normalMagnitudeSq))
				return true;
			normal *= PxRecipSqrt(normalMagnitudeSq);

			Dy::AvbdRigidTriangleSurfaceTriangle triangle;
			triangle.p0 = vertices[0];
			triangle.p1 = vertices[1];
			triangle.p2 = vertices[2];
			triangle.sourceTriangleIndex =
				sourceTriangleIndex;
			triangle.normal = normal;
			triangle.friction = friction;
			triangle.frictionCombineMode =
				frictionCombineMode;
			const PxU32 triangleIndex =
				surface.triangles.size();

			const PxU32 edgeEndpoints[3][2] =
			{
				{vertices[0], vertices[1]},
				{vertices[0], vertices[2]},
				{vertices[1], vertices[2]}
			};
			PxU32* triangleEdges[3] =
				{&triangle.edge0, &triangle.edge1,
				 &triangle.edge2};
			for(PxU32 localEdge = 0;
				localEdge < 3; ++localEdge)
			{
				const PxU32 edge0 = PxMin(
					edgeEndpoints[localEdge][0],
					edgeEndpoints[localEdge][1]);
				const PxU32 edge1 = PxMax(
					edgeEndpoints[localEdge][0],
					edgeEndpoints[localEdge][1]);
				const PxU64 edgeKey =
					(PxU64(edge0) << 32) | PxU64(edge1);
				const PxHashMap<PxU64, PxU32>::Entry* entry =
					edgeMap.find(edgeKey);
				PxU32 edgeIndex = PX_MAX_U32;
				if(entry)
					edgeIndex = entry->second;
				else
				{
					Dy::AvbdRigidTriangleSurfaceEdge edge;
					edge.p0 = edge0;
					edge.p1 = edge1;
					edge.outward = PxVec3(0.0f);
					edge.friction = friction;
					edge.frictionCombineMode =
						frictionCombineMode;
					edge.sourceTriangleIndex =
						sourceTriangleIndex;
					edgeIndex = surface.edges.size();
					surface.edges.pushBack(edge);
					edgeMap.insert(edgeKey, edgeIndex);
				}
				if(edgeIndex >= surface.edges.size())
					return false;
				Dy::AvbdRigidTriangleSurfaceEdge& edge =
					surface.edges[edgeIndex];
				if(edge.adjacentCount == 0)
					edge.triangle0 = triangleIndex;
				else if(edge.adjacentCount == 1)
					edge.triangle1 = triangleIndex;
				++edge.adjacentCount;
				edge.outward += normal;
				*triangleEdges[localEdge] = edgeIndex;
			}

			for(PxU32 endpoint = 0;
				endpoint < 3; ++endpoint)
				surface.vertices[vertices[endpoint]].
					outward += normal;
			surface.triangles.pushBack(triangle);
			return true;
		}

bool AvbdCpuSoftScene::finalizeTriangleSurfaceTopology(
			Dy::AvbdRigidTriangleSurface& surface,
			bool suppressBoundaryEdges) {
			if(surface.vertices.size() < 3 ||
				surface.triangles.empty())
				return false;
			surface.localBounds = PxBounds3::empty();
			surface.localRadius = 0.0f;
			for(PxU32 vertexIndex = 0;
				vertexIndex < surface.vertices.size();
				++vertexIndex)
			{
				Dy::AvbdRigidTriangleSurfaceVertex& vertex =
					surface.vertices[vertexIndex];
				if(!vertex.point.isFinite())
					return false;
				surface.localBounds.include(vertex.point);
				surface.localRadius = PxMax(
					surface.localRadius,
					vertex.point.magnitude());
				const PxReal normalMagnitudeSq =
					vertex.outward.magnitudeSquared();
				if(normalMagnitudeSq > 1.0e-12f &&
					PxIsFinite(normalMagnitudeSq))
					vertex.outward *=
						PxRecipSqrt(normalMagnitudeSq);
				else
					vertex.outward =
						PxVec3(0.0f, 1.0f, 0.0f);
			}

			for(PxU32 edgeIndex = 0;
				edgeIndex < surface.edges.size();
				++edgeIndex)
			{
				Dy::AvbdRigidTriangleSurfaceEdge& edge =
					surface.edges[edgeIndex];
				edge.active = false;
				if(edge.adjacentCount == 1)
					edge.active = !suppressBoundaryEdges;
				else if(edge.adjacentCount == 2 &&
					edge.triangle0 < surface.triangles.size() &&
					edge.triangle1 < surface.triangles.size())
				{
					const Dy::AvbdRigidTriangleSurfaceTriangle&
						triangle0 =
							surface.triangles[edge.triangle0];
					const Dy::AvbdRigidTriangleSurfaceTriangle&
						triangle1 =
							surface.triangles[edge.triangle1];
					PxU32 opposite0 = triangle0.p0;
					if(opposite0 == edge.p0 ||
						opposite0 == edge.p1)
						opposite0 = triangle0.p1;
					if(opposite0 == edge.p0 ||
						opposite0 == edge.p1)
						opposite0 = triangle0.p2;
					if(opposite0 >= surface.vertices.size() ||
						triangle1.p0 >=
							surface.vertices.size())
						return false;
					const PxReal oppositePlaneDistance =
						triangle1.normal.dot(
							surface.vertices[opposite0].point -
							surface.vertices[
								triangle1.p0].point);
					const PxReal normalDot =
						triangle0.normal.dot(
							triangle1.normal);
					edge.active =
						(oppositePlaneDistance < 0.0f &&
						 normalDot < 0.999999f) ||
						normalDot < -0.999f;
				}
				const PxReal normalMagnitudeSq =
					edge.outward.magnitudeSquared();
				if(normalMagnitudeSq > 1.0e-12f &&
					PxIsFinite(normalMagnitudeSq))
					edge.outward *=
						PxRecipSqrt(normalMagnitudeSq);
				else if(edge.triangle0 <
					surface.triangles.size())
					edge.outward =
						surface.triangles[
							edge.triangle0].normal;
				if(edge.active)
				{
					if(edge.p0 < surface.vertices.size())
						surface.vertices[edge.p0].active = true;
					if(edge.p1 < surface.vertices.size())
						surface.vertices[edge.p1].active = true;
				}
			}
			return !surface.localBounds.isEmpty() &&
				PxIsFinite(surface.localRadius) &&
				surface.localRadius > 0.0f;
		}

PxBounds3 AvbdCpuSoftScene::getRigidTriangleSurfaceTriangleBounds(
			const Dy::AvbdRigidTriangleSurface& surface,
			PxU32 triangleIndex) {
			PxBounds3 bounds(PxBounds3::empty());
			if(triangleIndex >= surface.triangles.size())
				return bounds;
			const Dy::AvbdRigidTriangleSurfaceTriangle& triangle =
				surface.triangles[triangleIndex];
			if(triangle.p0 >= surface.vertices.size() ||
				triangle.p1 >= surface.vertices.size() ||
				triangle.p2 >= surface.vertices.size())
				return bounds;
			bounds.include(surface.vertices[triangle.p0].point);
			bounds.include(surface.vertices[triangle.p1].point);
			bounds.include(surface.vertices[triangle.p2].point);
			return bounds;
		}

PxU32 AvbdCpuSoftScene::buildRigidTriangleSurfaceBvhNode(
			Dy::AvbdRigidTriangleSurface& surface,
			PxU32 firstPrimitive, PxU32 primitiveCount) {
			const PxU32 nodeIndex = surface.triangleBvhNodes.size();
			Dy::AvbdRigidTriangleSurfaceBvhNode node;
			node.minimum = PxVec3(PX_MAX_F32);
			node.maximum = PxVec3(-PX_MAX_F32);
			node.leftChild = PX_MAX_U32;
			node.rightChild = PX_MAX_U32;
			node.firstPrimitive = firstPrimitive;
			node.primitiveCount = primitiveCount;
			for(PxU32 entry = firstPrimitive;
				entry < firstPrimitive + primitiveCount; ++entry)
			{
				const PxBounds3 bounds =
					getRigidTriangleSurfaceTriangleBounds(
						surface,
						surface.triangleBvhTriangleIndices[entry]);
				if(!bounds.isEmpty())
				{
					node.minimum = node.minimum.minimum(bounds.minimum);
					node.maximum = node.maximum.maximum(bounds.maximum);
				}
			}
			surface.triangleBvhNodes.pushBack(node);
			if(primitiveCount <= 4)
				return nodeIndex;

			const PxVec3 extent = node.maximum - node.minimum;
			const PxU32 axis = extent.y > extent.x && extent.y >= extent.z
				? 1u : extent.z > extent.x && extent.z > extent.y ? 2u : 0u;
			PxSort(
				surface.triangleBvhTriangleIndices.begin() +
					firstPrimitive,
				primitiveCount,
				[&surface, axis](PxU32 lhs, PxU32 rhs)
				{
					const PxBounds3 lhsBounds =
						getRigidTriangleSurfaceTriangleBounds(surface, lhs);
					const PxBounds3 rhsBounds =
						getRigidTriangleSurfaceTriangleBounds(surface, rhs);
					const PxVec3 lhsCenter =
						(lhsBounds.minimum + lhsBounds.maximum) * 0.5f;
					const PxVec3 rhsCenter =
						(rhsBounds.minimum + rhsBounds.maximum) * 0.5f;
					const PxReal lhsValue = axis == 0 ? lhsCenter.x :
						axis == 1 ? lhsCenter.y : lhsCenter.z;
					const PxReal rhsValue = axis == 0 ? rhsCenter.x :
						axis == 1 ? rhsCenter.y : rhsCenter.z;
					return lhsValue == rhsValue ? lhs < rhs :
						lhsValue < rhsValue;
				});
			const PxU32 leftCount = primitiveCount / 2;
			const PxU32 leftChild = buildRigidTriangleSurfaceBvhNode(
				surface, firstPrimitive, leftCount);
			const PxU32 rightChild = buildRigidTriangleSurfaceBvhNode(
				surface, firstPrimitive + leftCount,
				primitiveCount - leftCount);
			surface.triangleBvhNodes[nodeIndex].leftChild = leftChild;
			surface.triangleBvhNodes[nodeIndex].rightChild = rightChild;
			return nodeIndex;
		}

void AvbdCpuSoftScene::buildRigidTriangleSurfaceBvh(
			Dy::AvbdRigidTriangleSurface& surface) {
			surface.triangleBvhTriangleIndices.clear();
			surface.triangleBvhNodes.clear();
			const PxU32 triangleCount = surface.triangles.size();
			if(triangleCount == 0)
				return;
			surface.triangleBvhTriangleIndices.reserve(triangleCount);
			surface.triangleBvhQueryCandidates.reserve(triangleCount);
			surface.edgeBvhQueryCandidates.reserve(surface.edges.size());
			surface.vertexBvhQueryCandidates.reserve(surface.vertices.size());
			surface.edgeBvhCandidateStamps.resize(surface.edges.size());
			surface.vertexBvhCandidateStamps.resize(surface.vertices.size());
			for(PxU32 edgeIndex = 0;
				edgeIndex < surface.edgeBvhCandidateStamps.size(); ++edgeIndex)
				surface.edgeBvhCandidateStamps[edgeIndex] = 0;
			for(PxU32 vertexIndex = 0;
				vertexIndex < surface.vertexBvhCandidateStamps.size(); ++vertexIndex)
				surface.vertexBvhCandidateStamps[vertexIndex] = 0;
			surface.featureBvhCandidateStamp = 0;
			for(PxU32 triangleIndex = 0;
				triangleIndex < triangleCount; ++triangleIndex)
				surface.triangleBvhTriangleIndices.pushBack(triangleIndex);
			buildRigidTriangleSurfaceBvhNode(
				surface, 0, triangleCount);
		}

bool AvbdCpuSoftScene::compileTriangleMeshTopology(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			const PxTriangleMeshGeometry& geometry,
			Dy::AvbdRigidTriangleSurface& surface) {
			PxTriangleMesh* mesh = geometry.triangleMesh;
			if(!mesh || !geometry.isValid())
				return false;
			surface.vertices.clear();
			surface.edges.clear();
			surface.triangles.clear();
			PxHashMap<PxU32, PxU32> vertexMap;
			PxHashMap<PxU64, PxU32> edgeMap;
			for(PxU32 triangleIndex = 0;
				triangleIndex < mesh->getNbTriangles();
				++triangleIndex)
			{
				PxTriangle triangle;
				PxU32 vertexIndices[3] =
					{PX_MAX_U32, PX_MAX_U32, PX_MAX_U32};
				PxMeshQuery::getTriangle(
					geometry, PxTransform(PxIdentity),
					triangleIndex, triangle, vertexIndices);
				PxReal friction = 0.5f;
				PxU8 frictionCombineMode =
					PxU8(PxCombineMode::eAVERAGE);
				getRigidMaterialValues(
					shape, materialManager,
					mesh->getTriangleMaterialIndex(
						triangleIndex),
					friction, frictionCombineMode);
				if(!appendTriangleSurfaceTriangle(
						triangle, vertexIndices,
						triangleIndex, friction,
						frictionCombineMode,
						vertexMap, edgeMap, surface))
					return false;
			}
			if(!finalizeTriangleSurfaceTopology(surface, false))
				return false;
			buildRigidTriangleSurfaceBvh(surface);
			return true;
		}

bool AvbdCpuSoftScene::compileHeightFieldTopology(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			const PxHeightFieldGeometry& geometry,
			Dy::AvbdRigidTriangleSurface& surface) {
			PxHeightField* heightField =
				geometry.heightField;
			if(!heightField || !geometry.isValid())
				return false;
			const PxU32 rows = heightField->getNbRows();
			const PxU32 columns =
				heightField->getNbColumns();
			if(rows < 2 || columns < 2)
				return false;
			surface.vertices.clear();
			surface.edges.clear();
			surface.triangles.clear();
			PxHashMap<PxU32, PxU32> vertexMap;
			PxHashMap<PxU64, PxU32> edgeMap;
			for(PxU32 row = 0; row + 1 < rows; ++row)
			{
				for(PxU32 column = 0;
					column + 1 < columns; ++column)
				{
					for(PxU32 localTriangle = 0;
						localTriangle < 2;
						++localTriangle)
					{
						const PxU32 triangleIndex =
							2 * (row * columns + column) +
							localTriangle;
						const PxMaterialTableIndex materialIndex =
							heightField->
								getTriangleMaterialIndex(
									triangleIndex);
						if(materialIndex ==
							PxHeightFieldMaterial::eHOLE)
							continue;
						PxTriangle triangle;
						PxU32 vertexIndices[3] =
							{PX_MAX_U32, PX_MAX_U32,
							 PX_MAX_U32};
						PxMeshQuery::getTriangle(
							geometry,
							PxTransform(PxIdentity),
							triangleIndex, triangle,
							vertexIndices);
						PxReal friction = 0.5f;
						PxU8 frictionCombineMode =
							PxU8(PxCombineMode::eAVERAGE);
						getRigidMaterialValues(
							shape, materialManager,
							materialIndex, friction,
							frictionCombineMode);
						if(!appendTriangleSurfaceTriangle(
								triangle, vertexIndices,
								triangleIndex, friction,
								frictionCombineMode,
								vertexMap, edgeMap, surface))
							return false;
					}
				}
			}
			const bool suppressBoundaryEdges =
				(heightField->getFlags() &
					PxHeightFieldFlag::eNO_BOUNDARY_EDGES)
				? true : false;
			if(!finalizeTriangleSurfaceTopology(
					surface, suppressBoundaryEdges))
				return false;
			buildRigidTriangleSurfaceBvh(surface);
			return true;
		}

bool AvbdCpuSoftScene::sameTriangleSurfaceVec3(
			const PxVec3& lhs, const PxVec3& rhs) {
			return lhs.x == rhs.x && lhs.y == rhs.y &&
				lhs.z == rhs.z;
		}

bool AvbdCpuSoftScene::sameTriangleSurfaceQuat(
			const PxQuat& lhs, const PxQuat& rhs) {
			return lhs.x == rhs.x && lhs.y == rhs.y &&
				lhs.z == rhs.z && lhs.w == rhs.w;
		}

bool AvbdCpuSoftScene::getTriangleMeshMaterialValues(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			const PxTriangleMesh& mesh,
			PxU32 sourceTriangleIndex, PxReal& friction,
			PxU8& frictionCombineMode) {
			if(sourceTriangleIndex >= mesh.getNbTriangles())
				return false;
			getRigidMaterialValues(
				shape, materialManager,
				mesh.getTriangleMaterialIndex(sourceTriangleIndex),
				friction, frictionCombineMode);
			return true;
		}

bool AvbdCpuSoftScene::getHeightFieldMaterialValues(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			const PxHeightField& heightField,
			PxU32 sourceTriangleIndex, PxReal& friction,
			PxU8& frictionCombineMode) {
			const PxMaterialTableIndex materialIndex =
				heightField.getTriangleMaterialIndex(
					sourceTriangleIndex);
			if(materialIndex == PxHeightFieldMaterial::eHOLE)
				return false;
			getRigidMaterialValues(
				shape, materialManager, materialIndex, friction,
				frictionCombineMode);
			return true;
		}

bool AvbdCpuSoftScene::refreshTriangleMeshSurfaceMaterials(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			const PxTriangleMeshGeometry& geometry,
			Dy::AvbdRigidTriangleSurface& surface) {
			PxTriangleMesh* mesh = geometry.triangleMesh;
			if(!mesh || !geometry.isValid())
				return false;
			for(PxU32 triangleIndex = 0;
				triangleIndex < surface.triangles.size(); ++triangleIndex)
			{
				Dy::AvbdRigidTriangleSurfaceTriangle& triangle =
					surface.triangles[triangleIndex];
				if(!getTriangleMeshMaterialValues(
						shape, materialManager, *mesh,
						triangle.sourceTriangleIndex, triangle.friction,
						triangle.frictionCombineMode))
					return false;
			}
			for(PxU32 vertexIndex = 0;
				vertexIndex < surface.vertices.size(); ++vertexIndex)
			{
				Dy::AvbdRigidTriangleSurfaceVertex& vertex =
					surface.vertices[vertexIndex];
				if(!getTriangleMeshMaterialValues(
						shape, materialManager, *mesh,
						vertex.sourceTriangleIndex, vertex.friction,
						vertex.frictionCombineMode))
					return false;
			}
			for(PxU32 edgeIndex = 0;
				edgeIndex < surface.edges.size(); ++edgeIndex)
			{
				Dy::AvbdRigidTriangleSurfaceEdge& edge =
					surface.edges[edgeIndex];
				if(!getTriangleMeshMaterialValues(
						shape, materialManager, *mesh,
						edge.sourceTriangleIndex, edge.friction,
						edge.frictionCombineMode))
					return false;
			}
			return true;
		}

bool AvbdCpuSoftScene::refreshHeightFieldSurfaceMaterials(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			const PxHeightFieldGeometry& geometry,
			Dy::AvbdRigidTriangleSurface& surface) {
			PxHeightField* heightField = geometry.heightField;
			if(!heightField || !geometry.isValid())
				return false;
			for(PxU32 triangleIndex = 0;
				triangleIndex < surface.triangles.size(); ++triangleIndex)
			{
				Dy::AvbdRigidTriangleSurfaceTriangle& triangle =
					surface.triangles[triangleIndex];
				if(!getHeightFieldMaterialValues(
						shape, materialManager, *heightField,
						triangle.sourceTriangleIndex, triangle.friction,
						triangle.frictionCombineMode))
					return false;
			}
			for(PxU32 vertexIndex = 0;
				vertexIndex < surface.vertices.size(); ++vertexIndex)
			{
				Dy::AvbdRigidTriangleSurfaceVertex& vertex =
					surface.vertices[vertexIndex];
				if(!getHeightFieldMaterialValues(
						shape, materialManager, *heightField,
						vertex.sourceTriangleIndex, vertex.friction,
						vertex.frictionCombineMode))
					return false;
			}
			for(PxU32 edgeIndex = 0;
				edgeIndex < surface.edges.size(); ++edgeIndex)
			{
				Dy::AvbdRigidTriangleSurfaceEdge& edge =
					surface.edges[edgeIndex];
				if(!getHeightFieldMaterialValues(
						shape, materialManager, *heightField,
						edge.sourceTriangleIndex, edge.friction,
						edge.frictionCombineMode))
					return false;
			}
			return true;
		}

bool AvbdCpuSoftScene::triangleMeshTopologyMatches(
			const PxTriangleMeshGeometry& geometry,
			const Dy::AvbdRigidTriangleSurface& surface) {
			return surface.topologyGeometryType ==
					PxU8(PxGeometryType::eTRIANGLEMESH) &&
				surface.topologySource == geometry.triangleMesh &&
				sameTriangleSurfaceVec3(
					surface.topologyScale, geometry.scale.scale) &&
				sameTriangleSurfaceQuat(
					surface.topologyScaleRotation,
					geometry.scale.rotation);
		}

bool AvbdCpuSoftScene::heightFieldTopologyMatches(
			const PxHeightFieldGeometry& geometry,
			const Dy::AvbdRigidTriangleSurface& surface) {
			return surface.topologyGeometryType ==
					PxU8(PxGeometryType::eHEIGHTFIELD) &&
				surface.topologySource == geometry.heightField &&
				geometry.heightField &&
				surface.topologyHeightScale == geometry.heightScale &&
				surface.topologyRowScale == geometry.rowScale &&
				surface.topologyColumnScale == geometry.columnScale &&
				surface.topologyContentTimestamp ==
					geometry.heightField->getTimestamp();
		}

void AvbdCpuSoftScene::setTriangleMeshTopologyIdentity(
			const PxTriangleMeshGeometry& geometry,
			Dy::AvbdRigidTriangleSurface& surface) {
			surface.topologySource = geometry.triangleMesh;
			surface.topologyGeometryType =
				PxU8(PxGeometryType::eTRIANGLEMESH);
			surface.topologyScale = geometry.scale.scale;
			surface.topologyScaleRotation = geometry.scale.rotation;
			surface.topologyHeightScale = 0.0f;
			surface.topologyRowScale = 0.0f;
			surface.topologyColumnScale = 0.0f;
			surface.topologyContentTimestamp = 0;
		}

void AvbdCpuSoftScene::setHeightFieldTopologyIdentity(
			const PxHeightFieldGeometry& geometry,
			Dy::AvbdRigidTriangleSurface& surface) {
			surface.topologySource = geometry.heightField;
			surface.topologyGeometryType =
				PxU8(PxGeometryType::eHEIGHTFIELD);
			surface.topologyScale = PxVec3(1.0f);
			surface.topologyScaleRotation = PxQuat(PxIdentity);
			surface.topologyHeightScale = geometry.heightScale;
			surface.topologyRowScale = geometry.rowScale;
			surface.topologyColumnScale = geometry.columnScale;
			surface.topologyContentTimestamp =
				geometry.heightField->getTimestamp();
		}

bool AvbdCpuSoftScene::refreshRigidTriangleSurfaceTopology(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager,
			Dy::AvbdRigidTriangleSurface& surface) {
			if(shape.getGeometryType() ==
				PxGeometryType::eTRIANGLEMESH)
			{
				const PxTriangleMeshGeometry& geometry =
					static_cast<const PxTriangleMeshGeometry&>(
						shape.getGeometry());
				if(triangleMeshTopologyMatches(geometry, surface) &&
					refreshTriangleMeshSurfaceMaterials(
						shape, materialManager, geometry, surface))
					return true;
				if(!compileTriangleMeshTopology(
						shape, materialManager, geometry, surface))
					return false;
				setTriangleMeshTopologyIdentity(geometry, surface);
				return true;
			}
			if(shape.getGeometryType() ==
				PxGeometryType::eHEIGHTFIELD)
			{
				const PxHeightFieldGeometry& geometry =
					static_cast<const PxHeightFieldGeometry&>(
						shape.getGeometry());
				if(heightFieldTopologyMatches(geometry, surface) &&
					refreshHeightFieldSurfaceMaterials(
						shape, materialManager, geometry, surface))
					return true;
				if(!compileHeightFieldTopology(
						shape, materialManager, geometry, surface))
					return false;
				setHeightFieldTopologyIdentity(geometry, surface);
				return true;
			}
			return false;
		}

bool AvbdCpuSoftScene::compileConvexTopology(
			const PxConvexMeshGeometry& geometry,
			Dy::AvbdRigidConvex& convex) {
			PxConvexMesh* mesh = geometry.convexMesh;
			if(!mesh ||
				!geometry.scale.isValidForConvexMesh() ||
				!geometry.scale.rotation.isFinite())
				return false;
			const PxU32 vertexCount = mesh->getNbVertices();
			const PxU32 polygonCount = mesh->getNbPolygons();
			const PxVec3* sourceVertices = mesh->getVertices();
			const PxU8* polygonIndices = mesh->getIndexBuffer();
			if(vertexCount < 4 || polygonCount < 4 ||
				!sourceVertices || !polygonIndices)
				return false;

			convex.vertices.resize(vertexCount);
			convex.vertexNormals.resize(vertexCount);
			convex.faces.clear();
			convex.edges.clear();
			convex.triangles.clear();
			PxVec3 centroid(0.0f);
			for(PxU32 vertexIndex = 0;
				vertexIndex < vertexCount; ++vertexIndex)
			{
				const PxVec3 vertex =
					geometry.scale.transform(
						sourceVertices[vertexIndex]);
				if(!vertex.isFinite())
					return false;
				convex.vertices[vertexIndex] = vertex;
				convex.vertexNormals[vertexIndex] =
					PxVec3(0.0f);
				centroid += vertex;
			}
			centroid *= 1.0f / PxReal(vertexCount);

			for(PxU32 polygonIndex = 0;
				polygonIndex < polygonCount; ++polygonIndex)
			{
				PxHullPolygon polygon;
				if(!mesh->getPolygonData(
						polygonIndex, polygon) ||
					polygon.mNbVerts < 3)
					return false;
				const PxU32 firstVertex =
					polygonIndices[polygon.mIndexBase];
				if(firstVertex >= vertexCount)
					return false;
				PxVec3 faceNormal(0.0f);
				PxU32 normalVertex1 = PX_MAX_U32;
				PxU32 normalVertex2 = PX_MAX_U32;
				for(PxU32 localVertex = 1;
					localVertex + 1 < polygon.mNbVerts;
					++localVertex)
				{
					const PxU32 vertex1 =
						polygonIndices[
							polygon.mIndexBase +
							localVertex];
					const PxU32 vertex2 =
						polygonIndices[
							polygon.mIndexBase +
							localVertex + 1];
					if(vertex1 >= vertexCount ||
						vertex2 >= vertexCount)
						return false;
					const PxVec3 candidate =
						(convex.vertices[vertex1] -
							convex.vertices[firstVertex]).
							cross(
								convex.vertices[vertex2] -
								convex.vertices[
									firstVertex]);
					if(candidate.magnitudeSquared() >
						1.0e-12f)
					{
						faceNormal = candidate.getNormalized();
						normalVertex1 = vertex1;
						normalVertex2 = vertex2;
						break;
					}
				}
				if(normalVertex1 == PX_MAX_U32 ||
					normalVertex2 == PX_MAX_U32 ||
					!faceNormal.isFinite())
					return false;
				const bool reverseWinding =
					faceNormal.dot(
						convex.vertices[firstVertex] -
							centroid) < 0.0f;
				if(reverseWinding)
					faceNormal = -faceNormal;

				Dy::AvbdRigidConvexFace face;
				face.normal = faceNormal;
				face.offset =
					faceNormal.dot(
						convex.vertices[firstVertex]);
				const PxU32 faceIndex = convex.faces.size();
				convex.faces.pushBack(face);

				for(PxU32 localVertex = 0;
					localVertex < polygon.mNbVerts;
					++localVertex)
				{
					const PxU32 vertex =
						polygonIndices[
							polygon.mIndexBase +
							localVertex];
					const PxU32 nextVertex =
						polygonIndices[
							polygon.mIndexBase +
							((localVertex + 1) %
								polygon.mNbVerts)];
					if(vertex >= vertexCount ||
						nextVertex >= vertexCount)
						return false;
					convex.vertexNormals[vertex] +=
						faceNormal;
					const PxU32 edge0 =
						PxMin(vertex, nextVertex);
					const PxU32 edge1 =
						PxMax(vertex, nextVertex);
					PxU32 edgeIndex = PX_MAX_U32;
					for(PxU32 candidateIndex = 0;
						candidateIndex < convex.edges.size();
						++candidateIndex)
					{
						if(convex.edges[candidateIndex].p0 ==
								edge0 &&
							convex.edges[candidateIndex].p1 ==
								edge1)
						{
							edgeIndex = candidateIndex;
							break;
						}
					}
					if(edgeIndex == PX_MAX_U32)
					{
						Dy::AvbdRigidConvexEdge edge;
						edge.p0 = edge0;
						edge.p1 = edge1;
						edge.outward = faceNormal;
						convex.edges.pushBack(edge);
					}
					else
						convex.edges[edgeIndex].outward +=
							faceNormal;
				}

				for(PxU32 localTriangle = 0;
					localTriangle + 2 <
						polygon.mNbVerts;
					++localTriangle)
				{
					const PxU32 fan1 =
						polygonIndices[
							polygon.mIndexBase +
							localTriangle + 1];
					const PxU32 fan2 =
						polygonIndices[
							polygon.mIndexBase +
							localTriangle + 2];
					if(fan1 >= vertexCount ||
						fan2 >= vertexCount)
						return false;
					Dy::AvbdRigidConvexTriangle triangle;
					triangle.p0 = firstVertex;
					triangle.p1 = reverseWinding
						? fan2 : fan1;
					triangle.p2 = reverseWinding
						? fan1 : fan2;
					triangle.faceIndex = faceIndex;
					convex.triangles.pushBack(triangle);
				}
			}

			convex.localRadius = 0.0f;
			for(PxU32 vertexIndex = 0;
				vertexIndex < vertexCount; ++vertexIndex)
			{
				const PxReal normalMagnitudeSq =
					convex.vertexNormals[vertexIndex].
						magnitudeSquared();
				if(normalMagnitudeSq <= 1.0e-12f ||
					!PxIsFinite(normalMagnitudeSq))
					return false;
				convex.vertexNormals[vertexIndex] *=
					PxRecipSqrt(normalMagnitudeSq);
				convex.localRadius = PxMax(
					convex.localRadius,
					convex.vertices[vertexIndex].magnitude());
			}
			for(PxU32 edgeIndex = 0;
				edgeIndex < convex.edges.size(); ++edgeIndex)
			{
				const PxReal normalMagnitudeSq =
					convex.edges[edgeIndex].outward.
						magnitudeSquared();
				if(normalMagnitudeSq <= 1.0e-12f ||
					!PxIsFinite(normalMagnitudeSq))
					return false;
				convex.edges[edgeIndex].outward *=
					PxRecipSqrt(normalMagnitudeSq);
			}
			return PxIsFinite(convex.localRadius) &&
				convex.localRadius > 0.0f &&
				!convex.triangles.empty();
		}

bool AvbdCpuSoftScene::compileDynamicConvex(
			const DynamicShapeEntry& entry,
			Dy::AvbdRigidConvex& convex) const {
			BodySim* bodySim = entry.core->getSim();
			if(!bodySim || bodySim->isArticulationLink())
				return false;
			const ShapeCore& shape = *entry.shape;
			if(!(shape.getFlags() &
					PxShapeFlag::eSIMULATION_SHAPE) ||
				shape.getGeometryType() !=
					PxGeometryType::eCONVEXMESH)
				return false;
			const PxConvexMeshGeometry& geometry =
				static_cast<const PxConvexMeshGeometry&>(
					shape.getGeometry());
			if(!compileConvexTopology(geometry, convex))
				return false;

			const PxsBodyCore& bodyCore =
				entry.core->getCore();
			const PxTransform previousActorToWorld =
				bodyCore.body2World *
					bodyCore.getBody2Actor().getInverse();
			const PxTransform previousShapeToWorld =
				previousActorToWorld * shape.getShape2Actor();
			PxTransform bodyToWorld = bodyCore.body2World;
			if(bodySim->isKinematic())
			{
				PxTransform targetPose;
				if(entry.core->getKinematicTarget(targetPose))
					bodyToWorld = targetPose;
			}
			const PxTransform actorToWorld =
				bodyToWorld *
					bodyCore.getBody2Actor().getInverse();
			const PxTransform shapeToWorld =
				actorToWorld * shape.getShape2Actor();
			if(!shapeToWorld.isValid() ||
				!previousShapeToWorld.isValid())
				return false;
			convex.center = shapeToWorld.p;
			convex.rotation = shapeToWorld.q;
			convex.previousCenter = previousShapeToWorld.p;
			convex.previousRotation = previousShapeToWorld.q;
			convex.friction = getStaticFriction(
				shape, mRigidMaterialManager);
			convex.frictionCombineMode =
				getStaticFrictionCombineMode(
					shape, mRigidMaterialManager);
			convex.primitiveKey = entry.primitiveKey;
			return true;
		}

bool AvbdCpuSoftScene::compileDynamicTriangleSurface(
			const DynamicShapeEntry& entry,
			Dy::AvbdRigidTriangleSurface& surface) const {
			BodySim* bodySim = entry.core->getSim();
			if(!bodySim || bodySim->isArticulationLink())
				return false;
			const ShapeCore& shape = *entry.shape;
			if(!(shape.getFlags() &
					PxShapeFlag::eSIMULATION_SHAPE))
				return false;
			if(!refreshRigidTriangleSurfaceTopology(
					shape, mRigidMaterialManager, surface))
				return false;

			const PxsBodyCore& bodyCore =
				entry.core->getCore();
			const PxTransform previousActorToWorld =
				bodyCore.body2World *
					bodyCore.getBody2Actor().getInverse();
			const PxTransform previousShapeToWorld =
				previousActorToWorld * shape.getShape2Actor();
			PxTransform bodyToWorld = bodyCore.body2World;
			if(bodySim->isKinematic())
			{
				PxTransform targetPose;
				if(entry.core->getKinematicTarget(targetPose))
					bodyToWorld = targetPose;
			}
			const PxTransform actorToWorld =
				bodyToWorld *
					bodyCore.getBody2Actor().getInverse();
			const PxTransform shapeToWorld =
				actorToWorld * shape.getShape2Actor();
			if(!shapeToWorld.isValid() ||
				!previousShapeToWorld.isValid())
				return false;
			surface.center = shapeToWorld.p;
			surface.rotation = shapeToWorld.q;
			surface.previousCenter = previousShapeToWorld.p;
			surface.previousRotation =
				previousShapeToWorld.q;
			surface.primitiveKey = entry.primitiveKey;
			return true;
		}

bool AvbdCpuSoftScene::compileDynamicBox(
			const DynamicShapeEntry& entry,
			Dy::AvbdRigidBox& box) const {
			BodySim* bodySim = entry.core->getSim();
			if(!bodySim || bodySim->isArticulationLink())
				return false;
			const ShapeCore& shape = *entry.shape;
			if(!(shape.getFlags() &
					PxShapeFlag::eSIMULATION_SHAPE) ||
				shape.getGeometryType() !=
					PxGeometryType::eBOX)
				return false;

			const PxsBodyCore& bodyCore =
				entry.core->getCore();
			const PxTransform previousActorToWorld =
				bodyCore.body2World *
				bodyCore.getBody2Actor().getInverse();
			const PxTransform previousShapeToWorld =
				previousActorToWorld * shape.getShape2Actor();
			PxTransform bodyToWorld = bodyCore.body2World;
			if(bodySim->isKinematic())
			{
				PxTransform targetPose;
				if(entry.core->getKinematicTarget(targetPose))
					bodyToWorld = targetPose;
			}
			const PxTransform actorToWorld =
				bodyToWorld *
				bodyCore.getBody2Actor().getInverse();
			const PxTransform shapeToWorld =
				actorToWorld * shape.getShape2Actor();
			if(!shapeToWorld.isValid())
				return false;

			const PxBoxGeometry& geometry =
				static_cast<const PxBoxGeometry&>(
					shape.getGeometry());
			box.center = shapeToWorld.p;
			box.rotation = shapeToWorld.q;
			box.previousCenter = previousShapeToWorld.p;
			box.previousRotation = previousShapeToWorld.q;
			box.halfExtent = geometry.halfExtents;
			box.friction = getStaticFriction(
				shape, mRigidMaterialManager);
			box.frictionCombineMode =
				getStaticFrictionCombineMode(
					shape, mRigidMaterialManager);
			box.primitiveKey = entry.primitiveKey;
			return true;
		}

bool AvbdCpuSoftScene::compileDynamicSphere(
			const DynamicShapeEntry& entry,
			Dy::AvbdRigidSphere& sphere) const {
			BodySim* bodySim = entry.core->getSim();
			if(!bodySim || bodySim->isArticulationLink())
				return false;
			const ShapeCore& shape = *entry.shape;
			if(!(shape.getFlags() &
					PxShapeFlag::eSIMULATION_SHAPE) ||
				shape.getGeometryType() !=
					PxGeometryType::eSPHERE)
				return false;

			const PxSphereGeometry& geometry =
				static_cast<const PxSphereGeometry&>(
					shape.getGeometry());
			if(geometry.radius <= 0.0f ||
				!PxIsFinite(geometry.radius))
				return false;
			const PxsBodyCore& bodyCore =
				entry.core->getCore();
			const PxTransform previousActorToWorld =
				bodyCore.body2World *
					bodyCore.getBody2Actor().getInverse();
			const PxTransform previousShapeToWorld =
				previousActorToWorld * shape.getShape2Actor();
			PxTransform bodyToWorld = bodyCore.body2World;
			if(bodySim->isKinematic())
			{
				PxTransform targetPose;
				if(entry.core->getKinematicTarget(targetPose))
					bodyToWorld = targetPose;
			}
			const PxTransform actorToWorld =
				bodyToWorld *
					bodyCore.getBody2Actor().getInverse();
			const PxTransform shapeToWorld =
				actorToWorld * shape.getShape2Actor();
			if(!shapeToWorld.isValid() ||
				!previousShapeToWorld.isValid())
				return false;

			sphere.center = shapeToWorld.p;
			sphere.rotation = shapeToWorld.q;
			sphere.previousCenter = previousShapeToWorld.p;
			sphere.previousRotation = previousShapeToWorld.q;
			sphere.radius = geometry.radius;
			sphere.friction = getStaticFriction(
				shape, mRigidMaterialManager);
			sphere.frictionCombineMode =
				getStaticFrictionCombineMode(
					shape, mRigidMaterialManager);
			sphere.primitiveKey = entry.primitiveKey;
			return true;
		}

bool AvbdCpuSoftScene::compileDynamicCapsule(
			const DynamicShapeEntry& entry,
			Dy::AvbdRigidCapsule& capsule) const {
			BodySim* bodySim = entry.core->getSim();
			if(!bodySim || bodySim->isArticulationLink())
				return false;
			const ShapeCore& shape = *entry.shape;
			if(!(shape.getFlags() &
					PxShapeFlag::eSIMULATION_SHAPE) ||
				shape.getGeometryType() !=
					PxGeometryType::eCAPSULE)
				return false;

			const PxCapsuleGeometry& geometry =
				static_cast<const PxCapsuleGeometry&>(
					shape.getGeometry());
			if(geometry.radius <= 0.0f ||
				geometry.halfHeight < 0.0f ||
				!PxIsFinite(geometry.radius) ||
				!PxIsFinite(geometry.halfHeight))
				return false;
			const PxsBodyCore& bodyCore =
				entry.core->getCore();
			const PxTransform previousActorToWorld =
				bodyCore.body2World *
					bodyCore.getBody2Actor().getInverse();
			const PxTransform previousShapeToWorld =
				previousActorToWorld * shape.getShape2Actor();
			PxTransform bodyToWorld = bodyCore.body2World;
			if(bodySim->isKinematic())
			{
				PxTransform targetPose;
				if(entry.core->getKinematicTarget(targetPose))
					bodyToWorld = targetPose;
			}
			const PxTransform actorToWorld =
				bodyToWorld *
					bodyCore.getBody2Actor().getInverse();
			const PxTransform shapeToWorld =
				actorToWorld * shape.getShape2Actor();
			if(!shapeToWorld.isValid() ||
				!previousShapeToWorld.isValid())
				return false;

			capsule.center = shapeToWorld.p;
			capsule.rotation = shapeToWorld.q;
			capsule.previousCenter = previousShapeToWorld.p;
			capsule.previousRotation = previousShapeToWorld.q;
			capsule.radius = geometry.radius;
			capsule.halfHeight = geometry.halfHeight;
			capsule.friction = getStaticFriction(
				shape, mRigidMaterialManager);
			capsule.frictionCombineMode =
				getStaticFrictionCombineMode(
					shape, mRigidMaterialManager);
			capsule.primitiveKey = entry.primitiveKey;
			return true;
		}

const PxsDeformableVolumeMaterialCore* AvbdCpuSoftScene::getMaterial(
			const DeformableVolumeCore& core,
			const PxsDeformableVolumeMaterialManager& materialManager) {
			const PxArray<PxU16>& handles =
				core.getCore().materialHandles;
			if(handles.empty() ||
				handles[0] == MATERIAL_INVALID_HANDLE ||
				handles[0] >= materialManager.getMaxSize())
				return NULL;
			const PxsDeformableVolumeMaterialCore* material =
				materialManager.getMaterial(handles[0]);
			return material->mMaterialIndex == handles[0]
				? material : NULL;
		}

PxReal AvbdCpuSoftScene::getStaticFriction(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager) {
			const PxU16* materialIndices =
				shape.getMaterialIndices();
			if(!materialIndices ||
				shape.getNbMaterialIndices() == 0 ||
				materialIndices[0] == MATERIAL_INVALID_HANDLE ||
				materialIndices[0] >= materialManager.getMaxSize())
				return 0.5f;
			const PxsMaterialCore* material =
				materialManager.getMaterial(materialIndices[0]);
			return material->mMaterialIndex == materialIndices[0]
				? PxMax(material->dynamicFriction, 0.0f) : 0.5f;
		}

PxU8 AvbdCpuSoftScene::getStaticFrictionCombineMode(
			const ShapeCore& shape,
			const PxsMaterialManager& materialManager) {
			const PxU16* materialIndices =
				shape.getMaterialIndices();
			if(!materialIndices ||
				shape.getNbMaterialIndices() == 0 ||
				materialIndices[0] == MATERIAL_INVALID_HANDLE ||
				materialIndices[0] >= materialManager.getMaxSize())
				return PxU8(PxCombineMode::eAVERAGE);
			const PxsMaterialCore* material =
				materialManager.getMaterial(materialIndices[0]);
			return material->mMaterialIndex == materialIndices[0]
				? PxU8(material->getFrictionCombineMode())
				: PxU8(PxCombineMode::eAVERAGE);
		}

Dy::AvbdRigidTriangleSurface& AvbdCpuSoftScene::getRigidTriangleSurface(
			PxU64 primitiveKey) {
			for(PxU32 surfaceIndex = 0;
				surfaceIndex < mRigidTriangleSurfaces.size();
				++surfaceIndex)
			{
				if(mRigidTriangleSurfaces[surfaceIndex].primitiveKey ==
					primitiveKey)
					return mRigidTriangleSurfaces[surfaceIndex];
			}
			const PxU32 newSurfaceIndex =
				mRigidTriangleSurfaces.size();
			mRigidTriangleSurfaces.pushBack(
				Dy::AvbdRigidTriangleSurface());
			mRigidTriangleSurfaces[newSurfaceIndex].primitiveKey =
				primitiveKey;
			return mRigidTriangleSurfaces[newSurfaceIndex];
		}

} // namespace Sc
} // namespace physx
