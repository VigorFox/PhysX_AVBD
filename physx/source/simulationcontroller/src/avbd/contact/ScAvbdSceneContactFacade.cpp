// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "avbd/scene/ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

		ActorCore* AvbdCpuSoftScene::findRigidCoreForPrimitive(
			PxU64 primitiveKey) const
		{
			for(PxU32 i = 0; i < mStaticShapes.size(); ++i)
				if(mStaticShapes[i].primitiveKey == primitiveKey)
					return static_cast<ActorCore*>(
						mStaticShapes[i].core);
			for(PxU32 i = 0; i < mDynamicShapes.size(); ++i)
				if(mDynamicShapes[i].primitiveKey == primitiveKey)
					return static_cast<ActorCore*>(
						mDynamicShapes[i].core);
			return NULL;
		}

		ActorCore* AvbdCpuSoftScene::findSoftCoreForContactBody(
			const Dy::AvbdSoftBody* bodies,
			PxU32 numBodies,
			ActorCore* const* softCores,
			PxU32 particleIndex) const
		{
			for(PxU32 bodyIndex = 0;
				bodyIndex < numBodies; ++bodyIndex)
			{
				const Dy::AvbdSoftBodyCompiledData& compiled =
					bodies[bodyIndex].compiled;
				if(particleIndex < compiled.particleStart ||
					particleIndex >=
						compiled.particleStart +
						compiled.particleCount)
					continue;
				if(softCores)
					return softCores[bodyIndex];
				if(bodies == mBodies.begin() ||
					bodies == mCollisionProxy.collisionBodies.begin())
				{
					for(PxU32 entryIndex = 0;
						entryIndex < mEntries.size(); ++entryIndex)
					{
						const Entry& entry = mEntries[entryIndex];
						if(entry.bodyIndex == bodyIndex)
							return entry.getActorCore();
					}
				}
				return NULL;
			}
			return NULL;
		}

		ActorCore* AvbdCpuSoftScene::findSoftCoreForContactBodyIndex(
			const Dy::AvbdSoftBody* bodies,
			PxU32 numBodies,
			ActorCore* const* softCores,
			PxU32 bodyIndex) const
		{
			if(bodyIndex >= numBodies)
				return NULL;
			if(softCores)
				return softCores[bodyIndex];
			if(bodies == mBodies.begin() ||
				bodies == mCollisionProxy.collisionBodies.begin())
			{
				for(PxU32 entryIndex = 0;
					entryIndex < mEntries.size(); ++entryIndex)
				{
					const Entry& entry = mEntries[entryIndex];
					if(entry.bodyIndex == bodyIndex)
						return entry.getActorCore();
				}
			}
			return NULL;
		}

		const Dy::AvbdSoftBody*
		AvbdCpuSoftScene::findSoftBodyForContactParticle(
			const Dy::AvbdSoftBody* bodies,
			PxU32 numBodies,
			PxU32 particleIndex) const
		{
			for(PxU32 bodyIndex = 0;
				bodyIndex < numBodies; ++bodyIndex)
			{
				const Dy::AvbdSoftBodyCompiledData& compiled =
					bodies[bodyIndex].compiled;
				if(particleIndex >= compiled.particleStart &&
					particleIndex <
						compiled.particleStart +
							compiled.particleCount)
					return &bodies[bodyIndex];
			}
			return NULL;
		}

		bool AvbdCpuSoftScene::isRigidActorContactFiltered(
			const Dy::AvbdSoftBody& body,
			ActorCore& softCore,
			ActorCore& rigidCore,
			PxU32 particleIndex) const
		{
			bool hasMatchingFilter = false;
			for(PxU32 filterIndex = 0;
				filterIndex < mRigidActorFilters.size();
				++filterIndex)
			{
				const RigidActorFilterEntry& filter =
					mRigidActorFilters[filterIndex];
				if(filter.softCore != &softCore ||
					filter.rigidCore != &rigidCore)
					continue;
				hasMatchingFilter = true;
				if(filter.filterAllElements)
					return true;
			}
			if(!hasMatchingFilter ||
				particleIndex < body.compiled.particleStart)
				return false;
			const PxU32 localParticle =
				particleIndex - body.compiled.particleStart;
			if(localParticle >=
				body.compiled.elementAdjacency.size())
				return false;

			// Rigid contact generation is particle-sampled on the collision
			// proxy. Surface filters own public triangles and Volume filters own
			// public collision tetrahedra directly. In both domains, remove the
			// objective only when every incident
			// element is covered by the union of active filter objects.
			const PxArray<Dy::AvbdParticleElementRef>& incident =
				body.compiled.triElements.empty()
					? body.compiled.elementAdjacency[
						localParticle].tetRefs
					: body.compiled.elementAdjacency[
						localParticle].triRefs;
			if(incident.empty())
				return false;
			const bool volumeOwnership =
				body.compiled.triElements.empty();
			for(PxU32 refIndex = 0;
				refIndex < incident.size(); ++refIndex)
			{
				const PxU32 compiledElementIndex =
					incident[refIndex].index;
				PxU32 sourceElementIndex = PX_MAX_U32;
				if(volumeOwnership)
				{
					if(compiledElementIndex >=
						body.compiled.tetElements.size())
						return false;
					sourceElementIndex =
						body.compiled.tetElements[
							compiledElementIndex].
								sourceElementIndex;
				}
				else
				{
					if(compiledElementIndex >=
						body.compiled.triElements.size())
						return false;
					sourceElementIndex =
						body.compiled.triElements[
							compiledElementIndex].
								sourceElementIndex;
				}
				bool elementFiltered = false;
				for(PxU32 filterIndex = 0;
					filterIndex < mRigidActorFilters.size();
					++filterIndex)
				{
					const RigidActorFilterEntry& filter =
						mRigidActorFilters[filterIndex];
					if(filter.softCore == &softCore &&
						filter.rigidCore == &rigidCore &&
						filter.containsElement(
							sourceElementIndex))
					{
						elementFiltered = true;
						break;
					}
				}
				if(!elementFiltered)
					return false;
			}
			return true;
		}

		bool AvbdCpuSoftScene::isDeformablePairContactFiltered(
			const Dy::AvbdSoftBody& queryBody,
			ActorCore& queryCore,
			ActorCore& targetCore,
			PxU32 queryParticleIndex,
			PxU32 targetSourceElementIndex) const
		{
			if(targetSourceElementIndex == PX_MAX_U32 ||
				queryParticleIndex <
					queryBody.compiled.particleStart)
				return false;
			const PxU32 localParticle =
				queryParticleIndex -
					queryBody.compiled.particleStart;
			if(localParticle >=
				queryBody.compiled.elementAdjacency.size())
				return false;
			const bool volumeOwnership =
				queryBody.compiled.triElements.empty();
			const PxArray<Dy::AvbdParticleElementRef>& incident =
				volumeOwnership
					? queryBody.compiled.elementAdjacency[
						localParticle].tetRefs
					: queryBody.compiled.elementAdjacency[
						localParticle].triRefs;
			if(incident.empty())
				return false;

			// Contact detection samples a query particle against one
			// explicit target boundary face. A shared query particle
			// belongs to every incident source element, so the prepared
			// objective is removed only when the union of active filter
			// objects covers every query/target source-element pair.
			for(PxU32 refIndex = 0;
				refIndex < incident.size(); ++refIndex)
			{
				const PxU32 compiledElementIndex =
					incident[refIndex].index;
				PxU32 querySourceElementIndex = PX_MAX_U32;
				if(volumeOwnership)
				{
					if(compiledElementIndex >=
						queryBody.compiled.tetElements.size())
						return false;
					querySourceElementIndex =
						queryBody.compiled.tetElements[
							compiledElementIndex].
								sourceElementIndex;
				}
				else
				{
					if(compiledElementIndex >=
						queryBody.compiled.triElements.size())
						return false;
					querySourceElementIndex =
						queryBody.compiled.triElements[
							compiledElementIndex].
								sourceElementIndex;
				}
				bool pairFiltered = false;
				for(PxU32 filterIndex = 0;
					filterIndex < mDeformablePairFilters.size();
					++filterIndex)
				{
					if(mDeformablePairFilters[filterIndex].
						containsPair(
							queryCore,
							querySourceElementIndex,
							targetCore,
							targetSourceElementIndex))
					{
						pairFiltered = true;
						break;
					}
				}
				if(!pairFiltered)
					return false;
			}
			return true;
		}

		bool AvbdCpuSoftScene::removeRigidActorFilteredContacts(
			const Dy::AvbdSoftBody* bodies,
			PxU32 numBodies,
			ActorCore* const* softCores,
			PxArray<Dy::AvbdSoftContact>& contacts,
			Dy::AvbdOgcGeometryEpochSidecar* geometrySidecar) const
		{
			if(geometrySidecar &&
				geometrySidecar->contactTriangleCoreIndices.size() !=
					contacts.size())
				return false;
			if(mRigidActorFilters.empty())
				return true;
			PxU32 writeIndex = 0;
			for(PxU32 contactIndex = 0;
				contactIndex < contacts.size(); ++contactIndex)
			{
				const Dy::AvbdSoftContact& contact =
					contacts[contactIndex];
				const Dy::AvbdSoftContactGeometry& geometry =
					contact.geometry;
				const bool rigidSource =
					geometry.source.type ==
						Dy::AvbdSoftContactSource::eGROUND ||
					geometry.source.type ==
						Dy::AvbdSoftContactSource::eRIGID_SDF;
				bool filtered = false;
				if(rigidSource)
				{
					ActorCore* softCore =
						findSoftCoreForContactBody(
							bodies, numBodies, softCores,
							geometry.particleIdx);
					ActorCore* rigidCore =
						findRigidCoreForPrimitive(
							geometry.source.primitiveKey);
					const Dy::AvbdSoftBody* softBody =
						findSoftBodyForContactParticle(
							bodies, numBodies,
							geometry.particleIdx);
					if(softCore && rigidCore && softBody)
						filtered =
							isRigidActorContactFiltered(
								*softBody, *softCore,
								*rigidCore,
								geometry.particleIdx);
				}
				if(!filtered)
				{
					if(geometrySidecar &&
						!geometrySidecar->moveContactMapping(
							contactIndex, writeIndex))
						return false;
					if(writeIndex != contactIndex)
						contacts[writeIndex] =
							contacts[contactIndex];
					++writeIndex;
				}
			}
			contacts.resize(writeIndex);
			return !geometrySidecar ||
				geometrySidecar->finalizeContactCompaction(writeIndex);
		}

		bool AvbdCpuSoftScene::removeDeformablePairFilteredContacts(
			const Dy::AvbdSoftBody* bodies,
			PxU32 numBodies,
			ActorCore* const* softCores,
			PxArray<Dy::AvbdSoftContact>& contacts,
			Dy::AvbdOgcGeometryEpochSidecar* geometrySidecar) const
		{
			if(geometrySidecar &&
				geometrySidecar->contactTriangleCoreIndices.size() !=
					contacts.size())
				return false;
			if(mDeformablePairFilters.empty())
				return true;
			PxU32 writeIndex = 0;
			for(PxU32 contactIndex = 0;
				contactIndex < contacts.size(); ++contactIndex)
			{
				const Dy::AvbdSoftContact& contact =
					contacts[contactIndex];
				const Dy::AvbdSoftContactGeometry& geometry =
					contact.geometry;
				bool filtered = false;
				if(geometry.source.type ==
					Dy::AvbdSoftContactSource::eSOFT_SURFACE)
				{
					ActorCore* queryCore =
						findSoftCoreForContactBody(
							bodies, numBodies, softCores,
							geometry.particleIdx);
					ActorCore* targetCore =
						findSoftCoreForContactBodyIndex(
							bodies, numBodies, softCores,
							geometry.source.targetBodyIndex);
					const Dy::AvbdSoftBody* queryBody =
						findSoftBodyForContactParticle(
							bodies, numBodies,
							geometry.particleIdx);
					if(queryCore && targetCore && queryBody)
						filtered =
							isDeformablePairContactFiltered(
								*queryBody, *queryCore,
								*targetCore,
								geometry.particleIdx,
								geometry.
									targetSourceElementIndex);
				}
				if(!filtered)
				{
					if(geometrySidecar &&
						!geometrySidecar->moveContactMapping(
							contactIndex, writeIndex))
						return false;
					if(writeIndex != contactIndex)
						contacts[writeIndex] =
							contacts[contactIndex];
					++writeIndex;
				}
			}
			contacts.resize(writeIndex);
			return !geometrySidecar ||
				geometrySidecar->finalizeContactCompaction(writeIndex);
		}


} // namespace Sc
} // namespace physx
