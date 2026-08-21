// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "ScAvbdIslandSelectionPlan.h"

namespace physx
{
namespace Sc
{

bool compileAvbdIslandSelectionExecutionPlan(
			AvbdIslandSelectionStorage& storage,
			PxU32 numParticles, PxU32 numRigidBodies)
		{
			if(numParticles == 0 ||
				storage.bodies.size() != storage.entryIndices.size())
				return false;
			storage.ogcPairContactStarts.clear();
			storage.ogcPairContactCounts.clear();
			storage.ogcPairContactRefs.clear();

			storage.particleBodyIndices.resize(numParticles);
			for(PxU32 particleIndex = 0;
				particleIndex < numParticles; particleIndex++)
				storage.particleBodyIndices[particleIndex] = PX_MAX_U32;
			for(PxU32 bodyIndex = 0;
				bodyIndex < storage.bodies.size(); bodyIndex++)
			{
				const Dy::AvbdSoftBody& body = storage.bodies[bodyIndex];
				const PxU32 particleStart =
					body.compiled.particleStart;
				const PxU32 particleCount =
					body.compiled.particleCount;
				if(particleStart > numParticles ||
					particleCount > numParticles - particleStart)
					return false;
				for(PxU32 localParticleIndex = 0;
					localParticleIndex < particleCount;
					localParticleIndex++)
				{
					const PxU32 particleIndex =
						particleStart + localParticleIndex;
					if(storage.particleBodyIndices[particleIndex] !=
						PX_MAX_U32)
						return false;
					storage.particleBodyIndices[particleIndex] = bodyIndex;
				}
			}
			for(PxU32 particleIndex = 0;
				particleIndex < numParticles; particleIndex++)
				if(storage.particleBodyIndices[particleIndex] ==
					PX_MAX_U32)
					return false;

			// This is the same stable, source-ordered CSR used by the component
			// soft path.  Compiling it at the provider boundary lets the native
			// mixed solver consume one immutable execution program instead of
			// rebuilding ownership and incidence from raw contact rows.
			storage.contactStarts.resize(numParticles + 1);
			storage.contactCounts.resize(numParticles);
			// Every dynamic rigid endpoint needs the same immutable contact
			// ownership program as its soft counterpart.  Omitting the mirror for a
			// one-body island saved a tiny setup pass, but it also made the sole 6DOF
			// rigid block bypass the shared OGC pair state.  Keep the CSR for one
			// body as well: the O(contact) setup is bounded and makes mixed OGC
			// scheduling independent of island cardinality.
			const bool mayBuildRigidTargetContactPlan =
				numRigidBodies > 0 && !storage.contacts.empty();
			PxU32 rigidTargetContactCount = 0;
			storage.rigidTargetContactStarts.clear();
			storage.rigidTargetContactCounts.clear();
			storage.rigidTargetContactRefs.clear();
			if(mayBuildRigidTargetContactPlan)
			{
				storage.rigidTargetContactCounts.resize(numRigidBodies);
				for(PxU32 rigidBodyIndex = 0;
					rigidBodyIndex < numRigidBodies; rigidBodyIndex++)
					storage.rigidTargetContactCounts[rigidBodyIndex] = 0;
			}
			for(PxU32 particleIndex = 0;
				particleIndex < numParticles; particleIndex++)
				storage.contactCounts[particleIndex] = 0;
			for(PxU32 contactIndex = 0;
				contactIndex < storage.contacts.size(); contactIndex++)
			{
				const Dy::AvbdSoftContactGeometry& geometry =
					storage.contacts[contactIndex].geometry;
				if(mayBuildRigidTargetContactPlan &&
					geometry.hasRigidBodyTarget() &&
					geometry.targetIndex < numRigidBodies)
				{
					storage.rigidTargetContactCounts[
						geometry.targetIndex]++;
					rigidTargetContactCount++;
				}
				PxU32 particleIndices[
					Dy::AVBD_CONTACT_MAX_PARTICLES];
				const PxU32 particleIndexCount =
					Dy::avbdCollectSoftContactParticleIndices(
						geometry, particleIndices);
				for(PxU32 particleOffset = 0;
					particleOffset < particleIndexCount;
					particleOffset++)
				{
					const PxU32 particleIndex =
						particleIndices[particleOffset];
					if(particleIndex >= numParticles)
						continue;
					if(PxAbs(
						Dy::avbdGetSoftContactParticleJacobianScale(
							geometry, particleIndex)) > 1e-12f)
						storage.contactCounts[particleIndex]++;
				}
			}
			storage.contactStarts[0] = 0;
			for(PxU32 particleIndex = 0;
				particleIndex < numParticles; particleIndex++)
				storage.contactStarts[particleIndex + 1] =
					storage.contactStarts[particleIndex] +
					storage.contactCounts[particleIndex];
			storage.contactRefs.resize(
				storage.contactStarts[numParticles]);

			// Reuse the two source-order contact passes required by the particle
			// CSR.  The rigid mirror is published only when it cuts at least half
			// the repeated body/contact visits; otherwise direct scans retain the
			// lower setup cost.  This keeps one-body dense dynamic contact on its
			// existing fast path.
			const PxU64 legacyRigidContactVisits =
				PxU64(numRigidBodies) * PxU64(storage.contacts.size());
			const bool publishRigidTargetContactPlan =
				mayBuildRigidTargetContactPlan &&
				PxU64(rigidTargetContactCount) * 2u <=
					legacyRigidContactVisits;
			if(publishRigidTargetContactPlan)
			{
				storage.rigidTargetContactStarts.resize(numRigidBodies + 1);
				storage.rigidTargetContactStarts[0] = 0;
				for(PxU32 rigidBodyIndex = 0;
					rigidBodyIndex < numRigidBodies; rigidBodyIndex++)
					storage.rigidTargetContactStarts[rigidBodyIndex + 1] =
						storage.rigidTargetContactStarts[rigidBodyIndex] +
						storage.rigidTargetContactCounts[rigidBodyIndex];
				storage.rigidTargetContactRefs.resize(
					storage.rigidTargetContactStarts[numRigidBodies]);
				for(PxU32 rigidBodyIndex = 0;
					rigidBodyIndex < numRigidBodies; rigidBodyIndex++)
					storage.rigidTargetContactCounts[rigidBodyIndex] = 0;
			}
			else
				storage.rigidTargetContactCounts.clear();
			for(PxU32 particleIndex = 0;
				particleIndex < numParticles; particleIndex++)
				storage.contactCounts[particleIndex] = 0;
			for(PxU32 contactIndex = 0;
				contactIndex < storage.contacts.size(); contactIndex++)
			{
				const Dy::AvbdSoftContactGeometry& geometry =
					storage.contacts[contactIndex].geometry;
				if(publishRigidTargetContactPlan &&
					geometry.hasRigidBodyTarget() &&
					geometry.targetIndex < numRigidBodies)
				{
					const PxU32 rigidBodyIndex = geometry.targetIndex;
					storage.rigidTargetContactRefs[
						storage.rigidTargetContactStarts[rigidBodyIndex] +
						storage.rigidTargetContactCounts[rigidBodyIndex]++] =
							contactIndex;
				}
				PxU32 particleIndices[
					Dy::AVBD_CONTACT_MAX_PARTICLES];
				const PxU32 particleIndexCount =
					Dy::avbdCollectSoftContactParticleIndices(
						geometry, particleIndices);
				for(PxU32 particleOffset = 0;
					particleOffset < particleIndexCount;
					particleOffset++)
				{
					const PxU32 particleIndex =
						particleIndices[particleOffset];
					if(particleIndex >= numParticles)
						continue;
					const PxReal jacobianScale =
						Dy::avbdGetSoftContactParticleJacobianScale(
							geometry, particleIndex);
					if(PxAbs(jacobianScale) <= 1e-12f)
						continue;
					storage.contactRefs[
						storage.contactStarts[particleIndex] +
						storage.contactCounts[particleIndex]++] =
							Dy::AvbdSoftContactParticleRef(
								contactIndex, jacobianScale);
				}
			}

			// Compile a second, geometry-only incidence program for triangle/OBB
			// core rows.  The normal contact CSR above deliberately contains only
			// the compact AL query support.  OGC admission, however, has to see the
			// independent embeddings of all three collision-triangle vertices or a
			// material update can move an unreferenced corner through a box.
			const PxU32 maxCoreSafetyParticles =
				3u * Dy::AVBD_CONTACT_POINT_MAX_SUPPORT;
			auto collectTriangleCoreSafetyParticles =
				[&](PxU32 contactIndex,
					const Dy::AvbdSoftContactGeometry& geometry,
					PxU32* particleIndices) -> PxU32
			{
				const Dy::AvbdOgcTriangleCoreCertificate* certificate =
					storage.ogcGeometrySidecar.getTriangleCore(contactIndex);
				if(!certificate ||
					(!geometry.hasRigidBodyTarget() &&
					 !geometry.hasWorldStaticTarget()))
					return 0;
				PxU32 count = 0;
				for(PxU32 vertex = 0; vertex < 3; ++vertex)
				{
					const Dy::AvbdWeightedContactPoint& point =
						certificate->points[vertex];
					if(point.count == 0 ||
						point.count > Dy::AVBD_CONTACT_POINT_MAX_SUPPORT)
						return PX_MAX_U32;
					for(PxU32 support = 0; support < point.count; ++support)
					{
						const PxU32 particleIndex =
							point.particleIndices[support];
						if(particleIndex >= numParticles)
							return PX_MAX_U32;
						bool duplicate = false;
						for(PxU32 prior = 0; prior < count; ++prior)
							duplicate |= particleIndices[prior] == particleIndex;
						if(!duplicate)
						{
							if(count >= maxCoreSafetyParticles)
								return PX_MAX_U32;
							particleIndices[count++] = particleIndex;
						}
					}
				}
				return count;
			};

			storage.triangleCoreSafetyStarts.resize(numParticles + 1);
			storage.triangleCoreSafetyCounts.resize(numParticles);
			for(PxU32 particleIndex = 0;
				particleIndex < numParticles; ++particleIndex)
				storage.triangleCoreSafetyCounts[particleIndex] = 0;
			for(PxU32 contactIndex = 0;
				contactIndex < storage.contacts.size(); ++contactIndex)
			{
				PxU32 coreParticles[3 * Dy::AVBD_CONTACT_POINT_MAX_SUPPORT];
				const PxU32 coreParticleCount =
					collectTriangleCoreSafetyParticles(
						contactIndex,
						storage.contacts[contactIndex].geometry, coreParticles);
				if(coreParticleCount == PX_MAX_U32)
					continue;
				for(PxU32 coreParticle = 0;
					coreParticle < coreParticleCount; ++coreParticle)
					storage.triangleCoreSafetyCounts[
						coreParticles[coreParticle]]++;
			}
			storage.triangleCoreSafetyStarts[0] = 0;
			for(PxU32 particleIndex = 0;
				particleIndex < numParticles; ++particleIndex)
				storage.triangleCoreSafetyStarts[particleIndex + 1] =
					storage.triangleCoreSafetyStarts[particleIndex] +
					storage.triangleCoreSafetyCounts[particleIndex];
			storage.triangleCoreSafetyRefs.resize(
				storage.triangleCoreSafetyStarts[numParticles]);
			for(PxU32 particleIndex = 0;
				particleIndex < numParticles; ++particleIndex)
				storage.triangleCoreSafetyCounts[particleIndex] = 0;
			for(PxU32 contactIndex = 0;
				contactIndex < storage.contacts.size(); ++contactIndex)
			{
				PxU32 coreParticles[3 * Dy::AVBD_CONTACT_POINT_MAX_SUPPORT];
				const PxU32 coreParticleCount =
					collectTriangleCoreSafetyParticles(
						contactIndex,
						storage.contacts[contactIndex].geometry, coreParticles);
				if(coreParticleCount == PX_MAX_U32)
					continue;
				for(PxU32 coreParticle = 0;
					coreParticle < coreParticleCount; ++coreParticle)
				{
					const PxU32 particleIndex = coreParticles[coreParticle];
					storage.triangleCoreSafetyRefs[
						storage.triangleCoreSafetyStarts[particleIndex] +
						storage.triangleCoreSafetyCounts[particleIndex]++] =
						Dy::AvbdSoftContactParticleRef(contactIndex, 1.0f);
				}
			}

			return true;
		}

} // namespace Sc
} // namespace physx
