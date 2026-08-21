// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include "ScAvbdCpuSoftScene.h"

namespace physx
{
namespace Sc
{

		void AvbdCpuSoftScene::refreshSurfaceFlattening(Entry& entry)
		{
			if(entry.kind != eSURFACE || !entry.surfaceCore ||
				entry.bodyIndex >= mBodies.size())
				return;
			const bool flatteningEnabled =
				(entry.surfaceCore->getSurfaceFlags() &
					PxDeformableSurfaceFlag::eENABLE_FLATTENING)
				? true : false;
			mBodies[entry.bodyIndex].compiled.
				compileBendingRestAngles(flatteningEnabled);
		}

		void AvbdCpuSoftScene::applyDeformablePreintegrationControls(
			Entry& entry)
		{
			const PxReal maxLinearVelocity =
				PxMax(entry.getBodyCore().maxLinearVelocity, 0.0f);
			const PxU32 particleStart = getParticleStart(entry);
			const PxU32 particleCount = getParticleCount(entry);
			for(PxU32 i = 0; i < particleCount; i++)
			{
				Dy::AvbdSoftParticle& particle =
					mParticles[particleStart + i];
				if(particle.invMass <= 0.0f ||
					!particle.velocity.isFinite())
					continue;
				const PxReal maxComponent = PxMax(
					PxAbs(particle.velocity.x),
					PxMax(
						PxAbs(particle.velocity.y),
						PxAbs(particle.velocity.z)));
				if(maxComponent == 0.0f)
					continue;
				const PxVec3 scaledVelocity =
					particle.velocity / maxComponent;
				const PxReal scaledMagnitude =
					scaledVelocity.magnitude();
				const PxReal limitedMaxComponent =
					maxLinearVelocity / scaledMagnitude;
				if(maxComponent <= limitedMaxComponent)
					continue;
				particle.velocity =
					scaledVelocity * limitedMaxComponent;
				particle.prevVelocity = particle.velocity;
			}
		}

		void AvbdCpuSoftScene::syncHostInputs(
			Entry& entry,
			const PxsDeformableVolumeMaterialManager& materialManager)
		{
			Dy::DeformableBodyCore& bodyCore =
				entry.getBodyCore();
			bool syncPositions = false;
			bool syncVelocities = false;
			bool syncRestPositions = false;
			if(entry.kind == eVOLUME)
			{
				const Dy::DeformableVolumeCore& core =
					entry.volumeCore->getCore();
				syncPositions = core.dirtyFlags &
					PxDeformableVolumeDataFlag::
						eSIM_POSITION_INVMASS;
				syncVelocities = core.dirtyFlags &
					PxDeformableVolumeDataFlag::eSIM_VELOCITY;
			}
			else
			{
				const Dy::DeformableSurfaceCore& core =
					entry.surfaceCore->getCore();
				syncPositions = core.dirtyFlags &
					PxDeformableSurfaceDataFlag::ePOSITION_INVMASS;
				syncVelocities = core.dirtyFlags &
					PxDeformableSurfaceDataFlag::eVELOCITY;
				syncRestPositions = core.dirtyFlags &
					PxDeformableSurfaceDataFlag::eREST_POSITION;
				if(syncRestPositions)
				{
					const bool rebuilt =
						rebuildSurfaceRestState(entry);
					PX_ASSERT(rebuilt);
					PX_UNUSED(rebuilt);
				}
			}

			Dy::AvbdSoftBody& body = mBodies[entry.bodyIndex];
			body.compiled.selfCollisionFilterDistance =
				PxMax(bodyCore.selfCollisionFilterDistance, 0.0f);
			// Rest position changes rebuild the compiled body above. A filter
			// distance change keeps topology intact, but must refresh the bounded
			// immutable rest-space exclusion cache before the next OGC query.
			body.compiled.ensureSelfCollisionRestVertexTriangleFilter();
			body.compiled.maxDepenetrationVelocity =
				PxMax(-bodyCore.maxPenetrationBias, 0.0f);
			body.compiled.selfCollisionStressTolerance =
				bodyCore.selfCollisionStressTolerance;
			body.compiled.speculativeCCDEnabled =
				bodyCore.bodyFlags.isSet(
					PxDeformableBodyFlag::
						eENABLE_SPECULATIVE_CCD);
			refreshSurfaceFlattening(entry);
			if(entry.kind == eVOLUME)
			{
				const PxsDeformableVolumeMaterialCore* material =
					getMaterial(*entry.volumeCore, materialManager);
				body.material.youngsModulus =
					material ? material->youngs : 1.0e5f;
				body.material.poissonsRatio =
					material ? material->poissons : 0.3f;
				body.material.damping =
					bodyCore.linearDamping +
					(material ? material->elasticityDamping : 0.0f);
				body.material.bendingStiffness = 0.0f;
				body.material.bendingDamping = 0.0f;
				body.material.thickness = 0.01f;
				body.material.dynamicFriction = material
					? PxMax(material->dynamicFriction, 0.0f)
					: 0.5f;
				const bool coRotationalVolumeModel =
					!material ||
					material->materialModel ==
						PxDeformableVolumeMaterialModel::
							eCO_ROTATIONAL;
				if(body.material.coRotationalVolumeModel !=
					coRotationalVolumeModel)
				{
					body.material.coRotationalVolumeModel =
						coRotationalVolumeModel;
					// Incidence packets are topology-only.  A material switch
					// changes the evaluator selected at the solve boundary, not
					// the canonical packet program itself.
					body.compiled.buildTetIncidencePacketProgram();
				}
			}
			else
			{
				const PxsDeformableSurfaceMaterialCore* material =
					getSurfaceMaterial(*entry.surfaceCore);
				body.material.youngsModulus =
					material ? material->youngs : 1.0e5f;
				body.material.poissonsRatio =
					material ? material->poissons : 0.3f;
				body.material.damping =
					bodyCore.linearDamping +
					(material ? material->elasticityDamping : 0.0f);
				body.material.bendingStiffness =
					material ? material->bendingStiffness : 0.0f;
				body.material.bendingDamping =
					material ? material->bendingDamping : 0.0f;
				body.material.thickness = material
					? PxMax(material->thickness, 1.0e-4f)
					: 0.01f;
				body.material.dynamicFriction = material
					? PxMax(material->dynamicFriction, 0.0f)
					: 0.5f;
				body.material.coRotationalVolumeModel = true;
			}
			body.material.computeLameParameters();
			const PxU32 particleStart = getParticleStart(entry);
			const PxU32 particleCount = getParticleCount(entry);
			const PxReal gravityScale =
				(entry.getActorFlags() &
					PxActorFlag::eDISABLE_GRAVITY)
				? 0.0f : 1.0f;
			for(PxU32 i = 0; i < particleCount; i++)
				mParticles[particleStart + i].gravityScale =
					gravityScale;

			PxVec4* positions = entry.getPositionInvMass();
			PxVec4* velocities = entry.getVelocity();
			PX_ASSERT(positions && velocities);
			if(syncPositions || syncVelocities || bodyCore.dirty)
			{
				for(PxU32 i = 0; i < particleCount; i++)
				{
					Dy::AvbdSoftParticle& particle =
						mParticles[particleStart + i];
					if(syncPositions)
					{
						const PxVec4& positionInvMass =
							positions[i];
						particle.position =
							positionInvMass.getXYZ();
						if(entry.kind == eVOLUME)
							particle.initialPosition =
								particle.position;
						particle.predictedPosition = particle.position;
						particle.outerPosition = particle.position;
						particle.invMass =
							PxMax(positionInvMass.w, 0.0f);
						particle.mass = particle.invMass > 0.0f
							? 1.0f / particle.invMass : 0.0f;
						particle.elasticK = 0.0f;
					}
					if(syncVelocities)
					{
						particle.velocity =
							velocities[i].getXYZ();
						particle.prevVelocity = particle.velocity;
					}
					particle.damping = body.material.damping;
				}
			}
			applyDeformablePreintegrationControls(entry);
			bodyCore.dirty = false;
			if(entry.kind == eVOLUME)
				entry.volumeCore->getCore().dirtyFlags =
					PxDeformableVolumeDataFlags(0);
			else
				entry.surfaceCore->getCore().dirtyFlags =
					PxDeformableSurfaceDataFlags(0);
		}

		void AvbdCpuSoftScene::writeBack(Entry& entry)
		{
			if(entry.sleeping)
				return;
			const PxU32 particleStart = getParticleStart(entry);
			const PxU32 particleCount = getParticleCount(entry);
			if(entry.kind == eSURFACE)
			{
				Dy::DeformableSurfaceCore& core =
					entry.surfaceCore->getCore();
				for(PxU32 i = 0; i < particleCount; i++)
				{
					const Dy::AvbdSoftParticle& particle =
						mParticles[particleStart + i];
					core.positionInvMass[i] =
						PxVec4(particle.position, particle.invMass);
					const PxReal velocityW = core.velocity[i].w;
					core.velocity[i] =
						PxVec4(particle.velocity, velocityW);
				}
				return;
			}

			Dy::DeformableVolumeCore& core =
				entry.volumeCore->getCore();
			for(PxU32 i = 0; i < particleCount; i++)
			{
				const Dy::AvbdSoftParticle& particle =
					mParticles[particleStart + i];
				core.simPositionInvMass[i] =
					PxVec4(particle.position, particle.invMass);
				core.simVelocity[i] =
					PxVec4(particle.velocity, particle.invMass);
			}

			const PxU32 collisionVertexCount =
				entry.collisionMesh->getNbVertices();
			if(entry.collisionMesh == entry.simulationMesh &&
				collisionVertexCount == particleCount)
			{
				for(PxU32 i = 0; i < collisionVertexCount; i++)
				{
					const PxReal invMass =
						core.positionInvMass[i].w;
					core.positionInvMass[i] = PxVec4(
						mParticles[particleStart + i].position,
						invMass);
				}
				return;
			}

			Gu::DeformableVolumeAuxData& auxData =
				static_cast<Gu::DeformableVolumeAuxData&>(
					*entry.auxData);
			const PxU32* remap =
				auxData.mVertsRemapInGridModel;
			const PxReal* barycentrics =
				auxData.mVertsBarycentricInGridModel;
			if(!remap || !barycentrics)
			{
				// Distinct meshes are rejected at registration when cooked mapping
				// is absent. Never reinterpret the first N simulation vertices as
				// collision vertices here.
				PX_ASSERT(false);
				return;
			}

			const bool has16BitIndices =
				entry.simulationMesh->getTetrahedronMeshFlags() &
				PxTetrahedronMeshFlag::e16_BIT_INDICES;
			const PxU16* tets16 = has16BitIndices
				? static_cast<const PxU16*>(
					entry.simulationMesh->getTetrahedrons()) : NULL;
			const PxU32* tets32 = has16BitIndices ? NULL
				: static_cast<const PxU32*>(
					entry.simulationMesh->getTetrahedrons());
			for(PxU32 i = 0; i < collisionVertexCount; i++)
			{
				const PxU32 tetIndex = remap[i];
				if(tetIndex >= entry.simulationMesh->getNbTetrahedrons())
				{
					PX_ASSERT(false);
					return;
				}
				PxVec3 position(0.0f);
				for(PxU32 j = 0; j < 4; j++)
				{
					const PxU32 localParticle = has16BitIndices
						? tets16[4 * tetIndex + j]
						: tets32[4 * tetIndex + j];
					position +=
						mParticles[
							particleStart + localParticle].position *
						barycentrics[4 * i + j];
				}
				const PxReal invMass = core.positionInvMass[i].w;
				core.positionInvMass[i] =
					PxVec4(position, invMass);
			}
		}

} // namespace Sc
} // namespace physx
