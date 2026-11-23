// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.

#include "foundation/PxPreprocessor.h"

#if PX_SUPPORT_DEFORMABLE_VOLUME

#include "ScDeformableVolumeSim.h"
#include "geometry/PxTetrahedronMesh.h"
#include "foundation/PxBounds3.h"
#include "foundation/PxVec4.h"
#include "GuTetrahedronMesh.h"

using namespace physx;
using namespace Dy;

Sc::DeformableVolumeSim::DeformableVolumeSim(DeformableVolumeCore& core, Scene& scene) :
	GPUActorSim(scene, core, NULL)
{
	mLLDeformableVolume = scene.createLLDeformableVolume(this);

	mNodeIndex = scene.getSimpleIslandManager()->addNode(false, false, IG::Node::eDEFORMABLE_VOLUME_TYPE, mLLDeformableVolume);

	scene.getSimpleIslandManager()->activateNode(mNodeIndex);

	mLLDeformableVolume->setElementId(mShapeSim.getElementID());
#if !PX_SUPPORT_GPU_PHYSX
    auto& llCore=mLLDeformableVolume->getCore();auto& view=llCore.cpuGeometry;
    view.state=&mCpuNp;view.flags=&llCore.bodyFlags;view.wakeCounter=&llCore.wakeCounter;
    view.friction=&llCore.cpuDynamicFriction;view.selfFilterDistance=&llCore.selfCollisionFilterDistance;
    mScene.getLowLevelContext()->mCpuDeformables.pushBack(&view);
#endif
}

Sc::DeformableVolumeSim::~DeformableVolumeSim()
{
	if (!mLLDeformableVolume)
		return;

#if !PX_SUPPORT_GPU_PHYSX
    auto& views=mScene.getLowLevelContext()->mCpuDeformables;
    views.findAndReplaceWithLast(&mLLDeformableVolume->getCore().cpuGeometry);
    mLLDeformableVolume->getCore().cpuGeometry.state=NULL;
#endif
	mScene.destroyLLDeformableVolume(*mLLDeformableVolume);

	mScene.getSimpleIslandManager()->removeNode(mNodeIndex);

	mCore.setSim(NULL);
}

bool Sc::DeformableVolumeSim::isSleeping() const
{
	IG::IslandSim& sim = mScene.getSimpleIslandManager()->getAccurateIslandSim();
	return sim.getActiveNodeIndex(mNodeIndex) == PX_INVALID_NODE;
}

void Sc::DeformableVolumeSim::onSetWakeCounter()
{
	mScene.getSimulationController()->setSoftBodyWakeCounter(mLLDeformableVolume);
	if (mLLDeformableVolume->getCore().wakeCounter > 0.f)
		mScene.getSimpleIslandManager()->activateNode(mNodeIndex);
	else
		mScene.getSimpleIslandManager()->deactivateNode(mNodeIndex);
}

void Sc::DeformableVolumeSim::attachShapeCore(ShapeCore* core)
{
	mShapeSim.setCore(core);
	{
		PX_ASSERT(getWorldBounds().isFinite());

		const PxU32 index = mShapeSim.getElementID();

		mScene.getBoundsArray().setBounds(getWorldBounds(), index);

		addToAABBMgr(Bp::FilterType::DEFORMABLE_VOLUME);
	}

	mLLDeformableVolume->setShapeCore(core);
#if !PX_SUPPORT_GPU_PHYSX
    auto& view=mLLDeformableVolume->getCore().cpuGeometry;
    view.transformCache=mShapeSim.getTransformCacheID();view.contactOffset=mShapeSim.getContactOffset();
    const PxTetrahedronMesh* mesh=mLLDeformableVolume->getCollisionMesh();
    view.collision={mesh->getTetrahedrons(),mesh->getNbTetrahedrons(),mesh->getNbVertices(),4,
        PxU8(mesh->getTetrahedronMeshFlags()&PxTetrahedronMeshFlag::e16_BIT_INDICES?2:4)};
    if(view.state && view.simulation.valid())view.state->mesh.bake(view);
#endif
}

PxBounds3 Sc::DeformableVolumeSim::getWorldBounds(PxReal* maximumSpeed) const
{
    if(maximumSpeed)*maximumSpeed=0;
	if(mLLDeformableVolume)
	{
		const Dy::DeformableVolumeCore& core = mLLDeformableVolume->getCore();
		const PxTetrahedronMesh* simMesh = mLLDeformableVolume->getSimulationMesh();
		if(core.simPositionInvMass && simMesh)
		{
			const PxU32 n = simMesh->getNbVertices();
			PxBounds3 b = PxBounds3::empty();
			for(PxU32 i = 0; i < n; ++i)
			{
				const PxVec3 p = core.simPositionInvMass[i].getXYZ();
                if(maximumSpeed && core.simVelocity)*maximumSpeed=PxMax(*maximumSpeed,core.simVelocity[i].getXYZ().magnitude());
				if(p.isFinite())
					b.include(p);
			}
			if(b.isValid())
				return b;
		}
	}

	const PxTetrahedronMeshGeometry& tetGeom = static_cast<const PxTetrahedronMeshGeometry&>(mShapeSim.getCore().getGeometry());

	return tetGeom.tetrahedronMesh->getLocalBounds();
}

void Sc::DeformableVolumeSim::attachSimulationMesh(PxTetrahedronMesh* simulationMesh, PxDeformableVolumeAuxData* simulationState)
{
	mLLDeformableVolume->setSimShapeCore(simulationMesh, simulationState);
#if !PX_SUPPORT_GPU_PHYSX
    auto& llCore=mLLDeformableVolume->getCore();auto& view=llCore.cpuGeometry;
    view.positions=&llCore.simPositionInvMass;view.velocities=&llCore.simVelocity;view.restPositions=&llCore.restPosition;
    view.simulation=simulationMesh?PxsDeformableTopology{simulationMesh->getTetrahedrons(),
        simulationMesh->getNbTetrahedrons(),simulationMesh->getNbVertices(),4,
        PxU8(simulationMesh->getTetrahedronMeshFlags()&PxTetrahedronMeshFlag::e16_BIT_INDICES?2:4)}:PxsDeformableTopology{};
    const auto* aux=static_cast<const Gu::DeformableVolumeAuxData*>(simulationState);
    view.embeddingElements=aux?aux->mVertsRemapInGridModel:NULL;
    view.embeddingWeights=aux?aux->mVertsBarycentricInGridModel:NULL;
    if(view.state && view.collision.valid())view.state->mesh.bake(view);
#endif
}

PxTetrahedronMesh* Sc::DeformableVolumeSim::getSimulationMesh()
{
	return mLLDeformableVolume->getSimulationMesh();
}

PxDeformableVolumeAuxData* Sc::DeformableVolumeSim::getAuxData()
{
	return mLLDeformableVolume->getAuxData();
}

PxTetrahedronMesh* Sc::DeformableVolumeSim::getCollisionMesh()
{
	return mLLDeformableVolume->getCollisionMesh();
}

void Sc::DeformableVolumeSim::enableSelfCollision()
{
	if (isActive())
	{
		mScene.getSimulationController()->activateSoftbodySelfCollision(mLLDeformableVolume);
	}
}

void Sc::DeformableVolumeSim::disableSelfCollision()
{
	if (isActive())
	{
		mScene.getSimulationController()->deactivateSoftbodySelfCollision(mLLDeformableVolume);
	}
}

/*void Sc::DeformableVolumeSim::activate()
{
	// Activate body
	//{
	//	PX_ASSERT((!isKinematic()) || notInScene() || readInternalFlag(InternalFlags(BF_KINEMATIC_MOVED | BF_KINEMATIC_SURFACE_VELOCITY)));	// kinematics should only get activated when a target is set.
	//																																		// exception: object gets newly added, then the state change will happen later
	//	if (!isArticulationLink())
	//	{
	//		mLLBody.mInternalFlags &= (~PxsRigidBody::eFROZEN);
	//		// Put in list of activated bodies. The list gets cleared at the end of a sim step after the sleep callbacks have been fired.
	//		getScene().onBodyWakeUp(this);
	//	}

	//	BodyCore& core = getBodyCore();
	//	if (core.getFlags() & PxRigidBodyFlag::eENABLE_POSE_INTEGRATION_PREVIEW)
	//	{
	//		PX_ASSERT(!getScene().isInPosePreviewList(*this));
	//		getScene().addToPosePreviewList(*this);
	//	}
	//	createSqBounds();
	//}

	activateInteractions(*this);
}

void Sc::DeformableVolumeSim::deactivate()
{
	deactivateInteractions(*this);

	// Deactivate body
	//{
	//	PX_ASSERT((!isKinematic()) || notInScene() || !readInternalFlag(BF_KINEMATIC_MOVED));	// kinematics should only get deactivated when no target is set.
	//																							// exception: object gets newly added, then the state change will happen later
	//	BodyCore& core = getBodyCore();
	//	if (!readInternalFlag(BF_ON_DEATHROW))
	//	{
	//		// Set velocity to 0.
	//		// Note: this is also fine if the method gets called because the user puts something to sleep (this behavior is documented in the API)
	//		PX_ASSERT(core.getWakeCounter() == 0.0f);
	//		const PxVec3 zero(0.0f);
	//		core.setLinearVelocityInternal(zero);
	//		core.setAngularVelocityInternal(zero);

	//		setForcesToDefaults(!(mLLBody.mInternalFlags & PxsRigidBody::eDISABLE_GRAVITY));
	//	}

	//	if (!isArticulationLink())  // Articulations have their own sleep logic.
	//		getScene().onBodySleep(this);

	//	if (core.getFlags() & PxRigidBodyFlag::eENABLE_POSE_INTEGRATION_PREVIEW)
	//	{
	//		PX_ASSERT(getScene().isInPosePreviewList(*this));
	//		getScene().removeFromPosePreviewList(*this);
	//	}
	//	destroySqBounds();
	//}
}*/

PxU32 Sc::DeformableVolumeSim::getGpuIndex() const
{
	return mLLDeformableVolume->getGpuIndex();
}

#endif //PX_SUPPORT_DEFORMABLE_VOLUME
