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

#if PX_SUPPORT_DEFORMABLE_SURFACE

#include "ScDeformableSurfaceSim.h"

#include "geometry/PxTriangleMesh.h"

using namespace physx;
using namespace Dy;

Sc::DeformableSurfaceSim::DeformableSurfaceSim(DeformableSurfaceCore& core, Scene& scene) :
	GPUActorSim(scene, core, NULL)
{
	mLLDeformableSurface = scene.createLLDeformableSurface(this);

	mNodeIndex = scene.getSimpleIslandManager()->addNode(false, false, IG::Node::eDEFORMABLE_SURFACE_TYPE, mLLDeformableSurface);

	scene.getSimpleIslandManager()->activateNode(mNodeIndex);

	mLLDeformableSurface->setElementId(mShapeSim.getElementID());
#if !PX_SUPPORT_GPU_PHYSX
    auto& llCore=mLLDeformableSurface->getCore();auto& view=llCore.cpuGeometry;
    view.state=&mCpuNp;view.flags=&llCore.bodyFlags;view.wakeCounter=&llCore.wakeCounter;
    view.friction=&llCore.cpuDynamicFriction;view.selfFilterDistance=&llCore.selfCollisionFilterDistance;
    mScene.getLowLevelContext()->mCpuDeformables.pushBack(&view);
#endif
}

Sc::DeformableSurfaceSim::~DeformableSurfaceSim()
{
	if (!mLLDeformableSurface)
		return;

#if !PX_SUPPORT_GPU_PHYSX
    auto& views=mScene.getLowLevelContext()->mCpuDeformables;
    views.findAndReplaceWithLast(&mLLDeformableSurface->getCore().cpuGeometry);
    mLLDeformableSurface->getCore().cpuGeometry.state=NULL;
#endif
	mScene.destroyLLDeformableSurface(*mLLDeformableSurface);

	mScene.getSimpleIslandManager()->removeNode(mNodeIndex);

	mCore.setSim(NULL);
}

bool Sc::DeformableSurfaceSim::isSleeping() const
{
	IG::IslandSim& sim = mScene.getSimpleIslandManager()->getAccurateIslandSim();
	return sim.getActiveNodeIndex(mNodeIndex) == PX_INVALID_NODE;
}

void Sc::DeformableSurfaceSim::onSetWakeCounter()
{
#if PX_SUPPORT_GPU_PHYSX
	mScene.getSimulationController()->setClothWakeCounter(mLLDeformableSurface);
#endif
	if (mLLDeformableSurface->getCore().wakeCounter > 0.f)
		mScene.getSimpleIslandManager()->activateNode(mNodeIndex);
	else
		mScene.getSimpleIslandManager()->deactivateNode(mNodeIndex);
}

void Sc::DeformableSurfaceSim::attachShapeCore(ShapeCore* core)
{
	mShapeSim.setCore(core);
	// Bounds reads the mesh through the low-level Surface.  Publish the Host
	// shape fact before the first bounds evaluation.
	mLLDeformableSurface->setShapeCore(core);
#if !PX_SUPPORT_GPU_PHYSX
    auto& llCore=mLLDeformableSurface->getCore();auto& view=llCore.cpuGeometry;
    view.transformCache=mShapeSim.getTransformCacheID();view.contactOffset=mShapeSim.getContactOffset();
    const PxTriangleMesh* mesh=mLLDeformableSurface->getTriangleMesh();
    view.positions=&llCore.positionInvMass;view.velocities=&llCore.velocity;view.restPositions=&llCore.restPosition;
    view.collision=view.simulation={mesh->getTriangles(),mesh->getNbTriangles(),mesh->getNbVertices(),3,
        PxU8(mesh->getTriangleMeshFlags()&PxTriangleMeshFlag::e16_BIT_INDICES?2:4)};
    if(view.state)view.state->mesh.bake(view);
#endif
	{
		PX_ASSERT(getWorldBounds().isFinite());

		const PxU32 index = mShapeSim.getElementID();

		mScene.getBoundsArray().setBounds(getWorldBounds(), index);

		addToAABBMgr(Bp::FilterType::DEFORMABLE_SURFACE);
	}

}

PxBounds3 Sc::DeformableSurfaceSim::getWorldBounds(PxReal* maximumSpeed) const
{
    if(maximumSpeed)*maximumSpeed=0;
	if(mLLDeformableSurface)
	{
		const Dy::DeformableSurfaceCore& core = mLLDeformableSurface->getCore();
		PxTriangleMesh* mesh = mLLDeformableSurface->getTriangleMesh();
		if(core.positionInvMass && mesh)
		{
			PxBounds3 bounds = PxBounds3::empty();
			for(PxU32 i = 0; i < mesh->getNbVertices(); ++i)
			{
				const PxVec3 p = core.positionInvMass[i].getXYZ();
                if(maximumSpeed && core.velocity)*maximumSpeed=PxMax(*maximumSpeed,core.velocity[i].getXYZ().magnitude());
				if(p.isFinite())
					bounds.include(p);
			}
			if(bounds.isValid())
				return bounds;
		}
	}
	const PxTriangleMeshGeometry& triGeom = static_cast<const PxTriangleMeshGeometry&>(mShapeSim.getCore().getGeometry());

	// PT: are you sure you want to go through the Px API here?
	return triGeom.triangleMesh->getLocalBounds();
}

#endif //PX_SUPPORT_DEFORMABLE_SURFACE
