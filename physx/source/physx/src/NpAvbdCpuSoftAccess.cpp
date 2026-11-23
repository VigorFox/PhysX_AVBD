// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Smoke/host diagnostics. Deformables are created and attached through the
// public Host actor path; this file intentionally exposes no simulation sidecar.

#include "NpScene.h"
#include "ScScene.h"
#include "DyAvbdCpuSoftAccess.h"
#include "common/PxPhysXCommonConfig.h"
#include "PxsSimpleIslandManager.h"

using namespace physx;
using namespace Dy;

namespace
{
Context* tryDynamicsContext(PxScene* scene)
{
	if(!scene)
		return NULL;
	return static_cast<NpScene*>(scene)->getScScene().getDynamicsContext();
}
} // namespace

extern "C"
{

// P4a: count active IG eDEFORMABLE_VOLUME nodes whose object is Dy::DeformableVolume*.
PX_PHYSX_CORE_API PxU32 avbdHostActiveDeformableVolumeNodeCount(PxScene* scene)
{
	if(!scene)
		return 0;
	Sc::Scene& sc = static_cast<NpScene*>(scene)->getScScene();
	const IG::IslandSim& islandSim = sc.getSimpleIslandManager()->getAccurateIslandSim();
	const PxU32 nb = islandSim.getNbActiveNodes(IG::Node::eDEFORMABLE_VOLUME_TYPE);
	const PxNodeIndex* nodes = islandSim.getActiveNodes(IG::Node::eDEFORMABLE_VOLUME_TYPE);
	PxU32 count = 0;
	for(PxU32 i = 0; i < nb; ++i)
	{
		const IG::Node& node = islandSim.getNode(nodes[i]);
		if(node.mObject)
			++count;
	}
	return count;
}

PX_PHYSX_CORE_API bool avbdHostKernelTableInfo(PxScene* scene, PxU32* deviceKind,
	bool* ldlt6Bound, bool* ldlt3Bound)
{
	PxU32 device = PX_MAX_U32;
	bool a = false;
	bool b = false;
	if(!avbdKernelTableInfo(tryDynamicsContext(scene), device, a, b))
		return false;
	if(deviceKind)
		*deviceKind = device;
	if(ldlt6Bound)
		*ldlt6Bound = a;
	if(ldlt3Bound)
		*ldlt3Bound = b;
	return true;
}

PX_PHYSX_CORE_API bool avbdHostKernelLdltSelfCheck()
{
	return avbdKernelLdltSelfCheck();
}

}
