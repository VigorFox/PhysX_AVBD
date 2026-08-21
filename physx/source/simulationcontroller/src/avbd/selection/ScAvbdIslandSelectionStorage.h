// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SC_AVBD_ISLAND_SELECTION_STORAGE_H
#define SC_AVBD_ISLAND_SELECTION_STORAGE_H

#include "ScActorCore.h"
#include "ScBodyCore.h"
#include "avbd/pipeline/DyAvbdDynamics.h"
#include "avbd/solver/soft/DyAvbdSoftBodyComponent.h"
#include "avbd/ogc/DyAvbdOgcGeometryEpoch.h"
#include "avbd/solver/post_al/DyAvbdPostAl.h"

namespace physx
{
namespace Sc
{

// Scene-owned native-island workspace. Selection policy and lifetime remain
// in AvbdCpuSoftScene; this value type is the stable hand-off for the Scene
// facade and future task-owned selection compiler.
struct AvbdIslandSelectionStorage
{
	IG::IslandId nativeIslandId;
	bool touched;
	PxU32 selectedIsland;
	PxArray<PxU32> entryIndices;
	PxArray<ActorCore*> softCores;
	bool usesCanonicalParticleRange;
	PxU32 canonicalParticleStart;
	PxU32 canonicalParticleCount;
	PxArray<PxU32> globalParticleIndices;
	PxArray<Dy::AvbdSoftParticle> particles;
	PxArray<Dy::AvbdSoftBody> bodies;
	PxArray<PxU32> particleBodyIndices;
	PxArray<PxU32> contactStarts;
	PxArray<PxU32> contactCounts;
	PxArray<Dy::AvbdSoftContactParticleRef> contactRefs;
	PxArray<PxU32> triangleCoreSafetyStarts;
	PxArray<PxU32> triangleCoreSafetyCounts;
	PxArray<Dy::AvbdSoftContactParticleRef> triangleCoreSafetyRefs;
	PxArray<PxU32> rigidTargetContactStarts;
	PxArray<PxU32> rigidTargetContactCounts;
	PxArray<PxU32> rigidTargetContactRefs;
	PxArray<Dy::AvbdSelfCollisionAdjacency> selfCollisionAdjacencies;
	PxArray<PxU8> selfCollisionEnabled;
	PxArray<Dy::AvbdRigidBox> rigidBoxes;
	PxArray<Dy::AvbdRigidBox> selectedDynamicBoxes;
	PxArray<Dy::AvbdSoftBody> terminalCollisionBodies;
	PxArray<Dy::AvbdWeightedContactPoint> terminalCollisionVertexMappings;
	PxArray<Dy::AvbdOgcPairState> ogcPairStates;
	Dy::AvbdOgcGeometryEpochSidecar ogcGeometrySidecar;
	PxArray<PxU32> ogcPairIndices;
	PxArray<PxU32> ogcPairContactStarts;
	PxArray<PxU32> ogcPairContactCounts;
	PxArray<PxU32> ogcPairContactRefs;
	PxArray<Dy::AvbdRigidSphere> rigidSpheres;
	PxArray<Dy::AvbdRigidSphere> selectedDynamicSpheres;
	PxArray<Dy::AvbdRigidCapsule> rigidCapsules;
	PxArray<Dy::AvbdRigidCapsule> selectedDynamicCapsules;
	PxArray<Dy::AvbdRigidConvex> rigidConvexes;
	PxArray<Dy::AvbdRigidConvex> selectedDynamicConvexes;
	PxArray<Dy::AvbdSoftContact> contacts;
	PxArray<Dy::AvbdSoftContact> probeContacts;
	Dy::AvbdPostAlWorkspace postAlWorkspace;

	AvbdIslandSelectionStorage()
		: nativeIslandId(IG_INVALID_ISLAND), touched(false),
		  selectedIsland(PX_MAX_U32),
		  usesCanonicalParticleRange(false),
		  canonicalParticleStart(PX_MAX_U32),
		  canonicalParticleCount(0)
	{
	}
};

// Persistent storage pool used by the Scene facade.  The pool owns storage
// lifetime and reuse; island selection policy remains with AvbdCpuSoftScene.
class AvbdIslandSelectionStoragePool
{
public:
	~AvbdIslandSelectionStoragePool()
	{
		release();
	}

	PxU32 size() const
	{
		return mStorages.size();
	}

	AvbdIslandSelectionStorage*& operator[](PxU32 index)
	{
		return mStorages[index];
	}

	AvbdIslandSelectionStorage* const& operator[](PxU32 index) const
	{
		return mStorages[index];
	}

	void pushBack(AvbdIslandSelectionStorage* storage)
	{
		mStorages.pushBack(storage);
	}

	// Matches the legacy caller contract: release() destroys objects, while
	// clear() only drops already-destroyed pointers after explicit teardown.
	void clear()
	{
		mStorages.clear();
	}

	void release()
	{
		for(PxU32 i = 0; i < mStorages.size(); ++i)
		{
			AvbdIslandSelectionStorage* storage = mStorages[i];
			if(storage)
			{
				storage->~AvbdIslandSelectionStorage();
				PX_FREE(storage);
			}
		}
		mStorages.clear();
	}

	void invalidate()
	{
		for(PxU32 i = 0; i < mStorages.size(); ++i)
		{
			AvbdIslandSelectionStorage* const storage = mStorages[i];
			if(!storage)
				continue;
			storage->contacts.clear();
			storage->particleBodyIndices.clear();
			storage->contactStarts.clear();
			storage->contactCounts.clear();
			storage->contactRefs.clear();
			storage->triangleCoreSafetyStarts.clear();
			storage->triangleCoreSafetyCounts.clear();
			storage->triangleCoreSafetyRefs.clear();
			storage->ogcPairContactStarts.clear();
			storage->ogcPairContactCounts.clear();
			storage->ogcPairContactRefs.clear();
			storage->ogcGeometrySidecar.clear();
			storage->rigidTargetContactStarts.clear();
			storage->rigidTargetContactCounts.clear();
			storage->rigidTargetContactRefs.clear();
			storage->softCores.clear();
			storage->touched = false;
			storage->selectedIsland = PX_MAX_U32;
			storage->entryIndices.clear();
		}
	}

	AvbdIslandSelectionStorage* acquire(IG::IslandId nativeIslandId)
	{
		for(PxU32 i = 0; i < mStorages.size(); ++i)
		{
			AvbdIslandSelectionStorage* const storage = mStorages[i];
			if(storage->touched && storage->nativeIslandId == nativeIslandId)
				return storage;
		}
		for(PxU32 i = 0; i < mStorages.size(); ++i)
		{
			AvbdIslandSelectionStorage* const storage = mStorages[i];
			if(!storage->touched)
			{
				if(storage->nativeIslandId != nativeIslandId)
				{
					storage->contacts.clear();
					storage->ogcGeometrySidecar.clear();
					storage->softCores.clear();
				}
				storage->nativeIslandId = nativeIslandId;
				storage->touched = true;
				return storage;
			}
		}

		void* memory = PX_ALLOC(
			sizeof(AvbdIslandSelectionStorage),
			"AVBD CPU soft island selection storage");
		AvbdIslandSelectionStorage* const storage = memory
			? PX_PLACEMENT_NEW(memory, AvbdIslandSelectionStorage)()
			: NULL;
		if(!storage)
			return NULL;
		storage->nativeIslandId = nativeIslandId;
		storage->touched = true;
		mStorages.pushBack(storage);
		return storage;
	}

private:
	PxArray<AvbdIslandSelectionStorage*> mStorages;
};

} // namespace Sc
} // namespace physx

#endif // SC_AVBD_ISLAND_SELECTION_STORAGE_H
