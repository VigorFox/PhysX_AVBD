#ifndef SNIPPET_DEFORMABLE_VOLUME_AVBD_H
#define SNIPPET_DEFORMABLE_VOLUME_AVBD_H

#include "PxPhysicsAPI.h"

extern physx::PxScene* gVolumeAvbdScene;
void reportVolumeAvbdAllocations(const char* phase);
void sendVolumeAvbdPvdMesh();
physx::PxU32 getVolumeAvbdRenderCount();
physx::PxDeformableVolume* getVolumeAvbdRenderVolume(physx::PxU32 index);
void createVolumeAvbdSphereMesh(physx::PxArray<physx::PxVec3>& vertices,
	physx::PxArray<physx::PxU32>& triangles, physx::PxReal radius,
	physx::PxReal maxEdgeLength);

// Optional headless input provider; the visual scene and stepping stay shared.
typedef physx::PxDeformableVolumeMesh* (*VolumeAvbdMeshFactory)(physx::PxPhysics&,
	const physx::PxArray<physx::PxVec3>&, const physx::PxArray<physx::PxU32>&,
	physx::PxU32, const char*);
bool initPhysics(bool interactive, physx::PxU32 dispatcherThreads = 4,
	VolumeAvbdMeshFactory meshFactory = NULL);
void stepPhysics(bool interactive);
void cleanupPhysics(bool interactive);
void keyPress(unsigned char key, const physx::PxTransform& camera);

#endif
