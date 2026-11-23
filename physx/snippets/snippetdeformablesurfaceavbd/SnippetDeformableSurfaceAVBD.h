#ifndef SNIPPET_DEFORMABLE_SURFACE_AVBD_H
#define SNIPPET_DEFORMABLE_SURFACE_AVBD_H

#include "PxPhysicsAPI.h"

extern physx::PxScene* gSurfaceAvbdScene;
extern physx::PxDeformableSurface* gSurfaceAvbdRenderSurface;
extern physx::PxTriangleMesh* gSurfaceAvbdRenderMesh;

bool initPhysics(bool interactive);
void stepPhysics(bool interactive);
void cleanupPhysics(bool interactive);
void keyPress(unsigned char key, const physx::PxTransform& camera);

#endif
