#ifndef SNIPPET_DEFORMABLE_VOLUME_AVBD_COMMON_H
#define SNIPPET_DEFORMABLE_VOLUME_AVBD_COMMON_H
#include "PxPhysicsAPI.h"
#include "cooking/PxCooking.h"
#include "extensions/PxDeformableVolumeExt.h"

namespace VolumeAvbdSnippet {
using namespace physx;
inline void hostFree(void* p){if(p)PxGetFoundation().getAllocatorCallback().deallocate(p);}
inline PxDeformableVolumeMesh* cookCube(PxPhysics& physics,const PxVec3& center,PxReal size){
	PxArray<PxVec3> v;const PxReal h=size*0.5f;for(PxU32 i=0;i<8;++i)v.pushBack(center+PxVec3((i&1)?h:-h,(i&2)?h:-h,(i&4)?h:-h));
	static const PxU32 cells[5][4]={{0,1,2,4},{1,3,2,7},{1,4,5,7},{2,4,6,7},{1,2,4,7}};PxArray<PxU32> t;for(PxU32 i=0;i<5;++i)for(PxU32 j=0;j<4;++j)t.pushBack(cells[i][j]);
	PxTetrahedronMeshDesc md(v,t);PxDeformableVolumeSimulationDataDesc sd;PxCookingParams p(physics.getTolerancesScale());return PxCreateDeformableVolumeMesh(p,md,md,sd,physics.getPhysicsInsertionCallback());
}
inline PxDeformableVolume* createCube(PxPhysics& physics,PxScene& scene,PxDeformableVolumeMesh*& mesh,const PxVec3& center,PxReal size,PxReal density=100.0f){
	mesh=cookCube(physics,center,size);if(!mesh)return NULL;PxDeformableVolume* volume=physics.createDeformableVolume();if(!volume)return NULL;
	PxDeformableVolumeMaterial* mat=physics.createDeformableVolumeMaterial(2.e5f,0.3f,0.1f);PxTetrahedronMeshGeometry geometry(mesh->getCollisionMesh());PxShape* shape=physics.createShape(geometry,&mat,1,true,PxShapeFlag::eVISUALIZATION|PxShapeFlag::eSCENE_QUERY_SHAPE|PxShapeFlag::eSIMULATION_SHAPE);
	const bool attached=shape&&volume->attachShape(*shape)&&volume->attachSimulationMesh(*mesh->getSimulationMesh(),*mesh->getDeformableVolumeAuxData());PX_RELEASE(shape);PX_RELEASE(mat);if(!attached){volume->release();return NULL;}
	PxVec4 *sp=NULL,*sv=NULL,*cp=NULL,*rp=NULL;PxDeformableVolumeExt::allocateAndInitializeHostMirror(*volume,NULL,sp,sv,cp,rp);PxDeformableVolumeExt::updateMass(*volume,density,50.0f,sp);PxDeformableVolumeExt::copyToDevice(*volume,PxDeformableVolumeDataFlag::eALL,sp,sv,cp,rp);hostFree(sp);hostFree(sv);hostFree(cp);hostFree(rp);
	if(!scene.addActor(*volume)){volume->release();return NULL;}return volume;
}
inline void bounds(PxDeformableVolume& v,PxReal& minY,PxReal& maxY,PxVec3& com,bool& finite){const PxU32 n=v.getSimulationMesh()->getNbVertices();const PxVec4* p=v.getSimPositionInvMassBufferD();minY=PX_MAX_F32;maxY=-PX_MAX_F32;com=PxVec3(0);finite=p!=NULL&&n>0;for(PxU32 i=0;i<n&&finite;++i){const PxVec3 q=p[i].getXYZ();finite=q.isFinite()&&PxIsFinite(p[i].w);minY=PxMin(minY,q.y);maxY=PxMax(maxY,q.y);com+=q;}if(n)com/=PxReal(n);}

inline void appendSubdividedQuad(PxArray<PxVec3>& vertices,
	PxArray<PxU32>& triangles, const PxVec3& origin,
	const PxVec3& axisU, const PxVec3& axisV, PxU32 subdivisions)
{
	const PxU32 base = vertices.size();
	for(PxU32 v = 0; v <= subdivisions; ++v)
		for(PxU32 u = 0; u <= subdivisions; ++u)
			vertices.pushBack(origin + axisU * (PxReal(u) / PxReal(subdivisions)) +
				axisV * (PxReal(v) / PxReal(subdivisions)));
	const PxU32 row = subdivisions + 1;
	for(PxU32 v = 0; v < subdivisions; ++v)
		for(PxU32 u = 0; u < subdivisions; ++u)
		{
			const PxU32 i00 = base + v * row + u;
			const PxU32 i10 = i00 + 1;
			const PxU32 i01 = i00 + row;
			const PxU32 i11 = i01 + 1;
			triangles.pushBack(i00); triangles.pushBack(i10); triangles.pushBack(i11);
			triangles.pushBack(i00); triangles.pushBack(i11); triangles.pushBack(i01);
		}
}

inline void createSubdividedCubeSurface(PxArray<PxVec3>& vertices,
	PxArray<PxU32>& triangles, PxReal sideLength, PxU32 subdivisions)
{
	vertices.clear();
	triangles.clear();
	const PxReal h = 0.5f * sideLength;
	appendSubdividedQuad(vertices, triangles, PxVec3(h,-h,-h),
		PxVec3(0,sideLength,0), PxVec3(0,0,sideLength), subdivisions);
	appendSubdividedQuad(vertices, triangles, PxVec3(-h,-h,-h),
		PxVec3(0,0,sideLength), PxVec3(0,sideLength,0), subdivisions);
	appendSubdividedQuad(vertices, triangles, PxVec3(-h,h,-h),
		PxVec3(0,0,sideLength), PxVec3(sideLength,0,0), subdivisions);
	appendSubdividedQuad(vertices, triangles, PxVec3(-h,-h,-h),
		PxVec3(sideLength,0,0), PxVec3(0,0,sideLength), subdivisions);
	appendSubdividedQuad(vertices, triangles, PxVec3(-h,-h,h),
		PxVec3(sideLength,0,0), PxVec3(0,sideLength,0), subdivisions);
	appendSubdividedQuad(vertices, triangles, PxVec3(-h,-h,-h),
		PxVec3(0,sideLength,0), PxVec3(sideLength,0,0), subdivisions);
}

inline void createLayeredConeSurface(PxArray<PxVec3>& vertices,
	PxArray<PxU32>& triangles, PxReal radius, PxReal height,
	PxU32 ringSegments, PxU32 heightSegments)
{
	vertices.clear();
	triangles.clear();
	// The showcase authors a fixed 32-segment ring. Runtime sin/cos differs in
	// the last float bits between Debug and optimized MSVC builds; vertices that
	// land on a voxel boundary then select a different public-cooking cell and
	// change the simulation mass by one voxel. Canonical authored directions
	// keep the cooking input identical without bypassing cooking in either build.
	static const PxVec3 unitRing32[32] =
	{
		PxVec3(0.0f,0.0f,1.0f),
		PxVec3(0.195090322f,0.0f,0.98078528f),
		PxVec3(0.382683426f,0.0f,0.923879504f),
		PxVec3(0.555570245f,0.0f,0.831469595f),
		PxVec3(0.707106769f,0.0f,0.707106769f),
		PxVec3(0.831469595f,0.0f,0.555570245f),
		PxVec3(0.923879504f,0.0f,0.382683426f),
		PxVec3(0.98078528f,0.0f,0.195090322f),
		PxVec3(1.0f,0.0f,0.0f),
		PxVec3(0.98078528f,0.0f,-0.195090322f),
		PxVec3(0.923879504f,0.0f,-0.382683426f),
		PxVec3(0.831469595f,0.0f,-0.555570245f),
		PxVec3(0.707106769f,0.0f,-0.707106769f),
		PxVec3(0.555570245f,0.0f,-0.831469595f),
		PxVec3(0.382683426f,0.0f,-0.923879504f),
		PxVec3(0.195090322f,0.0f,-0.98078528f),
		PxVec3(0.0f,0.0f,-1.0f),
		PxVec3(-0.195090322f,0.0f,-0.98078528f),
		PxVec3(-0.382683426f,0.0f,-0.923879504f),
		PxVec3(-0.555570245f,0.0f,-0.831469595f),
		PxVec3(-0.707106769f,0.0f,-0.707106769f),
		PxVec3(-0.831469595f,0.0f,-0.555570245f),
		PxVec3(-0.923879504f,0.0f,-0.382683426f),
		PxVec3(-0.98078528f,0.0f,-0.195090322f),
		PxVec3(-1.0f,0.0f,0.0f),
		PxVec3(-0.98078528f,0.0f,0.195090322f),
		PxVec3(-0.923879504f,0.0f,0.382683426f),
		PxVec3(-0.831469595f,0.0f,0.555570245f),
		PxVec3(-0.707106769f,0.0f,0.707106769f),
		PxVec3(-0.555570245f,0.0f,0.831469595f),
		PxVec3(-0.382683426f,0.0f,0.923879504f),
		PxVec3(-0.195090322f,0.0f,0.98078528f)
	};
	for(PxU32 layer = 0; layer < heightSegments; ++layer)
	{
		const PxReal t = PxReal(layer) / PxReal(heightSegments);
		const PxReal r = radius * (1.0f - t);
		for(PxU32 segment = 0; segment < ringSegments; ++segment)
		{
			if(ringSegments==32)
				vertices.pushBack(PxVec3(r*unitRing32[segment].x,height*t,
					r*unitRing32[segment].z));
			else
			{
				const PxReal angle=2.0f*PxPi*PxReal(segment)/PxReal(ringSegments);
				vertices.pushBack(PxVec3(r*PxSin(angle),height*t,r*PxCos(angle)));
			}
		}
	}
	const PxU32 apex = vertices.size();
	vertices.pushBack(PxVec3(0.0f, height, 0.0f));
	// The ring advances from +Z toward +X. Wind every face outward so
	// renderMesh's edge cross product and back-face culling agree.
	for(PxU32 layer = 0; layer + 1 < heightSegments; ++layer)
	{
		const PxU32 lower = layer * ringSegments;
		const PxU32 upper = (layer + 1) * ringSegments;
		for(PxU32 segment = 0; segment < ringSegments; ++segment)
		{
			const PxU32 next = (segment + 1) % ringSegments;
			triangles.pushBack(upper+segment); triangles.pushBack(lower+segment); triangles.pushBack(lower+next);
			triangles.pushBack(upper+segment); triangles.pushBack(lower+next); triangles.pushBack(upper+next);
		}
	}
	const PxU32 top = (heightSegments - 1) * ringSegments;
	for(PxU32 segment = 0; segment < ringSegments; ++segment)
	{
		triangles.pushBack(apex);
		triangles.pushBack(top + segment);
		triangles.pushBack(top + (segment + 1) % ringSegments);
	}
	const PxU32 base = vertices.size();
	vertices.pushBack(PxVec3(0.0f));
	for(PxU32 segment = 0; segment < ringSegments; ++segment)
	{
		triangles.pushBack(base);
		triangles.pushBack((segment + 1) % ringSegments);
		triangles.pushBack(segment);
	}
}

inline PxDeformableVolumeMesh* cookVisualMesh(PxPhysics& physics,
	const PxArray<PxVec3>& vertices, const PxArray<PxU32>& triangles,
	PxU32 voxels)
{
	PxSimpleTriangleMesh surface;
	surface.points.count = vertices.size();
	surface.points.stride = sizeof(PxVec3);
	surface.points.data = vertices.begin();
	surface.triangles.count = triangles.size() / 3;
	surface.triangles.stride = 3 * sizeof(PxU32);
	surface.triangles.data = triangles.begin();
	PxCookingParams cooking(physics.getTolerancesScale());
	cooking.buildGPUData = false;
	cooking.meshWeldTolerance = 0.001f;
	cooking.meshPreprocessParams = PxMeshPreprocessingFlag::eWELD_VERTICES;
	return PxDeformableVolumeExt::createDeformableVolumeMesh(cooking, surface,
		voxels, physics.getPhysicsInsertionCallback());
}

inline PxDeformableVolume* createVisualVolume(PxPhysics& physics, PxScene& scene,
	PxDeformableVolumeMesh& mesh, PxDeformableVolumeMaterial& material,
	const PxTransform& pose, const PxVec3& scale, PxReal density)
{
	PxDeformableVolume* volume = physics.createDeformableVolume();
	if(!volume)
		return NULL;
	PxDeformableVolumeMaterial* materials[] = { &material };
	PxShape* shape = physics.createShape(
		PxTetrahedronMeshGeometry(mesh.getCollisionMesh()), materials, 1, true,
		PxShapeFlag::eVISUALIZATION | PxShapeFlag::eSCENE_QUERY_SHAPE |
		PxShapeFlag::eSIMULATION_SHAPE);
	const bool attached = shape && volume->attachShape(*shape) &&
		volume->attachSimulationMesh(*mesh.getSimulationMesh(),
			*mesh.getDeformableVolumeAuxData());
	PX_RELEASE(shape);
	if(!attached)
	{
		volume->release();
		return NULL;
	}
	PxVec4 *sim = NULL, *vel = NULL, *collision = NULL, *rest = NULL;
	PxDeformableVolumeExt::allocateAndInitializeHostMirror(
		*volume, NULL, sim, vel, collision, rest);
	if(!sim || !vel || !collision || !rest)
	{
		hostFree(sim); hostFree(vel); hostFree(collision); hostFree(rest);
		volume->release();
		return NULL;
	}
	const PxU32 simCount = mesh.getSimulationMesh()->getNbVertices();
	for(PxU32 i = 0; i < simCount; ++i)
	{
		const PxVec3 p = sim[i].getXYZ();
		sim[i] = PxVec4(pose.transform(PxVec3(
			p.x * scale.x, p.y * scale.y, p.z * scale.z)), sim[i].w);
		vel[i] = PxVec4(0.0f, 0.0f, 0.0f, sim[i].w);
	}
	PxDeformableVolumeExt::updateMass(*volume, density, 50.0f, sim);
	for(PxU32 i = 0; i < simCount; ++i)
		vel[i].w = sim[i].w;
	PxDeformableVolumeExt::updateEmbeddedCollisionMesh(*volume, sim, collision);
	for(PxU32 i = 0; i < mesh.getCollisionMesh()->getNbVertices(); ++i)
		rest[i] = collision[i];
	PxDeformableVolumeExt::copyToDevice(*volume,
		PxDeformableVolumeDataFlag::eALL, sim, vel, collision, rest);
	hostFree(sim); hostFree(vel); hostFree(collision); hostFree(rest);
	volume->setSolverIterationCounts(16, 1);
	volume->setLinearDamping(0.005f);
	if(!scene.addActor(*volume))
	{
		volume->release();
		return NULL;
	}
	return volume;
}

inline bool setVisualVelocity(PxDeformableVolume& volume, const PxVec3& value)
{
	PxVec4* velocities = volume.getSimVelocityBufferD();
	if(!velocities)
		return false;
	for(PxU32 i = 0; i < volume.getSimulationMesh()->getNbVertices(); ++i)
		velocities[i] = PxVec4(value, velocities[i].w);
	volume.markDirty(PxDeformableVolumeDataFlag::eSIM_VELOCITY);
	return true;
}
}
#endif
