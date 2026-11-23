#pragma once
#include "PxPhysicsAPI.h"

namespace SurfaceAvbdSnippet
{
inline physx::PxU32 vertexId(physx::PxU32 x, physx::PxU32 y, physx::PxU32 height)
{
	return x * height + y;
}

inline physx::PxDeformableSurface* createHangingMembrane(physx::PxPhysics& physics,
	physx::PxScene& scene, physx::PxTriangleMesh*& retainedMesh,
	physx::PxDeformableSurfaceMaterial*& retainedMaterial, physx::PxU32 width = 17,
	physx::PxU32 height = 17, physx::PxReal elasticityDamping = 0.1f)
{
	physx::PxArray<physx::PxVec3> vertices;
	physx::PxArray<physx::PxU32> triangles;
	vertices.reserve(width * height);
	triangles.reserve((width - 1) * (height - 1) * 6);
	for(physx::PxU32 x = 0; x < width; ++x)
	{
		for(physx::PxU32 y = 0; y < height; ++y)
		{
			const physx::PxReal fx = physx::PxReal(x) / physx::PxReal(width - 1);
			const physx::PxReal fy = physx::PxReal(y) / physx::PxReal(height - 1);
			vertices.pushBack(physx::PxVec3(-3.0f + 6.0f * fx, 3.0f + 6.0f * fy, 0.0f));
		}
	}
	for(physx::PxU32 x = 1; x < width; ++x)
	{
		for(physx::PxU32 y = 1; y < height; ++y)
		{
			const physx::PxU32 a = vertexId(x - 1, y - 1, height);
			const physx::PxU32 b = vertexId(x, y - 1, height);
			const physx::PxU32 c = vertexId(x - 1, y, height);
			const physx::PxU32 d = vertexId(x, y, height);
			triangles.pushBack(a); triangles.pushBack(b); triangles.pushBack(c);
			triangles.pushBack(c); triangles.pushBack(b); triangles.pushBack(d);
		}
	}
	physx::PxTriangleMeshDesc desc;
	desc.points.count = vertices.size();
	desc.points.stride = sizeof(physx::PxVec3);
	desc.points.data = vertices.begin();
	desc.triangles.count = triangles.size() / 3;
	desc.triangles.stride = sizeof(physx::PxU32) * 3;
	desc.triangles.data = triangles.begin();
	retainedMesh = PxCreateTriangleMesh(physx::PxCookingParams(physics.getTolerancesScale()),
		desc, physics.getPhysicsInsertionCallback());
	if(!retainedMesh)
		return NULL;
	retainedMaterial = physics.createDeformableSurfaceMaterial(2.0e4f, 0.3f,
		0.2f, 0.04f, 0.0f, elasticityDamping, 0.0f);
	if(!retainedMaterial)
		return NULL;
	physx::PxDeformableSurfaceMaterial* materials[] = { retainedMaterial };
	physx::PxShape* shape = physics.createShape(physx::PxTriangleMeshGeometry(retainedMesh),
		materials, 1, true, physx::PxShapeFlag::eVISUALIZATION |
		physx::PxShapeFlag::eSCENE_QUERY_SHAPE | physx::PxShapeFlag::eSIMULATION_SHAPE);
	if(!shape)
		return NULL;
	physx::PxDeformableSurface* surface = physics.createDeformableSurface(
		physx::PxDeformableSurfaceBackend::eCPU_AVBD);
	if(!surface || !surface->attachShape(*shape))
	{
		PX_RELEASE(surface);
		shape->release();
		return NULL;
	}
	shape->release();
	physx::PxVec4* positions = surface->getPositionInvMassBufferH();
	physx::PxVec4* velocities = surface->getVelocityBufferH();
	physx::PxVec4* rest = surface->getRestPositionBufferH();
	if(!positions || !velocities || !rest)
	{
		surface->release();
		return NULL;
	}
	for(physx::PxU32 i = 0; i < vertices.size(); ++i)
	{
		const bool pinned = i == vertexId(0, height - 1, height) ||
			i == vertexId(width - 1, height - 1, height);
		positions[i] = physx::PxVec4(vertices[i], pinned ? 0.0f : 20.0f);
		velocities[i] = physx::PxVec4(0.0f);
		rest[i] = physx::PxVec4(vertices[i], 0.0f);
	}
	surface->setSolverIterationCounts(12, 1);
	physx::PxDeformableSurfaceDataFlags dirty(physx::PxDeformableSurfaceDataFlag::ePOSITION_INVMASS);
	dirty |= physx::PxDeformableSurfaceDataFlag::eVELOCITY;
	dirty |= physx::PxDeformableSurfaceDataFlag::eREST_POSITION;
	surface->markDirty(dirty);
	scene.addActor(*surface);
	return surface;
}

struct DrapeGrid
{
	physx::PxU32 width, height;
	physx::PxReal spacing;
};
static constexpr DrapeGrid kDrapeGrid = {50, 50, 0.1f};
static constexpr DrapeGrid kReferenceDrapeGrid = {25, 25, 0.2f};

inline physx::PxDeformableSurface* createDrapedSurface(physx::PxPhysics& physics,
	physx::PxScene& scene, physx::PxTriangleMesh*& retainedMesh,
	physx::PxDeformableSurfaceMaterial*& retainedMaterial, physx::PxReal bending = 0.2f,
	const DrapeGrid& grid = kDrapeGrid, const physx::PxVec3& origin = physx::PxVec3(0.0f))
{
	using namespace physx;
	const PxU32 width = grid.width;
	const PxU32 height = grid.height;
	const PxReal spacing = grid.spacing;
	PxArray<PxVec3> vertices;
	PxArray<PxU32> triangles;
	vertices.reserve(width * height);
	triangles.reserve(6 * (width - 1) * (height - 1));
	for(PxU32 z = 0; z < height; ++z)
		for(PxU32 x = 0; x < width; ++x)
			vertices.pushBack(origin + PxVec3(
				(PxReal(x) - 0.5f * PxReal(width - 1)) * spacing,
				4.2f,
				(PxReal(z) - 0.5f * PxReal(height - 1)) * spacing));
	for(PxU32 z = 0; z + 1 < height; ++z)
	{
		for(PxU32 x = 0; x + 1 < width; ++x)
		{
			const PxU32 v00 = z * width + x;
			const PxU32 v10 = v00 + 1;
			const PxU32 v01 = v00 + width;
			const PxU32 v11 = v01 + 1;
			if((x + z) & 1)
			{
				triangles.pushBack(v00); triangles.pushBack(v01); triangles.pushBack(v11);
				triangles.pushBack(v00); triangles.pushBack(v11); triangles.pushBack(v10);
			}
			else
			{
				triangles.pushBack(v00); triangles.pushBack(v01); triangles.pushBack(v10);
				triangles.pushBack(v01); triangles.pushBack(v11); triangles.pushBack(v10);
			}
		}
	}
	PxTriangleMeshDesc desc;
	desc.points.count = vertices.size();
	desc.points.stride = sizeof(PxVec3);
	desc.points.data = vertices.begin();
	desc.triangles.count = triangles.size() / 3;
	desc.triangles.stride = 3 * sizeof(PxU32);
	desc.triangles.data = triangles.begin();
	PxCookingParams cooking(physics.getTolerancesScale());
	cooking.buildGPUData = false;
	cooking.buildTriangleAdjacencies = true;
	retainedMesh = PxCreateTriangleMesh(cooking, desc,
		physics.getPhysicsInsertionCallback());
	if(!retainedMesh)
		return NULL;
	// The reference drape is authored with E*h in N/m. The public material
	// takes a 3D Young modulus in Pa; the membrane kernel integrates thickness.
	const PxReal arealYoungs = 2.0e4f;
	const PxReal thickness = 0.02f;
	retainedMaterial = physics.createDeformableSurfaceMaterial(
		arealYoungs / thickness, 0.3f, 0.08f, thickness, bending);
	PxDeformableSurface* surface = physics.createDeformableSurface(
		PxDeformableSurfaceBackend::eCPU_AVBD);
	if(!retainedMaterial || !surface)
		return NULL;
	PxDeformableSurfaceMaterial* materials[] = { retainedMaterial };
	PxShape* shape = physics.createShape(PxTriangleMeshGeometry(retainedMesh),
		materials, 1, true, PxShapeFlag::eVISUALIZATION |
		PxShapeFlag::eSCENE_QUERY_SHAPE | PxShapeFlag::eSIMULATION_SHAPE);
	if(!shape)
		return NULL;
	shape->setContactOffset(0.04f);
	shape->setRestOffset(0.01f);
	const bool attached = surface->attachShape(*shape);
	shape->release();
	if(!attached)
		return NULL;
	PxVec4* positions = surface->getPositionInvMassBufferH();
	PxVec4* velocities = surface->getVelocityBufferH();
	PxVec4* rest = surface->getRestPositionBufferH();
	if(!positions || !velocities || !rest)
		return NULL;
	// Preserve the authored sheet mass of the original 625 one-kilogram samples.
	const PxReal inverseNodeMass = PxReal(vertices.size()) / 625.0f;
	for(PxU32 i = 0; i < vertices.size(); ++i)
	{
		positions[i] = PxVec4(vertices[i], inverseNodeMass);
		velocities[i] = PxVec4(0.0f);
		rest[i] = PxVec4(vertices[i], 0.0f);
	}
	surface->setLinearDamping(0.2f);
	surface->setSolverIterationCounts(12, 1);
	surface->setSleepThreshold(5.0e-4f);
	surface->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, false);
	surface->markDirty(PxDeformableSurfaceDataFlag::eALL);
	if(!scene.addActor(*surface))
		return NULL;
	return surface;
}

inline physx::PxDeformableSurface* createBendingProbe(physx::PxPhysics& physics,
	physx::PxScene& scene, physx::PxTriangleMesh*& retainedMesh,
	physx::PxDeformableSurfaceMaterial*& retainedMaterial)
{
	using namespace physx;
	const PxVec3 restVertices[4] =
	{
		PxVec3(0.0f, 0.0f, 0.0f), PxVec3(1.0f, 0.0f, 0.0f),
		PxVec3(0.0f, 0.0f, 1.0f), PxVec3(1.0f, 0.0f, 1.0f)
	};
	const PxU32 triangles[6] = {0, 2, 1, 2, 3, 1};
	PxTriangleMeshDesc desc;
	desc.points.count = 4; desc.points.stride = sizeof(PxVec3);
	desc.points.data = restVertices;
	desc.triangles.count = 2; desc.triangles.stride = 3 * sizeof(PxU32);
	desc.triangles.data = triangles;
	PxCookingParams cooking(physics.getTolerancesScale());
	cooking.buildGPUData = false;
	cooking.buildTriangleAdjacencies = true;
	retainedMesh = PxCreateTriangleMesh(cooking, desc,
		physics.getPhysicsInsertionCallback());
	retainedMaterial = physics.createDeformableSurfaceMaterial(
		2.0e4f, 0.3f, 0.0f, 0.02f, 10.0f);
	PxDeformableSurface* surface = physics.createDeformableSurface(
		PxDeformableSurfaceBackend::eCPU_AVBD);
	if(!retainedMesh || !retainedMaterial || !surface) return NULL;
	PxDeformableSurfaceMaterial* materials[] = {retainedMaterial};
	PxShape* shape = physics.createShape(PxTriangleMeshGeometry(retainedMesh),
		materials, 1, true, PxShapeFlag::eSIMULATION_SHAPE |
		PxShapeFlag::eSCENE_QUERY_SHAPE);
	const bool attached = shape && surface->attachShape(*shape);
	PX_RELEASE(shape);
	if(!attached) { surface->release(); return NULL; }
	PxVec4* positions = surface->getPositionInvMassBufferH();
	PxVec4* velocities = surface->getVelocityBufferH();
	PxVec4* rest = surface->getRestPositionBufferH();
	if(!positions || !velocities || !rest) { surface->release(); return NULL; }
	for(PxU32 i = 0; i < 4; ++i)
	{
		const PxVec3 initial = i == 3 ?
			PxVec3(0.5f, PxSqrt(0.5f), 0.5f) : restVertices[i];
		positions[i] = PxVec4(initial, i == 3 ? 1.0f : 0.0f);
		velocities[i] = PxVec4(0.0f);
		rest[i] = PxVec4(restVertices[i], 0.0f);
	}
	surface->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
	surface->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
	surface->setLinearDamping(2.0f);
	surface->setSolverIterationCounts(16, 1);
	surface->setSleepThreshold(1.0e-8f);
	surface->markDirty(PxDeformableSurfaceDataFlag::eALL);
	if(!scene.addActor(*surface)) { surface->release(); return NULL; }
	return surface;
}

inline void resetDrapedSurface(physx::PxDeformableSurface& surface,
	const physx::PxTriangleMesh& mesh)
{
	using namespace physx;
	PxVec4* positions = surface.getPositionInvMassBufferH();
	PxVec4* velocities = surface.getVelocityBufferH();
	PxVec4* rest = surface.getRestPositionBufferH();
	const PxVec3* vertices = mesh.getVertices();
	if(!positions || !velocities || !rest || !vertices)
		return;
	for(PxU32 i = 0; i < mesh.getNbVertices(); ++i)
	{
		positions[i] = PxVec4(vertices[i], 1.0f);
		velocities[i] = PxVec4(0.0f);
		rest[i] = PxVec4(vertices[i], 0.0f);
	}
	surface.markDirty(PxDeformableSurfaceDataFlag::eALL);
	surface.setWakeCounter(0.4f);
}
}
