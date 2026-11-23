#include "SnippetDeformableVolumeAVBDHeadless.h"
#include "SnippetDeformableVolumeAVBD.h"
#include "SnippetDeformableVolumeAVBDCommon.h"
#include "SnippetDeformableVolumeAVBDBenchmark.h"
#include "SnippetDeformableVolumeAVBDEmbeddingCheck.h"
#include "SnippetDeformableVolumeAVBDEnergy.h"
#include "SnippetDeformableVolumeAVBDCookedMesh.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetSoftTipObservation.h"
#include "../snippetcommon/SnippetSoftReboundObservation.h"
#include "GuIntersectionTriangleBox.h"
#include "geometry/PxGjkQuery.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

using namespace physx;
namespace
{
bool volumeAngularVelocity(PxDeformableVolume& volume, PxVec3& omega);
// Read-only headless motion trace. Standard
// storage is intentional: its lifetime may outlive the PhysX foundation.
void traceShowcaseMotion(unsigned frame, double stepMs)
{
	if(!Snippets::isEnabledEnvironmentValue(std::getenv("PHYSX_AVBD_SHOWCASE_MOTION"))) return;
	static std::vector<PxVec3> previous[5];
	for(PxU32 body = 0; body < 5; ++body)
	{
		PxDeformableVolume* volume = getVolumeAvbdRenderVolume(body);
		if(!volume || !volume->getScene() || !volume->getSimulationMesh()) continue;
		const PxU32 count = volume->getSimulationMesh()->getNbVertices();
		const PxVec4* positions = volume->getSimPositionInvMassBufferD();
		const PxVec4* velocities = volume->getSimVelocityBufferD();
		if(!positions || !velocities) continue;
		const bool initial = frame == 0 || previous[body].size() != count;
		previous[body].resize(count);
		PxVec3d center(0), momentum(0);
		double mass = 0, kinetic = 0;
		PxReal minY = PX_MAX_F32, maxStep = 0.0f, maxSpeed = 0.0f;
		for(PxU32 vertex = 0; vertex < count; ++vertex)
		{
			const PxVec3 p = positions[vertex].getXYZ();
			const PxReal m = positions[vertex].w > 0.0f ? 1.0f / positions[vertex].w : 0.0f;
			const auto v = velocities[vertex].getXYZ();
			center += PxVec3d(p.x,p.y,p.z) * double(m);
			momentum += PxVec3d(v.x,v.y,v.z) * double(m);
			mass += m;
			kinetic += .5 * m * v.magnitudeSquared();
			minY = PxMin(minY, p.y);
			if(!initial) maxStep = PxMax(maxStep, (p - previous[body][vertex]).magnitude());
			maxSpeed = PxMax(maxSpeed, velocities[vertex].getXYZ().magnitude());
			previous[body][vertex] = p;
		}
		if(mass > 0.0f) center /= mass;
		PxDeformableVolumeMaterial* material = NULL;
		volume->getShape()->getDeformableVolumeMaterials(&material, 1);
		double minJ = HUGE_VAL;
		const double strain = avbdProbeVolumeEnergy(*volume->getSimulationMesh(), positions,
			material->getYoungsModulus(), material->getPoissons(), minJ);
		const auto gravity = gVolumeAvbdScene->getGravity();
		const double gravitational = -mass * center.dot(PxVec3d(gravity.x,gravity.y,gravity.z));
		PxVec3 omega(0);
		const bool hasOmega=volumeAngularVelocity(*volume,omega);
		std::fprintf(stderr, "[AVBD_MOTION] frame=%u body=%u cx=%.9g cy=%.9g cz=%.9g minY=%.9g maxStep=%.9g maxSpeed=%.9g stepMs=%.9g kinetic=%.9g comKinetic=%.9g strain=%.9g gravitational=%.9g mass=%.9g angularSpeed=%.9g\n",
			frame, body, double(center.x), double(center.y), double(center.z),
			double(minY), double(maxStep), double(maxSpeed), stepMs,
			kinetic, mass>0 ? .5*momentum.magnitudeSquared()/mass : 0, strain, gravitational, mass,
			hasOmega?double(omega.magnitude()):double(NAN));
	}
	std::fflush(stderr);
}

struct Fixture
{
	Snippets::TrackingErrorCallback errors;
	PxDefaultAllocator allocator;
	PxFoundation* foundation;
	PxPhysics* physics;
	PxDefaultCpuDispatcher* dispatcher;
	PxScene* scene;
	PxMaterial* rigidMaterial;
	PxDeformableVolumeMaterial* deformableMaterial;
	PxDeformableVolume* volume;
	PxDeformableVolume* volumeB;
	PxDeformableVolumeMesh* mesh;
	PxDeformableVolumeMesh* meshB;
	PxRigidDynamic* dynamicBox;
	PxVec3 dynamicBoxHalfExtent;
	bool extensions;
	Fixture() : foundation(NULL), physics(NULL), dispatcher(NULL), scene(NULL),
		rigidMaterial(NULL), deformableMaterial(NULL), volume(NULL), volumeB(NULL), mesh(NULL), meshB(NULL),
		dynamicBox(NULL), dynamicBoxHalfExtent(0.0f), extensions(false) {}
	~Fixture()
	{
		PX_RELEASE(volume);
		PX_RELEASE(volumeB);
		PX_RELEASE(mesh);
		PX_RELEASE(meshB);
		PX_RELEASE(scene);
		PX_RELEASE(deformableMaterial);
		PX_RELEASE(rigidMaterial);
		PX_RELEASE(dispatcher);
		if(extensions) PxCloseExtensions();
		PX_RELEASE(physics);
		PX_RELEASE(foundation);
	}
	bool init(const Snippets::HeadlessOptions& options, const char* caseName)
	{
		foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
		if(!foundation) return false;
		physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation,
			PxTolerancesScale(), true, NULL);
		if(!physics) return false;
		extensions = PxInitExtensions(*physics, NULL);
		dispatcher = PxDefaultCpuDispatcherCreate(options.dispatcherThreads);
		PxSceneDesc desc(physics->getTolerancesScale());
		desc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
		desc.solverType = options.solverType;
		desc.cpuDispatcher = dispatcher;
		desc.filterShader = PxDefaultSimulationFilterShader;
		scene = physics->createScene(desc);
		rigidMaterial = physics->createMaterial(0.5f, 0.5f, 0.0f);
		if(!scene || !rigidMaterial) return false;
		PxRigidStatic* ground = PxCreatePlane(*physics,
			PxPlane(0.0f, 1.0f, 0.0f, 0.0f), *rigidMaterial);
		if(!ground || !scene->addActor(*ground)) return false;

		const bool staticBox = Snippets::equalsIgnoreCase(caseName, "static-box");
		const bool dynamic = Snippets::equalsIgnoreCase(caseName, "dynamic-box");
		const bool embedded = Snippets::equalsIgnoreCase(caseName, "embedded-sync");
		const bool embeddedSameCount = Snippets::equalsIgnoreCase(caseName, "embedded-same-count");
		const bool softSoft = Snippets::equalsIgnoreCase(caseName, "soft-soft");
		const bool softSoftRecovery = Snippets::equalsIgnoreCase(caseName,
			"soft-soft-recovery");
		const bool softSoftInitialContainment = Snippets::equalsIgnoreCase(caseName,
			"soft-soft-initial-containment");
		const bool rigidSqueeze = Snippets::equalsIgnoreCase(caseName, "rigid-squeeze");
		const bool separatedRigid = Snippets::equalsIgnoreCase(caseName, "rigid-on-soft-separated");
		const bool rigidOnSoft = Snippets::equalsIgnoreCase(caseName, "rigid-on-soft") || separatedRigid;
		const bool freeFallTriplet = Snippets::equalsIgnoreCase(caseName, "free-fall-triplet");
		const bool rollingSphere = Snippets::equalsIgnoreCase(caseName, "rolling-sphere");
		if(rollingSphere)
		{
			PxArray<PxVec3> vertices;
			PxArray<PxU32> triangles;
			createVolumeAvbdSphereMesh(vertices, triangles, 1.0f, 0.45f);
			mesh = VolumeAvbdSnippet::cookVisualMesh(*physics, vertices, triangles, 6);
			deformableMaterial = physics->createDeformableVolumeMaterial(
				1.2e5f, 0.4f, 0.6f, 0.01f);
			if(!mesh || !deformableMaterial) return false;
			volume = VolumeAvbdSnippet::createVisualVolume(*physics, *scene, *mesh,
				*deformableMaterial, PxTransform(PxVec3(0.0f, 1.04f, 0.0f)),
				PxVec3(1.0f), 130.0f);
			if(!volume) return false;
			volume->setLinearDamping(0.0f);
			// Seed the state immediately after a centered horizontal impact.  This
			// isolates whether the supported soft sphere converts translation into
			// angular momentum, independently of projectile/contact timing.
			if(!VolumeAvbdSnippet::setVisualVelocity(*volume,
				PxVec3(2.0f, 0.0f, 0.0f))) return false;
		}
		else if(staticBox)
		{
			PxRigidStatic* box = physics->createRigidStatic(PxTransform(PxVec3(0.0f, 0.5f, 0.0f)));
			PxRigidActorExt::createExclusiveShape(*box, PxBoxGeometry(2.0f, 0.5f, 2.0f),
				*rigidMaterial);
			scene->addActor(*box);
		}
		if(dynamic)
		{
			dynamicBoxHalfExtent = PxVec3(2.0f, 0.5f, 2.0f);
			dynamicBox = PxCreateDynamic(*physics, PxTransform(PxVec3(0.0f, 0.6f, 0.0f)),
				PxBoxGeometry(dynamicBoxHalfExtent), *rigidMaterial, 20.0f);
			scene->addActor(*dynamicBox);
		}
		if(!rollingSphere)
		{
		if(embeddedSameCount)
		{
			// Same counts and centroid, different mapping. Copying by vertex ID
			// expands the collision tetrahedron while the centroid gate stays green.
			PxArray<PxVec3> sim, collision;
			sim.pushBack(PxVec3(0.0f)); sim.pushBack(PxVec3(1.0f, 0.0f, 0.0f));
			sim.pushBack(PxVec3(0.0f, 1.0f, 0.0f)); sim.pushBack(PxVec3(0.0f, 0.0f, 1.0f));
			for(PxU32 vertex = 0; vertex < sim.size(); ++vertex)
				collision.pushBack(PxVec3(0.25f) + 0.5f * (sim[vertex] - PxVec3(0.25f)));
			PxArray<PxU32> indices;
			for(PxU32 vertex = 0; vertex < 4; ++vertex) indices.pushBack(vertex);
			const PxTetrahedronMeshDesc simDesc(sim, indices), collDesc(collision, indices);
			mesh = PxCreateDeformableVolumeMesh(PxCookingParams(physics->getTolerancesScale()),
				simDesc, collDesc, PxDeformableVolumeSimulationDataDesc(),
				physics->getPhysicsInsertionCallback());
			deformableMaterial = physics->createDeformableVolumeMaterial(1.2e5f, 0.4f, 0.1f);
			if(!mesh || !deformableMaterial) return false;
			volume = VolumeAvbdSnippet::createVisualVolume(*physics, *scene, *mesh,
				*deformableMaterial, PxTransform(PxVec3(0.0f, 1.0f, 0.0f)), PxVec3(1.0f), 100.0f);
		}
		else if(embedded)
		{
			PxArray<PxVec3> vertices;
			PxArray<PxU32> triangles;
			VolumeAvbdSnippet::createSubdividedCubeSurface(vertices, triangles, 2.0f, 3);
			mesh = VolumeAvbdSnippet::cookVisualMesh(*physics, vertices, triangles, 8);
			deformableMaterial = physics->createDeformableVolumeMaterial(1.2e5f, 0.4f, 0.1f);
			if(!mesh || !deformableMaterial) return false;
			volume = VolumeAvbdSnippet::createVisualVolume(*physics, *scene, *mesh,
				*deformableMaterial, PxTransform(PxVec3(0.0f, 4.0f, 0.0f)),
				PxVec3(1.0f), 100.0f);
		}
		else if(softSoft || softSoftRecovery || softSoftInitialContainment ||
			rigidSqueeze || freeFallTriplet)
		{
			if(softSoftInitialContainment)
			{
				// A small 5-tet cube starts wholly inside another 5-tet cube.  This
				// isolates containment topology from the 840-face showcase mesh and
				// keeps the gate fast enough to run in every qualification build.
				volume = VolumeAvbdSnippet::createCube(*physics, *scene, mesh,
					PxVec3(0.0f, 5.0f, 0.0f), 2.0f);
				volumeB = VolumeAvbdSnippet::createCube(*physics, *scene, meshB,
					PxVec3(0.0f, 5.70f, 0.0f), 0.40f);
				if(!volume || !volumeB)
					return false;
				volume->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
				volumeB->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
			}
			else
			{
			PxArray<PxVec3> vertices;
			PxArray<PxU32> triangles;
			VolumeAvbdSnippet::createSubdividedCubeSurface(vertices, triangles, 2.0f, 3);
			mesh = VolumeAvbdSnippet::cookVisualMesh(*physics, vertices, triangles, 5);
			deformableMaterial = physics->createDeformableVolumeMaterial(1.2e5f, 0.4f, 0.1f);
			if(!mesh || !deformableMaterial) return false;
			if(softSoft || softSoftRecovery)
			{
				const PxReal lowerY = softSoftRecovery ? 5.0f : 1.05f;
				const PxReal upperY = softSoftRecovery ? 8.0f : 4.0f;
				volume = VolumeAvbdSnippet::createVisualVolume(*physics, *scene, *mesh,
					*deformableMaterial, PxTransform(PxVec3(0.0f, lowerY, 0.0f)),
					PxVec3(1.0f), 100.0f);
				volumeB = VolumeAvbdSnippet::createVisualVolume(*physics, *scene, *mesh,
					*deformableMaterial, PxTransform(PxVec3(0.0f, upperY, 0.0f)),
					PxVec3(1.0f), 100.0f);
				if(softSoftRecovery && volume && volumeB)
				{
					volume->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
					volumeB->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
					if(!VolumeAvbdSnippet::setVisualVelocity(*volume,
						PxVec3(0.0f, 2.0f, 0.0f)) ||
						!VolumeAvbdSnippet::setVisualVelocity(*volumeB,
						PxVec3(0.0f, -2.0f, 0.0f))) return false;
				}
			}
			else
			{
				const PxQuat orientation(0.36f, PxVec3(0.0f, 1.0f, 0.0f));
				const PxVec3 center(0.0f, freeFallTriplet ? 25.0f : 5.0f, 0.0f);
				volume = VolumeAvbdSnippet::createVisualVolume(*physics, *scene, *mesh,
					*deformableMaterial, PxTransform(center + PxVec3(0.0f, 1.10f, 0.0f), orientation),
					PxVec3(1.7f, 0.7f, 0.9f), 160.0f);
				volumeB = VolumeAvbdSnippet::createVisualVolume(*physics, *scene, *mesh,
					*deformableMaterial, PxTransform(center - PxVec3(0.0f, 1.10f, 0.0f), orientation),
					PxVec3(1.7f, 0.7f, 0.9f), 160.0f);
				dynamicBoxHalfExtent = PxVec3(0.5f, 0.35f, 0.7f);
				dynamicBox = PxCreateDynamic(*physics, PxTransform(center, orientation),
					PxBoxGeometry(dynamicBoxHalfExtent), *rigidMaterial, 10.0f);
				if(!volume || !volumeB || !dynamicBox || !scene->addActor(*dynamicBox))
					return false;
				if(freeFallTriplet)
				{
					// This gate isolates common-mode gravity from contact. Give all
					// three bodies the same undamped inertial contract; authored
					// damping is covered by its own material tests.
					deformableMaterial->setElasticityDamping(0.0f);
					volume->setLinearDamping(0.0f);
					volumeB->setLinearDamping(0.0f);
					dynamicBox->setLinearDamping(0.0f);
					dynamicBox->setAngularDamping(0.0f);
				}
				if(!freeFallTriplet &&
					(!VolumeAvbdSnippet::setVisualVelocity(*volume, PxVec3(0.0f, -0.5f, 0.0f)) ||
					 !VolumeAvbdSnippet::setVisualVelocity(*volumeB, PxVec3(0.0f, 0.5f, 0.0f))))
					return false;
			}
			}
		}
		else if(rigidOnSoft)
		{
			// Showcase sandwich: visual cube on a static pedestal, 10 kg
			// dynamic box overlapping the top.  The 5-tet light-box cell did
			// not reproduce the frozen clamp.
			PxRigidStatic* pedestal = physics->createRigidStatic(
				PxTransform(PxVec3(0.0f, 0.55f, 0.0f)));
			if(!pedestal || !PxRigidActorExt::createExclusiveShape(*pedestal,
				PxBoxGeometry(0.7f, 0.55f, 2.2f), *rigidMaterial) ||
				!scene->addActor(*pedestal))
				return false;
			PxArray<PxVec3> vertices;
			PxArray<PxU32> triangles;
			VolumeAvbdSnippet::createSubdividedCubeSurface(vertices, triangles, 2.0f, 3);
			mesh = VolumeAvbdSnippet::cookVisualMesh(*physics, vertices, triangles, 5);
			deformableMaterial = physics->createDeformableVolumeMaterial(
				1.2e5f, 0.4f, 0.2f, 0.01f);
			if(!mesh || !deformableMaterial)
				return false;
			deformableMaterial->setMaterialModel(
				PxDeformableVolumeMaterialModel::eNEO_HOOKEAN);
			const PxQuat orientation(0.36f, PxVec3(0.0f, 1.0f, 0.0f));
			volume = VolumeAvbdSnippet::createVisualVolume(*physics, *scene, *mesh,
				*deformableMaterial, PxTransform(PxVec3(0.0f, 1.85f, 0.0f), orientation),
				PxVec3(1.7f, 0.7f, 0.9f), 160.0f);
			dynamicBoxHalfExtent = PxVec3(0.5f, 0.35f, 0.7f);
			dynamicBox = physics->createRigidDynamic(
				PxTransform(PxVec3(0.0f, separatedRigid ? 3.40f : 2.80f, 0.0f), orientation));
			if(!volume || !dynamicBox || !PxRigidActorExt::createExclusiveShape(
				*dynamicBox, PxBoxGeometry(dynamicBoxHalfExtent), *rigidMaterial) ||
				!PxRigidBodyExt::setMassAndUpdateInertia(*dynamicBox, 10.0f))
				return false;
			dynamicBox->setLinearDamping(0.015f);
			dynamicBox->setAngularDamping(0.05f);
			if(!scene->addActor(*dynamicBox))
				return false;
		}
		else
		{
			const PxReal y = staticBox ? 2.2f : (dynamic ? 2.3f : 2.0f);
			volume = VolumeAvbdSnippet::createCube(*physics, *scene, mesh,
				PxVec3(0.0f, y, 0.0f), 1.0f);
		}
		}
		return extensions && volume != NULL &&
			(!(softSoft || softSoftRecovery || softSoftInitialContainment ||
				rigidSqueeze || freeFallTriplet) ||
				volumeB != NULL) &&
			(!rigidOnSoft || dynamicBox != NULL);
	}
};

bool known(const char* name)
{
	return Snippets::equalsIgnoreCase(name, "ground") ||
		Snippets::equalsIgnoreCase(name, "static-box") ||
		Snippets::equalsIgnoreCase(name, "dynamic-box") ||
		Snippets::equalsIgnoreCase(name, "rigid-on-soft") ||
		Snippets::equalsIgnoreCase(name, "rigid-on-soft-separated") ||
		Snippets::equalsIgnoreCase(name, "buffer-mutation") ||
		Snippets::equalsIgnoreCase(name, "embedded-sync") ||
		Snippets::equalsIgnoreCase(name, "embedded-same-count") ||
		Snippets::equalsIgnoreCase(name, "soft-soft") ||
		Snippets::equalsIgnoreCase(name, "soft-soft-recovery") ||
		Snippets::equalsIgnoreCase(name, "soft-soft-initial-containment") ||
		Snippets::equalsIgnoreCase(name, "rigid-squeeze") ||
		Snippets::equalsIgnoreCase(name, "free-fall-triplet") ||
		Snippets::equalsIgnoreCase(name, "rolling-sphere") ||
		Snippets::equalsIgnoreCase(name, "resting-sphere") ||
		Snippets::equalsIgnoreCase(name, "pedestal-tip") ||
		Snippets::equalsIgnoreCase(name, "pedestal-tip-loaded") ||
		Snippets::equalsIgnoreCase(name, "resting-cone") ||
		Snippets::equalsIgnoreCase(name, "visual-showcase");
}

bool volumeAngularVelocity(PxDeformableVolume& volume, PxVec3& omega)
{
	omega = PxVec3(0.0f);
	const PxTetrahedronMesh* mesh = volume.getSimulationMesh();
	const PxVec4* positions = volume.getSimPositionInvMassBufferD();
	const PxVec4* velocities = volume.getSimVelocityBufferD();
	if(!mesh || !positions || !velocities || !mesh->getNbVertices()) return false;
	PxVec3 centroid(0.0f);
	PxReal totalMass = 0.0f;
	for(PxU32 i = 0; i < mesh->getNbVertices(); ++i)
	{
		if(!positions[i].isFinite() || !velocities[i].isFinite() ||
			positions[i].w <= 0.0f) return false;
		const PxReal mass = 1.0f / positions[i].w;
		centroid += positions[i].getXYZ() * mass;
		totalMass += mass;
	}
	if(totalMass <= 0.0f || !PxIsFinite(totalMass)) return false;
	centroid *= 1.0f / totalMass;
	PxMat33 inertia(PxZero);
	PxVec3 angularMomentum(0.0f);
	for(PxU32 i = 0; i < mesh->getNbVertices(); ++i)
	{
		const PxReal mass = 1.0f / positions[i].w;
		const PxVec3 offset = positions[i].getXYZ() - centroid;
		inertia += (PxMat33::createDiagonal(PxVec3(offset.magnitudeSquared())) -
			PxMat33::outer(offset, offset)) * mass;
		angularMomentum += offset.cross(velocities[i].getXYZ() * mass);
	}
	const PxReal determinant = inertia.getDeterminant();
	if(!PxIsFinite(determinant) || PxAbs(determinant) <= 1.0e-12f) return false;
	omega = inertia.getInverse() * angularMomentum;
	return omega.isFinite();
}

bool deformableKineticState(PxDeformableVolume& volume, PxReal& kineticEnergy,
	PxReal& totalMass, PxVec3& centerVelocity)
{
	kineticEnergy = 0.0f;
	totalMass = 0.0f;
	centerVelocity = PxVec3(0.0f);
	const PxTetrahedronMesh* mesh = volume.getSimulationMesh();
	const PxVec4* positions = volume.getSimPositionInvMassBufferD();
	const PxVec4* velocities = volume.getSimVelocityBufferD();
	if(!mesh || !positions || !velocities || !mesh->getNbVertices()) return false;
	PxF64 energy = 0.0, massSum = 0.0;
	PxVec3 momentum(0.0f);
	for(PxU32 i = 0; i < mesh->getNbVertices(); ++i)
	{
		const PxReal invMass = positions[i].w;
		const PxVec3 velocity = velocities[i].getXYZ();
		if(!PxIsFinite(invMass) || invMass < 0.0f || !velocity.isFinite()) return false;
		if(invMass == 0.0f) continue;
		const PxReal mass = 1.0f / invMass;
		energy += 0.5 * PxF64(mass) * PxF64(velocity.magnitudeSquared());
		massSum += mass;
		momentum += velocity * mass;
	}
	if(!(massSum > 0.0) || !PxIsFinite(PxReal(energy)) ||
		!PxIsFinite(PxReal(massSum))) return false;
	kineticEnergy = PxReal(energy);
	totalMass = PxReal(massSum);
	centerVelocity = momentum * (1.0f / totalMass);
	return centerVelocity.isFinite();
}

bool centroid(const PxVec4* positions, PxU32 count, PxVec3& value)
{
	value = PxVec3(0.0f);
	if(!positions || !count) return false;
	for(PxU32 i = 0; i < count; ++i)
	{
		if(!positions[i].isFinite()) return false;
		value += positions[i].getXYZ();
	}
	value /= PxReal(count);
	return value.isFinite();
}

PxReal maximumTranslationRemovedDisplacement(const PxVec4* positions,
	const PxArray<PxVec3>& rest, const PxVec3& currentCenter,
	const PxVec3& initialCenter, bool& finite)
{
	finite = positions && rest.size();
	PxReal maximum = 0.0f;
	for(PxU32 i = 0; finite && i < rest.size(); ++i)
	{
		finite = positions[i].isFinite();
		if(finite)
			maximum = PxMax(maximum, ((positions[i].getXYZ() - currentCenter) -
				(rest[i] - initialCenter)).magnitude());
	}
	return maximum;
}

bool collectBoundaryFaces(const PxTetrahedronMesh& mesh, PxArray<PxU32>& faces);

PxReal triangleAabbSatSignedGap(const PxVec3& t0, const PxVec3& t1,
	const PxVec3& t2, const PxVec3& halfExtent)
{
	auto axisGap = [&](const PxVec3& axis) -> PxReal
	{
		const PxReal lengthSq = axis.magnitudeSquared();
		if(lengthSq <= 1.0e-16f) return -PX_MAX_F32;
		const PxVec3 n = axis * (1.0f / PxSqrt(lengthSq));
		const PxReal radius = halfExtent.x * PxAbs(n.x) + halfExtent.y * PxAbs(n.y) +
			halfExtent.z * PxAbs(n.z);
		const PxReal d0 = t0.dot(n);
		const PxReal d1 = t1.dot(n);
		const PxReal d2 = t2.dot(n);
		const PxReal tmin = PxMin(d0, PxMin(d1, d2));
		const PxReal tmax = PxMax(d0, PxMax(d1, d2));
		if(tmin > radius) return tmin - radius;
		if(tmax < -radius) return -radius - tmax;
		return -PxMin(radius - tmin, tmax + radius);
	};
	PxReal gap = axisGap(PxVec3(1.0f, 0.0f, 0.0f));
	gap = PxMax(gap, axisGap(PxVec3(0.0f, 1.0f, 0.0f)));
	gap = PxMax(gap, axisGap(PxVec3(0.0f, 0.0f, 1.0f)));
	gap = PxMax(gap, axisGap((t1 - t0).cross(t2 - t0)));
	const PxVec3 edges[3] = { t1 - t0, t2 - t1, t0 - t2 };
	const PxVec3 boxE[3] = {
		PxVec3(1.0f, 0.0f, 0.0f), PxVec3(0.0f, 1.0f, 0.0f), PxVec3(0.0f, 0.0f, 1.0f) };
	for(PxU32 e = 0; e < 3; ++e)
		for(PxU32 b = 0; b < 3; ++b)
			gap = PxMax(gap, axisGap(boxE[b].cross(edges[e])));
	return gap;
}

// Diagnostic only. holes = uninflated AABB. satNear = 15-axis SAT < margin
// with vertices outside the VF shell. inflated = AABB vs half+margin.
PxU32 countApproachingInteriorHoles(PxDeformableVolume& volume,
	const PxTransform& boxPose, const PxVec3& halfExtent, PxReal margin,
	bool& finite, PxReal& minSat, PxU32& satNear, PxReal& minSat15,
	PxU32& inflated)
{
	finite = true;
	minSat = PX_MAX_F32;
	minSat15 = PX_MAX_F32;
	satNear = 0;
	inflated = 0;
	const PxTetrahedronMesh* mesh = volume.getCollisionMesh();
	const PxVec4* positions = volume.getPositionInvMassBufferD();
	PxArray<PxU32> faces;
	if(!mesh || !positions || !collectBoundaryFaces(*mesh, faces))
	{ finite = false; return 0; }
	const PxVec3 inflatedHalf = halfExtent + PxVec3(margin);
	PxU32 holes = 0;
	for(PxU32 face = 0; face < faces.size() / 3; ++face)
	{
		PxVec3 t0, t1, t2;
		for(PxU32 corner = 0; corner < 3; ++corner)
		{
			const PxU32 vertex = faces[3 * face + corner];
			if(vertex >= mesh->getNbVertices() || !positions[vertex].isFinite())
			{ finite = false; return holes; }
			const PxVec3 local = boxPose.transformInv(positions[vertex].getXYZ());
			if(corner == 0) t0 = local;
			else if(corner == 1) t1 = local;
			else t2 = local;
		}
		if(Gu::intersectTriangleBox_Unsafe(PxVec3(0.0f), halfExtent, t0, t1, t2))
			continue;
		auto vertexSdf = [&](const PxVec3& local) -> PxReal
		{
			const PxVec3 q(PxAbs(local.x) - halfExtent.x,
				PxAbs(local.y) - halfExtent.y, PxAbs(local.z) - halfExtent.z);
			return (q.x <= 0.0f && q.y <= 0.0f && q.z <= 0.0f) ?
				PxMax(q.x, PxMax(q.y, q.z)) :
				PxVec3(PxMax(q.x, 0.0f), PxMax(q.y, 0.0f),
					PxMax(q.z, 0.0f)).magnitude();
		};
		const bool vertsOut = vertexSdf(t0) > margin && vertexSdf(t1) > margin &&
			vertexSdf(t2) > margin;
		const PxReal sat15 = triangleAabbSatSignedGap(t0, t1, t2, halfExtent);
		minSat15 = PxMin(minSat15, sat15);
		if(vertsOut && sat15 < margin)
			++satNear;
		const PxVec3 minimum = t0.minimum(t1).minimum(t2);
		const PxVec3 maximum = t0.maximum(t1).maximum(t2);
		const bool aabbInflated =
			!(maximum.x < -inflatedHalf.x || minimum.x > inflatedHalf.x ||
			maximum.y < -inflatedHalf.y || minimum.y > inflatedHalf.y ||
			maximum.z < -inflatedHalf.z || minimum.z > inflatedHalf.z);
		if(vertsOut && aabbInflated)
			++inflated;
		if(maximum.x < -halfExtent.x || minimum.x > halfExtent.x ||
			maximum.y < -halfExtent.y || minimum.y > halfExtent.y ||
			maximum.z < -halfExtent.z || minimum.z > halfExtent.z)
			continue;
		if(!vertsOut)
			continue;
		++holes;
		minSat = PxMin(minSat, sat15);
	}
	return holes;
}

bool readTet(const PxTetrahedronMesh& mesh, PxU32 tet, PxU32 (&indices)[4])
{
	if(tet >= mesh.getNbTetrahedrons() || !mesh.getTetrahedrons()) return false;
	if(mesh.getTetrahedronMeshFlags().isSet(PxTetrahedronMeshFlag::e16_BIT_INDICES))
	{
		const PxU16* source = static_cast<const PxU16*>(mesh.getTetrahedrons());
		for(PxU32 i = 0; i < 4; ++i) indices[i] = source[4 * tet + i];
	}
	else
	{
		const PxU32* source = static_cast<const PxU32*>(mesh.getTetrahedrons());
		for(PxU32 i = 0; i < 4; ++i) indices[i] = source[4 * tet + i];
	}
	for(PxU32 i = 0; i < 4; ++i)
		if(indices[i] >= mesh.getNbVertices()) return false;
	return true;
}

PxReal minimumTetDetF(const PxTetrahedronMesh& mesh, const PxVec4* positions,
	const PxArray<PxVec3>& rest, bool& finite)
{
	finite = positions != NULL && rest.size() == mesh.getNbVertices();
	if(!finite) return -PX_MAX_F32;
	PxReal minimum = PX_MAX_F32;
	for(PxU32 tet = 0; tet < mesh.getNbTetrahedrons(); ++tet)
	{
		PxU32 ids[4];
		if(!readTet(mesh, tet, ids)) { finite = false; return minimum; }
		for(PxU32 corner = 0; corner < 4; ++corner)
			if(ids[corner] >= rest.size() || !positions[ids[corner]].isFinite())
			{ finite = false; return minimum; }
		const PxMat33 restBasis(rest[ids[1]] - rest[ids[0]],
			rest[ids[2]] - rest[ids[0]], rest[ids[3]] - rest[ids[0]]);
		const PxReal restDet = restBasis.getDeterminant();
		if(!PxIsFinite(restDet) || PxAbs(restDet) <= 1.0e-12f)
		{ finite = false; return minimum; }
		const PxVec3 p0 = positions[ids[0]].getXYZ();
		const PxMat33 currentBasis(positions[ids[1]].getXYZ() - p0,
			positions[ids[2]].getXYZ() - p0, positions[ids[3]].getXYZ() - p0);
		const PxReal detF = currentBasis.getDeterminant() / restDet;
		if(!PxIsFinite(detF)) { finite = false; return minimum; }
		minimum = PxMin(minimum, detF);
	}
	return minimum;
}

PxReal maximumTetEdgeStrain(const PxTetrahedronMesh& mesh,
	const PxVec4* positions, const PxArray<PxVec3>& rest, bool& finite)
{
	finite = positions != NULL && rest.size() == mesh.getNbVertices();
	PxReal maximum = 0.0f;
	static const PxU32 edges[6][2] =
	{
		{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
	};
	for(PxU32 tet = 0; finite && tet < mesh.getNbTetrahedrons(); ++tet)
	{
		PxU32 ids[4];
		if(!readTet(mesh, tet, ids)) { finite = false; break; }
		for(PxU32 edge = 0; edge < 6; ++edge)
		{
			const PxU32 a = ids[edges[edge][0]], b = ids[edges[edge][1]];
			if(a >= rest.size() || b >= rest.size() || !positions[a].isFinite() ||
				!positions[b].isFinite()) { finite = false; break; }
			const PxReal restLength = (rest[b] - rest[a]).magnitude();
			if(restLength <= 1.0e-8f) { finite = false; break; }
			const PxReal currentLength =
				(positions[b].getXYZ() - positions[a].getXYZ()).magnitude();
			maximum = PxMax(maximum, PxAbs(currentLength - restLength) / restLength);
		}
	}
	return maximum;
}

bool pointStrictlyInsideTet(const PxVec3& point, const PxVec3& a,
	const PxVec3& b, const PxVec3& c, const PxVec3& d)
{
	const PxMat33 basis(b - a, c - a, d - a);
	const PxReal determinant = basis.getDeterminant();
	if(PxAbs(determinant) <= 1.0e-12f) return false;
	const PxVec3 weights123 = basis.getInverse() * (point - a);
	const PxReal weights[4] = {1.0f - weights123.x - weights123.y - weights123.z,
		weights123.x, weights123.y, weights123.z};
	const PxReal interior = 1.0e-4f;
	return weights[0] > interior && weights[1] > interior &&
		weights[2] > interior && weights[3] > interior;
}

bool collectBoundaryFaces(const PxTetrahedronMesh& mesh, PxArray<PxU32>& faces)
{
	struct FaceUse { PxU32 v[3]; PxU32 count; };
	PxArray<FaceUse> uses;
	static const PxU32 faceCorners[4][3] =
	{
		{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}
	};
	for(PxU32 tet = 0; tet < mesh.getNbTetrahedrons(); ++tet)
	{
		PxU32 ids[4];
		if(!readTet(mesh, tet, ids)) return false;
		for(PxU32 face = 0; face < 4; ++face)
		{
			PxU32 key[3] = {ids[faceCorners[face][0]], ids[faceCorners[face][1]],
				ids[faceCorners[face][2]]};
			if(key[0] > key[1]) PxSwap(key[0], key[1]);
			if(key[1] > key[2]) PxSwap(key[1], key[2]);
			if(key[0] > key[1]) PxSwap(key[0], key[1]);
			PxU32 existing = PX_MAX_U32;
			for(PxU32 i = 0; i < uses.size(); ++i)
				if(uses[i].v[0] == key[0] && uses[i].v[1] == key[1] &&
					uses[i].v[2] == key[2]) { existing = i; break; }
			if(existing == PX_MAX_U32)
			{
				FaceUse use = {{key[0], key[1], key[2]}, 1};
				uses.pushBack(use);
			}
			else ++uses[existing].count;
		}
	}
	faces.clear();
	for(PxU32 i = 0; i < uses.size(); ++i)
		if(uses[i].count == 1)
			for(PxU32 corner = 0; corner < 3; ++corner)
				faces.pushBack(uses[i].v[corner]);
	return !faces.empty();
}

PxU64 meshFingerprint(const PxTetrahedronMesh& mesh, bool includeGeometry)
{
	PxU64 hash = 14695981039346656037ull;
	auto add = [&](PxU32 value)
	{
		for(PxU32 byte = 0; byte < 4; ++byte)
		{
			hash ^= (value >> (8 * byte)) & 0xff;
			hash *= 1099511628211ull;
		}
	};
	add(mesh.getNbVertices()); add(mesh.getNbTetrahedrons());
	for(PxU32 tet = 0; tet < mesh.getNbTetrahedrons(); ++tet)
	{
		PxU32 ids[4];
		if(!readTet(mesh, tet, ids)) return 0;
		for(PxU32 corner = 0; corner < 4; ++corner) add(ids[corner]);
	}
	if(includeGeometry)
		for(PxU32 vertex = 0; vertex < mesh.getNbVertices(); ++vertex)
			for(PxU32 axis = 0; axis < 3; ++axis)
			{
				PxU32 bits;
				const PxReal value = mesh.getVertices()[vertex][axis];
				std::memcpy(&bits, &value, sizeof(bits));
				add(bits);
			}
	return hash;
}

void reportVisualMeshDiagnostics(PxDeformableVolume& volume, PxU32 volumeIndex)
{
	const PxTetrahedronMesh* simulation=volume.getSimulationMesh();
	const PxTetrahedronMesh* collision=volume.getCollisionMesh();
	const PxVec4* simulationPositions=volume.getSimPositionInvMassBufferD();
	const PxVec4* collisionPositions=volume.getPositionInvMassBufferD();
	if(!simulation || !collision || !simulationPositions || !collisionPositions)
		return;
	PxReal minimumSixVolume=PX_MAX_F32,totalVolume=0.0f,totalMass=0.0f;
	PxU32 invalidTets=0;
	for(PxU32 vertex=0;vertex<simulation->getNbVertices();++vertex)
		if(simulationPositions[vertex].w>0.0f)
			totalMass+=1.0f/simulationPositions[vertex].w;
	for(PxU32 tet=0;tet<simulation->getNbTetrahedrons();++tet)
	{
		PxU32 ids[4];
		if(!readTet(*simulation,tet,ids)) { ++invalidTets; continue; }
		const PxVec3 a=simulationPositions[ids[0]].getXYZ();
		const PxReal sixVolume=PxAbs((simulationPositions[ids[1]].getXYZ()-a).dot(
			(simulationPositions[ids[2]].getXYZ()-a).cross(
				simulationPositions[ids[3]].getXYZ()-a)));
		if(!PxIsFinite(sixVolume)) ++invalidTets;
		else
		{
			minimumSixVolume=PxMin(minimumSixVolume,sixVolume);
			totalVolume+=sixVolume/6.0f;
		}
	}
	PxArray<PxU32> faces;
	PxReal minimumDoubleArea=PX_MAX_F32;
	PxU32 invalidFaces=0;
	if(collectBoundaryFaces(*collision,faces))
		for(PxU32 face=0;face<faces.size()/3;++face)
		{
			const PxVec3 a=collisionPositions[faces[face*3]].getXYZ();
			const PxReal doubleArea=(collisionPositions[faces[face*3+1]].getXYZ()-a).cross(
				collisionPositions[faces[face*3+2]].getXYZ()-a).magnitude();
			if(!PxIsFinite(doubleArea) || doubleArea<=0.0f) ++invalidFaces;
			else minimumDoubleArea=PxMin(minimumDoubleArea,doubleArea);
		}
	std::printf("[AVBD_MESH] volume=%u simVertices=%u simTets=%u collisionVertices=%u collisionTets=%u boundaryFaces=%u minSixVolume=%.9g totalVolume=%.9g totalMass=%.9g minBoundaryDoubleArea=%.9g invalidTets=%u invalidFaces=%u\n",
		volumeIndex,simulation->getNbVertices(),simulation->getNbTetrahedrons(),
		collision->getNbVertices(),collision->getNbTetrahedrons(),faces.size()/3,
		double(minimumSixVolume),double(totalVolume),double(totalMass),
		double(minimumDoubleArea),invalidTets,invalidFaces);
	std::printf("[AVBD_MESH_HASH] volume=%u simTopology=%016llx collisionTopology=%016llx simRestGeometry=%016llx collisionRestGeometry=%016llx\n",
		volumeIndex, static_cast<unsigned long long>(meshFingerprint(*simulation, false)),
		static_cast<unsigned long long>(meshFingerprint(*collision, false)),
		static_cast<unsigned long long>(meshFingerprint(*simulation, true)),
		static_cast<unsigned long long>(meshFingerprint(*collision, true)));
	std::fflush(stdout);
}

PxReal minCollisionVertexBoxSdf(PxDeformableVolume& volume,
	const PxTransform& boxPose, const PxVec3& halfExtent, bool& finite)
{
	finite = true;
	const PxTetrahedronMesh* mesh = volume.getCollisionMesh();
	const PxVec4* positions = volume.getPositionInvMassBufferD();
	if(!mesh || !positions)
	{
		finite = false;
		return PX_MAX_F32;
	}
	PxReal minSdf = PX_MAX_F32;
	const PxU32 count = mesh->getNbVertices();
	for(PxU32 vertex = 0; vertex < count; ++vertex)
	{
		if(!positions[vertex].isFinite())
		{
			finite = false;
			return minSdf;
		}
		const PxVec3 local = boxPose.transformInv(positions[vertex].getXYZ());
		const PxVec3 q(PxAbs(local.x) - halfExtent.x,
			PxAbs(local.y) - halfExtent.y, PxAbs(local.z) - halfExtent.z);
		const PxReal sdf = (q.x <= 0.0f && q.y <= 0.0f && q.z <= 0.0f) ?
			PxMax(q.x, PxMax(q.y, q.z)) :
			PxVec3(PxMax(q.x, 0.0f), PxMax(q.y, 0.0f), PxMax(q.z, 0.0f)).magnitude();
		minSdf = PxMin(minSdf, sdf);
	}
	return minSdf;
}

PxU32 countBoxCoreTriangleIntersections(PxDeformableVolume& volume,
	const PxTransform& boxPose, const PxVec3& halfExtent, bool& finite)
{
	finite = true;
	const PxTetrahedronMesh* mesh = volume.getCollisionMesh();
	const PxVec4* positions = volume.getPositionInvMassBufferD();
	PxArray<PxU32> faces;
	if(!mesh || !positions || !collectBoundaryFaces(*mesh, faces))
	{ finite = false; return 0; }
	PxU32 hits = 0;
	for(PxU32 face = 0; face < faces.size() / 3; ++face)
	{
		PxVec3 triangle[3];
		for(PxU32 corner = 0; corner < 3; ++corner)
		{
			const PxU32 vertex = faces[3 * face + corner];
			if(vertex >= mesh->getNbVertices() || !positions[vertex].isFinite())
			{ finite = false; return hits; }
			triangle[corner] = boxPose.transformInv(positions[vertex].getXYZ());
		}
		if(Gu::intersectTriangleBox_Unsafe(PxVec3(0.0f), halfExtent,
			triangle[0], triangle[1], triangle[2])) ++hits;
	}
	return hits;
}

bool segmentProperlyCrossesTriangle(const PxVec3& p0, const PxVec3& p1,
	const PxVec3& a, const PxVec3& b, const PxVec3& c);

PxU32 countBoxEdgeTriangleCrossings(PxDeformableVolume& volume,
	const PxTransform& boxPose, const PxVec3& halfExtent, bool& finite)
{
	finite = true;
	const PxTetrahedronMesh* mesh = volume.getCollisionMesh();
	const PxVec4* positions = volume.getPositionInvMassBufferD();
	PxArray<PxU32> faces;
	if(!mesh || !positions || !collectBoundaryFaces(*mesh, faces))
	{ finite = false; return 0; }
	static const PxU32 edges[12][2] =
	{
		{0,1},{2,3},{4,5},{6,7}, {0,2},{1,3},{4,6},{5,7},
		{0,4},{1,5},{2,6},{3,7}
	};
	auto boxVertex = [&](PxU32 index)
	{
		return boxPose.transform(PxVec3((index & 1) ? halfExtent.x : -halfExtent.x,
			(index & 2) ? halfExtent.y : -halfExtent.y,
			(index & 4) ? halfExtent.z : -halfExtent.z));
	};
	PxU32 crossings = 0;
	for(PxU32 face = 0; face < faces.size() / 3; ++face)
	{
		PxVec3 triangle[3];
		for(PxU32 corner = 0; corner < 3; ++corner)
		{
			const PxU32 vertex = faces[3 * face + corner];
			if(vertex >= mesh->getNbVertices() || !positions[vertex].isFinite())
			{ finite = false; return crossings; }
			triangle[corner] = positions[vertex].getXYZ();
		}
		for(PxU32 edge = 0; edge < 12; ++edge)
		{
			if(!segmentProperlyCrossesTriangle(boxVertex(edges[edge][0]),
				boxVertex(edges[edge][1]), triangle[0], triangle[1], triangle[2]))
				continue;
			++crossings;
			break;
		}
	}
	return crossings;
}

// Hanging centroid outside the full box while the clip centroid is inside.
// Diagnostic only: leftover-open vs core-hit timing, not solver geometry.
PxU32 countLeftoverOpenTriangles(PxDeformableVolume& volume,
	const PxTransform& boxPose, const PxVec3& halfExtent, bool& finite,
	PxReal& minClipSdf, PxReal& minChebyshev, PxReal& maxChebyshev,
	PxU32& intersecting, PxU32& covering, PxReal& minCoveringHangingSdf)
{
	finite = true;
	minClipSdf = PX_MAX_F32;
	minChebyshev = PX_MAX_F32;
	maxChebyshev = 0.0f;
	intersecting = 0;
	covering = 0;
	minCoveringHangingSdf = PX_MAX_F32;
	const PxTetrahedronMesh* mesh = volume.getCollisionMesh();
	const PxVec4* positions = volume.getPositionInvMassBufferD();
	PxArray<PxU32> faces;
	if(!mesh || !positions || !collectBoundaryFaces(*mesh, faces))
	{ finite = false; return 0; }
	PxU32 leftover = 0;
	for(PxU32 face = 0; face < faces.size() / 3; ++face)
	{
		PxVec3 t0, t1, t2;
		for(PxU32 corner = 0; corner < 3; ++corner)
		{
			const PxU32 vertex = faces[3 * face + corner];
			if(vertex >= mesh->getNbVertices() || !positions[vertex].isFinite())
			{ finite = false; return leftover; }
			const PxVec3 local = boxPose.transformInv(positions[vertex].getXYZ());
			if(corner == 0) t0 = local;
			else if(corner == 1) t1 = local;
			else t2 = local;
		}
		if(!Gu::intersectTriangleBox_Unsafe(PxVec3(0.0f), halfExtent, t0, t1, t2))
			continue;
		++intersecting;
		const PxVec3 hanging = (t0 + t1 + t2) * (1.0f / 3.0f);
		const PxVec3 hangingQ(PxAbs(hanging.x) - halfExtent.x,
			PxAbs(hanging.y) - halfExtent.y, PxAbs(hanging.z) - halfExtent.z);
		const PxReal hangingSdf = (hangingQ.x <= 0.0f && hangingQ.y <= 0.0f &&
			hangingQ.z <= 0.0f) ? PxMax(hangingQ.x, PxMax(hangingQ.y, hangingQ.z)) :
			PxVec3(PxMax(hangingQ.x, 0.0f), PxMax(hangingQ.y, 0.0f),
				PxMax(hangingQ.z, 0.0f)).magnitude();
		if(hangingSdf < -1.0e-4f)
		{
			++covering;
			minCoveringHangingSdf = PxMin(minCoveringHangingSdf, hangingSdf);
		}
		struct ClipVertex { PxVec3 local; };
		ClipVertex clipIn[16] = {{t0},{t1},{t2}};
		ClipVertex clipOut[16];
		PxU32 clipCount = 3;
		bool clipValid = true;
		for(PxU32 axis = 0; axis < 3 && clipValid; ++axis)
		{
			for(PxU32 side = 0; side < 2 && clipValid; ++side)
			{
				const PxReal bound = side == 0 ? halfExtent[axis] : -halfExtent[axis];
				const bool upper = side == 0;
				PxU32 outCount = 0;
				ClipVertex previous = clipIn[clipCount - 1];
				bool previousInside = upper ? previous.local[axis] <= bound :
					previous.local[axis] >= bound;
				for(PxU32 vertex = 0; vertex < clipCount; ++vertex)
				{
					const ClipVertex current = clipIn[vertex];
					const bool currentInside = upper ? current.local[axis] <= bound :
						current.local[axis] >= bound;
					if(currentInside != previousInside)
					{
						const PxReal denom = current.local[axis] - previous.local[axis];
						if(PxAbs(denom) <= 1.0e-12f || outCount >= 16)
						{
							clipValid = false;
							break;
						}
						const PxReal t = PxClamp((bound - previous.local[axis]) / denom,
							0.0f, 1.0f);
						clipOut[outCount++].local = previous.local +
							(current.local - previous.local) * t;
					}
					if(currentInside)
					{
						if(outCount >= 16)
						{
							clipValid = false;
							break;
						}
						clipOut[outCount++] = current;
					}
					previous = current;
					previousInside = currentInside;
				}
				if(!clipValid || outCount < 3)
				{
					clipValid = false;
					break;
				}
				clipCount = outCount;
				for(PxU32 vertex = 0; vertex < clipCount; ++vertex)
					clipIn[vertex] = clipOut[vertex];
			}
		}
		PxReal clipSdf = PX_MAX_F32;
		if(clipValid && clipCount >= 3)
		{
			PxVec3 clipLocal(0.0f);
			for(PxU32 vertex = 0; vertex < clipCount; ++vertex)
				clipLocal += clipIn[vertex].local;
			clipLocal *= 1.0f / PxReal(clipCount);
			const PxVec3 clipQ(PxAbs(clipLocal.x) - halfExtent.x,
				PxAbs(clipLocal.y) - halfExtent.y,
				PxAbs(clipLocal.z) - halfExtent.z);
			clipSdf = (clipQ.x <= 0.0f && clipQ.y <= 0.0f && clipQ.z <= 0.0f) ?
				PxMax(clipQ.x, PxMax(clipQ.y, clipQ.z)) :
				PxVec3(PxMax(clipQ.x, 0.0f), PxMax(clipQ.y, 0.0f),
					PxMax(clipQ.z, 0.0f)).magnitude();
		}
		if(hangingSdf > 1.0e-4f && clipSdf < -1.0e-4f)
		{
			++leftover;
			minClipSdf = PxMin(minClipSdf, clipSdf);
			PxReal lo = 0.0f;
			PxReal hi = PxMin(halfExtent.x, PxMin(halfExtent.y, halfExtent.z));
			for(PxU32 iteration = 0; iteration < 16; ++iteration)
			{
				const PxReal mid = 0.5f * (lo + hi);
				const PxVec3 shrunk(halfExtent.x - mid, halfExtent.y - mid,
					halfExtent.z - mid);
				if(shrunk.x <= 1.0e-8f || shrunk.y <= 1.0e-8f || shrunk.z <= 1.0e-8f)
				{
					hi = mid;
					continue;
				}
				if(Gu::intersectTriangleBox_Unsafe(PxVec3(0.0f), shrunk, t0, t1, t2))
					lo = mid;
				else
					hi = mid;
			}
			minChebyshev = PxMin(minChebyshev, lo);
			maxChebyshev = PxMax(maxChebyshev, lo);
		}
	}
	return leftover;
}

void dumpBoxCoreTriangleHits(const char* volumeName, PxU32 frame,
	PxDeformableVolume& volume, const PxRigidDynamic& box,
	const PxVec3& halfExtent, const PxVec3& coreExtent)
{
	const PxTetrahedronMesh* mesh = volume.getCollisionMesh();
	const PxVec4* positions = volume.getPositionInvMassBufferD();
	PxArray<PxU32> faces;
	if(!mesh || !positions || !collectBoundaryFaces(*mesh, faces))
		return;
	const PxTransform pose = box.getGlobalPose();
	const PxVec3 omega = box.getAngularVelocity();
	const PxVec3 lin = box.getLinearVelocity();
	static const PxU32 edges[12][2] =
	{
		{0,1},{2,3},{4,5},{6,7}, {0,2},{1,3},{4,6},{5,7},
		{0,4},{1,5},{2,6},{3,7}
	};
	auto boxVertex = [&](PxU32 index, const PxVec3& extent)
	{
		return pose.transform(PxVec3((index & 1) ? extent.x : -extent.x,
			(index & 2) ? extent.y : -extent.y,
			(index & 4) ? extent.z : -extent.z));
	};
	PxU32 dumped = 0;
	for(PxU32 face = 0; face < faces.size() / 3; ++face)
	{
		PxVec3 triangle[3];
		PxU32 verticesInsideFull = 0, verticesInsideCore = 0;
		for(PxU32 corner = 0; corner < 3; ++corner)
		{
			const PxU32 vertex = faces[3 * face + corner];
			if(vertex >= mesh->getNbVertices() || !positions[vertex].isFinite())
				return;
			triangle[corner] = positions[vertex].getXYZ();
			const PxVec3 local = pose.transformInv(triangle[corner]);
			if(PxAbs(local.x) <= halfExtent.x && PxAbs(local.y) <= halfExtent.y &&
				PxAbs(local.z) <= halfExtent.z)
				++verticesInsideFull;
			if(PxAbs(local.x) <= coreExtent.x && PxAbs(local.y) <= coreExtent.y &&
				PxAbs(local.z) <= coreExtent.z)
				++verticesInsideCore;
		}
		const PxVec3 t0 = pose.transformInv(triangle[0]);
		const PxVec3 t1 = pose.transformInv(triangle[1]);
		const PxVec3 t2 = pose.transformInv(triangle[2]);
		if(!Gu::intersectTriangleBox_Unsafe(PxVec3(0.0f), coreExtent, t0, t1, t2))
			continue;
		PxU32 edgeCrossings = 0;
		PxVec3 hitLocal(0.0f);
		for(PxU32 edge = 0; edge < 12; ++edge)
		{
			const PxVec3 p0 = boxVertex(edges[edge][0], halfExtent);
			const PxVec3 p1 = boxVertex(edges[edge][1], halfExtent);
			if(!segmentProperlyCrossesTriangle(p0, p1, triangle[0], triangle[1],
				triangle[2]))
				continue;
			++edgeCrossings;
			hitLocal = pose.transformInv(p0 + (p1 - p0) * 0.5f);
		}
		const PxVec3 minimum = t0.minimum(t1).minimum(t2);
		const PxVec3 maximum = t0.maximum(t1).maximum(t2);
		const PxReal exits[6] =
		{
			halfExtent.x - minimum.x, maximum.x + halfExtent.x,
			halfExtent.y - minimum.y, maximum.y + halfExtent.y,
			halfExtent.z - minimum.z, maximum.z + halfExtent.z
		};
		PxU32 exitFace = 0;
		for(PxU32 candidate = 1; candidate < 6; ++candidate)
			if(exits[candidate] < exits[exitFace])
				exitFace = candidate;
		const PxVec3 hangingLocal = (t0 + t1 + t2) * (1.0f / 3.0f);
		const PxVec3 hangingQ(PxAbs(hangingLocal.x) - halfExtent.x,
			PxAbs(hangingLocal.y) - halfExtent.y,
			PxAbs(hangingLocal.z) - halfExtent.z);
		const PxReal hangingSdf = (hangingQ.x <= 0.0f && hangingQ.y <= 0.0f &&
			hangingQ.z <= 0.0f) ? PxMax(hangingQ.x, PxMax(hangingQ.y, hangingQ.z)) :
			PxVec3(PxMax(hangingQ.x, 0.0f), PxMax(hangingQ.y, 0.0f),
				PxMax(hangingQ.z, 0.0f)).magnitude();
		struct ClipVertex { PxVec3 local; };
		ClipVertex clipIn[16] = {{t0},{t1},{t2}};
		ClipVertex clipOut[16];
		PxU32 clipCount = 3;
		bool clipValid = true;
		for(PxU32 axis = 0; axis < 3 && clipValid; ++axis)
		{
			for(PxU32 side = 0; side < 2 && clipValid; ++side)
			{
				const PxReal bound = side == 0 ? halfExtent[axis] : -halfExtent[axis];
				const bool upper = side == 0;
				PxU32 outCount = 0;
				ClipVertex previous = clipIn[clipCount - 1];
				bool previousInside = upper ? previous.local[axis] <= bound :
					previous.local[axis] >= bound;
				for(PxU32 vertex = 0; vertex < clipCount; ++vertex)
				{
					const ClipVertex current = clipIn[vertex];
					const bool currentInside = upper ? current.local[axis] <= bound :
						current.local[axis] >= bound;
					if(currentInside != previousInside)
					{
						const PxReal denom = current.local[axis] - previous.local[axis];
						if(PxAbs(denom) <= 1.0e-12f || outCount >= 16)
						{
							clipValid = false;
							break;
						}
						const PxReal t = PxClamp((bound - previous.local[axis]) / denom,
							0.0f, 1.0f);
						clipOut[outCount++].local = previous.local +
							(current.local - previous.local) * t;
					}
					if(currentInside)
					{
						if(outCount >= 16)
						{
							clipValid = false;
							break;
						}
						clipOut[outCount++] = current;
					}
					previous = current;
					previousInside = currentInside;
				}
				if(!clipValid || outCount < 3)
				{
					clipValid = false;
					break;
				}
				clipCount = outCount;
				for(PxU32 vertex = 0; vertex < clipCount; ++vertex)
					clipIn[vertex] = clipOut[vertex];
			}
		}
		PxVec3 clipLocal(0.0f);
		PxReal clipSdf = PX_MAX_F32;
		if(clipValid && clipCount >= 3)
		{
			for(PxU32 vertex = 0; vertex < clipCount; ++vertex)
				clipLocal += clipIn[vertex].local;
			clipLocal *= 1.0f / PxReal(clipCount);
			const PxVec3 clipQ(PxAbs(clipLocal.x) - halfExtent.x,
				PxAbs(clipLocal.y) - halfExtent.y,
				PxAbs(clipLocal.z) - halfExtent.z);
			clipSdf = (clipQ.x <= 0.0f && clipQ.y <= 0.0f && clipQ.z <= 0.0f) ?
				PxMax(clipQ.x, PxMax(clipQ.y, clipQ.z)) :
				PxVec3(PxMax(clipQ.x, 0.0f), PxMax(clipQ.y, 0.0f),
					PxMax(clipQ.z, 0.0f)).magnitude();
		}
		std::printf("[AVBD_C01B] frame=%u volume=%s boxY=%.6g |w|=%.6g |v|=%.6g tri=%u verticesInsideFull=%u verticesInsideCore=%u edgeCrossings=%u hitLocal=%.6g,%.6g,%.6g exitFace=%u exit=%.6g clipLocal=%.6g,%.6g,%.6g clipSdf=%.6g hangingSdf=%.6g v0=%.6g,%.6g,%.6g v1=%.6g,%.6g,%.6g v2=%.6g,%.6g,%.6g\n",
			frame, volumeName, double(pose.p.y), double(omega.magnitude()),
			double(lin.magnitude()), face, verticesInsideFull,
			verticesInsideCore, edgeCrossings,
			double(hitLocal.x), double(hitLocal.y), double(hitLocal.z),
			exitFace, double(exits[exitFace]),
			double(clipLocal.x), double(clipLocal.y), double(clipLocal.z),
			double(clipSdf), double(hangingSdf),
			double(t0.x), double(t0.y), double(t0.z),
			double(t1.x), double(t1.y), double(t1.z),
			double(t2.x), double(t2.y), double(t2.z));
		if(++dumped >= 8)
			break;
	}
}

bool segmentProperlyCrossesTriangle(const PxVec3& p0, const PxVec3& p1,
	const PxVec3& a, const PxVec3& b, const PxVec3& c)
{
	const PxVec3 direction = p1 - p0;
	const PxVec3 edge0 = b - a, edge1 = c - a;
	const PxVec3 cross = direction.cross(edge1);
	const PxReal determinant = edge0.dot(cross);
	if(PxAbs(determinant) <= 1.0e-10f) return false;
	const PxReal inverse = 1.0f / determinant;
	const PxVec3 offset = p0 - a;
	const PxReal u = offset.dot(cross) * inverse;
	const PxVec3 q = offset.cross(edge0);
	const PxReal v = direction.dot(q) * inverse;
	const PxReal t = edge1.dot(q) * inverse;
	const PxReal epsilon = 1.0e-4f;
	return t > epsilon && t < 1.0f - epsilon && u > epsilon &&
		v > epsilon && u + v < 1.0f - epsilon;
}

PxU32 countProperBoundaryCrossings(PxDeformableVolume& a,
	PxDeformableVolume& b, bool& finite)
{
	finite = true;
	const PxTetrahedronMesh* meshA = a.getCollisionMesh();
	const PxTetrahedronMesh* meshB = b.getCollisionMesh();
	const PxVec4* positionsA = a.getPositionInvMassBufferD();
	const PxVec4* positionsB = b.getPositionInvMassBufferD();
	PxArray<PxU32> facesA, facesB;
	if(!meshA || !meshB || !positionsA || !positionsB ||
		!collectBoundaryFaces(*meshA, facesA) || !collectBoundaryFaces(*meshB, facesB))
	{ finite = false; return 0; }
	PxU32 crossings = 0;
	for(PxU32 ia = 0; ia < facesA.size() / 3; ++ia)
		for(PxU32 ib = 0; ib < facesB.size() / 3; ++ib)
		{
			PxVec3 ta[3], tb[3];
			for(PxU32 corner = 0; corner < 3; ++corner)
			{
				const PxU32 va = facesA[3 * ia + corner];
				const PxU32 vb = facesB[3 * ib + corner];
				if(va >= meshA->getNbVertices() || vb >= meshB->getNbVertices() ||
					!positionsA[va].isFinite() || !positionsB[vb].isFinite())
				{ finite = false; return crossings; }
				ta[corner] = positionsA[va].getXYZ();
				tb[corner] = positionsB[vb].getXYZ();
			}
			bool crossed = false;
			for(PxU32 edge = 0; edge < 3 && !crossed; ++edge)
				crossed = segmentProperlyCrossesTriangle(ta[edge], ta[(edge + 1) % 3],
					tb[0], tb[1], tb[2]) ||
					segmentProperlyCrossesTriangle(tb[edge], tb[(edge + 1) % 3],
						ta[0], ta[1], ta[2]);
			if(crossed) ++crossings;
		}
	return crossings;
}

PxU32 countInsideCollisionVertices(PxDeformableVolume& query,
	PxDeformableVolume& target, bool& finite)
{
	finite = true;
	const PxTetrahedronMesh* queryMesh = query.getCollisionMesh();
	const PxTetrahedronMesh* targetMesh = target.getCollisionMesh();
	const PxVec4* queryPositions = query.getPositionInvMassBufferD();
	const PxVec4* targetPositions = target.getPositionInvMassBufferD();
	if(!queryMesh || !targetMesh || !queryPositions || !targetPositions)
	{
		finite = false;
		return 0;
	}
	const PxU32 queryCount = queryMesh->getNbVertices();
	const PxU32 targetCount = targetMesh->getNbVertices();
	for(PxU32 i = 0; i < queryCount; ++i)
		if(!queryPositions[i].isFinite()) { finite = false; return 0; }
	for(PxU32 i = 0; i < targetCount; ++i)
		if(!targetPositions[i].isFinite()) { finite = false; return 0; }
	struct FaceUse { PxU32 v[3]; PxU32 count; };
	PxArray<FaceUse> faceUses;
	static const PxU32 faceCorners[4][3] =
	{
		{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}
	};
	for(PxU32 tet = 0; tet < queryMesh->getNbTetrahedrons(); ++tet)
	{
		PxU32 ids[4];
		if(!readTet(*queryMesh, tet, ids)) { finite = false; return 0; }
		for(PxU32 face = 0; face < 4; ++face)
		{
			PxU32 key[3] = {ids[faceCorners[face][0]], ids[faceCorners[face][1]],
				ids[faceCorners[face][2]]};
			if(key[0] > key[1]) PxSwap(key[0], key[1]);
			if(key[1] > key[2]) PxSwap(key[1], key[2]);
			if(key[0] > key[1]) PxSwap(key[0], key[1]);
			PxU32 existing = PX_MAX_U32;
			for(PxU32 i = 0; i < faceUses.size(); ++i)
				if(faceUses[i].v[0] == key[0] && faceUses[i].v[1] == key[1] &&
					faceUses[i].v[2] == key[2]) { existing = i; break; }
			if(existing == PX_MAX_U32)
			{
				FaceUse use = {{key[0], key[1], key[2]}, 1};
				faceUses.pushBack(use);
			}
			else ++faceUses[existing].count;
		}
	}
	PxArray<PxU8> boundaryVertex;
	boundaryVertex.resize(queryCount);
	PxMemZero(boundaryVertex.begin(), boundaryVertex.size() * sizeof(PxU8));
	for(PxU32 face = 0; face < faceUses.size(); ++face)
		if(faceUses[face].count == 1)
			for(PxU32 corner = 0; corner < 3; ++corner)
				if(faceUses[face].v[corner] < queryCount)
					boundaryVertex[faceUses[face].v[corner]] = 1;
	PxU32 inside = 0;
	for(PxU32 vertex = 0; vertex < queryCount; ++vertex)
	{
		if(!boundaryVertex[vertex]) continue;
		for(PxU32 tet = 0; tet < targetMesh->getNbTetrahedrons(); ++tet)
		{
			PxU32 indices[4];
			if(!readTet(*targetMesh, tet, indices)) { finite = false; return inside; }
			if(indices[0] >= targetCount || indices[1] >= targetCount ||
				indices[2] >= targetCount || indices[3] >= targetCount)
			{
				finite = false;
				return inside;
			}
			if(pointStrictlyInsideTet(queryPositions[vertex].getXYZ(),
				targetPositions[indices[0]].getXYZ(), targetPositions[indices[1]].getXYZ(),
				targetPositions[indices[2]].getXYZ(), targetPositions[indices[3]].getXYZ()))
			{
				++inside;
				break;
			}
		}
	}
	return inside;
}

void closestPointTriangle(const PxVec3& point, const PxVec3& a,
	const PxVec3& b, const PxVec3& c, PxVec3& closest)
{
	const PxVec3 ab = b - a, ac = c - a, ap = point - a;
	const PxReal d1 = ab.dot(ap), d2 = ac.dot(ap);
	if(d1 <= 0.0f && d2 <= 0.0f) { closest = a; return; }
	const PxVec3 bp = point - b;
	const PxReal d3 = ab.dot(bp), d4 = ac.dot(bp);
	if(d3 >= 0.0f && d4 <= d3) { closest = b; return; }
	const PxReal vc = d1 * d4 - d3 * d2;
	if(vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
	{
		closest = a + ab * (d1 / (d1 - d3)); return;
	}
	const PxVec3 cp = point - c;
	const PxReal d5 = ab.dot(cp), d6 = ac.dot(cp);
	if(d6 >= 0.0f && d5 <= d6) { closest = c; return; }
	const PxReal vb = d5 * d2 - d1 * d6;
	if(vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
	{
		closest = a + ac * (d2 / (d2 - d6)); return;
	}
	const PxReal va = d3 * d6 - d5 * d4;
	if(va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f)
	{
		closest = b + (c - b) * ((d4 - d3) /
			((d4 - d3) + (d5 - d6))); return;
	}
	const PxReal inverse = 1.0f / (va + vb + vc);
	const PxReal v = vb * inverse, w = vc * inverse;
	closest = a * (1.0f - v - w) + b * v + c * w;
}

bool pointInsideCollisionMesh(const PxVec3& point,
	const PxTetrahedronMesh& mesh, const PxVec4* positions)
{
	for(PxU32 tet = 0; tet < mesh.getNbTetrahedrons(); ++tet)
	{
		PxU32 indices[4];
		if(!readTet(mesh, tet, indices)) return false;
		const PxVec3 a = positions[indices[0]].getXYZ();
		const PxMat33 basis(positions[indices[1]].getXYZ() - a,
			positions[indices[2]].getXYZ() - a,
			positions[indices[3]].getXYZ() - a);
		if(PxAbs(basis.getDeterminant()) <= 1.0e-12f) continue;
		const PxVec3 w = basis.getInverse() * (point - a);
		if(1.0f - w.x - w.y - w.z >= 0.0f &&
			w.x >= 0.0f && w.y >= 0.0f && w.z >= 0.0f) return true;
	}
	return false;
}

bool directedInitialSoftContactPotential(PxDeformableVolume& query,
	PxDeformableVolume& target, PxReal margin, PxF64& energy,
	PxU32& contactCount)
{
	const PxTetrahedronMesh* queryMesh = query.getCollisionMesh();
	const PxTetrahedronMesh* targetMesh = target.getCollisionMesh();
	const PxVec4* queryPositions = query.getPositionInvMassBufferD();
	const PxVec4* targetPositions = target.getPositionInvMassBufferD();
	PxArray<PxU32> queryFaces, targetFaces;
	if(!queryMesh || !targetMesh || !queryPositions || !targetPositions ||
		!collectBoundaryFaces(*queryMesh, queryFaces) ||
		!collectBoundaryFaces(*targetMesh, targetFaces)) return false;
	PxArray<PxU8> boundary(queryMesh->getNbVertices(), 0);
	for(PxU32 i = 0; i < queryFaces.size(); ++i)
	{
		if(queryFaces[i] >= boundary.size()) return false;
		boundary[queryFaces[i]] = 1;
	}
	for(PxU32 vertex = 0; vertex < queryMesh->getNbVertices(); ++vertex)
	{
		if(!queryPositions[vertex].isFinite() || !boundary[vertex]) continue;
		const PxVec3 point = queryPositions[vertex].getXYZ();
		if(!pointInsideCollisionMesh(point, *targetMesh, targetPositions)) continue;
		PxReal bestDistanceSquared = PX_MAX_F32;
		for(PxU32 face = 0; face < targetFaces.size() / 3; ++face)
		{
			const PxU32 i0 = targetFaces[3 * face], i1 = targetFaces[3 * face + 1],
				i2 = targetFaces[3 * face + 2];
			if(i0 >= targetMesh->getNbVertices() || i1 >= targetMesh->getNbVertices() ||
				i2 >= targetMesh->getNbVertices() || !targetPositions[i0].isFinite() ||
				!targetPositions[i1].isFinite() || !targetPositions[i2].isFinite()) return false;
			PxVec3 closest;
			closestPointTriangle(point, targetPositions[i0].getXYZ(),
				targetPositions[i1].getXYZ(), targetPositions[i2].getXYZ(), closest);
			bestDistanceSquared = PxMin(bestDistanceSquared,
				(point - closest).magnitudeSquared());
		}
		if(bestDistanceSquared == PX_MAX_F32) return false;
		const PxF64 penetration = PxF64(PxSqrt(PxMax(0.0f,
			bestDistanceSquared)) + margin);
		energy += 0.5 * 1.0e5 * penetration * penetration;
		++contactCount;
	}
	return true;
}

bool initialSoftContactPotential(PxDeformableVolume& a,
	PxDeformableVolume& b, PxReal fallbackMargin, PxReal& energy,
	PxU32& contactCount)
{
	PxShape* shapeA = a.getShape();
	PxShape* shapeB = b.getShape();
	const PxReal margin = PxMax(fallbackMargin,
		(shapeA ? shapeA->getContactOffset() : 0.0f) +
		(shapeB ? shapeB->getContactOffset() : 0.0f));
	PxF64 sum = 0.0;
	contactCount = 0;
	const bool valid = directedInitialSoftContactPotential(a, b, margin,
		sum, contactCount) && directedInitialSoftContactPotential(b, a,
			margin, sum, contactCount);
	energy = PxReal(sum);
	return valid && PxIsFinite(energy);
}

PxU32 countPointsInsideCollisionVolume(const PxVec3* points, PxU32 pointCount,
	PxDeformableVolume& target, bool& finite)
{
	finite = points != NULL;
	const PxTetrahedronMesh* targetMesh = target.getCollisionMesh();
	const PxVec4* targetPositions = target.getPositionInvMassBufferD();
	if(!finite || !targetMesh || !targetPositions)
	{
		finite = false;
		return 0;
	}
	const PxU32 targetCount = targetMesh->getNbVertices();
	for(PxU32 vertex = 0; vertex < targetCount; ++vertex)
		if(!targetPositions[vertex].isFinite())
		{
			finite = false;
			return 0;
		}
	PxU32 inside = 0;
	for(PxU32 point = 0; point < pointCount; ++point)
	{
		if(!points[point].isFinite())
		{
			finite = false;
			return inside;
		}
		for(PxU32 tet = 0; tet < targetMesh->getNbTetrahedrons(); ++tet)
		{
			PxU32 indices[4];
			if(!readTet(*targetMesh, tet, indices) ||
				indices[0] >= targetCount || indices[1] >= targetCount ||
				indices[2] >= targetCount || indices[3] >= targetCount)
			{
				finite = false;
				return inside;
			}
			if(pointStrictlyInsideTet(points[point],
				targetPositions[indices[0]].getXYZ(),
				targetPositions[indices[1]].getXYZ(),
				targetPositions[indices[2]].getXYZ(),
				targetPositions[indices[3]].getXYZ()))
			{
				++inside;
				break;
			}
		}
	}
	return inside;
}

PxReal tetVolumeRatio(const PxTetrahedronMesh& mesh, const PxVec4* positions,
	const PxArray<PxVec3>& rest, bool& finite)
{
	finite = positions && rest.size() == mesh.getNbVertices();
	PxReal currentVolume = 0.0f;
	PxReal restVolume = 0.0f;
	for(PxU32 tet = 0; finite && tet < mesh.getNbTetrahedrons(); ++tet)
	{
		PxU32 ids[4];
		finite = readTet(mesh, tet, ids);
		for(PxU32 corner = 0; finite && corner < 4; ++corner)
			finite = ids[corner] < rest.size() && positions[ids[corner]].isFinite();
		if(!finite) break;
		const PxVec3 current0 = positions[ids[0]].getXYZ();
		const PxVec3 rest0 = rest[ids[0]];
		currentVolume += PxAbs((positions[ids[1]].getXYZ() - current0).dot(
			(positions[ids[2]].getXYZ() - current0).cross(
				positions[ids[3]].getXYZ() - current0)));
		restVolume += PxAbs((rest[ids[1]] - rest0).dot(
			(rest[ids[2]] - rest0).cross(rest[ids[3]] - rest0)));
	}
	finite = finite && restVolume > 1.0e-12f && PxIsFinite(currentVolume);
	return finite ? currentVolume / restVolume : 0.0f;
}

#include "SnippetDeformableVolumeAVBDShowcaseGeometry.h"

// First compression and recovery along the fixed impact normal, from published mass-weighted motion.
// A later roll off the sphere must not be counted as impact restitution.


// Release the authored volume at the outside of the plane's contact band,
// stress-free and at rest. With no moving obstacle the initial mechanical
// energy bounds the subsequent kinetic + elastic + gravitational energy.
int runRestingShowcaseBody(const Snippets::HeadlessOptions& options,bool cone)
{
    if(!initPhysics(false, options.dispatcherThreads,
        VolumeAvbdCookedMesh::enabled() ? VolumeAvbdCookedMesh::create : NULL))
    {cleanupPhysics(false);return Snippets::eHEADLESS_GATE_FAILED;}
    const PxU32 body=cone?2:1;
    auto& scene=*gVolumeAvbdScene;
    auto& volume=*getVolumeAvbdRenderVolume(body);
    for(PxU32 b=0;b<5;++b)if(b!=body)scene.removeActor(*getVolumeAvbdRenderVolume(b));
    PxActor* actors[3]={};
    const PxU32 count=scene.getActors(PxActorTypeFlag::eRIGID_DYNAMIC|PxActorTypeFlag::eRIGID_STATIC,actors,3);
    PxReal band=volume.getShape()->getContactOffset();
    for(PxU32 i=0;i<count;++i)
    {
        PxShape* shape=NULL; static_cast<PxRigidActor*>(actors[i])->getShapes(&shape,1);
        if(shape->getGeometry().getType()==PxGeometryType::ePLANE)band+=shape->getContactOffset();
        else actors[i]->release();
    }
    const auto& mesh=*volume.getSimulationMesh();
    auto* x=volume.getSimPositionInvMassBufferD();auto* v=volume.getSimVelocityBufferD();
    auto* collision=volume.getPositionInvMassBufferD();
    // Place one side generator of the 0.8 m x 3 m cone on the horizontal plane.
    const PxQuat q=cone?PxQuat(PxHalfPi+PxAtan(0.8f/3.f),PxVec3(0,0,1)):PxQuat(PxIdentity);
    for(PxU32 i=0;i<mesh.getNbVertices();++i)
    {x[i]=PxVec4(q.rotate(mesh.getVertices()[i]),x[i].w);v[i]=PxVec4(0,0,0,v[i].w);}
    PxDeformableVolumeExt::updateEmbeddedCollisionMesh(volume,x,collision);
    PxReal low=PX_MAX_F32;
    for(PxU32 i=0;i<volume.getCollisionMesh()->getNbVertices();++i)low=PxMin(low,collision[i].y);
    for(PxU32 i=0;i<mesh.getNbVertices();++i)x[i].y+=band-low;
    PxDeformableVolumeExt::updateEmbeddedCollisionMesh(volume,x,collision);
    PxDeformableVolumeDataFlags dirty(PxDeformableVolumeDataFlag::eSIM_POSITION_INVMASS);
    dirty|=PxDeformableVolumeDataFlag::eSIM_VELOCITY;dirty|=PxDeformableVolumeDataFlag::ePOSITION_INVMASS;
    volume.markDirty(dirty);
    PxDeformableVolumeMaterial* material=NULL;volume.getShape()->getDeformableVolumeMaterials(&material,1);
    double initial=0,excess=0,peakLateSpeed=0,drift=0;
    PxVec3d lateCenter(0);const PxU32 start=PxU32(std::ceil(5./options.dt));
    PxU32 frame=0;
    bool valid=true;
    for(;;)
    {
        double mass=0,kinetic=0;PxVec3d center(0);
        for(PxU32 i=0;i<mesh.getNbVertices();++i)if(x[i].w>0)
        {
            const double m=1./x[i].w;
            mass+=m;center+=m*PxVec3d(x[i].x,x[i].y,x[i].z);
            kinetic+=.5*m*v[i].getXYZ().magnitudeSquared();
        }
        center/=mass;
        double minJ=PX_MAX_F64;
        const double energy=(avbdProbeVolumeEnergy(mesh,x,material->getYoungsModulus(),
            material->getPoissons(),minJ)+kinetic)/mass-scene.getGravity().y*center.y;
        valid &= std::isfinite(energy) && minJ>0 && center.isFinite();
        if(!frame)initial=energy;
        excess=PxMax(excess,energy-initial);
        if(frame==start)lateCenter=center;
        if(frame>=start)
        {
            peakLateSpeed=PxMax(peakLateSpeed,std::sqrt(2*kinetic/mass));
            drift=PxMax(drift,std::hypot(center.x-lateCenter.x,center.z-lateCenter.z));
        }
        traceShowcaseMotion(frame,0.);
        if(!valid || frame==options.frames)break;
        if(!scene.simulate(options.dt) || !scene.fetchResults(true))break;
        ++frame;
    }
    // One contact skin of horizontal drift and the authored settling speed
    // bound residual motion after five seconds. Allow 1 mm of gravity work
    // per kg in the independent physical-energy measurement.
    const bool pass=valid && frame==options.frames && frame>start && excess<=.001*std::abs(scene.getGravity().y)
        && drift<=band && peakLateSpeed<=volume.getSettlingThreshold();
    std::printf("[GROUND_REST] case=%s frames=%u initialJperKg=%.9g excessJperKg=%.9g latePeakRms=%.9g horizontalExcursion=%.9g band=%.9g status=%s\n",
        options.caseName.c_str(),frame,initial,excess,peakLateSpeed,drift,band,pass?"PASS":"FAIL");
    cleanupPhysics(false);return pass?Snippets::eHEADLESS_PASS:Snippets::eHEADLESS_GATE_FAILED;
}

// Remove the impact and upper loading to measure gravity-driven tipping of
// the same unsupported overhang. Both the collision mesh and the mass axis
// are read after publication; the benchmark never applies an angular kick.
int runPedestalTip(const Snippets::HeadlessOptions& options,bool loaded)
{
    if(!initPhysics(false,options.dispatcherThreads,
        VolumeAvbdCookedMesh::enabled()?VolumeAvbdCookedMesh::create:NULL))
    {cleanupPhysics(false);return Snippets::eHEADLESS_GATE_FAILED;}
    auto& scene=*gVolumeAvbdScene;
    auto& jaw=*getVolumeAvbdRenderVolume(4);
    for(PxU32 i=0;i<(loaded?3u:4u);++i)scene.removeActor(*getVolumeAvbdRenderVolume(i));
    PxActor* actors[3]={};
    const PxU32 count=scene.getActors(PxActorTypeFlag::eRIGID_DYNAMIC|PxActorTypeFlag::eRIGID_STATIC,actors,3);
    PxBounds3 base=PxBounds3::empty();
    const auto jawBand=jaw.getShape()->getContactOffset();
    PxReal band=jawBand;
    PxRigidDynamic* box=NULL;
    for(PxU32 i=0;i<count;++i)
    {
        if(auto* dynamic=actors[i]->is<PxRigidDynamic>())
        {if(loaded)box=dynamic;else dynamic->release();continue;}
        PxShape* shape=NULL;auto* rigid=static_cast<PxRigidActor*>(actors[i]);rigid->getShapes(&shape,1);
        if(shape->getGeometry().getType()==PxGeometryType::eBOX)
        {base=rigid->getWorldBounds(1);band+=shape->getContactOffset();}
    }
    if(base.isEmpty() || (loaded && !box)){cleanupPhysics(false);return Snippets::eHEADLESS_GATE_FAILED;}
    auto* p=jaw.getSimPositionInvMassBufferD();auto* v=jaw.getSimVelocityBufferD();
    auto* collision=jaw.getPositionInvMassBufferD();const auto& mesh=*jaw.getSimulationMesh();
    const auto floorGap=[&](){
        PxReal y=PX_MAX_F32;
        for(PxU32 i=0;i<jaw.getCollisionMesh()->getNbVertices();++i)y=PxMin(y,collision[i].y);
        return y;
    };
    const PxReal drop=base.maximum.y+band-floorGap();
    Snippets::SoftTipObservation axis;
    const auto mass=[&](PxU32 i){return p[i].w>0?1./p[i].w:0.;};
    axis.bake(mesh.getNbVertices(),[&](PxU32 i){return mesh.getVertices()[i].x;},mass);
    double total=0,lever=0,inertia=0;
    const auto addMass=[&](const PxVec3& pos,double m)
    {
        const double x=pos.x-base.minimum.x,y=pos.y-base.maximum.y;
        total+=m;lever-=m*x;inertia+=m*(x*x+y*y);
    };
    PxDeformableVolumeDataFlags dirty(PxDeformableVolumeDataFlag::eSIM_POSITION_INVMASS);
    dirty|=PxDeformableVolumeDataFlag::eSIM_VELOCITY;dirty|=PxDeformableVolumeDataFlag::ePOSITION_INVMASS;
    for(PxU32 b=loaded?3u:4u;b<=4;++b)
    {
        auto& vol=*getVolumeAvbdRenderVolume(b);
        auto* pos=vol.getSimPositionInvMassBufferD();auto* vel=vol.getSimVelocityBufferD();
        for(PxU32 i=0;i<vol.getSimulationMesh()->getNbVertices();++i)
        {
            pos[i].y+=drop;vel[i]=PxVec4(0,0,0,vel[i].w);
            if(pos[i].w>0)addMass(pos[i].getXYZ(),1./pos[i].w);
        }
        PxDeformableVolumeExt::updateEmbeddedCollisionMesh(vol,pos,vol.getPositionInvMassBufferD());
        vol.markDirty(dirty);
    }
    if(box)
    {
        auto pose=box->getGlobalPose();pose.p.y+=drop;box->setGlobalPose(pose);
        const auto com=pose*box->getCMassLocalPose();
        const auto z=com.q.rotateInv(PxVec3(0,0,1));
        addMass(com.p,box->getMass());inertia+=z.dot(box->getMassSpaceInertiaTensor().multiply(z));
    }
    const double target=PxPi/12, g=-scene.getGravity().y;
    const double ideal=std::sqrt(2*target*inertia/(g*lever));
    double angle=0,rate=0,maxRate=0,tipTime=0,floorTime=0;
    PxU32 frame=0;
    bool ok=lever>0 && total>0;
    while(ok)
    {
        ok=axis.sample([&](PxU32 i){return p[i].getXYZ();},[&](PxU32 i){return v[i].getXYZ();},angle,rate);
        if(!frame)axis.initialAngle=angle;
        if(!tipTime && angle>=target)tipTime=frame*options.dt;
        if(!floorTime && floorGap()<=jawBand)floorTime=frame*options.dt;
        maxRate=PxMax(maxRate,rate);
        if(!ok || frame==options.frames)break;
        ok=scene.simulate(options.dt) && scene.fetchResults(true);
        if(ok)++frame;
    }
    // A factor of two accommodates the initial contact band, elasticity,
    // rocking and the finite pivot patch. The scale is set by gravity and
    // the actual mass distribution of the released actor.
    const bool pass=ok && tipTime>0 && tipTime<=2*ideal && floorTime>=tipTime;
    std::printf("[PEDESTAL_TIP] loaded=%u frames=%u overhangM=%.7g rigidTimeScaleS=%.7g tip15S=%.7g floorBandS=%.7g peakRateRadS=%.7g status=%s\n",
        unsigned(loaded),frame,lever/total,ideal,tipTime,floorTime,maxRate,pass?"PASS":"FAIL");
    cleanupPhysics(false);
    return pass?Snippets::eHEADLESS_PASS:Snippets::eHEADLESS_GATE_FAILED;
}

// Measure the actual boundary edges drawn by renderDeformableVolume. An
// interior simulation edge and an embedded surface edge need not coincide.
struct ShowcaseSkinStretch
{
    struct Edge { PxU32 a,b; PxReal restSquared; };
    std::vector<Edge> edges;
    PxReal peakRatio=1,ratio=1;
    PxU32 first=0,second=0,peakFrame=0,currentRun=0,longestRun=0;
    bool bake(PxDeformableVolume& volume)
    {
        PxArray<PxU32> faces;
        if(!collectBoundaryFaces(*volume.getCollisionMesh(),faces))return false;
        const auto* x=volume.getPositionInvMassBufferD();
        std::vector<std::pair<PxU32,PxU32>> pairs;
        for(PxU32 t=0;t<faces.size();t+=3)for(PxU32 i=0;i<3;++i)
        {
            const auto a=faces[t+i],b=faces[t+(i+1)%3];
            pairs.emplace_back(PxMin(a,b),PxMax(a,b));
        }
        std::sort(pairs.begin(),pairs.end());pairs.erase(std::unique(pairs.begin(),pairs.end()),pairs.end());
        for(const auto& p:pairs)
        {
            const PxReal rr=(x[p.first].getXYZ()-x[p.second].getXYZ()).magnitudeSquared();
            if(!(rr>0) || !PxIsFinite(rr))return false;
            edges.push_back({p.first,p.second,rr});
        }
        return !edges.empty();
    }
    bool sample(const PxVec4* x,PxU32 frame)
    {
        PxReal maximum=0;
        for(const auto& e:edges)
        {
            const PxReal rr=(x[e.a].getXYZ()-x[e.b].getXYZ()).magnitudeSquared()/e.restSquared;
            if(!PxIsFinite(rr))return false;
            if(rr>maximum){maximum=rr;first=e.a;second=e.b;}
        }
        ratio=PxSqrt(maximum);
        if(ratio>peakRatio){peakRatio=ratio;peakFrame=frame;}
        currentRun=ratio>=2.f?currentRun+1:0;
        longestRun=PxMax(longestRun,currentRun);
        return true;
    }
};

int runVisualShowcaseHeadless(const Snippets::HeadlessOptions& options)
{
	bool ok = initPhysics(false, options.dispatcherThreads,
		VolumeAvbdCookedMesh::enabled() ? VolumeAvbdCookedMesh::create : NULL);
	Snippets::TrackingErrorCallback errors;
	const bool trackingErrors=ok;
	if(trackingErrors) PxGetFoundation().registerErrorCallback(errors);
	const bool reportProgress = Snippets::isEnabledEnvironmentValue(
		std::getenv("PHYSX_AVBD_HEADLESS_PROGRESS"));
	int result = Snippets::eHEADLESS_GATE_FAILED;
	{
	VolumeEmbeddingCheck embedding, restEmbedding, partitionEmbedding;
	const bool embeddingOracleValidated = ok && VolumeEmbeddingCheck::detectsCorruption();
	PxArray<PxVec3> rest[5];
    ShowcaseSkinStretch skin[5];
	PxU32 simulationVertices = 0, simulationTets = 0;
	for(PxU32 volumeIndex = 0; ok && volumeIndex < 5; ++volumeIndex)
	{
		PxDeformableVolume* volume = getVolumeAvbdRenderVolume(volumeIndex);
		const PxTetrahedronMesh* mesh = volume ? volume->getSimulationMesh() : NULL;
		const PxVec4* positions = volume ? volume->getSimPositionInvMassBufferD() : NULL;
		ok = mesh && positions;
		if(!ok) break;
        ok=skin[volumeIndex].bake(*volume);
        if(!ok)break;
		embedding.sample(*volume);
		restEmbedding.sampleRest(*volume, false);
		partitionEmbedding.sampleRest(*volume, true);
		simulationVertices += mesh->getNbVertices();
		simulationTets += mesh->getNbTetrahedrons();
		if(std::getenv("PHYSX_SNIPPET_ALLOCATION_STATS"))
		{
			PxReal kinetic=0,mass=0;PxVec3 momentum(0);deformableKineticState(*volume,kinetic,mass,momentum);
			std::printf("[PARTICLE3_SHOWCASE_BODY] body=%u mass=%.9g simulationVertices=%u simulationTets=%u\n",volumeIndex,mass,mesh->getNbVertices(),mesh->getNbTetrahedrons());
		}
		rest[volumeIndex].resize(mesh->getNbVertices());
		for(PxU32 vertex = 0; ok && vertex < rest[volumeIndex].size(); ++vertex)
		{
			ok = positions[vertex].isFinite();
			rest[volumeIndex][vertex] = positions[vertex].getXYZ();
		}
	}
	if(ok && Snippets::isEnabledEnvironmentValue(
		std::getenv("PHYSX_AVBD_MESH_DIAGNOSTICS")))
		for(PxU32 volumeIndex=0;volumeIndex<5;++volumeIndex)
			reportVisualMeshDiagnostics(*getVolumeAvbdRenderVolume(volumeIndex),
				volumeIndex);

	traceShowcaseMotion(0, 0.0);
	reportVolumeAvbdAllocations("particle3-initialized");
	PxU32 completed = 0, nonFinite = 0;
	// Hash all published soft positions/velocities and rigid poses per frame.
	// Timing excludes this diagnostic, just like the geometric gate checks.
	PxU64 trajectoryHash = 14695981039346656037ull;
	auto hashReal = [&](PxReal value)
	{
		PxU32 bits;
		std::memcpy(&bits, &value, sizeof(bits));
		for(PxU32 byte = 0; byte < sizeof(bits); ++byte)
		{
			trajectoryHash ^= (bits >> (8 * byte)) & 0xff;
			trajectoryHash *= 1099511628211ull;
		}
	};
	PxArray<PxActor*> rigidActors;
	const PxActorTypeFlags rigidTypes = PxActorTypeFlag::eRIGID_DYNAMIC;
	const PxU32 rigidCount = gVolumeAvbdScene ?
		gVolumeAvbdScene->getNbActors(rigidTypes) : 0;
	rigidActors.resize(rigidCount);
	if(rigidCount)
		gVolumeAvbdScene->getActors(rigidTypes, rigidActors.begin(), rigidCount);
	PxReal minimumDetF = PX_MAX_F32;
	PxReal minimumVolumeRatio = PX_MAX_F32, maximumVolumeRatio = -PX_MAX_F32;
	PxReal peakEdgeStrain = 0.0f, finalEdgeStrain = 0.0f;
	PxReal bodyMinimumDetF[5] = {PX_MAX_F32, PX_MAX_F32, PX_MAX_F32,
		PX_MAX_F32, PX_MAX_F32};
	PxReal bodyMinimumVolumeRatio[5] = {PX_MAX_F32, PX_MAX_F32, PX_MAX_F32,
		PX_MAX_F32, PX_MAX_F32};
	PxReal bodyPeakEdgeStrain[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	PxReal bodyFinalEdgeStrain[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	PxReal bodyCurrentDetF[5] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	PxReal bodyCurrentVolumeRatio[5] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	PxU32 firstVolumeRatioFailFrame = PX_MAX_U32;
	PxU32 firstBody4VolumeRatioFailFrame = PX_MAX_U32;
	PxF64 totalStepMs = 0.0, maximumStepMs = 0.0;
    std::vector<PxF64> stepTimes;
    ShowcaseGeometry geometry;
    Snippets::ReboundObservation coneImpact,prismImpact;
    PxU32 pedestalFrame=0;
    Snippets::SoftTipObservation lowerAxis;
    PxU32 floorFrame=0, tipFrame=0;
    double impactAngle=0,tipAtHalfSecond=0,tipAtOneSecond=0,peakTipRate=0;
    if(ok)
    {
        auto* jaw=getVolumeAvbdRenderVolume(4);
        const auto* mesh=jaw->getSimulationMesh();
        const auto* p=jaw->getSimPositionInvMassBufferD();
        lowerAxis.bake(mesh->getNbVertices(),[&](PxU32 i){return mesh->getVertices()[i].x;},
            [&](PxU32 i){return p[i].w>0 ? 1./p[i].w : 0.;});
        double restAngle=0, restRate=0;
        ok=lowerAxis.sample([&](PxU32 i){return p[i].getXYZ();},[](PxU32){return PxVec3(0);},restAngle,restRate);
        lowerAxis.initialAngle=restAngle;
    }
    PxReal beforeImpact[3]={},incoming[3]={},upward[3]={},minHeight[3]={},recovery[3]={},minLowerVolume=1,unloadedLowerVolume=0;
    PxReal previousFlightVelocity=0,maxFlightVelocityError=0,impactBoxHeight=0,boxDropOneSecond=0;
    PxReal unloadingLow[2]={PX_MAX_F32,PX_MAX_F32},unloadingHigh[2]={-PX_MAX_F32,-PX_MAX_F32};
    PxU32 worstFlightFrame=0,flightSamples=0;
    const PxU32 pedestalWindow=PxMax(1u,PxU32(PxCeil(1.f/options.dt)));
    const bool traceSandwich=std::getenv("PHYSX_AVBD_SHOWCASE_MOTION")!=NULL;
    PxVec3 prismNormal(0),previousRelativeVelocity(0);
    bool encounteredSphere=false;
    PxU32 releaseFrame=0;
    PxU32 consecutiveContact=0,longestContact=0,firstDeparture=0,proximityFrames=0;
    PxU32 airborneContact=0,longestAirborneContact=0;
    PxU32 clearFrames=0,sustainedSeparation=0;
    bool sustainedEncounter=false;
    bool landed=false;
    PxReal maxGroundedSeparation=0;
	for(PxU32 frame = 0; ok && frame < options.frames; ++frame)
	{
		sendVolumeAvbdPvdMesh();
		PxTime timer;
		ok = gVolumeAvbdScene->simulate(options.dt) && gVolumeAvbdScene->fetchResults(true) && errors.getFatalCount()==0;
		const PxF64 stepMs = timer.getElapsedSeconds() * 1000.0;
		if(!ok) break;
		totalStepMs += stepMs;
		maximumStepMs = PxMax(maximumStepMs, stepMs);
        stepTimes.push_back(stepMs);
		++completed;
        if(!geometry.observe(*gVolumeAvbdScene)){ok=false;break;}
        PxReal bodyMass[5]={};
        PxVec3 center[5]={};
        PxVec3 speed[5]={};
        const auto& prism=geometry.bodies[0].bounds;
        const auto& sphere=geometry.bodies[1].bounds;
        const auto gap=(prism.minimum-sphere.maximum).maximum(sphere.minimum-prism.maximum).maximum(PxVec3(0));
        const PxReal distance=gap.magnitude();
        const PxReal contactBand=getVolumeAvbdRenderVolume(0)->getShape()->getContactOffset()+
            getVolumeAvbdRenderVolume(1)->getShape()->getContactOffset();
        encounteredSphere |= distance<=contactBand;
        if(encounteredSphere && prism.minimum.y<=contactBand)
        {
            maxGroundedSeparation=PxMax(maxGroundedSeparation,distance);
            if(!releaseFrame && distance>contactBand)releaseFrame=completed;
        }
		finalEdgeStrain = 0.0f;
		for(PxU32 volumeIndex = 0; volumeIndex < 5; ++volumeIndex)
			bodyFinalEdgeStrain[volumeIndex] = 0.0f;
		for(PxU32 volumeIndex = 0; volumeIndex < 5; ++volumeIndex)
		{
			PxDeformableVolume* volume = getVolumeAvbdRenderVolume(volumeIndex);
			const PxTetrahedronMesh* mesh = volume ? volume->getSimulationMesh() : NULL;
			const PxVec4* positions = volume ? volume->getSimPositionInvMassBufferD() : NULL;
			if(!mesh || !positions) { ++nonFinite; continue; }
			embedding.sample(*volume);
            if(!skin[volumeIndex].sample(volume->getPositionInvMassBufferD(),completed))++nonFinite;
            if(reportProgress && completed%30==0 && skin[volumeIndex].ratio>1.5f)
            {
                const auto& s=skin[volumeIndex];const auto* x=volume->getPositionInvMassBufferD();
                std::fprintf(stderr,"[STRETCHED_EDGE] frame=%u body=%u ratio=%.6g ids=%u,%u a=%.6g,%.6g,%.6g b=%.6g,%.6g,%.6g\n",
                    completed,volumeIndex,s.ratio,s.first,s.second,x[s.first].x,x[s.first].y,x[s.first].z,x[s.second].x,x[s.second].y,x[s.second].z);
            }
			const PxVec4* velocities = volume->getSimVelocityBufferD();
            if(!velocities) { ++nonFinite; continue; }
            PxF64 mass=0;PxVec3d massPosition(0),momentum(0);
            for(PxU32 vertex = 0; vertex < mesh->getNbVertices(); ++vertex)
            {
                if(positions[vertex].w>0)
                {
                    const PxF64 m=1./positions[vertex].w;
                    mass+=m;
                    const auto& x=positions[vertex];const auto& v=velocities[vertex];
                    massPosition+=m*PxVec3d(x.x,x.y,x.z);momentum+=m*PxVec3d(v.x,v.y,v.z);
                }
                for(PxU32 axis = 0; axis < 4; ++axis)
					hashReal(positions[vertex][axis]);
				for(PxU32 axis = 0; axis < 3; ++axis)
					hashReal(velocities[vertex][axis]);
			}
            {
                if(!(mass>0)){++nonFinite;continue;}
                bodyMass[volumeIndex]=PxReal(mass);massPosition/=mass;momentum/=mass;
                center[volumeIndex]=PxVec3(PxReal(massPosition.x),PxReal(massPosition.y),PxReal(massPosition.z));
                speed[volumeIndex]=PxVec3(PxReal(momentum.x),PxReal(momentum.y),PxReal(momentum.z));
            }
            bool finiteDet = false, finiteStrain = false, finiteVolume = false;
            const PxReal detF = minimumTetDetF(*mesh, positions,
				rest[volumeIndex], finiteDet);
			const PxReal strain = maximumTetEdgeStrain(*mesh, positions,
				rest[volumeIndex], finiteStrain);
			const PxReal volumeRatio = tetVolumeRatio(*mesh, positions,
				rest[volumeIndex], finiteVolume);
			bodyCurrentDetF[volumeIndex] = detF;
			bodyCurrentVolumeRatio[volumeIndex] = volumeRatio;
			if(!finiteDet || !finiteStrain || !finiteVolume) ++nonFinite;
			minimumDetF = PxMin(minimumDetF, detF);
			minimumVolumeRatio = PxMin(minimumVolumeRatio, volumeRatio);
			maximumVolumeRatio = PxMax(maximumVolumeRatio, volumeRatio);
			peakEdgeStrain = PxMax(peakEdgeStrain, strain);
			finalEdgeStrain = PxMax(finalEdgeStrain, strain);
			bodyMinimumDetF[volumeIndex] = PxMin(bodyMinimumDetF[volumeIndex], detF);
			bodyMinimumVolumeRatio[volumeIndex] = PxMin(
				bodyMinimumVolumeRatio[volumeIndex], volumeRatio);
			bodyPeakEdgeStrain[volumeIndex] = PxMax(
				bodyPeakEdgeStrain[volumeIndex], strain);
			bodyFinalEdgeStrain[volumeIndex] = strain;
			if(volumeRatio <= 0.85f && firstVolumeRatioFailFrame == PX_MAX_U32)
				firstVolumeRatioFailFrame = frame;
			if(volumeIndex == 4 && volumeRatio <= 0.85f &&
				firstBody4VolumeRatioFailFrame == PX_MAX_U32)
				firstBody4VolumeRatioFailFrame = frame;
		}
		for(PxU32 actor = 0; actor < rigidCount; ++actor)
		{
			const PxTransform pose = static_cast<PxRigidActor*>(
				rigidActors[actor])->getGlobalPose();
			hashReal(pose.p.x); hashReal(pose.p.y); hashReal(pose.p.z);
			hashReal(pose.q.x); hashReal(pose.q.y); hashReal(pose.q.z); hashReal(pose.q.w);
		}
        if(nonFinite || rigidCount!=1){ok=false;break;}
        const auto* box=static_cast<PxRigidDynamic*>(rigidActors[0]);
        const PxReal heights[]={center[4].y,box->getGlobalPose().p.y,center[3].y};
        const PxReal vy[]={speed[4].y,box->getLinearVelocity().y,speed[3].y};
        const PxReal combinedVy=(vy[0]*bodyMass[4]+vy[1]*box->getMass()+vy[2]*bodyMass[3])/
            (bodyMass[4]+box->getMass()+bodyMass[3]);
        PxBounds3 sandwich=geometry.bodies[4].bounds;
        sandwich.include(geometry.bodies[3].bounds);sandwich.include(box->getWorldBounds(1.f));
        const PxReal reach=geometry.pedestalBand+PxAbs(previousFlightVelocity)*options.dt;
        sandwich.fattenFast(reach);
        bool freeFlight=!pedestalFrame && sandwich.minimum.y>geometry.pedestal.bounds.maximum.y;
        for(PxU32 k=0;k<3;++k)freeFlight &= !sandwich.intersects(geometry.bodies[k].bounds);
        if(freeFlight && completed>1)
        {
            ++flightSamples;
            const PxReal residual=PxAbs(combinedVy-previousFlightVelocity-gVolumeAvbdScene->getGravity().y*options.dt);
            if(residual>maxFlightVelocityError){maxFlightVelocityError=residual;worstFlightFrame=completed;}
        }
        previousFlightVelocity=combinedVy;
        PxVec3 axis;
        if(!pedestalFrame && ShowcaseGeometry::impactNormal(geometry.bodies[4],geometry.pedestal,geometry.pedestalBand,axis))
        {
            pedestalFrame=completed;impactBoxHeight=heights[1];
            for(PxU32 k=0;k<3;++k){incoming[k]=-beforeImpact[k];minHeight[k]=heights[k];}
        }
        if(pedestalFrame && completed-pedestalFrame<=2*pedestalWindow)
        {
            auto* jaw=getVolumeAvbdRenderVolume(4);
            const auto* p=jaw->getSimPositionInvMassBufferD();
            const auto* v=jaw->getSimVelocityBufferD();
            double angle=0,rate=0;
            if(!lowerAxis.sample([&](PxU32 i){return p[i].getXYZ();},[&](PxU32 i){return v[i].getXYZ();},angle,rate))
                ++nonFinite;
            if(completed==pedestalFrame)impactAngle=angle;
            if(!tipFrame && angle-impactAngle>=PxPi/12)tipFrame=completed;
            if(!floorFrame && geometry.bodies[4].bounds.minimum.y<=jaw->getShape()->getContactOffset())floorFrame=completed;
            if(completed-pedestalFrame==pedestalWindow/2)tipAtHalfSecond=angle;
            if(completed-pedestalFrame==pedestalWindow)tipAtOneSecond=angle;
            peakTipRate=PxMax(peakTipRate,rate);
            if(traceSandwich)std::fprintf(stderr,"[JAW_TIP] frame=%u angleRad=%.8g rateRadS=%.8g floorGap=%.8g\n",
                completed,angle,rate,geometry.bodies[4].bounds.minimum.y);
        }
        if(pedestalFrame && completed-pedestalFrame<=pedestalWindow)
        {
            if(bodyCurrentVolumeRatio[4]<minLowerVolume)
            {minLowerVolume=bodyCurrentVolumeRatio[4];unloadedLowerVolume=minLowerVolume;}
            if(vy[0]>0)unloadedLowerVolume=PxMax(unloadedLowerVolume,bodyCurrentVolumeRatio[4]);
            // Either direction is physical for an off-centre sandwich. Measure
            // whether the rigid and upper body remain mobile during unloading.
            if(bodyCurrentVolumeRatio[4]-minLowerVolume>.01f)
                for(PxU32 k=0;k<2;++k)
                {
                    unloadingLow[k]=PxMin(unloadingLow[k],heights[k+1]);
                    unloadingHigh[k]=PxMax(unloadingHigh[k],heights[k+1]);
                }
            for(PxU32 k=0;k<3;++k)
            {
                upward[k]=PxMax(upward[k],vy[k]);minHeight[k]=PxMin(minHeight[k],heights[k]);
                recovery[k]=PxMax(recovery[k],heights[k]-minHeight[k]);
            }
        }
        for(PxU32 k=0;k<3;++k)beforeImpact[k]=vy[k];
        if(pedestalFrame && completed-pedestalFrame==pedestalWindow)boxDropOneSecond=impactBoxHeight-heights[1];
        if(traceSandwich)std::fprintf(stderr,"[PEDESTAL_MOTION] frame=%u lowerY=%.8g lowerVy=%.8g boxY=%.8g boxVy=%.8g upperY=%.8g upperVy=%.8g lowerVolume=%.8g\n",
            completed,heights[0],vy[0],heights[1],vy[1],heights[2],vy[2],bodyCurrentVolumeRatio[4]);
        const PxReal coneGap=geometry.bodies[2].bounds.minimum.y;
        coneImpact.sample(completed,coneGap<=getVolumeAvbdRenderVolume(2)->getShape()->getContactOffset(),
            center[2].y,speed[2].y,bodyCurrentVolumeRatio[2],coneGap);
        const PxVec3 relativeVelocity=speed[0]-speed[1];
        PxVec3 contactNormal;
        const bool inContact=distance<=contactBand &&
            geometry.impactNormal(geometry.bodies[0],geometry.bodies[1],contactBand,contactNormal);
        landed |= prism.minimum.y<=getVolumeAvbdRenderVolume(0)->getShape()->getContactOffset();
        airborneContact=!landed && inContact ? airborneContact+1 : 0;
        longestAirborneContact=PxMax(longestAirborneContact,airborneContact);
        if(inContact)
        {
            ++proximityFrames;
            longestContact=PxMax(longestContact,++consecutiveContact);
            sustainedEncounter |= consecutiveContact>=3;
            clearFrames=0;
        }
        else
        {
            if(consecutiveContact && !firstDeparture)firstDeparture=completed;
            consecutiveContact=0;
            if(sustainedEncounter && ++clearFrames==3 && !sustainedSeparation)sustainedSeparation=completed;
        }
        if(!prismImpact.contactFrame && inContact)
        {
            prismNormal=contactNormal;
            prismImpact.incoming=PxMax(0.f,-prismNormal.dot(previousRelativeVelocity));
        }
        if(!prismNormal.isZero())
            prismImpact.sample(completed,inContact,prismNormal.dot(center[0]-center[1]),prismNormal.dot(relativeVelocity),
                bodyCurrentVolumeRatio[0],distance);
        previousRelativeVelocity=relativeVelocity;
        traceShowcaseMotion(frame + 1, stepMs);
		if(reportProgress && ((frame + 1) % 30 == 0 || frame + 1 == options.frames))
		{
			std::fprintf(stderr,
				"[AVBD_PROGRESS] case=visual-showcase frame=%u stepMs=%.9g bodyFinalEdgeStrain=%.6g,%.6g,%.6g,%.6g,%.6g body4MinDetF=%.6g body4VolumeRatio=%.6g\n",
				frame + 1, stepMs, double(bodyFinalEdgeStrain[0]),
				double(bodyFinalEdgeStrain[1]), double(bodyFinalEdgeStrain[2]),
				double(bodyFinalEdgeStrain[3]), double(bodyFinalEdgeStrain[4]),
				double(bodyCurrentDetF[4]), double(bodyCurrentVolumeRatio[4]));
			std::fflush(stderr);
		}
	}
	const PxF64 averageStepMs = completed ? totalStepMs / completed : 0.0;
    if(!Snippets::writeHeadlessStepTimes(stepTimes)) ok = false;
    std::sort(stepTimes.begin(),stepTimes.end());
    std::printf("[VOLUME_STEP_TIMING] completed=%u meanMs=%.6f p95Ms=%.6f\n",completed,averageStepMs,
        stepTimes.empty()?0:stepTimes[(stepTimes.size()-1)*95/100]);
	std::printf("[AVBD_BENCH] case=visual-showcase simulationVertices=%u simulationTets=%u trajectoryHash=%016llx sceneDispatcherThreads=%u\n",
		simulationVertices, simulationTets,
		static_cast<unsigned long long>(trajectoryHash), options.dispatcherThreads);
    // The matched Rigid6 showcase uses one percent of its 0.2 m minimum cell.
    const double depthLimit=.002;
    std::printf("[FIVE_GEOMETRY] ground=%.9g rigid=%.9g between=%.9g self=%.9g limit=%.9g\n",
        geometry.ground,geometry.rigid,geometry.between,geometry.self,depthLimit);
    for(PxU32 i=0;i<2;++i)
    {
        const auto& r=i?prismImpact:coneImpact;
        std::printf("[FIVE_REBOUND] pair=%s axis=%s windowFrames=60 contactFrame=%u apexFrame=%u departureFrame=%u incoming=%.8g up=%.8g rise=%.8g "
            "minVolume=%.8g recoveredVolume=%.8g clearance=%.8g\n",i?"prism-sphere":"cone-ground",i?"incidentNormal":"worldY",
            r.contactFrame,r.apexFrame,r.departureFrame,double(r.incoming),double(r.up),double(r.rise),double(r.minVolume),
            double(r.apexVolume),double(r.clearance));
    }
    std::printf("[FIVE_RELEASE] encountered=%u firstGroundedSeparation=%u maxGroundedGap=%.8g\n",
        unsigned(encounteredSphere),releaseFrame,double(maxGroundedSeparation));
    std::printf("[FIVE_CONTACT_RESIDENCE] pair=prism-sphere firstEntry=%u firstExit=%u longestFrames=%u "
        "longestAirborneFrames=%u nearFrames=%u sustainedExit=%u\n",
        prismImpact.contactFrame,firstDeparture,longestContact,longestAirborneContact,proximityFrames,sustainedSeparation);
    const bool pedestalWindowComplete=pedestalFrame && completed-pedestalFrame>=pedestalWindow;
    std::printf("[FIVE_TIP] impactFrame=%u impactAngleRad=%.8g postImpact15Frame=%u floorBandFrame=%u halfSecondRad=%.8g oneSecondRad=%.8g peakRateRadS=%.8g\n",
        pedestalFrame,impactAngle,tipFrame,floorFrame,tipAtHalfSecond,tipAtOneSecond,peakTipRate);
    const PxReal boxTravel=unloadingHigh[0]>=unloadingLow[0]?unloadingHigh[0]-unloadingLow[0]:0;
    const PxReal upperTravel=unloadingHigh[1]>=unloadingLow[1]?unloadingHigh[1]-unloadingLow[1]:0;
    const bool pedestalPass=pedestalWindowComplete && flightSamples && maxFlightVelocityError<PxAbs(gVolumeAvbdScene->getGravity().y)*options.dt &&
        unloadedLowerVolume-minLowerVolume>.01f && recovery[0]>.02f && boxTravel>.02f && upperTravel>.02f;
    std::printf("[FIVE_PEDESTAL] status=%s contactFrame=%u lowerIn=%.8g boxIn=%.8g upperIn=%.8g lowerUp=%.8g boxUp=%.8g upperUp=%.8g lowerRise=%.8g boxRise=%.8g upperRise=%.8g minLowerVolume=%.8g unloadedLowerVolume=%.8g flightDvError=%.8g worstFlightFrame=%u boxDropOneSecond=%.8g boxUnloadingTravel=%.8g upperUnloadingTravel=%.8g\n",
        pedestalPass?"PASS":(pedestalWindowComplete?"FAIL":"INCOMPLETE"),pedestalFrame,incoming[0],incoming[1],incoming[2],upward[0],upward[1],upward[2],recovery[0],recovery[1],recovery[2],minLowerVolume,unloadedLowerVolume,maxFlightVelocityError,worstFlightFrame,boxDropOneSecond,boxTravel,upperTravel);
    bool skinPass=true;
    for(PxU32 b=0;b<5;++b)
    {
        std::printf("[FIVE_SKIN_STRETCH] body=%u peakLengthRatio=%.8g peakFrame=%u longestDoubledS=%.8g finalLengthRatio=%.8g\n",
            b,skin[b].peakRatio,skin[b].peakFrame,skin[b].longestRun*options.dt,skin[b].ratio);
        skinPass &= skin[b].longestRun*options.dt<.5f;
    }
    const bool physical = skinPass && (options.frames*options.dt<5.f || pedestalPass) && geometry.external<=depthLimit && geometry.self<=depthLimit && minimumDetF >= 0.049f && minimumVolumeRatio > 0.85f &&
		maximumVolumeRatio < 1.15f && finalEdgeStrain < 0.20f &&
		embeddingOracleValidated && embedding.passed() && restEmbedding.passed() && partitionEmbedding.passed();
	const PxU32 fatalErrors=errors.getFatalCount();
	const bool pass = ok && completed == options.frames && nonFinite == 0 && fatalErrors == 0 && physical;
	const char* reason = !ok ? "initialization_or_fetch" :
		(nonFinite || fatalErrors) ? "runtime_error" : !physical ? "physical_contract" : "none";
	std::printf("[AVBD_GATE] schema=1 snippet=SnippetDeformableVolumeAVBD solver=avbd case=visual-showcase execution=%s frames=%u completedFrames=%u status=%s reason=%s capability=SUPPORTED validation=GATED simVertices=%u simulationTets=%u trajectoryHash=%016llx firstVolumeRatioFailFrame=%u firstBody4VolumeRatioFailFrame=%u minDetF=%.9g minVolumeRatio=%.9g maxVolumeRatio=%.9g peakEdgeStrain=%.9g finalEdgeStrain=%.9g body0MinDetF=%.9g body1MinDetF=%.9g body2MinDetF=%.9g body3MinDetF=%.9g body4MinDetF=%.9g body0MinVolumeRatio=%.9g body1MinVolumeRatio=%.9g body2MinVolumeRatio=%.9g body3MinVolumeRatio=%.9g body4MinVolumeRatio=%.9g body0PeakEdgeStrain=%.9g body1PeakEdgeStrain=%.9g body2PeakEdgeStrain=%.9g body3PeakEdgeStrain=%.9g body4PeakEdgeStrain=%.9g body0FinalEdgeStrain=%.9g body1FinalEdgeStrain=%.9g body2FinalEdgeStrain=%.9g body3FinalEdgeStrain=%.9g body4FinalEdgeStrain=%.9g averageStepMs=%.9g maxStepMs=%.9g nonFinite=%u fatalErrors=%u embeddingMaxVertexError=%.9g embeddingMaxErrorRatio=%.9g embeddingSamples=%llu embeddingInvalid=%u embeddingOracleValidated=%u embeddingRestError=%.9g embeddingRestErrorRatio=%.9g embeddingPartitionError=%.9g embeddingPartitionErrorRatio=%.9g\n",
		Snippets::getExecutionName(options.execution), options.frames, completed,
		pass ? "PASS" : "FAIL", reason, simulationVertices, simulationTets,
		static_cast<unsigned long long>(trajectoryHash),
		firstVolumeRatioFailFrame, firstBody4VolumeRatioFailFrame, double(minimumDetF),
		double(minimumVolumeRatio), double(maximumVolumeRatio),
		double(peakEdgeStrain), double(finalEdgeStrain),
		double(bodyMinimumDetF[0]), double(bodyMinimumDetF[1]),
		double(bodyMinimumDetF[2]), double(bodyMinimumDetF[3]),
		double(bodyMinimumDetF[4]), double(bodyMinimumVolumeRatio[0]),
		double(bodyMinimumVolumeRatio[1]), double(bodyMinimumVolumeRatio[2]),
		double(bodyMinimumVolumeRatio[3]), double(bodyMinimumVolumeRatio[4]),
		double(bodyPeakEdgeStrain[0]),
		double(bodyPeakEdgeStrain[1]), double(bodyPeakEdgeStrain[2]),
		double(bodyPeakEdgeStrain[3]), double(bodyPeakEdgeStrain[4]),
		double(bodyFinalEdgeStrain[0]), double(bodyFinalEdgeStrain[1]),
		double(bodyFinalEdgeStrain[2]), double(bodyFinalEdgeStrain[3]),
			double(bodyFinalEdgeStrain[4]), averageStepMs, maximumStepMs, nonFinite, fatalErrors,
		embedding.maxVertexError, embedding.maxErrorRatio,
		static_cast<unsigned long long>(embedding.samples), embedding.invalid,
		embeddingOracleValidated ? 1u : 0u, restEmbedding.maxVertexError, restEmbedding.maxErrorRatio,
		partitionEmbedding.maxVertexError, partitionEmbedding.maxErrorRatio);
	std::fflush(stdout);
	reportVolumeAvbdAllocations("particle3-completed");
	// PxArray releases through the PhysX allocator.  Empty the local mesh
	// snapshots while the foundation is still alive; otherwise their scope
	// destructor runs after cleanupPhysics() has released that allocator.
	rigidActors.reset();
	for(PxU32 volumeIndex = 0; volumeIndex < 5; ++volumeIndex)
	{
		rest[volumeIndex].clear();
		rest[volumeIndex].shrink();
	}
	result = pass ? Snippets::eHEADLESS_PASS : Snippets::eHEADLESS_GATE_FAILED;
	}
	// All local PxArray instances must be destroyed while the foundation
	// allocator still exists.  A size/capacity operation is not a lifetime
	// boundary and cannot make post-foundation destructors safe.
	if(trackingErrors) PxGetFoundation().deregisterErrorCallback(errors);
	cleanupPhysics(false);
	reportVolumeAvbdAllocations("particle3-released");
	return result;
}
}

bool snippetDeformableVolumeAvbdHeadlessShouldRun(int argc, const char* const* argv)
{
	if(Snippets::isEnabledEnvironmentValue(std::getenv("PHYSX_SNIPPET_HEADLESS"))) return true;
	for(int i = 1; i < argc; ++i)
		if(argv[i] && !std::strcmp(argv[i], "--headless")) return true;
	return false;
}

int snippetDeformableVolumeAvbdHeadlessMain(int argc, const char* const* argv)
{
	Snippets::HeadlessOptions defaults;
	defaults.headless = true;
	defaults.solverType = PxSolverType::eAVBD;
	defaults.caseName = "visual-showcase";
	defaults.frames = 360;
	defaults.dispatcherThreads = 4;
	Snippets::HeadlessOptions options;
	std::string error;
	if(!Snippets::parseCommonHeadlessOptions(argc, argv, defaults, options, error) ||
		!known(options.caseName.c_str()))
	{
		std::fprintf(stderr, "[AVBD_GATE_CONFIG_ERROR] snippet=SnippetDeformableVolumeAVBD reason=%s\n",
			error.empty() ? "unsupported_case" : error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	for(int i = 1; i < argc; ++i)
		if(!Snippets::isCommonHeadlessOption(argv[i])) return Snippets::eHEADLESS_CONFIG_ERROR;
	if(options.solverType != PxSolverType::eAVBD)
	{
		std::printf("[AVBD_GATE] schema=1 snippet=SnippetDeformableVolumeAVBD solver=%s case=%s status=ERROR reason=cpu_volume_requires_avbd capability=UNSUPPORTED validation=GATED\n",
			Snippets::getSolverTypeName(options.solverType), options.caseName.c_str());
		return Snippets::eHEADLESS_UNSUPPORTED;
	}
	Snippets::applyExecutionEnvironment(options);
	Snippets::printHeadlessConfig("SnippetDeformableVolumeAVBD", options);
	const bool restingCone=Snippets::equalsIgnoreCase(options.caseName.c_str(),"resting-cone");
	if(restingCone || Snippets::equalsIgnoreCase(options.caseName.c_str(),"resting-sphere"))
		return runRestingShowcaseBody(options,restingCone);
	const bool loadedTip=Snippets::equalsIgnoreCase(options.caseName.c_str(),"pedestal-tip-loaded");
	if(loadedTip || Snippets::equalsIgnoreCase(options.caseName.c_str(),"pedestal-tip"))return runPedestalTip(options,loadedTip);
	if(Snippets::equalsIgnoreCase(options.caseName.c_str(), "visual-showcase"))
		return runVisualShowcaseHeadless(options);
	Fixture fixture;
	bool ok = fixture.init(options, options.caseName.c_str());
	VolumeAvbdBenchmark benchmark;
	if(ok) ok = benchmark.begin(*fixture.scene, *fixture.volume,
		options.caseName.c_str(), options.dt);
	const bool mutation = Snippets::equalsIgnoreCase(options.caseName.c_str(), "buffer-mutation");
	const bool embedded = Snippets::equalsIgnoreCase(options.caseName.c_str(), "embedded-sync");
	const bool embeddedSameCount = Snippets::equalsIgnoreCase(options.caseName.c_str(), "embedded-same-count");
	const bool rigidSqueeze = Snippets::equalsIgnoreCase(options.caseName.c_str(), "rigid-squeeze");
	const bool rigidOnSoft = Snippets::equalsIgnoreCase(options.caseName.c_str(), "rigid-on-soft") ||
		Snippets::equalsIgnoreCase(options.caseName.c_str(), "rigid-on-soft-separated");
	const bool freeFallTriplet = Snippets::equalsIgnoreCase(options.caseName.c_str(), "free-fall-triplet");
	const bool rollingSphere = Snippets::equalsIgnoreCase(options.caseName.c_str(), "rolling-sphere");
	const bool softSoftRecovery = Snippets::equalsIgnoreCase(options.caseName.c_str(),
		"soft-soft-recovery");
	const bool softSoftInitialContainment = Snippets::equalsIgnoreCase(
		options.caseName.c_str(), "soft-soft-initial-containment");
	const bool softSoft = Snippets::equalsIgnoreCase(options.caseName.c_str(), "soft-soft") ||
		softSoftRecovery || softSoftInitialContainment || rigidSqueeze || freeFallTriplet;
	const bool staticBox = Snippets::equalsIgnoreCase(options.caseName.c_str(), "static-box");
	const bool dynamic = Snippets::equalsIgnoreCase(options.caseName.c_str(), "dynamic-box") ||
		rigidSqueeze || freeFallTriplet;
	bool mutationReadback = !mutation;
	VolumeEmbeddingCheck embedding, restEmbedding, partitionEmbedding;
	const bool embeddingOracleValidated = ok && VolumeEmbeddingCheck::detectsCorruption();
	if(ok)
	{
		embedding.sample(*fixture.volume);
		restEmbedding.sampleRest(*fixture.volume, false);
		partitionEmbedding.sampleRest(*fixture.volume, true);
		if(fixture.volumeB)
		{
			embedding.sample(*fixture.volumeB);
			restEmbedding.sampleRest(*fixture.volumeB, false);
			partitionEmbedding.sampleRest(*fixture.volumeB, true);
		}
	}
	PxU32 simCount = 0, collisionCount = 0;
	PxReal initialEmbeddingIdentityError = 0.0f;
	PxArray<PxVec3> simulationRest, simulationRestB;
	PxVec3 simInitial(0.0f), collisionInitial(0.0f), simFinal(0.0f), collisionFinal(0.0f);
	PxVec3 simInitialB(0.0f);
	if(ok)
	{
		simCount = fixture.volume->getSimulationMesh()->getNbVertices();
		collisionCount = fixture.volume->getCollisionMesh()->getNbVertices();
		if(simCount == collisionCount)
			for(PxU32 vertex = 0; vertex < simCount; ++vertex)
				initialEmbeddingIdentityError = PxMax(initialEmbeddingIdentityError,
					(fixture.volume->getSimPositionInvMassBufferD()[vertex].getXYZ() -
					 fixture.volume->getPositionInvMassBufferD()[vertex].getXYZ()).magnitude());
		ok = centroid(fixture.volume->getSimPositionInvMassBufferD(), simCount, simInitial) &&
			centroid(fixture.volume->getPositionInvMassBufferD(), collisionCount, collisionInitial);
		const PxVec4* initial = fixture.volume->getSimPositionInvMassBufferD();
		simulationRest.resize(simCount);
		for(PxU32 i = 0; ok && i < simCount; ++i)
		{
			ok = initial[i].isFinite();
			simulationRest[i] = initial[i].getXYZ();
		}
		if(ok && softSoft)
		{
			const PxU32 countB = fixture.volumeB->getSimulationMesh()->getNbVertices();
			const PxVec4* initialB = fixture.volumeB->getSimPositionInvMassBufferD();
			simulationRestB.resize(countB);
			for(PxU32 i = 0; ok && i < countB; ++i)
			{
				ok = initialB[i].isFinite();
				simulationRestB[i] = initialB[i].getXYZ();
			}
			ok = ok && centroid(initialB, countB, simInitialB);
		}
	}
	PxArray<PxVec3> prevSimA = simulationRest;
	PxArray<PxVec3> prevSimB = simulationRestB;
	PxVec3 prevComA = simInitial;
	PxVec3 prevComB = simInitialB;
	if(ok && mutation)
	{
		PxVec4* positions = fixture.volume->getSimPositionInvMassBufferD();
		if(positions)
		{
			const PxReal x = positions[0].x;
			positions[0].x = x + 0.01f;
			mutationReadback = positions[0].x == x + 0.01f;
			positions[0].x = x;
		}
	}

	PxU32 initialInsideParticles = 0;
	PxU32 initialSoftContactCount = 0;
	PxReal initialSoftContactPotentialEnergy = 0.0f;
	PxReal initialSoftPairKineticEnergy = 0.0f;
	PxReal maximumSoftPairKineticEnergy = 0.0f;
	PxReal finalSoftPairKineticEnergy = 0.0f;
	PxReal maximumSoftPairRelativeComSpeed = 0.0f;
	if(ok && softSoft)
	{
		PxReal kineticA = 0.0f, kineticB = 0.0f, massA = 0.0f, massB = 0.0f;
		PxVec3 velocityA(0.0f), velocityB(0.0f);
		ok = deformableKineticState(*fixture.volume, kineticA, massA, velocityA) &&
			deformableKineticState(*fixture.volumeB, kineticB, massB, velocityB);
		initialSoftPairKineticEnergy = kineticA + kineticB;
		maximumSoftPairKineticEnergy = initialSoftPairKineticEnergy;
		finalSoftPairKineticEnergy = initialSoftPairKineticEnergy;
		maximumSoftPairRelativeComSpeed = (velocityA - velocityB).magnitude();
	}
	if(ok && softSoftInitialContainment)
	{
		bool finiteAB = false, finiteBA = false;
		initialInsideParticles = countInsideCollisionVertices(*fixture.volume,
			*fixture.volumeB, finiteAB) + countInsideCollisionVertices(*fixture.volumeB,
				*fixture.volume, finiteBA);
		// This independent geometry oracle integrates the same zero-multiplier
		// unilateral potential over the initial directed vertex/face contacts.
		// It is an energy reservoir, not a velocity or displacement cap.
		const PxReal fallbackMargin = 0.04f *
			fixture.physics->getTolerancesScale().length;
		ok = finiteAB && finiteBA && initialSoftContactPotential(*fixture.volume,
			*fixture.volumeB, fallbackMargin, initialSoftContactPotentialEnergy,
			initialSoftContactCount);
	}
	PxU32 completed = 0, nonFinite = 0;
	PxU32 maximumInsideParticles = initialInsideParticles, finalInsideParticles =
		initialInsideParticles, firstInsideFrame = initialInsideParticles ? 0 : PX_MAX_U32;
	PxU32 maximumBoxCoreTriangleHits = 0, maximumSoftBoundaryCrossings = 0;
	PxU32 firstBoxCoreHitFrame = PX_MAX_U32;
	PxU32 firstSoftBoundaryCrossingFrame = PX_MAX_U32;
	PxU32 maximumBoxEdgeTriangleCrossings = 0;
	PxU32 firstBoxEdgeTriangleCrossingFrame = PX_MAX_U32;
	PxU32 maximumLeftoverOpenTriangles = 0;
	PxU32 firstLeftoverOpenFrame = PX_MAX_U32;
	PxReal firstLeftoverMinClipSdf = 0.0f;
	PxReal firstLeftoverMinChebyshev = 0.0f;
	PxReal firstLeftoverMaxChebyshev = 0.0f;
	PxU32 currentLeftoverOpen = 0;
	PxU32 currentIntersecting = 0;
	PxU32 currentCovering = 0;
	PxU32 currentBoxEdges = 0;
	PxU32 currentHoles = 0;
	PxU32 currentSatNear = 0;
	PxU32 currentInflated = 0;
	PxReal currentHoleSat = PX_MAX_F32;
	PxReal currentMinSat15 = PX_MAX_F32;
	PxReal currentDCom = 0.0f;
	PxReal currentDShape = 0.0f;
	PxReal currentMinVertexSdf = PX_MAX_F32;
	PxReal leftoverMinClipSdf = PX_MAX_F32;
	PxReal leftoverMinChebyshev = PX_MAX_F32;
	PxReal leftoverMaxChebyshev = 0.0f;
	PxReal coveringHangingSdf = PX_MAX_F32;
	PxU32 maximumRigidCentersInsideSoft = 0, maximumRigidVerticesInsideSoft = 0;
	PxReal minY = PX_MAX_F32, maxY = -PX_MAX_F32;
	PxReal minimumDetF = PX_MAX_F32;
	PxReal maxComY = -PX_MAX_F32, minComY = PX_MAX_F32;
	const PxReal boxInitial = fixture.dynamicBox ? fixture.dynamicBox->getGlobalPose().p.y : 0.0f;
	PxReal boxFinal = boxInitial;
	PxReal maximumTripletRelativeDrift = 0.0f;
	PxReal maximumTripletShapeError = 0.0f;
	PxReal maximumSoftPairEdgeStrain = 0.0f;
	PxReal finalSoftPairEdgeStrain = 0.0f;
	PxReal peakRollingAngularSpeed = 0.0f;
	PxF64 totalStepMs = 0.0, maximumStepMs = 0.0;
	for(PxU32 frame = 0; ok && frame < options.frames; ++frame)
	{
		currentLeftoverOpen = 0;
		currentIntersecting = 0;
		currentCovering = 0;
		currentBoxEdges = 0;
		currentHoles = 0;
		currentSatNear = 0;
		currentInflated = 0;
		currentHoleSat = PX_MAX_F32;
		currentMinSat15 = PX_MAX_F32;
		currentDCom = 0.0f;
		currentDShape = 0.0f;
		currentMinVertexSdf = PX_MAX_F32;
		leftoverMinClipSdf = PX_MAX_F32;
		leftoverMinChebyshev = PX_MAX_F32;
		leftoverMaxChebyshev = 0.0f;
		coveringHangingSdf = PX_MAX_F32;
		if(frame == 0 && (rigidSqueeze || rigidOnSoft) && fixture.dynamicBox)
		{
			bool finiteOpenA = false, finiteOpenB = false;
			PxReal clipA = PX_MAX_F32, chebMinA = PX_MAX_F32, chebMaxA = 0.0f;
			PxReal clipB = PX_MAX_F32, chebMinB = PX_MAX_F32, chebMaxB = 0.0f;
			PxU32 intersectingA = 0, coveringA = 0, intersectingB = 0, coveringB = 0;
			PxReal hangA = PX_MAX_F32, hangB = PX_MAX_F32;
			const PxU32 leftoverOpen =
				countLeftoverOpenTriangles(*fixture.volume,
					fixture.dynamicBox->getGlobalPose(),
					fixture.dynamicBoxHalfExtent, finiteOpenA, clipA, chebMinA, chebMaxA,
					intersectingA, coveringA, hangA) +
				(fixture.volumeB ? countLeftoverOpenTriangles(*fixture.volumeB,
					fixture.dynamicBox->getGlobalPose(),
					fixture.dynamicBoxHalfExtent, finiteOpenB, clipB, chebMinB, chebMaxB,
					intersectingB, coveringB, hangB) : 0);
			if(!finiteOpenA || (fixture.volumeB && !finiteOpenB)) ++nonFinite;
			std::printf("[AVBD_C01B] preStep leftoverOpen=%u intersecting=%u covering=%u boxY=%.6g minClipSdf=%.6g minChebyshev=%.6g maxChebyshev=%.6g minCoveringHangingSdf=%.6g\n",
				leftoverOpen, intersectingA + intersectingB, coveringA + coveringB,
				double(fixture.dynamicBox->getGlobalPose().p.y),
				double(PxMin(clipA, clipB)),
				double(PxMin(chebMinA, chebMinB)),
				double(PxMax(chebMaxA, chebMaxB)),
				double(PxMin(hangA, hangB)));
		}
		const bool activeAtStart = !fixture.volume->isSleeping();
		PxTime stepTimer;
		if(!fixture.scene->simulate(options.dt) || !fixture.scene->fetchResults(true))
		{
			ok = false;
			break;
		}
		const PxF64 stepMs = stepTimer.getElapsedSeconds() * 1000.0;
		totalStepMs += stepMs;
		maximumStepMs = PxMax(maximumStepMs, stepMs);
		if(!benchmark.frame(stepMs, activeAtStart)) { ok = false; break; }
		++completed;
		embedding.sample(*fixture.volume);
		if(fixture.volumeB) embedding.sample(*fixture.volumeB);
		PxVec3 com;
		bool finite;
		VolumeAvbdSnippet::bounds(*fixture.volume, minY, maxY, com, finite);
		if(!finite) ++nonFinite;
		minComY = PxMin(minComY, com.y);
		maxComY = PxMax(maxComY, com.y);
		if(rollingSphere)
		{
			PxVec3 omega(0.0f);
			if(!volumeAngularVelocity(*fixture.volume, omega)) ++nonFinite;
			else peakRollingAngularSpeed = PxMax(peakRollingAngularSpeed,
				PxAbs(omega.z));
		}
		bool finiteDet = false;
		minimumDetF = PxMin(minimumDetF, minimumTetDetF(
			*fixture.volume->getSimulationMesh(),
			fixture.volume->getSimPositionInvMassBufferD(), simulationRest, finiteDet));
		if(!finiteDet) ++nonFinite;
		if(rigidOnSoft)
		{
			bool finiteStrain = false;
			const PxReal strain = maximumTetEdgeStrain(
				*fixture.volume->getSimulationMesh(),
				fixture.volume->getSimPositionInvMassBufferD(), simulationRest,
				finiteStrain);
			if(!finiteStrain) ++nonFinite;
			maximumSoftPairEdgeStrain = PxMax(maximumSoftPairEdgeStrain, strain);
			finalSoftPairEdgeStrain = strain;
		}
		if(fixture.dynamicBox)
		{
			boxFinal = fixture.dynamicBox->getGlobalPose().p.y;
			if(!fixture.dynamicBox->getGlobalPose().isFinite()) ++nonFinite;
		}
		if(staticBox || dynamic || rigidOnSoft)
		{
			bool finiteBox = false;
			const PxTransform pose = fixture.dynamicBox ? fixture.dynamicBox->getGlobalPose() :
				PxTransform(PxVec3(0.0f, 0.5f, 0.0f));
			const PxVec3 coreExtent = fixture.dynamicBox ?
				fixture.dynamicBoxHalfExtent - PxVec3(0.01f) : PxVec3(1.99f, 0.49f, 1.99f);
			const PxU32 hits = countBoxCoreTriangleIntersections(*fixture.volume,
				pose, coreExtent, finiteBox);
			if(!finiteBox) ++nonFinite;
			maximumBoxCoreTriangleHits = PxMax(maximumBoxCoreTriangleHits, hits);
			if(hits && firstBoxCoreHitFrame == PX_MAX_U32)
				firstBoxCoreHitFrame = frame;
			{
				bool finiteEdges = false;
				PxU32 edgeHits = countBoxEdgeTriangleCrossings(*fixture.volume,
					pose, fixture.dynamicBox ? fixture.dynamicBoxHalfExtent :
					PxVec3(2.0f, 0.5f, 2.0f), finiteEdges);
				if(!finiteEdges) ++nonFinite;
				if(rigidSqueeze)
				{
					bool finiteEdgesB = false;
					edgeHits += countBoxEdgeTriangleCrossings(*fixture.volumeB,
						pose, fixture.dynamicBoxHalfExtent, finiteEdgesB);
					if(!finiteEdgesB) ++nonFinite;
				}
				currentBoxEdges = edgeHits;
				maximumBoxEdgeTriangleCrossings = PxMax(
					maximumBoxEdgeTriangleCrossings, edgeHits);
				if(edgeHits && firstBoxEdgeTriangleCrossingFrame == PX_MAX_U32)
				{
					firstBoxEdgeTriangleCrossingFrame = frame;
					// Gate diagnostics use the exact full box here.  The regular core
					// dump is deliberately inset by 1 cm and therefore cannot identify
					// a strict box-edge/triangle crossing whose vertices all remain out.
					if(rigidSqueeze && fixture.dynamicBox)
					{
						dumpBoxCoreTriangleHits("A-edge", frame, *fixture.volume,
							*fixture.dynamicBox, fixture.dynamicBoxHalfExtent,
							fixture.dynamicBoxHalfExtent);
						dumpBoxCoreTriangleHits("B-edge", frame, *fixture.volumeB,
							*fixture.dynamicBox, fixture.dynamicBoxHalfExtent,
							fixture.dynamicBoxHalfExtent);
					}
				}
			}
			if((rigidSqueeze || rigidOnSoft) && fixture.dynamicBox)
			{
				bool finiteOpenA = false, finiteOpenB = false;
				PxReal clipA = PX_MAX_F32, chebMinA = PX_MAX_F32, chebMaxA = 0.0f;
				PxReal clipB = PX_MAX_F32, chebMinB = PX_MAX_F32, chebMaxB = 0.0f;
				PxU32 intersectingA = 0, coveringA = 0, intersectingB = 0, coveringB = 0;
				PxReal hangA = PX_MAX_F32, hangB = PX_MAX_F32;
				currentLeftoverOpen =
					countLeftoverOpenTriangles(*fixture.volume,
						pose, fixture.dynamicBoxHalfExtent, finiteOpenA,
						clipA, chebMinA, chebMaxA, intersectingA, coveringA, hangA) +
					(fixture.volumeB ? countLeftoverOpenTriangles(*fixture.volumeB,
						pose, fixture.dynamicBoxHalfExtent, finiteOpenB,
						clipB, chebMinB, chebMaxB, intersectingB, coveringB, hangB) : 0);
				currentIntersecting = intersectingA + intersectingB;
				currentCovering = coveringA + coveringB;
				leftoverMinClipSdf = PxMin(clipA, clipB);
				leftoverMinChebyshev = PxMin(chebMinA, chebMinB);
				leftoverMaxChebyshev = PxMax(chebMaxA, chebMaxB);
				coveringHangingSdf = PxMin(hangA, hangB);
				bool finiteSdfA = false, finiteSdfB = true;
				currentMinVertexSdf = minCollisionVertexBoxSdf(*fixture.volume,
					pose, fixture.dynamicBoxHalfExtent, finiteSdfA);
				if(fixture.volumeB)
				{
					currentMinVertexSdf = PxMin(currentMinVertexSdf,
						minCollisionVertexBoxSdf(*fixture.volumeB, pose,
							fixture.dynamicBoxHalfExtent, finiteSdfB));
				}
				if(!finiteSdfA || !finiteSdfB) ++nonFinite;
				if(!finiteOpenA || (fixture.volumeB && !finiteOpenB)) ++nonFinite;
				const PxReal vfMargin = 0.02f;
				bool finiteHoleA = false, finiteHoleB = true;
				PxReal holeSatA = PX_MAX_F32, holeSatB = PX_MAX_F32;
				PxReal sat15A = PX_MAX_F32, sat15B = PX_MAX_F32;
				PxU32 satNearA = 0, satNearB = 0, inflatedA = 0, inflatedB = 0;
				PxU32 holes = countApproachingInteriorHoles(*fixture.volume, pose,
					fixture.dynamicBoxHalfExtent, vfMargin, finiteHoleA, holeSatA,
					satNearA, sat15A, inflatedA);
				if(fixture.volumeB)
				{
					holes += countApproachingInteriorHoles(*fixture.volumeB, pose,
						fixture.dynamicBoxHalfExtent, vfMargin, finiteHoleB, holeSatB,
						satNearB, sat15B, inflatedB);
				}
				if(!finiteHoleA || !finiteHoleB) ++nonFinite;
				const PxReal holeSat = PxMin(holeSatA, holeSatB);
				const PxU32 satNear = satNearA + satNearB;
				const PxU32 inflated = inflatedA + inflatedB;
				const PxReal minSat15 = PxMin(sat15A, sat15B);
				PxVec3 comA(0.0f), comB(0.0f);
				bool finiteComA = centroid(fixture.volume->getSimPositionInvMassBufferD(),
					simCount, comA);
				PxReal dCom = 0.0f, dShape = 0.0f;
				if(finiteComA)
				{
					bool finiteShapeA = false;
					dCom = (comA - prevComA).magnitude();
					dShape = maximumTranslationRemovedDisplacement(
						fixture.volume->getSimPositionInvMassBufferD(), prevSimA,
						comA, prevComA, finiteShapeA);
					if(!finiteShapeA) ++nonFinite;
					prevComA = comA;
					const PxVec4* nowA = fixture.volume->getSimPositionInvMassBufferD();
					for(PxU32 i = 0; i < prevSimA.size(); ++i)
						prevSimA[i] = nowA[i].getXYZ();
				}
				else ++nonFinite;
				if(fixture.volumeB)
				{
					const PxU32 countB = fixture.volumeB->getSimulationMesh()->getNbVertices();
					if(centroid(fixture.volumeB->getSimPositionInvMassBufferD(), countB, comB))
					{
						bool finiteShapeB = false;
						dCom = PxMax(dCom, (comB - prevComB).magnitude());
						dShape = PxMax(dShape, maximumTranslationRemovedDisplacement(
							fixture.volumeB->getSimPositionInvMassBufferD(), prevSimB,
							comB, prevComB, finiteShapeB));
						if(!finiteShapeB) ++nonFinite;
						prevComB = comB;
						const PxVec4* nowB = fixture.volumeB->getSimPositionInvMassBufferD();
						for(PxU32 i = 0; i < prevSimB.size(); ++i)
							prevSimB[i] = nowB[i].getXYZ();
					}
					else ++nonFinite;
				}
				currentHoles = holes;
				currentSatNear = satNear;
				currentInflated = inflated;
				currentHoleSat = holeSat;
				currentMinSat15 = minSat15;
				currentDCom = dCom;
				currentDShape = dShape;
				if(currentLeftoverOpen && firstLeftoverOpenFrame == PX_MAX_U32)
				{
					firstLeftoverOpenFrame = frame;
					firstLeftoverMinClipSdf = leftoverMinClipSdf;
					firstLeftoverMinChebyshev = leftoverMinChebyshev;
					firstLeftoverMaxChebyshev = leftoverMaxChebyshev;
				}
				maximumLeftoverOpenTriangles = PxMax(
					maximumLeftoverOpenTriangles, currentLeftoverOpen);
				if(rigidOnSoft && (frame % 20u == 0u || firstLeftoverOpenFrame == frame))
				{
					std::printf("[AVBD_C01B] timeline frame=%u boxY=%.6g minDetF=%.6g cores=%u leftoverOpen=%u leftoverNow=%u intersecting=%u covering=%u minClipSdf=%.6g minChebyshev=%.6g maxChebyshev=%.6g minCoveringHangingSdf=%.6g minVertexSdf=%.6g boxEdges=%u boxEdgesNow=%u holes=%u minHoleSat=%.6g satNear=%u inflated=%u minSat15=%.6g dCom=%.6g dShape=%.6g\n",
						frame, double(fixture.dynamicBox->getGlobalPose().p.y),
						double(minimumDetF), maximumBoxCoreTriangleHits,
						maximumLeftoverOpenTriangles, currentLeftoverOpen,
						currentIntersecting, currentCovering,
						double(leftoverMinClipSdf), double(leftoverMinChebyshev),
						double(leftoverMaxChebyshev), double(coveringHangingSdf),
						double(currentMinVertexSdf),
						maximumBoxEdgeTriangleCrossings, currentBoxEdges,
						currentHoles, double(currentHoleSat),
						currentSatNear, currentInflated, double(currentMinSat15),
						double(currentDCom), double(currentDShape));
				}
			}
			if(rigidSqueeze)
			{
				bool finiteBoxB = false;
				const PxU32 hitsB = countBoxCoreTriangleIntersections(*fixture.volumeB,
					pose, coreExtent, finiteBoxB);
				if(!finiteBoxB) ++nonFinite;
				maximumBoxCoreTriangleHits = PxMax(maximumBoxCoreTriangleHits, hitsB);
				if(hitsB && firstBoxCoreHitFrame == PX_MAX_U32)
					firstBoxCoreHitFrame = frame;
				if(fixture.dynamicBox && firstBoxCoreHitFrame == frame)
				{
					dumpBoxCoreTriangleHits("A", frame, *fixture.volume,
						*fixture.dynamicBox, fixture.dynamicBoxHalfExtent, coreExtent);
					dumpBoxCoreTriangleHits("B", frame, *fixture.volumeB,
						*fixture.dynamicBox, fixture.dynamicBoxHalfExtent, coreExtent);
				}
			}
			if((rigidSqueeze || freeFallTriplet) && fixture.dynamicBox && fixture.volumeB)
			{
				const PxVec3 erodedExtent = fixture.dynamicBoxHalfExtent - PxVec3(0.01f);
				PxVec3 rigidPoints[9];
				rigidPoints[0] = pose.p;
				for(PxU32 vertex = 0; vertex < 8; ++vertex)
				{
					const PxVec3 local((vertex & 1) ? erodedExtent.x : -erodedExtent.x,
						(vertex & 2) ? erodedExtent.y : -erodedExtent.y,
						(vertex & 4) ? erodedExtent.z : -erodedExtent.z);
					rigidPoints[1 + vertex] = pose.transform(local);
				}
				bool finiteCenterA = false, finiteCenterB = false;
				bool finiteVerticesA = false, finiteVerticesB = false;
				const PxU32 centersInside = countPointsInsideCollisionVolume(
					rigidPoints, 1, *fixture.volume, finiteCenterA) +
					countPointsInsideCollisionVolume(rigidPoints, 1,
						*fixture.volumeB, finiteCenterB);
				const PxU32 verticesInside = countPointsInsideCollisionVolume(
					rigidPoints + 1, 8, *fixture.volume, finiteVerticesA) +
					countPointsInsideCollisionVolume(rigidPoints + 1, 8,
						*fixture.volumeB, finiteVerticesB);
				if(!finiteCenterA || !finiteCenterB || !finiteVerticesA ||
					!finiteVerticesB) ++nonFinite;
				maximumRigidCentersInsideSoft = PxMax(maximumRigidCentersInsideSoft,
					centersInside);
				maximumRigidVerticesInsideSoft = PxMax(maximumRigidVerticesInsideSoft,
					verticesInside);
			}
		}
		if(softSoft)
		{
			PxVec3 simCenterB(0.0f);
			const PxU32 simCountB = fixture.volumeB->getSimulationMesh()->getNbVertices();
			if(!centroid(fixture.volumeB->getSimPositionInvMassBufferD(), simCountB,
				simCenterB)) ++nonFinite;
			bool finiteDetB = false;
			minimumDetF = PxMin(minimumDetF, minimumTetDetF(
				*fixture.volumeB->getSimulationMesh(),
				fixture.volumeB->getSimPositionInvMassBufferD(), simulationRestB, finiteDetB));
			if(!finiteDetB) ++nonFinite;
			bool finiteStrainA = false, finiteStrainB = false;
			const PxReal strainA = maximumTetEdgeStrain(
				*fixture.volume->getSimulationMesh(),
				fixture.volume->getSimPositionInvMassBufferD(), simulationRest,
				finiteStrainA);
			const PxReal strainB = maximumTetEdgeStrain(
				*fixture.volumeB->getSimulationMesh(),
				fixture.volumeB->getSimPositionInvMassBufferD(), simulationRestB,
				finiteStrainB);
			if(!finiteStrainA || !finiteStrainB) ++nonFinite;
			finalSoftPairEdgeStrain = PxMax(strainA, strainB);
			maximumSoftPairEdgeStrain = PxMax(maximumSoftPairEdgeStrain,
				finalSoftPairEdgeStrain);
			PxReal kineticA = 0.0f, kineticB = 0.0f, massA = 0.0f, massB = 0.0f;
			PxVec3 velocityA(0.0f), velocityB(0.0f);
			if(!deformableKineticState(*fixture.volume, kineticA, massA, velocityA) ||
				!deformableKineticState(*fixture.volumeB, kineticB, massB, velocityB))
				++nonFinite;
			else
			{
				finalSoftPairKineticEnergy = kineticA + kineticB;
				maximumSoftPairKineticEnergy = PxMax(maximumSoftPairKineticEnergy,
					finalSoftPairKineticEnergy);
				maximumSoftPairRelativeComSpeed = PxMax(
					maximumSoftPairRelativeComSpeed,
					(velocityA - velocityB).magnitude());
			}
			bool finiteAB = false, finiteBA = false;
			const PxU32 inside = countInsideCollisionVertices(*fixture.volume,
				*fixture.volumeB, finiteAB) + countInsideCollisionVertices(*fixture.volumeB,
					*fixture.volume, finiteBA);
			if(!finiteAB || !finiteBA) ++nonFinite;
			bool finiteCrossings = false;
			const PxU32 crossings = countProperBoundaryCrossings(*fixture.volume,
				*fixture.volumeB, finiteCrossings);
			if(!finiteCrossings) ++nonFinite;
			maximumSoftBoundaryCrossings = PxMax(maximumSoftBoundaryCrossings,
				crossings);
			if(crossings && firstSoftBoundaryCrossingFrame == PX_MAX_U32)
				firstSoftBoundaryCrossingFrame = frame;
			if(rigidSqueeze && (frame % 20u == 0u ||
				(firstBoxCoreHitFrame == frame) ||
				(firstBoxEdgeTriangleCrossingFrame == frame) ||
				(firstLeftoverOpenFrame == frame) ||
				(firstSoftBoundaryCrossingFrame == frame) ||
				frame <= 20u))
			{
				PxReal minYB = PX_MAX_F32, maxYB = -PX_MAX_F32;
				PxVec3 comB(0.0f);
				bool finiteB = false;
				VolumeAvbdSnippet::bounds(*fixture.volumeB, minYB, maxYB, comB, finiteB);
				if(!finiteB) ++nonFinite;
				std::printf("[AVBD_C01B] timeline frame=%u boxY=%.6g minDetF=%.6g cores=%u leftoverOpen=%u leftoverNow=%u intersecting=%u covering=%u minClipSdf=%.6g minChebyshev=%.6g maxChebyshev=%.6g minCoveringHangingSdf=%.6g minVertexSdf=%.6g crossings=%u inside=%u boxEdges=%u boxEdgesNow=%u volA=%.6g,%.6g,%.6g volB=%.6g,%.6g,%.6g holes=%u minHoleSat=%.6g satNear=%u inflated=%u minSat15=%.6g dCom=%.6g dShape=%.6g\n",
					frame, double(fixture.dynamicBox ?
						fixture.dynamicBox->getGlobalPose().p.y : 0.0),
					double(minimumDetF), maximumBoxCoreTriangleHits,
					maximumLeftoverOpenTriangles, currentLeftoverOpen,
					currentIntersecting, currentCovering,
					double(leftoverMinClipSdf), double(leftoverMinChebyshev),
					double(leftoverMaxChebyshev), double(coveringHangingSdf),
					double(currentMinVertexSdf),
					crossings, inside, maximumBoxEdgeTriangleCrossings, currentBoxEdges,
					double(minY), double(maxY), double(com.y),
					double(minYB), double(maxYB), double(comB.y),
					currentHoles, double(currentHoleSat),
					currentSatNear, currentInflated, double(currentMinSat15),
					double(currentDCom), double(currentDShape));
			}
			maximumInsideParticles = PxMax(maximumInsideParticles, inside);
			finalInsideParticles = inside;
			if(inside && firstInsideFrame == PX_MAX_U32) firstInsideFrame = frame;
			if(freeFallTriplet && fixture.dynamicBox)
			{
				const PxVec3 boxPosition = fixture.dynamicBox->getGlobalPose().p;
				maximumTripletRelativeDrift = PxMax(maximumTripletRelativeDrift,
					PxMax(PxAbs((com.y - boxPosition.y) - (simInitial.y - boxInitial)),
						PxAbs((simCenterB.y - boxPosition.y) - (simInitialB.y - boxInitial))));
				bool finiteShapeA = false, finiteShapeB = false;
				maximumTripletShapeError = PxMax(maximumTripletShapeError,
					maximumTranslationRemovedDisplacement(
						fixture.volume->getSimPositionInvMassBufferD(), simulationRest,
						com, simInitial, finiteShapeA));
				maximumTripletShapeError = PxMax(maximumTripletShapeError,
					maximumTranslationRemovedDisplacement(
						fixture.volumeB->getSimPositionInvMassBufferD(), simulationRestB,
						simCenterB, simInitialB, finiteShapeB));
				if(!finiteShapeA || !finiteShapeB) ++nonFinite;
			}
		}
	}
	if(ok)
	{
		if(!centroid(fixture.volume->getSimPositionInvMassBufferD(), simCount, simFinal) ||
			!centroid(fixture.volume->getPositionInvMassBufferD(), collisionCount, collisionFinal))
			++nonFinite;
	}
	ok = benchmark.finish() && ok;
	std::printf("[AVBD_C01B] maxBoxEdgeTriangleCrossings=%u\n",
		maximumBoxEdgeTriangleCrossings);
	std::printf("[AVBD_C01B] maxLeftoverOpenTriangles=%u firstLeftoverOpenFrame=%u firstMinClipSdf=%.6g firstMinChebyshev=%.6g firstMaxChebyshev=%.6g\n",
		maximumLeftoverOpenTriangles, firstLeftoverOpenFrame,
		double(firstLeftoverMinClipSdf), double(firstLeftoverMinChebyshev),
		double(firstLeftoverMaxChebyshev));
	std::printf("[AVBD_BENCH] case=%s simulationVertices=%u simulationTets=%u sceneDispatcherThreads=%u averageStepMs=%.9g maxStepMs=%.9g\n",
		options.caseName.c_str(), simCount, fixture.volume ? fixture.volume->getSimulationMesh()->getNbTetrahedrons() : 0,
		options.dispatcherThreads, completed ? totalStepMs / completed : 0.0, maximumStepMs);
	const PxReal simTravel = (simFinal - simInitial).magnitude();
	const PxReal collisionTravel = (collisionFinal - collisionInitial).magnitude();
	const PxReal embeddingError = ((simFinal - simInitial) -
		(collisionFinal - collisionInitial)).magnitude();
	PxReal softPairCenterDistanceY = 0.0f;
	bool physical = minY > -0.15f && maxY < (freeFallTriplet ? 30.0f : 20.0f) &&
		minComY > -0.2f &&
		minimumDetF >= 0.049f && embeddingOracleValidated && embedding.passed() &&
		restEmbedding.passed() && partitionEmbedding.passed();
	if(staticBox) physical = physical && minY > 0.75f &&
		maximumBoxCoreTriangleHits == 0;
	if(dynamic && !rigidSqueeze) physical = physical && boxFinal < boxInitial - 0.02f &&
		minY > 0.55f && maximumBoxCoreTriangleHits == 0;
	if(rigidSqueeze) physical = physical && boxFinal < boxInitial - 0.02f &&
		maximumBoxCoreTriangleHits == 0 && maximumRigidCentersInsideSoft == 0 &&
		maximumRigidVerticesInsideSoft == 0 && maximumBoxEdgeTriangleCrossings == 0;
	if(freeFallTriplet) physical = physical && boxInitial - boxFinal > 4.5f &&
		maximumBoxCoreTriangleHits == 0 && minimumDetF > 0.98f &&
		maximumRigidCentersInsideSoft == 0 && maximumRigidVerticesInsideSoft == 0 &&
		maximumTripletRelativeDrift < 0.02f && maximumTripletShapeError < 0.02f;
	if(embedded) physical = physical && simCount != collisionCount && simTravel > 0.5f &&
		collisionTravel > 0.5f && embeddingError < 0.08f;
	if(embeddedSameCount) physical = physical && simCount == collisionCount &&
		initialEmbeddingIdentityError > 0.1f;
	if(rollingSphere) physical = physical && PxAbs(simFinal.x - simInitial.x) > 0.1f &&
		peakRollingAngularSpeed > 0.2f;
	if(softSoft)
	{
		PxVec3 centerA(0.0f), centerB(0.0f);
		const bool finiteCenters = centroid(fixture.volume->getPositionInvMassBufferD(),
			fixture.volume->getCollisionMesh()->getNbVertices(), centerA) &&
			centroid(fixture.volumeB->getPositionInvMassBufferD(),
				fixture.volumeB->getCollisionMesh()->getNbVertices(), centerB);
		softPairCenterDistanceY = PxAbs(centerA.y - centerB.y);
		if(softSoftInitialContainment)
			physical = physical && finiteCenters && initialInsideParticles > 0 &&
				finalInsideParticles == 0 && softPairCenterDistanceY > 0.70f &&
				finalSoftPairEdgeStrain < 0.20f && initialSoftContactCount > 0 &&
				initialSoftContactPotentialEnergy > 0.0f &&
				maximumSoftPairKineticEnergy <=
					initialSoftContactPotentialEnergy * 1.005f + 1.0e-3f;
		else
			physical = physical && finiteCenters && maximumInsideParticles == 0 &&
				finalInsideParticles == 0 && maximumSoftBoundaryCrossings == 0 &&
				(rigidSqueeze || freeFallTriplet || PxAbs(centerA.y - centerB.y) > 0.70f);
		if(softSoftRecovery)
			physical = physical && softPairCenterDistanceY > 2.5f &&
				finalSoftPairEdgeStrain < 0.08f;
	}
	if(rigidOnSoft) physical = physical && boxFinal > 1.60f && boxFinal < 4.00f &&
		maximumBoxCoreTriangleHits == 0 && finalSoftPairEdgeStrain < 0.20f &&
		minimumDetF > 0.20f && minComY > 0.80f;
	const char* reason = "none";
	if(!ok) reason = "initialization_or_fetch";
	else if(nonFinite || fixture.errors.getFatalCount()) reason = "runtime_error";
	else if(!mutationReadback) reason = "host_buffer_not_mutable";
	else if(!physical) reason = "physical_contract";
	const bool pass = ok && !nonFinite && !fixture.errors.getFatalCount() &&
		mutationReadback && physical;
	std::printf("[AVBD_GATE] schema=1 snippet=SnippetDeformableVolumeAVBD solver=avbd case=%s execution=%s frames=%u completedFrames=%u status=%s reason=%s capability=SUPPORTED validation=GATED minY=%.9g maxY=%.9g minComY=%.9g maxComY=%.9g minDetF=%.9g simVertices=%u collisionVertices=%u simTravel=%.9g collisionTravel=%.9g embeddingError=%.9g initialEmbeddingIdentityError=%.9g initialInsideParticles=%u maxInsideParticles=%u finalInsideParticles=%u firstInsideFrame=%u maxBoxCoreTriangleHits=%u firstBoxCoreHitFrame=%u maxBoxEdgeTriangleCrossings=%u firstBoxEdgeTriangleCrossingFrame=%u maxRigidCentersInsideSoft=%u maxRigidVerticesInsideSoft=%u maxSoftBoundaryCrossings=%u softPairCenterDistanceY=%.9g maxSoftPairEdgeStrain=%.9g finalSoftPairEdgeStrain=%.9g initialSoftContactCount=%u initialSoftContactPotentialEnergy=%.9g initialSoftPairKineticEnergy=%.9g maxSoftPairKineticEnergy=%.9g finalSoftPairKineticEnergy=%.9g maxSoftPairRelativeComSpeed=%.9g boxInitialY=%.9g boxFinalY=%.9g maxTripletRelativeDrift=%.9g maxTripletShapeError=%.9g peakRollingAngularSpeed=%.9g mutationReadback=%u nonFinite=%u fatalErrors=%u embeddingMaxVertexError=%.9g embeddingMaxErrorRatio=%.9g embeddingSamples=%llu embeddingInvalid=%u embeddingOracleValidated=%u embeddingRestError=%.9g embeddingRestErrorRatio=%.9g embeddingPartitionError=%.9g embeddingPartitionErrorRatio=%.9g\n",
		options.caseName.c_str(), Snippets::getExecutionName(options.execution), options.frames,
		completed, pass ? "PASS" : "FAIL", reason, double(minY), double(maxY),
		double(minComY), double(maxComY), double(minimumDetF), simCount, collisionCount, double(simTravel),
		double(collisionTravel), double(embeddingError), double(initialEmbeddingIdentityError), initialInsideParticles,
		maximumInsideParticles,
		finalInsideParticles, firstInsideFrame, maximumBoxCoreTriangleHits,
		firstBoxCoreHitFrame, maximumBoxEdgeTriangleCrossings,
		firstBoxEdgeTriangleCrossingFrame, maximumRigidCentersInsideSoft,
		maximumRigidVerticesInsideSoft,
		maximumSoftBoundaryCrossings, double(softPairCenterDistanceY),
		double(maximumSoftPairEdgeStrain), double(finalSoftPairEdgeStrain),
		initialSoftContactCount, double(initialSoftContactPotentialEnergy),
		double(initialSoftPairKineticEnergy), double(maximumSoftPairKineticEnergy),
		double(finalSoftPairKineticEnergy), double(maximumSoftPairRelativeComSpeed),
		double(boxInitial), double(boxFinal),
		double(maximumTripletRelativeDrift), double(maximumTripletShapeError),
		double(peakRollingAngularSpeed),
		mutationReadback ? 1u : 0u, nonFinite, fixture.errors.getFatalCount(),
		embedding.maxVertexError, embedding.maxErrorRatio,
		static_cast<unsigned long long>(embedding.samples), embedding.invalid,
		embeddingOracleValidated ? 1u : 0u, restEmbedding.maxVertexError, restEmbedding.maxErrorRatio,
		partitionEmbedding.maxVertexError, partitionEmbedding.maxErrorRatio);
	return pass ? Snippets::eHEADLESS_PASS : Snippets::eHEADLESS_GATE_FAILED;
}
