#include "SnippetDeformableSurfaceAVBDHeadless.h"
#include "../snippetjointsoftbody/SnippetJointSoftBodyPvd.h"
#include "SnippetDeformableSurfaceAVBDCommon.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "GuIntersectionTriangleBox.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

using namespace physx;
namespace
{
static const PxU32 kWidth = 9, kHeight = 9;

struct SurfaceStrain
{
	PxReal edge, principal, area;
	SurfaceStrain() : edge(0.0f), principal(0.0f), area(0.0f) {}
};

// Observe the cooked triangles in every direction, including shear. A final
// sample of only one grid axis misses transient and diagonal over-stretch.
SurfaceStrain measureStrain(const PxTriangleMesh& mesh, const PxVec4* rest,
	const PxVec4* p)
{
	SurfaceStrain result;
	double restArea = 0.0, area = 0.0;
	const bool small = mesh.getTriangleMeshFlags().isSet(PxTriangleMeshFlag::e16_BIT_INDICES);
	const PxU16* ids16 = static_cast<const PxU16*>(mesh.getTriangles());
	const PxU32* ids32 = static_cast<const PxU32*>(mesh.getTriangles());
	for(PxU32 t = 0; t < mesh.getNbTriangles(); ++t)
	{
		PxU32 ids[3];
		for(PxU32 k = 0; k < 3; ++k) ids[k] = small ? ids16[3*t+k] : ids32[3*t+k];
		const PxVec3 r0 = rest[ids[1]].getXYZ() - rest[ids[0]].getXYZ();
		const PxVec3 r1 = rest[ids[2]].getXYZ() - rest[ids[0]].getXYZ();
		const PxVec3 e0 = p[ids[1]].getXYZ() - p[ids[0]].getXYZ();
		const PxVec3 e1 = p[ids[2]].getXYZ() - p[ids[0]].getXYZ();
		const PxReal a = r0.magnitude(), twiceArea = r0.cross(r1).magnitude();
		if(a <= 0.0f || twiceArea <= 0.0f) continue;
		const PxVec3 f0 = e0 / a;
		const PxVec3 f1 = (e1 - f0 * (r0.dot(r1) / a)) * (a / twiceArea);
		const double c00 = f0.dot(f0), c01 = f0.dot(f1), c11 = f1.dot(f1);
		const double discriminant = (c00-c11)*(c00-c11) + 4.0*c01*c01;
		result.principal = PxMax(result.principal,
			PxReal(std::sqrt(0.5*(c00+c11+std::sqrt(discriminant)))));
		for(PxU32 k = 0; k < 3; ++k)
		{
			const PxU32 i = ids[k], j = ids[(k+1)%3];
			result.edge = PxMax(result.edge, (p[j].getXYZ()-p[i].getXYZ()).magnitude() /
				(rest[j].getXYZ()-rest[i].getXYZ()).magnitude());
		}
		restArea += twiceArea;
		area += e0.cross(e1).magnitude();
	}
	result.area = restArea > 0.0 ? PxReal(area/restArea) : 0.0f;
	return result;
}

void writeTrajectory(FILE* file, PxU32 frame, const PxTriangleMesh& mesh, const PxVec4* p)
{
	if(!file) return;
	for(PxU32 i = 0; i < mesh.getNbVertices(); ++i)
		std::fprintf(file, "%u,%u,%.9g,%.9g,%.9g\n", frame, i,
			double(p[i].x), double(p[i].y), double(p[i].z));
}

// K + gravitational potential is a lower bound on mechanical energy here:
// the drape starts flat and at rest, with zero nonnegative elastic energy.
// Thus exceeding its initial value witnesses energy creation independently
// of the production membrane/contact formulas.
double kineticGravityEnergy(PxDeformableSurface& surface, PxU32 count)
{
	const PxVec4* p = surface.getPositionInvMassBufferH();
	const PxVec4* v = surface.getVelocityBufferH();
	double energy = 0.0;
	for(PxU32 i = 0; i < count; ++i)
		if(p[i].w > 0.0f)
			energy += (0.5 * double(v[i].getXYZ().magnitudeSquared()) + 9.81 * double(p[i].y)) / double(p[i].w);
	return energy;
}

struct Fixture
{
	JointSoftBodySnippet::PvdVisualization pvd;
	Snippets::TrackingErrorCallback errors;
	PxDefaultAllocator allocator;
	PxFoundation* foundation;
	PxPhysics* physics;
	PxDefaultCpuDispatcher* dispatcher;
	PxScene* scene;
	PxDeformableSurface* surface;
	PxTriangleMesh* mesh;
	PxDeformableSurfaceMaterial* material;
	PxMaterial* rigidMaterial;
	PxU32 width;
	PxU32 height;
	bool extensions;
	Fixture() : foundation(NULL), physics(NULL), dispatcher(NULL), scene(NULL),
		surface(NULL), mesh(NULL), material(NULL), rigidMaterial(NULL), width(kWidth),
		height(kHeight), extensions(false) {}
	~Fixture()
	{
		PX_RELEASE(surface);
		PX_RELEASE(mesh);
		PX_RELEASE(scene);
		PX_RELEASE(material);
		PX_RELEASE(rigidMaterial);
		PX_RELEASE(dispatcher);
		if(extensions) PxCloseExtensions();
		PX_RELEASE(physics);
		pvd.release();
		PX_RELEASE(foundation);
	}
	bool init(const Snippets::HeadlessOptions& options, const char* caseName, const SurfaceAvbdSnippet::DrapeGrid& grid)
	{
		foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
		if(!foundation || !pvd.init(*foundation, false)) return false;
		physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale(), true, pvd.get());
		if(!physics) return false;
		extensions = PxInitExtensions(*physics, pvd.get());
		dispatcher = PxDefaultCpuDispatcherCreate(options.dispatcherThreads);
		PxSceneDesc desc(physics->getTolerancesScale());
		desc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
		desc.solverType = options.solverType;
		desc.cpuDispatcher = dispatcher;
		desc.filterShader = PxDefaultSimulationFilterShader;
		scene = physics->createScene(desc);
		if(!scene) return false;
		pvd.configureScene(*scene);
		const bool groundCase = Snippets::equalsIgnoreCase(caseName, "ground");
		const bool boxCase = Snippets::equalsIgnoreCase(caseName, "static-box");
		const bool freeFallFlat = Snippets::equalsIgnoreCase(caseName, "free-fall-flat");
		const bool freeFlight = Snippets::equalsIgnoreCase(caseName, "free-flight");
		const bool bendingCase = Snippets::equalsIgnoreCase(caseName, "bending");
		if(groundCase || boxCase)
		{
			rigidMaterial = physics->createMaterial(0.7f, 0.6f, 0.0f);
			if(!rigidMaterial) return false;
			PxRigidStatic* ground = PxCreatePlane(*physics, PxPlane(0.0f, 1.0f, 0.0f, 0.0f),
				*rigidMaterial);
			if(!ground || !scene->addActor(*ground)) return false;
			if(boxCase)
			{
				PxRigidStatic* obstacle = PxCreateStatic(*physics,
					PxTransform(PxVec3(0.0f, 1.0f, 0.0f)),
					PxBoxGeometry(1.15f, 1.0f, 1.15f), *rigidMaterial);
				if(!obstacle || !scene->addActor(*obstacle)) return false;
			}
			width = grid.width; height = grid.height;
			surface = SurfaceAvbdSnippet::createDrapedSurface(*physics, *scene,
				mesh, material, 0.2f, grid);
		}
		else if(freeFallFlat || freeFlight)
		{
			width = grid.width; height = grid.height;
			surface = SurfaceAvbdSnippet::createDrapedSurface(*physics, *scene,
				mesh, material,freeFlight?2.e4f*.02f*.02f/12:0.0f, grid);
            if(freeFlight && surface)
            {
                scene->setGravity(PxVec3(0));surface->setLinearDamping(0);
                for(PxU32 i=0;i<width*height;++i)surface->getVelocityBufferH()[i]=PxVec4(.5f,-.25f,.2f,0);
                surface->markDirty(PxDeformableSurfaceDataFlag::eVELOCITY);
            }
		}
		else if(bendingCase)
		{
			width = height = 2;
			surface = SurfaceAvbdSnippet::createBendingProbe(*physics, *scene,
				mesh, material);
		}
		else
			surface = SurfaceAvbdSnippet::createHangingMembrane(*physics, *scene,
				mesh, material, kWidth, kHeight,Snippets::equalsIgnoreCase(caseName,"membrane")?0.0f:0.1f);
        // These are isolated material probes. Contact scenes keep their
        // authored collision flags and must pass the complete NP path.
        if(surface && !groundCase && !boxCase)
            surface->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION,true);
        // Minimal constitutive contract: damping has its own future operator
        // and is explicitly absent in this authored test, not silently ignored.
		return extensions && surface != NULL;
	}
};

bool supported(const char* name)
{
	return Snippets::equalsIgnoreCase(name, "stretch") ||
        Snippets::equalsIgnoreCase(name, "membrane") ||
		Snippets::equalsIgnoreCase(name, "free-fall-flat") ||
		Snippets::equalsIgnoreCase(name, "free-flight") ||
		Snippets::equalsIgnoreCase(name, "bending") ||
		Snippets::equalsIgnoreCase(name, "buffer-mutation") ||
		Snippets::equalsIgnoreCase(name, "ground") ||
		Snippets::equalsIgnoreCase(name, "static-box");
}

const char* unsupportedReason(const char* name)
{
	if(Snippets::equalsIgnoreCase(name, "dynamic-box")) return "cpu_surface_dynamic_rigid_contact_not_implemented";
	if(Snippets::equalsIgnoreCase(name, "world-pin")) return "cpu_surface_attachment_not_implemented";
	if(Snippets::equalsIgnoreCase(name, "sleep-wake")) return "cpu_surface_sleep_not_implemented";
	if(Snippets::equalsIgnoreCase(name, "self-collision")) return "surface_self_collision_not_implemented";
	if(Snippets::equalsIgnoreCase(name, "soft-soft") || Snippets::equalsIgnoreCase(name, "surface-volume")) return "soft_soft_not_implemented";
	if(Snippets::equalsIgnoreCase(name, "skinning")) return "surface_skinning_not_implemented";
	return NULL;
}

bool segmentIntersectsBoxCore(const PxVec3& p0, const PxVec3& p1)
{
	// This is a geometric non-penetration witness, not a contact activation
	// tolerance.  Erode the rendered obstacle by one millimetre so touching a
	// face is accepted while a cloth edge through the box core is not.
	const PxVec3 center(0.0f, 1.0f, 0.0f);
	const PxVec3 halfExtent(1.149f, 0.999f, 1.149f);
	const PxVec3 local0 = p0 - center;
	const PxVec3 direction = p1 - p0;
	PxReal t0 = 0.0f, t1 = 1.0f;
	for(PxU32 axis = 0; axis < 3; ++axis)
	{
		if(PxAbs(direction[axis]) <= 1.0e-9f)
		{
			if(PxAbs(local0[axis]) >= halfExtent[axis])
				return false;
			continue;
		}
		PxReal enter = (-halfExtent[axis] - local0[axis]) / direction[axis];
		PxReal exit = (halfExtent[axis] - local0[axis]) / direction[axis];
		if(enter > exit) PxSwap(enter, exit);
		t0 = PxMax(t0, enter);
		t1 = PxMin(t1, exit);
		if(t0 > t1) return false;
	}
	return t1 >= 0.0f && t0 <= 1.0f;
}

PxU32 countBoxCoreEdgeIntersections(const PxTriangleMesh& mesh,
	const PxVec4* positions, bool& finite)
{
	finite = positions != NULL;
	if(!finite) return 0;
	const PxU32 vertexCount = mesh.getNbVertices();
	for(PxU32 i = 0; i < vertexCount; ++i)
		if(!positions[i].isFinite()) { finite = false; return 0; }
	const bool indices16 = mesh.getTriangleMeshFlags().isSet(
		PxTriangleMeshFlag::e16_BIT_INDICES);
	const PxU16* triangles16 = static_cast<const PxU16*>(mesh.getTriangles());
	const PxU32* triangles32 = static_cast<const PxU32*>(mesh.getTriangles());
	PxU32 hits = 0;
	for(PxU32 triangle = 0; triangle < mesh.getNbTriangles(); ++triangle)
	{
		PxU32 ids[3];
		for(PxU32 corner = 0; corner < 3; ++corner)
			ids[corner] = indices16 ? PxU32(triangles16[3 * triangle + corner]) :
				triangles32[3 * triangle + corner];
		for(PxU32 edge = 0; edge < 3; ++edge)
		{
			const PxU32 a = ids[edge], b = ids[(edge + 1) % 3];
			if(a >= vertexCount || b >= vertexCount)
			{
				finite = false;
				return hits;
			}
			if(segmentIntersectsBoxCore(positions[a].getXYZ(), positions[b].getXYZ()))
				++hits;
		}
	}
	return hits;
}

PxU32 countBoxCoreTriangleIntersections(const PxTriangleMesh& mesh,
	const PxVec4* positions, bool& finite)
{
	finite = positions != NULL;
	if(!finite) return 0;
	const PxU32 vertexCount = mesh.getNbVertices();
	const bool indices16 = mesh.getTriangleMeshFlags().isSet(
		PxTriangleMeshFlag::e16_BIT_INDICES);
	const PxU16* triangles16 = static_cast<const PxU16*>(mesh.getTriangles());
	const PxU32* triangles32 = static_cast<const PxU32*>(mesh.getTriangles());
	PxU32 hits = 0;
	for(PxU32 triangle = 0; triangle < mesh.getNbTriangles(); ++triangle)
	{
		PxU32 ids[3];
		for(PxU32 corner = 0; corner < 3; ++corner)
		{
			ids[corner] = indices16 ? PxU32(triangles16[3 * triangle + corner]) :
				triangles32[3 * triangle + corner];
			if(ids[corner] >= vertexCount || !positions[ids[corner]].isFinite())
			{ finite = false; return hits; }
		}
		if(Gu::intersectTriangleBox_Unsafe(PxVec3(0.0f, 1.0f, 0.0f),
			PxVec3(1.149f, 0.999f, 1.149f), positions[ids[0]].getXYZ(),
			positions[ids[1]].getXYZ(), positions[ids[2]].getXYZ()))
			++hits;
	}
	return hits;
}

// Each replica owns its public objects. All four sheets share one scene step,
// dispatcher and ground plane. Sample quality only after fetchResults.
struct DrapeReplica
{
	PxDeformableSurface* surface = NULL;
	PxTriangleMesh* mesh = NULL;
	PxDeformableSurfaceMaterial* material = NULL;
	PxRigidStatic* obstacle = NULL;
	PxVec3 origin = PxVec3(0.0f);
	std::vector<PxVec4> local;
	double initialEnergy = 0.0, peakEnergy = 0.0;
	PxReal peakStretch = 0.0f, finalStretch = 0.0f, centerY = 0.0f, minY = PX_MAX_F32;
	PxU32 coreHits = 0;
	bool finite = true;

	~DrapeReplica()
	{
		PX_RELEASE(surface);
		PX_RELEASE(mesh);
		PX_RELEASE(material);
		PX_RELEASE(obstacle);
	}

	bool init(Fixture& fixture, const SurfaceAvbdSnippet::DrapeGrid& grid, const PxVec3& offset)
	{
		origin = offset;
		obstacle = PxCreateStatic(*fixture.physics, PxTransform(origin + PxVec3(0,1,0)),
			PxBoxGeometry(1.15f, 1.0f, 1.15f), *fixture.rigidMaterial);
		if(!obstacle || !fixture.scene->addActor(*obstacle)) return false;
		surface = SurfaceAvbdSnippet::createDrapedSurface(*fixture.physics, *fixture.scene,
			mesh, material, 0.2f, grid, origin);
		if(!surface) return false;
		local.resize(mesh->getNbVertices());
		initialEnergy = kineticGravityEnergy(*surface, mesh->getNbVertices());
		return true;
	}

	void sample(PxU32 center)
	{
		const PxVec4* p = surface->getPositionInvMassBufferH();
		const PxVec4* v = surface->getVelocityBufferH();
		for(PxU32 i = 0; i < mesh->getNbVertices(); ++i)
		{
			finite &= p[i].isFinite() && v[i].isFinite();
			local[i] = PxVec4(p[i].getXYZ() - origin, p[i].w);
			minY = PxMin(minY, p[i].y);
		}
		if(!finite) return;
		centerY = p[center].y;
		finalStretch = measureStrain(*mesh, surface->getRestPositionBufferH(), p).principal;
		peakStretch = PxMax(peakStretch, finalStretch);
		peakEnergy = PxMax(peakEnergy, kineticGravityEnergy(*surface, mesh->getNbVertices()));
		bool valid = false;
		coreHits += countBoxCoreTriangleIntersections(*mesh, local.data(), valid);
		finite &= valid;
	}

	bool passed() const
	{
		return finite && !coreHits && centerY > 1.90f && centerY < 2.10f && minY > -0.03f &&
			peakStretch < 1.6f && finalStretch < 1.3f && peakEnergy <= 1.05 * initialEnergy;
	}
};
}

bool snippetDeformableSurfaceAvbdHeadlessShouldRun(int argc, const char* const* argv)
{
	if(Snippets::isEnabledEnvironmentValue(std::getenv("PHYSX_SNIPPET_HEADLESS"))) return true;
	for(int i = 1; i < argc; ++i)
		if(argv[i] && (!std::strcmp(argv[i], "--headless") || !std::strncmp(argv[i], "--case=", 7)))
			return true;
	return false;
}

int snippetDeformableSurfaceAvbdHeadlessMain(int argc, const char* const* argv)
{
	Snippets::HeadlessOptions defaults;
	defaults.headless = true;
	defaults.caseName = "static-box";
	defaults.solverType = PxSolverType::eAVBD;
	defaults.frames = 360;
	defaults.dispatcherThreads = 4;
	Snippets::HeadlessOptions options;
	std::string error;
	if(!Snippets::parseCommonHeadlessOptions(argc, argv, defaults, options, error))
		return Snippets::eHEADLESS_CONFIG_ERROR;
	for(int i = 1; i < argc; ++i)
		if(!Snippets::isCommonHeadlessOption(argv[i])) return Snippets::eHEADLESS_CONFIG_ERROR;
	const char* sceneCase = options.caseName.c_str();
	const bool fourSheets = Snippets::equalsIgnoreCase(sceneCase, "static-box-four-sheets");
	SurfaceAvbdSnippet::DrapeGrid grid = SurfaceAvbdSnippet::kDrapeGrid;
	if(Snippets::equalsIgnoreCase(sceneCase, "static-box-25"))
	{
		grid = SurfaceAvbdSnippet::kReferenceDrapeGrid;
		sceneCase = "static-box";
	}
	else if(Snippets::equalsIgnoreCase(sceneCase, "static-box-50"))
	{
		grid = {50, 50, 0.1f};
		sceneCase = "static-box";
	}
	else if(Snippets::equalsIgnoreCase(sceneCase, "static-box-99"))
	{
		// Split each of the 49 x 49 cells twice along each axis at the same sheet size.
		grid = {99, 99, 0.05f};
		sceneCase = "static-box";
	}
	else if(fourSheets) sceneCase = "static-box";
	const char* gated = unsupportedReason(sceneCase);
	if(!supported(sceneCase))
	{
		if(!gated) return Snippets::eHEADLESS_CONFIG_ERROR;
		Snippets::printHeadlessConfig("SnippetDeformableSurfaceAVBD", options);
		std::printf("[AVBD_GATE] schema=1 snippet=SnippetDeformableSurfaceAVBD solver=%s case=%s status=ERROR reason=%s capability=UNSUPPORTED validation=GATED\n",
			Snippets::getSolverTypeName(options.solverType), options.caseName.c_str(), gated);
		return Snippets::eHEADLESS_UNSUPPORTED;
	}
	if(options.solverType != PxSolverType::eAVBD)
	{
		std::printf("[AVBD_GATE] schema=1 snippet=SnippetDeformableSurfaceAVBD solver=%s case=%s status=ERROR reason=cpu_surface_requires_avbd capability=UNSUPPORTED validation=GATED\n",
			Snippets::getSolverTypeName(options.solverType), options.caseName.c_str());
		return Snippets::eHEADLESS_UNSUPPORTED;
	}
	Snippets::applyExecutionEnvironment(options);
	Snippets::printHeadlessConfig("SnippetDeformableSurfaceAVBD", options);
	Fixture fixture;
	bool ok = fixture.init(options, sceneCase, grid);
	DrapeReplica replicas[3];
	for(PxU32 i = 0; ok && fourSheets && i < 3; ++i)
		ok = replicas[i].init(fixture, grid, PxVec3((i+1)%2 * 8.f, 0.f, (i+1)/2 * 8.f));
	if(ok && Snippets::equalsIgnoreCase(sceneCase, "static-box"))
		std::printf("[DRAPE_GRID] width=%u height=%u spacing=%.8g vertices=%u triangles=%u\n",
			grid.width, grid.height, double(grid.spacing), fixture.mesh->getNbVertices(), fixture.mesh->getNbTriangles());
	FILE* trajectory = NULL;
	if(ok)
		if(const char* path = std::getenv("PHYSX_AVBD_SURFACE_TRAJECTORY"))
		{
			trajectory = std::fopen(path, "wb");
			if(!trajectory) ok = false;
			else std::fprintf(trajectory, "frame,vertex,x,y,z\n");
		}
	const bool bendingCase = Snippets::equalsIgnoreCase(options.caseName.c_str(), "bending");
	const bool freeFlight = Snippets::equalsIgnoreCase(options.caseName.c_str(), "free-flight");
	const bool freeFallFlat = Snippets::equalsIgnoreCase(options.caseName.c_str(),
		"free-fall-flat");
	bool mutationReadback = !Snippets::equalsIgnoreCase(options.caseName.c_str(), "buffer-mutation");
	const PxU32 center = bendingCase ? 3 : SurfaceAvbdSnippet::vertexId(
		fixture.width / 2, fixture.height / 2, fixture.height);
	const PxU32 pin0 = bendingCase ? 0 : SurfaceAvbdSnippet::vertexId(
		0, fixture.height - 1, fixture.height);
	const PxU32 pin1 = bendingCase ? 1 : SurfaceAvbdSnippet::vertexId(
		fixture.width - 1, fixture.height - 1, fixture.height);
	PxVec3 initialCenter(0.0f), initialPin0(0.0f), initialPin1(0.0f);
	if(ok)
	{
		PxVec4* p = fixture.surface->getPositionInvMassBufferH();
		initialCenter = p[center].getXYZ(); initialPin0 = p[pin0].getXYZ(); initialPin1 = p[pin1].getXYZ();
		if(!mutationReadback)
		{
			p[center].z += 0.01f;
			fixture.surface->markDirty(PxDeformableSurfaceDataFlag::ePOSITION_INVMASS);
			mutationReadback = PxAbs(p[center].z - 0.01f) < 1.0e-6f;
		}
		writeTrajectory(trajectory, 0, *fixture.mesh, p);
	}
	PxU32 completed = 0, nonFinite = 0, maximumBoxCoreEdgeHits = 0,
		maximumBoxCoreTriangleHits = 0;
	PxU32 firstBoxCoreEdgeHitFrame = PX_MAX_U32, firstBoxCoreTriangleHitFrame = PX_MAX_U32;
	PxReal maximumPlanarityError = 0.0f;
    PxReal maximumBallisticError=0,maximumMomentumError=0;
    const bool drape=Snippets::equalsIgnoreCase(sceneCase,"static-box");
    PxVec3 settledCenter(0);
    double settledSpeedSquared=0;
    PxReal settledDrift=0;
    PxU32 settledFrames=0;
    double initialRimY=0,finalRimY=0,previousRimY=0,previousRimTravel=0;
    double earlyRimY=0,earlyDrop=0,peakBrakingTravel=0;
    PxU32 rimSamples=0;
	SurfaceStrain finalStrain, peakStrain;
	PxU32 peakStrainFrame = 0;
	const double initialKineticGravity = ok ? kineticGravityEnergy(*fixture.surface,
		fixture.mesh->getNbVertices()) : 0.0;
	double peakKineticGravity = initialKineticGravity;
	std::vector<double> stepMs;
	for(PxU32 frame = 0; ok && frame < options.frames; ++frame)
	{
		if(fixture.pvd.connected())
			fixture.pvd.wireMesh(*fixture.scene, *fixture.mesh, fixture.surface->getPositionInvMassBufferH(), 0xfff25729);
		const auto start = std::chrono::steady_clock::now();
		if(!fixture.scene->simulate(options.dt) || !fixture.scene->fetchResults(true) || fixture.errors.getFatalCount()) { ok = false; break; }
		stepMs.push_back(std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - start).count());
		++completed;
		if(fourSheets)
			for(DrapeReplica& replica : replicas) replica.sample(center);
		const PxVec4* p = fixture.surface->getPositionInvMassBufferH();
		writeTrajectory(trajectory, completed, *fixture.mesh, p);
		const PxVec4* v = fixture.surface->getVelocityBufferH();
        if(drape && completed>=38 && completed<=90)
        {
            // Post-impact motion of the authored free perimeter, in world metres.
            // The centre is already supported by the box during this interval.
            double y=0;
            for(PxU32 x=0;x<fixture.width;++x)
                y+=p[SurfaceAvbdSnippet::vertexId(x,0,fixture.height)].y+
                   p[SurfaceAvbdSnippet::vertexId(x,fixture.height-1,fixture.height)].y;
            for(PxU32 z=1;z+1<fixture.height;++z)
                y+=p[SurfaceAvbdSnippet::vertexId(0,z,fixture.height)].y+
                   p[SurfaceAvbdSnippet::vertexId(fixture.width-1,z,fixture.height)].y;
            y/=2*(fixture.width+fixture.height)-4;
            const double travel=y-previousRimY;
            if(completed>=40)peakBrakingTravel=PxMax(peakBrakingTravel,travel-previousRimTravel);
            previousRimTravel=travel;previousRimY=y;
            if(completed==50)earlyRimY=y;
            if(completed==66)earlyDrop=earlyRimY-y;
            if(completed>=60)
            {
                if(!rimSamples)initialRimY=y;
                finalRimY=y;
                ++rimSamples;
            }
        }
        if(drape && completed*options.dt>=20.f)
        {
            PxVec3 centerOfMass(0);PxReal mass=0;double speedSquared=0;
            for(PxU32 i=0;i<fixture.mesh->getNbVertices();++i)if(p[i].w>0)
            {
                const PxReal m=1.f/p[i].w;
                centerOfMass+=m*p[i].getXYZ();mass+=m;
                speedSquared+=double(m)*v[i].getXYZ().magnitudeSquared();
            }
            if(!(mass>0)){ok=false;break;}
            centerOfMass/=mass;
            if(settledFrames==0)settledCenter=centerOfMass;
            const auto delta=centerOfMass-settledCenter;
            settledDrift=PxMax(settledDrift,PxSqrt(delta.x*delta.x+delta.z*delta.z));
            settledSpeedSquared+=speedSquared/mass;++settledFrames;
        }
        if(freeFlight)
        {
            PxVec3 momentum(0);PxReal mass=0;
            for(PxU32 i=0;i<fixture.width*fixture.height;++i)
            {
                maximumBallisticError=PxMax(maximumBallisticError,(p[i].getXYZ()-
                    fixture.surface->getRestPositionBufferH()[i].getXYZ()-PxVec3(.5f,-.25f,.2f)*(completed*options.dt)).magnitude());
                if(p[i].w>0){mass+=1/p[i].w;momentum+=v[i].getXYZ()/p[i].w;}
            }
            const PxVec3 initialMomentum=PxVec3(.5f,-.25f,.2f)*mass;
            maximumMomentumError=PxMax(maximumMomentumError,(momentum-initialMomentum).magnitude()/initialMomentum.magnitude());
        }
		for(PxU32 i = 0; i < fixture.width * fixture.height; ++i)
			if(!p[i].isFinite() || !v[i].isFinite()) ++nonFinite;
		if(!nonFinite)
		{
			finalStrain = measureStrain(*fixture.mesh, fixture.surface->getRestPositionBufferH(), p);
			if(finalStrain.principal > peakStrain.principal) peakStrainFrame = completed;
			peakStrain.edge = PxMax(peakStrain.edge, finalStrain.edge);
			peakStrain.principal = PxMax(peakStrain.principal, finalStrain.principal);
			peakStrain.area = PxMax(peakStrain.area, finalStrain.area);
			peakKineticGravity = PxMax(peakKineticGravity,
				kineticGravityEnergy(*fixture.surface, fixture.mesh->getNbVertices()));
		}
		if(freeFallFlat || drape)
		{
			PxReal meanY = 0.0f;
			for(PxU32 i = 0; i < fixture.width * fixture.height; ++i)
				meanY += p[i].y;
			meanY /= PxReal(fixture.width * fixture.height);
			if(freeFallFlat || freeFlight || meanY > 2.6f)
				for(PxU32 i = 0; i < fixture.width * fixture.height; ++i)
					maximumPlanarityError = PxMax(maximumPlanarityError,
						PxAbs(p[i].y - meanY));
		}
		if(drape)
		{
			bool finite = false;
			const PxU32 hits = countBoxCoreEdgeIntersections(*fixture.mesh, p, finite);
			if(!finite) ++nonFinite;
			maximumBoxCoreEdgeHits = PxMax(maximumBoxCoreEdgeHits, hits);
			if(hits && firstBoxCoreEdgeHitFrame == PX_MAX_U32)
				firstBoxCoreEdgeHitFrame = frame;
			const PxU32 triangleHits = countBoxCoreTriangleIntersections(
				*fixture.mesh, p, finite);
			if(!finite) ++nonFinite;
			maximumBoxCoreTriangleHits = PxMax(maximumBoxCoreTriangleHits, triangleHits);
			if(triangleHits && firstBoxCoreTriangleHitFrame == PX_MAX_U32)
				firstBoxCoreTriangleHitFrame = frame;
		}
	}
	if(trajectory && std::fclose(trajectory) != 0) ok = false;
	double meanStepMs = 0.0;
	for(size_t i = 0; i < stepMs.size(); ++i) meanStepMs += stepMs[i];
	if(!stepMs.empty()) meanStepMs /= double(stepMs.size());
	if(!Snippets::writeHeadlessStepTimes(stepMs)) ok = false;
	std::sort(stepMs.begin(), stepMs.end());
    if(rimSamples)std::printf("[DRAPE_FOLD] samples=%u perimeterDrop=%.9g earlyDrop=%.9g peakBrakingTravel=%.9g gridSpacing=%.9g\n",
        rimSamples,initialRimY-finalRimY,earlyDrop,peakBrakingTravel,double(grid.spacing));
    if(freeFlight)std::printf("[NODAL_FLIGHT_QUALITY] maxBallisticError=%.9g maxMomentumError=%.9g\n",
        double(maximumBallisticError),double(maximumMomentumError));
	std::printf("[AVBD_SURFACE_METRICS] youngs=%.9g thickness=%.9g arealYoungs=%.9g "
		"finalEdgeStretch=%.9g peakEdgeStretch=%.9g finalPrincipalStretch=%.9g "
		"peakPrincipalStretch=%.9g peakStrainFrame=%u finalAreaRatio=%.9g peakAreaRatio=%.9g "
		"initialKineticGravity=%.9g peakKineticGravity=%.9g "
		"meanStepMs=%.9g p95StepMs=%.9g maxStepMs=%.9g\n",
		fixture.material ? double(fixture.material->getYoungsModulus()) : 0.0,
		fixture.material ? double(fixture.material->getThickness()) : 0.0,
		fixture.material ? double(fixture.material->getYoungsModulus()*fixture.material->getThickness()) : 0.0,
		double(finalStrain.edge), double(peakStrain.edge), double(finalStrain.principal),
		double(peakStrain.principal), peakStrainFrame, double(finalStrain.area), double(peakStrain.area),
		initialKineticGravity, peakKineticGravity,
		meanStepMs, stepMs.empty() ? 0.0 : stepMs[(stepMs.size()-1)*95/100],
		stepMs.empty() ? 0.0 : stepMs.back());
	PxReal centerY = PX_MAX_F32, pinError = 0.0f, maxStretch = 0.0f;
	PxReal minY = PX_MAX_F32, minBoxSdf = PX_MAX_F32, averageY = 0.0f;
	PxVec3 minBoxPosition(0.0f);
	if(ok)
	{
		const PxVec4* p = fixture.surface->getPositionInvMassBufferH();
		centerY = p[center].y;
		if(Snippets::equalsIgnoreCase(options.caseName.c_str(), "stretch") ||
            Snippets::equalsIgnoreCase(options.caseName.c_str(), "membrane") ||
			Snippets::equalsIgnoreCase(options.caseName.c_str(), "buffer-mutation"))
			pinError = PxMax((p[pin0].getXYZ() - initialPin0).magnitude(),
				(p[pin1].getXYZ() - initialPin1).magnitude());
		for(PxU32 i = 0; i < fixture.width * fixture.height; ++i)
		{
			const PxVec3 x = p[i].getXYZ();
			minY = PxMin(minY, x.y);
			averageY += x.y;
			const PxVec3 q(PxAbs(x.x) - 1.15f, PxAbs(x.y - 1.0f) - 1.0f,
				PxAbs(x.z) - 1.15f);
			const PxVec3 outside(PxMax(q.x, 0.0f), PxMax(q.y, 0.0f), PxMax(q.z, 0.0f));
			const PxReal sdf = (q.x <= 0.0f && q.y <= 0.0f && q.z <= 0.0f) ?
				PxMax(q.x, PxMax(q.y, q.z)) : outside.magnitude();
			if(sdf < minBoxSdf)
			{
				minBoxSdf = sdf;
				minBoxPosition = x;
			}
		}
		averageY /= PxReal(fixture.width * fixture.height);
		const PxVec4* rest = fixture.surface->getRestPositionBufferH();
		for(PxU32 x = 0; x < fixture.width; ++x)
			for(PxU32 y = 1; y < fixture.height; ++y)
		{
			const PxU32 a = SurfaceAvbdSnippet::vertexId(x, y - 1, fixture.height), b = a + 1;
			maxStretch = PxMax(maxStretch, (p[b].getXYZ() - p[a].getXYZ()).magnitude() /
				(rest[b].getXYZ() - rest[a].getXYZ()).magnitude());
		}
	}
	const PxReal requiredSag = PxMin(0.02f, 0.0001f * PxReal(options.frames));
	const bool groundCase = Snippets::equalsIgnoreCase(options.caseName.c_str(), "ground");
	const bool boxCase = drape;
	bool physical = centerY < initialCenter.y - requiredSag && centerY > -20.0f &&
		pinError < 1.0e-5f && maxStretch < 1.6f;
	if(groundCase)
		physical = minY > -0.03f && averageY < 0.5f && maxStretch < 1.6f;
	else if(boxCase)
		physical = minY > -0.03f && minBoxSdf > -0.04f && centerY > 1.90f &&
			centerY < initialCenter.y - requiredSag && averageY < 2.5f && maxStretch < 1.8f &&
			peakStrain.principal < 1.6f && finalStrain.principal < 1.3f &&
			peakKineticGravity <= 1.05 * initialKineticGravity &&
			maximumBoxCoreEdgeHits == 0 && maximumBoxCoreTriangleHits == 0 &&
			maximumPlanarityError < 0.02f;
	else if(freeFlight)
        physical=completed>=60 && maximumBallisticError<1.e-3f && maximumMomentumError<5.e-3f &&
            maximumPlanarityError<1.e-4f && maxStretch<1.001f;
	else if(freeFallFlat)
		physical = centerY < initialCenter.y - 0.5f &&
			maximumPlanarityError < 1.0e-4f && maxStretch < 1.001f;
	else if(bendingCase)
	{
		const PxVec4* p = fixture.surface->getPositionInvMassBufferH();
		pinError = PxMax(pinError,
			(p[2].getXYZ() - fixture.surface->getRestPositionBufferH()[2].getXYZ()).magnitude());
		physical = pinError < 1.0e-5f && initialCenter.y - centerY > 0.05f &&
			PxAbs(centerY) < initialCenter.y - 0.05f && maxStretch < 1.05f;
	}
    if(drape && fixture.surface && options.frames*options.dt>=40.f)
    {
        const double rms=settledFrames?std::sqrt(settledSpeedSquared/settledFrames):PX_MAX_F64;
        // At most a quarter cell of drift and two percent of sheet thickness per frame.
        const double rmsLimit=0.02*fixture.material->getThickness()/options.dt;
        physical &= settledFrames && settledDrift<0.05f && rms<=rmsLimit;
        std::printf("[DRAPE_SETTLE] samples=%u maxLateralDrift=%.8g rmsNodeSpeed=%.8g rmsSpeedLimit=%.8g\n",
            settledFrames,double(settledDrift),rms,rmsLimit);
    }
	if(fourSheets && ok)
		for(PxU32 i = 0; i < 3; ++i)
		{
			const DrapeReplica& r = replicas[i];
			physical &= r.passed();
			std::printf("[DRAPE_REPLICA] sheet=%u centerY=%.8g peakStretch=%.8g finalStretch=%.8g coreHits=%u energyRatio=%.8g status=%s\n",
				i+1, double(r.centerY), double(r.peakStretch), double(r.finalStretch), r.coreHits,
				r.peakEnergy / r.initialEnergy, r.passed() ? "PASS" : "FAIL");
		}
    const bool pass = ok && nonFinite == 0 && fixture.errors.getFatalCount() == 0 &&
		mutationReadback && physical;
	const char* reason = !ok ? "initialization_or_fetch" : nonFinite ? "non_finite" :
		fixture.errors.getFatalCount() ? "runtime_error" : !mutationReadback ? "host_buffer_not_mutable" :
		!physical ? "physical_contract" : "none";
	std::printf("[AVBD_GATE] schema=1 snippet=SnippetDeformableSurfaceAVBD solver=avbd case=%s execution=%s frames=%u completedFrames=%u status=%s reason=%s capability=SUPPORTED validation=GATED centerY=%.9g minY=%.9g averageY=%.9g minBoxSdf=%.9g minBoxX=%.9g minBoxY=%.9g minBoxZ=%.9g maxBoxCoreEdgeHits=%u maxBoxCoreTriangleHits=%u firstBoxCoreEdgeHitFrame=%u firstBoxCoreTriangleHitFrame=%u pinError=%.9g maxStretch=%.9g maxPlanarityError=%.9g mutationReadback=%u nonFinite=%u fatalErrors=%u\n",
		options.caseName.c_str(), Snippets::getExecutionName(options.execution), options.frames,
		completed, pass ? "PASS" : "FAIL", reason, double(centerY), double(minY),
		double(averageY), double(minBoxSdf), double(minBoxPosition.x),
		double(minBoxPosition.y), double(minBoxPosition.z), maximumBoxCoreEdgeHits,
		maximumBoxCoreTriangleHits, firstBoxCoreEdgeHitFrame,
		firstBoxCoreTriangleHitFrame, double(pinError),
		double(maxStretch), double(maximumPlanarityError), mutationReadback ? 1u : 0u,
		nonFinite, fixture.errors.getFatalCount());
	return pass ? Snippets::eHEADLESS_PASS : Snippets::eHEADLESS_GATE_FAILED;
}
