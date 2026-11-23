#include "SnippetDeformableMeshHeadless.h"
#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetHeadless.h"
#include <cstdio>
#include <cstring>
#include <string>

using namespace physx;

namespace
{
static const PxU32 GRID = 8;
static const PxReal STEP = 512.0f / PxReal(GRID - 1);
static const PxReal MINIMUM = -400.0f;
static const PxU32 SUBSTEPS = 8;
static int gMeshTag, gSphereTag;

struct Triangle
{
	PxU32 a, b, c;
};

struct Metrics
{
	PxU32 completed, contacts, points, firstContact, nonFinite, fetchFailures;
	PxReal minY, finalY, maxSpeed, maxPostContactVy;
	Metrics() : completed(0), contacts(0), points(0), firstContact(PX_MAX_U32),
		nonFinite(0), fetchFailures(0), minY(PX_MAX_F32), finalY(0.0f),
		maxSpeed(0.0f), maxPostContactVy(-PX_MAX_F32) {}
};

struct Callback : PxSimulationEventCallback
{
	Metrics* metrics;
	Callback() : metrics(NULL) {}
	void onConstraintBreak(PxConstraintInfo*, PxU32) {}
	void onWake(PxActor**, PxU32) {}
	void onSleep(PxActor**, PxU32) {}
	void onTrigger(PxTriggerPair*, PxU32) {}
	void onAdvance(const PxRigidBody*const*, const PxTransform*, const PxU32) {}
	void onContact(const PxContactPairHeader& header, const PxContactPair* pairs,
		PxU32 count)
	{
		if(!metrics)
			return;
		const bool hit = (header.actors[0]->userData == &gMeshTag &&
			header.actors[1]->userData == &gSphereTag) ||
			(header.actors[1]->userData == &gMeshTag &&
			header.actors[0]->userData == &gSphereTag);
		if(!hit)
			return;
		for(PxU32 i = 0; i < count; ++i)
		{
			if(metrics->firstContact == PX_MAX_U32)
				metrics->firstContact = metrics->completed;
			++metrics->contacts;
			metrics->points += pairs[i].contactCount;
		}
	}
};

static PxFilterFlags filter(PxFilterObjectAttributes, PxFilterData,
	PxFilterObjectAttributes, PxFilterData, PxPairFlags& pairFlags,
	const void*, PxU32)
{
	pairFlags = PxPairFlag::eCONTACT_DEFAULT | PxPairFlag::eNOTIFY_TOUCH_FOUND |
		PxPairFlag::eNOTIFY_TOUCH_PERSISTS | PxPairFlag::eNOTIFY_CONTACT_POINTS;
	return PxFilterFlags();
}

struct Fixture
{
	Snippets::TrackingErrorCallback errors;
	PxDefaultAllocator allocator;
	PxFoundation* foundation;
	PxPhysics* physics;
	PxDefaultCpuDispatcher* dispatcher;
	PxScene* scene;
	PxMaterial* material;
	PxTriangleMesh* mesh;
	PxRigidStatic* meshActor;
	PxRigidDynamic* sphere;
	Callback callback;
	Metrics metrics;
	bool extensions;
	PxReal time;

	Fixture() : foundation(NULL), physics(NULL), dispatcher(NULL), scene(NULL),
		material(NULL), mesh(NULL), meshActor(NULL), sphere(NULL), extensions(false),
		time(0.0f)
	{
		callback.metrics = &metrics;
	}
	~Fixture() { cleanup(); }

	void cleanup()
	{
		PX_RELEASE(scene);
		PX_RELEASE(mesh);
		PX_RELEASE(material);
		PX_RELEASE(dispatcher);
		if(extensions)
		{
			PxCloseExtensions();
			extensions = false;
		}
		PX_RELEASE(physics);
		PX_RELEASE(foundation);
	}

	static void update(PxVec3* vertices, PxReal amplitude)
	{
		for(PxU32 z = 0; z < GRID; ++z)
			for(PxU32 x = 0; x < GRID; ++x)
			{
				const PxReal cx = PxReal(x) / PxReal(GRID);
				const PxReal cz = PxReal(z) / PxReal(GRID);
				vertices[z * GRID + x] = PxVec3(MINIMUM + PxReal(x) * STEP,
					20.0f + PxSin(cx * PxTwoPi) * PxCos(cz * PxTwoPi) * amplitude,
					MINIMUM + PxReal(z) * STEP);
			}
	}

	bool init(const Snippets::HeadlessOptions& options)
	{
		foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
		if(!foundation)
			return false;
		physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation,
			PxTolerancesScale(), true, NULL);
		if(!physics)
			return false;
		extensions = PxInitExtensions(*physics, NULL);
		dispatcher = PxDefaultCpuDispatcherCreate(options.dispatcherThreads);
		PxSceneDesc desc(physics->getTolerancesScale());
		desc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
		desc.solverType = options.solverType;
		desc.cpuDispatcher = dispatcher;
		desc.filterShader = filter;
		desc.simulationEventCallback = &callback;
		scene = physics->createScene(desc);
		material = physics->createMaterial(0.5f, 0.5f, 0.0f);
		if(!extensions || !dispatcher || !scene || !material)
			return false;

		PxVec3 vertices[GRID * GRID];
		Triangle triangles[2 * (GRID - 1) * (GRID - 1)];
		update(vertices, 0.0f);
		for(PxU32 z = 0; z + 1 < GRID; ++z)
			for(PxU32 x = 0; x + 1 < GRID; ++x)
			{
				const PxU32 t = 2 * (z * (GRID - 1) + x);
				triangles[t] = {z * GRID + x + 1, z * GRID + x,
					(z + 1) * GRID + x + 1};
				triangles[t + 1] = {(z + 1) * GRID + x + 1,
					z * GRID + x, (z + 1) * GRID + x};
			}
		PxTriangleMeshDesc meshDesc;
		meshDesc.points.data = vertices;
		meshDesc.points.count = GRID * GRID;
		meshDesc.points.stride = sizeof(PxVec3);
		meshDesc.triangles.data = triangles;
		meshDesc.triangles.count = 2 * (GRID - 1) * (GRID - 1);
		meshDesc.triangles.stride = sizeof(Triangle);
		PxCookingParams cooking(physics->getTolerancesScale());
		cooking.midphaseDesc.setToDefault(PxMeshMidPhase::eBVH34);
		cooking.midphaseDesc.mBVH34Desc.quantized = false;
		cooking.meshPreprocessParams = PxMeshPreprocessingFlag::eDISABLE_CLEAN_MESH;
		mesh = PxCreateTriangleMesh(cooking, meshDesc,
			physics->getPhysicsInsertionCallback());
		if(!mesh)
			return false;
		meshActor = physics->createRigidStatic(PxTransform(PxVec3(0.0f, 2.0f, 0.0f)));
		PxShape* meshShape = physics->createShape(PxTriangleMeshGeometry(mesh), *material);
		if(!meshActor || !meshShape)
			return false;
		meshShape->setContactOffset(0.02f);
		meshShape->setRestOffset(-0.5f);
		meshActor->attachShape(*meshShape);
		meshShape->release();
		meshActor->userData = &gMeshTag;
		scene->addActor(*meshActor);

		sphere = PxCreateDynamic(*physics, PxTransform(PxVec3(0.0f, 55.0f, 0.0f)),
			PxSphereGeometry(3.0f), *material, 3.0f);
		if(!sphere)
			return false;
		sphere->setLinearVelocity(PxVec3(0.0f, -200.0f, 0.0f));
		sphere->userData = &gSphereTag;
		scene->addActor(*sphere);
		return true;
	}

	bool step(PxReal frameDt)
	{
		const PxReal dt = frameDt / PxReal(SUBSTEPS);
		for(PxU32 substep = 0; substep < SUBSTEPS; ++substep)
		{
			PxVec3* vertices = mesh->getVerticesForModification();
			time += 0.01f / PxReal(SUBSTEPS);
			update(vertices, PxSin(time) * 20.0f);
			mesh->refitBVH();
			scene->resetFiltering(*meshActor);
			if(!scene->simulate(dt) || !scene->fetchResults(true))
			{
				++metrics.fetchFailures;
				return false;
			}
		}
		++metrics.completed;
		const PxTransform pose = sphere->getGlobalPose();
		const PxVec3 velocity = sphere->getLinearVelocity();
		if(!pose.isFinite() || !velocity.isFinite() ||
			!sphere->getAngularVelocity().isFinite())
			++metrics.nonFinite;
		metrics.finalY = pose.p.y;
		metrics.minY = PxMin(metrics.minY, pose.p.y);
		metrics.maxSpeed = PxMax(metrics.maxSpeed, velocity.magnitude());
		if(metrics.firstContact != PX_MAX_U32)
			metrics.maxPostContactVy = PxMax(metrics.maxPostContactVy, velocity.y);
		return true;
	}
};
}

bool snippetDeformableMeshHeadlessShouldRun(int argc, const char*const* argv)
{
	if(Snippets::isEnabledEnvironmentValue(std::getenv("PHYSX_SNIPPET_HEADLESS")))
		return true;
	for(int i = 1; i < argc; ++i)
		if(argv[i] && std::strcmp(argv[i], "--headless") == 0)
			return true;
	return false;
}

int snippetDeformableMeshHeadlessMain(int argc, const char*const* argv)
{
	Snippets::HeadlessOptions defaults;
	defaults.headless = true;
	defaults.caseName = "sphere-shot";
	defaults.frames = 180;
	Snippets::HeadlessOptions options;
	std::string error;
	if(!Snippets::parseCommonHeadlessOptions(argc, argv, defaults, options, error) ||
		!Snippets::equalsIgnoreCase(options.caseName.c_str(), "sphere-shot"))
	{
		std::fprintf(stderr, "[AVBD_GATE_CONFIG_ERROR] snippet=SnippetDeformableMesh reason=%s\n",
			error.empty() ? "unsupported_case" : error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	for(int i = 1; i < argc; ++i)
		if(!Snippets::isCommonHeadlessOption(argv[i]))
			return Snippets::eHEADLESS_CONFIG_ERROR;
	if(!Snippets::applyExecutionEnvironment(options))
		return Snippets::eHEADLESS_CONFIG_ERROR;
	Snippets::printHeadlessConfig("SnippetDeformableMesh", options);
	Fixture fixture;
	bool ok = fixture.init(options);
	for(PxU32 frame = 0; ok && frame < options.frames; ++frame)
		ok = fixture.step(options.dt);
	const Metrics& m = fixture.metrics;
	const bool response = m.contacts > 0 && m.points > 0 &&
		m.maxPostContactVy > -20.0f;
	const bool retained = m.minY > 10.0f && m.finalY > 10.0f;
	const bool bounded = m.maxSpeed < 250.0f;
	const bool pass = ok && m.completed == options.frames && !m.nonFinite &&
		!fixture.errors.getFatalCount() && response && retained && bounded;
	const char* reason = !ok ? "initialization_or_fetch" :
		(m.nonFinite || fixture.errors.getFatalCount()) ? "runtime_error" :
		!response ? "missing_surface_response" : !retained ? "sphere_tunneled" :
		!bounded ? "unbounded_energy" : "none";
	std::printf("[AVBD_GATE] schema=1 snippet=SnippetDeformableMesh solver=%s case=%s execution=%s frames=%u completedFrames=%u status=%s reason=%s capability=SUPPORTED validation=GATED substeps=%u contacts=%u contactPoints=%u firstContactFrame=%u minSphereY=%.9g finalSphereY=%.9g maxSphereSpeed=%.9g maxPostContactVy=%.9g nonFinite=%u fetchFailures=%u fatalErrors=%u\n",
		Snippets::getSolverTypeName(options.solverType), options.caseName.c_str(),
		Snippets::getExecutionName(options.execution), options.frames, m.completed,
		pass ? "PASS" : "FAIL", reason, SUBSTEPS, m.contacts, m.points,
		m.firstContact, double(m.minY), double(m.finalY), double(m.maxSpeed),
		double(m.maxPostContactVy), m.nonFinite, m.fetchFailures,
		fixture.errors.getFatalCount());
	return pass ? Snippets::eHEADLESS_PASS : Snippets::eHEADLESS_GATE_FAILED;
}
