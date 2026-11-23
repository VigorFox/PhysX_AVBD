#include "SnippetGearJointHeadless.h"
#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetHeadless.h"
#include <cstdio>
#include <cstring>
#include <string>

using namespace physx;

namespace
{
static const PxReal RATIO = 2.5f;

struct Fixture
{
	Snippets::TrackingErrorCallback errors;
	PxDefaultAllocator allocator;
	PxFoundation* foundation;
	PxPhysics* physics;
	PxDefaultCpuDispatcher* dispatcher;
	PxScene* scene;
	PxMaterial* material;
	PxRigidDynamic* gears[2];
	PxRevoluteJoint* hinges[2];
	PxGearJoint* gearJoint;
	bool extensions;

	Fixture() : foundation(NULL), physics(NULL), dispatcher(NULL), scene(NULL),
		material(NULL), gearJoint(NULL), extensions(false)
	{
		gears[0] = gears[1] = NULL;
		hinges[0] = hinges[1] = NULL;
	}
	~Fixture() { cleanup(); }

	void cleanup()
	{
		PX_RELEASE(gearJoint);
		PX_RELEASE(hinges[0]);
		PX_RELEASE(hinges[1]);
		PX_RELEASE(scene);
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

	PxRigidDynamic* createGear(const PxVec3& position, PxReal radius,
		PxReal halfDepth, PxU32 shapeCount)
	{
		PxRigidDynamic* actor = physics->createRigidDynamic(PxTransform(position));
		if(!actor)
			return NULL;
		const PxBoxGeometry geometry(radius, radius, halfDepth);
		for(PxU32 i = 0; i < shapeCount; ++i)
		{
			const PxReal angle = PxHalfPi * PxReal(i) / PxReal(shapeCount);
			PxShape* shape = physics->createShape(geometry, *material, true);
			if(!shape)
				continue;
			shape->setLocalPose(PxTransform(PxVec3(0.0f),
				PxQuat(angle, PxVec3(0.0f, 0.0f, 1.0f))));
			actor->attachShape(*shape);
			shape->release();
		}
		PxRigidBodyExt::updateMassAndInertia(*actor, 1.0f);
		return actor;
	}

	bool init(const Snippets::HeadlessOptions& options, bool driven)
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
		desc.filterShader = PxDefaultSimulationFilterShader;
		scene = physics->createScene(desc);
		material = physics->createMaterial(0.5f, 0.5f, 0.6f);
		if(!extensions || !dispatcher || !scene || !material)
			return false;

		const PxReal radius0 = 5.0f;
		const PxReal radius1 = 2.0f;
		const PxReal extra = ((radius0 * PxSqrt(2.0f) - radius0) +
			(radius1 * PxSqrt(2.0f) - radius1)) * 0.75f;
		const PxVec3 positions[2] = {
			PxVec3(0.0f, 10.0f, 0.0f),
			PxVec3(radius0 + radius1 + extra, 10.0f, 0.0f) };
		gears[0] = createGear(positions[0], radius0, 0.5f, 5);
		gears[1] = createGear(positions[1], radius1, 0.25f, 2);
		if(!gears[0] || !gears[1])
			return false;
		scene->addActor(*gears[0]);
		scene->addActor(*gears[1]);

		const PxQuat xToZ = PxShortestRotation(PxVec3(1.0f, 0.0f, 0.0f),
			PxVec3(0.0f, 0.0f, 1.0f));
		hinges[0] = PxRevoluteJointCreate(*physics, NULL,
			PxTransform(positions[0], xToZ), gears[0],
			PxTransform(PxVec3(0.0f), xToZ));
		hinges[1] = PxRevoluteJointCreate(*physics, NULL,
			PxTransform(positions[1], xToZ), gears[1],
			PxTransform(PxVec3(0.0f), xToZ));
		gearJoint = PxGearJointCreate(*physics, gears[0],
			PxTransform(PxVec3(0.0f), xToZ), gears[1],
			PxTransform(PxVec3(0.0f), xToZ));
		if(!hinges[0] || !hinges[1] || !gearJoint)
			return false;
		gearJoint->setHinges(hinges[0], hinges[1]);
		gearJoint->setGearRatio(RATIO);
		if(driven)
		{
			hinges[0]->setDriveVelocity(0.5f);
			hinges[0]->setRevoluteJointFlag(
				PxRevoluteJointFlag::eDRIVE_ENABLED, true);
		}
		else
			gears[0]->setAngularVelocity(PxVec3(0.0f, 0.0f, 0.5f));
		return true;
	}
};

static bool isSupported(const char* value)
{
	return Snippets::equalsIgnoreCase(value, "drive") ||
		Snippets::equalsIgnoreCase(value, "passive");
}
}

bool snippetGearJointHeadlessShouldRun(int argc, const char*const* argv)
{
	if(Snippets::isEnabledEnvironmentValue(std::getenv("PHYSX_SNIPPET_HEADLESS")))
		return true;
	for(int i = 1; i < argc; ++i)
		if(argv[i] && std::strcmp(argv[i], "--headless") == 0)
			return true;
	return false;
}

int snippetGearJointHeadlessMain(int argc, const char*const* argv)
{
	Snippets::HeadlessOptions defaults;
	defaults.headless = true;
	defaults.caseName = "drive";
	defaults.frames = 240;
	Snippets::HeadlessOptions options;
	std::string error;
	if(!Snippets::parseCommonHeadlessOptions(argc, argv, defaults, options, error) ||
		!isSupported(options.caseName.c_str()))
	{
		std::fprintf(stderr, "[AVBD_GATE_CONFIG_ERROR] snippet=SnippetGearJoint reason=%s\n",
			error.empty() ? "unsupported_case" : error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	for(int i = 1; i < argc; ++i)
		if(!Snippets::isCommonHeadlessOption(argv[i]))
			return Snippets::eHEADLESS_CONFIG_ERROR;
	if(!Snippets::applyExecutionEnvironment(options))
		return Snippets::eHEADLESS_CONFIG_ERROR;
	Snippets::printHeadlessConfig("SnippetGearJoint", options);

	const bool driven = Snippets::equalsIgnoreCase(options.caseName.c_str(), "drive");
	Fixture fixture;
	bool ok = fixture.init(options, driven);
	PxReal maxResidual = 0.0f;
	PxReal travel[2] = { 0.0f, 0.0f };
	PxReal maxCenterDrift = 0.0f;
	PxU32 completed = 0;
	PxU32 nonFinite = 0;
	const PxReal secondX = 7.0f +
		((5.0f * PxSqrt(2.0f) - 5.0f) +
		 (2.0f * PxSqrt(2.0f) - 2.0f)) * 0.75f;
	const PxVec3 initial[2] = { PxVec3(0.0f, 10.0f, 0.0f),
		PxVec3(secondX, 10.0f, 0.0f) };
	for(PxU32 frame = 0; ok && frame < options.frames; ++frame)
	{
		ok = fixture.scene->simulate(options.dt) &&
			fixture.scene->fetchResults(true);
		if(!ok)
			break;
		++completed;
		const PxReal w0 = fixture.gears[0]->getAngularVelocity().z;
		const PxReal w1 = fixture.gears[1]->getAngularVelocity().z;
		travel[0] += w0 * options.dt;
		travel[1] += w1 * options.dt;
		if(frame >= 60)
			maxResidual = PxMax(maxResidual, PxAbs(RATIO * w0 + w1));
		for(PxU32 i = 0; i < 2; ++i)
		{
			const PxTransform pose = fixture.gears[i]->getGlobalPose();
			if(!pose.isFinite() || !fixture.gears[i]->getLinearVelocity().isFinite() ||
				!fixture.gears[i]->getAngularVelocity().isFinite())
				++nonFinite;
			maxCenterDrift = PxMax(maxCenterDrift, (pose.p - initial[i]).magnitude());
		}
	}
	const PxReal w0 = ok ? fixture.gears[0]->getAngularVelocity().z : 0.0f;
	const PxReal w1 = ok ? fixture.gears[1]->getAngularVelocity().z : 0.0f;
	const bool moved = PxAbs(travel[0]) > 0.1f && PxAbs(travel[1]) > 0.1f;
	const bool coupled = maxResidual < 0.25f && PxAbs(RATIO * w0 + w1) < 0.25f;
	const bool bounded = maxCenterDrift < 0.05f && PxAbs(w0) < 5.0f && PxAbs(w1) < 12.5f;
	const bool pass = ok && completed == options.frames && nonFinite == 0 &&
		fixture.errors.getFatalCount() == 0 && moved && coupled && bounded;
	const char* reason = !ok ? "initialization_or_fetch" :
		(nonFinite || fixture.errors.getFatalCount()) ? "runtime_error" :
		!moved ? "drive_or_coupling_inactive" : !coupled ? "gear_ratio_error" :
		!bounded ? "unbounded_motion" : "none";
	std::printf("[AVBD_GATE] schema=1 snippet=SnippetGearJoint solver=%s case=%s execution=%s frames=%u completedFrames=%u status=%s reason=%s capability=SUPPORTED validation=GATED finalW0=%.9g finalW1=%.9g travel0=%.9g travel1=%.9g maxRatioResidual=%.9g maxCenterDrift=%.9g nonFinite=%u fatalErrors=%u\n",
		Snippets::getSolverTypeName(options.solverType), options.caseName.c_str(),
		Snippets::getExecutionName(options.execution), options.frames, completed,
		pass ? "PASS" : "FAIL", reason, double(w0), double(w1),
		double(travel[0]), double(travel[1]), double(maxResidual),
		double(maxCenterDrift), nonFinite, fixture.errors.getFatalCount());
	return pass ? Snippets::eHEADLESS_PASS : Snippets::eHEADLESS_GATE_FAILED;
}
