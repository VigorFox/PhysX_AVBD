#include "SnippetJointDriveHeadless.h"

#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetHeadless.h"

#include <cstdio>
#include <cstring>
#include <string>

using namespace physx;

namespace
{
enum DriveKind
{
	eLINEAR_X,
	eTWIST,
	eSWING1,
	eSWING2,
	eSLERP
};

struct CaseDesc
{
	const char* name;
	DriveKind drive;
	PxQuat frameA;
	PxVec3 localAxis;
	PxReal expectedSign;
	bool position;
	bool forceLimited;
};

static const CaseDesc gCases[] =
{
	{ "velocity-x", eLINEAR_X, PxQuat(PxIdentity), PxVec3(1.0f, 0.0f, 0.0f), 1.0f, false, false },
	{ "velocity-x-frame", eLINEAR_X, PxQuat(-PxPi / 4.0f, PxVec3(0.0f, 0.0f, 1.0f)), PxVec3(1.0f, 0.0f, 0.0f), 1.0f, false, false },
	{ "velocity-twist", eTWIST, PxQuat(PxIdentity), PxVec3(1.0f, 0.0f, 0.0f), -1.0f, false, false },
	{ "velocity-swing1", eSWING1, PxQuat(PxIdentity), PxVec3(0.0f, 1.0f, 0.0f), -1.0f, false, false },
	{ "velocity-swing2", eSWING2, PxQuat(PxIdentity), PxVec3(0.0f, 0.0f, 1.0f), -1.0f, false, false },
	{ "velocity-slerp", eSLERP, PxQuat(PxIdentity), PxVec3(0.0f, 1.0f, 0.0f), 1.0f, false, false },
	{ "velocity-slerp-frame", eSLERP, PxQuat(-PxPi / 4.0f, PxVec3(1.0f, 0.0f, 0.0f)), PxVec3(0.0f, 1.0f, 0.0f), 1.0f, false, false },
	{ "force-limit-x", eLINEAR_X, PxQuat(PxIdentity), PxVec3(1.0f, 0.0f, 0.0f), 1.0f, false, true },
	{ "position-x", eLINEAR_X, PxQuat(PxIdentity), PxVec3(1.0f, 0.0f, 0.0f), 1.0f, true, false },
	{ "position-x-frame", eLINEAR_X, PxQuat(-PxPi / 4.0f, PxVec3(0.0f, 0.0f, 1.0f)), PxVec3(1.0f, 0.0f, 0.0f), 1.0f, true, false },
	{ "position-twist", eTWIST, PxQuat(PxIdentity), PxVec3(1.0f, 0.0f, 0.0f), 1.0f, true, false },
	{ "position-swing1", eSWING1, PxQuat(PxIdentity), PxVec3(0.0f, 1.0f, 0.0f), 1.0f, true, false },
	{ "position-swing2", eSWING2, PxQuat(PxIdentity), PxVec3(0.0f, 0.0f, 1.0f), 1.0f, true, false },
	{ "position-slerp", eSLERP, PxQuat(PxIdentity), PxVec3(0.0f, 1.0f, 0.0f), 1.0f, true, false }
};

static const CaseDesc* findCase(const char* name)
{
	for(PxU32 i = 0; i < PX_ARRAY_SIZE(gCases); ++i)
		if(Snippets::equalsIgnoreCase(name, gCases[i].name))
			return &gCases[i];
	return NULL;
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
	PxD6Joint* joint;
	PxRigidDynamic* driven;
	bool extensions;

	Fixture() : foundation(NULL), physics(NULL), dispatcher(NULL), scene(NULL),
		material(NULL), joint(NULL), driven(NULL), extensions(false) {}

	~Fixture() { cleanup(); }

	void cleanup()
	{
		PX_RELEASE(joint);
		PX_RELEASE(scene);
		driven = NULL;
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

	bool init(const Snippets::HeadlessOptions& options, const CaseDesc& desc)
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
		PxSceneDesc sceneDesc(physics->getTolerancesScale());
		sceneDesc.gravity = PxVec3(0.0f);
		sceneDesc.solverType = options.solverType;
		sceneDesc.cpuDispatcher = dispatcher;
		sceneDesc.filterShader = PxDefaultSimulationFilterShader;
		sceneDesc.flags |= PxSceneFlag::eDISABLE_SLEEPING;
		scene = physics->createScene(sceneDesc);
		material = physics->createMaterial(0.0f, 0.0f, 0.0f);
		if(!extensions || !dispatcher || !scene || !material)
			return false;

		const PxBoxGeometry geometry(0.5f, 0.5f, 0.5f);
		PxRigidStatic* fixed = PxCreateStatic(*physics,
			PxTransform(PxVec3(0.0f, 2.0f, 0.0f)), geometry, *material);
		driven = PxCreateDynamic(*physics,
			PxTransform(PxVec3(2.0f, 2.0f, 0.0f)), geometry, *material, 1.0f);
		if(!fixed || !driven)
		{
			PX_RELEASE(fixed);
			PX_RELEASE(driven);
			return false;
		}
		driven->setLinearDamping(0.0f);
		driven->setAngularDamping(0.0f);
		driven->setMaxAngularVelocity(100.0f);
		scene->addActor(*fixed);
		scene->addActor(*driven);

		joint = PxD6JointCreate(*physics, fixed, PxTransform(PxVec3(0.0f), desc.frameA),
			driven, PxTransform(PxIdentity));
		if(!joint)
			return false;
		for(PxU32 axis = 0; axis < PxD6Axis::eCOUNT; ++axis)
			joint->setMotion(PxD6Axis::Enum(axis), PxD6Motion::eFREE);

		PxD6JointDriveFlags driveFlags =
			(desc.position || desc.forceLimited) ? PxD6JointDriveFlags() :
			PxD6JointDriveFlags(PxD6JointDriveFlag::eACCELERATION);
		const PxD6JointDrive drive(desc.position ? 100.0f : 0.0f,
			desc.position ? 20.0f : 1000.0f,
			desc.forceLimited ? 1.0f : PX_MAX_F32,
			driveFlags);
		if(desc.forceLimited)
			joint->setConstraintFlag(
				PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES, true);
		switch(desc.drive)
		{
		case eLINEAR_X:
			joint->setDrive(PxD6Drive::eX, drive);
			joint->setDriveVelocity(PxVec3(1.0f, 0.0f, 0.0f), PxVec3(0.0f));
			break;
		case eTWIST:
			joint->setAngularDriveConfig(PxD6AngularDriveConfig::eSWING_TWIST);
			joint->setDrive(PxD6Drive::eTWIST, drive);
			joint->setDriveVelocity(PxVec3(0.0f), PxVec3(1.0f, 0.0f, 0.0f));
			break;
		case eSWING1:
			joint->setAngularDriveConfig(PxD6AngularDriveConfig::eSWING_TWIST);
			joint->setDrive(PxD6Drive::eSWING1, drive);
			joint->setDriveVelocity(PxVec3(0.0f), PxVec3(0.0f, 1.0f, 0.0f));
			break;
		case eSWING2:
			joint->setAngularDriveConfig(PxD6AngularDriveConfig::eSWING_TWIST);
			joint->setDrive(PxD6Drive::eSWING2, drive);
			joint->setDriveVelocity(PxVec3(0.0f), PxVec3(0.0f, 0.0f, 1.0f));
			break;
		case eSLERP:
			joint->setAngularDriveConfig(PxD6AngularDriveConfig::eSLERP);
			joint->setDrive(PxD6Drive::eSLERP, drive);
			joint->setDriveVelocity(PxVec3(0.0f), PxVec3(0.0f, 1.0f, 0.0f));
			break;
		}
		if(desc.position)
		{
			const PxVec3 linearTarget = desc.drive == eLINEAR_X ?
				PxVec3(0.5f, 0.0f, 0.0f) : PxVec3(0.0f);
			const PxQuat angularTarget = desc.drive == eLINEAR_X ?
				PxQuat(PxIdentity) : PxQuat(0.5f, desc.localAxis);
			joint->setDrivePosition(PxTransform(linearTarget, angularTarget));
			joint->setDriveVelocity(PxVec3(0.0f), PxVec3(0.0f));
		}
		return true;
	}
};
}

bool snippetJointDriveHeadlessShouldRun(int argc, const char*const* argv)
{
	if(Snippets::isEnabledEnvironmentValue(std::getenv("PHYSX_SNIPPET_HEADLESS")))
		return true;
	for(int i = 1; i < argc; ++i)
		if(argv[i] && std::strcmp(argv[i], "--headless") == 0)
			return true;
	return false;
}

int snippetJointDriveHeadlessMain(int argc, const char*const* argv)
{
	Snippets::HeadlessOptions defaults;
	defaults.headless = true;
	defaults.caseName = "velocity-x";
	defaults.frames = 180;
	Snippets::HeadlessOptions options;
	std::string error;
	if(!Snippets::parseCommonHeadlessOptions(argc, argv, defaults, options, error))
	{
		std::fprintf(stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetJointDrive reason=%s\n",
			error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	for(int i = 1; i < argc; ++i)
		if(!Snippets::isCommonHeadlessOption(argv[i]))
			return Snippets::eHEADLESS_CONFIG_ERROR;
	const CaseDesc* desc = findCase(options.caseName.c_str());
	if(!desc || !Snippets::applyExecutionEnvironment(options))
	{
		std::fprintf(stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetJointDrive reason=%s\n",
			desc ? "execution_environment" : "unsupported_case");
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	Snippets::printHeadlessConfig("SnippetJointDrive", options);

	Fixture fixture;
	bool ok = fixture.init(options, *desc);
	PxU32 completed = 0;
	PxU32 nonFinite = 0;
	PxReal minSignedSpeed = PX_MAX_F32;
	PxReal maxOffAxisSpeed = 0.0f;
	PxReal maxLateTargetError = 0.0f;
	PxReal firstSignedSpeed = 0.0f;
	const PxVec3 worldAxis = desc->frameA.rotate(desc->localAxis).getNormalized();
	for(PxU32 frame = 0; ok && frame < options.frames; ++frame)
	{
		ok = fixture.scene->simulate(options.dt) && fixture.scene->fetchResults(true);
		if(!ok)
			break;
		++completed;
		const PxVec3 velocity = desc->drive == eLINEAR_X
			? fixture.driven->getLinearVelocity()
			: fixture.driven->getAngularVelocity();
		if(!velocity.isFinite() || !fixture.driven->getGlobalPose().isFinite())
			++nonFinite;
		if(frame == 0)
			firstSignedSpeed = velocity.dot(worldAxis) * desc->expectedSign;
		if(frame + 30 >= options.frames)
		{
			const PxReal signedSpeed = velocity.dot(worldAxis) * desc->expectedSign;
			minSignedSpeed = PxMin(minSignedSpeed, signedSpeed);
			maxOffAxisSpeed = PxMax(maxOffAxisSpeed,
				(velocity - worldAxis * velocity.dot(worldAxis)).magnitude());
			if(desc->position)
			{
				const PxTransform relative = fixture.joint->getRelativeTransform();
				if(desc->drive == eLINEAR_X)
					maxLateTargetError = PxMax(maxLateTargetError,
						PxAbs(relative.p.x - 0.5f));
				else
				{
					const PxQuat target(0.5f, desc->localAxis);
					maxLateTargetError = PxMax(maxLateTargetError,
						2.0f * PxAcos(PxClamp(PxAbs(relative.q.dot(target)), 0.0f, 1.0f)));
				}
			}
		}
	}
	const PxVec3 finalVelocity = ok ? (desc->drive == eLINEAR_X
		? fixture.driven->getLinearVelocity() : fixture.driven->getAngularVelocity())
		: PxVec3(0.0f);
	const PxReal finalSignedSpeed = finalVelocity.dot(worldAxis) * desc->expectedSign;
	// The authored drive target is 1 for every backend. Finite drives own
	// their relative damping; no additional solver-wide velocity decay applies.
	const PxReal expectedSpeed = 1.0f;
	const bool targetHeld = desc->position ? maxLateTargetError < 0.03f :
		(desc->forceLimited ?
			(finalSignedSpeed > 0.95f && finalSignedSpeed < 1.01f &&
			 minSignedSpeed > 0.90f) :
			(PxAbs(finalSignedSpeed - expectedSpeed) < 0.015f &&
			 PxAbs(minSignedSpeed - expectedSpeed) < 0.02f));
	const bool axisHeld = desc->position ? finalVelocity.magnitude() < 0.1f :
		maxOffAxisSpeed < 0.05f;
	const bool forceLimitHeld = !desc->forceLimited ||
		(firstSignedSpeed > 0.01f && firstSignedSpeed < 0.025f);
	const bool pass = ok && completed == options.frames && nonFinite == 0 &&
		fixture.errors.getFatalCount() == 0 && targetHeld && axisHeld && forceLimitHeld;
	const char* reason = !ok ? "initialization_or_fetch" :
		(nonFinite || fixture.errors.getFatalCount()) ? "runtime_error" :
		!targetHeld ? "velocity_target_error" : !axisHeld ? "drive_axis_error" :
		!forceLimitHeld ? "force_limit_error" : "none";
	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetJointDrive solver=%s case=%s "
		"execution=%s frames=%u completedFrames=%u status=%s reason=%s "
		"capability=SUPPORTED validation=GATED finalSignedSpeed=%.9g "
		"firstSignedSpeed=%.9g minLateSignedSpeed=%.9g maxOffAxisSpeed=%.9g "
		"maxLateTargetError=%.9g "
		"nonFinite=%u fatalErrors=%u\n",
		Snippets::getSolverTypeName(options.solverType), options.caseName.c_str(),
		Snippets::getExecutionName(options.execution), options.frames, completed,
		pass ? "PASS" : "FAIL", reason, double(finalSignedSpeed),
		double(firstSignedSpeed), double(minSignedSpeed), double(maxOffAxisSpeed),
		double(maxLateTargetError), nonFinite,
		fixture.errors.getFatalCount());
	return pass ? Snippets::eHEADLESS_PASS : Snippets::eHEADLESS_GATE_FAILED;
}
