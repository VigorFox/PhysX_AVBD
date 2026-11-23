// PhysX AVBD chainmail visual sample. Validation lives in SnippetChainmailHeadless.cpp.
#include <ctype.h>
#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"
#include "SnippetChainmailHeadless.h"

using namespace physx;

static PxDefaultAllocator gAllocator;
static PxDefaultErrorCallback gErrorCallback;
static PxFoundation* gFoundation = NULL;
static PxPhysics* gPhysics = NULL;
static PxDefaultCpuDispatcher* gDispatcher = NULL;
static PxScene* gScene = NULL;
static PxMaterial* gMaterial = NULL;
static PxPvd* gPvd = NULL;
static bool gExtensionsInitialized = false;

static const PxU32 gWidth = 30, gHeight = 30;
static const PxReal gSpacing = 0.65f, gMeshY = 35.0f;
static PxRigidDynamic* gNodes[gWidth * gHeight] = {};

static void addStrut(PxRigidDynamic& body, const PxVec3& direction)
{
	const PxReal radius = 0.06f;
	PxShape* shape = gPhysics->createShape(PxCapsuleGeometry(radius, gSpacing * 0.5f - radius), *gMaterial, true);
	const PxQuat rotation = direction.x > 0.5f ? PxQuat(PxIdentity) : PxQuat(-PxHalfPi, PxVec3(0, 1, 0));
	shape->setLocalPose(PxTransform(direction * (gSpacing * 0.5f), rotation));
	PxFilterData data; data.word0 = 1;
	shape->setSimulationFilterData(data);
	body.attachShape(*shape);
	shape->release();
}

static PxFilterFlags chainmailFilter(PxFilterObjectAttributes, PxFilterData a,
	PxFilterObjectAttributes, PxFilterData b, PxPairFlags& pairFlags, const void*, PxU32)
{
	if(a.word0 == 1 && b.word0 == 1)
		return PxFilterFlag::eSUPPRESS;
	pairFlags = PxPairFlag::eCONTACT_DEFAULT;
	return PxFilterFlags();
}

void initPhysics(bool)
{
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	gPvd = PxCreatePvd(*gFoundation);
	PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
	gPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);
	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	gExtensionsInitialized = PxInitExtensions(*gPhysics, gPvd);
	PxSceneDesc desc(gPhysics->getTolerancesScale());
	desc.gravity = PxVec3(0, -9.81f, 0);
	desc.solverType = PxSolverType::eAVBD;
	gDispatcher = PxDefaultCpuDispatcherCreate(4);
	desc.cpuDispatcher = gDispatcher;
	desc.filterShader = chainmailFilter;
	gScene = gPhysics->createScene(desc);
	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.2f);
	PxRigidStatic* ground = PxCreatePlane(*gPhysics, PxPlane(0, 1, 0, 0), *gMaterial);
	gScene->addActor(*ground);

	const PxReal x0 = -gSpacing * PxReal(gWidth - 1) * 0.5f;
	const PxReal z0 = -gSpacing * PxReal(gHeight - 1) * 0.5f;
	for(PxU32 r = 0; r < gHeight; ++r)
		for(PxU32 c = 0; c < gWidth; ++c)
		{
			const PxU32 i = r * gWidth + c;
			const PxVec3 p(x0 + c * gSpacing, gMeshY, z0 + r * gSpacing);
			gNodes[i] = PxCreateDynamic(*gPhysics, PxTransform(p), PxSphereGeometry(0.12f), *gMaterial, 3.0f);
			gNodes[i]->setAngularDamping(0.5f);
			if(c + 1 < gWidth) addStrut(*gNodes[i], PxVec3(1, 0, 0));
			if(r + 1 < gHeight) addStrut(*gNodes[i], PxVec3(0, 0, 1));
			PxShape* shapes[3]; const PxU32 n = gNodes[i]->getShapes(shapes, 3);
			PxFilterData data; data.word0 = 1;
			for(PxU32 s = 0; s < n; ++s) shapes[s]->setSimulationFilterData(data);
			const bool corner = (r == 0 || r + 1 == gHeight) && (c == 0 || c + 1 == gWidth);
			if(corner)
			{
				gNodes[i]->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
			}
			gScene->addActor(*gNodes[i]);
			if(corner) gNodes[i]->setKinematicTarget(PxTransform(p));
		}
	const PxVec3 hx(gSpacing * 0.5f, 0, 0), hz(0, 0, gSpacing * 0.5f);
	for(PxU32 r = 0; r < gHeight; ++r)
		for(PxU32 c = 0; c < gWidth; ++c)
		{
			const PxU32 i = r * gWidth + c;
			if(c + 1 < gWidth) PxSphericalJointCreate(*gPhysics, gNodes[i], PxTransform(hx), gNodes[i + 1], PxTransform(-hx));
			if(r + 1 < gHeight) PxSphericalJointCreate(*gPhysics, gNodes[i], PxTransform(hz), gNodes[i + gWidth], PxTransform(-hz));
		}
	PxRigidDynamic* ball = PxCreateDynamic(*gPhysics, PxTransform(PxVec3(0, 70, 0)), PxSphereGeometry(2.0f), *gMaterial, 300.0f);
	ball->setAngularDamping(0.5f);
	gScene->addActor(*ball);
}

void stepPhysics(bool) { gScene->simulate(1.0f / 60.0f); gScene->fetchResults(true); }

void cleanupPhysics(bool)
{
	PX_RELEASE(gScene); PX_RELEASE(gMaterial); PX_RELEASE(gDispatcher);
	if(gExtensionsInitialized) { PxCloseExtensions(); gExtensionsInitialized = false; }
	PX_RELEASE(gPhysics);
	if(gPvd) { PxPvdTransport* transport = gPvd->getTransport(); PX_RELEASE(gPvd); PX_RELEASE(transport); }
	PX_RELEASE(gFoundation);
}

void keyPress(unsigned char key, const PxTransform& camera)
{
	if(key == ' ')
	{
		PxRigidDynamic* ball = PxCreateDynamic(*gPhysics, camera, PxSphereGeometry(1), *gMaterial, 100.0f);
		ball->setLinearVelocity(camera.rotate(PxVec3(0, 0, -1)) * 50.0f);
		gScene->addActor(*ball);
	}
}

int snippetMain(int argc, const char* const* argv)
{
	if(snippetChainmailHeadlessShouldRun(argc, argv))
		return snippetChainmailHeadlessMain(argc, argv);
#ifdef RENDER_SNIPPET
	extern void renderLoop(); renderLoop();
#else
	initPhysics(false); for(PxU32 i = 0; i < 600; ++i) stepPhysics(false); cleanupPhysics(false);
#endif
	return 0;
}
