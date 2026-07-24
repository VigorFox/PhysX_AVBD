// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.  

// ****************************************************************************
// This snippet shows how to use custom geometries in PhysX.
// ****************************************************************************

#include <ctype.h>
#include "PxPhysicsAPI.h"

#ifdef RENDER_SNIPPET

#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"
#include "../snippetrender/SnippetRender.h"

#include "VoxelMap.h"
#include <cstdio>
#include <string>

using namespace physx;

static PxDefaultAllocator gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation* gFoundation = NULL;
static PxPhysics* gPhysics = NULL;
static PxDefaultCpuDispatcher* gDispatcher = NULL;
static PxScene* gScene = NULL;
static PxMaterial* gMaterial = NULL;
static PxPvd* gPvd = NULL;
static PxRigidStatic* gActor = NULL;
static PxRigidDynamic* gHeadlessBody = NULL;
static bool gExtensionsInitialized = false;

static const int gVoxelMapDim = 20;
static const float gVoxelMapSize = 80.0f;
static VoxelMap* gVoxelMap;
static Snippets::HeadlessOptions gHeadlessOptions;

struct CustomGeometryMetrics
{
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 callbackCount;
	PxU32 pairCount;
	PxU32 reportedPoints;
	PxU32 nonzeroImpulseCount;
	PxU32 identityErrors;
	PxU32 nonFinite;
	PxReal impulseSum;
	PxReal maxImpulse;
	PxReal surfaceY;
	PxReal minBodyY;
	PxVec3 initialPosition;
	PxVec3 finalPosition;
	PxVec3 finalVelocity;
	PxU32 cleanupComplete;

	CustomGeometryMetrics()
	: completedFrames(0), fetchFailures(0), callbackCount(0), pairCount(0),
	  reportedPoints(0), nonzeroImpulseCount(0), identityErrors(0),
	  nonFinite(0), impulseSum(0.0f), maxImpulse(0.0f), surfaceY(0.0f),
	  minBodyY(PX_MAX_F32), initialPosition(0.0f), finalPosition(0.0f),
	  finalVelocity(0.0f), cleanupComplete(0)
	{
	}
};

static CustomGeometryMetrics gMetrics;

static bool isHeadlessCase(const char* name)
{
	return Snippets::equalsIgnoreCase(
		gHeadlessOptions.caseName.c_str(), name);
}

static bool parseHeadlessOptions(
	int argc, const char* const* argv, std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 180;
	defaults.caseName = "drop";
	defaults.solverType = PxSolverType::eAVBD;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
		return false;
	for(int i = 1; i < argc; ++i)
	{
		if(!Snippets::isCommonHeadlessOption(argv[i]))
		{
			error = std::string("unknown option: ") +
				(argv[i] ? argv[i] : "<null>");
			return false;
		}
	}
	if(!isHeadlessCase("drop") && !isHeadlessCase("impact"))
	{
		error = "unsupported --case";
		return false;
	}
	if(gHeadlessOptions.frames < 120)
	{
		error = "--frames must be at least 120";
		return false;
	}
	return true;
}

static PxFilterFlags contactReportFilterShader(
	PxFilterObjectAttributes attributes0, PxFilterData filterData0,
	PxFilterObjectAttributes attributes1, PxFilterData filterData1,
	PxPairFlags& pairFlags, const void* constantBlock,
	PxU32 constantBlockSize)
{
	PX_UNUSED(attributes0);
	PX_UNUSED(attributes1);
	PX_UNUSED(filterData0);
	PX_UNUSED(filterData1);
	PX_UNUSED(constantBlock);
	PX_UNUSED(constantBlockSize);
	pairFlags = PxPairFlag::eCONTACT_DEFAULT |
		PxPairFlag::eNOTIFY_TOUCH_FOUND |
		PxPairFlag::eNOTIFY_TOUCH_PERSISTS |
		PxPairFlag::eNOTIFY_CONTACT_POINTS;
	return PxFilterFlag::eDEFAULT;
}

class ContactReportCallback : public PxSimulationEventCallback
{
	void onConstraintBreak(PxConstraintInfo*, PxU32) {}
	void onWake(PxActor**, PxU32) {}
	void onSleep(PxActor**, PxU32) {}
	void onTrigger(PxTriggerPair*, PxU32) {}
	void onAdvance(
		const PxRigidBody*const*, const PxTransform*, const PxU32) {}
	void onContact(const PxContactPairHeader& pairHeader,
		const PxContactPair* pairs, PxU32 nbPairs)
	{
		if(!gHeadlessOptions.headless)
			return;
		++gMetrics.callbackCount;
		gMetrics.pairCount += nbPairs;
		const bool identityValid =
			(pairHeader.actors[0] == gHeadlessBody &&
			 pairHeader.actors[1] == gActor) ||
			(pairHeader.actors[1] == gHeadlessBody &&
			 pairHeader.actors[0] == gActor);
		if(!identityValid)
			++gMetrics.identityErrors;
		PxArray<PxContactPairPoint> points;
		for(PxU32 i = 0; i < nbPairs; ++i)
		{
			const PxU32 count = pairs[i].contactCount;
			if(!count)
				continue;
			points.resize(count);
			const PxU32 extracted =
				pairs[i].extractContacts(points.begin(), count);
			for(PxU32 j = 0; j < extracted; ++j)
			{
				const PxContactPairPoint& point = points[j];
				const PxReal impulse = point.impulse.magnitude();
				++gMetrics.reportedPoints;
				gMetrics.impulseSum += impulse;
				gMetrics.maxImpulse = PxMax(gMetrics.maxImpulse, impulse);
				if(impulse > 1e-5f)
					++gMetrics.nonzeroImpulseCount;
				if(!point.position.isFinite() || !point.normal.isFinite() ||
					!point.impulse.isFinite() ||
					!PxIsFinite(point.separation))
					++gMetrics.nonFinite;
			}
		}
	}
};

static ContactReportCallback gContactReportCallback;

static PxArray<PxVec3> gVertices;
static PxArray<PxU32> gIndices;
static PxU32 gVertexCount;
static PxU32 gIndexCount;
static PxGeometryHolder gVoxelGeometryHolder;
static const PxU32 gVertexOrder[12] = {
	0, 2, 1, 2, 3, 1,
	0, 1, 2, 2, 1, 3
};

PX_INLINE void cookVoxelFace(bool reverseWinding) {
	for (int i = 0; i < 6; ++i) {
		gIndices[gIndexCount + i] = gVertexCount + gVertexOrder[i + (reverseWinding ? 6 : 0)];
	}
	gVertexCount += 4;
	gIndexCount += 6;
}

void cookVoxelMesh() {

	int faceCount = 0;
	gVertexCount = 0;
	gIndexCount = 0;

	float vx[2] = {gVoxelMap->voxelSize().x * -0.5f, gVoxelMap->voxelSize().x * 0.5f};
	float vy[2] = {gVoxelMap->voxelSize().y * -0.5f, gVoxelMap->voxelSize().y * 0.5f};
	float vz[2] = {gVoxelMap->voxelSize().z * -0.5f, gVoxelMap->voxelSize().z * 0.5f};

	for (int x = 0; x < gVoxelMap->dimX(); ++x)
		for (int y = 0; y < gVoxelMap->dimY(); ++y)
			for (int z = 0; z < gVoxelMap->dimZ(); ++z)
				if (gVoxelMap->voxel(x, y, z))
				{
					if (!gVoxelMap->voxel(x+1, y, z)) {faceCount++;}
					if (!gVoxelMap->voxel(x-1, y, z)) {faceCount++;}
					if (!gVoxelMap->voxel(x, y+1, z)) {faceCount++;}
					if (!gVoxelMap->voxel(x, y-1, z)) {faceCount++;}
					if (!gVoxelMap->voxel(x, y, z+1)) {faceCount++;}
					if (!gVoxelMap->voxel(x, y, z-1)) {faceCount++;}
				}

	gVertices.resize(faceCount*4);
	gIndices.resize(faceCount*6);

	for (int x = 0; x < gVoxelMap->dimX(); ++x)
	{
		for (int y = 0; y < gVoxelMap->dimY(); ++y)
		{
			for (int z = 0; z < gVoxelMap->dimZ(); ++z)
			{
				PxVec3 voxelPos = gVoxelMap->voxelPos(x, y, z);

				if (gVoxelMap->voxel(x, y, z))
				{
					if (!gVoxelMap->voxel(x+1, y, z)) {
						gVertices[gVertexCount + 0] = voxelPos + PxVec3(vx[1], vy[0], vz[0]);
						gVertices[gVertexCount + 1] = voxelPos + PxVec3(vx[1], vy[0], vz[1]);
						gVertices[gVertexCount + 2] = voxelPos + PxVec3(vx[1], vy[1], vz[0]);
						gVertices[gVertexCount + 3] = voxelPos + PxVec3(vx[1], vy[1], vz[1]);
						cookVoxelFace(false);
					}
					if (!gVoxelMap->voxel(x-1, y, z)) {
						gVertices[gVertexCount + 0] = voxelPos + PxVec3(vx[0], vy[0], vz[0]);
						gVertices[gVertexCount + 1] = voxelPos + PxVec3(vx[0], vy[0], vz[1]);
						gVertices[gVertexCount + 2] = voxelPos + PxVec3(vx[0], vy[1], vz[0]);
						gVertices[gVertexCount + 3] = voxelPos + PxVec3(vx[0], vy[1], vz[1]);
						cookVoxelFace(true);
					}
					if (!gVoxelMap->voxel(x, y+1, z)) {
						gVertices[gVertexCount + 0] = voxelPos + PxVec3(vx[0], vy[1], vz[0]);
						gVertices[gVertexCount + 1] = voxelPos + PxVec3(vx[0], vy[1], vz[1]);
						gVertices[gVertexCount + 2] = voxelPos + PxVec3(vx[1], vy[1], vz[0]);
						gVertices[gVertexCount + 3] = voxelPos + PxVec3(vx[1], vy[1], vz[1]);
						cookVoxelFace(true);
					}
					if (!gVoxelMap->voxel(x, y-1, z)) {
						gVertices[gVertexCount + 0] = voxelPos + PxVec3(vx[0], vy[0], vz[0]);
						gVertices[gVertexCount + 1] = voxelPos + PxVec3(vx[0], vy[0], vz[1]);
						gVertices[gVertexCount + 2] = voxelPos + PxVec3(vx[1], vy[0], vz[0]);
						gVertices[gVertexCount + 3] = voxelPos + PxVec3(vx[1], vy[0], vz[1]);
						cookVoxelFace(false);
					}
					if (!gVoxelMap->voxel(x, y, z+1)) {
						gVertices[gVertexCount + 0] = voxelPos + PxVec3(vx[0], vy[0], vz[1]);
						gVertices[gVertexCount + 1] = voxelPos + PxVec3(vx[0], vy[1], vz[1]);
						gVertices[gVertexCount + 2] = voxelPos + PxVec3(vx[1], vy[0], vz[1]);
						gVertices[gVertexCount + 3] = voxelPos + PxVec3(vx[1], vy[1], vz[1]);
						cookVoxelFace(false);
					}
					if (!gVoxelMap->voxel(x, y, z-1)) {
						gVertices[gVertexCount + 0] = voxelPos + PxVec3(vx[0], vy[0], vz[0]);
						gVertices[gVertexCount + 1] = voxelPos + PxVec3(vx[0], vy[1], vz[0]);
						gVertices[gVertexCount + 2] = voxelPos + PxVec3(vx[1], vy[0], vz[0]);
						gVertices[gVertexCount + 3] = voxelPos + PxVec3(vx[1], vy[1], vz[0]);
						cookVoxelFace(true);
					}
				}
			}
		}
	}

	const PxTolerancesScale scale;
	PxCookingParams params(scale);
	params.midphaseDesc.setToDefault(PxMeshMidPhase::eBVH34);
	params.meshPreprocessParams |= PxMeshPreprocessingFlag::eDISABLE_ACTIVE_EDGES_PRECOMPUTE;
	params.meshPreprocessParams |= PxMeshPreprocessingFlag::eDISABLE_CLEAN_MESH;
	PxTriangleMeshDesc triangleMeshDesc;
	triangleMeshDesc.points.count = gVertexCount;
	triangleMeshDesc.points.data = gVertices.begin();
	triangleMeshDesc.points.stride = sizeof(PxVec3);
	triangleMeshDesc.triangles.count = gIndexCount / 3;
	triangleMeshDesc.triangles.data = gIndices.begin();
	triangleMeshDesc.triangles.stride = 3 * sizeof(PxU32);
	PxTriangleMesh* gTriangleMesh = PxCreateTriangleMesh(params, triangleMeshDesc);
	gVoxelGeometryHolder.storeAny( PxTriangleMeshGeometry(gTriangleMesh) );
}

static PxRigidDynamic* createDynamic(const PxTransform& t, const PxGeometry& geometry, const PxVec3& velocity = PxVec3(0), PxReal density = 1.0f)
{
	PxRigidDynamic* dynamic = PxCreateDynamic(*gPhysics, t, geometry, *gMaterial, density);
	dynamic->setLinearVelocity(velocity);
	gScene->addActor(*dynamic);
	return dynamic;
}

static void createStack(const PxTransform& t, PxU32 size, PxReal halfExtent)
{
	PxShape* shape = gPhysics->createShape(PxBoxGeometry(halfExtent, halfExtent, halfExtent), *gMaterial);
	for (PxU32 i = 0; i < size; i++)
	{
		for (PxU32 j = 0; j < size - i; j++)
		{
			PxTransform localTm(PxVec3(PxReal(j * 2) - PxReal(size - i), PxReal(i * 2 + 1), 0) * halfExtent);
			PxRigidDynamic* body = gPhysics->createRigidDynamic(t.transform(localTm));
			body->attachShape(*shape);
			PxRigidBodyExt::updateMassAndInertia(*body, 10.0f);
			gScene->addActor(*body);
		}
	}
	shape->release();
}

void initVoxelMap()
{
	gVoxelMap = PX_NEW(VoxelMap);
	gVoxelMap->setDimensions(gVoxelMapDim, gVoxelMapDim, gVoxelMapDim);
	gVoxelMap->setVoxelSize(gVoxelMapSize / gVoxelMapDim, gVoxelMapSize / gVoxelMapDim, gVoxelMapSize / gVoxelMapDim);
	gVoxelMap->setWaveVoxels();
}

void initPhysics(bool /*interactive*/)
{
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);

	if(!gHeadlessOptions.headless)
	{
		gPvd = PxCreatePvd(*gFoundation);
		PxPvdTransport* transport =
			PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
		gPvd->connect(*transport, PxPvdInstrumentationFlag::eALL);
	}

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	gExtensionsInitialized = PxInitExtensions(*gPhysics, gPvd);

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	gDispatcher = PxDefaultCpuDispatcherCreate(
		gHeadlessOptions.headless ?
		gHeadlessOptions.dispatcherThreads : 2);
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader = gHeadlessOptions.headless ?
		contactReportFilterShader : PxDefaultSimulationFilterShader;
	sceneDesc.simulationEventCallback = gHeadlessOptions.headless ?
		&gContactReportCallback : NULL;
	sceneDesc.solverType = gHeadlessOptions.solverType;

	gScene = gPhysics->createScene(sceneDesc);
	if(!gHeadlessOptions.headless)
	{
		gScene->setVisualizationParameter(
			PxVisualizationParameter::eCOLLISION_SHAPES, 1.0f);
		gScene->setVisualizationParameter(
			PxVisualizationParameter::eSCALE, 1.0f);
		PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
		if (pvdClient)
		{
			pvdClient->setScenePvdFlag(
				PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
			pvdClient->setScenePvdFlag(
				PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
			pvdClient->setScenePvdFlag(
				PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
		}
	}

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);

	// Create voxel map actor
	initVoxelMap();
	PxRigidStatic* voxelMapActor = gPhysics->createRigidStatic(PxTransform(PxVec3(0, gVoxelMapSize * 0.5f, 0)));
	PxShape* shape = PxRigidActorExt::createExclusiveShape(*voxelMapActor, PxCustomGeometry(*gVoxelMap), *gMaterial);
	shape->setFlag(PxShapeFlag::eVISUALIZATION, true);
	gScene->addActor(*voxelMapActor);
	gActor = voxelMapActor;
	gActor->setActorFlag(PxActorFlag::eVISUALIZATION, true);

	if(gHeadlessOptions.headless)
	{
		const int x = gVoxelMapDim / 2;
		const int z = gVoxelMapDim / 2;
		int highestY = -1;
		for(int y = 0; y < gVoxelMapDim; ++y)
			if(gVoxelMap->voxel(x, y, z))
				highestY = y;
		if(highestY >= 0)
		{
			const PxVec3 topVoxel = gVoxelMap->voxelPos(x, highestY, z);
			gMetrics.surfaceY = gVoxelMapSize * 0.5f + topVoxel.y +
				gVoxelMap->voxelSizeY() * 0.5f;
			const PxVec3 position(
				topVoxel.x, gMetrics.surfaceY +
					(isHeadlessCase("impact") ? 10.0f : 8.0f),
				topVoxel.z);
			const PxVec3 velocity(
				0.0f, isHeadlessCase("impact") ? -40.0f : 0.0f, 0.0f);
			gHeadlessBody = createDynamic(
				PxTransform(position), PxSphereGeometry(2.0f),
				velocity, 1.0f);
			gHeadlessBody->setLinearDamping(0.0f);
			gHeadlessBody->setAngularDamping(0.0f);
			gHeadlessBody->setSleepThreshold(0.0f);
			gMetrics.initialPosition = position;
			gMetrics.finalPosition = position;
			gMetrics.finalVelocity = velocity;
		}
	}
	else
	{
		// Ground plane
		PxRigidStatic* planeActor = gPhysics->createRigidStatic(
			PxTransform(PxQuat(PX_PIDIV2, PxVec3(0, 0, 1))));
		PxRigidActorExt::createExclusiveShape(
			*planeActor, PxPlaneGeometry(), *gMaterial);
		gScene->addActor(*planeActor);
		createStack(PxTransform(PxVec3(0, 22, 0)), 10, 2.0f);
		cookVoxelMesh();
	}
}

void debugRender()
{
	PxTransform pose = gActor->getGlobalPose();
	Snippets::renderGeoms(1, &gVoxelGeometryHolder, &pose, false, PxVec3(0.5f));
}

void stepPhysics(bool /*interactive*/)
{
	const PxReal dt = gHeadlessOptions.headless ?
		gHeadlessOptions.dt : 1.0f / 60.0f;
	gScene->simulate(dt);
	const bool fetched = gScene->fetchResults(true);
	if(gHeadlessOptions.headless)
	{
		if(!fetched)
			++gMetrics.fetchFailures;
		if(gHeadlessBody)
		{
			const PxTransform pose = gHeadlessBody->getGlobalPose();
			const PxVec3 velocity = gHeadlessBody->getLinearVelocity();
			gMetrics.finalPosition = pose.p;
			gMetrics.finalVelocity = velocity;
			gMetrics.minBodyY = PxMin(gMetrics.minBodyY, pose.p.y);
			if(!pose.isFinite() || !velocity.isFinite())
				++gMetrics.nonFinite;
		}
		++gMetrics.completedFrames;
	}
}

void cleanupPhysics(bool /*interactive*/)
{
	PX_RELEASE(gScene);
	gHeadlessBody = NULL;
	gActor = NULL;
	PX_DELETE(gVoxelMap);
	PX_RELEASE(gMaterial);
	PX_RELEASE(gDispatcher);
	if(gExtensionsInitialized)
	{
		PxCloseExtensions();
		gExtensionsInitialized = false;
	}
	PX_RELEASE(gPhysics);
	gVertices.reset();
	gIndices.reset();
	if (gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		PX_RELEASE(gPvd);
		PX_RELEASE(transport);
	}
	PX_RELEASE(gFoundation);
	gMetrics.cleanupComplete =
		!gScene && !gMaterial && !gDispatcher && !gPhysics && !gPvd &&
		!gFoundation && !gVoxelMap && !gHeadlessBody && !gActor &&
		!gExtensionsInitialized ? 1u : 0u;

	printf("SnippetCustomGeometry done.\n");
}

void keyPress(unsigned char key, const PxTransform& camera)
{
	switch (toupper(key))
	{
	case ' ':	createDynamic(camera, PxSphereGeometry(3.0f), camera.rotate(PxVec3(0, 0, -1)) * 200, 3.0f);	break;
	}
}

static int runHeadless()
{
	std::setvbuf(stdout, NULL, _IONBF, 0);
	Snippets::printHeadlessConfig("SnippetCustomGeometry", gHeadlessOptions);
	initPhysics(false);
	if(gVoxelMap)
		gVoxelMap->resetContactStats();
	const bool initialized =
		gFoundation && gPhysics && gExtensionsInitialized && gDispatcher &&
		gScene && gMaterial && gVoxelMap && gActor && gHeadlessBody &&
		gMetrics.surfaceY > 0.0f;
	if(initialized)
	{
		for(PxU32 frame = 0; frame < gHeadlessOptions.frames; ++frame)
			stepPhysics(false);
	}

	const PxU32 generateCalls =
		gVoxelMap ? gVoxelMap->generateCalls() : 0;
	const PxU32 generatedContacts =
		gVoxelMap ? gVoxelMap->generatedContacts() : 0;
	const PxU32 nonFiniteGenerated =
		gVoxelMap ? gVoxelMap->nonFiniteGeneratedContacts() : 0;
	const char* reason = "none";
	bool passed = true;
	if(!initialized)
	{
		passed = false;
		reason = "initialization_failed";
	}
	else if(gMetrics.completedFrames != gHeadlessOptions.frames ||
		gMetrics.fetchFailures != 0)
	{
		passed = false;
		reason = "incomplete_simulation";
	}
	else if(gMetrics.nonFinite != 0 || nonFiniteGenerated != 0 ||
		gErrorCallback.getFatalCount() != 0)
	{
		passed = false;
		reason = "runtime_error";
	}
	else if(generateCalls == 0 || generatedContacts == 0)
	{
		passed = false;
		reason = "missing_generated_contacts";
	}
	else if(gMetrics.callbackCount == 0 || gMetrics.pairCount == 0 ||
		gMetrics.reportedPoints == 0)
	{
		passed = false;
		reason = "missing_contact_report";
	}
	else if(gMetrics.identityErrors != 0)
	{
		passed = false;
		reason = "contact_identity_mismatch";
	}
	else if(gMetrics.nonzeroImpulseCount == 0 ||
		gMetrics.maxImpulse <= 1e-5f)
	{
		passed = false;
		reason = "generated_contact_not_consumed";
	}
	else if(gMetrics.minBodyY < gMetrics.surfaceY ||
		gMetrics.finalPosition.y < gMetrics.surfaceY + 1.5f)
	{
		passed = false;
		reason = "body_fell_through_voxel_geometry";
	}

	cleanupPhysics(false);
	if(!gMetrics.cleanupComplete && passed)
	{
		passed = false;
		reason = "cleanup_incomplete";
	}
	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetCustomGeometry solver=%s "
		"case=%s execution=%s frames=%u completedFrames=%u status=%s "
		"reason=%s validation=GATED generateCalls=%u generatedContacts=%u "
		"callbackCount=%u pairCount=%u reportedPoints=%u "
		"nonzeroImpulseCount=%u identityErrors=%u impulseSum=%.9g "
		"maxImpulse=%.9g surfaceY=%.9g minBodyY=%.9g "
		"initialX=%.9g initialY=%.9g initialZ=%.9g "
		"finalX=%.9g finalY=%.9g finalZ=%.9g "
		"finalVx=%.9g finalVy=%.9g finalVz=%.9g nonFinite=%u "
		"fetchFailures=%u fatalErrors=%u cleanupComplete=%u pvd=0\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gMetrics.completedFrames,
		passed ? "PASS" : "FAIL", reason, generateCalls,
		generatedContacts, gMetrics.callbackCount, gMetrics.pairCount,
		gMetrics.reportedPoints, gMetrics.nonzeroImpulseCount,
		gMetrics.identityErrors, double(gMetrics.impulseSum),
		double(gMetrics.maxImpulse), double(gMetrics.surfaceY),
		double(gMetrics.minBodyY), double(gMetrics.initialPosition.x),
		double(gMetrics.initialPosition.y),
		double(gMetrics.initialPosition.z),
		double(gMetrics.finalPosition.x),
		double(gMetrics.finalPosition.y),
		double(gMetrics.finalPosition.z),
		double(gMetrics.finalVelocity.x),
		double(gMetrics.finalVelocity.y),
		double(gMetrics.finalVelocity.z), gMetrics.nonFinite,
		gMetrics.fetchFailures, gErrorCallback.getFatalCount(),
		gMetrics.cleanupComplete);
	return passed ? Snippets::eHEADLESS_PASS :
		Snippets::eHEADLESS_GATE_FAILED;
}

int snippetMain(int argc, const char* const* argv)
{
	std::string error;
	if(!parseHeadlessOptions(argc, argv, error))
	{
		std::fprintf(stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetCustomGeometry "
			"reason=%s\n", error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
	{
		std::fprintf(stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetCustomGeometry "
			"reason=execution_environment_failed\n");
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(gHeadlessOptions.headless)
		return runHeadless();

	extern void renderLoop();
	renderLoop();

	return 0;
}

#else
int snippetMain(int, const char* const*)
{
	return 0;
}

#endif
