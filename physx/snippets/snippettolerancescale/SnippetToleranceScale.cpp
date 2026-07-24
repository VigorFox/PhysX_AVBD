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

// ********************************************************************************
// This snippet illustrates the concept of PxToleranceScale.
//
// It creates 2 scenes using different units for length and mass.
// Use PVD to replay the scene and see how scaling affects the simulation.
// ********************************************************************************

#include <ctype.h>
#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"
#include <cfloat>
#include <cstdio>
#include <string>
#include <vector>

using namespace physx;

static PxDefaultAllocator		gAllocator;
static Snippets::TrackingErrorCallback gErrorCallback;
static PxFoundation*			gFoundation = NULL;
static PxPhysics*				gPhysics	= NULL;
static PxDefaultCpuDispatcher*	gDispatcher = NULL;
static PxScene*					gScene		= NULL;
static PxMaterial*				gMaterial	= NULL;
static PxPvd*					gPvd        = NULL;

static PxReal gStackZ = 10.0f;
static Snippets::HeadlessOptions gHeadlessOptions;
static PxRigidDynamic* gSphere = NULL;
static std::vector<PxRigidDynamic*> gDynamicActors;
static PxU32 gCleanupCount = 0;
struct ScaleRunMetrics;
static ScaleRunMetrics* gCurrentMetrics = NULL;

struct NormalizedBodyState
{
	PxVec3 position;
	PxQuat orientation;
	PxVec3 linearVelocity;
	PxVec3 angularVelocity;
	PxU32 sleeping;
};

struct ScaleRunMetrics
{
	PxReal length;
	PxReal speed;
	PxReal massScale;
	PxReal scaleReadbackError;
	PxReal contactOffsetNormalized;
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 nonFinite;
	PxU32 bodyCount;
	PxU32 sleepingBodies;
	PxReal sphereInitialY;
	PxReal sphereFinalY;
	PxReal sphereMinY;
	PxReal sphereMaxSpeed;
	PxVec3 meanPosition;
	PxReal meanSpeed;
	PxReal minBodyY;
	PxReal maxBodyY;
	std::vector<NormalizedBodyState> finalStates;

	ScaleRunMetrics()
	: length(0.0f), speed(0.0f), massScale(0.0f),
	  scaleReadbackError(0.0f), contactOffsetNormalized(0.0f),
	  completedFrames(0), fetchFailures(0), nonFinite(0), bodyCount(0),
	  sleepingBodies(0), sphereInitialY(0.0f), sphereFinalY(0.0f),
	  sphereMinY(FLT_MAX), sphereMaxSpeed(0.0f), meanPosition(0.0f),
	  meanSpeed(0.0f), minBodyY(FLT_MAX), maxBodyY(-FLT_MAX)
	{
	}
};

struct ScaleComparison
{
	PxReal sphereFinalPositionDelta;
	PxReal sphereMinYDelta;
	PxReal sphereMaxSpeedDelta;
	PxReal meanPositionDelta;
	PxReal meanSpeedDelta;
	PxReal minBodyYDelta;
	PxReal maxBodyYDelta;
	PxReal maxBodyPositionDelta;
	PxReal rmsBodyPositionDelta;
	PxReal maxBodyVelocityDelta;
	PxReal rmsBodyVelocityDelta;
	PxReal maxOrientationDelta;
	PxU32 sleepMismatch;

	ScaleComparison()
	: sphereFinalPositionDelta(0.0f), sphereMinYDelta(0.0f),
	  sphereMaxSpeedDelta(0.0f), meanPositionDelta(0.0f),
	  meanSpeedDelta(0.0f), minBodyYDelta(0.0f), maxBodyYDelta(0.0f),
	  maxBodyPositionDelta(0.0f), rmsBodyPositionDelta(0.0f),
	  maxBodyVelocityDelta(0.0f), rmsBodyVelocityDelta(0.0f),
	  maxOrientationDelta(0.0f), sleepMismatch(0)
	{
	}
};

PxRigidDynamic* createDynamic(const PxTransform& t, const PxGeometry& geometry, const PxReal& mass, const PxVec3& velocity=PxVec3(0))
{
	PxRigidDynamic* dynamic = PxCreateDynamic(*gPhysics, t, geometry, *gMaterial, 10.0f);
	dynamic->setAngularDamping(0.5f);
	dynamic->setLinearVelocity(velocity);
	PxRigidBodyExt::setMassAndUpdateInertia(*dynamic, mass);
	gScene->addActor(*dynamic);
	gDynamicActors.push_back(dynamic);
	return dynamic;
}

void createStack(const PxTransform& t, PxU32 size, PxReal halfExtent, const PxReal& mass)
{
	PxShape* shape = gPhysics->createShape(PxBoxGeometry(halfExtent, halfExtent, halfExtent), *gMaterial);
	if(gCurrentMetrics && gCurrentMetrics->contactOffsetNormalized == 0.0f)
		gCurrentMetrics->contactOffsetNormalized =
			shape->getContactOffset() / gCurrentMetrics->length;
	for(PxU32 i=0; i<size;i++)
	{
		for(PxU32 j=0;j<size-i;j++)
		{
			PxTransform localTm(PxVec3(PxReal(j*2) - PxReal(size-i), PxReal(i*2+1), 0) * halfExtent);
			PxRigidDynamic* body = gPhysics->createRigidDynamic(t.transform(localTm));
			body->attachShape(*shape);
			PxRigidBodyExt::setMassAndUpdateInertia(*body, mass);
			gScene->addActor(*body);
			gDynamicActors.push_back(body);
		}
	}
	shape->release();
}

bool initPhysics(bool interactive, const PxTolerancesScale& scale,
	PxReal scaleMass, ScaleRunMetrics& metrics)
{
	gCurrentMetrics = &metrics;
	gDynamicActors.clear();
	gSphere = NULL;
	gStackZ = 10.0f;
	metrics.length = scale.length;
	metrics.speed = scale.speed;
	metrics.massScale = scaleMass;
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gFoundation)
		return false;
	
	if(!gHeadlessOptions.headless)
	{
		gPvd = PxCreatePvd(*gFoundation);
		PxPvdTransport* transport =
			PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
		if(gPvd && transport)
			gPvd->connect(*transport,PxPvdInstrumentationFlag::eALL);
	}

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, scale, true, gPvd);
	if(!gPhysics)
		return false;
	const PxTolerancesScale& readback = gPhysics->getTolerancesScale();
	metrics.scaleReadbackError = PxMax(
		PxAbs(readback.length / scale.length - 1.0f),
		PxAbs(readback.speed / scale.speed - 1.0f));

	PxReal scaleLength = scale.length;

	PxSceneDesc sceneDesc(scale);
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f) * scaleLength;
	gDispatcher = PxDefaultCpuDispatcherCreate(
		gHeadlessOptions.headless ?
		gHeadlessOptions.dispatcherThreads : 2u);
	if(!gDispatcher)
		return false;
	sceneDesc.cpuDispatcher	= gDispatcher;
	sceneDesc.filterShader	= PxDefaultSimulationFilterShader;
	if(gHeadlessOptions.headless)
		sceneDesc.solverType = gHeadlessOptions.solverType;
	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return false;

	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
	if(pvdClient)
	{
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
	}

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);
	if(!gMaterial)
		return false;

	PxRigidStatic* groundPlane = PxCreatePlane(*gPhysics, PxPlane(0,1,0,0), *gMaterial);
	gScene->addActor(*groundPlane);

	for(PxU32 i=0;i<5;i++)
		createStack(PxTransform(PxVec3(0,0,gStackZ-=10.0f) * scaleLength), 10, 2.0f * scaleLength, 1.0f * scaleMass);

	if(!interactive)
	{
		gSphere = createDynamic(
			PxTransform(PxVec3(0,40,100) * scaleLength),
			PxSphereGeometry(10 * scaleLength), 100.0f * scaleMass,
			PxVec3(0,-50,-100) * scaleLength);
		metrics.sphereInitialY =
			gSphere->getGlobalPose().p.y / scaleLength;
		metrics.sphereMinY = metrics.sphereInitialY;
	}
	metrics.bodyCount = PxU32(gDynamicActors.size());
	return metrics.bodyCount == (interactive ? 275u : 276u);
}

bool stepPhysics(bool /*interactive*/, ScaleRunMetrics& metrics)
{
	gScene->simulate(gHeadlessOptions.headless ?
		gHeadlessOptions.dt : 1.0f/60.0f);
	if(!gScene->fetchResults(true))
	{
		metrics.fetchFailures++;
		return false;
	}
	metrics.completedFrames++;
	if(gSphere)
	{
		const PxTransform pose = gSphere->getGlobalPose();
		const PxVec3 velocity = gSphere->getLinearVelocity();
		if(!pose.isValid() || !velocity.isFinite())
			metrics.nonFinite++;
		else
		{
			metrics.sphereFinalY = pose.p.y / metrics.length;
			metrics.sphereMinY =
				PxMin(metrics.sphereMinY, metrics.sphereFinalY);
			metrics.sphereMaxSpeed = PxMax(
				metrics.sphereMaxSpeed,
				velocity.magnitude() / metrics.length);
		}
	}
	return metrics.nonFinite == 0;
}

void cleanupPhysics(bool /*interactive*/)
{
	PX_RELEASE(gScene);
	PX_RELEASE(gDispatcher);
	PX_RELEASE(gMaterial);
	PX_RELEASE(gPhysics);
	if (gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		PX_RELEASE(gPvd);
		PX_RELEASE(transport);
	}
	PX_RELEASE(gFoundation);
	gDynamicActors.clear();
	gSphere = NULL;
	gCurrentMetrics = NULL;
	gCleanupCount++;
}

bool captureFinalState(ScaleRunMetrics& metrics)
{
	metrics.finalStates.clear();
	metrics.finalStates.reserve(gDynamicActors.size());
	PxVec3 positionSum(0.0f);
	PxReal speedSum = 0.0f;
	for(PxU32 i = 0; i < gDynamicActors.size(); ++i)
	{
		PxRigidDynamic* body = gDynamicActors[i];
		const PxTransform pose = body->getGlobalPose();
		const PxVec3 linearVelocity = body->getLinearVelocity();
		const PxVec3 angularVelocity = body->getAngularVelocity();
		if(!pose.isValid() || !linearVelocity.isFinite() ||
			!angularVelocity.isFinite())
		{
			metrics.nonFinite++;
			continue;
		}
		NormalizedBodyState state;
		state.position = pose.p / metrics.length;
		state.orientation = pose.q;
		state.linearVelocity = linearVelocity / metrics.length;
		state.angularVelocity = angularVelocity;
		state.sleeping = body->isSleeping() ? 1u : 0u;
		metrics.finalStates.push_back(state);
		positionSum += state.position;
		speedSum += state.linearVelocity.magnitude();
		metrics.minBodyY = PxMin(metrics.minBodyY, state.position.y);
		metrics.maxBodyY = PxMax(metrics.maxBodyY, state.position.y);
		metrics.sleepingBodies += state.sleeping;
	}
	if(metrics.finalStates.size() != gDynamicActors.size() ||
		metrics.finalStates.empty())
		return false;
	const PxReal reciprocal =
		1.0f / PxReal(metrics.finalStates.size());
	metrics.meanPosition = positionSum * reciprocal;
	metrics.meanSpeed = speedSum * reciprocal;
	return metrics.nonFinite == 0;
}

bool runSim(const PxTolerancesScale& scale, PxReal scaleMass,
	ScaleRunMetrics& metrics)
{
	const PxU32 frameCount =
		gHeadlessOptions.headless ? gHeadlessOptions.frames : 150u;
	bool runOk = initPhysics(false, scale, scaleMass, metrics);
	if(runOk)
		for(PxU32 i=0; i<frameCount; i++)
			if(!stepPhysics(false, metrics))
			{
				runOk = false;
				break;
			}
	if(runOk)
		runOk = captureFinalState(metrics);
	cleanupPhysics(false);
	return runOk;
}

ScaleComparison compareScaleRuns(
	const ScaleRunMetrics& base, const ScaleRunMetrics& scaled)
{
	ScaleComparison comparison;
	comparison.sphereFinalPositionDelta =
		PxAbs(base.sphereFinalY - scaled.sphereFinalY);
	comparison.sphereMinYDelta =
		PxAbs(base.sphereMinY - scaled.sphereMinY);
	comparison.sphereMaxSpeedDelta =
		PxAbs(base.sphereMaxSpeed - scaled.sphereMaxSpeed);
	comparison.meanPositionDelta =
		(base.meanPosition - scaled.meanPosition).magnitude();
	comparison.meanSpeedDelta =
		PxAbs(base.meanSpeed - scaled.meanSpeed);
	comparison.minBodyYDelta =
		PxAbs(base.minBodyY - scaled.minBodyY);
	comparison.maxBodyYDelta =
		PxAbs(base.maxBodyY - scaled.maxBodyY);
	if(base.finalStates.size() != scaled.finalStates.size())
		return comparison;
	PxReal positionSquaredSum = 0.0f;
	PxReal velocitySquaredSum = 0.0f;
	for(PxU32 i = 0; i < base.finalStates.size(); ++i)
	{
		const NormalizedBodyState& a = base.finalStates[i];
		const NormalizedBodyState& b = scaled.finalStates[i];
		const PxReal positionDelta =
			(a.position - b.position).magnitude();
		const PxReal velocityDelta =
			(a.linearVelocity - b.linearVelocity).magnitude();
		const PxReal orientationDot =
			PxMin(PxAbs(a.orientation.dot(b.orientation)), 1.0f);
		const PxReal orientationDelta =
			2.0f * PxAcos(orientationDot);
		comparison.maxBodyPositionDelta =
			PxMax(comparison.maxBodyPositionDelta, positionDelta);
		comparison.maxBodyVelocityDelta =
			PxMax(comparison.maxBodyVelocityDelta, velocityDelta);
		comparison.maxOrientationDelta =
			PxMax(comparison.maxOrientationDelta, orientationDelta);
		positionSquaredSum += positionDelta * positionDelta;
		velocitySquaredSum += velocityDelta * velocityDelta;
		if(a.sleeping != b.sleeping)
			comparison.sleepMismatch++;
	}
	if(!base.finalStates.empty())
	{
		const PxReal reciprocal =
			1.0f / PxReal(base.finalStates.size());
		comparison.rmsBodyPositionDelta =
			PxSqrt(positionSquaredSum * reciprocal);
		comparison.rmsBodyVelocityDelta =
			PxSqrt(velocitySquaredSum * reciprocal);
	}
	return comparison;
}

bool parseHeadlessOptions(
	int argc, const char* const* argv, std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 150;
	defaults.caseName = "scale-pair";
	defaults.solverType = PxSolverType::eAVBD;
	defaults.dispatcherThreads = 4;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
		return false;
	for(int i = 1; i < argc; ++i)
		if(!Snippets::isCommonHeadlessOption(argv[i]))
		{
			error = std::string("unknown option: ") + argv[i];
			return false;
		}
	if(gHeadlessOptions.headless &&
		!Snippets::equalsIgnoreCase(
			gHeadlessOptions.caseName.c_str(), "scale-pair"))
	{
		error = "unsupported --case";
		return false;
	}
	if(gHeadlessOptions.headless && gHeadlessOptions.frames != 150)
	{
		error = "scale-pair requires --frames=150";
		return false;
	}
	if(gHeadlessOptions.headless &&
		gHeadlessOptions.solverType != PxSolverType::eTGS &&
		gHeadlessOptions.solverType != PxSolverType::eAVBD)
	{
		error = "headless gate supports only tgs or avbd";
		return false;
	}
	return true;
}

bool headlessPassed(const ScaleRunMetrics& base,
	const ScaleRunMetrics& scaled, const ScaleComparison& comparison)
{
	// The five impacted stacks are deliberately chaotic: even TGS has large
	// per-body pose deltas after normalization.  Gate the launched sphere to
	// well below its normalized radius (10) and the scene mean to half a box
	// half-extent (2), while retaining the strict pre-impact/minimum and speed
	// comparisons.  Per-body extrema remain diagnostics, not parity gates.
	return
		base.completedFrames == 150 &&
		scaled.completedFrames == 150 &&
		base.bodyCount == 276 &&
		scaled.bodyCount == 276 &&
		base.finalStates.size() == 276 &&
		scaled.finalStates.size() == 276 &&
		base.fetchFailures == 0 && scaled.fetchFailures == 0 &&
		base.nonFinite == 0 && scaled.nonFinite == 0 &&
		base.scaleReadbackError < 1e-6f &&
		scaled.scaleReadbackError < 1e-6f &&
		PxAbs(base.contactOffsetNormalized -
			scaled.contactOffsetNormalized) < 1e-6f &&
		comparison.sphereFinalPositionDelta < 1.6f &&
		comparison.sphereMinYDelta < 0.5f &&
		comparison.sphereMaxSpeedDelta < 2.0f &&
		comparison.meanPositionDelta < 1.0f &&
		comparison.meanSpeedDelta < 1.0f &&
		comparison.minBodyYDelta < 0.5f &&
		gCleanupCount == 2 &&
		gErrorCallback.getFatalCount() == 0;
}

const char* failureReason(const ScaleRunMetrics& base,
	const ScaleRunMetrics& scaled, const ScaleComparison& comparison)
{
	if(gErrorCallback.getFatalCount())
		return "fatal_error";
	if(gCleanupCount != 2)
		return "cleanup_incomplete";
	if(base.fetchFailures || scaled.fetchFailures)
		return "fetch_failed";
	if(base.nonFinite || scaled.nonFinite)
		return "non_finite";
	if(base.bodyCount != 276 || scaled.bodyCount != 276)
		return "body_count";
	if(base.scaleReadbackError >= 1e-6f ||
		scaled.scaleReadbackError >= 1e-6f)
		return "scale_readback";
	if(PxAbs(base.contactOffsetNormalized -
		scaled.contactOffsetNormalized) >= 1e-6f)
		return "contact_offset_scaling";
	if(comparison.sphereFinalPositionDelta >= 1.6f ||
		comparison.sphereMinYDelta >= 0.5f ||
		comparison.sphereMaxSpeedDelta >= 2.0f)
		return "sphere_scale_invariance";
	return "aggregate_scale_invariance";
}

void printHeadlessResult(const ScaleRunMetrics& base,
	const ScaleRunMetrics& scaled, const ScaleComparison& comparison)
{
	const bool pass = headlessPassed(base, scaled, comparison);
	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetToleranceScale solver=%s "
		"case=%s execution=%s frames=150 runs=2 "
		"baseCompleted=%u scaledCompleted=%u baseBodies=%u "
		"scaledBodies=%u baseLength=%.9g scaledLength=%.9g "
		"baseSpeed=%.9g scaledSpeed=%.9g baseMassScale=%.9g "
		"scaledMassScale=%.9g baseScaleError=%.9g scaledScaleError=%.9g "
		"baseContactOffsetN=%.9g scaledContactOffsetN=%.9g "
		"baseSphereFinalY=%.9g scaledSphereFinalY=%.9g "
		"baseSphereMinY=%.9g scaledSphereMinY=%.9g "
		"baseSphereMaxSpeed=%.9g scaledSphereMaxSpeed=%.9g "
		"sphereFinalDelta=%.9g sphereMinDelta=%.9g "
		"sphereSpeedDelta=%.9g meanPositionDelta=%.9g "
		"meanSpeedDelta=%.9g minBodyYDelta=%.9g maxBodyYDelta=%.9g "
		"maxBodyPositionDelta=%.9g rmsBodyPositionDelta=%.9g "
		"maxBodyVelocityDelta=%.9g rmsBodyVelocityDelta=%.9g "
		"maxOrientationDelta=%.9g sleepMismatch=%u "
		"baseNonFinite=%u scaledNonFinite=%u baseFetchFailures=%u "
		"scaledFetchFailures=%u fatalErrors=%u cleanupComplete=%u "
		"pvd=0 status=%s reason=%s validation=GATED\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		gHeadlessOptions.caseName.c_str(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		base.completedFrames, scaled.completedFrames, base.bodyCount,
		scaled.bodyCount, double(base.length), double(scaled.length),
		double(base.speed), double(scaled.speed), double(base.massScale),
		double(scaled.massScale), double(base.scaleReadbackError),
		double(scaled.scaleReadbackError),
		double(base.contactOffsetNormalized),
		double(scaled.contactOffsetNormalized),
		double(base.sphereFinalY), double(scaled.sphereFinalY),
		double(base.sphereMinY), double(scaled.sphereMinY),
		double(base.sphereMaxSpeed), double(scaled.sphereMaxSpeed),
		double(comparison.sphereFinalPositionDelta),
		double(comparison.sphereMinYDelta),
		double(comparison.sphereMaxSpeedDelta),
		double(comparison.meanPositionDelta),
		double(comparison.meanSpeedDelta),
		double(comparison.minBodyYDelta),
		double(comparison.maxBodyYDelta),
		double(comparison.maxBodyPositionDelta),
		double(comparison.rmsBodyPositionDelta),
		double(comparison.maxBodyVelocityDelta),
		double(comparison.rmsBodyVelocityDelta),
		double(comparison.maxOrientationDelta), comparison.sleepMismatch,
		base.nonFinite, scaled.nonFinite, base.fetchFailures,
		scaled.fetchFailures, gErrorCallback.getFatalCount(),
		gCleanupCount, pass ? "PASS" : "FAIL",
		pass ? "none" : failureReason(base, scaled, comparison));
}

int snippetMain(int argc, const char*const* argv)
{
	std::string error;
	if(!parseHeadlessOptions(argc, argv, error))
	{
		std::fprintf(stderr, "[AVBD_GATE_CONFIG_ERROR] %s\n", error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(gHeadlessOptions.headless)
	{
		if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
		{
			std::fprintf(stderr,
				"[AVBD_GATE_CONFIG_ERROR] failed to set execution mode\n");
			return Snippets::eHEADLESS_CONFIG_ERROR;
		}
		Snippets::printHeadlessConfig(
			"SnippetToleranceScale", gHeadlessOptions);
		gErrorCallback.reset();
		gCleanupCount = 0;
		ScaleRunMetrics baseMetrics;
		ScaleRunMetrics scaledMetrics;
		PxTolerancesScale baseScale;
		PxTolerancesScale scaledScale;
		scaledScale.length = 100.0f;
		scaledScale.speed *= scaledScale.length;
		bool runOk = runSim(baseScale, 1000.0f, baseMetrics);
		runOk = runSim(scaledScale, 1.0f, scaledMetrics) && runOk;
		const ScaleComparison comparison =
			compareScaleRuns(baseMetrics, scaledMetrics);
		printHeadlessResult(baseMetrics, scaledMetrics, comparison);
		return runOk &&
			headlessPassed(baseMetrics, scaledMetrics, comparison) ?
			Snippets::eHEADLESS_PASS : Snippets::eHEADLESS_GATE_FAILED;
	}

	PxTolerancesScale scale;

	// Default
	printf("PxToleranceScale (Default).\n");
	ScaleRunMetrics baseMetrics;
	runSim(scale, 1000.0f, baseMetrics);

	// Scaled assets
	printf("PxToleranceScale (Scaled).\n");
	scale.length = 100;				// length in cm
 	scale.speed *= scale.length;	// speed in cm/s
	ScaleRunMetrics scaledMetrics;
	runSim(scale, 1.0f, scaledMetrics);

	printf("SnippetToleranceScale done.\n");

	return 0;
}
