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
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.

// This snippet demonstrates qA + gearRatio*qB + offset = 0 for two degrees
// of freedom in one articulation. Its headless path is the AVBD failure-first
// authority for articulation-internal mimic coupling.

#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetutils/SnippetUtils.h"

#include <cmath>
#include <cstdio>
#include <cstring>
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
static PxPvdTransport* gPvdTransport = NULL;
static PxArticulationReducedCoordinate* gArticulation = NULL;
static PxArticulationLink* gRootLink = NULL;
static PxArticulationLink* gLinkA = NULL;
static PxArticulationLink* gLinkB = NULL;
static PxArticulationLink* gLinkC = NULL;
static PxArticulationLink* gLinkD = NULL;
static PxArticulationJointReducedCoordinate* gJointA = NULL;
static PxArticulationJointReducedCoordinate* gJointB = NULL;
static PxArticulationJointReducedCoordinate* gJointC = NULL;
static PxArticulationJointReducedCoordinate* gJointD = NULL;
static PxArticulationJointReducedCoordinate* gDriveJoint = NULL;
static PxArticulationJointReducedCoordinate* gDriveJoint2 = NULL;
static PxArticulationMimicJoint* gMimicJoint = NULL;
static PxArticulationMimicJoint* gMimicJoint2 = NULL;
static bool gExtensionsInitialized = false;

static const PxArticulationAxis::Enum gAxisA =
    PxArticulationAxis::eTWIST;
static const PxArticulationAxis::Enum gAxisB = PxArticulationAxis::eX;
static const PxArticulationAxis::Enum gAxisC = PxArticulationAxis::eY;
static const PxArticulationAxis::Enum gAxisD = PxArticulationAxis::eZ;
static const PxReal gDrivePositionMin = -0.6f;
static const PxReal gDrivePositionMax = 0.6f;
static const PxReal gDriveSpeed = 1.5f;
static const PxReal gSecondRatio = -1.5f;
static const PxReal gSecondOffset = -0.15f;

enum MimicHeadlessCase
{
	eDRIVE_A,
	eDRIVE_B
};

enum MimicHeadlessMode
{
	eSINGLE_HARD,
	eSINGLE_COMPLIANT,
	eMULTI_HARD
};

struct MimicHeadlessConfig
{
	PxReal ratio;
	PxReal offset;
	PxReal naturalFrequency;
	PxReal dampingRatio;
	MimicHeadlessMode mode;

	MimicHeadlessConfig()
	: ratio(1.0f), offset(0.25f), naturalFrequency(0.0f),
	  dampingRatio(0.0f), mode(eSINGLE_HARD)
	{
	}
};

struct MimicMetrics
{
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 nonFinite;
	PxU32 responseSamples;
	PxU32 directionSamples;
	PxU32 directionViolations;
	PxReal maxPositionError;
	PxReal positionErrorSquaredSum;
	PxReal maxVelocityError;
	PxReal velocityErrorSquaredSum;
	PxReal minQA;
	PxReal maxQA;
	PxReal minQB;
	PxReal maxQB;
	PxReal finalQA;
	PxReal finalQB;
	PxReal finalVelocityA;
	PxReal finalVelocityB;
	PxReal finalPublicQA;
	PxReal finalPublicQB;
	PxReal maxPublicCoordinateError;
	PxU32 secondResponseSamples;
	PxU32 secondDirectionSamples;
	PxU32 secondDirectionViolations;
	PxReal secondMaxPositionError;
	PxReal secondPositionErrorSquaredSum;
	PxReal secondMaxVelocityError;
	PxReal secondVelocityErrorSquaredSum;
	PxReal minQC;
	PxReal maxQC;
	PxReal minQD;
	PxReal maxQD;
	PxReal finalQC;
	PxReal finalQD;
	PxReal finalVelocityC;
	PxReal finalVelocityD;
	PxReal finalPublicQC;
	PxReal finalPublicQD;
	PxReal secondMaxPublicCoordinateError;
	PxU32 cleanupComplete;

	MimicMetrics()
	: completedFrames(0), fetchFailures(0), nonFinite(0),
	  responseSamples(0), directionSamples(0), directionViolations(0),
	  maxPositionError(0.0f), positionErrorSquaredSum(0.0f),
	  maxVelocityError(0.0f), velocityErrorSquaredSum(0.0f),
	  minQA(PX_MAX_F32), maxQA(-PX_MAX_F32), minQB(PX_MAX_F32),
	  maxQB(-PX_MAX_F32), finalQA(0.0f), finalQB(0.0f),
	  finalVelocityA(0.0f), finalVelocityB(0.0f), finalPublicQA(0.0f),
	  finalPublicQB(0.0f), maxPublicCoordinateError(0.0f),
	  secondResponseSamples(0), secondDirectionSamples(0),
	  secondDirectionViolations(0), secondMaxPositionError(0.0f),
	  secondPositionErrorSquaredSum(0.0f), secondMaxVelocityError(0.0f),
	  secondVelocityErrorSquaredSum(0.0f), minQC(PX_MAX_F32),
	  maxQC(-PX_MAX_F32), minQD(PX_MAX_F32), maxQD(-PX_MAX_F32),
	  finalQC(0.0f), finalQD(0.0f), finalVelocityC(0.0f),
	  finalVelocityD(0.0f), finalPublicQC(0.0f), finalPublicQD(0.0f),
	  secondMaxPublicCoordinateError(0.0f),
	  cleanupComplete(0)
	{
	}
};

static Snippets::HeadlessOptions gHeadlessOptions;
static MimicHeadlessConfig gHeadlessConfig;
static MimicHeadlessCase gHeadlessCase = eDRIVE_A;
static MimicMetrics gMetrics;
static PxReal gDrivePosition = 0.0f;
static PxReal gDriveDirection = 1.0f;

static const char* getMimicCaseName()
{
	return gHeadlessCase == eDRIVE_A ? "drive-a" : "drive-b";
}

static const char* getMimicModeName()
{
	switch(gHeadlessConfig.mode)
	{
	case eSINGLE_HARD:
		return "hard";
	case eSINGLE_COMPLIANT:
		return "compliant";
	case eMULTI_HARD:
		return "multi";
	}
	return "unknown";
}

static bool parseMimicMode(const char* value, MimicHeadlessMode& result)
{
	if(Snippets::equalsIgnoreCase(value, "hard"))
	{
		result = eSINGLE_HARD;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "compliant"))
	{
		result = eSINGLE_COMPLIANT;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "multi"))
	{
		result = eMULTI_HARD;
		return true;
	}
	return false;
}

static bool parseMimicCase(const char* value, MimicHeadlessCase& result)
{
	if(Snippets::equalsIgnoreCase(value, "drive-a"))
	{
		result = eDRIVE_A;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "drive-b"))
	{
		result = eDRIVE_B;
		return true;
	}
	return false;
}

static bool parseHeadlessOptions(int argc, const char* const* argv,
	std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 360;
	defaults.caseName = "drive-a";
	defaults.solverType = PxSolverType::eAVBD;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
		return false;

	bool ratioSeen = false;
	bool offsetSeen = false;
	bool modeSeen = false;
	bool naturalFrequencySeen = false;
	bool dampingRatioSeen = false;
	for(int i = 1; i < argc; ++i)
	{
		const char* arg = argv[i];
		if(Snippets::isCommonHeadlessOption(arg))
			continue;
		if(Snippets::hasOptionPrefix(arg, "--ratio="))
		{
			if(ratioSeen || !Snippets::parseReal(
				arg + std::strlen("--ratio="), -10.0f, 10.0f,
				gHeadlessConfig.ratio) ||
				PxAbs(gHeadlessConfig.ratio) < 0.1f)
			{
				error = ratioSeen ? "duplicate --ratio" : "invalid --ratio";
				return false;
			}
			ratioSeen = true;
			continue;
		}
		if(Snippets::hasOptionPrefix(arg, "--offset="))
		{
			if(offsetSeen || !Snippets::parseReal(
				arg + std::strlen("--offset="), -2.0f, 2.0f,
				gHeadlessConfig.offset))
			{
				error = offsetSeen ? "duplicate --offset" : "invalid --offset";
				return false;
			}
			offsetSeen = true;
			continue;
		}
		if(Snippets::hasOptionPrefix(arg, "--mimic-mode="))
		{
			if(modeSeen || !parseMimicMode(
				arg + std::strlen("--mimic-mode="), gHeadlessConfig.mode))
			{
				error = modeSeen ? "duplicate --mimic-mode" :
					"invalid --mimic-mode";
				return false;
			}
			modeSeen = true;
			continue;
		}
		if(Snippets::hasOptionPrefix(arg, "--natural-frequency="))
		{
			if(naturalFrequencySeen || !Snippets::parseReal(
				arg + std::strlen("--natural-frequency="), 0.0f, 100.0f,
				gHeadlessConfig.naturalFrequency))
			{
				error = naturalFrequencySeen ?
					"duplicate --natural-frequency" :
					"invalid --natural-frequency";
				return false;
			}
			naturalFrequencySeen = true;
			continue;
		}
		if(Snippets::hasOptionPrefix(arg, "--damping-ratio="))
		{
			if(dampingRatioSeen || !Snippets::parseReal(
				arg + std::strlen("--damping-ratio="), 0.0f, 10.0f,
				gHeadlessConfig.dampingRatio))
			{
				error = dampingRatioSeen ? "duplicate --damping-ratio" :
					"invalid --damping-ratio";
				return false;
			}
			dampingRatioSeen = true;
			continue;
		}
		error = std::string("unknown option: ") + (arg ? arg : "<null>");
		return false;
	}

	if(!parseMimicCase(gHeadlessOptions.caseName.c_str(), gHeadlessCase))
	{
		error = "unsupported --case (expected drive-a or drive-b)";
		return false;
	}
	if(gHeadlessConfig.mode == eSINGLE_COMPLIANT)
	{
		if(!naturalFrequencySeen)
			gHeadlessConfig.naturalFrequency = 6.0f;
		if(!dampingRatioSeen)
			gHeadlessConfig.dampingRatio = 1.0f;
		if(gHeadlessConfig.naturalFrequency <= 0.0f ||
			gHeadlessConfig.dampingRatio <= 0.0f)
		{
			error = "compliant mode requires positive frequency and damping";
			return false;
		}
	}
	else if(gHeadlessConfig.naturalFrequency != 0.0f ||
		gHeadlessConfig.dampingRatio != 0.0f)
	{
		error = "hard/multi mode requires zero compliance parameters";
		return false;
	}
	if(gHeadlessOptions.frames < 180)
	{
		error = "--frames must be at least 180";
		return false;
	}
	return true;
}

static PxArticulationReducedCoordinate* createArticulation(
	PxArticulationJointReducedCoordinate*& driveJoint)
{
	PxArticulationReducedCoordinate* articulation =
		gPhysics->createArticulationReducedCoordinate();
	if(!articulation)
		return NULL;

	articulation->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);
	articulation->setArticulationFlag(
		PxArticulationFlag::eDISABLE_SELF_COLLISION, true);
	articulation->setSolverIterationCounts(32, 4);
	articulation->setSleepThreshold(0.0f);

	gRootLink = articulation->createLink(NULL, PxTransform(PxIdentity));
	if(!gRootLink)
		return articulation;
	gRootLink->setCfmScale(0.0f);
	gRootLink->setLinearDamping(0.0f);
	gRootLink->setAngularDamping(0.0f);
	PxShape* rootShape =
		gPhysics->createShape(PxSphereGeometry(0.25f), *gMaterial, true);
	if(rootShape)
	{
		gRootLink->attachShape(*rootShape);
		rootShape->release();
	}

	const bool multi = gHeadlessConfig.mode == eMULTI_HARD;
	const PxBoxGeometry linkGeometry(PxVec3(0.35f));
	gLinkA = articulation->createLink(gRootLink, PxTransform(PxIdentity));
	gLinkB = articulation->createLink(gRootLink, PxTransform(PxIdentity));
	if(multi)
	{
		gLinkC = articulation->createLink(
			gRootLink, PxTransform(PxIdentity));
		gLinkD = articulation->createLink(
			gRootLink, PxTransform(PxIdentity));
	}
	if(!gLinkA || !gLinkB || (multi && (!gLinkC || !gLinkD)))
		return articulation;

	PxArticulationLink* links[4] = {gLinkA, gLinkB, gLinkC, gLinkD};
	const PxU32 linkCount = multi ? 4u : 2u;
	for(PxU32 i = 0; i < linkCount; ++i)
	{
		links[i]->setCfmScale(0.0f);
		links[i]->setLinearDamping(0.0f);
		links[i]->setAngularDamping(0.0f);
		PxShape* shape =
			gPhysics->createShape(linkGeometry, *gMaterial, true);
		if(shape)
		{
			links[i]->attachShape(*shape);
			shape->release();
		}
		PxRigidBodyExt::updateMassAndInertia(*links[i], 1.0f);
	}

	gJointA = gLinkA->getInboundJoint();
	gJointB = gLinkB->getInboundJoint();
	if(multi)
	{
		gJointC = gLinkC->getInboundJoint();
		gJointD = gLinkD->getInboundJoint();
	}
	if(!gJointA || !gJointB || (multi && (!gJointC || !gJointD)))
		return articulation;

	gJointA->setJointType(PxArticulationJointType::eREVOLUTE);
	gJointA->setMotion(gAxisA, PxArticulationMotion::eFREE);
	gJointA->setParentPose(PxTransform(PxVec3(-2.0f, 0.0f, 0.0f)));
	gJointA->setChildPose(PxTransform(PxIdentity));
	gJointA->setFrictionCoefficient(0.0f);

	gJointB->setJointType(PxArticulationJointType::ePRISMATIC);
	gJointB->setMotion(gAxisB, PxArticulationMotion::eFREE);
	gJointB->setParentPose(PxTransform(PxVec3(2.0f, 0.0f, 0.0f)));
	gJointB->setChildPose(PxTransform(PxIdentity));
	gJointB->setFrictionCoefficient(0.0f);

	if(multi)
	{
		gJointC->setJointType(PxArticulationJointType::ePRISMATIC);
		gJointC->setMotion(gAxisC, PxArticulationMotion::eFREE);
		gJointC->setParentPose(PxTransform(PxVec3(0.0f, -2.0f, 0.0f)));
		gJointC->setChildPose(PxTransform(PxIdentity));
		gJointC->setFrictionCoefficient(0.0f);

		gJointD->setJointType(PxArticulationJointType::ePRISMATIC);
		gJointD->setMotion(gAxisD, PxArticulationMotion::eFREE);
		gJointD->setParentPose(PxTransform(PxVec3(0.0f, 0.0f, 2.0f)));
		gJointD->setChildPose(PxTransform(PxIdentity));
		gJointD->setFrictionCoefficient(0.0f);
	}

	if(gHeadlessCase == eDRIVE_A)
	{
		driveJoint = gJointA;
		driveJoint->setDriveParams(
			gAxisA, PxArticulationDrive(10000.0f, 100.0f, PX_MAX_F32));
		if(multi)
		{
			gDriveJoint2 = gJointC;
			gDriveJoint2->setDriveParams(
				gAxisC, PxArticulationDrive(10000.0f, 100.0f, PX_MAX_F32));
		}
	}
	else
	{
		driveJoint = gJointB;
		driveJoint->setDriveParams(
			gAxisB, PxArticulationDrive(10000.0f, 100.0f, PX_MAX_F32));
		if(multi)
		{
			gDriveJoint2 = gJointD;
			gDriveJoint2->setDriveParams(
				gAxisD, PxArticulationDrive(10000.0f, 100.0f, PX_MAX_F32));
		}
	}

	gMimicJoint = articulation->createMimicJoint(
		*gJointA, gAxisA, *gJointB, gAxisB, gHeadlessConfig.ratio,
		gHeadlessConfig.offset, gHeadlessConfig.naturalFrequency,
		gHeadlessConfig.dampingRatio);
	if(multi)
	{
		gMimicJoint2 = articulation->createMimicJoint(
			*gJointC, gAxisC, *gJointD, gAxisD, gSecondRatio,
			gSecondOffset);
	}
	return articulation;
}

void initPhysics(bool interactive)
{
	gErrorCallback.reset();
	gFoundation =
		PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	if(!gFoundation)
		return;

	if(interactive)
	{
		gPvd = PxCreatePvd(*gFoundation);
		gPvdTransport =
			PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
		if(gPvd && gPvdTransport)
			gPvd->connect(*gPvdTransport, PxPvdInstrumentationFlag::eALL);
	}

	gPhysics = PxCreatePhysics(
		PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);
	if(!gPhysics)
		return;
	if(!PxInitExtensions(*gPhysics, gPvd))
		return;
	gExtensionsInitialized = true;

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.0f);
	if(!gMaterial)
		return;

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f);
	const PxU32 dispatcherThreads =
		interactive ? 2u : gHeadlessOptions.dispatcherThreads;
	gDispatcher = PxDefaultCpuDispatcherCreate(dispatcherThreads);
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader = PxDefaultSimulationFilterShader;
	sceneDesc.solverType =
		interactive ? PxSolverType::eAVBD : gHeadlessOptions.solverType;
	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return;

	gArticulation = createArticulation(gDriveJoint);
	if(gArticulation && gMimicJoint)
		gScene->addArticulation(*gArticulation);
}

static void recordHeadlessState(PxU32 frame)
{
	if(!gRootLink || !gLinkA || !gLinkB || !gJointA || !gJointB)
		return;

	const PxTransform rootPose = gRootLink->getGlobalPose();
	const PxTransform linkAPose = gLinkA->getGlobalPose();
	const PxTransform linkBPose = gLinkB->getGlobalPose();
	const PxTransform parentFrameA =
		rootPose * gJointA->getParentPose();
	const PxTransform childFrameA =
		linkAPose * gJointA->getChildPose();
	PxQuat relativeA =
		parentFrameA.q.getConjugate() * childFrameA.q;
	if(relativeA.w < 0.0f)
		relativeA = -relativeA;
	const PxReal qA = 2.0f * PxAtan2(relativeA.x, relativeA.w);

	const PxTransform parentFrameB =
		rootPose * gJointB->getParentPose();
	const PxTransform childFrameB =
		linkBPose * gJointB->getChildPose();
	const PxVec3 axisA = parentFrameA.q.getBasisVector0();
	const PxVec3 axisB = parentFrameB.q.getBasisVector0();
	const PxReal qB = (childFrameB.p - parentFrameB.p).dot(axisB);
	const PxReal vA =
		(gLinkA->getAngularVelocity() -
			gRootLink->getAngularVelocity()).dot(axisA);
	const PxReal vB =
		(gLinkB->getLinearVelocity() -
			gRootLink->getLinearVelocity()).dot(axisB);
	const PxReal publicQA = gJointA->getJointPosition(gAxisA);
	const PxReal publicQB = gJointB->getJointPosition(gAxisB);
	PxReal qC = 0.0f;
	PxReal qD = 0.0f;
	PxReal vC = 0.0f;
	PxReal vD = 0.0f;
	PxReal publicQC = 0.0f;
	PxReal publicQD = 0.0f;
	const bool multi = gHeadlessConfig.mode == eMULTI_HARD;
	if(multi)
	{
		if(!gLinkC || !gLinkD || !gJointC || !gJointD)
		{
			++gMetrics.nonFinite;
			return;
		}
		const PxTransform linkCPose = gLinkC->getGlobalPose();
		const PxTransform linkDPose = gLinkD->getGlobalPose();
		const PxTransform parentFrameC =
			rootPose * gJointC->getParentPose();
		const PxTransform childFrameC =
			linkCPose * gJointC->getChildPose();
		const PxTransform parentFrameD =
			rootPose * gJointD->getParentPose();
		const PxTransform childFrameD =
			linkDPose * gJointD->getChildPose();
		const PxVec3 axisC = parentFrameC.q.getBasisVector1();
		const PxVec3 axisD = parentFrameD.q.getBasisVector2();
		qC = (childFrameC.p - parentFrameC.p).dot(axisC);
		qD = (childFrameD.p - parentFrameD.p).dot(axisD);
		vC = (gLinkC->getLinearVelocity() -
			gRootLink->getLinearVelocity()).dot(axisC);
		vD = (gLinkD->getLinearVelocity() -
			gRootLink->getLinearVelocity()).dot(axisD);
		publicQC = gJointC->getJointPosition(gAxisC);
		publicQD = gJointD->getJointPosition(gAxisD);
		if(!linkCPose.isFinite() || !linkDPose.isFinite())
		{
			++gMetrics.nonFinite;
			return;
		}
	}
	if(!PxIsFinite(qA) || !PxIsFinite(qB) ||
		!PxIsFinite(vA) || !PxIsFinite(vB) ||
		!PxIsFinite(publicQA) || !PxIsFinite(publicQB) ||
		!PxIsFinite(qC) || !PxIsFinite(qD) ||
		!PxIsFinite(vC) || !PxIsFinite(vD) ||
		!PxIsFinite(publicQC) || !PxIsFinite(publicQD) ||
		!rootPose.isFinite() || !linkAPose.isFinite() ||
		!linkBPose.isFinite())
	{
		++gMetrics.nonFinite;
		return;
	}

	gMetrics.finalQA = qA;
	gMetrics.finalQB = qB;
	gMetrics.finalVelocityA = vA;
	gMetrics.finalVelocityB = vB;
	gMetrics.finalPublicQA = publicQA;
	gMetrics.finalPublicQB = publicQB;
	gMetrics.maxPublicCoordinateError = PxMax(
		gMetrics.maxPublicCoordinateError,
		PxMax(PxAbs(publicQA - qA), PxAbs(publicQB - qB)));
	if(multi)
	{
		gMetrics.finalQC = qC;
		gMetrics.finalQD = qD;
		gMetrics.finalVelocityC = vC;
		gMetrics.finalVelocityD = vD;
		gMetrics.finalPublicQC = publicQC;
		gMetrics.finalPublicQD = publicQD;
		gMetrics.secondMaxPublicCoordinateError = PxMax(
			gMetrics.secondMaxPublicCoordinateError,
			PxMax(PxAbs(publicQC - qC), PxAbs(publicQD - qD)));
	}
	if(frame < 60)
		return;

	const PxReal positionError =
		qA + gHeadlessConfig.ratio * qB + gHeadlessConfig.offset;
	const PxReal velocityError = vA + gHeadlessConfig.ratio * vB;
	gMetrics.maxPositionError =
		PxMax(gMetrics.maxPositionError, PxAbs(positionError));
	gMetrics.positionErrorSquaredSum += positionError * positionError;
	gMetrics.maxVelocityError =
		PxMax(gMetrics.maxVelocityError, PxAbs(velocityError));
	gMetrics.velocityErrorSquaredSum += velocityError * velocityError;
	gMetrics.minQA = PxMin(gMetrics.minQA, qA);
	gMetrics.maxQA = PxMax(gMetrics.maxQA, qA);
	gMetrics.minQB = PxMin(gMetrics.minQB, qB);
	gMetrics.maxQB = PxMax(gMetrics.maxQB, qB);
	++gMetrics.responseSamples;

	if(PxAbs(vA) > 0.02f && PxAbs(vB) > 0.02f)
	{
		++gMetrics.directionSamples;
		if(vA * gHeadlessConfig.ratio * vB >= 0.0f)
			++gMetrics.directionViolations;
	}

	if(multi)
	{
		const PxReal secondPositionError =
			qC + gSecondRatio * qD + gSecondOffset;
		const PxReal secondVelocityError = vC + gSecondRatio * vD;
		gMetrics.secondMaxPositionError = PxMax(
			gMetrics.secondMaxPositionError, PxAbs(secondPositionError));
		gMetrics.secondPositionErrorSquaredSum +=
			secondPositionError * secondPositionError;
		gMetrics.secondMaxVelocityError = PxMax(
			gMetrics.secondMaxVelocityError, PxAbs(secondVelocityError));
		gMetrics.secondVelocityErrorSquaredSum +=
			secondVelocityError * secondVelocityError;
		gMetrics.minQC = PxMin(gMetrics.minQC, qC);
		gMetrics.maxQC = PxMax(gMetrics.maxQC, qC);
		gMetrics.minQD = PxMin(gMetrics.minQD, qD);
		gMetrics.maxQD = PxMax(gMetrics.maxQD, qD);
		++gMetrics.secondResponseSamples;
		if(PxAbs(vC) > 0.02f && PxAbs(vD) > 0.02f)
		{
			++gMetrics.secondDirectionSamples;
			if(vC * gSecondRatio * vD >= 0.0f)
				++gMetrics.secondDirectionViolations;
		}
	}
}

void stepPhysics(bool interactive)
{
	gDrivePosition +=
		gDriveSpeed * gDriveDirection *
		(interactive ? 1.0f / 60.0f : gHeadlessOptions.dt);
	if(gDrivePosition > gDrivePositionMax)
	{
		gDrivePosition = gDrivePositionMax;
		gDriveDirection = -1.0f;
	}
	else if(gDrivePosition < gDrivePositionMin)
	{
		gDrivePosition = gDrivePositionMin;
		gDriveDirection = 1.0f;
	}

	const PxArticulationAxis::Enum driveAxis =
		gHeadlessCase == eDRIVE_A ? gAxisA : gAxisB;
	gDriveJoint->setDriveTarget(driveAxis, gDrivePosition);
	if(gDriveJoint2)
	{
		const PxArticulationAxis::Enum driveAxis2 =
			gHeadlessCase == eDRIVE_A ? gAxisC : gAxisD;
		gDriveJoint2->setDriveTarget(driveAxis2, -gDrivePosition);
	}
	gScene->simulate(interactive ? 1.0f / 60.0f : gHeadlessOptions.dt);
	if(!gScene->fetchResults(true))
	{
		++gMetrics.fetchFailures;
		return;
	}
	if(!interactive)
		recordHeadlessState(gMetrics.completedFrames);
	++gMetrics.completedFrames;
}

void cleanupPhysics(bool interactive)
{
	PX_UNUSED(interactive);
	PX_RELEASE(gArticulation);
	gMimicJoint = NULL;
	gMimicJoint2 = NULL;
	gDriveJoint = NULL;
	gDriveJoint2 = NULL;
	gJointA = NULL;
	gJointB = NULL;
	gJointC = NULL;
	gJointD = NULL;
	gRootLink = NULL;
	gLinkA = NULL;
	gLinkB = NULL;
	gLinkC = NULL;
	gLinkD = NULL;
	PX_RELEASE(gScene);
	PX_RELEASE(gDispatcher);
	PX_RELEASE(gMaterial);
	if(gExtensionsInitialized)
	{
		PxCloseExtensions();
		gExtensionsInitialized = false;
	}
	PX_RELEASE(gPhysics);
	PX_RELEASE(gPvd);
	PX_RELEASE(gPvdTransport);
	PX_RELEASE(gFoundation);
	gMetrics.cleanupComplete =
		!gArticulation && !gScene && !gDispatcher && !gMaterial &&
		!gPhysics && !gPvd && !gPvdTransport && !gFoundation;
	if(interactive)
		std::printf("SnippetMimicJoint done.\n");
}

static int runHeadless()
{
	Snippets::printHeadlessConfig("SnippetMimicJoint", gHeadlessOptions);
	initPhysics(false);
	const bool initialized =
		gFoundation && gPhysics && gDispatcher && gScene && gMaterial &&
		gArticulation && gRootLink && gLinkA && gLinkB && gJointA && gJointB &&
		gDriveJoint && gMimicJoint &&
		(gHeadlessConfig.mode != eMULTI_HARD ||
			(gLinkC && gLinkD && gJointC && gJointD && gDriveJoint2 &&
				gMimicJoint2));

	if(initialized)
	{
		for(PxU32 frame = 0; frame < gHeadlessOptions.frames; ++frame)
		{
			stepPhysics(false);
			if(gMetrics.fetchFailures || gMetrics.nonFinite)
				break;
		}
	}

	const PxReal positionRms =
		gMetrics.responseSamples
			? PxSqrt(gMetrics.positionErrorSquaredSum /
				PxReal(gMetrics.responseSamples))
			: PX_MAX_F32;
	const PxReal velocityRms =
		gMetrics.responseSamples
			? PxSqrt(gMetrics.velocityErrorSquaredSum /
				PxReal(gMetrics.responseSamples))
			: PX_MAX_F32;
	const PxReal rangeA = gMetrics.maxQA - gMetrics.minQA;
	const PxReal rangeB = gMetrics.maxQB - gMetrics.minQB;
	const PxReal secondPositionRms =
		gMetrics.secondResponseSamples
			? PxSqrt(gMetrics.secondPositionErrorSquaredSum /
				PxReal(gMetrics.secondResponseSamples))
			: 0.0f;
	const PxReal secondVelocityRms =
		gMetrics.secondResponseSamples
			? PxSqrt(gMetrics.secondVelocityErrorSquaredSum /
				PxReal(gMetrics.secondResponseSamples))
			: 0.0f;
	const PxReal rangeC = gMetrics.secondResponseSamples ?
		gMetrics.maxQC - gMetrics.minQC : 0.0f;
	const PxReal rangeD = gMetrics.secondResponseSamples ?
		gMetrics.maxQD - gMetrics.minQD : 0.0f;
	const PxU32 expectedMimicCount =
		gHeadlessConfig.mode == eMULTI_HARD ? 2u : 1u;
	const bool readbackValid =
		gMimicJoint &&
		PxAbs(gMimicJoint->getGearRatio() - gHeadlessConfig.ratio) < 1e-6f &&
		PxAbs(gMimicJoint->getOffset() - gHeadlessConfig.offset) < 1e-6f &&
		PxAbs(gMimicJoint->getNaturalFrequency() -
			gHeadlessConfig.naturalFrequency) < 1e-6f &&
		PxAbs(gMimicJoint->getDampingRatio() -
			gHeadlessConfig.dampingRatio) < 1e-6f &&
		gMimicJoint->getAxisA() == gAxisA &&
		gMimicJoint->getAxisB() == gAxisB &&
		&gMimicJoint->getJointA() == gJointA &&
		&gMimicJoint->getJointB() == gJointB &&
		gArticulation->getNbMimicJoints() == expectedMimicCount &&
		(gHeadlessConfig.mode != eMULTI_HARD ||
			(gMimicJoint2 &&
				PxAbs(gMimicJoint2->getGearRatio() - gSecondRatio) < 1e-6f &&
				PxAbs(gMimicJoint2->getOffset() - gSecondOffset) < 1e-6f &&
				gMimicJoint2->getAxisA() == gAxisC &&
				gMimicJoint2->getAxisB() == gAxisD &&
				&gMimicJoint2->getJointA() == gJointC &&
				&gMimicJoint2->getJointB() == gJointD));

	const char* reason = "none";
	bool passed = true;
	if(!initialized)
	{
		passed = false;
		reason = "initialization_failed";
	}
	else if(!readbackValid)
	{
		passed = false;
		reason = "api_readback_mismatch";
	}
	else if(gMetrics.completedFrames != gHeadlessOptions.frames ||
		gMetrics.fetchFailures != 0)
	{
		passed = false;
		reason = "incomplete_simulation";
	}
	else if(gMetrics.nonFinite != 0 || gErrorCallback.getFatalCount() != 0)
	{
		passed = false;
		reason = "runtime_error";
	}
	else if(gMetrics.responseSamples < gHeadlessOptions.frames - 62)
	{
		passed = false;
		reason = "missing_samples";
	}
	else if(gHeadlessConfig.mode == eMULTI_HARD &&
		gMetrics.secondResponseSamples < gHeadlessOptions.frames - 62)
	{
		passed = false;
		reason = "missing_second_samples";
	}
	else if(rangeA < 0.15f || rangeB < 0.15f)
	{
		passed = false;
		reason = "missing_coupled_response";
	}
	else if(gMetrics.directionSamples < 30 ||
		gMetrics.directionViolations * 10 > gMetrics.directionSamples)
	{
		passed = false;
		reason = "wrong_coupling_direction";
	}
	else if(gHeadlessConfig.mode == eMULTI_HARD &&
		(rangeC < 0.15f || rangeD < 0.15f))
	{
		passed = false;
		reason = "missing_second_coupled_response";
	}
	else if(gHeadlessConfig.mode == eMULTI_HARD &&
		(gMetrics.secondDirectionSamples < 30 ||
			gMetrics.secondDirectionViolations * 10 >
				gMetrics.secondDirectionSamples))
	{
		passed = false;
		reason = "wrong_second_coupling_direction";
	}
	else if(gHeadlessConfig.mode != eSINGLE_COMPLIANT &&
		(positionRms > 0.04f || gMetrics.maxPositionError > 0.12f))
	{
		passed = false;
		reason = "mimic_position_error";
	}
	else if(gHeadlessConfig.mode != eSINGLE_COMPLIANT &&
		(velocityRms > 0.12f || gMetrics.maxVelocityError > 0.5f))
	{
		passed = false;
		reason = "mimic_velocity_error";
	}
	else if(gHeadlessConfig.mode == eSINGLE_COMPLIANT &&
		(positionRms > 0.6f || gMetrics.maxPositionError > 1.5f ||
			velocityRms > 2.0f || gMetrics.maxVelocityError > 6.0f))
	{
		passed = false;
		reason = "compliant_mimic_unbounded";
	}
	else if(gHeadlessConfig.mode == eMULTI_HARD &&
		(secondPositionRms > 0.04f ||
			gMetrics.secondMaxPositionError > 0.12f ||
			secondVelocityRms > 0.12f ||
			gMetrics.secondMaxVelocityError > 0.5f))
	{
		passed = false;
		reason = "second_mimic_error";
	}
	else if(gMetrics.maxPublicCoordinateError > 0.002f)
	{
		passed = false;
		reason = "joint_state_mismatch";
	}
	else if(gMetrics.secondMaxPublicCoordinateError > 0.002f)
	{
		passed = false;
		reason = "second_joint_state_mismatch";
	}

	const PxReal ratioReadback =
		gMimicJoint ? gMimicJoint->getGearRatio() : 0.0f;
	const PxReal offsetReadback =
		gMimicJoint ? gMimicJoint->getOffset() : 0.0f;
	const PxReal naturalFrequencyReadback =
		gMimicJoint ? gMimicJoint->getNaturalFrequency() : 0.0f;
	const PxReal dampingRatioReadback =
		gMimicJoint ? gMimicJoint->getDampingRatio() : 0.0f;
	const PxU32 mimicCountReadback =
		gArticulation ? gArticulation->getNbMimicJoints() : 0u;
	cleanupPhysics(false);
	if(!gMetrics.cleanupComplete && passed)
	{
		passed = false;
		reason = "cleanup_incomplete";
	}

	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetMimicJoint solver=%s case=%s "
		"execution=%s frames=%u completedFrames=%u status=%s reason=%s "
		"validation=GATED mimicMode=%s mimicCount=%u ratio=%.9g "
		"ratioReadback=%.9g offset=%.9g "
		"offsetReadback=%.9g axisA=twist axisB=x responseSamples=%u "
		"naturalFrequency=%.9g naturalFrequencyReadback=%.9g "
		"dampingRatio=%.9g dampingRatioReadback=%.9g "
		"directionSamples=%u directionViolations=%u rangeA=%.9g rangeB=%.9g "
		"positionErrorMax=%.9g positionErrorRms=%.9g "
		"velocityErrorMax=%.9g velocityErrorRms=%.9g "
		"finalQA=%.9g finalQB=%.9g finalVelocityA=%.9g finalVelocityB=%.9g "
		"finalPublicQA=%.9g finalPublicQB=%.9g "
		"maxPublicCoordinateError=%.9g "
		"secondRatio=%.9g secondOffset=%.9g secondResponseSamples=%u "
		"secondDirectionSamples=%u secondDirectionViolations=%u "
		"rangeC=%.9g rangeD=%.9g "
		"secondPositionErrorMax=%.9g secondPositionErrorRms=%.9g "
		"secondVelocityErrorMax=%.9g secondVelocityErrorRms=%.9g "
		"finalQC=%.9g finalQD=%.9g finalVelocityC=%.9g finalVelocityD=%.9g "
		"finalPublicQC=%.9g finalPublicQD=%.9g "
		"secondMaxPublicCoordinateError=%.9g "
		"nonFinite=%u fetchFailures=%u fatalErrors=%u cleanupComplete=%u pvd=0\n",
		Snippets::getSolverTypeName(gHeadlessOptions.solverType),
		getMimicCaseName(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gMetrics.completedFrames,
		passed ? "PASS" : "FAIL", reason,
		getMimicModeName(), mimicCountReadback,
		double(gHeadlessConfig.ratio), double(ratioReadback),
		double(gHeadlessConfig.offset), double(offsetReadback),
		gMetrics.responseSamples,
		double(gHeadlessConfig.naturalFrequency),
		double(naturalFrequencyReadback),
		double(gHeadlessConfig.dampingRatio), double(dampingRatioReadback),
		gMetrics.directionSamples,
		gMetrics.directionViolations, double(rangeA), double(rangeB),
		double(gMetrics.maxPositionError), double(positionRms),
		double(gMetrics.maxVelocityError), double(velocityRms),
		double(gMetrics.finalQA), double(gMetrics.finalQB),
		double(gMetrics.finalVelocityA), double(gMetrics.finalVelocityB),
		double(gMetrics.finalPublicQA), double(gMetrics.finalPublicQB),
		double(gMetrics.maxPublicCoordinateError),
		double(gSecondRatio), double(gSecondOffset),
		gMetrics.secondResponseSamples, gMetrics.secondDirectionSamples,
		gMetrics.secondDirectionViolations, double(rangeC), double(rangeD),
		double(gMetrics.secondMaxPositionError), double(secondPositionRms),
		double(gMetrics.secondMaxVelocityError), double(secondVelocityRms),
		double(gMetrics.finalQC), double(gMetrics.finalQD),
		double(gMetrics.finalVelocityC), double(gMetrics.finalVelocityD),
		double(gMetrics.finalPublicQC), double(gMetrics.finalPublicQD),
		double(gMetrics.secondMaxPublicCoordinateError),
		gMetrics.nonFinite, gMetrics.fetchFailures,
		gErrorCallback.getFatalCount(), gMetrics.cleanupComplete);
	return passed ? Snippets::eHEADLESS_PASS :
		Snippets::eHEADLESS_GATE_FAILED;
}

int snippetMain(int argc, const char* const* argv)
{
	std::string error;
	if(!parseHeadlessOptions(argc, argv, error))
	{
		std::fprintf(stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetMimicJoint reason=%s\n",
			error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
	{
		std::fprintf(stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetMimicJoint "
			"reason=execution_environment_failed\n");
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}

	if(gHeadlessOptions.headless)
		return runHeadless();

#ifdef RENDER_SNIPPET
	extern void renderLoop();
	renderLoop();
#else
	initPhysics(false);
	for(PxU32 i = 0; i < 100; ++i)
		stepPhysics(false);
	cleanupPhysics(false);
#endif
	return 0;
}
