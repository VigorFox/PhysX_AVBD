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

// This snippet demonstrates a fixed tendon that couples two articulation
// joint coordinates. Its headless path is the AVBD failure-first authority
// for fixed-tendon spring/damping routing.

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
static PxArticulationJointReducedCoordinate* gJointA = NULL;
static PxArticulationJointReducedCoordinate* gJointB = NULL;
static PxArticulationJointReducedCoordinate* gDriveJoint = NULL;
static PxArticulationFixedTendon* gTendon = NULL;
static PxArticulationTendonJoint* gTendonRoot = NULL;
static PxArticulationTendonJoint* gTendonJointA = NULL;
static PxArticulationTendonJoint* gTendonJointB = NULL;
static bool gExtensionsInitialized = false;

static const PxReal gTendonOffset = 0.2f;
static const PxReal gTendonStiffness = 5000.0f;
static const PxReal gTendonDamping = 100.0f;
static const PxReal gTendonLimitStiffness = 5000.0f;
static const PxReal gTendonLowLimit = -0.2f;
static const PxReal gTendonHighLimit = 0.2f;
static const PxReal gDriveMin = -0.5f;
static const PxReal gDriveMax = 0.5f;
static const PxReal gDriveSpeed = 1.2f;

enum FixedTendonCase
{
	eDRIVE_A,
	eDRIVE_B,
	eOFFSET_ACTUATION
};

enum FixedTendonMode
{
	eSERIAL_ANGULAR,
	eBRANCH_ANGULAR,
	eBRANCH_LINEAR,
	eLIMIT_ANGULAR
};

struct FixedTendonMetrics
{
	PxU32 completedFrames;
	PxU32 fetchFailures;
	PxU32 nonFinite;
	PxU32 responseSamples;
	PxU32 directionSamples;
	PxU32 directionViolations;
	PxReal maxLengthError;
	PxReal lengthErrorSquaredSum;
	PxReal maxLengthVelocity;
	PxReal lengthVelocitySquaredSum;
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
	PxReal minLength;
	PxReal maxLength;
	PxReal maxLimitViolation;
	PxU32 limitActiveSamples;
	PxU32 cleanupComplete;

	FixedTendonMetrics()
	: completedFrames(0), fetchFailures(0), nonFinite(0),
	  responseSamples(0), directionSamples(0), directionViolations(0),
	  maxLengthError(0.0f), lengthErrorSquaredSum(0.0f),
	  maxLengthVelocity(0.0f), lengthVelocitySquaredSum(0.0f),
	  minQA(PX_MAX_F32), maxQA(-PX_MAX_F32), minQB(PX_MAX_F32),
	  maxQB(-PX_MAX_F32), finalQA(0.0f), finalQB(0.0f),
	  finalVelocityA(0.0f), finalVelocityB(0.0f),
	  finalPublicQA(0.0f), finalPublicQB(0.0f),
	  maxPublicCoordinateError(0.0f), minLength(PX_MAX_F32),
	  maxLength(-PX_MAX_F32), maxLimitViolation(0.0f),
	  limitActiveSamples(0), cleanupComplete(0)
	{
	}
};

static Snippets::HeadlessOptions gHeadlessOptions;
static FixedTendonCase gHeadlessCase = eDRIVE_A;
static FixedTendonMode gHeadlessMode = eSERIAL_ANGULAR;
static FixedTendonMetrics gMetrics;
static PxReal gDriveTarget = 0.0f;
static PxReal gDriveDirection = 1.0f;
static PxReal gCurrentTendonOffset = gTendonOffset;
static PxArticulationAxis::Enum gAxis = PxArticulationAxis::eSWING2;

static const char* getCaseName()
{
	switch(gHeadlessCase)
	{
	case eDRIVE_A:
		return "drive-a";
	case eDRIVE_B:
		return "drive-b";
	case eOFFSET_ACTUATION:
		return "offset-actuation";
	}
	return "unknown";
}

static const char* getModeName()
{
	switch(gHeadlessMode)
	{
	case eSERIAL_ANGULAR:
		return "serial-angular";
	case eBRANCH_ANGULAR:
		return "branch-angular";
	case eBRANCH_LINEAR:
		return "branch-linear";
	case eLIMIT_ANGULAR:
		return "branch-limit-angular";
	}
	return "unknown";
}

static const char* getTopologyName()
{
	return gHeadlessMode == eBRANCH_ANGULAR ||
		gHeadlessMode == eBRANCH_LINEAR ||
		gHeadlessMode == eLIMIT_ANGULAR ? "branch" : "serial";
}

static const char* getAxisName()
{
	return gAxis == PxArticulationAxis::eX ? "x" : "swing2";
}

static bool parseCase(const char* value, FixedTendonCase& result)
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
	if(Snippets::equalsIgnoreCase(value, "offset-actuation"))
	{
		result = eOFFSET_ACTUATION;
		return true;
	}
	return false;
}

static bool parseMode(const char* value, FixedTendonMode& result)
{
	if(Snippets::equalsIgnoreCase(value, "serial-angular"))
	{
		result = eSERIAL_ANGULAR;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "branch-angular"))
	{
		result = eBRANCH_ANGULAR;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "branch-linear"))
	{
		result = eBRANCH_LINEAR;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "branch-limit-angular"))
	{
		result = eLIMIT_ANGULAR;
		return true;
	}
	return false;
}

static bool parseHeadlessOptions(
	int argc, const char* const* argv, std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 480;
	defaults.caseName = "drive-a";
	defaults.solverType = PxSolverType::eAVBD;
	if(!Snippets::parseCommonHeadlessOptions(
		argc, argv, defaults, gHeadlessOptions, error))
		return false;

	bool modeSeen = false;
	for(int i = 1; i < argc; ++i)
	{
		const char* arg = argv[i];
		if(Snippets::isCommonHeadlessOption(arg))
			continue;
		if(Snippets::hasOptionPrefix(arg, "--tendon-mode="))
		{
			if(modeSeen || !parseMode(
				arg + std::strlen("--tendon-mode="), gHeadlessMode))
			{
				error = modeSeen ? "duplicate --tendon-mode" :
					"invalid --tendon-mode";
				return false;
			}
			modeSeen = true;
			continue;
		}
		else
		{
			error = std::string("unknown option: ") +
				(arg ? arg : "<null>");
			return false;
		}
	}
	if(!parseCase(gHeadlessOptions.caseName.c_str(), gHeadlessCase))
	{
		error =
			"unsupported --case (expected drive-a, drive-b or offset-actuation)";
		return false;
	}
	if(gHeadlessCase == eOFFSET_ACTUATION &&
		gHeadlessMode != eSERIAL_ANGULAR)
	{
		error = "offset-actuation requires serial-angular mode";
		return false;
	}
	if(gHeadlessOptions.frames < 240)
	{
		error = "--frames must be at least 240";
		return false;
	}
	return true;
}

static PxArticulationReducedCoordinate* createArticulation()
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
	gLinkA = articulation->createLink(
		gRootLink, PxTransform(PxIdentity));
	gLinkB = articulation->createLink(
		gHeadlessMode == eBRANCH_ANGULAR ||
			gHeadlessMode == eBRANCH_LINEAR ||
			gHeadlessMode == eLIMIT_ANGULAR ? gRootLink : gLinkA,
		PxTransform(PxIdentity));
	if(!gLinkA || !gLinkB)
		return articulation;

	PxArticulationLink* links[3] = {gRootLink, gLinkA, gLinkB};
	for(PxU32 i = 0; i < 3; ++i)
	{
		links[i]->setCfmScale(0.0f);
		links[i]->setLinearDamping(0.0f);
		links[i]->setAngularDamping(0.0f);
		PxShape* shape = gPhysics->createShape(
			PxBoxGeometry(PxVec3(0.25f)), *gMaterial, true);
		if(shape)
		{
			links[i]->attachShape(*shape);
			shape->release();
		}
		if(i != 0)
			PxRigidBodyExt::updateMassAndInertia(*links[i], 1.0f);
	}

	gJointA = gLinkA->getInboundJoint();
	gJointB = gLinkB->getInboundJoint();
	if(!gJointA || !gJointB)
		return articulation;
	PxArticulationJointReducedCoordinate* joints[2] = {gJointA, gJointB};
	gAxis = gHeadlessMode == eBRANCH_LINEAR ?
		PxArticulationAxis::eX : PxArticulationAxis::eSWING2;
	for(PxU32 i = 0; i < 2; ++i)
	{
		joints[i]->setJointType(
			gHeadlessMode == eBRANCH_LINEAR ?
				PxArticulationJointType::ePRISMATIC :
				PxArticulationJointType::eREVOLUTE);
		joints[i]->setMotion(gAxis, PxArticulationMotion::eFREE);
		joints[i]->setParentPose(PxTransform(PxIdentity));
		joints[i]->setChildPose(PxTransform(PxIdentity));
		joints[i]->setFrictionCoefficient(0.0f);
	}

	gDriveJoint =
		gHeadlessCase == eDRIVE_B ? gJointB : gJointA;
	if(gHeadlessCase == eDRIVE_A || gHeadlessCase == eDRIVE_B)
	{
		gDriveJoint->setDriveParams(
			gAxis, PxArticulationDrive(10000.0f, 200.0f, PX_MAX_F32));
	}

	gTendon = articulation->createFixedTendon();
	if(!gTendon)
		return articulation;
	const bool limitMode = gHeadlessMode == eLIMIT_ANGULAR;
	gTendon->setLimitStiffness(
		limitMode ? gTendonLimitStiffness : 0.0f);
	gTendon->setStiffness(limitMode ? 0.0f : gTendonStiffness);
	gTendon->setDamping(limitMode ? 0.0f : gTendonDamping);
	gTendon->setRestLength(0.0f);
	gCurrentTendonOffset = limitMode ? 0.0f : gTendonOffset;
	gTendon->setOffset(gCurrentTendonOffset);
	if(limitMode)
	{
		PxArticulationTendonLimit limits;
		limits.lowLimit = gTendonLowLimit;
		limits.highLimit = gTendonHighLimit;
		gTendon->setLimitParameters(limits);
	}
	gTendonRoot = gTendon->createTendonJoint(
		NULL, gAxis, 1.0f, 1.0f, gRootLink);
	gTendonJointA = gTendon->createTendonJoint(
		gTendonRoot, gAxis, 1.0f, 1.0f, gLinkA);
	gTendonJointB = gTendon->createTendonJoint(
		gHeadlessMode == eBRANCH_ANGULAR ||
			gHeadlessMode == eBRANCH_LINEAR ||
			gHeadlessMode == eLIMIT_ANGULAR ?
			gTendonRoot : gTendonJointA,
		gAxis, -1.0f, -1.0f, gLinkB);
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
			gPvd->connect(
				*gPvdTransport, PxPvdInstrumentationFlag::eALL);
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
	gDispatcher = PxDefaultCpuDispatcherCreate(
		interactive ? 2u : gHeadlessOptions.dispatcherThreads);
	sceneDesc.cpuDispatcher = gDispatcher;
	sceneDesc.filterShader = PxDefaultSimulationFilterShader;
	sceneDesc.solverType =
		interactive ? PxSolverType::eAVBD : gHeadlessOptions.solverType;
	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return;

	gArticulation = createArticulation();
	if(gArticulation && gTendonJointA && gTendonJointB)
		gScene->addArticulation(*gArticulation);
}

static PxReal getJointCoordinate(
	const PxArticulationLink& parent,
	const PxArticulationLink& child,
	const PxArticulationJointReducedCoordinate& joint)
{
	const PxTransform parentFrame =
		parent.getGlobalPose() * joint.getParentPose();
	const PxTransform childFrame =
		child.getGlobalPose() * joint.getChildPose();
	if(gAxis == PxArticulationAxis::eX)
		return (childFrame.p - parentFrame.p).dot(
			parentFrame.q.getBasisVector0());
	PxQuat relative = parentFrame.q.getConjugate() * childFrame.q;
	if(relative.w < 0.0f)
		relative = -relative;
	return 2.0f * PxAtan2(relative.z, relative.w);
}

static PxReal getJointVelocity(
	const PxArticulationLink& parent,
	const PxArticulationLink& child,
	const PxArticulationJointReducedCoordinate& joint)
{
	const PxTransform parentFrame =
		parent.getGlobalPose() * joint.getParentPose();
	if(gAxis == PxArticulationAxis::eX)
		return (child.getLinearVelocity() -
			parent.getLinearVelocity()).dot(parentFrame.q.getBasisVector0());
	const PxVec3 axis = parentFrame.q.getBasisVector2();
	return (child.getAngularVelocity() - parent.getAngularVelocity()).dot(axis);
}

static void recordHeadlessState(PxU32 frame)
{
	if(!gRootLink || !gLinkA || !gLinkB || !gJointA || !gJointB)
		return;
	const PxReal qA =
		getJointCoordinate(*gRootLink, *gLinkA, *gJointA);
	const PxArticulationLink& parentB =
		gHeadlessMode == eBRANCH_ANGULAR ||
			gHeadlessMode == eBRANCH_LINEAR ||
			gHeadlessMode == eLIMIT_ANGULAR ? *gRootLink : *gLinkA;
	const PxReal qB =
		getJointCoordinate(parentB, *gLinkB, *gJointB);
	const PxReal vA =
		getJointVelocity(*gRootLink, *gLinkA, *gJointA);
	const PxReal vB =
		getJointVelocity(parentB, *gLinkB, *gJointB);
	const PxReal publicQA = gJointA->getJointPosition(gAxis);
	const PxReal publicQB = gJointB->getJointPosition(gAxis);
	if(!PxIsFinite(qA) || !PxIsFinite(qB) ||
		!PxIsFinite(vA) || !PxIsFinite(vB) ||
		!PxIsFinite(publicQA) || !PxIsFinite(publicQB) ||
		!gRootLink->getGlobalPose().isFinite() ||
		!gLinkA->getGlobalPose().isFinite() ||
		!gLinkB->getGlobalPose().isFinite())
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
	if(frame < 60)
		return;

	const PxReal lengthError = qA - qB + gCurrentTendonOffset;
	const PxReal lengthVelocity = vA - vB;
	const PxReal length = qA - qB + gCurrentTendonOffset;
	gMetrics.maxLengthError =
		PxMax(gMetrics.maxLengthError, PxAbs(lengthError));
	gMetrics.lengthErrorSquaredSum += lengthError * lengthError;
	gMetrics.maxLengthVelocity =
		PxMax(gMetrics.maxLengthVelocity, PxAbs(lengthVelocity));
	gMetrics.lengthVelocitySquaredSum +=
		lengthVelocity * lengthVelocity;
	gMetrics.minQA = PxMin(gMetrics.minQA, qA);
	gMetrics.maxQA = PxMax(gMetrics.maxQA, qA);
	gMetrics.minQB = PxMin(gMetrics.minQB, qB);
	gMetrics.maxQB = PxMax(gMetrics.maxQB, qB);
	gMetrics.minLength = PxMin(gMetrics.minLength, length);
	gMetrics.maxLength = PxMax(gMetrics.maxLength, length);
	if(gHeadlessMode == eLIMIT_ANGULAR)
	{
		const PxReal violation =
			length < gTendonLowLimit ? gTendonLowLimit - length :
			(length > gTendonHighLimit ? length - gTendonHighLimit : 0.0f);
		gMetrics.maxLimitViolation =
			PxMax(gMetrics.maxLimitViolation, violation);
		if(PxAbs(length) > 0.15f)
			++gMetrics.limitActiveSamples;
	}
	++gMetrics.responseSamples;

	if(PxAbs(vA) > 0.02f && PxAbs(vB) > 0.02f)
	{
		++gMetrics.directionSamples;
		const bool wrongDirection =
			gHeadlessCase == eOFFSET_ACTUATION ?
				vA * vB >= 0.0f : vA * vB <= 0.0f;
		if(wrongDirection)
			++gMetrics.directionViolations;
	}
}

void stepPhysics(bool interactive)
{
	const PxReal dt = interactive ? 1.0f / 60.0f : gHeadlessOptions.dt;
	if(gHeadlessCase == eDRIVE_A || gHeadlessCase == eDRIVE_B)
	{
		gDriveTarget += gDriveSpeed * gDriveDirection * dt;
		if(gDriveTarget > gDriveMax)
		{
			gDriveTarget = gDriveMax;
			gDriveDirection = -1.0f;
		}
		else if(gDriveTarget < gDriveMin)
		{
			gDriveTarget = gDriveMin;
			gDriveDirection = 1.0f;
		}
		gDriveJoint->setDriveTarget(gAxis, gDriveTarget);
	}
	else
	{
		const PxReal phase =
			PxTwoPi * PxReal(gMetrics.completedFrames) / 240.0f;
		gCurrentTendonOffset =
			gTendonOffset + 0.15f * PxSin(phase);
		gTendon->setOffset(gCurrentTendonOffset);
	}
	gScene->simulate(dt);
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
	gTendon = NULL;
	gTendonRoot = NULL;
	gTendonJointA = NULL;
	gTendonJointB = NULL;
	gDriveJoint = NULL;
	gJointA = NULL;
	gJointB = NULL;
	gRootLink = NULL;
	gLinkA = NULL;
	gLinkB = NULL;
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
		std::printf("SnippetFixedTendon done.\n");
}

static int runHeadless()
{
	Snippets::printHeadlessConfig("SnippetFixedTendon", gHeadlessOptions);
	initPhysics(false);
	const bool initialized =
		gFoundation && gPhysics && gDispatcher && gScene && gMaterial &&
		gArticulation && gRootLink && gLinkA && gLinkB && gJointA && gJointB &&
		gDriveJoint && gTendon && gTendonRoot && gTendonJointA &&
		gTendonJointB;

	if(initialized)
	{
		for(PxU32 frame = 0; frame < gHeadlessOptions.frames; ++frame)
		{
			stepPhysics(false);
			if(gMetrics.fetchFailures || gMetrics.nonFinite)
				break;
		}
	}

	const PxReal lengthRms =
		gMetrics.responseSamples
			? PxSqrt(gMetrics.lengthErrorSquaredSum /
				PxReal(gMetrics.responseSamples))
			: PX_MAX_F32;
	const PxReal lengthVelocityRms =
		gMetrics.responseSamples
			? PxSqrt(gMetrics.lengthVelocitySquaredSum /
				PxReal(gMetrics.responseSamples))
			: PX_MAX_F32;
	const PxReal rangeA = gMetrics.maxQA - gMetrics.minQA;
	const PxReal rangeB = gMetrics.maxQB - gMetrics.minQB;

	PxArticulationAxis::Enum axisA = PxArticulationAxis::eCOUNT;
	PxArticulationAxis::Enum axisB = PxArticulationAxis::eCOUNT;
	PxReal coefficientA = 0.0f, reciprocalA = 0.0f;
	PxReal coefficientB = 0.0f, reciprocalB = 0.0f;
	if(gTendonJointA)
		gTendonJointA->getCoefficient(axisA, coefficientA, reciprocalA);
	if(gTendonJointB)
		gTendonJointB->getCoefficient(axisB, coefficientB, reciprocalB);
	const bool limitMode = gHeadlessMode == eLIMIT_ANGULAR;
	const PxReal expectedStiffness =
		limitMode ? 0.0f : gTendonStiffness;
	const PxReal expectedLimitStiffness =
		limitMode ? gTendonLimitStiffness : 0.0f;
	const PxReal expectedDamping =
		limitMode ? 0.0f : gTendonDamping;
	PxArticulationTendonLimit limitReadback;
	limitReadback.lowLimit = 0.0f;
	limitReadback.highLimit = 0.0f;
	if(gTendon)
		limitReadback = gTendon->getLimitParameters();
	const bool topologyReadback =
		gTendonRoot && !gTendonRoot->getParent() &&
		gTendonJointA && gTendonJointA->getParent() == gTendonRoot &&
		gTendonJointB &&
		gTendonJointB->getParent() ==
			(gHeadlessMode == eBRANCH_ANGULAR ||
				gHeadlessMode == eBRANCH_LINEAR ||
				gHeadlessMode == eLIMIT_ANGULAR ?
				gTendonRoot : gTendonJointA);
	const bool limitReadbackValid =
		limitMode ?
			PxAbs(limitReadback.lowLimit - gTendonLowLimit) < 1e-6f &&
			PxAbs(limitReadback.highLimit - gTendonHighLimit) < 1e-6f :
			limitReadback.lowLimit == PX_MAX_F32 &&
			limitReadback.highLimit == -PX_MAX_F32;
	const bool readbackValid =
		gTendon && gTendon->getNbTendonJoints() == 3 &&
		PxAbs(gTendon->getStiffness() - expectedStiffness) < 1e-4f &&
		PxAbs(gTendon->getDamping() - expectedDamping) < 1e-4f &&
		PxAbs(gTendon->getLimitStiffness() -
			expectedLimitStiffness) < 1e-4f &&
		PxAbs(gTendon->getOffset() - gCurrentTendonOffset) < 1e-6f &&
		PxAbs(gTendon->getRestLength()) < 1e-6f &&
		topologyReadback && limitReadbackValid &&
		axisA == gAxis && axisB == gAxis &&
		PxAbs(coefficientA - 1.0f) < 1e-6f &&
		PxAbs(reciprocalA - 1.0f) < 1e-6f &&
		PxAbs(coefficientB + 1.0f) < 1e-6f &&
		PxAbs(reciprocalB + 1.0f) < 1e-6f;

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
	else if(rangeA < (gHeadlessCase == eOFFSET_ACTUATION ? 0.04f : 0.15f) ||
		rangeB < (gHeadlessCase == eOFFSET_ACTUATION ? 0.04f : 0.15f))
	{
		passed = false;
		reason = "missing_coupled_response";
	}
	else if(!limitMode && (gMetrics.directionSamples < 30 ||
		gMetrics.directionViolations * 10 > gMetrics.directionSamples))
	{
		passed = false;
		reason = "wrong_coupling_direction";
	}
	else if(limitMode && (gMetrics.limitActiveSamples < 20 ||
		gMetrics.maxLimitViolation > 0.1f))
	{
		passed = false;
		reason = "tendon_limit_error";
	}
	else if(!limitMode &&
		(lengthRms > 0.08f || gMetrics.maxLengthError > 0.22f))
	{
		passed = false;
		reason = "tendon_length_error";
	}
	else if(!limitMode && (lengthVelocityRms > 0.4f ||
		gMetrics.maxLengthVelocity > 2.0f)
	)
	{
		passed = false;
		reason = "tendon_velocity_error";
	}
	else if(gMetrics.maxPublicCoordinateError > 0.002f)
	{
		passed = false;
		reason = "joint_state_mismatch";
	}

	cleanupPhysics(false);
	if(!gMetrics.cleanupComplete && passed)
	{
		passed = false;
		reason = "cleanup_incomplete";
	}

	std::printf(
		"[AVBD_GATE] schema=1 snippet=SnippetFixedTendon tendonMode=%s "
		"topology=%s axis=%s tendonJointCount=3 solver=%s case=%s "
		"execution=%s frames=%u completedFrames=%u status=%s reason=%s "
		"validation=GATED coefficientA=1 reciprocalA=1 "
		"coefficientB=-1 reciprocalB=-1 "
		"offsetBase=%.9g offsetFinal=%.9g stiffness=%.9g damping=%.9g "
		"limitStiffness=%.9g lowLimit=%.9g highLimit=%.9g "
		"responseSamples=%u "
		"directionSamples=%u directionViolations=%u rangeA=%.9g rangeB=%.9g "
		"lengthErrorMax=%.9g lengthErrorRms=%.9g "
		"lengthVelocityMax=%.9g lengthVelocityRms=%.9g "
		"minLength=%.9g maxLength=%.9g maxLimitViolation=%.9g "
		"limitActiveSamples=%u "
		"finalQA=%.9g finalQB=%.9g finalVelocityA=%.9g finalVelocityB=%.9g "
		"finalPublicQA=%.9g finalPublicQB=%.9g "
		"maxPublicCoordinateError=%.9g nonFinite=%u fetchFailures=%u "
		"fatalErrors=%u cleanupComplete=%u pvd=0\n",
		getModeName(), getTopologyName(), getAxisName(),
		Snippets::getSolverTypeName(gHeadlessOptions.solverType), getCaseName(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gMetrics.completedFrames,
		passed ? "PASS" : "FAIL", reason,
		double(gHeadlessMode == eLIMIT_ANGULAR ? 0.0f : gTendonOffset),
		double(gCurrentTendonOffset),
		double(expectedStiffness), double(expectedDamping),
		double(expectedLimitStiffness), double(limitReadback.lowLimit),
		double(limitReadback.highLimit),
		gMetrics.responseSamples, gMetrics.directionSamples,
		gMetrics.directionViolations, double(rangeA), double(rangeB),
		double(gMetrics.maxLengthError), double(lengthRms),
		double(gMetrics.maxLengthVelocity), double(lengthVelocityRms),
		double(gMetrics.minLength), double(gMetrics.maxLength),
		double(gMetrics.maxLimitViolation), gMetrics.limitActiveSamples,
		double(gMetrics.finalQA), double(gMetrics.finalQB),
		double(gMetrics.finalVelocityA), double(gMetrics.finalVelocityB),
		double(gMetrics.finalPublicQA), double(gMetrics.finalPublicQB),
		double(gMetrics.maxPublicCoordinateError), gMetrics.nonFinite,
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
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetFixedTendon reason=%s\n",
			error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
	{
		std::fprintf(stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetFixedTendon "
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
