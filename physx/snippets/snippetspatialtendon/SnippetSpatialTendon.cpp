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
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2026 NVIDIA Corporation. All rights reserved.

// This snippet demonstrates a spatial tendon whose path passes from a
// moving endpoint through a fixed-base attachment to another moving endpoint.
// Its headless path is the AVBD failure-first authority for spatial-tendon
// attachment geometry and compliant response.

#include "PxPhysicsAPI.h"
#include "../snippetcommon/SnippetHeadless.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetcommon/SnippetPrint.h"
#include "../snippetutils/SnippetUtils.h"

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
static PxArticulationLink* gMiddleLink = NULL;
static PxArticulationJointReducedCoordinate* gJointA = NULL;
static PxArticulationJointReducedCoordinate* gJointB = NULL;
static PxArticulationJointReducedCoordinate* gJointC = NULL;
static PxArticulationJointReducedCoordinate* gMiddleJoint = NULL;
static PxArticulationSpatialTendon* gTendon = NULL;
static PxArticulationAttachment* gRootAttachment = NULL;
static PxArticulationAttachment* gMiddleAttachment = NULL;
static PxArticulationAttachment* gLeafAttachment = NULL;
static PxArticulationAttachment* gSecondLeafAttachment = NULL;
static bool gExtensionsInitialized = false;

static PxArticulationAxis::Enum gAxis = PxArticulationAxis::eSWING2;
static const PxVec3 gLocalAttachmentA(-1.0f, 0.0f, 0.0f);
static const PxVec3 gLocalAttachmentB(1.0f, 0.0f, 0.0f);
static const PxVec3 gLocalAttachmentC(1.0f, 0.0f, 0.0f);
static const PxReal gMiddleHeight = 1.0f;
static const PxReal gOffsetBase = 0.2f;
static const PxReal gRestLength =
	2.0f * 1.4142135623730950488f + gOffsetBase;
static const PxReal gTendonStiffness = 5000.0f;
static const PxReal gTendonDamping = 100.0f;
static const PxReal gLimitStiffness = 5000.0f;
static const PxReal gLimitLow = gRestLength - 0.2f;
static const PxReal gLimitHigh = gRestLength + 0.2f;

enum SpatialTendonCase
{
	eOFFSET_ACTUATION,
	eATTACHMENT_ACTUATION,
	eMIDDLE_DRIVE
};

enum SpatialTendonMode
{
	eFIXED_MIDDLE,
	eMOVING_MIDDLE,
	eMULTI_LEAF,
	eLINEAR_AXIS,
	eLIMIT
};

struct SpatialTendonMetrics
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
	PxReal minQC;
	PxReal maxQC;
	PxReal minQM;
	PxReal maxQM;
	PxReal maxSecondLengthError;
	PxReal secondLengthErrorSquaredSum;
	PxReal maxLimitViolation;
	PxU32 limitActiveSamples;
	PxReal finalQA;
	PxReal finalQB;
	PxReal finalQC;
	PxReal finalQM;
	PxReal finalVelocityA;
	PxReal finalVelocityB;
	PxReal finalPublicQA;
	PxReal finalPublicQB;
	PxReal maxPublicCoordinateError;
	PxU32 cleanupComplete;

	SpatialTendonMetrics()
	: completedFrames(0), fetchFailures(0), nonFinite(0),
	  responseSamples(0), directionSamples(0), directionViolations(0),
	  maxLengthError(0.0f), lengthErrorSquaredSum(0.0f),
	  maxLengthVelocity(0.0f), lengthVelocitySquaredSum(0.0f),
	  minQA(PX_MAX_F32), maxQA(-PX_MAX_F32), minQB(PX_MAX_F32),
	  maxQB(-PX_MAX_F32), minQC(PX_MAX_F32), maxQC(-PX_MAX_F32),
	  minQM(PX_MAX_F32), maxQM(-PX_MAX_F32),
	  maxSecondLengthError(0.0f), secondLengthErrorSquaredSum(0.0f),
	  maxLimitViolation(0.0f), limitActiveSamples(0),
	  finalQA(0.0f), finalQB(0.0f), finalQC(0.0f), finalQM(0.0f),
	  finalVelocityA(0.0f), finalVelocityB(0.0f),
	  finalPublicQA(0.0f), finalPublicQB(0.0f),
	  maxPublicCoordinateError(0.0f), cleanupComplete(0)
	{
	}
};

static Snippets::HeadlessOptions gHeadlessOptions;
static SpatialTendonCase gHeadlessCase = eOFFSET_ACTUATION;
static SpatialTendonMode gHeadlessMode = eFIXED_MIDDLE;
static SpatialTendonMetrics gMetrics;
static PxReal gCurrentOffset = gOffsetBase;
static PxReal gCurrentMiddleHeight = gMiddleHeight;
static PxReal gMiddleDriveTarget = 0.0f;
static PxReal gMiddleDriveDirection = 1.0f;

PxRigidStatic** getAttachments()
{
	static PxRigidStatic* attachments[6] = {NULL};
	return attachments;
}

static const char* getCaseName()
{
	switch(gHeadlessCase)
	{
	case eOFFSET_ACTUATION:
		return "offset-actuation";
	case eATTACHMENT_ACTUATION:
		return "attachment-actuation";
	case eMIDDLE_DRIVE:
		return "middle-drive";
	}
	return "unknown";
}

static const char* getModeName()
{
	switch(gHeadlessMode)
	{
	case eFIXED_MIDDLE:
		return "fixed-middle";
	case eMOVING_MIDDLE:
		return "moving-middle";
	case eMULTI_LEAF:
		return "multi-leaf";
	case eLINEAR_AXIS:
		return "linear-axis";
	case eLIMIT:
		return "limit";
	}
	return "unknown";
}

static const char* getAxisName()
{
	return gAxis == PxArticulationAxis::eX ? "x" : "swing2";
}

static bool parseCase(const char* value, SpatialTendonCase& result)
{
	if(Snippets::equalsIgnoreCase(value, "offset-actuation"))
	{
		result = eOFFSET_ACTUATION;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "attachment-actuation"))
	{
		result = eATTACHMENT_ACTUATION;
		return true;
	}
	if(Snippets::equalsIgnoreCase(value, "middle-drive"))
	{
		result = eMIDDLE_DRIVE;
		return true;
	}
	return false;
}

static bool parseMode(const char* value, SpatialTendonMode& result)
{
	if(Snippets::equalsIgnoreCase(value, "fixed-middle"))
		result = eFIXED_MIDDLE;
	else if(Snippets::equalsIgnoreCase(value, "moving-middle"))
		result = eMOVING_MIDDLE;
	else if(Snippets::equalsIgnoreCase(value, "multi-leaf"))
		result = eMULTI_LEAF;
	else if(Snippets::equalsIgnoreCase(value, "linear-axis"))
		result = eLINEAR_AXIS;
	else if(Snippets::equalsIgnoreCase(value, "limit"))
		result = eLIMIT;
	else
		return false;
	return true;
}

static bool parseHeadlessOptions(
	int argc, const char* const* argv, std::string& error)
{
	Snippets::HeadlessOptions defaults;
	defaults.frames = 480;
	defaults.caseName = "offset-actuation";
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
		if(Snippets::hasOptionPrefix(arg, "--spatial-mode="))
		{
			if(modeSeen || !parseMode(
				arg + std::strlen("--spatial-mode="), gHeadlessMode))
			{
				error = modeSeen ? "duplicate --spatial-mode" :
					"invalid --spatial-mode";
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
		error = "unsupported --case (expected offset-actuation, "
			"attachment-actuation or middle-drive)";
		return false;
	}
	if(gHeadlessMode == eMOVING_MIDDLE &&
		gHeadlessCase != eMIDDLE_DRIVE)
	{
		error = "moving-middle mode requires middle-drive";
		return false;
	}
	if(gHeadlessMode != eMOVING_MIDDLE &&
		gHeadlessCase == eMIDDLE_DRIVE)
	{
		error = "middle-drive requires moving-middle mode";
		return false;
	}
	if(gHeadlessMode != eFIXED_MIDDLE &&
		gHeadlessMode != eMOVING_MIDDLE &&
		gHeadlessCase != eOFFSET_ACTUATION)
	{
		error = "multi-leaf, linear-axis and limit modes require "
			"offset-actuation";
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
	gLinkA = articulation->createLink(gRootLink, PxTransform(PxIdentity));
	gLinkB = articulation->createLink(gRootLink, PxTransform(PxIdentity));
	if(gHeadlessMode == eMOVING_MIDDLE)
		gMiddleLink =
			articulation->createLink(gRootLink, PxTransform(PxIdentity));
	if(gHeadlessMode == eMULTI_LEAF)
		gLinkC = articulation->createLink(
			gRootLink, PxTransform(PxIdentity));
	if(!gRootLink || !gLinkA || !gLinkB)
		return articulation;

	PxArticulationLink* links[5] =
		{gRootLink, gLinkA, gLinkB, gLinkC, gMiddleLink};
	for(PxU32 i = 0; i < 5; ++i)
	{
		if(!links[i])
			continue;
		links[i]->setCfmScale(0.0f);
		links[i]->setLinearDamping(0.0f);
		links[i]->setAngularDamping(0.0f);
		PxShape* shape = gPhysics->createShape(
			PxBoxGeometry(PxVec3(0.2f, 0.08f, 0.08f)),
			*gMaterial, true);
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
	gJointC = gLinkC ? gLinkC->getInboundJoint() : NULL;
	gMiddleJoint =
		gMiddleLink ? gMiddleLink->getInboundJoint() : NULL;
	if(!gJointA || !gJointB)
		return articulation;
	PxArticulationJointReducedCoordinate* joints[4] =
		{gJointA, gJointB, gJointC, gMiddleJoint};
	gAxis = gHeadlessMode == eLINEAR_AXIS ?
		PxArticulationAxis::eX : PxArticulationAxis::eSWING2;
	for(PxU32 i = 0; i < 4; ++i)
	{
		if(!joints[i])
			continue;
		joints[i]->setJointType(
			gHeadlessMode == eLINEAR_AXIS ?
				PxArticulationJointType::ePRISMATIC :
				PxArticulationJointType::eREVOLUTE);
		joints[i]->setMotion(gAxis, PxArticulationMotion::eFREE);
		joints[i]->setParentPose(PxTransform(PxIdentity));
		joints[i]->setChildPose(PxTransform(PxIdentity));
		joints[i]->setFrictionCoefficient(0.0f);
	}
	if(gMiddleJoint)
		gMiddleJoint->setDriveParams(
			gAxis, PxArticulationDrive(
				10000.0f, 200.0f, PX_MAX_F32));

	gTendon = articulation->createSpatialTendon();
	if(!gTendon)
		return articulation;
	const bool limitMode = gHeadlessMode == eLIMIT;
	gTendon->setStiffness(limitMode ? 0.0f : gTendonStiffness);
	gTendon->setDamping(limitMode ? 0.0f : gTendonDamping);
	gTendon->setLimitStiffness(limitMode ? gLimitStiffness : 0.0f);
	gTendon->setOffset(gOffsetBase);
	gRootAttachment = gTendon->createAttachment(
		NULL, 1.0f, gLocalAttachmentA, gLinkA);
	gMiddleAttachment = gTendon->createAttachment(
		gRootAttachment, 1.0f,
		PxVec3(0.0f, gMiddleHeight, 0.0f),
		gMiddleLink ? gMiddleLink : gRootLink);
	gLeafAttachment = gTendon->createAttachment(
		gMiddleAttachment, 1.0f, gLocalAttachmentB, gLinkB);
	if(gLeafAttachment)
	{
		gLeafAttachment->setRestLength(gRestLength);
		if(limitMode)
		{
			PxArticulationTendonLimit limits;
			limits.lowLimit = gLimitLow;
			limits.highLimit = gLimitHigh;
			gLeafAttachment->setLimitParameters(limits);
		}
	}
	if(gLinkC)
	{
		gSecondLeafAttachment = gTendon->createAttachment(
			gMiddleAttachment, 1.0f, gLocalAttachmentC, gLinkC);
		if(gSecondLeafAttachment)
			gSecondLeafAttachment->setRestLength(gRestLength);
	}
	return articulation;
}

static void createVisualizationAttachments()
{
	PxRigidStatic** attachments = getAttachments();
	for(PxU32 i = 0; i < 6; ++i)
	{
		attachments[i] =
			gPhysics->createRigidStatic(PxTransform(PxIdentity));
		if(!attachments[i])
			continue;
		PxShape* shape = gPhysics->createShape(
			PxSphereGeometry(0.08f), *gMaterial, false, PxShapeFlags(0));
		if(shape)
		{
			attachments[i]->setActorFlag(
				PxActorFlag::eDISABLE_SIMULATION, true);
			attachments[i]->attachShape(*shape);
			shape->release();
		}
	}
}

static void updateVisualizationAttachments()
{
	if(!gRootLink || !gLinkA || !gLinkB)
		return;
	const PxTransform poseA =
		gLinkA->getGlobalPose() * PxTransform(gLocalAttachmentA);
	const PxTransform poseM =
		gRootLink->getGlobalPose() *
		PxTransform(PxVec3(0.0f, gCurrentMiddleHeight, 0.0f));
	const PxTransform poseB =
		gLinkB->getGlobalPose() * PxTransform(gLocalAttachmentB);
	PxRigidStatic** attachments = getAttachments();
	if(attachments[0])
		attachments[0]->setGlobalPose(poseM);
	if(attachments[1])
		attachments[1]->setGlobalPose(poseA);
	if(attachments[2])
		attachments[2]->setGlobalPose(poseB);
	if(attachments[3])
		attachments[3]->setGlobalPose(poseM);
	if(attachments[4])
		attachments[4]->setGlobalPose(poseA);
	if(attachments[5])
		attachments[5]->setGlobalPose(poseB);
}

void initPhysics(bool interactive)
{
	gErrorCallback.reset();
	gFoundation = PxCreateFoundation(
		PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
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
		interactive ? PxSolverType::eTGS : gHeadlessOptions.solverType;
	gScene = gPhysics->createScene(sceneDesc);
	if(!gScene)
		return;

	gArticulation = createArticulation();
	if(gArticulation)
		gScene->addArticulation(*gArticulation);
	if(interactive)
		createVisualizationAttachments();
	updateVisualizationAttachments();
}

static PxReal getJointCoordinate(
	const PxArticulationLink& parent, const PxArticulationLink& child,
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
	const PxArticulationLink& parent, const PxArticulationLink& child,
	const PxArticulationJointReducedCoordinate& joint)
{
	const PxTransform parentFrame =
		parent.getGlobalPose() * joint.getParentPose();
	const PxVec3 axis = gAxis == PxArticulationAxis::eX ?
		parentFrame.q.getBasisVector0() : parentFrame.q.getBasisVector2();
	if(gAxis == PxArticulationAxis::eX)
		return (child.getLinearVelocity() -
			parent.getLinearVelocity()).dot(axis);
	return (child.getAngularVelocity() -
		parent.getAngularVelocity()).dot(axis);
}

static PxReal getCoordinateReadbackError(
	PxReal publicCoordinate, PxReal measuredCoordinate)
{
	if(gAxis == PxArticulationAxis::eX)
		return PxAbs(publicCoordinate - measuredCoordinate);
	const PxReal difference = publicCoordinate - measuredCoordinate;
	return PxAbs(PxAtan2(PxSin(difference), PxCos(difference)));
}

static bool getTendonState(
	PxReal& length, PxReal& lengthVelocity,
	PxReal& secondLength, PxReal& secondLengthVelocity,
	PxVec3& pointA, PxVec3& pointM, PxVec3& pointB, PxVec3& pointC)
{
	if(!gRootLink || !gLinkA || !gLinkB)
		return false;
	const PxTransform poseA = gLinkA->getGlobalPose();
	const PxArticulationLink* middleLink =
		gMiddleLink ? gMiddleLink : gRootLink;
	const PxTransform poseM = middleLink->getGlobalPose();
	const PxTransform poseB = gLinkB->getGlobalPose();
	pointA = poseA.transform(gLocalAttachmentA);
	pointM = poseM.transform(
		PxVec3(0.0f, gCurrentMiddleHeight, 0.0f));
	pointB = poseB.transform(gLocalAttachmentB);
	pointC = gLinkC ?
		gLinkC->getGlobalPose().transform(gLocalAttachmentC) :
		PxVec3(0.0f);
	const PxVec3 segmentA = pointA - pointM;
	const PxVec3 segmentB = pointB - pointM;
	const PxReal distanceA = segmentA.magnitude();
	const PxReal distanceB = segmentB.magnitude();
	if(distanceA <= 1e-6f || distanceB <= 1e-6f)
		return false;
	const PxVec3 armA = pointA - poseA.p;
	const PxVec3 armM = pointM - poseM.p;
	const PxVec3 armB = pointB - poseB.p;
	const PxVec3 velocityA =
		gLinkA->getLinearVelocity() +
		gLinkA->getAngularVelocity().cross(armA);
	const PxVec3 velocityM =
		middleLink->getLinearVelocity() +
		middleLink->getAngularVelocity().cross(armM);
	const PxVec3 velocityB =
		gLinkB->getLinearVelocity() +
		gLinkB->getAngularVelocity().cross(armB);
	length = distanceA + distanceB + gCurrentOffset;
	lengthVelocity =
		segmentA.dot(velocityA - velocityM) / distanceA +
		segmentB.dot(velocityB - velocityM) / distanceB;
	secondLength = 0.0f;
	secondLengthVelocity = 0.0f;
	if(gLinkC)
	{
		const PxTransform poseC = gLinkC->getGlobalPose();
		const PxVec3 segmentC = pointC - pointM;
		const PxReal distanceC = segmentC.magnitude();
		if(distanceC <= 1e-6f)
			return false;
		const PxVec3 armC = pointC - poseC.p;
		const PxVec3 velocityC =
			gLinkC->getLinearVelocity() +
			gLinkC->getAngularVelocity().cross(armC);
		secondLength = distanceA + distanceC + gCurrentOffset;
		secondLengthVelocity =
			segmentA.dot(velocityA - velocityM) / distanceA +
			segmentC.dot(velocityC - velocityM) / distanceC;
	}
	return PxIsFinite(length) && PxIsFinite(lengthVelocity) &&
		PxIsFinite(secondLength) && PxIsFinite(secondLengthVelocity);
}

static void recordHeadlessState(PxU32 frame)
{
	PxReal length = 0.0f, lengthVelocity = 0.0f;
	PxReal secondLength = 0.0f, secondLengthVelocity = 0.0f;
	PxVec3 pointA(0.0f), pointM(0.0f), pointB(0.0f), pointC(0.0f);
	const PxReal qA =
		getJointCoordinate(*gRootLink, *gLinkA, *gJointA);
	const PxReal qB =
		getJointCoordinate(*gRootLink, *gLinkB, *gJointB);
	const PxReal qC = gLinkC && gJointC ?
		getJointCoordinate(*gRootLink, *gLinkC, *gJointC) : 0.0f;
	const PxReal qM = gMiddleLink && gMiddleJoint ?
		getJointCoordinate(*gRootLink, *gMiddleLink, *gMiddleJoint) : 0.0f;
	const PxReal vA =
		getJointVelocity(*gRootLink, *gLinkA, *gJointA);
	const PxReal vB =
		getJointVelocity(*gRootLink, *gLinkB, *gJointB);
	const PxReal publicQA = gJointA->getJointPosition(gAxis);
	const PxReal publicQB = gJointB->getJointPosition(gAxis);
	const PxReal publicQC =
		gJointC ? gJointC->getJointPosition(gAxis) : 0.0f;
	const PxReal publicQM =
		gMiddleJoint ? gMiddleJoint->getJointPosition(gAxis) : 0.0f;
	if(!getTendonState(
			length, lengthVelocity, secondLength, secondLengthVelocity,
			pointA, pointM, pointB, pointC) ||
		!PxIsFinite(qA) || !PxIsFinite(qB) ||
		!PxIsFinite(qC) || !PxIsFinite(qM) ||
		!PxIsFinite(vA) || !PxIsFinite(vB) ||
		!PxIsFinite(publicQA) || !PxIsFinite(publicQB) ||
		!PxIsFinite(publicQC) || !PxIsFinite(publicQM) ||
		!pointA.isFinite() || !pointM.isFinite() || !pointB.isFinite() ||
		(gLinkC && !pointC.isFinite()))
	{
		++gMetrics.nonFinite;
		return;
	}

	gMetrics.finalQA = qA;
	gMetrics.finalQB = qB;
	gMetrics.finalQC = qC;
	gMetrics.finalQM = qM;
	gMetrics.finalVelocityA = vA;
	gMetrics.finalVelocityB = vB;
	gMetrics.finalPublicQA = publicQA;
	gMetrics.finalPublicQB = publicQB;
	gMetrics.maxPublicCoordinateError = PxMax(
		gMetrics.maxPublicCoordinateError,
		PxMax(getCoordinateReadbackError(publicQA, qA),
			getCoordinateReadbackError(publicQB, qB)));
	gMetrics.maxPublicCoordinateError = PxMax(
		gMetrics.maxPublicCoordinateError,
		PxMax(getCoordinateReadbackError(publicQC, qC),
			getCoordinateReadbackError(publicQM, qM)));
	if(frame < 60)
		return;

	const PxReal lengthError = length - gRestLength;
	gMetrics.maxLengthError =
		PxMax(gMetrics.maxLengthError, PxAbs(lengthError));
	gMetrics.lengthErrorSquaredSum += lengthError * lengthError;
	gMetrics.maxLengthVelocity =
		PxMax(gMetrics.maxLengthVelocity, PxAbs(lengthVelocity));
	gMetrics.lengthVelocitySquaredSum +=
		lengthVelocity * lengthVelocity;
	if(gLinkC)
	{
		const PxReal secondError = secondLength - gRestLength;
		gMetrics.maxSecondLengthError = PxMax(
			gMetrics.maxSecondLengthError, PxAbs(secondError));
		gMetrics.secondLengthErrorSquaredSum +=
			secondError * secondError;
		gMetrics.maxLengthVelocity = PxMax(
			gMetrics.maxLengthVelocity, PxAbs(secondLengthVelocity));
		gMetrics.lengthVelocitySquaredSum +=
			secondLengthVelocity * secondLengthVelocity;
	}
	if(gHeadlessMode == eLIMIT)
	{
		PxReal violation = 0.0f;
		if(length < gLimitLow)
			violation = gLimitLow - length;
		else if(length > gLimitHigh)
			violation = length - gLimitHigh;
		if(violation > 0.0f)
			++gMetrics.limitActiveSamples;
		gMetrics.maxLimitViolation =
			PxMax(gMetrics.maxLimitViolation, violation);
	}
	gMetrics.minQA = PxMin(gMetrics.minQA, qA);
	gMetrics.maxQA = PxMax(gMetrics.maxQA, qA);
	gMetrics.minQB = PxMin(gMetrics.minQB, qB);
	gMetrics.maxQB = PxMax(gMetrics.maxQB, qB);
	if(gLinkC)
	{
		gMetrics.minQC = PxMin(gMetrics.minQC, qC);
		gMetrics.maxQC = PxMax(gMetrics.maxQC, qC);
	}
	if(gMiddleLink)
	{
		gMetrics.minQM = PxMin(gMetrics.minQM, qM);
		gMetrics.maxQM = PxMax(gMetrics.maxQM, qM);
	}
	++gMetrics.responseSamples;
	if(PxAbs(vA) > 0.01f && PxAbs(vB) > 0.01f)
	{
		++gMetrics.directionSamples;
		if(vA * vB >= 0.0f)
			++gMetrics.directionViolations;
	}
}

void stepPhysics(bool interactive)
{
	const PxReal dt = interactive ? 1.0f / 60.0f : gHeadlessOptions.dt;
	const PxReal phase =
		PxTwoPi * PxReal(gMetrics.completedFrames) / 240.0f;
	if(gHeadlessCase == eOFFSET_ACTUATION)
	{
		const PxReal amplitude =
			gHeadlessMode == eLIMIT ? 0.45f : 0.15f;
		gCurrentOffset = gOffsetBase + amplitude * PxSin(phase);
		gCurrentMiddleHeight = gMiddleHeight;
		if(gTendon)
			gTendon->setOffset(gCurrentOffset);
	}
	else if(gHeadlessCase == eATTACHMENT_ACTUATION)
	{
		gCurrentOffset = gOffsetBase;
		gCurrentMiddleHeight =
			gMiddleHeight + 0.2f * PxSin(phase);
		if(gMiddleAttachment)
			gMiddleAttachment->setRelativeOffset(
				PxVec3(0.0f, gCurrentMiddleHeight, 0.0f));
		if(gArticulation)
			gArticulation->wakeUp();
	}
	else
	{
		gCurrentOffset = gOffsetBase;
		gCurrentMiddleHeight = gMiddleHeight;
		gMiddleDriveTarget +=
			1.2f * gMiddleDriveDirection * dt;
		if(gMiddleDriveTarget > 0.5f)
		{
			gMiddleDriveTarget = 0.5f;
			gMiddleDriveDirection = -1.0f;
		}
		else if(gMiddleDriveTarget < -0.5f)
		{
			gMiddleDriveTarget = -0.5f;
			gMiddleDriveDirection = 1.0f;
		}
		if(gMiddleJoint)
			gMiddleJoint->setDriveTarget(gAxis, gMiddleDriveTarget);
	}
	gScene->simulate(dt);
	if(!gScene->fetchResults(true))
	{
		++gMetrics.fetchFailures;
		return;
	}
	updateVisualizationAttachments();
	if(!interactive)
		recordHeadlessState(gMetrics.completedFrames);
	++gMetrics.completedFrames;
}

void cleanupPhysics(bool interactive)
{
	PX_UNUSED(interactive);
	for(PxU32 i = 0; i < 6; ++i)
		PX_RELEASE(getAttachments()[i]);
	PX_RELEASE(gArticulation);
	gTendon = NULL;
	gRootAttachment = NULL;
	gMiddleAttachment = NULL;
	gLeafAttachment = NULL;
	gSecondLeafAttachment = NULL;
	gJointA = NULL;
	gJointB = NULL;
	gJointC = NULL;
	gMiddleJoint = NULL;
	gRootLink = NULL;
	gLinkA = NULL;
	gLinkB = NULL;
	gLinkC = NULL;
	gMiddleLink = NULL;
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
		std::printf("SnippetSpatialTendon done.\n");
}

static int runHeadless()
{
	Snippets::printHeadlessConfig(
		"SnippetSpatialTendon", gHeadlessOptions);
	initPhysics(false);
	const bool initialized =
		gFoundation && gPhysics && gDispatcher && gScene && gMaterial &&
		gArticulation && gRootLink && gLinkA && gLinkB &&
		gJointA && gJointB && gTendon && gRootAttachment &&
		gMiddleAttachment && gLeafAttachment &&
		(gHeadlessMode != eMULTI_LEAF ||
			(gLinkC && gJointC && gSecondLeafAttachment)) &&
		(gHeadlessMode != eMOVING_MIDDLE ||
			(gMiddleLink && gMiddleJoint));
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
		gMetrics.responseSamples ?
			PxSqrt(gMetrics.lengthErrorSquaredSum /
				PxReal(gMetrics.responseSamples)) : PX_MAX_F32;
	const PxReal lengthVelocityRms =
		gMetrics.responseSamples ?
			PxSqrt(gMetrics.lengthVelocitySquaredSum /
				PxReal(gMetrics.responseSamples *
					(gLinkC ? 2u : 1u))) : PX_MAX_F32;
	const PxReal secondLengthRms =
		gMetrics.responseSamples && gLinkC ?
			PxSqrt(gMetrics.secondLengthErrorSquaredSum /
				PxReal(gMetrics.responseSamples)) : 0.0f;
	const PxReal rangeA = gMetrics.maxQA - gMetrics.minQA;
	const PxReal rangeB = gMetrics.maxQB - gMetrics.minQB;
	const PxReal rangeC = gLinkC ?
		gMetrics.maxQC - gMetrics.minQC : 0.0f;
	const PxReal rangeM = gMiddleLink ?
		gMetrics.maxQM - gMetrics.minQM : 0.0f;

	PxReal coefficientRoot = 0.0f;
	PxReal coefficientMiddle = 0.0f;
	PxReal coefficientLeaf = 0.0f;
	PxReal coefficientSecondLeaf = 0.0f;
	PxVec3 relativeRoot(0.0f);
	PxVec3 relativeMiddle(0.0f);
	PxVec3 relativeLeaf(0.0f);
	PxVec3 relativeSecondLeaf(0.0f);
	if(gRootAttachment)
	{
		coefficientRoot = gRootAttachment->getCoefficient();
		relativeRoot = gRootAttachment->getRelativeOffset();
	}
	if(gMiddleAttachment)
	{
		coefficientMiddle = gMiddleAttachment->getCoefficient();
		relativeMiddle = gMiddleAttachment->getRelativeOffset();
	}
	if(gLeafAttachment)
	{
		coefficientLeaf = gLeafAttachment->getCoefficient();
		relativeLeaf = gLeafAttachment->getRelativeOffset();
	}
	if(gSecondLeafAttachment)
	{
		coefficientSecondLeaf =
			gSecondLeafAttachment->getCoefficient();
		relativeSecondLeaf =
			gSecondLeafAttachment->getRelativeOffset();
	}
	const bool limitMode = gHeadlessMode == eLIMIT;
	const PxReal expectedStiffness =
		limitMode ? 0.0f : gTendonStiffness;
	const PxReal expectedDamping =
		limitMode ? 0.0f : gTendonDamping;
	const PxReal expectedLimitStiffness =
		limitMode ? gLimitStiffness : 0.0f;
	PxArticulationTendonLimit limitReadback;
	limitReadback.lowLimit = PX_MAX_F32;
	limitReadback.highLimit = -PX_MAX_F32;
	if(gLeafAttachment)
		limitReadback = gLeafAttachment->getLimitParameters();
	const PxU32 expectedAttachmentCount =
		gHeadlessMode == eMULTI_LEAF ? 4u : 3u;
	const bool readbackValid =
		gTendon &&
		gTendon->getNbAttachments() == expectedAttachmentCount &&
		PxAbs(gTendon->getStiffness() - expectedStiffness) < 1e-4f &&
		PxAbs(gTendon->getDamping() - expectedDamping) < 1e-4f &&
		PxAbs(gTendon->getLimitStiffness() -
			expectedLimitStiffness) < 1e-4f &&
		PxAbs(gTendon->getOffset() - gCurrentOffset) < 1e-6f &&
		PxAbs(coefficientRoot - 1.0f) < 1e-6f &&
		PxAbs(coefficientMiddle - 1.0f) < 1e-6f &&
		PxAbs(coefficientLeaf - 1.0f) < 1e-6f &&
		(!gSecondLeafAttachment ||
			PxAbs(coefficientSecondLeaf - 1.0f) < 1e-6f) &&
		(relativeRoot - gLocalAttachmentA).magnitude() < 1e-6f &&
		(relativeMiddle -
			PxVec3(0.0f, gCurrentMiddleHeight, 0.0f)).magnitude() < 1e-6f &&
		(relativeLeaf - gLocalAttachmentB).magnitude() < 1e-6f &&
		(!gSecondLeafAttachment ||
			(relativeSecondLeaf - gLocalAttachmentC).magnitude() < 1e-6f) &&
		PxAbs(gLeafAttachment->getRestLength() - gRestLength) < 1e-6f &&
		(!gSecondLeafAttachment ||
			PxAbs(gSecondLeafAttachment->getRestLength() -
				gRestLength) < 1e-6f) &&
		gRootAttachment->getParent() == NULL &&
		gMiddleAttachment->getParent() == gRootAttachment &&
		gLeafAttachment->getParent() == gMiddleAttachment &&
		(!gSecondLeafAttachment ||
			gSecondLeafAttachment->getParent() == gMiddleAttachment) &&
		!gRootAttachment->isLeaf() && !gMiddleAttachment->isLeaf() &&
		gLeafAttachment->isLeaf() &&
		(!gSecondLeafAttachment || gSecondLeafAttachment->isLeaf()) &&
		(!limitMode ||
			(PxAbs(limitReadback.lowLimit - gLimitLow) < 1e-6f &&
			 PxAbs(limitReadback.highLimit - gLimitHigh) < 1e-6f));

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
	else if(gMetrics.nonFinite != 0 ||
		gErrorCallback.getFatalCount() != 0)
	{
		passed = false;
		reason = "runtime_error";
	}
	else if(gMetrics.responseSamples < gHeadlessOptions.frames - 62)
	{
		passed = false;
		reason = "missing_samples";
	}
	else if(rangeA < 0.03f || rangeB < 0.03f)
	{
		passed = false;
		reason = "missing_coupled_response";
	}
	else if(gLinkC && rangeC < 0.03f)
	{
		passed = false;
		reason = "missing_second_leaf_response";
	}
	else if(gMiddleLink && rangeM < 0.3f)
	{
		passed = false;
		reason = "missing_middle_motion";
	}
	else if(gHeadlessMode != eMOVING_MIDDLE &&
		(gMetrics.directionSamples < 30 ||
		 gMetrics.directionViolations * 10 > gMetrics.directionSamples))
	{
		passed = false;
		reason = "wrong_coupling_direction";
	}
	else if(!limitMode &&
		(lengthRms > 0.10f || gMetrics.maxLengthError > 0.25f ||
		 secondLengthRms > 0.10f ||
		 gMetrics.maxSecondLengthError > 0.25f))
	{
		passed = false;
		reason = "tendon_length_error";
	}
	else if(!limitMode && (lengthVelocityRms > 0.6f ||
		gMetrics.maxLengthVelocity > 3.0f)
	)
	{
		passed = false;
		reason = "tendon_velocity_error";
	}
	else if(limitMode &&
		(gMetrics.limitActiveSamples < 20 ||
		 gMetrics.maxLimitViolation > 0.10f))
	{
		passed = false;
		reason = "tendon_limit_error";
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
		"[AVBD_GATE] schema=1 snippet=SnippetSpatialTendon spatialMode=%s "
		"axis=%s attachmentCount=%u leafCount=%u solver=%s case=%s "
		"execution=%s frames=%u completedFrames=%u status=%s reason=%s "
		"validation=GATED coefficientRoot=1 coefficientMiddle=1 "
		"coefficientLeaf=1 coefficientSecondLeaf=%.9g "
		"offsetBase=%.9g offsetFinal=%.9g restLength=%.9g "
		"stiffness=%.9g damping=%.9g limitStiffness=%.9g "
		"lowLimit=%.9g highLimit=%.9g responseSamples=%u "
		"directionSamples=%u directionViolations=%u "
		"rangeA=%.9g rangeB=%.9g rangeC=%.9g rangeM=%.9g "
		"lengthErrorMax=%.9g lengthErrorRms=%.9g "
		"secondLengthErrorMax=%.9g secondLengthErrorRms=%.9g "
		"lengthVelocityMax=%.9g lengthVelocityRms=%.9g "
		"maxLimitViolation=%.9g limitActiveSamples=%u "
		"finalQA=%.9g finalQB=%.9g finalVelocityA=%.9g finalVelocityB=%.9g "
		"finalQC=%.9g finalQM=%.9g "
		"finalPublicQA=%.9g finalPublicQB=%.9g "
		"maxPublicCoordinateError=%.9g nonFinite=%u fetchFailures=%u "
		"fatalErrors=%u cleanupComplete=%u pvd=0\n",
		getModeName(), getAxisName(), expectedAttachmentCount,
		gHeadlessMode == eMULTI_LEAF ? 2u : 1u,
		Snippets::getSolverTypeName(gHeadlessOptions.solverType), getCaseName(),
		Snippets::getExecutionName(gHeadlessOptions.execution),
		gHeadlessOptions.frames, gMetrics.completedFrames,
		passed ? "PASS" : "FAIL", reason,
		double(coefficientSecondLeaf),
		double(gOffsetBase), double(gCurrentOffset), double(gRestLength),
		double(expectedStiffness), double(expectedDamping),
		double(expectedLimitStiffness), double(limitReadback.lowLimit),
		double(limitReadback.highLimit), gMetrics.responseSamples,
		gMetrics.directionSamples, gMetrics.directionViolations,
		double(rangeA), double(rangeB), double(rangeC), double(rangeM),
		double(gMetrics.maxLengthError), double(lengthRms),
		double(gMetrics.maxSecondLengthError), double(secondLengthRms),
		double(gMetrics.maxLengthVelocity), double(lengthVelocityRms),
		double(gMetrics.maxLimitViolation), gMetrics.limitActiveSamples,
		double(gMetrics.finalQA),
		double(gMetrics.finalQB), double(gMetrics.finalVelocityA),
		double(gMetrics.finalVelocityB), double(gMetrics.finalQC),
		double(gMetrics.finalQM), double(gMetrics.finalPublicQA),
		double(gMetrics.finalPublicQB),
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
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetSpatialTendon reason=%s\n",
			error.c_str());
		return Snippets::eHEADLESS_CONFIG_ERROR;
	}
	if(!Snippets::applyExecutionEnvironment(gHeadlessOptions))
	{
		std::fprintf(stderr,
			"[AVBD_GATE_CONFIG_ERROR] snippet=SnippetSpatialTendon "
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
